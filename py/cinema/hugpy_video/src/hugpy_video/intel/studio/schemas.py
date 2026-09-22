"""Schemas over ad-hoc objects. Every one is a frozen, slotted dataclass.

The load-bearing type is `RenderManifest` (INV-1): a render is *defined* by its
manifest, and the pixels are a cache of it. `canonical_inputs()` / `content_hash()`
give a stable reproducibility + dedup + resume key (INV-6) that excludes metadata
(render_id, timestamps) and includes everything that changes the output.

One exception to "every one is a dataclass": the CLIP-LENGTH POLICY constants
(``WAN_FRAME_CADENCE`` / ``WAN_MAX_FRAMES`` / ``DEFAULT_FRAMES_REAL`` +
``snap_wan_frames``) live here too, deliberately — see the long WHY on that
section. They bound ``RenderManifest.requested_frames``, they are declared next to
the field they bound, and this is the only module BOTH the renderer and the
presets/wire layer can import without dragging numpy/PIL into app boot.

No pathlib anywhere. os.path only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from hugpy_video.intel.studio.enums import (
    AdapterKind,
    Capability,
    ControlKind,
    DeterminismClass,
    Framework,
    LicenseClass,
    LICENSE_AUTO_COMMERCIAL,
    PathClass,
    Precision,
    RiskFlag,
    Task,
)

# ---------------------------------------------------------------------------
# Value schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Resolution:
    width: int
    height: int
    fps: int  # nominal target fps for the model; real cadence is per-render

    def __post_init__(self) -> None:
        # Structurally-invalid geometry is programmer error, not runtime data.
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError(f"invalid Resolution {self.width}x{self.height}@{self.fps}")

    @property
    def area(self) -> int:
        return self.width * self.height

    def covers(self, target: "Resolution") -> bool:
        """True if this resolution is at least as large as `target` in both dims."""
        return self.width >= target.width and self.height >= target.height


@dataclass(frozen=True, slots=True)
class VramEnvelope:
    """VRAM cost per precision, in GB. Stored as a sorted tuple of pairs so the
    schema stays frozen/hashable. Values are PLANNING ESTIMATES (they move with
    resolution/frames/offload) - the router uses them for fit, not as a promise."""
    per_precision: tuple[tuple[Precision, float], ...]

    def __post_init__(self) -> None:
        if not self.per_precision:
            raise ValueError("VramEnvelope needs at least one precision")
        for prec, gb in self.per_precision:
            if gb <= 0:
                raise ValueError(f"non-positive VRAM for {prec}: {gb}")

    def as_map(self) -> dict[Precision, float]:
        return {p: gb for p, gb in self.per_precision}

    def min_gb(self) -> float:
        return min(gb for _, gb in self.per_precision)

    def fits(self, budget_gb: float) -> tuple[Precision, ...]:
        return tuple(p for p, gb in self.per_precision if gb <= budget_gb)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """All seeds captured, never implicit (INV-2)."""
    global_seed: int
    stage_seeds: tuple[tuple[str, int], ...] = ()   # (stage_name, seed)
    chunk_seed_base: int | None = None               # for autoregressive chunks


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    shift: float | None = None
    sigmas: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ControlRef:
    """A control signal referenced by content hash (INV-1). The pixels of the
    depth map / pose skeleton live in the store; the manifest carries the hash."""
    kind: ControlKind
    content_hash: str
    weight: float = 1.0
    target_frames: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterRef:
    kind: AdapterKind
    adapter_id: str
    weight: float
    weight_hash: str


@dataclass(frozen=True, slots=True)
class ProvenanceStub:
    """INV-7. C2PA is filled in at mastering; this is the internal stub."""
    operator: str
    created_at: str        # ISO-8601
    tool: str = "hugpy-studio"
    c2pa_pending: bool = True


# ---------------------------------------------------------------------------
# CLIP-LENGTH POLICY - the one place each length fact is spelled
# ---------------------------------------------------------------------------
# ⚠ WHY THESE LIVE IN schemas.py AND NOT NEXT TO resolve_frames (2026-07-27).
# They were born in ``runners/synthetic.py``, which is the right home for the
# LOGIC (``resolve_frames`` is still there, still the ONE decider). But the WIRE
# has to publish the same numbers - ``GET /video/render/presets`` serves a
# ``default_frames`` / ``max_frames`` per preset row out of ``studio/presets.py``
# - and the adversarial review that produced this change found the wire publishing
# 29 while the renderer produced 81. Two literals, 2.8x apart, with no import
# between them: a caller was told 29 frames and charged for 81 (~2.8x the latent
# tokens and the GPU minutes). That is a defaults-are-promises violation on the
# most expensive axis the studio has.
#
# The fix is ONE literal per fact, imported by both consumers. It could not live
# in the runner: ``presets`` is imported at ``import studio`` time (studio/__init__
# -> router -> presets) while ``runners/synthetic`` imports numpy + PIL at module
# top, so a presets -> runner import would drag the whole imaging stack into app
# boot - exactly what this package keeps out on purpose (studio/job.py's header:
# "no numpy/PIL pulled into app boot from here"). ``schemas`` imports only
# ``enums``, it is where ``RenderManifest.requested_frames`` is DECLARED, and both
# the runner and presets already sit downstream of it. So the facts live here and
# the two consumers import them.

# Wan's latent VAE compresses TIME 4:1, so every Wan pipeline accepts only
# ``num_frames == 4*k + 1`` (81 = 4*20 + 1). A frame count off this cadence is not
# "slightly wrong" - the pipeline either rejects it or silently renders a different
# length than was asked for, which is why the snap happens BEFORE the resume check
# and the generate call (they must agree on one number).
WAN_FRAME_CADENCE = 4

# HARD Wan ceiling, independent of the registry. Every Wan row in ``models_seed``
# declares ``max_frames=81``, and CAPABILITY-VIABILITY-MAP.md (measured on ae's
# 3090, 2026-07-27) confirms 81 is the real ceiling for all three Wan models that
# actually have weights on disk. Belt-and-braces over ``ModelConfig.max_frames`` so
# a future registry typo can never hand a Wan pipeline a 200-frame request.
WAN_MAX_FRAMES = 81

# DEFAULT clip length for a REAL model when the manifest requests none.
#
# 81 frames. Not a guess and not a placeholder: it is Wan's own reference length
# (81 @ 16fps = 5.0625s), it is the ``max_frames`` every Wan row declares, and it
# is MEASURED - wan2.1-t2v-1.3b at 832x480 x 81 frames renders in ~352s wall-clock
# on ae's 3090 (2026-07-27). Operator doctrine "defaults are promises": the
# previous default was ``fps * 2`` (32 frames -> snapped 29 = ~1.8s), a placeholder
# written for the no-model noise prover that silently governed every REAL render on
# this fleet. ~5s for ~6min of GPU is a success path a caller would actually want;
# a caller who wants a cheap preview has the ``requested_frames`` lever to ask for
# fewer. Heavier rows (the 14B i2v) pay more wall-clock for the SAME 81 frames -
# length is the request, not the cost model.
#
# ⚠ THIS IS THE NUMBER THE WIRE MUST PUBLISH. ``studio/presets.py`` imports it for
# every Wan preset's ``default_frames`` rather than restating 81 (or, as it did
# until this change, 29). If you change it here, the presets endpoint changes with
# it - that is the entire point.
DEFAULT_FRAMES_REAL = 81


def snap_wan_frames(n_frames: int) -> int:
    """Nearest legal Wan frame count AT OR BELOW ``n_frames`` (i.e. 4k+1).

    Snap DOWN, never up: up could push past the model's ceiling, which is the one
    direction that turns a clamp into an OOM. Used by ``runners.synthetic.
    resolve_frames`` at render time AND by the presets layer, which needs to show a
    caller the TRUE length before spending ~6 minutes of denoise on it - one
    implementation so the preview and the render can never disagree.

    NOTE the arithmetic this enforces: 4k+1 is always ODD, so ``frames / fps`` can
    never be a whole number of seconds at an EVEN fps (81/16 = 5.0625s, not 5s).
    FRAMES is therefore the exact unit; a duration in seconds is a REQUEST that must
    be resolved to on-cadence frames with the TRUE resulting duration reported back,
    never the requested one echoed."""
    n = max(1, int(n_frames))
    return ((n - 1) // WAN_FRAME_CADENCE) * WAN_FRAME_CADENCE + 1


# ---------------------------------------------------------------------------
# RenderManifest - the source of truth for a render (INV-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderManifest:
    render_id: str
    capability: Capability
    model_id: str
    weight_hash: str | None          # None only if the bound model is unpinned
    framework: Framework
    task: Task
    precision: Precision             # FIX-1: router-selected precision changes the
                                     # output (fp8 vs bf16), so it MUST be in the hash
    seeds: SeedBundle
    sampler: SamplerConfig
    resolution_ladder: tuple[Resolution, ...]
    controls: tuple[ControlRef, ...] = ()
    adapters: tuple[AdapterRef, ...] = ()
    identity_ids: tuple[str, ...] = ()           # character stable IDs (§3)
    identity_view_hashes: tuple[str, ...] = ()   # canonical multi-view refs (ID-3)
    determinism_class: DeterminismClass = DeterminismClass.SEEDED_APPROX
    env_snapshot: tuple[tuple[str, str], ...] = ()
    provenance: ProvenanceStub | None = None
    # Text conditioning (C-prompt): the prompt genuinely changes the output, so it
    # is part of the reproducibility key (canonical_inputs -> content_hash). "" is a
    # valid empty prompt (image-conditioned i2v). Appended (not inserted) so no
    # positional field shifts for existing construction sites.
    prompt: str = ""
    negative_prompt: str = ""
    # Source clip the render extends (B2 movie->studio chain). A DIFFERENT source
    # video genuinely changes the output (its last frame conditions an i2v extend),
    # so it is part of the reproducibility key (canonical_inputs -> content_hash).
    # "" is a valid "no source" (the common i2v/t2v case). Appended (not inserted)
    # so no positional field shifts for existing construction sites.
    source_video: str = ""
    # IDENTITY LOCK (id_lock, Wan VACE reference-to-video): the reference image path(s)
    # of the subject whose identity is preserved across the render. CANONICAL — a
    # DIFFERENT reference set genuinely changes the output (each is prepended as a VACE
    # reference latent), so it is part of the reproducibility key (canonical_inputs ->
    # content_hash). ORDER IS PRESERVED in the hash (the pipeline consumes them in
    # order). () is the valid "no reference" case (v2v restyle / plain i2v/t2v).
    reference_images: tuple[str, ...] = ()
    # OPTIONAL VACE CONTROL channel used for composition blocking when there is no
    # source_video: a single still (pose/depth/sketch) repeated across the frame count
    # as the pipeline's `video=` control input. CANONICAL when present (it changes the
    # generated composition). "" = no control image; control_kind is the control type
    # ("pose"|"depth"|"sketch"), "" when unused.
    control_image: str = ""
    control_kind: str = ""
    # VACE-EXTEND temporal conditioning (studio-movie splice motion-carry): the ORDERED
    # abs paths of the parent clip's TRAILING context frames (oldest -> newest, ending
    # at + including the branch frame). When non-empty, the VACE runner builds the
    # diffusers video+mask EXTEND idiom — these frames are the KEPT prefix (mask=0) and
    # the remaining num_frames-K positions are GENERATED (mask=1), carrying motion across
    # the splice instead of restarting from a single still. () = not an extend render.
    #
    # DELIBERATELY EXCLUDED FROM canonical_inputs()/content_hash (see below): unlike the
    # VACE runner's other conditioning inputs (source_video/reference_images/
    # control_image, which ARE hashed), these frames are extracted by the movie runner to
    # a JOB-SPECIFIC path (<movie_root>/segment_NN/context/). Hashing that path would
    # defeat same-job resume and needlessly re-address every existing clip. This mirrors
    # how the i2v ``start_image`` conditions a render without being hashed; the movie
    # runner's out_root isolation (each segment under segment_NN) provides resume
    # correctness. Carried in the manifest so the runner can consume it + the sidecar can
    # record it (provenance), just never as a content-hash input.
    vace_context_frames: tuple[str, ...] = ()
    # CLIP LENGTH, as a REQUEST. ``None`` = "unset — use the runner's model-aware
    # default"; an int = "the caller asked for exactly this many frames". The runner
    # RESOLVES it (clamp to the model ceiling, snap to the pipeline's 4k+1 temporal
    # cadence, max(1, n) floor) — see ``runners.synthetic.resolve_frames``, the ONE
    # place clip length is decided.
    #
    # ⚠ REACHABILITY (2026-07-27). The first cut of this field was a LIE: it existed
    # on the dataclass and in ``canonical_inputs``, the docstrings advertised "a
    # caller who wants a cheap preview now has a lever", and NO production path could
    # set it — ``_build_manifest`` / ``make_render_manifest`` (the ONE live build
    # path) took no such kwarg, ``StudioI2VSpec`` had no length field, and
    # ``produce_clip`` passed none. Only tests constructed it. It is now threaded end
    # to end: ``StudioI2VSpec.requested_frames`` -> ``produce_clip(requested_frames=)``
    # -> ``make_render_manifest(requested_frames=)`` -> here -> ``resolve_frames``.
    # ONE NAME at every seam on purpose, so no seam has to translate.
    #
    # FRAMES, not seconds, is the unit ON PURPOSE: Wan's latent VAE compresses time
    # 4:1, so its pipelines accept only ``num_frames == 4k+1`` — which is always ODD,
    # so ``frames / fps`` can never be a whole number of seconds at an EVEN fps
    # (81/16 = 5.0625s). A seconds request would have to be multiplied by fps and then
    # snapped anyway, i.e. the delivered duration would silently differ from the number
    # the caller typed. Frames is what the model actually accepts, it is the unit the
    # registry ceiling is expressed in (``ModelConfig.max_frames``), and seconds stay
    # derivable + reported as the TRUE resulting duration (``Artifact.duration_s`` =
    # resolved_frames / fps) rather than echoing the request back.
    #
    # CANONICAL (in the content_hash): a different requested length is a genuinely
    # different clip, so two requests that differ ONLY in length must not collide on
    # one content-addressed path (without this, a 33-frame request would RESUME an
    # existing 81-frame clip). Being canonical makes the manifest SIDECAR round-trip
    # load-bearing: ``render_manifest_to_dict`` MUST serialize this and
    # ``render_manifest_from_dict`` MUST restore it, or a rehydrated manifest
    # re-addresses to a different content hash and resume/dedup break for every
    # non-default length (the exact defect found on 2026-07-27 — the field was in
    # ``canonical_inputs`` but absent from both sides of the sidecar). Guarded by
    # ``tests/studio/test_clip_length.py``'s round-trip check.
    #
    # Appended (not inserted) so no positional field shifts for existing sites.
    requested_frames: int | None = None

    def canonical_inputs(self) -> dict:
        """Everything that changes the output; nothing that is mere metadata.
        Excludes render_id + provenance so two identical intents hash equal."""
        return {
            "capability": self.capability.value,
            "model_id": self.model_id,
            "weight_hash": self.weight_hash,
            "framework": self.framework.value,
            "task": self.task.value,
            "precision": self.precision.value,   # FIX-1: fp8 vs bf16 must not collide
            "seeds": {
                "global": self.seeds.global_seed,
                "stage": sorted(self.seeds.stage_seeds),
                "chunk_base": self.seeds.chunk_seed_base,
            },
            "sampler": {
                "sampler": self.sampler.sampler,
                "scheduler": self.sampler.scheduler,
                "steps": self.sampler.steps,
                "cfg": self.sampler.cfg,
                "shift": self.sampler.shift,
                "sigmas": list(self.sampler.sigmas),
            },
            "resolution_ladder": [
                [r.width, r.height, r.fps] for r in self.resolution_ladder
            ],
            "controls": sorted(
                [c.kind.value, c.content_hash, c.weight, list(c.target_frames)]
                for c in self.controls
            ),
            "adapters": sorted(
                [a.kind.value, a.adapter_id, a.weight, a.weight_hash]
                for a in self.adapters
            ),
            "identity_ids": sorted(self.identity_ids),
            "identity_view_hashes": sorted(self.identity_view_hashes),
            "determinism_class": self.determinism_class.value,
            "env_snapshot": sorted(self.env_snapshot),
            # C-prompt: text conditioning changes the pixels, so it is in the hash.
            # Empty prompt still participates (its value is just ""), which re-addresses
            # ALL prior content-addressed clips once — correct + acceptable for dev.
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            # Source-clip conditioning (B2 chain): a different source video -> a
            # different extend -> a different clip, so it participates in the hash.
            # "" (no source) still participates (re-addresses prior clips once, same
            # one-time cost + rationale as the empty-prompt case above).
            "source_video": self.source_video,
            # Identity-lock reference images (id_lock): CANONICAL, ORDER-PRESERVED (each
            # is a VACE reference latent prepended in order — a reorder is a different
            # render). () -> [] participates like the empty-source case. Different refs
            # -> different content_hash (proven in the id_lock suite).
            "reference_images": list(self.reference_images),
            # Optional VACE control channel (composition blocking): canonical when set.
            "control_image": self.control_image,
            "control_kind": self.control_kind,
            # CLIP LENGTH request: a different requested length is a different clip, so
            # it participates in the hash. None (unset -> model default) participates as
            # null, which RE-ADDRESSES every previously-rendered clip exactly once —
            # deliberate and REQUIRED here, not merely tolerated: this change also moves
            # the real-model default off the fps*2 placeholder (~1.8s) onto the model's
            # measured ceiling, so an unchanged hash would have RESUMED those old 29-frame
            # clips as if they were the new 81-frame default. Same one-time cost + rationale
            # as the empty-prompt / empty-source cases above.
            "requested_frames": self.requested_frames,
            # NOTE: ``vace_context_frames`` is DELIBERATELY absent here (not a content-hash
            # input) — it is extracted to a job-specific path by the movie runner, so
            # hashing it would break same-job resume + re-address every clip. It conditions
            # the render like the (also-unhashed) i2v start_image; resume correctness comes
            # from the movie runner's per-segment out_root isolation. See the field's
            # docstring on RenderManifest.
        }

    def content_hash(self) -> str:
        blob = json.dumps(self.canonical_inputs(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ModelConfig - one row of the zoo, as data (§2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    family: Framework
    tasks: tuple[Task, ...]
    capabilities: tuple[Capability, ...]
    vram: VramEnvelope
    resolutions: tuple[Resolution, ...]
    max_frames: int
    max_duration_s: float
    license: LicenseClass
    weight_uri: str                       # HF repo / GitHub / local
    source_url: str
    default_determinism: DeterminismClass
    path_class: PathClass = PathClass.OFFLINE
    native_audio: bool = False
    accepts_adapters: frozenset[AdapterKind] = frozenset()
    weight_hash: str | None = None        # pin before production
    unpinned: bool = False                # must be True if weight_hash is None
    verify_uri: bool = False              # repo path not confirmed this session
    synthetic: bool = False               # LAST-RESORT placeholder (no-model
                                          # procedural runner). The router ranks any
                                          # REAL model strictly above a synthetic
                                          # one, so it binds only when no real model
                                          # fits the request. Set True on synthetic
                                          # rows ONLY (see models_seed synthetic-i2v).
    notes: str = ""

    @property
    def commercial_auto(self) -> bool:
        return LICENSE_AUTO_COMMERCIAL[self.license]

    def supports_resolution(self, target: Resolution) -> bool:
        return any(r.covers(target) for r in self.resolutions)

    def best_native_area(self) -> int:
        return max(r.area for r in self.resolutions)


# ---------------------------------------------------------------------------
# Request / binding / job
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    """What a shot asks for. The router turns this into a ModelBinding or an Err."""
    capability: Capability
    target_resolution: Resolution
    vram_budget_gb: float
    commercial_use: bool = False
    allowed_licenses: frozenset[LicenseClass] = frozenset()  # empty = any
    latency_budget_ms: int | None = None    # set => streaming path required (STR-6)
    require_native_audio: bool = False
    preferred_framework: Framework | None = None
    risk_flags: frozenset[RiskFlag] = frozenset()
    min_frames: int = 0
    # A prior-tier clip (movie/scene mp4) the request extends (B2 movie->studio chain).
    # Carried through to the manifest; routing itself does not key on it (an i2v job
    # extends it from its LAST FRAME — see produce_clip / the i2v runners). None when
    # the request has no source clip (a plain i2v/t2v).
    source_video: str | None = None
    # DIRECT MODEL CHOICE (pin): the caller wants a SPECIFIC model_id, not the router's
    # auto-pick. When set, the router restricts its candidate set to exactly this model
    # and either binds it (if it declares the capability, serves it, and fits the live
    # budget/resolution) or returns a CLEAR Err-as-data — NEVER a silent fallback to a
    # different model (see CapabilityRouter.resolve + ErrorCode.PINNED_MODEL_UNAVAILABLE).
    # None (the default) = auto-pick, the historical behavior. Appended (not inserted)
    # so no positional field shifts for existing construction sites.
    pinned_model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """The router's resolved answer: which model, which runner, which precision."""
    model_id: str
    framework: Framework
    task: Task
    precision: Precision
    path_class: PathClass
    weight_uri: str
    weight_hash: str | None
    determinism_class: DeterminismClass   # FIX-3: carried from the bound model's
                                          # default_determinism so a manifest built
                                          # from this binding reflects the real class
                                          # (EXACT/SEEDED_APPROX/DRIFTING), not a
                                          # hardcoded literal.


@dataclass(frozen=True, slots=True)
class Job:
    """Frozen currency of the queue (ORCH-1). Lifecycle STATE is not stored here -
    it lives in the append-only ledger (ORCH-4); the Job itself never mutates."""
    job_id: str
    request: CapabilityRequest
    binding: ModelBinding
    manifest: RenderManifest
    priority: int = 100
    retake_budget: int = 2
    risk_flags: frozenset[RiskFlag] = frozenset()
    provenance: ProvenanceStub | None = None


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One append-only transition (ORCH-4/ORCH-6). Errors ride along as data."""
    job_id: str
    state: "JobStateT"
    at: str                       # ISO-8601
    detail: str = ""
    error: "StageErrorT | None" = None


# Late imports for annotations only (avoid a hard import cycle with errors/enums
# at module top while keeping the names available for tooling).
from hugpy_video.intel.studio.enums import JobState as JobStateT
from hugpy_video.intel.studio.errors import StageError as StageErrorT
