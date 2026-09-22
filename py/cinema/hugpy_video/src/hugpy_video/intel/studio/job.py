"""Studio i2v JOB SPEC — the durable, bus-carried intent for a studio clip (B2).

This is the media-bus currency for a studio image-to-video job: a small frozen,
validate-at-construction spec (house style mirrored from ``movie_schema.make_movie``
/ ``crop_schema.make_crop``) that carries everything a studio i2v job needs and
nothing that isn't JSON-safe. The bus serializes it via ``asdict`` -> ``json`` and
rehydrates it through ``studio_i2v_from_dict`` (reconstruct + RE-VALIDATE), exactly
like every other media job.

Deliberately DECOUPLED from the studio spine's rich value objects: the spec holds
plain primitives (capability as a string, geometry as ints), so it round-trips
through JSON with zero enum/dataclass ceremony. The bus RUNNER
(``video_intel/runners/studio_i2v.py``) is the one place those primitives are
lifted into ``CapabilityRequest`` / ``Resolution`` / ``SeedBundle`` and handed to
``produce_clip`` — keeping the studio spine importable-but-dormant at module top
(no numpy/PIL pulled into app boot from here).

``resolve_studio_env`` satisfies INV-5 by RESOLVING concrete paths (under the
media-store root) rather than demanding the operator set STUDIO_* env vars — a
worker enqueues a studio job with no environment wiring at all.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_platform.platform_facade import env_value
from hugpy_video.intel.studio.enums import Capability
from hugpy_video.intel.studio.env import StudioEnv

logger = logging.getLogger(__name__)

# Where studio clips land by default: a content-addressed tree UNDER the media
# store root, so the produced clip.mp4 is inside media_store.ingest's storage
# jail (DEFAULT_ROOT) and can be cataloged like any other media output.
STUDIO_ROOT = os.path.join(DEFAULT_ROOT, "video_intel", "studio")
DEFAULT_CLIPS_ROOT = os.path.join(STUDIO_ROOT, "clips")

# Studio "Send to Editor" (k12) INBOX — a STABLE first-class sibling of clips/,
# weights/ and manifests/ under STUDIO_ROOT. The one-click handoff writes a
# Filmora-native MP4 here; the operator's LAN Windows workstation (Filmora
# desktop) picks the folder up over a host-side Samba export. Defined ONCE here
# (not recomputed as os.path.join(STUDIO_ROOT, "editor-inbox") at each call site)
# so the route and its tests import the single canonical constant.
EDITOR_INBOX_ROOT = os.path.join(STUDIO_ROOT, "editor-inbox")

# This slice wires exactly one studio runner path (SYNTHETIC i2v — the no-model
# spine prover). A budget below the smallest real i2v footprint (8GB) makes the
# router deterministically bind synthetic-i2v; that is the produce-a-clip default.
# A larger budget routes to a real model whose runner is not wired yet, which
# comes back as JobResult(ok=False, RUNNER_MISSING) — errors as data, never a raise.
_DEFAULT_VRAM_BUDGET_GB = 0.5

# Sanity ranges for the optional sampler overrides (route passthrough). steps below 1
# is a no-op denoise (gray mush); a huge step count or cfg is almost always a typo that
# would burn a render. cfg 0 is valid (unguided). These bound the /video/studio/i2v
# route's 400 AND every non-route caller (make_studio_i2v is the single validator).
_MIN_STEPS, _MAX_STEPS = 1, 100
_MIN_CFG, _MAX_CFG = 0.0, 20.0

# CLIP LENGTH sanity range (route passthrough), deliberately WIDE. This is a
# TYPO/DoS guard, NOT the model ceiling — those are two different jobs:
#   * The MODEL ceiling (81 for Wan, 49 for cogvideox-5b, ...) is only knowable AFTER
#     the router binds a model, which happens long after this spec is built. An
#     over-ceiling ask is therefore NOT a caller error: ``resolve_frames`` CLAMPS it
#     with a recorded reason and renders the longest clip the bound model can do
#     ("give me the longest you can" must not be a 400).
#   * This bound only catches what is nonsense at ANY ceiling: zero/negative frames
#     (not a clip), and a fat-fingered 100000000 that would otherwise ride all the way
#     to a runner. 10_000 sits comfortably above every non-sentinel ``max_frames`` in
#     the registry (the largest real one is framepack-i2v-hy at 3600), so nothing a
#     model could actually deliver is refused here.
# bool is an int subclass and is rejected explicitly — ``requested_frames=True`` would
# otherwise mean "1 frame".
_MIN_REQUESTED_FRAMES, _MAX_REQUESTED_FRAMES = 1, 10_000

_VALID_CAPABILITIES = frozenset(c.value for c in Capability)

# IDENTITY LOCK (id_lock, Wan VACE reference-to-video). Multiple reference images ARE
# consumed by diffusers 0.39 (each prepended as a VACE reference latent), so we accept
# up to this many; more is rejected with a clean caller error (NEVER silently dropped).
_MAX_REFERENCE_IMAGES = 4
# The VACE control channel kinds (composition blocking). A control image needs a kind
# to be meaningful; a kind outside this set is a caller error.
_VALID_CONTROL_KINDS = frozenset({"pose", "depth", "sketch"})


@dataclass(frozen=True)
class StudioI2VSpec:
    """Frozen currency of a studio i2v bus job. Built ONLY via ``make_studio_i2v``
    (validate-at-construction); the bus rehydrates it via ``studio_i2v_from_dict``,
    which re-validates through the same factory. All fields are JSON-safe
    primitives so ``asdict`` -> ``json.dumps`` round-trips cleanly."""
    capability: str          # a Capability value, e.g. "i2v"
    width: int
    height: int
    fps: int
    # AUTOFIT budget. A number PINS the routing tier (manual override). ``None`` means
    # AUTOFIT: a BLANK budget is NOT a guaranteed-fail low guess — at render time the
    # shared render path (``runners.studio_i2v.render_clip``) sizes it to the serving
    # worker's MEASURED free VRAM (operator doctrine: "if a model needs 14GB and it's
    # blank, just do 14, otherwise a fail is 100% likely"). None is threaded verbatim
    # through the router probes / produce_clip only AFTER render_clip resolves it.
    vram_budget_gb: Optional[float]
    seed: int
    out_root: str            # output location (a dir under the media-store root)
    start_image: Optional[str] = None   # abs path to a still (i2v conditioning)
    negative: Optional[str] = None      # carried; synthetic ignores it
    prompt: Optional[str] = None        # carried; synthetic ignores it
    # optional human project NAME (auto-archive metadata). NON-CANONICAL: carried on the
    # spec + cataloged, but NEVER threaded into produce.py/manifest.py, so it is NOT part
    # of the render content_hash (addressing). Mirrors gen/scene/movie_schema's `project`.
    project: Optional[str] = None
    source_video: Optional[str] = None  # abs path to a prior-tier clip (movie/scene
                                        # mp4). The movie->studio chain input (B2): an
                                        # i2v job with no start_image EXTENDS this clip
                                        # from its LAST FRAME. Carried in the manifest
                                        # (part of the content_hash) either way.
    # SAMPLER OVERRIDES (route passthrough). None = "unset": the studio spine fills the
    # denoise settings from the BOUND model's family default (steps 32 / cfg 5.0 for
    # Wan, steps 1 for synthetic). A number here PINS that field and ALWAYS wins over the
    # model default. Range-validated in make_studio_i2v (steps 1-100, cfg 0-20) so an
    # out-of-range value is a clean caller error, never a bad render.
    steps: Optional[int] = None
    cfg: Optional[float] = None
    # DIRECT MODEL CHOICE (pin). None = auto-pick (the router chooses by capability +
    # budget + resolution). A model_id here is threaded into the CapabilityRequest as a
    # pin: the router binds THAT model or returns a clear Err-as-data (never a silent
    # fallback). Not validated against the registry here (that is a runtime routing
    # decision, surfaced as errors-as-data from produce_clip) — only shape-checked.
    model_id: Optional[str] = None
    # IDENTITY LOCK (id_lock): jailed abs paths of the subject reference image(s), in
    # order. CANONICAL (they define the identity → part of the content_hash). () = none
    # (a plain i2v/t2v/v2v render). The route jail-resolves + image-classifies each; the
    # VACE runner loads them as PIL and drives reference-to-video conditioning.
    reference_images: tuple[str, ...] = ()
    # OPTIONAL VACE control channel (composition blocking): a single jailed still
    # (control_image) + its kind (control_kind ∈ pose|depth|sketch), used as the VACE
    # `video=` control input (repeated across the frame count) when there is no
    # source_video. Both None = no control. CANONICAL when set.
    control_image: Optional[str] = None
    control_kind: Optional[str] = None
    # VACE-EXTEND temporal conditioning (studio-movie splice motion-carry): the ORDERED
    # abs paths of a parent clip's trailing context frames (oldest -> newest). Set ONLY
    # by the studio-movie runner for a ``vace_extend`` joint — it extracts these and
    # routes the segment through the VACE path (capability "v2v") so the render CONTINUES
    # the parent's motion instead of restarting from a single still. () = not an extend
    # render (every non-movie caller). Threaded into the manifest for the VACE runner;
    # NOT part of the render content_hash (see RenderManifest.vace_context_frames).
    vace_context_frames: tuple[str, ...] = ()
    # CLIP LENGTH, in FRAMES — the caller's ask. None = "unset": the studio spine uses
    # the BOUND model's default (81 frames for a real model — Wan's reference length,
    # measured ~352s on ae's 3090 at 832x480; ``fps * 2`` for the synthetic prover).
    # A number here is threaded verbatim into ``RenderManifest.requested_frames`` and
    # RESOLVED at render time by ``studio.runners.synthetic.resolve_frames`` (clamp to
    # the bound model's real ceiling, snap to the 4k+1 temporal cadence, max(1,n) floor,
    # each with a recorded reason). Plumbed EXACTLY like ``steps``/``cfg``: an optional
    # scalar the route passes through, range-checked here (see the constants above) so
    # every caller gets the same guard, never just the HTTP route.
    #
    # ⚠ THIS FIELD DID NOT EXIST until 2026-07-27, which is what made the whole lever a
    # lie: ``RenderManifest.requested_frames`` was hashed and documented, but no spec
    # could carry a length, so every real render on this fleet was forced to the
    # default. FRAMES (not seconds) is the unit because Wan's 4k+1 cadence is always
    # ODD — ``frames/fps`` is never a whole number of seconds at an even fps (81/16 =
    # 5.0625s), so seconds can only ever be a REQUEST resolved to frames, with the TRUE
    # duration reported back on the Artifact.
    requested_frames: Optional[int] = None


def make_studio_i2v(
    *,
    capability: str = Capability.I2V.value,
    width: int,
    height: int,
    fps: int,
    vram_budget_gb: float = _DEFAULT_VRAM_BUDGET_GB,
    seed: int = 0,
    out_root: Optional[str] = None,
    start_image: Optional[str] = None,
    negative: Optional[str] = None,
    prompt: Optional[str] = None,
    project: Optional[str] = None,
    source_video: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    model_id: Optional[str] = None,
    reference_images: Optional[tuple] = None,
    control_image: Optional[str] = None,
    control_kind: Optional[str] = None,
    vace_context_frames: Optional[tuple] = None,
    requested_frames: Optional[int] = None,
) -> StudioI2VSpec:
    """Validate every field and build the frozen ``StudioI2VSpec``. Raises
    ``ValueError``/``TypeError`` LOCALLY on any structural violation (house
    discipline: a structurally-invalid spec is programmer/caller error caught at
    the boundary, never carried across the bus). Runtime policy failures — an
    unroutable request, an unreadable start image — are NOT validated here; they
    surface as errors-as-data from ``produce_clip`` at run time."""
    if not (isinstance(capability, str) and capability in _VALID_CAPABILITIES):
        raise ValueError(
            f"capability must be one of {sorted(_VALID_CAPABILITIES)}; got {capability!r}")
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    # AUTOFIT: None is a LEGAL value (blank budget -> fit to the serving worker's free
    # VRAM at render time). A NUMBER is the manual override and must be positive; a bad
    # non-None value is still a clean caller error (400 at the route).
    if vram_budget_gb is not None:
        if not isinstance(vram_budget_gb, (int, float)) or isinstance(vram_budget_gb, bool) \
                or vram_budget_gb <= 0:
            raise ValueError(
                f"vram_budget_gb must be a positive number or None (autofit); "
                f"got {vram_budget_gb!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    if start_image is not None and not (isinstance(start_image, str) and start_image.strip()):
        raise ValueError(f"start_image must be a non-empty string or None; got {start_image!r}")
    if source_video is not None and not (isinstance(source_video, str) and source_video.strip()):
        raise ValueError(f"source_video must be a non-empty string or None; got {source_video!r}")
    if negative is not None and not isinstance(negative, str):
        raise ValueError(f"negative must be a string or None; got {negative!r}")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string or None; got {prompt!r}")
    # PROJECT: optional human auto-archive NAME (NON-CANONICAL metadata — never enters the
    # content_hash). A value must be a string; surrounding whitespace is stripped and an
    # all-blank name coerces to None (empty -> None is acceptable — mirrors the
    # gen/scene/movie_schema ``project=(project or None)`` coercion).
    if project is not None and not isinstance(project, str):
        raise ValueError(f"project must be a string or None; got {project!r}")
    project = (project.strip() or None) if isinstance(project, str) else None
    # SAMPLER OVERRIDES: None = unset (model default fills it). A value is range-checked
    # here so the same guard protects EVERY caller (route, preset apply, bus rehydrate),
    # not just the HTTP route. bool is an int subclass — reject it explicitly.
    if steps is not None:
        if not isinstance(steps, int) or isinstance(steps, bool) \
                or not (_MIN_STEPS <= steps <= _MAX_STEPS):
            raise ValueError(
                f"steps must be an int in [{_MIN_STEPS}, {_MAX_STEPS}] or None; got {steps!r}")
    if cfg is not None:
        if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) \
                or not (_MIN_CFG <= cfg <= _MAX_CFG):
            raise ValueError(
                f"cfg must be a number in [{_MIN_CFG}, {_MAX_CFG}] or None; got {cfg!r}")
    # DIRECT MODEL CHOICE: shape-check only (a non-empty string). Whether the model_id
    # actually EXISTS / serves the capability / fits is a routing decision surfaced as
    # errors-as-data (ErrorCode.PINNED_MODEL_UNAVAILABLE) at run time, not here.
    if model_id is not None and not (isinstance(model_id, str) and model_id.strip()):
        raise ValueError(f"model_id must be a non-empty string or None; got {model_id!r}")
    # CLIP LENGTH: None = unset (the bound model's default fills it). A value is
    # range-checked here — the same guard for every caller (route, preset apply, bus
    # rehydrate, movie runner), mirroring the steps/cfg treatment above. This bound is a
    # typo guard, NOT the model ceiling: an over-ceiling ask is legal and CLAMPS at
    # render time with a recorded reason (see _MIN/_MAX_REQUESTED_FRAMES).
    if requested_frames is not None:
        if not isinstance(requested_frames, int) or isinstance(requested_frames, bool) \
                or not (_MIN_REQUESTED_FRAMES <= requested_frames <= _MAX_REQUESTED_FRAMES):
            raise ValueError(
                f"requested_frames must be an int in [{_MIN_REQUESTED_FRAMES}, "
                f"{_MAX_REQUESTED_FRAMES}] or None (unset -> the bound model's default); "
                f"got {requested_frames!r}")

    # IDENTITY LOCK reference images: None -> (); coerce a list/tuple to a tuple (so an
    # asdict->json->from_dict round-trip lands a tuple). Each must be a non-empty string;
    # at most _MAX_REFERENCE_IMAGES (more is a clean caller error, never silently dropped).
    # Existence / jail / image-classification are the ROUTE's job (a runtime input check),
    # not this structural validator.
    if reference_images is None:
        reference_images = ()
    if isinstance(reference_images, (list, tuple)):
        reference_images = tuple(reference_images)
    else:
        raise ValueError(
            f"reference_images must be a list/tuple of paths or None; got {reference_images!r}")
    for i, r in enumerate(reference_images):
        if not (isinstance(r, str) and r.strip()):
            raise ValueError(
                f"reference_images[{i}] must be a non-empty string; got {r!r}")
    if len(reference_images) > _MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"at most {_MAX_REFERENCE_IMAGES} reference_images are accepted; "
            f"got {len(reference_images)}")

    # OPTIONAL VACE control channel: a control image needs a kind, and a kind needs an
    # image — both-or-neither. control_kind must be one of the supported kinds.
    if control_image is not None and not (isinstance(control_image, str) and control_image.strip()):
        raise ValueError(f"control_image must be a non-empty string or None; got {control_image!r}")
    if control_kind is not None:
        if not isinstance(control_kind, str) or control_kind not in _VALID_CONTROL_KINDS:
            raise ValueError(
                f"control_kind must be one of {sorted(_VALID_CONTROL_KINDS)} or None; "
                f"got {control_kind!r}")
    if (control_image is None) != (control_kind is None):
        raise ValueError(
            "control_image and control_kind must be set together (a control still needs "
            "a kind, and a kind needs a still) or both omitted")

    # VACE-EXTEND context frames: None -> (); coerce a list/tuple to a tuple (so an
    # asdict->json->from_dict round-trip lands a tuple). Each must be a non-empty string.
    # Existence is the movie runner's job (it extracts them); this is a structural check.
    if vace_context_frames is None:
        vace_context_frames = ()
    if isinstance(vace_context_frames, (list, tuple)):
        vace_context_frames = tuple(vace_context_frames)
    else:
        raise ValueError(
            f"vace_context_frames must be a list/tuple of paths or None; "
            f"got {vace_context_frames!r}")
    for i, f in enumerate(vace_context_frames):
        if not (isinstance(f, str) and f.strip()):
            raise ValueError(
                f"vace_context_frames[{i}] must be a non-empty string; got {f!r}")

    resolved_out = out_root if (isinstance(out_root, str) and out_root.strip()) \
        else DEFAULT_CLIPS_ROOT

    return StudioI2VSpec(
        capability=capability,
        width=width,
        height=height,
        fps=fps,
        # AUTOFIT: keep None verbatim (resolved at render time); coerce a number to float.
        vram_budget_gb=(float(vram_budget_gb) if vram_budget_gb is not None else None),
        seed=seed,
        out_root=os.path.abspath(resolved_out),
        start_image=start_image,
        negative=negative,
        prompt=prompt,
        project=project,
        source_video=source_video,
        steps=steps,
        cfg=(float(cfg) if cfg is not None else None),
        model_id=model_id,
        reference_images=reference_images,
        control_image=control_image,
        control_kind=control_kind,
        vace_context_frames=vace_context_frames,
        requested_frames=requested_frames,
    )


def studio_i2v_from_dict(d: dict) -> StudioI2VSpec:
    """Rebuild a ``StudioI2VSpec`` from its ``asdict`` form, THROUGH the validating
    factory (mirrors ``movie_schema``'s deserialize-then-revalidate) so a rehydrated
    spec is re-checked, never trusted blind. Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under the name ``"studio_i2v"``."""
    return make_studio_i2v(
        capability=d.get("capability", Capability.I2V.value),
        width=d["width"],
        height=d["height"],
        fps=d["fps"],
        vram_budget_gb=d.get("vram_budget_gb", _DEFAULT_VRAM_BUDGET_GB),
        seed=d.get("seed", 0),
        out_root=d.get("out_root"),
        start_image=d.get("start_image"),
        negative=d.get("negative"),
        prompt=d.get("prompt"),
        project=d.get("project"),
        source_video=d.get("source_video"),
        steps=d.get("steps"),
        cfg=d.get("cfg"),
        model_id=d.get("model_id"),
        reference_images=d.get("reference_images"),
        control_image=d.get("control_image"),
        control_kind=d.get("control_kind"),
        vace_context_frames=d.get("vace_context_frames"),
        # CLIP LENGTH: absent -> None (unset). Tolerates specs enqueued before this
        # field existed AND specs relayed to a studio GPU worker running an older
        # build — both land on the bound model's default, i.e. today's behaviour.
        requested_frames=d.get("requested_frames"),
    )


_FALLBACK_MAX_VRAM_GB = 24.0


def _device_vram_gb() -> Optional[float]:
    """Installed VRAM on THIS box, in GB, or None when unmeasurable.

    ⚠ WHY THIS EXISTS (2026-07-27). ``resolve_studio_env`` took
    ``max_vram_gb: float = 24.0`` and **neither caller ever overrode it** — see
    ``runners/studio_i2v.py::run_produce_clip``, which is the code path on BOTH the
    central and the GPU-worker side. So every studio placement decision the fleet
    has ever made was measured against a hardcoded 24.0, *including on computron,
    an 8 GiB RTX 4060*. The studio believed every box was a 3090.

    That is a defaults-are-promises violation on its own terms: the number that
    decides whether a pipeline goes whole-on-GPU or streams from the CPU was not
    read from any device.

    Sourced from ``_platform.hardware.total_vram_bytes`` — the same probe pair
    (torch ``mem_get_info`` → ``nvidia-smi``) the free-VRAM read uses, so the
    ceiling and the headroom come from ONE truth. It degrades to ``None``, never
    to 0, so "unmeasurable" stays distinguishable from "no card" and the caller
    keeps today's conservative behaviour instead of concluding zero headroom."""
    try:
        from hugpy_platform.hardware import total_vram_bytes
        total = total_vram_bytes()
    except Exception:  # noqa: BLE001 — a probe failure must never fail a render
        return None
    if not total or total <= 0:
        return None
    return float(total) / (1024 ** 3)


def resolve_studio_env(out_root: str, *, master_fps: int,
                       max_vram_gb: float = _FALLBACK_MAX_VRAM_GB) -> StudioEnv:
    """Resolve a concrete ``StudioEnv`` from sensible worker defaults (INV-5) WITHOUT
    requiring any STUDIO_* environment variable: every required field is filled with
    a resolved value (paths under the media-store root, house mastering defaults),
    so a bus runner constructs a complete env with no operator wiring. ``master_fps``
    is threaded from the job's target so the recorded env matches the render intent.
    ``allow_unpinned=True`` (dev posture); the synthetic runner is pinned regardless,
    and ``produce_clip`` never calls ``validate_registry``, so this only affects the
    recorded env snapshot, never routing.

    WEIGHTS ROOT — the SHARED-store seam (k4 fix, 2026-07-18). The studio weights are a
    SHARED asset (373GB Wan/Lightricks), NOT a per-box copy: on a split central+worker
    fleet the render runs on the GPU worker (ae), whose media-store root ``DEFAULT_ROOT``
    is its LOCAL hot drive (``/mnt/hot990/hugpy-worker``) — so deriving the weights root
    from ``DEFAULT_ROOT`` (the old behavior) pointed the manifest's captured
    ``STUDIO_WEIGHTS_ROOT`` at a hot path that never held the weights, and every Wan render
    failed ``weights_missing`` naming ``<DEFAULT_ROOT>/video_intel/studio/weights``. The
    fix: honor an EXPLICIT ``STUDIO_WEIGHTS_ROOT`` (the shared mount, e.g.
    ``/mnt/llm_storage/video_intel/studio/weights``, resolved via ``env_value`` — the .env
    file OR the process env) when the operator sets one, falling back to the
    ``DEFAULT_ROOT``-derived default for the self-hosted SINGLE-box case (where the media
    store IS where the weights live, so the default is correct). This is NOT a smart default
    on the strict ``env.py`` loader (INV-5 untouched); it is honoring an explicit operator
    override on the worker-defaults path, and the resulting value is what ``to_snapshot()``
    records — so the manifest snapshot still matches what actually resolved."""
    # ⚠ max_vram_gb STAYS A STABLE CONSTANT HERE — do NOT make it device-derived.
    # StudioEnv.to_snapshot() emits STUDIO_MAX_VRAM_GB (env.py _REQUIRED), and
    # schemas.py:231 puts env_snapshot INSIDE the clip content_hash. Sourcing this
    # from the card would therefore RE-ADDRESS every clip the moment two boxes
    # disagree (ae reads 23.56, computron 8.0), breaking resume/dedup: an identical
    # spec would re-render instead of reusing its clip. The PLACEMENT decision does
    # need the real card — it reads it live in the runner
    # (runners/wan_i2v._max_vram_gb), which is outside the manifest and so outside
    # addressing. Same split _hot_weights_root() already uses.
    weights_root = env_value("STUDIO_WEIGHTS_ROOT") or os.path.join(STUDIO_ROOT, "weights")
    return StudioEnv(
        output_root=os.path.abspath(out_root),
        weights_root=weights_root,
        manifest_root=os.path.join(STUDIO_ROOT, "manifests"),
        master_colorspace="rec709",
        master_fps=int(master_fps),
        max_vram_gb=float(max_vram_gb),
        loudness_target_lufs=-14.0,
        allow_unpinned=True,
    )
