"""RenderManifest factory + round-trip (de)serializer — the ENFORCEMENT point.

The manifest is the source of truth for a render (INV-1). Until now the three
verified P0-0 content_hash fixes lived only as dataclass DEFAULTS on
``RenderManifest`` (exercised by the conformance suite, never by a live build
path). This module closes that gap: ``make_render_manifest`` is the single
validating factory that BUILDS a manifest, and it threads the fix inputs from
their AUTHORITATIVE sources so they cannot be forgotten or hand-forged:

  * FIX-1  precision            <- binding.precision      (router-selected)
  * FIX-3  determinism_class    <- binding.determinism_class (bound model's class)
  * FIX-5  env_snapshot         <- env.to_snapshot()       (resolved env, INV-5)

Plus model_id / framework / task / weight_hash, also from the binding. These are
NOT free parameters: passing a binding + an env is the only way to fill them, so
a caller can never quietly diverge the recorded precision/determinism/env from
what the router and boot actually resolved.

House convention mirrored from ``movie_schema.make_movie``: validation RAISES
LOCALLY at construction (never across a boundary), and a from-dict path rebuilds
the nested value objects and RE-VALIDATES through the same core on the way back.
Construction-time violations raise ``ValueError`` — the same discipline the
studio value schemas use (``Resolution``/``VramEnvelope.__post_init__``): a
structurally-invalid manifest is programmer error, not runtime policy data (that
is what ``StageError``/``Err`` are for, INV-3).

No pathlib anywhere. os.path only (there is none here — pure data).

TODO(P0-3): register ``render_manifest_from_dict`` in
``media_bus.SPEC_DESERIALIZERS`` when the runner lands. Kept OUT of the bus for
this slice so studio stays dormant (nothing in ``src/`` imports it).
"""

from __future__ import annotations

from hugpy_video.intel.studio.enums import (
    AdapterKind,
    Capability,
    ControlKind,
    DeterminismClass,
    Framework,
    Precision,
    Task,
)
from hugpy_video.intel.studio.env import StudioEnv
from hugpy_video.intel.studio.schemas import (
    AdapterRef,
    ControlRef,
    ModelBinding,
    ProvenanceStub,
    RenderManifest,
    Resolution,
    SamplerConfig,
    SeedBundle,
)


# --------------------------------------------------------------------------- #
# Validating construction core — one place, two entry paths (factory + from_dict)
# --------------------------------------------------------------------------- #
def _build_manifest(
    *,
    render_id: str,
    capability: Capability,
    model_id: str,
    weight_hash: str | None,
    framework: Framework,
    task: Task,
    precision: Precision,
    seeds: SeedBundle,
    sampler: SamplerConfig,
    resolution_ladder: tuple[Resolution, ...],
    controls: tuple[ControlRef, ...],
    adapters: tuple[AdapterRef, ...],
    identity_ids: tuple[str, ...],
    identity_view_hashes: tuple[str, ...],
    determinism_class: DeterminismClass,
    env_snapshot: tuple[tuple[str, str], ...],
    provenance: ProvenanceStub | None,
    prompt: str = "",
    negative_prompt: str = "",
    source_video: str = "",
    reference_images: tuple[str, ...] = (),
    control_image: str = "",
    control_kind: str = "",
    vace_context_frames: tuple[str, ...] = (),
    requested_frames: int | None = None,
) -> RenderManifest:
    """Validate every field, then build the frozen manifest. Raises ``ValueError``
    LOCALLY on any structural violation. Shared by ``make_render_manifest`` (the
    live build path) and ``render_manifest_from_dict`` (rehydration) so both go
    through exactly the same enforcement — a rehydrated manifest is re-validated,
    never trusted blind."""
    # --- identity / metadata ---
    if not (isinstance(render_id, str) and render_id.strip()):
        raise ValueError(f"render_id must be a non-empty string; got {render_id!r}")
    if not isinstance(capability, Capability):
        raise ValueError(
            f"capability must be a Capability enum; got {type(capability).__name__}")

    # --- binding-derived fields (authoritative; here we only type-check them) ---
    if not (isinstance(model_id, str) and model_id.strip()):
        raise ValueError(f"model_id must be a non-empty string; got {model_id!r}")
    if not isinstance(framework, Framework):
        raise ValueError(
            f"framework must be a Framework enum; got {type(framework).__name__}")
    if not isinstance(task, Task):
        raise ValueError(f"task must be a Task enum; got {type(task).__name__}")
    if not isinstance(precision, Precision):
        raise ValueError(
            f"precision must be a Precision enum; got {type(precision).__name__}")
    if not isinstance(determinism_class, DeterminismClass):
        raise ValueError(
            f"determinism_class must be a DeterminismClass enum; "
            f"got {type(determinism_class).__name__}")
    # weight_hash: None is ALLOWED (unpinned/dev model). If present it must be a
    # non-empty string — an empty hash is a silent-collision landmine (INV-1).
    if weight_hash is not None and not (isinstance(weight_hash, str) and weight_hash.strip()):
        raise ValueError(
            f"weight_hash must be a non-empty string or None (unpinned); "
            f"got {weight_hash!r}")

    # --- render-input value objects ---
    if not isinstance(seeds, SeedBundle):
        raise ValueError(f"seeds must be a SeedBundle; got {type(seeds).__name__}")
    if not isinstance(sampler, SamplerConfig):
        raise ValueError(
            f"sampler must be a SamplerConfig; got {type(sampler).__name__}")

    resolution_ladder = tuple(resolution_ladder)
    if not resolution_ladder:
        raise ValueError("resolution_ladder must be non-empty (at least one Resolution)")
    for i, r in enumerate(resolution_ladder):
        if not isinstance(r, Resolution):
            raise ValueError(
                f"resolution_ladder[{i}] must be a Resolution; got {type(r).__name__}")

    controls = tuple(controls)
    for i, c in enumerate(controls):
        if not isinstance(c, ControlRef):
            raise ValueError(
                f"controls[{i}] must be a ControlRef; got {type(c).__name__}")
    adapters = tuple(adapters)
    for i, a in enumerate(adapters):
        if not isinstance(a, AdapterRef):
            raise ValueError(
                f"adapters[{i}] must be an AdapterRef; got {type(a).__name__}")

    identity_ids = tuple(identity_ids)
    for i, s in enumerate(identity_ids):
        if not isinstance(s, str):
            raise ValueError(
                f"identity_ids[{i}] must be a str; got {type(s).__name__}")
    identity_view_hashes = tuple(identity_view_hashes)
    for i, s in enumerate(identity_view_hashes):
        if not isinstance(s, str):
            raise ValueError(
                f"identity_view_hashes[{i}] must be a str; got {type(s).__name__}")

    # env_snapshot: sorted (str, str) pairs. Empty is tolerated by the dataclass
    # default, but a manifest built via the factory always has a populated one
    # (from env.to_snapshot()); here we only enforce the shape.
    norm_env: list[tuple[str, str]] = []
    for pair in env_snapshot:
        if len(pair) != 2:
            raise ValueError(
                f"env_snapshot entries must be (var, value) pairs; got {pair!r}")
        k, v = pair
        if not (isinstance(k, str) and isinstance(v, str)):
            raise ValueError(
                f"env_snapshot pair must be (str, str); got {pair!r}")
        norm_env.append((k, v))

    if provenance is not None and not isinstance(provenance, ProvenanceStub):
        raise ValueError(
            f"provenance must be a ProvenanceStub or None; "
            f"got {type(provenance).__name__}")

    # C-prompt: text conditioning. None is normalized to "" (a caller threading an
    # Optional spec field / an absent JSON key lands here as None); anything else
    # non-str is programmer error. "" is a valid empty prompt.
    if prompt is None:
        prompt = ""
    if negative_prompt is None:
        negative_prompt = ""
    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be a string or None; got {type(prompt).__name__}")
    if not isinstance(negative_prompt, str):
        raise ValueError(
            f"negative_prompt must be a string or None; got {type(negative_prompt).__name__}")

    # source_video (B2 chain): an Optional spec field / absent JSON key lands here as
    # None -> normalized to "" (no source). Anything else non-str is programmer error.
    if source_video is None:
        source_video = ""
    if not isinstance(source_video, str):
        raise ValueError(
            f"source_video must be a string or None; got {type(source_video).__name__}")

    # IDENTITY LOCK reference images (id_lock): None/absent -> () ; each must be a str.
    # ORDER PRESERVED (canonical). A caller threading a list (from JSON) is coerced to
    # a tuple so the frozen manifest stays hashable + order-stable.
    if reference_images is None:
        reference_images = ()
    reference_images = tuple(reference_images)
    for i, r in enumerate(reference_images):
        if not isinstance(r, str):
            raise ValueError(
                f"reference_images[{i}] must be a str; got {type(r).__name__}")

    # OPTIONAL VACE control channel: None/absent -> "" (no control). Both must be str.
    if control_image is None:
        control_image = ""
    if control_kind is None:
        control_kind = ""
    if not isinstance(control_image, str):
        raise ValueError(
            f"control_image must be a string or None; got {type(control_image).__name__}")
    if not isinstance(control_kind, str):
        raise ValueError(
            f"control_kind must be a string or None; got {type(control_kind).__name__}")

    # VACE-EXTEND context frames (movie splice motion-carry): None/absent -> (); each
    # must be a str. ORDER PRESERVED (oldest -> newest, ending at the branch frame). A
    # caller threading a list (from JSON) is coerced to a tuple so the frozen manifest
    # stays hashable + order-stable. NOT part of the content_hash (see the field docstring
    # on RenderManifest) — carried for the runner + provenance sidecar only.
    if vace_context_frames is None:
        vace_context_frames = ()
    vace_context_frames = tuple(vace_context_frames)
    for i, f in enumerate(vace_context_frames):
        if not isinstance(f, str):
            raise ValueError(
                f"vace_context_frames[{i}] must be a str; got {type(f).__name__}")

    # CLIP LENGTH REQUEST: None/absent -> None ("unset — use the bound model's default").
    # SHAPE is programmer error and raises here; RANGE is NOT. A request ABOVE the bound
    # model's ceiling is a legitimate ask that the runner CLAMPS with a recorded reason
    # (``resolve_frames``) — the caller cannot know a model's ceiling at manifest-build
    # time (the model is bound by the router, not by them), so refusing it here would turn
    # "give me the longest clip you can" into a 500. What IS rejected: a bool (an int
    # subclass — ``requested_frames=True`` would silently mean 1 frame), a non-int, and a
    # count below 1 (zero/negative frames is not a clip anyone can want, and the caller CAN
    # be told that without knowing anything about the model). ``resolve_frames`` keeps its
    # own max(1, n) floor regardless, for hand-forged manifests that bypass this factory.
    if requested_frames is not None:
        if not isinstance(requested_frames, int) or isinstance(requested_frames, bool):
            raise ValueError(
                f"requested_frames must be an int or None (unset); "
                f"got {requested_frames!r}")
        if requested_frames < 1:
            raise ValueError(
                f"requested_frames must be >= 1 when set (a clip has at least one "
                f"frame); got {requested_frames!r}")

    return RenderManifest(
        render_id=render_id,
        capability=capability,
        model_id=model_id,
        weight_hash=weight_hash,
        framework=framework,
        task=task,
        precision=precision,
        seeds=seeds,
        sampler=sampler,
        resolution_ladder=resolution_ladder,
        controls=controls,
        adapters=adapters,
        identity_ids=identity_ids,
        identity_view_hashes=identity_view_hashes,
        determinism_class=determinism_class,
        env_snapshot=tuple(norm_env),
        provenance=provenance,
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_video=source_video,
        reference_images=reference_images,
        control_image=control_image,
        control_kind=control_kind,
        vace_context_frames=vace_context_frames,
        requested_frames=requested_frames,
    )


# --------------------------------------------------------------------------- #
# The factory — the ONE live build path. Threads the fixes from their sources.
# --------------------------------------------------------------------------- #
def make_render_manifest(
    *,
    render_id: str,
    capability: Capability,
    binding: ModelBinding,
    seeds: SeedBundle,
    sampler: SamplerConfig,
    resolution_ladder: tuple[Resolution, ...],
    env: StudioEnv,
    controls: tuple[ControlRef, ...] = (),
    adapters: tuple[AdapterRef, ...] = (),
    identity_ids: tuple[str, ...] = (),
    identity_view_hashes: tuple[str, ...] = (),
    provenance: ProvenanceStub | None = None,
    prompt: str = "",
    negative_prompt: str = "",
    source_video: str = "",
    reference_images: tuple[str, ...] = (),
    control_image: str = "",
    control_kind: str = "",
    vace_context_frames: tuple[str, ...] = (),
    requested_frames: int | None = None,
) -> RenderManifest:
    """Build a validated ``RenderManifest`` from a resolved ``ModelBinding`` and a
    resolved ``StudioEnv``.

    ENFORCEMENT: the reproducibility-critical fields are threaded from their
    authoritative sources, never accepted as free parameters, so they cannot be
    forgotten or forged:

      * model_id / framework / task / precision / weight_hash / determinism_class
        come from ``binding`` (the router's answer — precision is FIX-1,
        determinism_class is FIX-3).
      * env_snapshot comes from ``env.to_snapshot()`` (FIX-5 / INV-5): the env
        that actually resolved at boot, as sorted (var, value) pairs.

    ``binding.weight_hash is None`` is threaded through as-is (allowed only for an
    unpinned/dev model). Everything else is validated in ``_build_manifest``;
    violations raise ``ValueError`` locally.

    ``requested_frames`` is the CLIP-LENGTH LEVER and is a FREE parameter (unlike
    precision/determinism/env, which are threaded from their authoritative sources):
    it is the CALLER's ask, not something the router or the boot env can know. It is
    plumbed exactly like the optional ``steps``/``cfg`` sampler overrides — ``None``
    means "unset, let the bound model's default decide". ⚠ Until 2026-07-27 this
    parameter did not exist, so ``RenderManifest.requested_frames`` was unreachable
    from every production path and every real render was forced to the default; the
    field, the docstrings advertising the lever, and the content-hash entry were all
    live while the only way to set it was to hand-construct a manifest in a test.
    """
    if not isinstance(binding, ModelBinding):
        raise ValueError(
            f"binding must be a ModelBinding (from CapabilityRouter.resolve); "
            f"got {type(binding).__name__}")
    if not isinstance(env, StudioEnv):
        raise ValueError(
            f"env must be a StudioEnv (from load_env); got {type(env).__name__}")

    return _build_manifest(
        render_id=render_id,
        capability=capability,
        # --- threaded from the binding (never hand-passed) ---
        model_id=binding.model_id,
        weight_hash=binding.weight_hash,
        framework=binding.framework,
        task=binding.task,
        precision=binding.precision,               # FIX-1
        determinism_class=binding.determinism_class,  # FIX-3
        # --- render inputs ---
        seeds=seeds,
        sampler=sampler,
        resolution_ladder=tuple(resolution_ladder),
        controls=tuple(controls),
        adapters=tuple(adapters),
        identity_ids=tuple(identity_ids),
        identity_view_hashes=tuple(identity_view_hashes),
        # --- threaded from the resolved env (never hand-passed) ---
        env_snapshot=env.to_snapshot(),            # FIX-5
        provenance=provenance,
        # --- text conditioning (C-prompt): part of the reproducibility key ---
        prompt=prompt,
        negative_prompt=negative_prompt,
        # --- source-clip conditioning (B2 chain): part of the reproducibility key ---
        source_video=source_video,
        # --- identity-lock reference images + optional control (part of the key) ---
        reference_images=tuple(reference_images),
        control_image=control_image,
        control_kind=control_kind,
        # --- VACE-extend temporal context (movie splice motion-carry); NOT hashed ---
        vace_context_frames=tuple(vace_context_frames),
        # --- CLIP LENGTH: the caller's ask (part of the reproducibility key). None =
        # unset -> the bound model's default. RESOLVED (clamped/snapped) at render
        # time by runners.synthetic.resolve_frames, never here: the manifest records
        # what was ASKED, the Artifact reports what was DELIVERED. ---
        requested_frames=requested_frames,
    )


# --------------------------------------------------------------------------- #
# Round-trip (de)serializer — JSON-safe dict <-> RenderManifest, re-validated.
# --------------------------------------------------------------------------- #
def render_manifest_to_dict(m: RenderManifest) -> dict:
    """A JSON-safe plain dict of the manifest (enums -> their string values,
    nested value objects -> dicts). The inverse of ``render_manifest_from_dict``:
    ``from_dict(to_dict(m)) == m``, and therefore
    ``from_dict(to_dict(m)).content_hash() == m.content_hash()``.

    ⚠ EVERY canonical field MUST appear here. A field that is in
    ``RenderManifest.canonical_inputs()`` but missing from this dict does not fail
    loudly — it silently rehydrates to its dataclass default, which re-addresses the
    manifest and breaks resume/dedup for anyone who set it. That is precisely how
    ``requested_frames`` shipped broken on 2026-07-27 (hashed, never serialized);
    ``tests/studio/test_clip_length.py`` now asserts the whole-object round trip."""
    return {
        "render_id": m.render_id,
        "capability": m.capability.value,
        "model_id": m.model_id,
        "weight_hash": m.weight_hash,
        "framework": m.framework.value,
        "task": m.task.value,
        "precision": m.precision.value,
        "seeds": {
            "global_seed": m.seeds.global_seed,
            "stage_seeds": [[name, seed] for name, seed in m.seeds.stage_seeds],
            "chunk_seed_base": m.seeds.chunk_seed_base,
        },
        "sampler": {
            "sampler": m.sampler.sampler,
            "scheduler": m.sampler.scheduler,
            "steps": m.sampler.steps,
            "cfg": m.sampler.cfg,
            "shift": m.sampler.shift,
            "sigmas": list(m.sampler.sigmas),
        },
        "resolution_ladder": [[r.width, r.height, r.fps] for r in m.resolution_ladder],
        "controls": [
            {
                "kind": c.kind.value,
                "content_hash": c.content_hash,
                "weight": c.weight,
                "target_frames": list(c.target_frames),
            }
            for c in m.controls
        ],
        "adapters": [
            {
                "kind": a.kind.value,
                "adapter_id": a.adapter_id,
                "weight": a.weight,
                "weight_hash": a.weight_hash,
            }
            for a in m.adapters
        ],
        "identity_ids": list(m.identity_ids),
        "identity_view_hashes": list(m.identity_view_hashes),
        "determinism_class": m.determinism_class.value,
        "env_snapshot": [[var, val] for var, val in m.env_snapshot],
        "prompt": m.prompt,
        "negative_prompt": m.negative_prompt,
        "source_video": m.source_video,
        "reference_images": list(m.reference_images),
        "control_image": m.control_image,
        "control_kind": m.control_kind,
        # VACE-extend context frames: recorded in the sidecar (provenance/round-trip),
        # even though NOT a content_hash input, so a rehydrated manifest reconstructs the
        # exact conditioning the runner consumed.
        "vace_context_frames": list(m.vace_context_frames),
        # CLIP LENGTH request: a CONTENT-HASH INPUT, so serializing it is load-bearing,
        # not cosmetic. ⚠ It was missing from BOTH sides of this round-trip on
        # 2026-07-27 while ``canonical_inputs`` already hashed it — harmless only while
        # the field was permanently None (which it was, being unreachable). The moment
        # the lever became settable, ``from_dict(to_dict(m))`` would have re-addressed
        # to a DIFFERENT content hash than ``m`` for every non-default length, silently
        # breaking resume/dedup: a resumed 33-frame render would look like a cache miss
        # and re-burn ~2 GPU-minutes, and the clip.mp4 sidecar would disagree with the
        # directory it sits in. ``None`` serializes as JSON null and restores as None.
        "requested_frames": m.requested_frames,
        "provenance": (
            None if m.provenance is None else {
                "operator": m.provenance.operator,
                "created_at": m.provenance.created_at,
                "tool": m.provenance.tool,
                "c2pa_pending": m.provenance.c2pa_pending,
            }
        ),
    }


def render_manifest_from_dict(d: dict) -> RenderManifest:
    """Rebuild a ``RenderManifest`` from its ``render_manifest_to_dict`` form.

    Every nested value object (SeedBundle / SamplerConfig / Resolution /
    ControlRef / AdapterRef / ProvenanceStub) is reconstructed, enums are rebuilt
    from their string values, and the whole thing is RE-VALIDATED through
    ``_build_manifest`` (mirroring ``movie_schema``'s deserialize-then-revalidate).
    The binding/env-derived fields (model_id, precision, determinism_class,
    env_snapshot, ...) are read from the serialized manifest — which is itself the
    authoritative record produced at build time — rather than re-threaded from a
    binding, since a rehydration has no live binding."""
    sd = d["seeds"]
    seeds = SeedBundle(
        global_seed=sd["global_seed"],
        stage_seeds=tuple((name, seed) for name, seed in sd["stage_seeds"]),
        chunk_seed_base=sd["chunk_seed_base"],
    )
    pd = d["sampler"]
    sampler = SamplerConfig(
        sampler=pd["sampler"],
        scheduler=pd["scheduler"],
        steps=pd["steps"],
        cfg=pd["cfg"],
        shift=pd["shift"],
        sigmas=tuple(pd["sigmas"]),
    )
    resolution_ladder = tuple(
        Resolution(width=r[0], height=r[1], fps=r[2]) for r in d["resolution_ladder"]
    )
    controls = tuple(
        ControlRef(
            kind=ControlKind(c["kind"]),
            content_hash=c["content_hash"],
            weight=c["weight"],
            target_frames=tuple(c["target_frames"]),
        )
        for c in d["controls"]
    )
    adapters = tuple(
        AdapterRef(
            kind=AdapterKind(a["kind"]),
            adapter_id=a["adapter_id"],
            weight=a["weight"],
            weight_hash=a["weight_hash"],
        )
        for a in d["adapters"]
    )
    prov_d = d["provenance"]
    provenance = (
        None if prov_d is None else ProvenanceStub(
            operator=prov_d["operator"],
            created_at=prov_d["created_at"],
            tool=prov_d["tool"],
            c2pa_pending=prov_d["c2pa_pending"],
        )
    )

    return _build_manifest(
        render_id=d["render_id"],
        capability=Capability(d["capability"]),
        model_id=d["model_id"],
        weight_hash=d["weight_hash"],
        framework=Framework(d["framework"]),
        task=Task(d["task"]),
        precision=Precision(d["precision"]),
        seeds=seeds,
        sampler=sampler,
        resolution_ladder=resolution_ladder,
        controls=controls,
        adapters=adapters,
        identity_ids=tuple(d["identity_ids"]),
        identity_view_hashes=tuple(d["identity_view_hashes"]),
        determinism_class=DeterminismClass(d["determinism_class"]),
        env_snapshot=tuple((str(k), str(v)) for k, v in d["env_snapshot"]),
        provenance=provenance,
        # C-prompt: tolerate manifests serialized before this field existed (absent -> "").
        prompt=d.get("prompt", ""),
        negative_prompt=d.get("negative_prompt", ""),
        # B2 chain: tolerate manifests serialized before source_video existed (absent -> "").
        source_video=d.get("source_video", ""),
        # id_lock: tolerate manifests serialized before these fields existed (absent -> ()/"").
        reference_images=tuple(d.get("reference_images", ())),
        control_image=d.get("control_image", ""),
        control_kind=d.get("control_kind", ""),
        # VACE-extend context: tolerate manifests serialized before this field existed
        # (absent -> ()).
        vace_context_frames=tuple(d.get("vace_context_frames", ())),
        # CLIP LENGTH: tolerate sidecars written before this field was serialized
        # (absent -> None = "unset"), which is also exactly what those older clips
        # hashed with, so an old manifest.json still rehydrates to its own address.
        requested_frames=d.get("requested_frames"),
    )
