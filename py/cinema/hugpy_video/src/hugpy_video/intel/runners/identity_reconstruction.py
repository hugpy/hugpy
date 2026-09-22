"""Identity RECONSTRUCTION bus runner (studio stage (b)) — the orchestrator that
turns a saved identity profile + a description into an identity-locked turnaround
set (one still per view) awaiting the operator's approval.

Pure ``(IdentityReconstructionSpec, job_id) -> JobResult`` (map §6): expected
failures are DATA (a ``JobResult(ok=False, JobError(...))``), never a raise; only a
genuine programmer error raises, and ``media_bus.run_claimed`` is the one place that
catches that. Like ``runners/studio_movie``, this is a FAT orchestrator: it drives
each per-view render INLINE through the shared render primitive (``render_clip``),
so a single bus ``job_id`` covers the whole set and the UI polls one job.

It does three things:
  1. For each requested view, render an identity-locked still via ``_render_identity_view``
     (the SWAP SEAM — see its docstring).
  2. On completion, hand the produced stills to
     ``identity_profiles.attach_reconstruction`` so they land in the identity's own
     directory (copied in, manifested, appended to the profile) for approval.
  3. Catalog the owned stills as image outputs on the JobResult.

Heavy imports (the studio spine, media_store, the frame runner) are LAZY — done
inside the functions — so importing this module (which ``runners/__init__`` does at
boot) stays cheap and can never break app boot. No pathlib anywhere; os.path only.
"""
from __future__ import annotations

import logging

from hugpy_video.intel.result_schema import JobError, JobResult

logger = logging.getLogger(__name__)


def _view_prompt(base: str, view: str) -> str:
    """The per-view render prompt: the base description (the profile's notes /
    request prompt) + a view-specific, neutral-turnaround suffix. An empty base is
    handled gracefully (no dangling leading comma)."""
    label = (view or "").replace("_", " ").strip()
    base = (base or "").strip()
    lead = f"{base}, " if base else ""
    return f"{lead}{label} view, neutral pose, plain background"


def _orbit_prompt(base: str) -> str:
    """The turntable render prompt: the base description (the profile's notes / request
    prompt) + a suffix that drives ONE full 360° turntable orbit of the SAME subject,
    so every rendered frame is a degree-view of the character. An empty base is handled
    gracefully (no dangling leading comma)."""
    base = (base or "").strip()
    lead = f"{base}, " if base else ""
    return (
        f"{lead}full body, the same subject slowly rotating a full 360 degrees on a "
        "turntable, T-pose neutral pose, static camera, plain neutral background, "
        "consistent lighting"
    )


def _render_identity_turntable(
    refs,
    prompt: str,
    seed: int,
    *,
    width: int,
    height: int,
    fps: int,
    render_id: str,
    max_frames: int,
    should_cancel=None,
    negative_prompt: str = "",
) -> "list[str] | None":
    """Turntable sibling of ``_render_identity_view`` (the SWAP SEAM): render ONE id_lock
    ``studio_i2v`` ORBIT clip (the subject rotating a full 360°) through the shared render
    primitive, then extract EVERY frame of that clip as an ordered angular sequence of
    stills — frame 0 = the start angle ... frame N = ~360°.

    A studio clip's LENGTH is spine-derived (~2s, model-capped) and canNOT be lengthened
    via a frame count, so this renders one short orbit clip and keeps ALL its frames at the
    clip's native fps (``make_frame_extract(fps=<clip fps>, max_frames=max_frames)``) to get
    the dense degree-views. ``max_frames`` is the LOUD cap the frame runner enforces.

    Returns the ORDERED list of frame uris (angular order), or ``None`` on any render /
    extract failure (the runner converts a ``None`` into a clean error-as-data JobResult
    for the whole reconstruction job — mirrors ``_render_identity_view``)."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot).
    from hugpy_video.intel import media_store
    from hugpy_video.intel.frame_schema import make_frame_extract
    from hugpy_video.intel.studio.job import make_studio_i2v
    from hugpy_video.intel.runners.ffmpeg_frames import run_frame_extract
    from hugpy_video.intel.runners.studio_i2v import render_clip

    # id_lock studio_i2v spec: geometry at the caller's (<=480p) budget, autofit VRAM,
    # the subject reference images driving the Wan-VACE reference-to-video conditioning.
    # The ORBIT prompt (not a frame count) is what makes the clip a 360° turntable.
    i2v_spec = make_studio_i2v(
        capability="id_lock",
        width=width,
        height=height,
        fps=fps,
        vram_budget_gb=None,       # autofit — size to the serving worker's free VRAM
        seed=seed,
        prompt=prompt,
        # CLEANUP-PROMPT slice: forward the TRUE negative (see _render_identity_view). ""
        # (default) -> byte-identical to today's turntable render call.
        negative=negative_prompt,
        reference_images=tuple(refs),
    )
    outcome = render_clip(i2v_spec, render_id=render_id, should_cancel=should_cancel)
    if not outcome.ok or not outcome.path:
        logger.warning("identity turntable render failed (%s): %s", render_id,
                       getattr(outcome.error, "message", "no clip path"))
        return None

    # Extract EVERY frame of the orbit clip via the EXISTING frame_extract job (reuse — no
    # new ffmpeg code). Sample at the clip's NATIVE fps (frames/duration when the render
    # reports both, else the requested fps) so each rendered frame surfaces exactly once,
    # in angular order. The clip lives on the shared store -> ingest it as a video MediaRef.
    clip_ref = media_store.ingest(outcome.path, kind_hint="video")
    clip_fps = float(fps)
    if outcome.frames and outcome.duration_s and outcome.duration_s > 0:
        clip_fps = outcome.frames / outcome.duration_s
    fe_spec = make_frame_extract(
        source=clip_ref, fps=clip_fps, quality=2, fmt="png", max_frames=max_frames,
    )
    fr = run_frame_extract(fe_spec, render_id + ":frames")
    if not fr.ok or not fr.outputs:
        logger.warning("identity turntable frame-extract failed (%s): %s", render_id,
                       getattr(fr.error, "message", "no frames"))
        return None
    # ffmpeg writes frame_%05d.png; run_frame_extract sorts the glob, so outputs are in
    # capture (== angular) order. One uri per degree-view.
    return [o.uri for o in fr.outputs]


def _render_identity_view(
    refs,
    prompt: str,
    seed: int,
    *,
    width: int,
    height: int,
    fps: int,
    render_id: str,
    should_cancel=None,
    negative_prompt: str = "",
) -> "str | None":
    """====================== SWAP SEAM (option b -> option a) ======================
    THE SINGLE place the identity-render backend is chosen. TODAY it drives the WORKING
    Wan-VACE id_lock path (option b): build an id_lock ``studio_i2v`` spec from the
    subject reference images, render a SHORT clip through the shared render primitive
    (``render_clip`` — delegated to the studio GPU worker when one is resolvable), then
    extract frame 0 of that clip as the still via the existing ``frame_extract`` job.

    To swap in the comfy still path (option a — currently provisioning-gated) LATER,
    REPLACE THE BODY of this one function with the comfy call and return the produced
    still's abs path. Nothing else in the reconstruction flow (the route, the runner
    loop, the store) references the render mechanism — this is the only coupling point.
    ==============================================================================

    CLEANUP-PROMPT slice: ``negative_prompt`` is a TRUE negative FORWARDED to the studio
    Wan-VACE render (``make_studio_i2v(negative=...)`` -> ``StudioI2VSpec.negative`` ->
    ``run_produce_clip`` -> ``produce_clip(negative_prompt=...)`` -> the manifest). The
    studio path ALREADY accepts it, so this is a FORWARD, not a new capability. Default
    "" -> the studio path's own default negative_prompt="" (``make_studio_i2v`` with a ""
    negative and ``run_produce_clip``'s ``"" or ""``), so an empty negative reproduces
    today's exact render call byte-for-byte.

    Returns the abs path of the produced still, or ``None`` on any render / extract
    failure (the runner converts a ``None`` into a clean error-as-data JobResult for
    the whole reconstruction job)."""
    # Lazy imports keep this module boot-cheap (runners/__init__ imports it at boot).
    from hugpy_video.intel import media_store
    from hugpy_video.intel.frame_schema import make_frame_extract
    from hugpy_video.intel.studio.job import make_studio_i2v
    from hugpy_video.intel.runners.ffmpeg_frames import run_frame_extract
    from hugpy_video.intel.runners.studio_i2v import render_clip

    # id_lock studio_i2v spec: geometry at the caller's (<=480p) budget, autofit VRAM,
    # the subject reference images driving the Wan-VACE reference-to-video conditioning.
    i2v_spec = make_studio_i2v(
        capability="id_lock",
        width=width,
        height=height,
        fps=fps,
        vram_budget_gb=None,       # autofit — size to the serving worker's free VRAM
        seed=seed,
        prompt=prompt,
        # CLEANUP-PROMPT slice: the TRUE negative rides StudioI2VSpec.negative. "" (default)
        # -> run_produce_clip forwards negative_prompt="" -> the studio path's byte-identical
        # today-behavior (produce_clip already defaults negative_prompt="").
        negative=negative_prompt,
        reference_images=tuple(refs),
    )
    outcome = render_clip(i2v_spec, render_id=render_id, should_cancel=should_cancel)
    if not outcome.ok or not outcome.path:
        logger.warning("identity view render failed (%s): %s", render_id,
                       getattr(outcome.error, "message", "no clip path"))
        return None

    # Extract frame 0 of the rendered clip via the EXISTING frame_extract job (reuse —
    # no new ffmpeg code). The clip lives on the shared store, so ingest it as a video
    # MediaRef and sample the first frame (fps=1 over a short clip -> frame ~0 first).
    clip_ref = media_store.ingest(outcome.path, kind_hint="video")
    fe_spec = make_frame_extract(source=clip_ref, fps=1, quality=2, fmt="png")
    fr = run_frame_extract(fe_spec, render_id + ":frame")
    if not fr.ok or not fr.outputs:
        logger.warning("identity view frame-extract failed (%s): %s", render_id,
                       getattr(fr.error, "message", "no frames"))
        return None
    return fr.outputs[0].uri   # frame 0 (the earliest sampled frame)


def run_identity_reconstruction(spec, job_id: str) -> JobResult:
    """Render one identity-locked still per requested view, then attach the set to the
    profile for approval. Returns ``JobResult(ok=True, outputs=(view stills,))`` on
    success; a per-view render failure or a vanished profile is an error-as-data."""
    from hugpy_video.intel import identity_profiles, media_store
    from hugpy_video.intel.media_bus import is_cancelling

    refs = list(spec.source_images)

    # Relay THIS reconstruction job's cancel down to the GPU render. render_clip's default
    # cancel probe watches the CHILD render_id (e.g. "<job>:<recon>:turntable"), which the
    # UI cancel — issued against the owning job_id — never sets, so a plain reconstruction
    # cancel was a no-op. is_cancelling(job_id) flips the worker's denoise-loop interrupt,
    # so cancelling now actually STOPS the render (mirrors studio_movie.py's should_cancel).
    should_cancel = lambda: is_cancelling(job_id)  # noqa: E731

    def _cancelled() -> JobResult:
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code="cancelled",
            message=f"identity reconstruction for profile {spec.slug!r} cancelled by user",
            retryable=False))

    # ---------------------------- TURNTABLE mode ------------------------------ #
    # ONE id_lock orbit clip, EVERY frame kept as an angular degree-view. The record's
    # ordered ``views`` hold the frames in angular order; the manifest exposes
    # ``mode``/``frame_count``/``degrees_per_frame`` so the UI can drive the scrub viewer.
    if getattr(spec, "mode", "sheet") == "turntable":
        orbit_prompt = _orbit_prompt(spec.base_prompt)
        frames = _render_identity_turntable(
            refs, orbit_prompt, spec.seed,
            width=spec.width, height=spec.height, fps=spec.fps,
            render_id=f"{job_id}:{spec.recon_id}:turntable",
            max_frames=spec.turntable_max_frames,
            should_cancel=should_cancel,
            # CLEANUP-PROMPT slice: forward the spec's TRUE negative. "" (default on the
            # spec) -> byte-identical to today. getattr keeps an OLD spec (no field) working.
            negative_prompt=getattr(spec, "negative_prompt", "") or "",
        )
        if not frames:
            if should_cancel():
                return _cancelled()
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="render_failed",
                message=(f"identity turntable reconstruction failed rendering the orbit "
                         f"clip for profile {spec.slug!r}"),
                retryable=True))
        frame_count = len(frames)
        record = identity_profiles.attach_reconstruction(
            spec.slug, spec.recon_id, frames,
            spec={"job_id": job_id, "mode": "turntable",
                  "frame_count": frame_count,
                  "degrees_per_frame": (round(360.0 / frame_count, 2)
                                        if frame_count else None),
                  "prompt": spec.base_prompt, "orbit_prompt": orbit_prompt,
                  "seed": spec.seed},
        )
        if record is None:
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="profile_gone",
                message=(f"identity profile {spec.slug!r} is no longer active; the "
                         "turntable frames rendered but could not be attached"),
                retryable=False))
        outputs: list = []
        for p in (record.get("views") or []):
            try:
                outputs.append(media_store.ingest(p, kind_hint="image"))
            except Exception:  # noqa: BLE001 — a frame that won't ingest never fails the job
                pass
        return JobResult(job_id=job_id, ok=True, outputs=tuple(outputs))

    # ------------------------------ SHEET mode -------------------------------- #
    # The EXISTING path (unchanged): N independent view-stills, one id_lock clip per named
    # view, only frame 0 kept.
    view_names = list(spec.views)
    view_paths: list = []
    prompts: list = []
    for i, view in enumerate(view_names):
        # Honor a cancel BETWEEN views (mirrors the movie's between-segment check) so a
        # multi-view sheet stops promptly, not only mid-clip.
        if should_cancel():
            return _cancelled()
        vp = _view_prompt(spec.base_prompt, view)
        prompts.append(vp)
        # A DISTINCT render_id per view (the worker keys renders by id — like a movie's
        # per-segment ids — so views are never deduped as one "exists" render).
        still = _render_identity_view(
            refs, vp, spec.seed,
            width=spec.width, height=spec.height, fps=spec.fps,
            render_id=f"{job_id}:{spec.recon_id}:view_{i:02d}",
            should_cancel=should_cancel,
            # CLEANUP-PROMPT slice: forward the spec's TRUE negative. "" (default on the
            # spec) -> byte-identical to today. getattr keeps an OLD spec (no field) working.
            negative_prompt=getattr(spec, "negative_prompt", "") or "",
        )
        if still is None:
            if should_cancel():
                return _cancelled()
            return JobResult(job_id=job_id, ok=False, error=JobError(
                code="render_failed",
                message=(f"identity reconstruction failed rendering view {view!r} "
                         f"(index {i}) for profile {spec.slug!r}"),
                retryable=True))
        view_paths.append(still)

    # On completion of the view renders: persist the produced stills into the identity's
    # own dir for approval (copies them in, writes the manifest, appends to the profile).
    record = identity_profiles.attach_reconstruction(
        spec.slug, spec.recon_id, view_paths,
        spec={"job_id": job_id, "prompt": spec.base_prompt, "seed": spec.seed,
              "prompts": prompts, "view_names": view_names},
    )
    if record is None:
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code="profile_gone",
            message=(f"identity profile {spec.slug!r} is no longer active; the "
                     "reconstruction stills rendered but could not be attached"),
            retryable=False))

    # Catalog the OWNED stills as image outputs so GET /video/jobs/<job_id> carries the
    # produced views (the UI also re-reads the profile for the full recon manifest).
    outputs: list = []
    for p in (record.get("views") or []):
        try:
            outputs.append(media_store.ingest(p, kind_hint="image"))
        except Exception:  # noqa: BLE001 — a still that won't ingest never fails the job
            pass
    return JobResult(job_id=job_id, ok=True, outputs=tuple(outputs))
