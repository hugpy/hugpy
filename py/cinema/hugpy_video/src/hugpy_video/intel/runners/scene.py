"""Pure `(diffusers, generate_scene)` runner — one query -> N consecutive frames.

`run_generate_scene(spec, job_id) -> JobResult`. Two coherence modes, chosen by
whether the spec carries a START-FRAME image part (the FIRST image part):

  * NO start frame -> v1: seed + prompt-schedule. The runner walks n_frames
    SEQUENTIALLY, derives a per-frame prompt (base + a positional tag + an
    optional motion schedule) and a per-frame seed (base_seed + i), and drives
    the SAME text-to-image plane (managers.dispatch.execute_prompt) once/frame.

  * START-FRAME present -> IMG2IMG (image-to-image plane). If img2img is not
    available on the fleet (the sd-turbo advertisement flip is HELD), returns a
    retryable `image_to_image_unavailable` JobError — an HONEST failure, never a
    silent fall back to text-to-image. When available:
      - chain=False (default): every frame conditions on the start frame (no
        drift) with base + motion step i.
      - chain=True: frame 0 conditions on the start frame; each later frame
        conditions on the PREVIOUS frame's saved output (true sequential
        chaining) with base + motion step i. This is the LEGACY pixel chain the
        oracle directive prohibits: the runner forces chain=False unless the
        operator opts in with HUGPY_LEGACY_CHAIN=1 (movie_schema.
        legacy_chain_enabled), and declares it on project.json as
        legacy_pixel_chain + label 'legacy (pixel-chained)' when it actually ran.

Each frame is materialized to a padded frame_%05d.png under DEFAULT_ROOT and
re-ingested. When spec.assemble, the frames are muxed into a browser-playable
H.264 mp4 (yuv420p + faststart) that is ingested LAST (so it classifies as
kind="video").

Pure discipline (map §6): EXPECTED failures (no prompt, an unresolved video part,
the plane raising / returning not-ok / no usable output, assembly failing) are
returned as JobResult(ok=False, JobError(...)) — DATA, never a raise. Only the
worker loop may catch an UNEXPECTED raise.

Import/purity discipline mirrors imagegen.py: the inference-plane import is LAZY
(inside the loop) so importing `video_intel.runners` never couples to the health
of the managers/dispatch plane.

CONCURRENCY BOUND: only the mp4 ASSEMBLY subprocess is serialized behind a
module-level BoundedSemaphore(1) (like ffmpeg_frames' fan-out). Generation is
remote-GPU-bound and is intentionally NOT gated here.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, replace

from hugpy_platform.binaries import resolve_bin
from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult
from hugpy_video.intel.scene_schema import GenerateSceneSpec, FRAME_CAP
from hugpy_video.intel.movie_schema import LEGACY_CHAIN_LABEL, effective_chain
from hugpy_video.intel.runners._gpu_guard import guard_gpu_worker
from hugpy_video.intel.runners._img2img import img2img_available, start_image_required

logger = logging.getLogger(__name__)

# Serialize ONLY the mp4 assembly subprocess (see module docstring). Generation
# is remote-GPU-bound and is not gated here.
_SCENE_SEM = threading.BoundedSemaphore(1)


def _assemble_scene_mp4(frame_dir: str, mp4_path: str, fps: int) -> None:
    """Mux frame_%05d.png in `frame_dir` into a browser-playable H.264 mp4.

    Raises RuntimeError on ffmpeg failure; the runner converts it into a
    scene_assembly_failed JobError (never a raise across the job boundary). The
    scale filter forces even dimensions (libx264 + yuv420p require them).
    """
    ffmpeg = resolve_bin("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path,
    ]
    with _SCENE_SEM:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(mp4_path):
        raise RuntimeError(
            f"ffmpeg assembly failed rc={result.returncode}: {(result.stderr or '')[-500:]}"
        )


def _scene_concurrency(model_id: str, n_frames: int) -> int:
    """How many frames to generate at once. HUGPY_SCENE_CONCURRENCY pins it;
    otherwise self-tune to the number of ONLINE workers assigned this model
    (the delegating plane's least-recently-picked routing spreads concurrent
    requests across them, and each worker serializes same-model generates via
    the runners' per-model lock — so extra in-flight frames just queue).
    Floor 1 = today's sequential behavior on a one-worker fleet."""
    env = (os.environ.get("HUGPY_SCENE_CONCURRENCY") or "").strip()
    if env:
        try:
            return max(1, min(int(env), n_frames))
        except ValueError:
            pass
    try:
        # Placement seam: fleet-installed registry via hugpy_engine.placement;
        # the null default lists nothing -> sequential.
        from hugpy_engine.placement import get_worker_registry
        live = sum(
            1 for w in get_worker_registry().list_workers(online_only=True)
            if w.get("status", "online") == "online" and model_id in (w.get("models") or [])
        )
        return max(1, min(live, n_frames))
    except Exception:
        return 1  # registry unavailable (e.g. worker-side import) — sequential


def _generate_frames_parallel(frame_kwargs: "list[dict]", out_dir: str,
                              job_id: str, model_id: str, on_frame_done=None):
    """Tier-1 scene fan-out: generate INDEPENDENT frames concurrently.

    Only for orders where frames don't feed each other — v1 (seed-schedule
    text-to-image) and img2img chain=False (every frame off the START frame).
    chain=True stays strictly sequential at the call site. Cooperative cancel:
    frames not yet started become no-ops once 'cancelling' is set; in-flight
    frames finish (same between-frames contract as the sequential path).

    ``on_frame_done(frame_path, i)`` (when given) is invoked IN FRAME ORDER as
    each frame lands — it re-ingests the frame and emits live progress. It runs
    in the CALLING thread (the map/loop body), never a pool worker, so its refs
    list stays single-threaded + ordered. Callers rely on it to populate refs;
    the returned path list is retained for symmetry only.
    Returns (ordered_frame_paths, None) or (None, JobResult error)."""
    from concurrent.futures import ThreadPoolExecutor
    from hugpy_video.intel.media_bus import is_cancelling

    n = len(frame_kwargs)
    workers = _scene_concurrency(model_id, n)
    if workers <= 1:
        paths: list = []
        for i, kw in enumerate(frame_kwargs):
            if is_cancelling(job_id):
                return None, JobResult(job_id, ok=False, error=JobError(
                    code="cancelled",
                    message=f"cancelled after {i} of {n} frame(s)",
                    retryable=False))
            frame_path, err = _generate_one_frame(kw, out_dir, i, job_id)
            if err is not None:
                return None, err
            paths.append(frame_path)
            if on_frame_done is not None:
                on_frame_done(frame_path, i)
        return paths, None

    logger.info("scene %s: fan-out %d frames across up to %d concurrent "
                "generates (model=%s)", job_id, n, workers, model_id)

    def _task(i: int, kw: dict):
        if is_cancelling(job_id):
            return i, None, "cancelled"
        frame_path, err = _generate_one_frame(kw, out_dir, i, job_id)
        return i, frame_path, err

    # pool.map yields in INPUT order, so emitting per landed frame here is
    # in-order + monotonic (a slow early frame just batches the later emits). We
    # stop emitting once any frame errors/cancels — refs is discarded on error.
    results: dict = {}
    saw_error = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame_path, err in pool.map(
                lambda args: _task(*args), enumerate(frame_kwargs)):
            results[i] = (frame_path, err)
            if err is not None:
                saw_error = True
            elif frame_path is not None and on_frame_done is not None and not saw_error:
                on_frame_done(frame_path, i)

    cancelled = sum(1 for p, e in results.values() if e == "cancelled")
    if cancelled:
        done = sum(1 for p, e in results.values() if e is None)
        return None, JobResult(job_id, ok=False, error=JobError(
            code="cancelled",
            message=f"cancelled after {done} of {n} frame(s)",
            retryable=False))
    for i in range(n):  # first REAL error by frame order, deterministic
        _path, err = results[i]
        if err is not None:
            return None, err
    return [results[i][0] for i in range(n)], None


def _generate_one_frame(kwargs: dict, out_dir: str, i: int, job_id: str):
    """Drive the EXISTING inference plane once and materialize the result to a
    SEQUENTIAL padded frame_%05d.png. Returns (frame_path, None) on success, or
    (None, JobResult) carrying a JobError on any expected failure.

    Shared by BOTH the v1 (text-to-image) and img2img loops so the plane-drive +
    materialize logic (and its DATA error shapes) exist in exactly one place. The
    inference-plane import stays LAZY (keeps this runner pure)."""
    try:
        from hugpy_video.intel.plane import execute_prompt
        from hugpy_platform.async_runtime import run
        res = run(execute_prompt(**kwargs))
    except Exception as exc:  # unknown model / registry / plane error -> DATA
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=f"inference plane raised on frame {i}: {type(exc).__name__}: {exc}",
            retryable=True,
        ))

    if not getattr(res, "ok", False):
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=f"frame {i}: {str(getattr(res, 'error', None) or 'unknown')}",
            retryable=True,
        ))

    images = getattr(res, "images", None) or ()
    if not images:
        return None, JobResult(job_id, ok=False, error=JobError(
            code="generation_no_output",
            message=f"plane returned ok but produced no image for frame {i}",
            retryable=True,
        ))

    frame_path = os.path.join(out_dir, f"frame_{i:05d}.png")
    img = images[0]
    src_path = getattr(img, "path", None)
    if src_path and os.path.isfile(src_path):
        shutil.copyfile(src_path, frame_path)
    else:
        # Plane served this remotely: its `path` lives on THAT machine; the
        # inline b64 bytes are how the artifact reaches this machine.
        b64 = getattr(img, "b64", None)
        if not b64:
            return None, JobResult(job_id, ok=False, error=JobError(
                code="generation_no_output",
                message=(f"frame {i}: generated image has no usable file path "
                         f"on disk and no inline bytes: {src_path!r}"),
                retryable=False,
            ))
        with open(frame_path, "wb") as fh:
            fh.write(base64.b64decode(b64))
    return frame_path, None


def _write_bundle(spec: GenerateSceneSpec, job_id: str, projectmeta: str,
                  frame_paths: "list[str]", mp4_path: "str | None",
                  base_prompt: str, started_at: float, finished_at: float,
                  per_frame_secs: "list[float]") -> str:
    """Copy the scene's frame PNGs + assembled mp4 into a self-contained
    ``<DEFAULT_ROOT>/assets/<projectmeta>/`` bundle and write ``project.json``
    beside them. Returns the absolute bundle dir.

    RAISES on IO error — the caller wraps this best-effort (mirrors the
    assembly-failure handling), so a bundle failure never crosses the job
    boundary. Kept module-level (no closures) so it is unit-testable with a
    stubbed frame set, GPU-free."""
    bundle_dir = os.path.join(DEFAULT_ROOT, "assets", projectmeta)
    os.makedirs(bundle_dir, exist_ok=True)

    frame_relnames: "list[str]" = []
    for fp in frame_paths:
        rel = os.path.basename(fp)
        shutil.copyfile(fp, os.path.join(bundle_dir, rel))
        frame_relnames.append(rel)

    mp4_rel = None
    if mp4_path and os.path.isfile(mp4_path):
        mp4_rel = "video.mp4"
        shutil.copyfile(mp4_path, os.path.join(bundle_dir, mp4_rel))

    seeds = ([spec.seed + i for i in range(spec.n_frames)]
             if spec.seed is not None else "random")
    # chaining only actually happens with a start frame AND the legacy opt-in;
    # without a start frame the v1 seed+schedule path never conditions on frame i.
    has_start = any(p.kind == "image" and p.media is not None for p in (spec.parts or ()))
    chained = effective_chain(spec.chain) and has_start
    manifest = {
        "project_name": spec.project,
        "project_uuid": job_id,
        "model_key": spec.model_id,
        "prompt": base_prompt,
        "negative": spec.negative,
        "chain": chained,
        "legacy_pixel_chain": chained,
        "label": (LEGACY_CHAIN_LABEL if chained else "independent"),
        "limitations": (["legacy pixel chain: frame i+1 conditions on frame i (runners/scene.py chain=True, "
                         "opted in with HUGPY_LEGACY_CHAIN=1). The oracle directive prohibits chained "
                         "generation; use the video.performance recipe for sibling segments, pass "
                         "chain=false, or unset HUGPY_LEGACY_CHAIN."] if chained else []),
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "guidance": spec.guidance,
        "n_frames": spec.n_frames,
        "fps": spec.fps,
        "strength": spec.strength,
        "seeds": seeds,
        "frames": frame_relnames,
        "mp4": mp4_rel,
        "started_at": started_at,
        "finished_at": finished_at,
        "per_frame_secs": per_frame_secs,
    }
    with open(os.path.join(bundle_dir, "project.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return bundle_dir


def render_scene_frames(
    *,
    model_id: str,
    base_prompt: str,
    n_frames: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed,
    motion,
    negative,
    strength,
    chain: bool,
    start_frame,
    out_dir: str,
    job_id: str,
    on_frame_done,
):
    """EXTRACTED frame-generation core — render ``n_frames`` frames into
    ``out_dir`` via the THREE frame shapes and return None on success or a
    JobResult(ok=False) on any EXPECTED failure (DATA, never a raise across the
    boundary).

    Shared verbatim by ``run_generate_scene`` (scene-level bundle/progress) and
    ``runners.movie.run_generate_movie`` (movie-level bundle/progress): the CALLER
    owns progress + bundle; this core only drives the inference plane and calls
    ``on_frame_done(frame_path, i)`` per landed frame IN ORDER so the caller can
    re-ingest + record it (its refs list stays single-threaded + ordered).

    The three shapes (chosen by start_frame + chain, exactly as before):
      * NO start_frame            -> v1 text-to-image (seed + prompt-schedule),
        fanned out across the fleet.
      * start_frame + chain=True  -> TRUE sequential img2img chaining (frame i+1
        conditions on frame i's output).
      * start_frame + chain=False -> img2img off the START frame, fanned out.

    The same up-front guards run HERE so both callers share ONE policy:
    start_image_required, the init-image size gate, img2img_available, and the
    shared GPU-worker guard. ``strength`` defaults to the LOCKED 0.45 when None."""
    from hugpy_video.intel.media_bus import is_cancelling

    # Runner-applied default when None (LOCKED CONTRACT): 0.45.
    strength = strength if strength is not None else 0.45

    # ---- HONEST early refusal (mirrors imagegen) ----
    # An image-to-image-ONLY model (a native edit model like Qwen-Image-Edit)
    # with NO start frame must refuse UP FRONT, not fall through to text-to-image
    # and die LATE inside the plane. Gated on img2img_available so a registry/
    # plane outage isn't mis-blamed on the caller.
    if start_image_required(model_id, start_frame is not None):
        return JobResult(job_id, ok=False, error=JobError(
            code="start_image_required",
            message="This model edits an image — add a start image.",
            retryable=False,
        ))

    # ---- init-image size gate ----
    # A degenerate init (a thumbnail attached instead of the full image) dies as
    # an opaque VAE error; refuse it HERE with the real dimensions. 64px floor.
    _MIN_INIT_PX = 64
    if start_frame is not None:
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(start_frame) as _im:
                _w, _h = _im.size
        except Exception as exc:  # unreadable init is the same honest failure
            return JobResult(job_id, ok=False, error=JobError(
                code="init_image_unreadable",
                message=f"start-frame image could not be read ({type(exc).__name__}: {exc})",
                retryable=False,
            ))
        if min(_w, _h) < _MIN_INIT_PX:
            return JobResult(job_id, ok=False, error=JobError(
                code="init_image_too_small",
                message=(f"start-frame image is {_w}x{_h}px — too small for "
                         f"image-to-image (minimum {_MIN_INIT_PX}px on the short "
                         f"side)."),
                retryable=False,
            ))

    # A start-frame image REQUIRES a servable img2img pair. NEVER silently fall
    # back to text-to-image; a model that genuinely can't serve img2img yields an
    # honest, retryable failure (the movie orchestrator uses this to decide
    # whether cross-segment carry is feasible).
    if start_frame is not None and not img2img_available(model_id):
        logger.info(
            "render_scene_frames %s: start-frame present but image-to-image is "
            "not available on the fleet (model=%s); honest failure", job_id, model_id,
        )
        return JobResult(job_id, ok=False, error=JobError(
            code="image_to_image_unavailable",
            message="image-to-image not available on the fleet",
            retryable=True,
        ))

    # ---- GPU-worker guard (shared refusal policy; ADDITIONAL to the img2img probe) ----
    _refusal = guard_gpu_worker(model_id, job_id)
    if _refusal is not None:
        return _refusal

    os.makedirs(out_dir, exist_ok=True)

    if start_frame is not None:
        # ---- img2img path: condition each frame on an init image ----
        mode = "chain" if chain else "parallel"
        logger.info(
            "render_scene_frames %s: IMG2IMG-%s (model=%s start_frame=%s "
            "strength=%s n=%d)", job_id, mode, model_id,
            os.path.basename(start_frame), strength, n_frames,
        )

        def _img2img_kwargs(i: int, cond_path: str) -> dict:
            # chain:    frame 0 = base prompt; frame i>0 = base + motion step i.
            # parallel: every frame = base + motion step i (all off the start frame).
            include_motion = bool(motion) and (chain is False or i > 0)
            prompt = base_prompt
            if include_motion:
                # .replace (NOT .format): user strings may contain stray braces.
                prompt += ", " + motion.replace("{i}", str(i + 1)).replace("{n}", str(n_frames))
            kwargs = dict(
                task="image-to-image",
                image_path=cond_path,
                strength=strength,
                prompt=prompt,
                model_key=model_id,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images=1,
                return_b64=True,
            )
            if seed is not None:
                kwargs["seed"] = seed + i
            if negative is not None:
                kwargs["negative_prompt"] = negative
            return kwargs

        if chain:
            # TRUE sequential chaining — frame i+1 consumes frame i's output.
            prev_output = start_frame
            for i in range(n_frames):
                # Cooperative cancel — honored BETWEEN frames (mid-frame
                # inference is never interrupted).
                if is_cancelling(job_id):
                    return JobResult(job_id, ok=False, error=JobError(
                        code="cancelled",
                        message=f"cancelled after {i} of {n_frames} frame(s)",
                        retryable=False))
                frame_path, err = _generate_one_frame(
                    _img2img_kwargs(i, prev_output), out_dir, i, job_id)
                if err is not None:
                    return err
                prev_output = frame_path
                on_frame_done(frame_path, i)
        else:
            # parallel mode: every frame conditions on the START frame — fan out.
            _paths, err = _generate_frames_parallel(
                [_img2img_kwargs(i, start_frame) for i in range(n_frames)],
                out_dir, job_id, model_id, on_frame_done)
            if err is not None:
                return err
    else:
        # ---- v1 path: seed + prompt-schedule (NO img2img) ----
        logger.info(
            "render_scene_frames %s: V1 text-to-image (seed+prompt-schedule; "
            "model=%s n=%d)", job_id, model_id, n_frames,
        )

        def _v1_kwargs(i: int) -> dict:
            prompt = base_prompt + f", frame {i + 1} of {n_frames}"
            if motion:
                prompt += ", " + motion.replace("{i}", str(i + 1)).replace("{n}", str(n_frames))
            kwargs = dict(
                task="text-to-image",
                prompt=prompt,
                model_key=model_id,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images=1,
                return_b64=True,
            )
            if seed is not None:
                kwargs["seed"] = seed + i
            if negative is not None:
                kwargs["negative_prompt"] = negative
            return kwargs

        _paths, err = _generate_frames_parallel(
            [_v1_kwargs(i) for i in range(n_frames)],
            out_dir, job_id, model_id, on_frame_done)
        if err is not None:
            return err

    return None


def run_generate_scene(spec: GenerateSceneSpec, job_id: str) -> JobResult:
    started_at = time.time()
    # ---- defensive frame cap (belt-and-suspenders; the factory already guards) ----
    if spec.n_frames > FRAME_CAP:
        return JobResult(job_id, ok=False, error=JobError(
            code="frame_cap_exceeded",
            message=f"n_frames={spec.n_frames} exceeds cap {FRAME_CAP}",
            retryable=False,
        ))

    # ---- safety net: video parts MUST be resolved to image parts upstream ----
    # (chains.resolve_video_parts_scene runs in the enqueue path). If a raw video
    # part reaches the runner, refuse loudly rather than silently drop it.
    for i, p in enumerate(spec.parts):
        if p.kind == "video":
            return JobResult(job_id, ok=False, error=JobError(
                code="unresolved_video_part",
                message=(
                    f"part[{i}] is a raw video part; it must be resolved to "
                    "image frames via chains.resolve_video_parts_scene before enqueue"
                ),
                retryable=False,
            ))

    # ---- resolve the base prompt: join text parts (ordered) ----
    texts = [p.text.strip() for p in spec.parts
             if p.kind == "text" and p.text and p.text.strip()]
    base_prompt = "\n".join(texts).strip()
    if not base_prompt:
        return JobResult(job_id, ok=False, error=JobError(
            code="no_prompt",
            message="generate_scene has no text part to build a prompt from",
            retryable=False,
        ))

    # ---- start-frame image (img2img seam) ----
    # The FIRST image part is the init/start frame. Reuse the existing parts list
    # (no new init_image field); its local path is media.uri (parts materialize
    # exactly as in generate_image).
    start_frame = next(
        (p.media.uri for p in spec.parts if p.kind == "image" and p.media is not None),
        None,
    )
    # The frame-generation core (render_scene_frames, below) owns the img2img/v1
    # guards (start_image_required, the init-image size gate, img2img_available)
    # + the shared GPU-worker guard + the three-shape frame loop, so BOTH
    # run_generate_scene and runners.movie.run_generate_movie share ONE policy.
    # This wrapper keeps only the scene-level prompt resolution, live progress,
    # assembly, and auto-archive bundle.

    # Legacy pixel chain gate (board or-k2 / or-p1): chain is forced OFF unless
    # the operator opted in with HUGPY_LEGACY_CHAIN=1 — shared with runners/movie.py.
    if spec.chain and not effective_chain(spec.chain):
        logger.info("scene %s: chain=true requested but HUGPY_LEGACY_CHAIN is off — "
                    "rendering every frame off the start frame (chain=false)", job_id)
        spec = replace(spec, chain=False)

    out_dir = os.path.join(DEFAULT_ROOT, "video_intel", "scenes", job_id)
    os.makedirs(out_dir, exist_ok=True)
    _write_spec_sidecar(out_dir, job_id, "generate_scene", spec)

    refs = []
    done_frame_paths: "list[str]" = []   # frame PNG paths, in order (for the bundle)
    per_frame_secs: "list[float]" = []   # wall-clock secs between frame completions
    total = spec.n_frames
    _frame_clock = [started_at]          # last-completion ts (mutable holder)
    _label_prompt = base_prompt if len(base_prompt) <= 80 else base_prompt[:79] + "…"
    _chained = bool(spec.chain and start_frame is not None)

    def _emit(stage: str, done: int, label: str) -> None:
        """Build + persist the live progress blob (best-effort — never raises)."""
        elapsed = time.time() - started_at
        eta = round((elapsed / done) * (total - done), 2) if done > 0 else None
        blob = {
            "done": done,
            "total": total,
            "stage": stage,
            "label": label,
            "model": spec.model_id,
            "legacy_pixel_chain": _chained,
            "mode": (LEGACY_CHAIN_LABEL if _chained else "independent"),
            # completed FRAMES only (kind=='image'); the mp4 is a terminal output,
            # never a gallery frame. Same MediaRef dict shape as result.outputs so
            # the UI's frame renderer works unchanged.
            "frames": [asdict(r) for r in refs if r.kind == "image"],
            "started_at": started_at,
            "eta_s": eta,
        }
        try:
            from hugpy_video.intel.media_bus import set_progress
            set_progress(job_id, blob)
        except Exception:
            logger.debug("scene %s: set_progress failed (non-fatal)",
                         job_id, exc_info=True)

    def on_frame_done(frame_path: str, i: int) -> None:
        """Per-frame hook: re-ingest the landed frame (dims/mime resolved, §9.2),
        record it, and emit live progress carrying every completed frame so far
        (live-as-generated gallery)."""
        now = time.time()
        per_frame_secs.append(round(now - _frame_clock[0], 3))
        _frame_clock[0] = now
        refs.append(ingest(frame_path))
        done_frame_paths.append(frame_path)
        done = len(done_frame_paths)
        _emit("generating", done, f"frame {done}/{total} — {_label_prompt}")

    # Initial heartbeat: the poller sees a progress object the moment the job runs
    # (the plane may still be loading the model before frame 1 lands).
    _emit("loading", 0, f"loading {spec.model_id} — 0/{total}")

    # ---- render the frames via the shared three-shape core ----
    # The core owns the img2img/v1 guards + the frame loop; it calls on_frame_done
    # per landed frame so this wrapper's refs/live-progress closures stay intact.
    err = render_scene_frames(
        model_id=spec.model_id,
        base_prompt=base_prompt,
        n_frames=spec.n_frames,
        width=spec.width,
        height=spec.height,
        steps=spec.steps,
        guidance=spec.guidance,
        seed=spec.seed,
        motion=spec.motion,
        negative=spec.negative,
        strength=spec.strength,
        chain=spec.chain,
        start_frame=start_frame,
        out_dir=out_dir,
        job_id=job_id,
        on_frame_done=on_frame_done,
    )
    if err is not None:
        return err

    # ---- optional assembly: frames -> browser-playable mp4 (ingested LAST) ----
    bundle_mp4 = None
    if spec.assemble:
        _emit("assembling", total, "assembling mp4…")
        mp4_path = os.path.join(out_dir, f"{job_id}.mp4")
        try:
            _assemble_scene_mp4(out_dir, mp4_path, spec.fps)
        except Exception as exc:  # ffmpeg mux failure -> DATA (not retryable)
            return JobResult(job_id, ok=False, error=JobError(
                code="scene_assembly_failed",
                message=str(exc),
                retryable=False,
            ))
        # ingest LAST so the video ref is outputs[-1] (classifies as kind="video").
        refs.append(ingest(mp4_path))
        bundle_mp4 = mp4_path

    # ---- auto-archive: self-contained assets/<projectmeta>/ bundle ----
    # projectmeta = slug(project name) or the job_id uuid when unnamed. Best-effort:
    # a bundle failure is logged and swallowed (mirrors the assembly guard) — it
    # must never fail an otherwise-successful generation across the job boundary.
    from hugpy_platform.utils import slugify
    projectmeta = slugify(spec.project) if spec.project else job_id
    _emit("archiving", total, f"archiving bundle assets/{projectmeta}")
    try:
        _write_bundle(
            spec=spec, job_id=job_id, projectmeta=projectmeta,
            frame_paths=done_frame_paths, mp4_path=bundle_mp4,
            base_prompt=base_prompt, started_at=started_at,
            finished_at=time.time(), per_frame_secs=per_frame_secs,
        )
    except Exception as exc:  # best-effort — do NOT raise across the job boundary
        logger.warning("scene %s: auto-archive bundle FAILED (non-fatal): %s: %s",
                       job_id, type(exc).__name__, exc)

    return JobResult(
        job_id, ok=True, outputs=tuple(refs),
        project={"name": spec.project, "uuid": job_id,
                 "dir": f"assets/{projectmeta}"},
    )


def _write_spec_sidecar(out_dir: str, job_id: str, kind: str, spec_obj) -> None:
    """Drop prompt.txt + manifest.json NEXT TO the render (2026-08-13, operator
    ask: "the prompt info is not associated clearly with the directory in which
    the media is saved" — until now the spec lived only in media_jobs.db and
    every output dir shipped bare). Best-effort — a sidecar failure must never
    fail a render."""
    try:
        import dataclasses as _dc
        import json as _json
        import time as _time
        d = _dc.asdict(spec_obj) if _dc.is_dataclass(spec_obj) else (
            spec_obj if isinstance(spec_obj, dict) else {"repr": repr(spec_obj)})
        prompts = [p.get("text") for p in (d.get("parts") or [])
                   if isinstance(p, dict) and p.get("kind") == "text" and p.get("text")]
        prompts += [g.get("prompt") for g in (d.get("goals") or [])
                    if isinstance(g, dict) and g.get("prompt")]
        if d.get("prompt"):
            prompts.append(d["prompt"])
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            _json.dump({"job_id": job_id, "kind": kind, "created": _time.time(),
                        "spec": d}, fh, indent=1, default=str)
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(p for p in prompts if p) + "\n")
    except Exception:  # noqa: BLE001 — sidecars are documentation, never load-bearing
        pass
