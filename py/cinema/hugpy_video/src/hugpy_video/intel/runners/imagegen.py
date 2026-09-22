"""Pure `(diffusers, generate_image)` runner — map §4.4 / §6.

`run_generate_image(spec, job_id) -> JobResult`. A THIN wrapper over the EXISTING
inference plane (`managers.dispatch.execute_prompt`) — it does NOT import
diffusers itself. The prompt is resolved from the spec's ordered parts (text
parts joined). When the spec carries an image part AND img2img is servable on the
fleet, the FIRST image is used as an img2img init image (strength knob; default
0.45); otherwise the image is ignored and text-to-image runs (v1 behavior). The
plane is driven once and the produced image file is re-ingested into a MediaRef.

Pure discipline (map §6): EXPECTED failures (no prompt, an unresolved video
part, the plane raising / returning not-ok, no usable output) are returned as
JobResult(ok=False, JobError(...)) — DATA, never a raise. Only the worker loop
may catch an UNEXPECTED raise.

The inference-plane import is LAZY (inside the function), NOT at module top. That
keeps merely importing `video_intel.runners` (which the ffmpeg crop/frame_extract
runners need) decoupled from the health of the managers/dispatch plane — so a
registry hiccup there can never break the media runners' import path.

ROUTING: `pool` is intentionally left UNSET (see the joined kwargs). With
HUGPY_ML_POOL empty / no pool passed, the plane runs generation in-process
(central/local). If a GPU worker should have taken it, that is a routing symptom
to surface — this runner does NOT force pool="ml".
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time

from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.gen_schema import GenerateImageSpec
from hugpy_video.intel.media_store import ingest
from hugpy_video.intel.result_schema import JobError, JobResult
from hugpy_video.intel.runners._img2img import img2img_available, start_image_required

logger = logging.getLogger(__name__)


def _write_image_bundle(spec: GenerateImageSpec, job_id: str, projectmeta: str,
                        image_path: str, prompt: str,
                        started_at: float, finished_at: float) -> str:
    """Copy the single generated image into ``<DEFAULT_ROOT>/assets/<projectmeta>/``
    and write ``project.json`` beside it. Returns the absolute bundle dir.

    RAISES on IO error — the caller wraps this best-effort so a bundle failure
    never crosses the job boundary. Module-level (no closures) so it is
    unit-testable GPU-free."""
    bundle_dir = os.path.join(DEFAULT_ROOT, "assets", projectmeta)
    os.makedirs(bundle_dir, exist_ok=True)
    rel = os.path.basename(image_path)
    shutil.copyfile(image_path, os.path.join(bundle_dir, rel))
    manifest = {
        "project_name": spec.project,
        "project_uuid": job_id,
        "model_key": spec.model_id,
        "prompt": prompt,
        "negative": spec.negative,
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "guidance": spec.guidance,
        "n_frames": 1,
        "strength": spec.strength,
        "seed": spec.seed if spec.seed is not None else "random",
        "frames": [rel],
        "image": rel,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    with open(os.path.join(bundle_dir, "project.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return bundle_dir


def run_generate_image(spec: GenerateImageSpec, job_id: str) -> JobResult:
    started_at = time.time()
    # ---- safety net: video parts MUST be resolved to image parts upstream ----
    # (video_intel.chains.resolve_video_parts runs in the enqueue path). If a raw
    # video part reaches the runner, refuse loudly rather than silently drop it.
    for i, p in enumerate(spec.parts):
        if p.kind == "video":
            return JobResult(job_id, ok=False, error=JobError(
                code="unresolved_video_part",
                message=(
                    f"part[{i}] is a raw video part; it must be resolved to "
                    "image frames via chains.resolve_video_parts before enqueue"
                ),
                retryable=False,
            ))

    # ---- resolve the prompt: join text parts (ordered); collect image paths ----
    texts = [p.text.strip() for p in spec.parts
             if p.kind == "text" and p.text and p.text.strip()]
    prompt = "\n".join(texts).strip()
    image_paths = [p.media.uri for p in spec.parts
                   if p.kind == "image" and p.media is not None]

    if not prompt:
        # sd-turbo is text-to-image; without any text we have nothing to condition
        # on (image2image conditioning is a future model flag — see below).
        return JobResult(job_id, ok=False, error=JobError(
            code="no_prompt",
            message="generate_image has no text part to build a prompt from",
            retryable=False,
        ))

    # ---- image-conditioning seam: use the FIRST image part as an init image via
    # img2img — but ONLY when img2img is servable on the fleet. Image-generation
    # models advertise image-to-image (config layer — models_config
    # ._augment_img2img); if the selected model can't serve img2img we DO NOT
    # regress the existing "image ignored, text-to-image runs" flow — we still
    # generate from the text and log LOUDLY.
    start_frame = image_paths[0] if image_paths else None

    # ---- HONEST early refusal: an image-to-image-ONLY model (a native edit
    # model like Qwen-Image-Edit — cfg.tasks lists image-to-image but NOT
    # text-to-image) with NO start image USED to fall through to
    # task="text-to-image" below and die LATE inside the plane ("model can't
    # serve t2i"). Refuse UP FRONT + truthfully instead. Gated on
    # img2img_available (inside start_image_required) so a registry/plane outage
    # is never mis-blamed on the caller. Matches the no_prompt error idiom above.
    if start_image_required(spec.model_id, start_frame is not None):
        return JobResult(job_id, ok=False, error=JobError(
            code="start_image_required",
            message="This model edits an image — add a start image.",
            retryable=False,
        ))

    strength = spec.strength if spec.strength is not None else 0.45   # documented default
    use_img2img = start_frame is not None and img2img_available(spec.model_id)
    if start_frame is not None and not use_img2img:
        logger.warning(
            "generate_image %s: %d image part(s) supplied but image-to-image is "
            "NOT available on the fleet (model=%s); ignoring the init image and "
            "running text-to-image (v1 behavior preserved)",
            job_id, len(image_paths), spec.model_id,
        )

    # ---- GPU-worker guard (2026-07-03 central-meltdown fix) ----
    # The guard block was extracted VERBATIM into the shared helper so every
    # generation runner (generate_image, generate_scene, ...) shares ONE refusal
    # policy — behavior is byte-identical (see runners/_gpu_guard.py for the full
    # rationale). Refuses with retryable data when a fleet EXISTS but no live
    # worker serves this model; override with HUGPY_VIDEOGEN_LOCAL=always.
    from hugpy_video.intel.runners._gpu_guard import guard_gpu_worker
    _refusal = guard_gpu_worker(spec.model_id, job_id)
    if _refusal is not None:
        return _refusal

    # ---- live progress: a single image is one shot, so emit ONE 'generating'
    # heartbeat (done=0/total=1) and rely on result.outputs at terminal. ----
    try:
        from hugpy_video.intel.media_bus import set_progress
        set_progress(job_id, {
            "done": 0, "total": 1, "stage": "generating",
            "label": (prompt if len(prompt) <= 80 else prompt[:79] + "…"),
            "model": spec.model_id, "frames": [],
            "started_at": started_at, "eta_s": None,
        })
    except Exception:
        logger.debug("generate_image %s: set_progress failed (non-fatal)",
                     job_id, exc_info=True)

    # ---- build kwargs; leave `pool` UNSET (routing note in the docstring) ----
    if use_img2img:
        logger.info(
            "generate_image %s: IMG2IMG path (model=%s init=%s strength=%s)",
            job_id, spec.model_id, os.path.basename(start_frame), strength,
        )
        kwargs = dict(
            task="image-to-image",
            image_path=start_frame,
            strength=strength,
            prompt=prompt,
            model_key=spec.model_id,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.steps,
            guidance_scale=spec.guidance,
            num_images=1,
            # The plane may serve this on a REMOTE GPU worker: the result's `path`
            # is then worker-local. The b64 bytes are how the artifact reaches
            # this machine (materialized below when the path isn't local).
            return_b64=True,
        )
    else:
        kwargs = dict(
            task="text-to-image",
            prompt=prompt,
            model_key=spec.model_id,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.steps,
            guidance_scale=spec.guidance,
            num_images=1,
            # The plane may serve this on a REMOTE GPU worker: the result's `path`
            # is then worker-local. The b64 bytes are how the artifact reaches
            # this machine (materialized below when the path isn't local).
            return_b64=True,
        )
    # only include optional knobs when set, so a None can't confuse a builder
    if spec.seed is not None:
        kwargs["seed"] = spec.seed
    if spec.negative is not None:
        kwargs["negative_prompt"] = spec.negative

    # ---- drive the EXISTING plane once (lazy import; keep the runner pure) ----
    try:
        from hugpy_video.intel.plane import execute_prompt
        from hugpy_platform.async_runtime import run
        res = run(execute_prompt(**kwargs))
    except Exception as exc:  # unknown model / registry / plane error -> DATA
        return JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=f"inference plane raised: {type(exc).__name__}: {exc}",
            retryable=True,
        ))

    if not getattr(res, "ok", False):
        return JobResult(job_id, ok=False, error=JobError(
            code="generation_failed",
            message=str(getattr(res, "error", None) or "unknown"),
            retryable=True,
        ))

    images = getattr(res, "images", None) or ()
    if not images:
        return JobResult(job_id, ok=False, error=JobError(
            code="generation_no_output",
            message="plane returned ok but produced no images",
            retryable=True,
        ))

    img = images[0]
    path = getattr(img, "path", None)
    if not path or not os.path.isfile(path):
        # The plane served this on a remote worker — its `path` lives on THAT
        # machine's disk. The result rides the bytes for exactly this case:
        # materialize them locally and continue.
        b64 = getattr(img, "b64", None)
        if not b64:
            return JobResult(job_id, ok=False, error=JobError(
                code="generation_no_output",
                message=(f"generated image has no usable file path on disk "
                         f"and no inline bytes: {path!r}"),
                retryable=False,
            ))
        import base64
        from hugpy_platform.constants import UPLOADS_HOME
        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{job_id}_0.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64))

    # Re-ingest so the output's dims/mime are authoritatively resolved (§9.2).
    # (ingest jails the path under UPLOADS_HOME/DEFAULT_ROOT.)
    ref = ingest(path)

    # ---- auto-archive: single image -> assets/<projectmeta>/ bundle ----
    # projectmeta = slug(project name) or the job_id uuid when unnamed. Best-effort:
    # a bundle failure is logged + swallowed (never crosses the job boundary).
    from hugpy_platform.utils import slugify
    projectmeta = slugify(spec.project) if spec.project else job_id
    try:
        _write_image_bundle(spec, job_id, projectmeta, ref.uri, prompt,
                            started_at, time.time())
    except Exception as exc:  # best-effort — do NOT raise across the job boundary
        logger.warning("generate_image %s: auto-archive bundle FAILED (non-fatal): "
                       "%s: %s", job_id, type(exc).__name__, exc)

    return JobResult(
        job_id, ok=True, outputs=(ref,),
        project={"name": spec.project, "uuid": job_id,
                 "dir": f"assets/{projectmeta}"},
    )
