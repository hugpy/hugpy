"""img2img availability probe for the video arm.

`img2img_available(model_id)` is ADDITIONAL to (never a replacement for) the GPU
guard in `_gpu_guard.py`. It answers ONE question: can the managers inference
plane actually serve ("transformers","image-to-image") for this model right now?

True iff BOTH:
  (a) the managers registry has ("transformers","image-to-image") wired — a
      runner AND a request builder are registered; AND
  (b) it is SERVABLE for `model_id` — the model advertises the task, i.e. a
      resolve for (model_id, task="image-to-image") would succeed. `resolve()`
      raises when the model's cfg.tasks does not list the task.

(b) is SERVABLE when the model's cfg.tasks lists "image-to-image". The config
layer now advertises image-to-image for every image-generation checkpoint
(SD/SDXL/flux-class diffusers, comfy SD-lineage checkpoints — see
models_config._augment_img2img), so image models return True here, while a
genuinely-incapable model (a text LLM, whisper, embeddings, …) returns the honest
FALSE -> "not available on the fleet".

Import is LAZY inside the function (mirrors the runners' discipline): merely
importing this module never couples to the health of the managers/dispatch plane.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_IMG2IMG_KEY = ("transformers", "image-to-image")
_TEXT2IMG_KEY = ("transformers", "text-to-image")


def _pair_wired(key) -> bool:
    """Is ``(framework, task)`` served by the engine — a runner AND a request
    builder registered? Asks the engine's task registry (``hugpy_engine.tasks``,
    where media/video plugins register) first, then the engine's own
    framework/builder tables (looked up by name so a table the engine retires
    reads as "not wired", never as an import error)."""
    import importlib

    framework, task = key
    try:
        from hugpy_engine.tasks import task_spec
        spec = task_spec(task)
        if spec is not None and spec.build_request is not None and (
                not spec.frameworks or framework in spec.frameworks):
            return True
    except Exception as exc:  # noqa: BLE001 - registry unavailable -> fall through
        logger.info("_pair_wired: task registry unavailable (%s)", exc)
    try:
        fw = importlib.import_module("hugpy_engine.resolvers.categories.frameworks")
        bl = importlib.import_module("hugpy_engine.resolvers.categories.builders")
    except Exception as exc:  # managers plane unhealthy -> not available
        logger.info("_pair_wired: engine registry import failed (%s)", exc)
        return False
    runners = getattr(fw, "FRAMEWORK_RUNNERS", {}) or {}
    builders = getattr(bl, "MODEL_REQUEST_BUILDERS", {}) or {}
    return key in runners and key in builders


def img2img_available(model_id: str) -> bool:
    """Return True iff the fleet can serve image-to-image for `model_id`."""
    # (a) registry has the pair wired (runner + builder).
    if not _pair_wired(_IMG2IMG_KEY):
        logger.info("img2img_available: ('transformers','image-to-image') not registered")
        return False

    # (b) servable for this model — resolve_model_key validates the model
    # advertises the task (task in cfg.tasks). Image-generation checkpoints now
    # advertise image-to-image (models_config._augment_img2img); a model that
    # genuinely can't (text LLM, whisper, embeddings, …) raises here => False.
    try:
        from hugpy_engine.resolvers.model_resolver import resolve_model_key
        resolve_model_key(model_key=model_id, task="image-to-image")
    except Exception as exc:
        logger.info("img2img_available: %s cannot serve image-to-image (%s)",
                    model_id, exc)
        return False
    return True


def text_to_image_available(model_id: str) -> bool:
    """Return True iff the fleet can serve text-to-image for `model_id`.

    Symmetric sibling of `img2img_available` (same two-part probe, same lazy
    import discipline). It answers: does this model advertise text-to-image AND
    is the pair wired in the plane? Crucially it is FALSE for an
    image-to-image-ONLY model — a native edit checkpoint like Qwen-Image-Edit
    whose cfg.tasks lists "image-to-image" but NOT "text-to-image" — because
    `resolve_model_key(..., task="text-to-image")` raises for it (task not in
    cfg.tasks). No hand-rolled catalog parsing: resolution is the single source
    of truth for "does this model support this task".

    True iff BOTH:
      (a) the managers registry has ("transformers","text-to-image") wired — a
          runner AND a request builder are registered; AND
      (b) it is SERVABLE for `model_id` — a resolve for
          (model_id, task="text-to-image") would succeed.
    """
    # (a) registry has the pair wired (runner + builder).
    if not _pair_wired(_TEXT2IMG_KEY):
        logger.info("text_to_image_available: ('transformers','text-to-image') not registered")
        return False

    # (b) servable for this model — resolve_model_key validates the model
    # advertises the task (task in cfg.tasks). An image-to-image-ONLY edit model
    # (Qwen-Image-Edit, flux-klein edit checkpoints, …) raises here => False.
    try:
        from hugpy_engine.resolvers.model_resolver import resolve_model_key
        resolve_model_key(model_key=model_id, task="text-to-image")
    except Exception as exc:
        logger.info("text_to_image_available: %s cannot serve text-to-image (%s)",
                    model_id, exc)
        return False
    return True


def start_image_required(model_id: str, has_start_image: bool) -> bool:
    """Guard predicate for generate_image / generate_scene: refuse EARLY and
    HONESTLY when a caller gives NO start image to an image-to-image-ONLY model.

    True iff (no start image) AND the model is image-to-image-ONLY, i.e.
    text-to-image is NOT servable for it but image-to-image IS. Without this an
    edit-only model (Qwen-Image-Edit) with no init image falls through to
    task="text-to-image" and dies LATE inside the plane ("model can't serve
    t2i"); this returns True so the runner can refuse up front with
    `start_image_required`.

    Gated on `img2img_available` so a registry/plane OUTAGE (which also makes
    text_to_image_available False) is never mis-reported to the user as "you
    forgot the image" — in that case this returns False and the normal (late,
    retryable) failure path is preserved.
    """
    if has_start_image:
        return False
    return not text_to_image_available(model_id) and img2img_available(model_id)
