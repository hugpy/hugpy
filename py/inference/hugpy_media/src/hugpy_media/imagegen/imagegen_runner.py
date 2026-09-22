"""Text-to-image runner.

Serves ("transformers", "text-to-image"). One diffusers pipeline per
model_key (class-level singleton cache — same pattern as
FeatureExtractionRunner), generation runs in a worker thread.

diffusers/torch are imported lazily inside the .pipeline property, so
importing this module doesn't require either library to be installed.
Only callers that actually generate pay the import cost — and if it
fails, the error fires at first use, not at dispatch import time.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
from typing import Any, Dict

from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_platform.constants import UPLOADS_HOME
from hugpy_storage.download_models import ensure_model

from hugpy_media.imagegen.schemas import GeneratedImage, ImageGenRequest, ImageGenResult
from hugpy_media.imagegen.vram_retry import is_retryable_vram_failure, settle_delay_s

logger = logging.getLogger(__name__)

# Per-model GENERATE serialization, shared by both runners. A diffusers
# pipeline object is NOT safe under concurrent __call__ (scheduler state
# races) — and the scene fan-out (video_intel/runners/scene.py) deliberately
# issues concurrent frame requests that may land on the same worker. Different
# models still generate in parallel; same-model calls queue.
_GEN_LOCKS: Dict[str, threading.Lock] = {}
_GEN_LOCKS_GUARD = threading.Lock()


def _generate_lock(model_key: str) -> threading.Lock:
    with _GEN_LOCKS_GUARD:
        lock = _GEN_LOCKS.get(model_key)
        if lock is None:
            lock = _GEN_LOCKS[model_key] = threading.Lock()
        return lock


def _record_battery(req, result, secs: float, axis: str) -> None:
    """One model-battery row per generation (ok AND failed). Telemetry only —
    no-raise by contract, so it can never alter the ImageGenResult returned or
    break the image path. The shared util (``abstract_hugpy_dev.model_battery``)
    is itself guarded; this is the belt to its braces."""
    try:
        from hugpy_media import model_battery

        if not model_battery.enabled():
            return
        run = model_battery.run_for_session("imagegen")
        if run is None:
            return
        uri = ""
        if result.ok and result.images:
            uri = result.images[0].path or ""
        run.record(
            model=req.model_key,
            axis=axis,
            ok=bool(result.ok),
            secs=secs,
            uri=uri,
            thumb_b64=model_battery.thumb_b64_for(uri),
            error=None if result.ok else (result.error or "unknown"),
        )
    except Exception:
        logger.debug("model battery record failed (non-fatal)", exc_info=True)


def _evict_idle_pipelines(cache: Dict[str, Any], keep: str) -> list[str]:
    """Free the VRAM held by cached pipelines other than ``keep``, skipping any
    model that is mid-generation (its per-model generate lock is held — it gets
    evicted on a later load instead). MUST be called while holding the caller's
    ``_LOCK`` and BEFORE the new pipeline loads, so a model switch frees the old
    model's VRAM before allocating the new one rather than transiently needing
    both.

    Without this the cache was UNBOUNDED: every distinct image model ever
    generated stayed resident forever, so a 24 GB card filled with ~20 GB of
    stale pipelines while central reported "0 models on GPU" (these live
    in-process, not in a tracked slot). Bounds each runner's cache to the single
    model in use (worst case one text2img + one img2img resident concurrently).

    A victim is torn down ONLY while we hold its per-model generate lock, taken
    non-blocking: if a thread is mid-generation with it (or about to be), the
    acquire fails and we skip it (evicted on a later load). Non-blocking is what
    keeps this deadlock-free — we already hold the cache _LOCK, and a generating
    thread holds the generate lock then wants _LOCK, so a *blocking* acquire here
    would be a classic ABBA. Holding the generate lock during teardown makes
    remove_all_hooks()/del atomic against generation, so we never strip the
    accelerate cpu-offload hooks off a pipeline another thread is calling."""
    victims: list[str] = []
    for k in list(cache):
        if k == keep:
            continue
        gl = _generate_lock(k)
        if not gl.acquire(blocking=False):
            continue                       # mid-generation — leave it for later
        try:
            pipe = cache.pop(k, None)
            try:
                if hasattr(pipe, "remove_all_hooks"):
                    pipe.remove_all_hooks()   # drop accelerate cpu-offload hooks
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            del pipe
            victims.append(k)
        finally:
            gl.release()
    if victims:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — no torch/cuda: nothing to release
            pass
        logger.info("imagegen: evicted idle pipeline(s) to free VRAM: %s", victims)
    return victims


# ---------------------------------------------------------------------------
# Placement (Slice C) — diffusers does NOT take device_map/max_memory the
# transformers way, so the honest spill mechanism for a t2i/i2v pipeline is
# diffusers' own CPU-offload API, driven by the SAME placement seam the
# transformers loaders read (spill.n_gpu_layers_intent / alloc_mode_env — no
# parallel intent reader is invented here):
#   * CPU-leaning intent (ram-only / n_gpu_layers "off"/0)  -> sequential CPU
#     offload: submodules stream to the GPU one at a time and return to RAM —
#     the smallest possible VRAM footprint (slowest), matching "keep it off the
#     card". Binds even with a GPU present (the operator asked for RAM).
#   * fit-and-spill (max-ram alloc_mode) -> model CPU offload: whole submodules
#     are offloaded to RAM and pulled onto the GPU only while active — big model,
#     one consumer card, no OOM.
#   * default (no intent / gpu-only / auto that fits) -> today's `.to(cuda)`,
#     BYTE-IDENTICAL when the seam is silent (defaults-are-promises).
# A pipeline class without an offload method (genuine capability gap) is logged
# ONCE and falls back to .to(device) rather than silently ignoring the mode.
def _place_diffusers_pipeline(pipe, cuda: bool, model_key: str) -> str:
    """Place a diffusers pipeline per the allocation seam. Returns a short label
    of what was applied (for the load log). Mutates ``pipe`` in place (both
    .to(...) and enable_*_cpu_offload() act on the object)."""
    if not cuda:
        pipe.to("cpu")
        return "cpu"

    # Read the placement intent from the shared seam — never a parallel reader.
    try:
        from hugpy_engine.spill import n_gpu_layers_intent, alloc_mode_env
        intent = n_gpu_layers_intent()          # "gpu" | "cpu" | "auto"
        alloc_mode = alloc_mode_env()           # "max-ram" | "explicit" | None
    except Exception as exc:  # noqa: BLE001 — no seam: today's path, logged
        logger.warning("imagegen: placement seam unavailable (%s); using "
                       ".to(cuda)", exc)
        pipe.to("cuda")
        return "cuda (seam unavailable)"

    def _offload(method: str, label: str) -> str:
        fn = getattr(pipe, method, None)
        if not callable(fn):
            # Genuine capability gap: SAY so once, don't silently ignore the mode.
            logger.warning(
                "imagegen: model=%s pipeline %s has no %s() — cannot honor the "
                "'%s' placement; loading fully on the GPU with .to(cuda) instead",
                model_key, type(pipe).__name__, method, label,
            )
            pipe.to("cuda")
            return f"cuda ({label} unsupported by this pipeline)"
        try:
            fn()                                # offload methods mutate in place
            return label
        except Exception as exc:  # noqa: BLE001 — offload failed: honest fallback
            logger.warning(
                "imagegen: model=%s %s() failed (%s) — falling back to .to(cuda)",
                model_key, method, exc,
            )
            pipe.to("cuda")
            return f"cuda ({label} failed)"

    if intent == "cpu":
        return _offload("enable_sequential_cpu_offload", "ram-only/sequential-offload")
    if alloc_mode == "max-ram":
        return _offload("enable_model_cpu_offload", "max-ram/model-offload")

    # gpu-only / auto / no intent -> unchanged historical path.
    pipe.to("cuda")
    return "cuda"


# ---------------------------------------------------------------------------
# Honest footprint pricing + quant election + deterministic VRAM unwind (k66).
#
# The RULING (operator 2026-07-31) for EVERY load call: "route -> is moe? is
# 4bit? BE that size -> queue evict -> allocate, serve." For a diffusers
# pipeline the honest size is the artifact's REAL bytes on disk, priced BEFORE
# the load — never an fp16 election that balloons past the card. The t2i path
# (ImageGenRunner) used to skip this entirely: it loaded fp16 and .to(cuda)
# with no ladder, so FLUX.2-klein (transformer ~18 GiB + Qwen3 text_encoder
# ~16 GiB of bf16 weights, despite the "bucket" in its name) ballooned to
# 22.37 GiB and OOM'd on a ~22.4-GiB-free card. Both image runners now share
# ONE priced loader: no path is exempt (load-call-pipeline ruling).
# ---------------------------------------------------------------------------

# bnb nf4 stores quantized params at ~0.5 byte/param plus a small scale/absmax
# overhead — empirically ~3.6x smaller than a bf16/fp16 param (2 bytes). Only
# the transformer + text_encoder are quantized (the VAE stays fp16), but those
# are ~99% of the weight bytes, so pricing the whole load against /_QUANT_SHRINK
# is honest-to-slightly-conservative.
_QUANT_SHRINK = 3.6


def _quantize_mode() -> str:
    """auto (default) | always | never. HUGPY_IMAGEGEN_QUANTIZE governs both
    image runners; the older HUGPY_IMG2IMG_QUANTIZE name is still honored so no
    existing operator override silently changes meaning."""
    return (os.environ.get("HUGPY_IMAGEGEN_QUANTIZE")
            or os.environ.get("HUGPY_IMG2IMG_QUANTIZE")
            or "auto").lower()


def _weight_bytes(model_dir: str) -> int:
    """The on-disk weight bytes (.safetensors/.bin) under ``model_dir`` — the
    honest footprint the pipeline occupies at its stored precision. This is the
    number stage 2 prices against, NOT a momentary free-VRAM guess."""
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for fn in files:
            if fn.endswith((".safetensors", ".bin")):
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    return total


def _free_vram_bytes() -> "int | None":
    """Budgetable free VRAM via the shared spill seam (operator reserve already
    subtracted), falling back to the raw torch probe. None when unmeasurable."""
    try:
        from hugpy_engine.spill import free_vram_bytes
        fv = free_vram_bytes()
        if fv is not None:
            return fv
    except Exception:  # noqa: BLE001 — no seam: try the raw probe
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001 — no cuda / can't tell
        pass
    return None


def _should_quantize(weight_bytes: int, free_vram: "int | None", mode: str) -> bool:
    """Pure pricing decision (stage 2): quantize to 4-bit when the honest fp16
    footprint would not fit the budgetable free VRAM (85% headroom for activations
    + the transient load arena). ``always``/``never`` are operator overrides. When
    free VRAM is unmeasurable in ``auto`` we do NOT quantize (fits assumed) — the
    load-and-place path stays the historical one rather than guessing blind."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    if free_vram is None:
        return False
    return weight_bytes > free_vram * 0.85


def _log_plan(plan: Dict[str, Any]) -> None:
    fv = plan.get("free_vram")
    logger.warning(
        "imagegen PLAN: model=%s weights=%.1fGiB free_vram=%s "
        "planned_footprint=%.1fGiB decision=%s",
        plan["model"], plan["weight_bytes"] / 2 ** 30,
        ("%.1fGiB" % (fv / 2 ** 30)) if fv else "unknown",
        plan["planned_bytes"] / 2 ** 30, plan["decision"],
    )


def _elect_quantization(model_dir: str, model_key: str, cuda: bool):
    """Stage 2 of the load pipeline: price the planned footprint from real disk
    bytes, decide fp16-vs-4bit, log the PLAN line, and build the quant config.
    Returns ``(quant_config_or_None, plan)``. Never raises — a missing
    bitsandbytes/diffusers quant stack degrades to a priced fp16 plan (and the
    load may then legitimately need CPU-offload or refuse; that is admission's
    call, not a silent OOM)."""
    weight_bytes = _weight_bytes(model_dir)
    plan: Dict[str, Any] = {
        "model": model_key, "weight_bytes": weight_bytes,
        "free_vram": None, "decision": "fp16", "planned_bytes": weight_bytes,
    }
    if not cuda:
        plan["decision"] = "cpu (fp32)"
        _log_plan(plan)
        return None, plan

    mode = _quantize_mode()
    free_vram = _free_vram_bytes()
    plan["free_vram"] = free_vram

    if not _should_quantize(weight_bytes, free_vram, mode):
        plan["decision"] = "fp16-whole"
        _log_plan(plan)
        return None, plan

    try:
        import bitsandbytes  # noqa: F401 — availability probe
        import torch
        from diffusers import PipelineQuantizationConfig
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
        plan["decision"] = "quantize-4bit+cpu-offload"
        plan["planned_bytes"] = int(weight_bytes / _QUANT_SHRINK)
        _log_plan(plan)
        return quant_config, plan
    except Exception as exc:  # noqa: BLE001 — no bnb/quant API: priced fp16 plan
        plan["decision"] = (
            "fp16 (4-bit wanted but quant stack unavailable: "
            f"{type(exc).__name__})")
        _log_plan(plan)
        return None, plan


def _release_cuda() -> None:
    """Return freed-but-cached CUDA blocks to the OS and collect host garbage.
    torch's caching allocator keeps freed blocks RESERVED (nvidia-smi attributes
    them to the process) until empty_cache() — so a caught OOM would otherwise
    leave the whole card reserved as a zombie (item I /
    worker-vram-leak-unattributed). This is the deterministic unwind."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — no torch/cuda: nothing to release
        pass


def _trim_host_ram() -> None:
    """Post-load: hand torch's CUDA cache and glibc's host arena back to the OS so
    RSS/VRAM don't stay pinned at the load high-water mark (mirrors the img2img
    idiom that used to live inline)."""
    _release_cuda()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 — non-glibc/musl: no malloc_trim
        pass


def _settle_for_vram_retry(cache: Dict[str, Any], lock: threading.Lock,
                           keep: str) -> None:
    """Between a retryable VRAM-class first failure and the ONE retry (k71, the
    in-process twin of the comfy seam's cancel+headroom step): re-drive the
    eviction machinery that already exists — evict idle sibling pipelines,
    return freed-but-reserved CUDA blocks to the OS — then give the allocator a
    bounded few seconds to settle. Only the settle sleep is new mechanism.
    Every step is best-effort; the retry proceeds regardless."""
    try:
        with lock:
            _evict_idle_pipelines(cache, keep)
    except Exception:  # noqa: BLE001 — best-effort eviction re-drive
        pass
    _release_cuda()
    delay = settle_delay_s()
    if delay > 0:
        time.sleep(delay)


def _load_diffusers_pipeline(auto_cls, model_dir: str, model_key: str,
                             *, place_fn=None):
    """The ONE priced diffusers loader, shared by both image runners (stages 2 &
    4 of the ruling). Prices + elects quant, loads (with a DiffusionPipeline
    fallback for natively-conditioned edit/flux2 classes ``AutoPipeline`` can't
    map), then places — CPU-offload when quantized/oversized, else the seam-aware
    ``place_fn`` (t2i) or a plain ``.to`` (img2img default). Returns
    ``(pipe, placement_label)``.

    CRITICAL (item I): on ANY failure it removes offload hooks, drops the partial
    pipeline reference, and releases the CUDA cache BEFORE re-raising — so a
    load-time OOM returns the process to baseline VRAM instead of zombifying the
    card until an /ops/restart."""
    import torch
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    quant_config, _plan = _elect_quantization(model_dir, model_key, cuda)
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    pipe = None
    try:
        # AutoPipeline maps the classic families (SD/SDXL/flux1); natively
        # image-conditioned or newer pipeline classes (Flux2KleinPipeline,
        # QwenImageEditPlusPipeline, …) are absent from its mapping and it raises
        # "can't find a pipeline linked to <cls>" — fall back to DiffusionPipeline,
        # which instantiates the concrete class straight from model_index.json.
        fallback = False
        try:
            pipe = auto_cls.from_pretrained(model_dir, **load_kwargs)
        except ValueError as exc:
            from diffusers import DiffusionPipeline
            logger.info(
                "imagegen: %s has no pipeline mapping for model=%s (%s); "
                "falling back to the concrete DiffusionPipeline class",
                getattr(auto_cls, "__name__", auto_cls), model_key, exc,
            )
            pipe = DiffusionPipeline.from_pretrained(model_dir, **load_kwargs)
            fallback = True

        if cuda and (quant_config is not None or fallback):
            # Quantized / oversized / natively-conditioned edit pipelines:
            # component CPU-offload spills inactive components to host RAM
            # (bnb-quantized components are already device-placed) instead of
            # OOMing at .to("cuda").
            try:
                pipe.enable_model_cpu_offload()
                placement = "model-cpu-offload" + ("+4bit" if quant_config else "")
            except Exception:  # noqa: BLE001 — offload gap: honest fallback
                try:
                    pipe = pipe.to("cuda")
                    placement = "cuda (offload unavailable)"
                except Exception:  # noqa: BLE001 — quantized parts already placed
                    placement = "device-placed (quantized)"
        elif place_fn is not None:
            placement = place_fn(pipe, cuda, model_key)   # seam-aware default path
        else:
            pipe = pipe.to("cuda" if cuda else "cpu")
            placement = "cuda" if cuda else "cpu"
    except BaseException:
        # Deterministic unwind: strip hooks, drop the partial pipe, empty the
        # allocator cache. Null the local BEFORE _release_cuda so the tensors are
        # actually collectable (a live reference would keep the blocks reserved).
        try:
            if pipe is not None and hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        pipe = None
        _release_cuda()
        raise

    _trim_host_ram()
    return pipe, placement


class ImageGenRunner:
    """Runner for diffusers text-to-image pipelines.

    Per-process singleton cache (_PIPELINES) means many runner instances
    for the same model_key share one loaded pipeline. The Runner wrapper
    itself is cheap; the pipeline isn't.
    """

    request_type = ImageGenRequest
    result_type = ImageGenResult

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- pipeline loading (lazy, singleton) ---------------------------------

    @property
    def pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached

            try:
                import torch  # noqa: F401 — availability probe
                from diffusers import AutoPipelineForText2Image
            except ImportError as exc:
                raise RuntimeError(
                    "diffusers + torch are required for text-to-image tasks "
                    "but are not installed. `pip install diffusers torch`."
                ) from exc

            # Free any idle prior pipeline BEFORE loading this one (bounds VRAM).
            _evict_idle_pipelines(self._PIPELINES, self.model_key)
            model_dir = ensure_model(self.model_key)
            # Priced, quant-electing loader shared with Img2ImgRunner: honest
            # footprint pricing (never a blind fp16 election), 4-bit-on-load when
            # the fp16 footprint won't fit, and a deterministic VRAM unwind if the
            # load fails. The seam-aware _place_diffusers_pipeline governs the
            # non-quantized default so byte-identical placement is preserved when
            # the model fits.
            pipe, placement = _load_diffusers_pipeline(
                AutoPipelineForText2Image, model_dir, self.model_key,
                place_fn=_place_diffusers_pipeline,
            )
            logger.info(
                "ImageGenRunner: loaded model=%s dir=%s placement=%s",
                self.model_key, model_dir, placement,
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

    # --- generation ---------------------------------------------------------

    def _generate(self, req: ImageGenRequest) -> list[GeneratedImage]:
        """Blocking generate. Called from a worker thread by .run().

        Only explicitly-set request fields reach the pipeline call, so the
        pipeline's per-model defaults govern everything the caller left out.
        """
        import torch

        call_kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "num_images_per_prompt": req.num_images,
        }
        for field in ("negative_prompt", "width", "height",
                      "num_inference_steps", "guidance_scale"):
            value = getattr(req, field)
            if value is not None:
                call_kwargs[field] = value
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)

        with _generate_lock(self.model_key):
            try:
                output = self.pipeline(**call_kwargs)
            except BaseException:
                # A generation OOM (or a load OOM reached via the .pipeline
                # property) leaves reserved allocator blocks the process still
                # owns — return them to the OS so failure doesn't zombie the card
                # until an /ops/restart (item I).
                _release_cuda()
                raise

        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)

        images: list[GeneratedImage] = []
        for index, image in enumerate(output.images):
            path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
            image.save(path, format="PNG")
            b64 = None
            if req.return_b64:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(GeneratedImage(
                path=path, b64=b64,
                width=image.width, height=image.height,
                seed=req.seed,
            ))
        return images

    # --- public API ---------------------------------------------------------

    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        t0 = time.monotonic()
        try:
            try:
                images = await asyncio.to_thread(self._generate, req)
            except Exception as first_exc:
                # ONE retry, only for the VRAM/eviction/OOM/allocation class
                # (k71): stale pre-eviction pool state fails the first attempt
                # and an identical second try succeeds once eviction settles.
                # Every other class surfaces as before; a second failure takes
                # the outer except, i.e. exactly today's error path.
                if not is_retryable_vram_failure(first_exc):
                    raise
                logger.warning(
                    "ImageGenRunner %s: retrying ONCE — first attempt failed "
                    "with a VRAM/allocation-class error (%s: %s); re-driving "
                    "idle-pipeline eviction and letting the pool settle before "
                    "the retry", self.model_key,
                    type(first_exc).__name__, first_exc)
                await asyncio.to_thread(
                    _settle_for_vram_retry, self._PIPELINES, self._LOCK,
                    self.model_key)
                images = await asyncio.to_thread(self._generate, req)
            result = ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                images=images,
                text=(f"generated {len(images)} image(s): "
                      + ", ".join(img.path for img in images)),
            )
        except Exception as exc:
            logger.exception(
                "ImageGenRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            result = ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        _record_battery(req, result, time.monotonic() - t0, axis="t2i")
        return result

    async def stream(self, req: ImageGenRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring VisionRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or "image generation failed")


class Img2ImgRunner:
    """Runner for diffusers image-to-image (img2img) pipelines.

    SIBLING of ImageGenRunner — same lazy/singleton/thread-offload pattern, but
    it drives ``AutoPipelineForImage2Image`` and conditions generation on an init
    image (req.image_path) with an optional denoising ``strength``. It REUSES
    ImageGenRequest/ImageGenResult (the remote factory wrappers copy request/
    result types straight off FRAMEWORK_RUNNERS, so reusing them means the worker
    offload path needs zero changes).

    Its pipeline cache is its OWN (_PIPELINES) — the img2img pipeline object is a
    different class from text2img's, so it must not share the text2img cache.

    INERT until a model advertises ("transformers","image-to-image"): the sd-turbo
    advertisement flip is HELD (see models_config.py) so live central never routes
    img2img to the old-wheel GPU worker.
    """

    request_type = ImageGenRequest
    result_type = ImageGenResult

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        self.model_key = cfg.model_key
        self._runtime_kwargs = runtime_kwargs

    # --- pipeline loading (lazy, singleton) ---------------------------------

    @property
    def pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached

            try:
                import torch  # noqa: F401 — availability probe
                from diffusers import AutoPipelineForImage2Image
            except ImportError as exc:
                raise RuntimeError(
                    "diffusers + torch are required for image-to-image tasks "
                    "but are not installed. `pip install diffusers torch`."
                ) from exc

            # Free any idle prior pipeline BEFORE loading this one (bounds VRAM).
            _evict_idle_pipelines(self._PIPELINES, self.model_key)
            model_dir = ensure_model(self.model_key)
            # Shared priced loader: honest footprint pricing, 4-bit-on-load when
            # the fp16 footprint won't fit (Qwen-Image-Edit ~55GB on a 24GB card),
            # component CPU-offload for the quantized/oversized case, and a
            # deterministic VRAM unwind if the load fails. No place_fn — img2img's
            # historical default is a plain .to(device), not the t2i alloc seam.
            pipe, placement = _load_diffusers_pipeline(
                AutoPipelineForImage2Image, model_dir, self.model_key,
            )
            logger.info(
                "Img2ImgRunner: loaded model=%s dir=%s class=%s placement=%s",
                self.model_key, model_dir, type(pipe).__name__, placement,
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

    # --- input helpers -------------------------------------------------------

    def _load_init_image(self, req: ImageGenRequest):
        """Load the init image (mirrors VisionAnalysisRunner._load_image). A
        clean error (raised here, caught by run() into an ok=False result) when
        no init image was provided — img2img has nothing to condition on."""
        from PIL import Image
        if not req.image_path:
            raise ValueError(
                "image-to-image requires an init image (image_path); none provided"
            )
        return Image.open(req.image_path).convert("RGB")

    # --- generation ---------------------------------------------------------

    def _generate(self, req: ImageGenRequest) -> list[GeneratedImage]:
        """Blocking img2img generate. Called from a worker thread by .run().

        Mirrors ImageGenRunner._generate but conditions on an init image. Only
        explicitly-set request fields reach the pipeline call.
        """
        import torch

        init_img = self._load_init_image(req)
        # The SD img2img pipeline derives the output size from the init image
        # (its __call__ takes no width/height), so honor the requested dims by
        # RESIZING the init here. This also keeps every chained scene frame the
        # same size, which the mp4 mux requires.
        if req.width is not None and req.height is not None:
            init_img = init_img.resize((req.width, req.height))

        call_kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "num_images_per_prompt": req.num_images,
            "image": init_img,
        }
        # width/height are handled via the resize above (the pipeline ignores
        # them), so they are intentionally NOT forwarded here.
        for field in ("negative_prompt", "num_inference_steps", "guidance_scale"):
            value = getattr(req, field)
            if value is not None:
                call_kwargs[field] = value
        if req.strength is not None:
            call_kwargs["strength"] = req.strength

        # sd-turbo numeric edge: diffusers computes effective steps as
        # int(num_inference_steps * strength) and RAISES when that is 0. sd-turbo
        # runs 1-4 steps, so a low strength (e.g. steps=2 * strength=0.3 -> 0
        # effective) detonates. Bump steps so int(steps*strength) >= 1 and log
        # LOUDLY, rather than letting the pipeline raise.
        steps = call_kwargs.get("num_inference_steps")
        strength = call_kwargs.get("strength")
        if (steps is not None and strength is not None
                and strength > 0 and int(steps * strength) < 1):
            import math
            bumped = int(math.ceil(1.0 / strength))
            logger.warning(
                "Img2ImgRunner: num_inference_steps=%s * strength=%s -> %d "
                "effective steps (0 raises in diffusers); bumping steps %s -> %d "
                "for model=%s", steps, strength, int(steps * strength),
                steps, bumped, self.model_key,
            )
            call_kwargs["num_inference_steps"] = bumped

        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)

        # Concrete edit pipelines (QwenImageEditPlus, Flux2Klein, …) don't share
        # the SD img2img signature — e.g. no `strength`, `true_cfg_scale` in
        # place of guidance. Filter kwargs to what THIS pipeline's __call__
        # actually accepts (a **kwargs pipeline keeps everything) and log the
        # drops, instead of detonating on an unexpected-keyword TypeError.
        import inspect
        # Resolve AND call the pipeline under the generate lock (mirrors
        # ImageGenRunner): the eviction pass honours a held generate lock, so a
        # concurrent load of a different model can't tear this pipeline down —
        # or strip its cpu-offload hooks — between resolve and call.
        with _generate_lock(self.model_key):
            pipe = self.pipeline
            try:
                sig = inspect.signature(pipe.__call__)
                has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                                 for p in sig.parameters.values())
                if not has_var_kw:
                    accepted = set(sig.parameters)
                    dropped = [k for k in call_kwargs if k not in accepted]
                    if dropped:
                        logger.warning(
                            "Img2ImgRunner: %s.__call__ does not accept %s — "
                            "dropping for model=%s",
                            type(pipe).__name__, dropped, self.model_key,
                        )
                        call_kwargs = {k: v for k, v in call_kwargs.items()
                                       if k in accepted}
            except (TypeError, ValueError):
                pass  # unsignaturable callable — send as-is
            try:
                output = pipe(**call_kwargs)
            except BaseException:
                # Unwind the transient allocation a failed/OOM'd generation left
                # reserved so the card returns to baseline (item I).
                _release_cuda()
                raise

        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)

        images: list[GeneratedImage] = []
        for index, image in enumerate(output.images):
            path = os.path.join(out_dir, f"{req.request_id}_{index}.png")
            image.save(path, format="PNG")
            b64 = None
            if req.return_b64:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(GeneratedImage(
                path=path, b64=b64,
                width=image.width, height=image.height,
                seed=req.seed,
            ))
        return images

    # --- public API ---------------------------------------------------------

    async def run(self, req: ImageGenRequest) -> ImageGenResult:
        t0 = time.monotonic()
        try:
            try:
                images = await asyncio.to_thread(self._generate, req)
            except Exception as first_exc:
                # ONE retry for the VRAM/allocation class only — see
                # ImageGenRunner.run (k71). Same seam, img2img cache.
                if not is_retryable_vram_failure(first_exc):
                    raise
                logger.warning(
                    "Img2ImgRunner %s: retrying ONCE — first attempt failed "
                    "with a VRAM/allocation-class error (%s: %s); re-driving "
                    "idle-pipeline eviction and letting the pool settle before "
                    "the retry", self.model_key,
                    type(first_exc).__name__, first_exc)
                await asyncio.to_thread(
                    _settle_for_vram_retry, self._PIPELINES, self._LOCK,
                    self.model_key)
                images = await asyncio.to_thread(self._generate, req)
            result = ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                images=images,
                text=(f"generated {len(images)} image(s): "
                      + ", ".join(img.path for img in images)),
            )
        except Exception as exc:
            logger.exception(
                "Img2ImgRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            result = ImageGenResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        _record_battery(req, result, time.monotonic() - t0, axis="i2i")
        return result

    async def stream(self, req: ImageGenRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring ImageGenRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or "image-to-image generation failed")
