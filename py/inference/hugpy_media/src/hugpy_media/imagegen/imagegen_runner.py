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
# Placement — diffusers does NOT take device_map/max_memory the transformers way.
# The cpu-vs-gpu decision is made in _load_diffusers_pipeline (the `cuda` flag,
# which folds ram-only intent + the worker's evict-to-fit reclaim); this function
# only applies it:
#   * cuda=False  -> FULLY on the CPU (pipe.to("cpu")): the GPU is NOT touched.
#     diffusers' cpu-offload APIs are deliberately NOT used — enable_sequential_
#     cpu_offload still STREAMS submodules THROUGH CUDA and OOMs on a full card
#     (incident 2026-09-25: sd-turbo on computron, 17.88 MiB free). ram-only that
#     still can't fit after eviction arrives here as cuda=False.
#   * cuda=True + max-ram alloc_mode -> model CPU offload: whole submodules stream
#     to RAM and are pulled onto the GPU only while active — big model, one card.
#   * cuda=True otherwise (gpu-only / auto / ram-only UPGRADED after reclaim) ->
#     `.to(cuda)`, the whole pipeline on the GPU.
# A pipeline class without the max-ram offload method (genuine capability gap) is
# logged ONCE and falls back to .to(cuda) rather than silently ignoring the mode.
def _place_diffusers_pipeline(pipe, cuda: bool, model_key: str,
                              device: "str | None" = None) -> str:
    """Place a diffusers pipeline per the allocation seam. Returns a short label
    of what was applied (for the load log). Mutates ``pipe`` in place (both
    .to(...) and enable_*_cpu_offload() act on the object).

    ``device`` is the CUDA device string the GPU branch places on — central's
    per-GPU pin resolved to ``"cuda"`` (primary card) or ``"cuda:N"``. Defaults to
    :func:`_cuda_device`, so the pin is honored without threading it through every
    caller, and a single-GPU box stays byte-identical (``"cuda"``).

    THE cpu-vs-gpu decision is the caller's (the ``cuda`` flag), NOT re-derived
    here: ``_load_diffusers_pipeline`` folds ram-only intent + the worker's
    evict-to-fit reclaim into ``cuda`` (False => fully CPU; True => the GPU is
    wanted, incl. a ram-only load UPGRADED after eviction freed the card). So this
    function must NEVER send a cuda=True load to the CPU by re-reading the intent —
    that would undo the upgrade (place a now-fitting model on the CPU while the
    idle squatter it evicted is gone). ram-only that still can't fit arrives here
    as cuda=False and takes the CPU branch below."""
    if not cuda:
        # Fully on the CPU — the GPU is not touched at all. diffusers' cpu-offload
        # APIs are deliberately NOT used (they stream submodules THROUGH CUDA and
        # OOM on a full card — incident 2026-09-25).
        pipe.to("cpu")
        return "cpu"

    # The GPU card central chose (primary "cuda" or "cuda:N"). enable_*_cpu_offload
    # deliberately keeps its default-device behavior (passing a device would break
    # pipelines whose offload API has no such arg); the whole-pipeline .to() path
    # honors the pin.
    device = device or _cuda_device()

    # GPU decided upstream. The only remaining device knob is max-ram, which
    # streams whole components to RAM and pulls them onto the GPU while active
    # (big model, one card). Everything else places the whole pipeline on the GPU.
    try:
        from hugpy_engine.spill import alloc_mode_env
        alloc_mode = alloc_mode_env()           # "max-ram" | "explicit" | None
    except Exception as exc:  # noqa: BLE001 — no seam: today's path, logged
        logger.warning("imagegen: placement seam unavailable (%s); using "
                       ".to(cuda)", exc)
        pipe.to(device)
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
            pipe.to(device)
            return f"cuda ({label} unsupported by this pipeline)"
        try:
            fn()                                # offload methods mutate in place
            return label
        except Exception as exc:  # noqa: BLE001 — offload failed: honest fallback
            logger.warning(
                "imagegen: model=%s %s() failed (%s) — falling back to .to(cuda)",
                model_key, method, exc,
            )
            pipe.to(device)
            return f"cuda ({label} failed)"

    if alloc_mode == "max-ram":
        return _offload("enable_model_cpu_offload", "max-ram/model-offload")

    # gpu-only / auto / no intent / ram-only-upgraded -> whole pipeline on the GPU.
    pipe.to(device)
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
            return int(torch.cuda.mem_get_info(_main_gpu_index())[0])
    except Exception:  # noqa: BLE001 — no cuda / can't tell
        pass
    return None


# ── evict-to-fit hook (2026-09-25) ──────────────────────────────────────────
# The in-process diffusers pipeline loads LAZILY inside the runner (AFTER the
# runner object is built + cached), so dispatch.ensure_headroom_for_load — which
# runs at runner-BUILD time — never covers the real VRAM allocation; and a
# ram-only designation makes the cross-tier make-room a no-op. So an idle slot
# squatter is never evicted and the diffusers load/generate CUDA-OOMs behind it
# (incident 2026-09-25: sd-turbo on computron behind flux2-klein's idle 6.99 GiB
# slot child). This hook lets the WORKER agent register its evict-to-fit routine
# (the SAME evict verb/policy comfy + every other path uses); the runner calls it
# with its priced footprint right before loading/generating. Package-shared
# (central imports this module), so — exactly like set_comfy_headroom_hook — it
# must NOT import worker/GPU internals; the worker registers the hook at boot and
# this module calls it if present, else a no-op (bare central / no-GPU is
# byte-identical). Best-effort: a failing hook NEVER blocks a generation.
_IMAGEGEN_HEADROOM_HOOK = None

# Per-model state the load path records for the generation path + honest errors.
_MODEL_NEED: Dict[str, int] = {}          # model_key -> priced fp16 footprint bytes
_PIPELINE_DEVICE: Dict[str, str] = {}     # model_key -> "cuda" | "cpu" (effective)
_LAST_HEADROOM: Dict[str, dict] = {}      # model_key -> last evict-to-fit telemetry


def set_imagegen_headroom_hook(fn) -> None:
    """Register the worker-side evict-to-fit routine for the in-process diffusers
    image runners: ``fn(model_key, need_bytes, job_id) -> dict|None`` — evict the
    minimum LRU set of eligible residents (same verb/policy every other path uses,
    honoring the busy/in-flight/static gates) until the GPU has room for
    ``need_bytes``. None (bare central / no-GPU) => the pre-load headroom step is a
    no-op, byte-identical to before."""
    global _IMAGEGEN_HEADROOM_HOOK
    _IMAGEGEN_HEADROOM_HOOK = fn


def _ensure_imagegen_headroom(model_key: str, need_bytes: "int | None",
                              job_id=None) -> "dict | None":
    """Call the registered evict-to-fit hook (if present) BEFORE an in-process
    diffusers load/generate, so the pipeline can land on the GPU instead of OOMing
    behind an idle squatter. Best-effort — never raises into the generation path;
    returns the worker's telemetry dict (evicted/skipped residents, free_after) or
    None when there is no hook / it failed."""
    hook = _IMAGEGEN_HEADROOM_HOOK
    if hook is None:
        return None
    try:
        return hook(model_key, need_bytes, job_id)
    except Exception:  # noqa: BLE001 — headroom prep never breaks a generation
        logger.warning("ensure-imagegen-headroom hook failed (proceeding with the "
                       "load/gen anyway)", exc_info=True)
        return None


# fp16 fit headroom: the whole model is priced as fitting the GPU when free VRAM
# >= weight_bytes / 0.85 (the same 0.85 activation/arena headroom _should_quantize
# uses), so the two agree by construction.
_FP16_FIT_HEADROOM = 0.85


def _cpu_forced(need: "int | None" = None) -> bool:
    """Should this diffusers load run ENTIRELY on the CPU?

    True when central's derived serve mode is ram-only / CPU-only
    (HUGPY_N_GPU_LAYERS off/0/cpu -> spill.n_gpu_layers_intent() == "cpu", set
    per-request by the worker agent's _apply_spill) AND the worker has NOT been
    able to reclaim enough VRAM to hold the model on the GPU.

    Honoring ram-only via diffusers' cpu-offload APIs is WRONG:
    enable_sequential_cpu_offload / enable_model_cpu_offload both still STREAM
    submodules THROUGH CUDA (and a 4-bit bnb load device-places its quantized
    components on CUDA), so both OOM on a nearly-full card — incident 2026-09-25
    01:12:58 (sd-turbo on computron, 17.88 MiB free behind flux2-klein's idle
    6.99 GiB slot child): ram-only priced fp16 against the tiny free VRAM, elected
    a 4-bit bnb load with enable_model_cpu_offload, and OOM'd inside bitsandbytes
    dequant on CUDA. So a forced-CPU load means the WHOLE pipeline, its dtype, and
    the seed generator stay on the CPU.

    But ram-only is central's decision from a CONTENDED snapshot: it derived
    ram-only precisely because an idle resident squatted the card. Once the worker
    has run evict-to-fit (``_ensure_imagegen_headroom``) and reclaimed room, the
    model may now fit the GPU — so ram-only is UPGRADED to a GPU load when the
    reclaimed free VRAM clearly holds the whole fp16 footprint. Only when it still
    does not fit does ram-only stay binding (CPU). ``need`` is the priced fp16
    footprint; call this AFTER the evict-to-fit pass so free VRAM is current."""
    try:
        from hugpy_engine.spill import n_gpu_layers_intent
        if n_gpu_layers_intent() != "cpu":
            return False
    except Exception:  # noqa: BLE001 — no seam: not forced (historical path)
        return False
    # ram-only intent. Honor it UNLESS the reclaimed card now clearly holds the
    # whole model on the GPU (worker owns its own contended VRAM).
    if need:
        free = _free_vram_bytes()
        if free is not None and free >= int(need / _FP16_FIT_HEADROOM):
            logger.info(
                "imagegen: ram-only intent UPGRADED to GPU — reclaimed %.2f GiB "
                "free >= %.2f GiB fp16 need after evict-to-fit",
                free / 2 ** 30, (need / _FP16_FIT_HEADROOM) / 2 ** 30)
            return False
    return True


def _main_gpu_index() -> int:
    """The GPU index central pinned this request to (HUGPY_MAIN_GPU, via the spill
    seam), or 0. An in-process diffusers load runs in the worker's own process,
    which enumerates every card, so it can't restrict CUDA_VISIBLE_DEVICES
    per-request the way a slot child does — it must address the chosen card by its
    GLOBAL index (``cuda:N``). 0 when unpinned (single-GPU / device 0)."""
    try:
        from hugpy_engine.spill import main_gpu
        n = main_gpu()
        return int(n) if n is not None else 0
    except Exception:  # noqa: BLE001 — no seam / unparseable: the primary card
        return 0


def _cuda_device() -> str:
    """The CUDA device string this pipeline occupies: ``"cuda"`` for the primary
    card (index 0 — byte-identical to the historical bare ``.to("cuda")``) or
    ``"cuda:N"`` for a central-chosen card on a multi-GPU box."""
    n = _main_gpu_index()
    return "cuda" if not n else f"cuda:{n}"


def _generator_device(model_key: "str | None" = None) -> str:
    """Device the seed generator must live on. It MUST match where the pipeline
    actually runs: the effective device recorded at load time (``_PIPELINE_DEVICE``)
    when known, else CUDA only when the GPU is genuinely in use. A cuda generator
    for a CPU-placed pipeline touches the very card ram-only exists to avoid and
    mismatches the CPU pipeline's latents."""
    if model_key is not None:
        dev = _PIPELINE_DEVICE.get(model_key)
        if dev:
            return dev
    try:
        import torch
        if torch.cuda.is_available() and not _cpu_forced():
            return _cuda_device()
    except Exception:  # noqa: BLE001 — no torch/cuda
        pass
    return "cpu"


def _vram_snapshot() -> str:
    """Human free/total VRAM for an honest OOM message, or "" when unmeasurable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(_main_gpu_index())
            return f"{free / 2 ** 30:.2f} GiB free of {total / 2 ** 30:.2f} GiB total"
    except Exception:  # noqa: BLE001 — no cuda / can't tell
        pass
    return ""


class _HonestVRAMError(RuntimeError):
    """A CUDA OOM during an image load/generation, rewritten to name the model,
    the phase, the derived placement intent and free VRAM alongside torch's own
    detail (which already names the other resident process). Still classifies as
    the retryable VRAM class (its text carries "out of memory") so the one-shot
    settle+retry seam is unchanged. Marked as a distinct type so it is never
    double-wrapped as it propagates."""


def _honest_vram_error(exc: BaseException, model_key: str,
                       phase: str) -> "_HonestVRAMError | None":
    """If ``exc`` is a CUDA OOM, return an _HonestVRAMError that surfaces the real
    reason — model, phase (load/generation), derived placement intent, and free
    VRAM — instead of an opaque torch traceback. Returns None when ``exc`` is not
    an OOM (caller re-raises the original) or is already an _HonestVRAMError."""
    if isinstance(exc, _HonestVRAMError):
        return None
    if not is_retryable_vram_failure(exc):
        return None
    try:
        from hugpy_engine.spill import n_gpu_layers_intent
        intent = n_gpu_layers_intent()
    except Exception:  # noqa: BLE001 — no seam
        intent = "unknown"
    snap = _vram_snapshot()
    ctx = f"placement intent={intent}"
    if snap:
        ctx += f"; VRAM {snap}"
    # Name what the worker's evict-to-fit pass could (not) do — so the failure
    # says WHY the card had no room (which residents wouldn't yield and why),
    # instead of only the raw OOM. Best-effort; absent when no pass ran.
    hr = _LAST_HEADROOM.get(model_key) or {}
    evicted = hr.get("evicted")
    if evicted:
        ctx += f"; evicted {evicted} but still short"
    skipped = hr.get("skipped") or []
    if skipped:
        names = ", ".join(
            f"{s.get('model_key')} ({s.get('reason')})" for s in skipped)
        ctx += f"; could NOT evict: {names}"
    return _HonestVRAMError(
        f"CUDA out of memory during {phase} of image model {model_key!r} "
        f"({ctx}). Underlying error: {type(exc).__name__}: {exc}"
    )


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
    eviction machinery that already exists — evict idle sibling pipelines, run the
    worker's cross-tier evict-to-fit (slot children / other-model residents the
    in-process cache can't see), return freed-but-reserved CUDA blocks to the OS —
    then give the allocator a bounded few seconds to settle. Only the settle sleep
    is new mechanism. Every step is best-effort; the retry proceeds regardless."""
    try:
        with lock:
            _evict_idle_pipelines(cache, keep)
    except Exception:  # noqa: BLE001 — best-effort eviction re-drive
        pass
    # Cross-tier: the sibling-pipeline eviction above only sees THIS process's
    # in-process image pipelines; re-drive the worker's evict-to-fit so the retry
    # also reclaims a slot child / other resident squatting the card (the same
    # pass the pre-load path runs). Best-effort; records to /llm/evictions.
    _LAST_HEADROOM[keep] = _ensure_imagegen_headroom(keep, _MODEL_NEED.get(keep)) or {}
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
    # THE priced footprint: honest on-disk fp16 weight bytes. Used both to price
    # quantization and as the evict-to-fit target below.
    need = _weight_bytes(model_dir)
    _MODEL_NEED[model_key] = need
    cuda_available = torch.cuda.is_available()
    # EVICT-TO-FIT BEFORE pricing (2026-09-25): reclaim the GPU by evicting the
    # minimum LRU set of idle eligible residents (same evict verb/policy every
    # other load path uses; busy/in-flight/static are never touched), so the
    # fit/quant/placement decision below is made against a card this load actually
    # gets — not one an idle squatter is holding. Each eviction is logged and shows
    # in /llm/evictions. No-op on bare central / no-GPU / no registered hook.
    # (Incident 2026-09-25: the idle flux2-klein slot child held 6.99 GiB and was
    # never evicted, so the diffusers load CUDA-OOM'd behind it.)
    if cuda_available:
        _LAST_HEADROOM[model_key] = _ensure_imagegen_headroom(model_key, need) or {}
    # `cuda` is the EFFECTIVE decision to use the GPU for this load, NOT merely
    # "is a card present": when central's derived serve mode is ram-only AND the
    # reclaimed card still can't hold the model (_cpu_forced), the whole load runs
    # on the CPU. This gates dtype (fp16 only on the GPU — fp16 on CPU is
    # unsupported/slow), quant election (no 4-bit bnb load off the card), and
    # placement, so the ram-only intent can no longer be dropped by the quant
    # branch bypassing the placement seam. _cpu_forced(need) reads free VRAM AFTER
    # the evict-to-fit pass, so a ram-only load that now fits the reclaimed card is
    # upgraded to the GPU instead of running slow on the CPU.
    cuda = cuda_available and not _cpu_forced(need=need)
    # The specific card central pinned this load to ("cuda"/"cuda:N"); recorded so
    # the generation path's seed generator lands on the SAME device as the weights.
    _dev = _cuda_device() if cuda else "cpu"
    _PIPELINE_DEVICE[model_key] = _dev
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
                    pipe = pipe.to(_dev)
                    placement = "cuda (offload unavailable)"
                except Exception:  # noqa: BLE001 — quantized parts already placed
                    placement = "device-placed (quantized)"
        elif place_fn is not None:
            placement = place_fn(pipe, cuda, model_key)   # seam-aware default path
        else:
            pipe = pipe.to(_dev)
            placement = "cuda" if cuda else "cpu"
    except BaseException as exc:
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
        # A load-time CUDA OOM surfaces as a clear, recorded failure naming the
        # free VRAM + the other resident process, not an opaque traceback.
        honest = _honest_vram_error(exc, model_key, "load")
        if honest is not None:
            raise honest from exc
        raise

    _trim_host_ram()
    return pipe, placement


from hugpy_platform.runner_config import RunnerConfig


class ImageGenRunner(RunnerConfig):
    """Runner for diffusers text-to-image pipelines.

    Per-process singleton cache (_PIPELINES) means many runner instances
    for the same model_key share one loaded pipeline. The Runner wrapper
    itself is cheap; the pipeline isn't.
    """

    request_type = ImageGenRequest
    result_type = ImageGenResult

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()


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
            # Match the pipeline's actual device: "cpu" whenever ram-only forces
            # the whole run off the card (a cuda generator would touch the very
            # GPU ram-only exists to avoid, and mismatch the CPU latents).
            device = _generator_device(self.model_key)
            call_kwargs["generator"] = torch.Generator(device).manual_seed(req.seed)

        with _generate_lock(self.model_key):
            try:
                output = self.pipeline(**call_kwargs)
            except BaseException as exc:
                # A generation OOM (or a load OOM reached via the .pipeline
                # property) leaves reserved allocator blocks the process still
                # owns — return them to the OS so failure doesn't zombie the card
                # until an /ops/restart (item I).
                _release_cuda()
                # Surface a CUDA OOM as a clear, recorded reason (free VRAM +
                # the other resident process) rather than an opaque traceback.
                honest = _honest_vram_error(exc, self.model_key, "generation")
                if honest is not None:
                    raise honest from exc
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


class Img2ImgRunner(RunnerConfig):
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
            # Match the pipeline's actual device: "cpu" whenever ram-only forces
            # the whole run off the card (see ImageGenRunner._generate).
            device = _generator_device(self.model_key)
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
            except BaseException as exc:
                # Unwind the transient allocation a failed/OOM'd generation left
                # reserved so the card returns to baseline (item I).
                _release_cuda()
                # Surface a CUDA OOM as a clear, recorded reason (free VRAM +
                # the other resident process) rather than an opaque traceback.
                honest = _honest_vram_error(exc, self.model_key, "generation")
                if honest is not None:
                    raise honest from exc
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
