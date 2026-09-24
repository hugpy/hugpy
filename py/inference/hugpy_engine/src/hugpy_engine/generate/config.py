# deepcoder/config.py
import os
import os.path as osp
import threading
from dataclasses import dataclass
from typing import Any, Optional

from abstract_essentials import get_logFile
from hugpy_engine.config.main import get_model_config, resolve_model_source
from hugpy_engine.peft_adapters import AdapterBaseUnavailable, resolve_adapter_pair, standalone_load_refusal
from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_engine.schemas.runner_schemas import StreamEvent
from hugpy_platform.constants import DEFAULT_LOCAL_FILES_ONLY
from hugpy_platform.module_imports import require
# Serve path: weights are local-or-central on a worker, never Hugging Face
# (hugpy_storage.provision.ensure_serving_weights; computron 2026-09-23).
from hugpy_storage.provision import ensure_serving_weights
logger = get_logFile("deepcoder")
_SENTINEL = object()


@dataclass(frozen=True)
class DeepCoderConfig:
    model_dir: str
    device: str
    # The REGISTRY key this config was built for. Carried so the worker's
    # in-process VRAM attribution can name the model holding the weights:
    # _inprocess_gpu_bytes walks coder.REGISTRY._instances and reads
    # `dc.cfg.model_key`, and without it every transformers causal-LM loaded
    # in-process reported model_key=None — VRAM that shows as an anonymous
    # `cuda_context` lump on the agent's own pid, which is UNATTRIBUTABLE and
    # therefore UNEVICTABLE (every eviction verb keys on model_key). That is the
    # 2026-07-26 report: a 4-bit load took 13.6 GiB, the console said "No models
    # are using this GPU's VRAM", and the next load refused against memory
    # nothing could reclaim. Default None keeps every existing caller valid.
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY
    # SECURITY: loading a model with custom code (auto_map / modeling_*.py)
    # EXECUTES the author's Python on load — i.e. arbitrary RCE from an untrusted
    # HF repo. OFF by default; only ever True via an explicit opt-in (per call, or
    # the operator switch HUGPY_TRUST_REMOTE_CODE). Never defaults on.
    trust_remote_code: bool = False
    model_key: Optional[str] = None

    # PEFT: when model_dir is a bare LoRA ADAPTER dir, model_dir is rewritten to
    # the BASE model and this holds the adapter. DeepCoder._load_model has read
    # `cfg.adapter_dir` since the adapter branch was written, but this field did
    # not exist and nothing ever set it — so the branch was dead and the adapter
    # dir went to transformers as if it were a model ("Unrecognized model in
    # <dir>. Should have a `model_type` key in its config.json"). Resolved in
    # build_deepcoder_runtime; see imports/src/peft_adapters.py.
    adapter_dir: Optional[str] = None

    max_new_tokens_cap: int = 16000

    cpu_threads: Optional[int] = None
    cpu_interop_threads: Optional[int] = 1
    max_concurrent_generations: int = 1

    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.trust_remote_code,
            self.adapter_dir,            # distinct cache slot per adapter
            self.max_new_tokens_cap,
            self.cpu_threads,
            self.cpu_interop_threads,
            self.max_concurrent_generations,
        )


def pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, Any]:
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is not None:
        return chosen, dtype

    if chosen == "cuda":
        return chosen, (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen, torch.bfloat16

    return chosen, torch.float32


def build_deepcoder_runtime(
    *,
    model_key: str = "deepcoder",
    device: Optional[str] = None,
    torch_dtype=None,
    use_quantization: bool = False,
    use_flash_attention: bool = False,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    max_new_tokens_cap: int = 16000,
    max_concurrent_generations: int = 1,
    cpu_threads: Optional[int] = None,
    cpu_interop_threads: Optional[int] = 1,
    auto_download: bool = True,
) -> DeepCoderConfig:
    torch = require("torch", reason="DeepCoder requires PyTorch")
    get_model_config(model_key)

    if auto_download:
        model_dir = str(ensure_serving_weights(model_key))
    else:
        model_dir = resolve_model_source(model_key)

        if not osp.exists(model_dir):
            raise FileNotFoundError(
                f"Model {model_key!r} is not on disk and auto_download=False; "
                f"call ensure_serving_weights({model_key!r}) first."
            )

    # PEFT / bare-adapter dirs. A LoRA dir has adapter_config.json and NO
    # config.json, so it has no `model_type` and transformers refuses it with
    # "Unrecognized model in <dir>" — the 2026-07-29 bench failure for
    # qwen3.5-test-stage1-lora, Qwen2.5-1.5B-LFGRPO-300S and
    # veeraragavan410~Llama-3.2-3B-sentiment. Rewrite model_dir to the BASE model
    # from the store and carry the adapter separately; DeepCoder._load_model
    # applies it with PeftModel.from_pretrained (peft is lazy-imported by
    # require_peft, which refuses with `pip install peft` if it's missing).
    #
    # Local store ONLY — an absent base raises AdapterBaseUnavailable naming the
    # id to acquire. It never becomes a silent multi-GB download.
    #
    # Ordinary model dirs come back unchanged, so this is inert for everything
    # that already worked. A dir that is neither loadable nor an adapter (a
    # bespoke-runtime repo mis-registered as transformers) is named for what it
    # is instead of dying inside from_pretrained.
    adapter_dir = None
    try:
        model_dir, adapter_dir = resolve_adapter_pair(model_dir)
    except AdapterBaseUnavailable as exc:
        raise RuntimeError(f"{model_key}: {exc}") from exc
    if adapter_dir:
        logger.info("%s is a PEFT adapter; base=%s adapter=%s",
                    model_key, model_dir, adapter_dir)
    else:
        refusal = standalone_load_refusal(model_dir)
        if refusal:
            raise RuntimeError(f"{model_key}: {refusal}")

    chosen_device, chosen_dtype = pick_device_and_dtype(torch, device, torch_dtype)

    # SECURITY: trust_remote_code lets a model repo run arbitrary Python on load.
    # Default OFF; True only via an explicit opt-in here OR the operator switch
    # HUGPY_TRUST_REMOTE_CODE (1/true/yes/on). Never on by default.
    allow_remote_code = bool(trust_remote_code) or (
        os.environ.get("HUGPY_TRUST_REMOTE_CODE", "").strip().lower()
        in ("1", "true", "yes", "on")
    )

    return DeepCoderConfig(
        model_dir=model_dir,
        model_key=model_key,        # carried for in-process VRAM attribution
        device=chosen_device,
        torch_dtype=chosen_dtype,
        use_quantization=use_quantization and chosen_device == "cuda",
        use_flash_attention=use_flash_attention and chosen_device == "cuda",
        local_files_only=local_files_only,
        trust_remote_code=allow_remote_code,
        max_new_tokens_cap=max_new_tokens_cap,
        max_concurrent_generations=max_concurrent_generations,
        cpu_threads=cpu_threads,
        cpu_interop_threads=cpu_interop_threads,
        adapter_dir=adapter_dir,
    )
build_deepcoder_config = build_deepcoder_runtime


class CancelStoppingCriteria:
    def __init__(self, cancel: threading.Event):
        self._cancel = cancel

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancel.is_set()
