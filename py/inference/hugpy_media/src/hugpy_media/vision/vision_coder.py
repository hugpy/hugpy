import logging
import gc
import math
from dataclasses import dataclass
from typing import Optional

import io
import base64
from PIL import Image

from hugpy_platform.constants import DEFAULT_LOCAL_FILES_ONLY
from hugpy_platform.module_imports import get_torch, get_transformers, require

from hugpy_media.vision.schemas import (
    BAD_PATH_STRINGS,
    QWEN_PATCH,
    QWEN_PIXELS_PER_TOKEN,
    VisionCoderConfig,
    VisionRequest,
)

logger = logging.getLogger("vision_coder")

def cleanup_cuda() -> None:
    torch = get_torch()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def open_image_from_request(req: "VisionRequest") -> Image.Image:
    if req.image_path is not None:
        return Image.open(req.image_path).convert("RGB")
    if req.image_b64 is not None:
        raw = base64.b64decode(req.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError(
        f"VisionRequest {req.request_id!r} has neither image_path nor image_b64"
    )


def _coerce_image_path(value) -> str:
    import os.path as osp

    if not isinstance(value, str):
        raise TypeError(
            f"image_path must be a string, got {type(value).__name__}: {value!r}"
        )

    cleaned = value.strip()
    if cleaned.lower() in BAD_PATH_STRINGS:
        raise ValueError(f"image_path looks like a serialization artifact: {value!r}")

    if not osp.exists(cleaned):
        raise FileNotFoundError(f"Image not found: {cleaned}")

    return cleaned


def _pick_device_and_dtype(torch, device: Optional[str], dtype) -> tuple[str, object]:
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is not None:
        return chosen_device, dtype

    if chosen_device == "cuda":
        return chosen_device, torch.float16

    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return chosen_device, torch.bfloat16

    return chosen_device, torch.float32


def fit_to_token_budget(image: Image.Image, max_tokens: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size

    pixel_budget = max_tokens * QWEN_PIXELS_PER_TOKEN
    current_pixels = width * height

    if current_pixels <= pixel_budget:
        return image

    scale = math.sqrt(pixel_budget / current_pixels)

    new_width = max(QWEN_PATCH, int(width * scale))
    new_height = max(QWEN_PATCH, int(height * scale))

    # Round down to patch multiples to avoid odd visual grids.
    new_width = max(QWEN_PATCH, new_width // QWEN_PATCH * QWEN_PATCH)
    new_height = max(QWEN_PATCH, new_height // QWEN_PATCH * QWEN_PATCH)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def __getattr__(name: str):
    # VISION_MODELS_REGISTRY / DEFAULT_VISION_MODEL are the engine's model
    # registry views; resolving them walks the model store, so they are
    # looked up on first use rather than at import (video reads them here).
    if name in ("VISION_MODELS_REGISTRY", "DEFAULT_VISION_MODEL"):
        from hugpy_engine.config.models import models_default
        return getattr(models_default, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_vision_model_key(model_key: Optional[str]) -> str:
    from hugpy_engine.config.models.models_default import DEFAULT_VISION_MODEL, VISION_MODELS_REGISTRY

    key = model_key or DEFAULT_VISION_MODEL

    if key not in VISION_MODELS_REGISTRY:
        available = list(VISION_MODELS_REGISTRY.keys())
        raise KeyError(f"Unknown vision model key {key!r}. Available: {available}")

    return key


def build_config(
    model_key: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype=None,
    min_tokens: int = 16,
    max_tokens: int = 128,
    device_map: Optional[str] = "auto",
    gpu_max_memory: str = "5GiB",
    cpu_max_memory: str = "24GiB",
    use_cache: bool = False,
    local_files_only:bool = DEFAULT_LOCAL_FILES_ONLY
) -> VisionCoderConfig:
    import os.path as osp

    torch = require("torch", reason="VisionCoder requires PyTorch")
    chosen_device, chosen_dtype = _pick_device_and_dtype(torch, device, torch_dtype)

    from hugpy_engine.config.main import get_model_path

    key = _resolve_vision_model_key(model_key)
    model_dir = get_model_path(key)

    if not osp.isdir(model_dir):
        raise FileNotFoundError(
            f"Vision model {key!r} does not appear to be downloaded locally.\n"
            f"Expected directory: {model_dir}"
        )

    return VisionCoderConfig(
        model_key=key,
        model_dir=model_dir,
        device=chosen_device,
        torch_dtype=chosen_dtype,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        device_map=device_map,
        gpu_max_memory=gpu_max_memory,
        cpu_max_memory=cpu_max_memory,
        use_cache=use_cache,
    )


class VisionCoder:
    def __init__(self, cfg: VisionCoderConfig):
        require("transformers", reason="VisionCoder requires HuggingFace transformers")

        self.cfg = cfg

        logger.info(
            "VisionCoder loading key=%s model=%s device=%s dtype=%s token_budget=[%d,%d]",
            cfg.model_key,
            cfg.model_dir,
            cfg.device,
            cfg.torch_dtype,
            cfg.min_tokens,
            cfg.max_tokens,
        )

        # AutoModelForImageTextToText dispatches on the checkpoint's own
        # model_type — a hardcoded Qwen2_5_VL class loaded ANY vision
        # checkpoint, which shape-mismatched every non-Qwen VLM (MiniCPM-V-4.6
        # died with ignore_mismatched_sizes on worker ae, 2026-08-06).
        AutoModelForImageTextToText = get_transformers(
            "AutoModelForImageTextToText"
        )
        AutoProcessor = get_transformers("AutoProcessor")

        model_kwargs = {
            "torch_dtype": cfg.torch_dtype,
            "trust_remote_code": True,
            "local_files_only": cfg.local_files_only,
            "low_cpu_mem_usage": True,
        }

        if cfg.device_map == "auto" and cfg.device == "cuda":
            # GPU/CPU spill: honor the SAME placement seam every other
            # transformers loader uses (t26/t27 — spill.transformers_max_memory)
            # so explicit HUGPY_GPU_MEM_GIB/HUGPY_CPU_MEM_GIB budgets and the
            # HUGPY_N_GPU_LAYERS placement intent (Max GPU / CPU only / auto)
            # apply to vision loads too, not just text (t31). Falls back to
            # this loader's own gpu_max_memory/cpu_max_memory (5GiB/24GiB by
            # default) ONLY when the seam has no better answer (e.g. VRAM is
            # unreadable) — no-operator-config behavior is unchanged.
            from hugpy_engine.spill import transformers_max_memory

            max_memory = transformers_max_memory()
            if not max_memory:
                max_memory = {
                    0: cfg.gpu_max_memory,
                    "cpu": cfg.cpu_max_memory,
                }

            model_kwargs.update({
                "device_map": "auto",
                "max_memory": max_memory,
            })

            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_dir,
                **model_kwargs,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_dir,
                **model_kwargs,
            ).to(cfg.device)

        self.model.eval()

        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.use_cache = cfg.use_cache
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_dir,
            trust_remote_code=True,
            local_files_only=cfg.local_files_only,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            use_fast=True,
        )

    def analyze_pil(
        self,
        image: Image.Image,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> str:
        torch = get_torch()

        budget = max_tokens if max_tokens is not None else self.cfg.max_tokens
        image = fit_to_token_budget(image, budget)

        logger.debug(
            "Vision input resized to %sx%s with token_budget=%s",
            image.size[0],
            image.size[1],
            budget,
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.cfg.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        except RuntimeError:
            cleanup_cuda()
            raise

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]

        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image.",
        max_new_tokens: int = 128,
        max_tokens: Optional[int] = None,
    ) -> str:
        path = _coerce_image_path(image_path)
        image = Image.open(path).convert("RGB")
        return self.analyze_pil(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_tokens=max_tokens,
        )


_INSTANCES: dict[tuple[str, int, int, str], VisionCoder] = {}


def get_vision_coder(
    model_key: Optional[str] = None,
    torch_dtype=None,
    max_tokens: int = 512,
    min_tokens: int = 64,
) -> VisionCoder:
    torch = get_torch()

    key = _resolve_vision_model_key(model_key)
    device, dtype = _pick_device_and_dtype(torch, None, torch_dtype)

    cache_key = (key, min_tokens, max_tokens, str(dtype))

    if cache_key not in _INSTANCES:
        cfg = build_config(
            model_key=key,
            torch_dtype=dtype,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        _INSTANCES[cache_key] = VisionCoder(cfg)

    return _INSTANCES[cache_key]


def deepcoder_image_analysis(
    image_path,
    prompt: str = "please describe this image",
    max_new_tokens: int = 100,
    model_key: Optional[str] = None,
    torch_dtype=None,
    max_tokens: int = 512,
    min_tokens: int = 64,
):
    vision = get_vision_coder(
        model_key=model_key,
        torch_dtype=torch_dtype,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    return vision.analyze_image(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
    )
