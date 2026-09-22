import math
import os.path as osp
from PIL import Image

QWEN_PATCH = 28          # 14x14 ViT * 2x2 spatial merge
QWEN_TOKEN_BYTES = QWEN_PATCH * QWEN_PATCH  # 784 px per visual token


def pick_cpu_dtype(torch):
    # bf16 if the CPU supports it, else fp32. Never fp16 on CPU.
    if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def fit_to_token_budget(image: Image.Image, token_budget: int) -> Image.Image:
    """Downscale (never upscale) so visual-token count <= token_budget.
    Snaps to multiples of QWEN_PATCH so the processor doesn't resize again."""
    w, h = image.size
    pixel_budget = token_budget * QWEN_TOKEN_BYTES
    if w * h <= pixel_budget:
        return image
    scale = math.sqrt(pixel_budget / (w * h))
    nw = max(QWEN_PATCH, int(w * scale) // QWEN_PATCH * QWEN_PATCH)
    nh = max(QWEN_PATCH, int(h * scale) // QWEN_PATCH * QWEN_PATCH)
    return image.resize((nw, nh), Image.LANCZOS)
