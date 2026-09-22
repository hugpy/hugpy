"""Model-catalog domain constants: task vocabulary, servable pairs, defaults.

The ``DEFAULT_*_MODEL`` values here are the env-configured defaults from
``hugpy_platform.constants``; ``config.models.models_default`` reconciles them
against the live registry after discovery and patches ``TASK_DEFAULTS`` /
``MEDIA_DEFAULTS`` in place.
"""
from typing import Dict
from hugpy_platform.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_DEPTH_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_DETECT_MODEL,
    DEFAULT_IMAGEGEN_MODEL,
    DEFAULT_IMG_CLASSIFY_MODEL,
    DEFAULT_KEYWORDS_MODEL,
    DEFAULT_SEGMENT_MODEL,
    DEFAULT_SUMMARIZE_MODEL,
)
# pipeline_tag -> task vocabulary. Every value is HF-derivable.
HF_TASK_TO_TASKS = {
    "text-generation": ["text-generation"],
    "image-text-to-text": ["image-text-to-text", "text-generation"],
    "automatic-speech-recognition": ["automatic-speech-recognition"],
    # SPEECH OUT (chatterbox seat). Its absence here is why a row whose card
    # says ``pipeline_tag: text-to-speech`` fell through to _base_tasks'
    # conservative floor and advertised ["text-generation"] — a voice-cloning
    # model offering itself as a chat model. The pair
    # ("transformers","text-to-speech") is in RUNNER_PAIRS below, so the task is
    # servable, not merely visible.
    "text-to-speech": ["text-to-speech"],
    "text-summarization": ["text-summarization"],
    "text2text-generation": ["text-summarization", "text2text-generation"],
    # KeyBERT rides any sentence-transformers model, so embedding models
    # also serve keyword-extraction.
    "feature-extraction": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
    "sentence-similarity": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
    "text-to-image": ["text-to-image"],
    # VIDEO (2026-07-27). Advertised truthfully, NOT servable by the LLM plane:
    # there is no ("transformers","text-to-video") RUNNER_PAIR, so these rows are
    # visible in the catalogue and excluded from chat routing — the studio arm
    # (video_intel/studio) serves them through its own registry. Before this, a
    # Wan T2V model fell through _base_tasks' "conservative floor" and advertised
    # ["text-generation"]: a video diffusion model offering itself as a chat model.
    "text-to-video": ["text-to-video"],
    "image-to-video": ["image-to-video"],
    # NOTE: "image-to-image" is intentionally NOT a DISCOVERY pipeline_tag key.
    # A raw HF pipeline_tag=image-to-image (native EDIT models like
    # Qwen-Image-Edit) is still not auto-advertised from that tag alone. Instead,
    # img2img capability is DERIVED at the config layer
    # (models_config._derive_tasks): any image-GENERATION checkpoint — an
    # SD/SDXL/flux-class diffusers model or a comfy SD-lineage checkpoint — serves
    # image-to-image from the SAME weights it serves text-to-image with
    # (AutoPipelineForImage2Image / the comfy image-conditioned graph), so it
    # gains "image-to-image" whenever it advertises "text-to-image". (Operator
    # ruling 2026-07-05: primary_task / pipeline_tag is NOT a definitive capability
    # marker — plenty of models do img2img without saying so.) The
    # ("transformers","image-to-image") / ("comfy","image-to-image") RUNNER_PAIRS
    # entries below back that derivation.
    # Vision-analysis family — HF pipeline tags map 1:1.
    "depth-estimation": ["depth-estimation"],
    "object-detection": ["object-detection"],
    "image-classification": ["image-classification"],
    "image-segmentation": ["image-segmentation"],
}
RUNNER_PAIRS = {
    ("transformers", "text-generation"), ("gguf", "text-generation"),
    ("transformers", "image-text-to-text"), ("gguf", "image-text-to-text"),
    ("transformers", "automatic-speech-recognition"),
    ("transformers", "text-to-speech"),
    ("transformers", "text-summarization"), ("transformers", "text2text-generation"),
    ("transformers", "feature-extraction"), ("transformers", "sentence-similarity"),
    ("transformers", "text-to-image"), ("transformers", "image-to-image"),
    # ComfyUI engine rows (slice B) — the checkpoint lives in the WORKER's own
    # ComfyUI install; hugpy holds no files for these.
    ("comfy", "text-to-image"), ("comfy", "image-to-image"),
    ("transformers", "keyword-extraction"),
    ("transformers", "depth-estimation"), ("transformers", "object-detection"),
    ("transformers", "image-classification"), ("transformers", "image-segmentation"),
}

# Frameworks whose ("<framework>","image-to-image") runner+builder pair is wired
# above. DERIVED from RUNNER_PAIRS so it can't drift (same discipline as
# frameworks.KNOWN_TASKS_REGISTRY). A generative-image checkpoint on one of these
# frameworks serves img2img from the SAME weights it serves text-to-image with,
# so models_config._derive_tasks advertises "image-to-image" for it whenever it
# advertises "text-to-image". Today this is {"transformers", "comfy"}.
IMG2IMG_CAPABLE_FRAMEWORKS: frozenset = frozenset(
    framework for (framework, task) in RUNNER_PAIRS if task == "image-to-image"
)
MEDIA_DEFAULTS: Dict[str, str] = {
    "document": DEFAULT_CHAT_MODEL,
    "code":     DEFAULT_CHAT_MODEL,
    "text":     DEFAULT_CHAT_MODEL,
    "image":    DEFAULT_VISION_MODEL,
    "audio":    DEFAULT_WHISPER_MODEL,
    "video":    DEFAULT_WHISPER_MODEL,
}

TASK_DEFAULTS: Dict[str, str] = {
    "text-generation":              DEFAULT_CHAT_MODEL,
    "image-text-to-text":           DEFAULT_VISION_MODEL,
    "automatic-speech-recognition": DEFAULT_WHISPER_MODEL,
    "text-summarization":           DEFAULT_SUMMARIZE_MODEL,
    "text2text-generation":         DEFAULT_SUMMARIZE_MODEL,
    "feature-extraction":           DEFAULT_EMBED_MODEL,
    "sentence-similarity":          DEFAULT_EMBED_MODEL,
    "text-to-image":                DEFAULT_IMAGEGEN_MODEL,
    "image-to-image":               DEFAULT_IMAGEGEN_MODEL,
    "keyword-extraction":           DEFAULT_KEYWORDS_MODEL,
    "depth-estimation":             DEFAULT_DEPTH_MODEL,
    "object-detection":             DEFAULT_DETECT_MODEL,
    "image-classification":         DEFAULT_IMG_CLASSIFY_MODEL,
    "image-segmentation":           DEFAULT_SEGMENT_MODEL,
}
