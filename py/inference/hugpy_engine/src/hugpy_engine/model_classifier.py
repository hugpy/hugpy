"""What a model dir IS, read from the dir itself — offline, no hub, no path.

WHY THIS EXISTS (operator incident, 2026-07-31: "flux2 never fires", 7 attempts)
    Every flux2 key was task-misclassified, and the wrong task was stamped into
    all three stores (central discovery, worker discovery, the per-model
    hugpy.json sidecar):

      * ``FLUX.2-klein-base-9B-bucket-uncensored`` — a COMPLETE diffusers
        pipeline whose own ``model_index.json`` says
        ``{"_class_name": "Flux2KleinPipeline"}`` — was stamped
        ``tasks: ["image-to-image"]`` ONLY, so every text-to-image call refused.
      * ``Flux-Uncensored-V2`` — a LoRA-only dir (``lora.safetensors``, no
        ``model_index.json``) — was left ``tasks: null``, and null silently
        defaulted to text-generation, so an image LoRA was offered as an LLM and
        every image call refused with a nonsense
        ``supported: ['text-generation']``.

    The truth was in the directory the whole time. This module reads it, and
    ONLY it: no hub metadata (the fetch-once cache is a bonus, never a
    dependency), no task-from-path (the sticky-task landmine), no filename
    guessing beyond the adapter shape.

THE THREE ANSWERS
    pipeline   — ``model_index.json`` is present: its ``_class_name`` is
                 AUTHORITATIVE and outranks any stamped task, because it is the
                 pipeline's own declaration of what it runs.
    adapter    — only lora/adapter weights: NOT a servable model. It gets the
                 non-servable ``adapter`` task (never null, never
                 text-generation) so the row stays VISIBLE and refuses with a
                 reason that names the base to apply it to.
    unknown    — ``{}``: nothing here says anything. The caller decides; what it
                 must NOT do is assume text-generation (defaults-are-promises —
                 that default promised nonsense).

WHY SENTINEL TASKS AND NOT ``tasks: []``
    An empty ``tasks`` list makes the row fail ``assess_config`` (required
    fields reject ``[]``), which DROPS it from the registry — the silent-
    unavailability defect the operator named on 2026-07-29. So "not servable" is
    encoded the way ``pipeline-component`` already encodes it: a real task string
    with no RUNNER_PAIR. The row is listed, flagged unserveable, and carries a
    CAUSE + FIX.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

__all__ = [
    "MODEL_INDEX_NAME",
    "ADAPTER_TASK", "NEEDS_CLASSIFICATION_TASK",
    "T2I", "I2I", "INPAINT",
    "read_model_index", "pipeline_class_name", "tasks_for_pipeline_class",
    "is_adapter_only_dir", "classify_model_dir", "is_speech_checkpoint_dir",
    "TTS",
    "adapter_refusal", "needs_classification_refusal",
]

MODEL_INDEX_NAME = "model_index.json"

# Non-servable task tokens. Neither has a RUNNER_PAIR, so a row carrying one is
# kept-and-visible but refuses with a reason (see models_config.derive_model_config_row).
ADAPTER_TASK = "adapter"
NEEDS_CLASSIFICATION_TASK = "needs-classification"

T2I = "text-to-image"
I2I = "image-to-image"
INPAINT = "image-inpainting"
TTS = "text-to-speech"

# What a CHATTERBOX-family speech checkpoint looks like on disk: a voice encoder,
# a T3 text-to-token model, an S3 generator and a tokenizer — the exact fileset
# ``ChatterboxTTS.from_local`` reads. Such a dir carries NO transformers
# ``config.json`` and NO diffusers ``model_index.json``, so before this arm it
# said "nothing" and fell to the conservative text-generation floor: a
# voice-cloning model offering itself as a chat model (the audio twin of the
# 2026-07-31 image-LoRA incident, and the reason ``Viral2AI~chatterbox`` carried
# ``tasks: ["text-generation"]`` while its own card said ``pipeline_tag:
# text-to-speech``).
#
# Deliberately a CONJUNCTION of three prefixes, not a name match: a dir that
# merely mentions "chatterbox" proves nothing, while these three weight families
# together are what the loader actually requires.
_SPEECH_WEIGHT_PREFIXES = (("ve.",), ("t3_", "t3."), ("s3gen",))


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 — absent/unreadable/malformed are all "no"
        return None
    return data if isinstance(data, dict) else None


def read_model_index(model_dir: Optional[str]) -> Optional[dict]:
    """``model_index.json`` parsed, or None when absent/unreadable."""
    if not model_dir:
        return None
    return _read_json(os.path.join(model_dir, MODEL_INDEX_NAME))


def pipeline_class_name(model_dir: Optional[str], *,
                        index: Optional[dict] = None) -> Optional[str]:
    """The diffusers pipeline class this dir declares, or None."""
    idx = index if index is not None else read_model_index(model_dir)
    cls = (idx or {}).get("_class_name")
    cls = cls.strip() if isinstance(cls, str) else ""
    return cls or None


# Pipeline families the VIDEO arm owns. Their model_index.json is a pipeline
# declaration too, but the video vocabulary (text-to-video / image-to-video) is
# derived in models_config._correct_video_task from the SAME file — classifying
# them here as image pipelines would fight that corrector. Deliberately deferred.
_VIDEO_PIPELINE_PREFIXES = ("wan", "ltx", "cog", "hunyuanvideo", "mochi",
                            "allegro", "stablevideo", "svd", "pyramid")

# Explicit class -> tasks. Small on purpose: the suffix fallback below handles
# every pipeline we have not met yet, and this map exists for the cases where the
# NAME alone would mislead. A t2i-capable pipeline ALWAYS gets image-to-image
# too — the same weights serve img2img through the AutoPipelineForImage2Image
# family (operator ruling 2026-07-05: pipeline_tag is not a capability marker).
_CLASS_TASKS: Dict[str, List[str]] = {
    "stablediffusionpipeline":            [T2I, I2I],
    "stablediffusionxlpipeline":          [T2I, I2I],
    "stablediffusion3pipeline":           [T2I, I2I],
    "fluxpipeline":                       [T2I, I2I],
    "flux2pipeline":                      [T2I, I2I],
    "flux2kleinpipeline":                 [T2I, I2I],   # the 2026-07-31 incident
    "qwenimagepipeline":                  [T2I, I2I],
    "kandinskypipeline":                  [T2I, I2I],
    "pixartalphapipeline":                [T2I, I2I],
    "stablediffusionimg2imgpipeline":     [I2I],
    "stablediffusionxlimg2imgpipeline":   [I2I],
    "fluxkontextpipeline":                [I2I],
    "qwenimageeditpipeline":              [I2I],
    "stablediffusioninpaintpipeline":     [I2I, INPAINT],
    "stablediffusionxlinpaintpipeline":   [I2I, INPAINT],
    "fluxinpaintpipeline":                [I2I, INPAINT],
}

# model_index.json component keys that prove this is a GENERATIVE image pipeline:
# something that encodes the prompt, plus something that denoises.
_TEXT_ENCODER_KEYS = ("text_encoder", "text_encoder_2", "text_encoder_3")
_DENOISER_KEYS = ("transformer", "unet")


def _declares_component(index: dict, keys) -> bool:
    return any(k in index for k in keys)


def tasks_for_pipeline_class(class_name: Optional[str], *,
                             index: Optional[dict] = None) -> Optional[List[str]]:
    """Tasks a diffusers ``_class_name`` declares, or None when it declares none.

    Order: the explicit map, then a suffix fallback for classes we have not met.
    The fallback is where the incident's real fix lives — a pipeline nobody has
    enumerated yet still classifies from its own shape:

      * ``*Img2Img*`` / ``*Edit*`` / ``*Kontext*``  -> image-to-image only (no
        text-only path; these need a start image).
      * ``*Inpaint*``                               -> image-to-image + inpainting.
      * any other ``*Pipeline`` that declares a text encoder AND a
        transformer/unet                            -> text-to-image + image-to-image.

    Returns None for video pipelines (the video corrector owns them) and for a
    class that is not a pipeline at all — "I have nothing to say" is a real
    answer, and it must not be confused with a task.
    """
    if not class_name:
        return None
    low = class_name.lower()
    if any(low.startswith(p) for p in _VIDEO_PIPELINE_PREFIXES):
        return None                                    # the video arm's vocabulary
    mapped = _CLASS_TASKS.get(low)
    if mapped:
        return list(mapped)
    if not low.endswith("pipeline"):
        return None
    if "inpaint" in low:
        return [I2I, INPAINT]
    if any(s in low for s in ("img2img", "imagetoimage", "edit", "kontext")):
        return [I2I]
    idx = index or {}
    if idx and not (_declares_component(idx, _TEXT_ENCODER_KEYS)
                    and _declares_component(idx, _DENOISER_KEYS)):
        # A pipeline that neither encodes text nor denoises is not an image
        # generator (an upscaler, a safety checker bundle, an audio pipeline).
        # Say nothing rather than mint a task it cannot serve.
        return None
    return [T2I, I2I]


# What a bare adapter dir looks like on disk. `lora` in the filename is the
# fleet's actual shape for diffusers LoRAs (``lora.safetensors``,
# ``pytorch_lora_weights.safetensors``) — those ship NO adapter_config.json, which
# is exactly why Flux-Uncensored-V2 read as "no information" and fell to the
# text-generation default.
_ADAPTER_MARKER_FILES = ("adapter_config.json", "adapter_model.safetensors",
                         "adapter_model.bin")
#
# TOKEN-BOUNDARY, not substring (verified against the real store, 2026-07-31):
# ``anyloracheckpoint-bakedvaeblessedfp16.safetensors`` is a COMPLETE SD
# checkpoint whose name merely contains the letters "lora", and a substring test
# demoted it to an adapter — turning a working text-to-image model into an
# unserveable row. The marker has to be a word in the filename
# (``lora.safetensors``, ``pytorch_lora_weights.safetensors``,
# ``adapter_model.safetensors``), not a fragment inside another word.
_ADAPTER_NAME_RE = re.compile(
    r"(?:^|[^a-z])(lora|adapter|lycoris|locon)(?:[^a-z]|$)", re.IGNORECASE)
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _weight_files(names) -> List[str]:
    return [n for n in names if n.lower().endswith(_WEIGHT_SUFFIXES)]


def is_adapter_only_dir(model_dir: Optional[str]) -> bool:
    """True iff this dir holds ONLY adapter weights — a delta, not a model.

    Refuses to fire when the dir can stand on its own: a ``model_index.json``
    (diffusers pipeline) or a ``config.json`` naming architectures/model_type
    (transformers model) both mean "loadable", and a merged checkpoint that
    happens to ship an adapter config alongside real weights is NOT an adapter.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    if read_model_index(model_dir) is not None:
        return False
    cfg = _read_json(os.path.join(model_dir, "config.json")) or {}
    if cfg.get("model_type") or cfg.get("architectures"):
        return False
    try:
        names = os.listdir(model_dir)
    except OSError:
        # "can't tell" is never reported as "adapter" (see the EMFILE
        # false-negative doctrine in peft_adapters.find_base_model_dir).
        return False
    weights = _weight_files(names)
    if not weights:
        return False
    has_peft_marker = any(m in names for m in _ADAPTER_MARKER_FILES)
    all_adapter_named = all(_ADAPTER_NAME_RE.search(w) for w in weights)
    return has_peft_marker or all_adapter_named


def _declared_base_model(model_dir: str) -> Optional[str]:
    """The base an adapter names in ``adapter_config.json``, or None."""
    cfg = _read_json(os.path.join(model_dir, "adapter_config.json")) or {}
    base = cfg.get("base_model_name_or_path")
    base = base.strip() if isinstance(base, str) else ""
    return base or None


def is_speech_checkpoint_dir(model_dir: Optional[str]) -> bool:
    """True iff this dir holds a chatterbox-family TTS checkpoint.

    CONTENT-AUTHORITATIVE and narrow (k61's rule, applied to audio): the answer
    comes from the weight files the backend's own loader reads, never from the
    directory name or a path segment. A dir that also carries a transformers
    ``config.json`` or a diffusers ``model_index.json`` is something else and is
    left alone — this speaks only for the shape that declares nothing else.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    if read_model_index(model_dir) is not None:
        return False
    cfg = _read_json(os.path.join(model_dir, "config.json")) or {}
    if cfg.get("model_type") or cfg.get("architectures"):
        return False
    try:
        names = [n.lower() for n in os.listdir(model_dir)]
    except OSError:
        return False                      # "can't tell" is never "yes"
    weights = _weight_files(names)
    if not weights:
        return False
    for prefixes in _SPEECH_WEIGHT_PREFIXES:
        if not any(w.startswith(prefixes) for w in weights):
            return False
    return any(n in ("tokenizer.json", "mtl_tokenizer.json") for n in names)


def classify_model_dir(model_dir: Optional[str]) -> dict:
    """What the dir says it is — ``{}`` when it says nothing.

    Keys when non-empty: ``tasks``, ``primary_task``, ``source``
    ("model_index" | "adapter_dir"), and ``adapter`` (True only for adapters).
    Never raises: discovery walks whatever is on disk, and an unreadable dir must
    degrade to "unknown", not break the walk.
    """
    index = read_model_index(model_dir)
    if index is not None:
        cls = pipeline_class_name(model_dir, index=index)
        tasks = tasks_for_pipeline_class(cls, index=index)
        if tasks:
            return {"tasks": tasks, "primary_task": tasks[0],
                    "pipeline_class": cls, "source": "model_index"}
        return {}
    if is_speech_checkpoint_dir(model_dir):
        # A speech checkpoint is SERVABLE (unlike the adapter/unclassified
        # verdicts below): ("transformers","text-to-speech") has a runner and a
        # builder, so this task makes the row usable rather than merely visible.
        return {"tasks": [TTS], "primary_task": TTS,
                "source": "speech_checkpoint"}
    if is_adapter_only_dir(model_dir):
        # A PEFT adapter that NAMES its base is PAIRABLE: the registry's
        # base-present gate and resolve_adapter_pair already serve it (base +
        # delta) or refuse by naming the base to acquire. Overriding its task
        # here would break that working path, so this defers — it speaks only for
        # the shape nothing can pair, the bare diffusers LoRA (a lone
        # ``lora.safetensors`` with no base recorded anywhere), which is the shape
        # that was left null and read as a chat model.
        if _declared_base_model(model_dir):
            return {}
        return {"tasks": [ADAPTER_TASK], "primary_task": ADAPTER_TASK,
                "adapter": True, "source": "adapter_dir"}
    return {}


def adapter_refusal(name: str, *, base_model: Optional[str] = None) -> str:
    """Why an adapter row cannot serve, and what to do about it."""
    base = f" (base {base_model!r})" if base_model else ""
    return (f"{name}: this directory holds ADAPTER weights only{base} — a LoRA "
            f"is a delta applied to a base model, not a servable model, so it "
            f"has no task of its own. FIX: call the BASE image model and pass "
            f"this adapter as a LoRA, or merge the adapter into its base and "
            f"register the merged checkpoint.")


def needs_classification_refusal(name: str) -> str:
    """Why an unclassified row cannot serve, and what to do about it.

    NEVER 'assume text-generation'. Null means unclassified, and unclassified
    means refuse with the remedy named — that is the whole point of k61.
    """
    return (f"{name}: unclassified — nothing in this model's own directory "
            f"(no model_index.json pipeline class, no config.json model_type/"
            f"architectures, no pipeline tag) says what task it serves, so it is "
            f"NOT assumed to be a chat model. FIX: re-derive from disk with "
            f"`hugpy reclassify-images --apply` (or POST /llm/models/reclassify-images "
            f"{{\"apply\": true}}), or stamp the real tasks into the model's "
            f"hugpy.json marker.")


# ---------------------------------------------------------------------------
# Lineage: which repo a model was derived from (moved down from
# hugpy_curation.review.screen so the registry and the review screen parse
# base_model identically).
# ---------------------------------------------------------------------------
def base_model_of(payload: dict) -> str | None:
    """The repo this one was derived from.

    card_data.base_model when we have the card, else the ``base_model:<id>``
    tag the Hub synthesises — which is all the central metadata cache keeps,
    and is the only lineage signal available for a GGUF-only repo.
    """
    bm = payload.get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    if isinstance(bm, str) and bm:
        return bm
    plain, derived = None, None
    for tag in (payload.get("tags") or []):
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            continue
        rest = tag.split(":", 1)[1]
        if ":" in rest:                       # base_model:quantized:<id>
            derived = derived or rest.split(":", 1)[1]
        else:
            plain = plain or rest
    return plain or derived
