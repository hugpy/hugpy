"""models_config.py — the registry.

MODELS is the authoritative base: the real, curated models. Discovery finds
everything else on disk (test downloads) and appends it at build time. Staples
are never overwritten — a discovered row is skipped if its model_key OR its
cleaned hub_id already belongs to a staple (prevents same-path collisions like
Falconsai-text-summarization vs a discovered text_summarization).

Build order:
    MODELS (curated)  +  discovery report (derived)  ->  ModelConfig registry

Import is cheap: it merges MODELS with whatever discovery report already exists
on disk. To re-walk the model tree (HF metadata, network), call
refresh_registry() explicitly — e.g. on hugpy module startup.
"""

import json
import os
import re
from dataclasses import MISSING, fields
from typing import Dict
from abstract_essentials import get_logFile, safe_dump_to_file, safe_load_from_json
from hugpy_engine.categories import HF_TASK_TO_TASKS, IMG2IMG_CAPABLE_FRAMEWORKS, RUNNER_PAIRS
from hugpy_engine.schemas.model_schemas import ModelConfig
from hugpy_storage.model_paths import route_destination
from hugpy_platform.constants import (
    DEFAULT_MAX_TOKENS,
    MODELS_DICT_PATH,
    MODELS_DISCOVERY_PATH,
    MODELS_HOME,
)
from hugpy_engine.model_classifier import (
    ADAPTER_TASK,
    NEEDS_CLASSIFICATION_TASK,
    TTS as TTS_TASK,
    adapter_refusal,
    classify_model_dir,
    is_speech_checkpoint_dir,
    needs_classification_refusal,
    pipeline_class_name,
    tasks_for_pipeline_class,
)

# EMFILE burst hardening (incident 2026-07-23): media_models.json lives on the
# virtiofs mount; the restart-open burst threw EMFILE here (logged via
# safe_read_from_json). Retry the store-read past the transient. Best-effort
# import so a packaging skew can never break config loading.
try:
    from hugpy_control.shared import retry_on_emfile
except Exception:  # pragma: no cover - degrade to un-retried read on import skew
    def retry_on_emfile(fn, **_kw):  # type: ignore[misc]
        return fn()

logger = get_logFile(__name__)


# ===========================================================================
# Base registry — the real models. Authoritative.
# ===========================================================================
# Stock fleet: exactly one default model per task (a model may serve its whole
# task group). Efficiency-first, non-gargantuan picks — ~17GB all-in. Bigger
# siblings (flan-t5-xl, whisper-large-v3, sdxl-turbo, Qwen2.5-VL-7B, LED-16384,
# gte-large) stay available as opt-in installs via discovery; they're just not
# staples. Every DEFAULT_* constant in constants.py points at a key below.
MODELS = {
    "Qwen2.5-3B-Instruct-GGUF": {
        "model_max_length": 32768, "include": None, "name": "Qwen2.5-3B-Instruct-GGUF",
        "framework": "gguf", "hub_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "folder": "Qwen/Qwen2.5-3B-Instruct-GGUF", "tasks": ["text-generation"],
        "primary_task": "text-generation", "port": None,
    },

    # Torch-free vision: llama.cpp serves the LM gguf with the mmproj CLIP
    # projector via --mmproj (find_mmproj auto-discovers it beside the model).
    # `include` pulls BOTH files; `filename` is the LM so get_gguf_file resolves
    # it and skips the projector. This is DEFAULT_VISION_MODEL — preferred over
    # the transformers variant on CPU/phone workers that can't install torch.
    "Qwen2.5-VL-3B-Instruct-GGUF": {
        "model_max_length": 32768,
        "include": ["Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                    "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"],
        "name": "Qwen2.5-VL-3B-Instruct-GGUF",
        "framework": "gguf", "hub_id": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "folder": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        "tasks": ["image-text-to-text", "text-generation"],
        "primary_task": "image-text-to-text", "port": None,
    },
    "whisper-large-v3-turbo": {
        "model_max_length": 448, "include": None, "name": "whisper-large-v3-turbo",
        "framework": "transformers", "hub_id": "openai/whisper-large-v3-turbo", "filename": None,
        "folder": "openai/whisper-large-v3-turbo", "tasks": ["automatic-speech-recognition"],
        "primary_task": "automatic-speech-recognition", "port": None,
    },
    "flan-t5-large": {
        "model_max_length": 1024, "include": None, "name": "flan-t5-large",
        "framework": "transformers", "hub_id": "google/flan-t5-large", "filename": None,
        "folder": "google/flan-t5-large", "tasks": ["text-summarization", "text2text-generation"],
        "primary_task": "text-summarization", "port": None,
    },
    "all-minilm-l6-v2": {
        "model_max_length": 512, "include": None, "name": "all-minilm-l6-v2",
        "framework": "transformers", "hub_id": "sentence-transformers/all-minilm-l6-v2",
        "filename": None, "folder": "sentence-transformers/all-minilm-l6-v2",
        "tasks": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
        "primary_task": "feature-extraction", "port": None,
    },
    # ComfyUI-backed staple (engine slice B, 2026-07-03): `filename` names the
    # checkpoint inside the WORKER's own ComfyUI models/checkpoints — hugpy
    # holds no files for comfy rows. Routes to whichever worker advertises
    # supports_comfy AND is assigned this model.
    "comfy-dreamshaper-8": {
        "model_max_length": 77, "include": None, "name": "comfy-dreamshaper-8",
        "framework": "comfy", "hub_id": "Lykon/DreamShaper",
        "filename": "DreamShaper_8_pruned.safetensors",
        "folder": "comfy/DreamShaper",
        "tasks": ["text-to-image", "image-to-image"],
        "primary_task": "text-to-image", "port": None,
    },

    "sd-turbo": {
        "model_max_length": 77, "include": None, "name": "sd-turbo",
        "framework": "transformers", "hub_id": "stabilityai/sd-turbo", "filename": None,
        "folder": "stabilityai/sd-turbo",
        # GO-LIVE flipped 2026-07-03 (was held): the fleet wheel is 0.1.95
        # (op/computron converged — carries Img2ImgRunner), so advertising
        # image-to-image now routes to workers that actually serve it.
        # Runner/builder/RUNNER_PAIRS were wired (inert) in the same release;
        # validate_registry accepts this.
        "tasks": ["text-to-image", "image-to-image"],
        "primary_task": "text-to-image", "port": None,
    },

    # Vision-analysis staples — small, permissively-licensed defaults for the
    # generic pipeline runner family (managers/vision_analysis). Each is the
    # TASK_DEFAULTS anchor for its task; heavier variants ride discovery.
    "depth-anything-v2-small": {
        "model_max_length": None, "include": None, "name": "depth-anything-v2-small",
        "framework": "transformers", "hub_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "filename": None, "folder": "depth-anything/Depth-Anything-V2-Small-hf",
        "tasks": ["depth-estimation"], "primary_task": "depth-estimation", "port": None,
    },
    "detr-resnet-50": {
        "model_max_length": None, "include": None, "name": "detr-resnet-50",
        "framework": "transformers", "hub_id": "facebook/detr-resnet-50",
        "filename": None, "folder": "facebook/detr-resnet-50",
        "tasks": ["object-detection"], "primary_task": "object-detection", "port": None,
    },
    "vit-base-patch16-224": {
        "model_max_length": None, "include": None, "name": "vit-base-patch16-224",
        "framework": "transformers", "hub_id": "google/vit-base-patch16-224",
        "filename": None, "folder": "google/vit-base-patch16-224",
        "tasks": ["image-classification"], "primary_task": "image-classification", "port": None,
    },
    "segformer-b0-ade": {
        "model_max_length": None, "include": None, "name": "segformer-b0-ade",
        "framework": "transformers", "hub_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "filename": None, "folder": "nvidia/segformer-b0-finetuned-ade-512-512",
        "tasks": ["image-segmentation"], "primary_task": "image-segmentation", "port": None,
    },
}


# ===========================================================================
# Derivation — discovery/manifest row -> ModelConfig-ready dict.
# Pure; no torch, no runner-stack import (so building the registry never drags
### the inference stack in). RUNNER_PAIRS mirrors FRAMEWORK_RUNNERS statically.
# ===========================================================================
DEFAULT_MAX_TOKENS_LOCAL = DEFAULT_MAX_TOKENS

_FAMILIES = {"gguf", "transformers", "misc", "datasets", "models"}


def _clean_repo_id(hub_id):
    """Strip storage-path leakage (gguf/text-generation/owner/repo, leading
    slashes) back to owner/repo — the only shape HF and routing accept."""
    parts = (hub_id or "").strip("/").split("/")
    while len(parts) > 2 and parts[0] in _FAMILIES:
        parts = parts[1:]
        if parts and parts[0] not in _FAMILIES:
            parts = parts[1:]
    return "/".join(parts)

def base_present(base_model: str) -> bool:
    """True if a PEFT adapter's base model is actually on disk.

    Non-adapters (base_model falsy) pass trivially. An adapter passes only
    if route_destination's base dir exists AND holds real weights — a bare
    dir with just a config doesn't count.
    """
    if not base_model:
        return True
    base_dir = route_destination(
        {"hub_id": base_model,
         "framework": "transformers",
         "primary_task": "text-generation"}
    )
    if not os.path.isdir(base_dir):
        return False
    try:
        return any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in os.listdir(base_dir)
        )
    except OSError:
        return False


_SEQ2SEQ = {"t5", "led", "bart", "pegasus", "mbart", "mt5", "longt5"}
_EMBED   = {"bert", "new", "roberta", "mpnet", "nomic_bert"}
_ASR     = {"whisper"}
_VISION  = {"qwen2_5_vl", "minicpmv4_6", "mllama", "idefics3", "internvl"}
# VIDEO diffusion model_types. Wan's own config.json SAYS what it is —
# {"_class_name": "WanModel", "model_type": "t2v", "_diffusers_version": "0.30.0"} —
# and _base_tasks already reads model_type. It just had no video vocabulary, so
# every Wan row fell through to the "conservative floor" and advertised
# ["text-generation"]: a video diffusion model offering itself as a chat model,
# which then earned it a 4-bit bitsandbytes lever, an LLM ctx, an LLM VRAM price,
# and eligibility for chat routing. Believe the model when it names itself.
_VIDEO_T2V = {"t2v", "ti2v"}
_VIDEO_I2V = {"i2v", "vace"}

def _safe_path_part(value):
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/\-]+", "_", value)
    value = re.sub(r"/+", "/", value)
    return value.strip("/")

def _runtime_folder(framework, hub_id, include=None, filename=None):
    framework = (framework or "").lower().strip()
    if framework == "gguf": return "gguf"
    if filename and filename.lower().endswith(".gguf"): return "gguf"
    if include:
        pats = include if isinstance(include, list) else [include]
        if any("gguf" in p.lower() for p in pats): return "gguf"
    return "transformers" if framework == "transformers" else "misc"

def _routed_folder(framework, task, hub_id, filename=None, include=None):
    """Predicted MODELS_HOME-relative folder — only used when the model isn't
    on disk yet, so there's no real dir to record."""
    if task == "dataset": return None
    return f"{_runtime_folder(framework, hub_id, include, filename)}/{_safe_path_part(task)}/{_safe_path_part(hub_id)}"

def _resolve_folder(row, framework, task, hub_id, filename, include):
    """Real dir wins; then an already-routed folder; then a prediction."""
    abs_dir = row.get("dir")
    if abs_dir and MODELS_HOME:
        rel = os.path.relpath(abs_dir, MODELS_HOME)
        if not rel.startswith(".."):
            return rel
    f = row.get("folder")
    if f and len(f.strip("/").split("/")) >= 3:   # looks like runtime/task/owner/repo already
        return f.strip("/")
    return _routed_folder(framework, task, hub_id, filename, include) or hub_id
def _derive_framework(name, hub_id, row):
    # HF-canonical vocabulary (operator directive 2026-07-05): "gguf" is HF
    # Hub's library tag for GGUF repos — "llama_cpp" is retired as a value.
    if row.get("framework"):
        return row["framework"]
    blob = f"{name} {hub_id}".lower()
    tags = [t.lower() for t in (row.get("tags") or [])]
    return "gguf" if ("gguf" in blob or "gguf" in tags) else "transformers"


def _augment_img2img(framework, tasks):
    """Advertise image-to-image for models that inherently support it.

    Operator ruling (2026-07-05): primary_task / pipeline_tag is NOT a definitive
    capability marker. A generative-IMAGE checkpoint — an SD/SDXL/flux-class
    diffusers model or a comfy SD-lineage checkpoint — serves image-to-image from
    the SAME weights it serves text-to-image with (AutoPipelineForImage2Image /
    the comfy image-conditioned graph). So whenever a model on an img2img-capable
    framework (transformers / comfy — derived in IMG2IMG_CAPABLE_FRAMEWORKS) lists
    text-to-image, it also gets image-to-image.

    Conservative by construction: it ONLY widens models that already generate
    images (text-to-image present). Text LLMs, VL chat, whisper, embeddings and
    the vision-analysis family never carry text-to-image, so none of them gain
    img2img. Idempotent — a model that already lists image-to-image (sd-turbo, the
    swept comfy checkpoints, native edit models) is returned unchanged.
    """
    if ("text-to-image" in tasks
            and framework in IMG2IMG_CAPABLE_FRAMEWORKS
            and "image-to-image" not in tasks):
        return [*tasks, "image-to-image"]               # new list — never mutate the row's
    return tasks


def _base_tasks(framework, row):
    tasks = row.get("tasks")
    if tasks:
        return tasks if isinstance(tasks, list) else [tasks]
    if framework == "gguf":
        # Vision GGUFs (a VL model + its mmproj sidecar) ride the chat path but
        # MUST advertise image-text-to-text, or they're dropped from that task
        # ("has no runner") and only ever answer text. Detect by model_type or a
        # "VL" marker in the id/folder/filename.
        _blob = " ".join(str(row.get(k) or "") for k in ("hub_id", "folder", "filename", "name")).lower()
        _mtype = (row.get("model_type") or "").lower()
        if _mtype in _VISION or any(s in _blob for s in ("-vl-", "-vl.", "vl-instruct", "qwen2.5-vl", "qwen2_5_vl")):
            return ["image-text-to-text", "text-generation"]
        return ["text-generation"]                      # all the gguf runner serves
    pt = row.get("pipeline_tag") or row.get("primary_task") or row.get("task")
    if pt in HF_TASK_TO_TASKS:
        return list(HF_TASK_TO_TASKS[pt])
    m = (row.get("model_type") or "").lower()
    if m in _SEQ2SEQ: return ["text-summarization", "text2text-generation"]
    if m in _EMBED:   return ["feature-extraction", "sentence-similarity"]
    if m in _ASR:     return ["automatic-speech-recognition"]
    if m in _VISION:  return ["image-text-to-text", "text-generation"]
    if m in _VIDEO_T2V: return ["text-to-video"]
    if m in _VIDEO_I2V: return ["image-to-video"]
    # THE FLOOR IS NOT A DEFAULT (k61, 2026-07-31). It stands only on POSITIVE
    # evidence that this is a transformers-shaped model: a model_type, an
    # architectures list, or a hub pipeline/library tag. A row with NONE of those
    # is not a chat model — it is a row nobody has classified, and the old
    # unconditional `["text-generation"]` is what let an image LoRA be offered as
    # an LLM (and refuse every image call with "supported: ['text-generation']").
    # Unclassified says so, and refuses with the remedy named.
    if (m or row.get("architectures") or row.get("pipeline_tag")
            or row.get("library_name")):
        return ["text-generation"]                      # conservative floor
    return [NEEDS_CLASSIFICATION_TASK]


def _derive_tasks(framework, row):
    """A row's servable task list. Base derivation (curated/discovery vocabulary),
    then the img2img capability widening (see _augment_img2img)."""
    return _augment_img2img(framework, _base_tasks(framework, row))


def _correct_gguf_vision(framework, tasks, row):
    """A GGUF is vision (image-text-to-text) ONLY if it actually ships an mmproj
    projector. HF pipeline_tag and the storage-layout task folder are unreliable
    (operator ruling 2026-07-05) — a TEXT gguf mis-tagged/mis-routed under
    image-text-to-text has no projector, so `model_looks_downloaded` demands an
    mmproj it never had and the model reads 'incomplete' (and can't be allocated).
    Content-authoritative: DOWNGRADE to text only when we can SEE the dir and it
    has no projector — never fight a not-yet-downloaded vision model we can't
    inspect, and never touch a genuine vision gguf (mmproj on disk / in include)."""
    if framework != "gguf" or "image-text-to-text" not in tasks:
        return tasks
    # Intrinsic vision signal (VL name / vision model_type) — same test as
    # _base_tasks. A real VL gguf stays vision even without the projector on disk;
    # model_looks_downloaded then flags it 'incomplete' (fetch the mmproj) rather
    # than silently serving a vision model as blind text. We only STRIP vision
    # when the label came PURELY from the path / HF pipeline_tag (no intrinsic
    # signal) AND no projector is declared or present.
    _blob = " ".join(str(row.get(k) or "") for k in ("hub_id", "folder", "filename", "name")).lower()
    if (row.get("model_type") or "").lower() in _VISION or any(
            s in _blob for s in ("-vl-", "-vl.", "vl-instruct", "qwen2.5-vl", "qwen2_5_vl")):
        return tasks                                    # intrinsically vision
    inc = row.get("include") or []
    _dir = row.get("dir")
    try:
        from hugpy_platform.utils import is_mmproj_file, find_mmproj
        declares = any(is_mmproj_file(x) for x in inc)
        has_on_disk = bool(_dir and find_mmproj(_dir))
    except Exception:  # noqa: BLE001 — can't inspect → leave the task untouched
        return tasks
    if declares or has_on_disk:
        return tasks                                    # genuine vision gguf
    if _dir:                                            # inspectable + no projector → text
        return [t for t in tasks if t != "image-text-to-text"] or ["text-generation"]
    return tasks                                        # not on disk, no declared mmproj


def _correct_video_task(framework, tasks, row):
    """A model whose own config says it is VIDEO must not advertise text-generation.

    CONTENT-AUTHORITATIVE OVERRIDE, deliberately shaped like ``_correct_gguf_vision``:
    it corrects an ALREADY-DERIVED task list from what is on disk, because by the
    time we get here the task may not have come from ``_base_tasks`` at all.

    ⚠ WHY A CORRECTOR AND NOT JUST A model_type BRANCH. ``_base_tasks`` short-circuits
    on a stored value:

        tasks = row.get("tasks")
        if tasks: return tasks

    Discovery stamps ``tasks`` from the STORAGE LAYOUT PATH (the sticky-task-from-path
    landmine), so every Wan row arrived carrying ``["text-generation"]`` and the
    model_type check below it never ran. Teaching ``_base_tasks`` about video was
    necessary and NOT sufficient — it fixed only rows that reach it undecided.

    Fires only when the config's own ``model_type`` names a video architecture, so a
    real text model can never be re-labelled by this. Absent/unreadable config → the
    task list is returned untouched."""
    if framework == "gguf" or not ({"text-generation", NEEDS_CLASSIFICATION_TASK}
                                   & set(tasks)):
        return tasks
    mt = (row.get("model_type") or "").lower()
    if not mt:
        enriched = _enrich_model_type(framework, row)
        mt = (enriched.get("model_type") or "").lower()
    if mt in _VIDEO_T2V:
        return ["text-to-video"]
    if mt in _VIDEO_I2V:
        return ["image-to-video"]
    return tasks


# Path segments that mark a GGUF as a COMPONENT of a diffusion/video pipeline
# rather than a standalone servable model. A diffusion checkpoint splits into
# sub-trees (the text encoder, the VAE, the transformer/unet); the encoder is
# very often a real LLM architecture in its own right (LTX-2 ships a Gemma-3
# text encoder), so it CANNOT be told apart from a chat model by its header —
# only by WHERE it lives. LTX-2.3-uncensored-fp8 was routed to the llama chat
# path and rejected at load because its registered file is
# ``split/text_encoders/gemma-3-12b-it-...gguf``.
_PIPELINE_COMPONENT_DIRS = ("text_encoders", "text_encoder", "vae",
                            "transformer", "unet", "image_encoder")


def _correct_pipeline_component(framework, tasks, row):
    """A GGUF that is a sub-component of a diffusion/video pipeline is NOT a
    chat model, however chat-like its architecture looks.

    PATH-AUTHORITATIVE, and deliberately narrow: it fires ONLY when the row's
    own file path contains a pipeline-component directory segment
    (``.../text_encoders/…``, ``.../vae/…``). A standalone Gemma/Qwen GGUF lives
    at its model root, never under one of those segments, so this can never
    re-label a genuine chat model. Returns ``["pipeline-component"]`` — a task
    with no runner, so the derive keeps the row visible but flags it
    ``serveable: False`` with an honest reason instead of letting it advertise
    text-generation and crash at the loader (silent-unavailability-by-
    misclassification, operator doctrine 2026-07-29: a model must be knowable,
    not hidden behind a misleading failure)."""
    if framework != "gguf":
        return tasks
    fn = str(row.get("filename") or row.get("effective_gguf") or "")
    segs = {s.lower() for s in fn.replace("\\", "/").split("/")}
    if segs & set(_PIPELINE_COMPONENT_DIRS):
        return ["pipeline-component"]
    return tasks


def _correct_diffusers_task(framework, tasks, row):
    """A diffusers pipeline's ``model_index.json`` is AUTHORITATIVE about its task.

    CONTENT-AUTHORITATIVE OVERRIDE, shaped like ``_correct_video_task`` and for the
    same reason: by the time we get here the task list may not have come from
    ``_base_tasks`` at all — ``_base_tasks`` short-circuits on a STORED value, and
    the stored value is exactly what was wrong.

    THE INCIDENT (2026-07-31). ``FLUX.2-klein-base-9B-bucket-uncensored`` is a
    complete pipeline — ``model_index.json`` says ``Flux2KleinPipeline`` — but all
    three task stores had it stamped ``["image-to-image"]`` only, so every
    text-to-image call refused and the operator's flux2 attempts all failed. The
    pipeline had been declaring the truth about itself the whole time; nothing read
    it. Now the declaration WINS over the stamp, which is what stops the fleet's
    hand-corrected data from regressing on the next walk.

    Narrow by construction: fires only when the dir actually carries a
    ``model_index.json`` whose ``_class_name`` maps to image tasks. Video pipelines
    (Wan/LTX/Cog/…) return None from the classifier and are left to
    ``_correct_video_task``; a dir with no model_index.json is untouched."""
    if framework == "gguf":
        return tasks
    d = row.get("dir")
    if not d:
        return tasks
    cls = pipeline_class_name(d)
    if not cls:
        return tasks
    derived = tasks_for_pipeline_class(cls, index=None)
    if not derived:
        return tasks
    if list(tasks) != list(derived):
        logger.info("classify: %s declares %s -> tasks %s (was %s)",
                    row.get("name") or row.get("hub_id") or d, cls, derived, tasks)
    return list(derived)


def _inherit_adapter_base_task(framework, tasks, row):
    """A PAIRABLE PEFT adapter serves whatever its BASE serves.

    An adapter is a delta: ``qwen3.5-test-stage1-lora`` has no task of its own, but
    it names ``base_model_name_or_path`` and the load path (resolve_adapter_pair)
    loads that base and applies the delta — so the base's task IS the row's task.
    Deriving it from the base is a READ, not a guess, which is what lets k61 remove
    the text-generation default without turning every working PEFT row into
    "unclassified". The base's own config.json is consulted; when the base is not
    on disk the row stays unclassified (and the base-present gate above already
    refuses by naming the base to acquire).

    Fires only on an otherwise-unclassified row — an adapter that states its task
    keeps it."""
    if tasks != [NEEDS_CLASSIFICATION_TASK]:
        return tasks
    base = row.get("base_model")
    if not base:
        return tasks
    try:
        from hugpy_engine.peft_adapters import find_base_model_dir
        base_dir = find_base_model_dir(base)
    except Exception:  # noqa: BLE001 — an unresolvable base is simply not here
        return tasks
    if not base_dir:
        return tasks
    inherited = _base_tasks(framework, _enrich_model_type(
        framework, {"dir": base_dir, "hub_id": base}))
    if inherited == [NEEDS_CLASSIFICATION_TASK]:
        return tasks
    logger.info("classify: %s is a delta on %s -> inherits tasks %s",
                row.get("name") or row.get("hub_id"), base, inherited)
    return inherited


def _correct_speech_task(framework, tasks, row):
    """A TTS checkpoint must not advertise text-generation.

    CONTENT/DECLARATION-AUTHORITATIVE OVERRIDE, shaped like ``_correct_video_task``
    and ``_correct_diffusers_task`` and for the identical reason: ``_base_tasks``
    short-circuits on a STORED ``tasks`` value, and the stored value is exactly
    what was wrong here. ``Viral2AI~chatterbox`` carried
    ``tasks: ["text-generation"]`` (a 2026-07-11 reconcile stamp) while its own
    card declared ``pipeline_tag: text-to-speech`` and its dir held a chatterbox
    checkpoint — so the oracle could only bind ``audio.tts`` by NAME MARKER and
    had to report the misclassification as a reason (k98).

    Two independent witnesses, either of which is the model speaking for itself:
      * the dir holds a chatterbox-family checkpoint
        (``model_classifier.is_speech_checkpoint_dir`` — the exact weight files
        the backend's own loader reads), or
      * the row's own hub declaration says ``pipeline_tag: text-to-speech``.

    Fires ONLY over the two "nobody classified this" verdicts (text-generation /
    needs-classification), so a row that already declares a real task of its own
    is never re-labelled."""
    if framework == "gguf" or not ({"text-generation", NEEDS_CLASSIFICATION_TASK}
                                   & set(tasks)):
        return tasks
    if is_speech_checkpoint_dir(row.get("dir")):
        return [TTS_TASK]
    if str(row.get("pipeline_tag") or "").lower() == TTS_TASK:
        return [TTS_TASK]
    return tasks


def _correct_adapter_only(framework, tasks, row):
    """A dir holding only LoRA/adapter weights is an ADAPTER, never a model.

    ``Flux-Uncensored-V2`` is a single ``lora.safetensors`` with no
    ``model_index.json`` and no ``config.json``. It carried ``tasks: null``, null
    defaulted to text-generation, and an image LoRA was offered as an LLM. An
    adapter now says what it is: the non-servable ``adapter`` task (never null,
    never text-generation), so the row stays visible and refuses by NAMING the
    base to apply it to — the same keep-and-explain treatment PEFT adapters and
    pipeline components already get."""
    if framework == "gguf" or row.get("base_model"):
        # A row carrying base_model is a PAIRABLE PEFT adapter: the base-present
        # gate above serves it (base + delta) or refuses by naming the base. Its
        # task is the BASE's task and must survive — this corrector speaks only
        # for the shape nothing can pair.
        return tasks
    d = row.get("dir")
    if not d:
        return tasks
    verdict = classify_model_dir(d)
    if not verdict.get("adapter"):
        return tasks
    if list(tasks) != [ADAPTER_TASK]:
        logger.info("classify: %s holds adapter weights only -> tasks ['%s'] (was %s)",
                    row.get("name") or row.get("hub_id") or d, ADAPTER_TASK, tasks)
    return [ADAPTER_TASK]


def _enrich_model_type(framework, row):
    """Fill ``model_type`` from the model's OWN ``config.json`` when the row lacks it.

    CONTENT-AUTHORITATIVE, the same discipline ``_correct_gguf_vision`` applies to
    the mmproj question: read what is on disk rather than trusting the path or a
    hub tag. Returns the row unchanged (never mutated in place) when there is
    nothing to add or nothing readable.

    WHY (2026-07-27): every Wan row reached ``_base_tasks`` with NO ``model_type``
    and NO ``pipeline_tag``, so it fell through to the "conservative floor" and
    advertised ``["text-generation"]`` — a video diffusion model offering itself as
    a chat model, which then earned a 4-bit bitsandbytes lever, an LLM ctx, an LLM
    VRAM price and eligibility for chat routing. The truth was in the file the
    whole time:

        Wan2.1-T2V-1.3B/config.json
        {"_class_name": "WanModel", "_diffusers_version": "0.30.0",
         "model_type": "t2v", "dim": 1536, "ffn_dim": 8960, "num_layers": 30}

    Only fills a MISSING field — an explicit row value always wins — and only for
    non-gguf rows (a GGUF's type comes from its own header, and ``_base_tasks``
    handles that branch before it ever reaches ``model_type``)."""
    if framework == "gguf" or row.get("model_type"):
        return row
    d = row.get("dir")
    if not d:
        return row
    mt = None
    # (a) the raw repo: config.json carries {"model_type": "t2v"|"vace"|...}
    try:
        with open(os.path.join(d, "config.json"), "r", encoding="utf-8") as fh:
            mt = json.load(fh).get("model_type")
    except Exception:  # noqa: BLE001 — unreadable/absent/malformed → try (b)
        mt = None
    # (b) the DIFFUSERS REPACK has no config.json model_type — it carries
    #     model_index.json {"_class_name": "WanVACEPipeline"|"WanPipeline"|...}.
    #     Both shapes exist side by side on this fleet (Wan2.1-VACE-1.3B vs
    #     Wan2.1-VACE-1.3B-diffusers), so reading only one leaves the other
    #     mis-classified. Map the pipeline class to the same vocabulary.
    if not (isinstance(mt, str) and mt.strip()):
        try:
            with open(os.path.join(d, "model_index.json"), "r", encoding="utf-8") as fh:
                cls = str(json.load(fh).get("_class_name") or "").lower()
        except Exception:  # noqa: BLE001
            cls = ""
        if cls.startswith("wan"):
            # WanVACEPipeline / WanImageToVideoPipeline -> reference/image driven;
            # WanPipeline -> text driven.
            mt = "vace" if ("vace" in cls or "imagetovideo" in cls) else "t2v"
    if not isinstance(mt, str) or not mt.strip():
        return row
    enriched = dict(row)
    enriched["model_type"] = mt.strip().lower()
    return enriched


def _derived_from(hub_id, row):
    """Best-effort HF LINEAGE: the repo this one was quantized / merged /
    finetuned from — the ``base_model:<id>`` tag the Hub synthesises, which is
    the only provenance a GGUF-only repo carries. Cache-only (the metadata
    store's own DB, zero network on a listing); a miss just yields None.

    DELIBERATELY not ``base_model``: that field means "PEFT adapter on this
    base" and trips the base_present gate, so a quant pointed there would read
    as a broken adapter — the same reason civitai_base_model is kept separate
    in the row below. Reuses model_classifier.base_model_of so the parse matches
    the discovery dossier's lineage exactly."""
    base = None
    try:
        from hugpy_engine.model_classifier import base_model_of
        base = base_model_of(row)                       # row's own card/tags, if any
        if not base:
            from hugpy_storage.model_metadata import model_metadata_store
            payload = model_metadata_store.get_repo_info(hub_id)   # cache-only
            if payload:
                base = base_model_of(payload)
    except Exception:                                   # never break the listing
        return None
    return base if (base and base != hub_id) else None


def derive_model_config_row(name, row):
    """One discovery/manifest row -> ModelConfig-ready dict, or (None, reason)."""
    hub_id = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
    if not hub_id or "/" not in hub_id:
        return None, f"unusable hub_id {row.get('hub_id')!r}"

    # PEFT adapter gate: an adapter is a delta on a base model, so it needs
    # base_model_name_or_path AND that base on disk before it can serve.
    #
    # It used to DROP the row (`return None, ...`). That is the silent-
    # unavailability defect the operator named on 2026-07-29: the model is
    # DOWNLOADED and on the models tab's disk, and vanishing it leaves the
    # operator with a directory nothing will explain. The row is KEPT and
    # flagged unserveable with a reason that names the base to acquire — same
    # treatment the no-runner case below already gets. The load path
    # (managers/generate/config.py -> resolve_adapter_pair) refuses with the
    # same text, and SERVES the row outright once the base lands.
    peft_base = row.get("base_model")
    peft_reason = None
    if peft_base and not base_present(peft_base):
        peft_reason = (f"PEFT adapter (base {peft_base!r}) — the adapter is on "
                       f"disk but its base model is NOT in this store, and an "
                       f"adapter cannot be loaded without it. FIX: acquire "
                       f"{peft_base!r} into the store, then this row serves.")

    framework = _derive_framework(name, hub_id, row)
    row = _enrich_model_type(framework, row)   # believe the model when it names itself
    tasks = _derive_tasks(framework, row)
    tasks = _correct_gguf_vision(framework, tasks, row)   # mmproj-authoritative, not pipeline_tag/path
    tasks = _correct_video_task(framework, tasks, row)    # config-authoritative: t2v/vace are NOT chat models
    tasks = _correct_diffusers_task(framework, tasks, row) # model_index-authoritative: the pipeline's own declaration wins
    tasks = _correct_speech_task(framework, tasks, row)    # content-authoritative: a TTS checkpoint is not a chat model
    tasks = _correct_adapter_only(framework, tasks, row)   # content-authoritative: a LoRA dir is a delta, not a model
    tasks = _inherit_adapter_base_task(framework, tasks, row)  # a pairable delta serves what its base serves
    tasks = _correct_pipeline_component(framework, tasks, row)  # path-authoritative: an encoder/vae split is not a chat model
    primary = row.get("primary_task") if row.get("primary_task") in tasks else tasks[0]
    no_runner = [t for t in tasks if (framework, t) not in RUNNER_PAIRS]
    # A pipeline component fails the runner test (no ("gguf","pipeline-component")
    # pair) so it is KEPT-but-unserveable; give the refusal a CAUSE + FIX rather
    # than a bare task name the UI can't explain.
    if no_runner == ["pipeline-component"] and not peft_reason:
        peft_reason = (f"{name}: this GGUF is a sub-component of a diffusion/"
                       f"video pipeline (its file lives under "
                       f"{row.get('filename')!r}), not a standalone chat model. "
                       f"FIX: serve it through its pipeline (Studio/video), or "
                       f"re-register the parent model under a video task.")
    # k61: the two "not a servable model" verdicts refuse with a CAUSE + FIX
    # instead of a bare task name — and, crucially, instead of silently reading
    # as a text-generation model.
    if not peft_reason and tasks == [ADAPTER_TASK]:
        peft_reason = adapter_refusal(name, base_model=peft_base)
    if not peft_reason and tasks == [NEEDS_CLASSIFICATION_TASK]:
        peft_reason = needs_classification_refusal(name)
    on_disk = bool(row.get("dir"))
    if no_runner and not on_disk:
        # Nothing can serve it AND it isn't downloaded → don't surface a dead
        # entry that can neither run nor be opened.
        return None, f"({framework},{no_runner}) has no runner"
    # A downloaded model ALWAYS appears on the models tab, even when no runner
    # can serve its task(s) — it's kept and flagged `serveable: False` (with the
    # offending tasks) so the UI can show it as present-but-unservable and the
    # serve path can refuse cleanly, instead of the model silently vanishing.
    folder = _resolve_folder(row, framework, primary, hub_id,
                             row.get("filename"), row.get("include"))
    return {
        "name": row.get("name") or name, "model_key": name,
        "hub_id": hub_id, "folder": folder,
        "dir": row.get("dir"),
        "framework": framework, "tasks": tasks, "primary_task": primary,
        "base_model": peft_base,                 # None for ordinary models
        # HF LINEAGE (additive, no serving semantics): the quant/merge/finetune
        # parent from the Hub's base_model tag, cache-only. Kept separate from
        # base_model on purpose — see _derived_from.
        "derived_from": _derived_from(hub_id, row),
        "model_max_length": row.get("model_max_length")
            or row.get("tokenizer_model_max_length")
            or row.get("max_position_embeddings") or DEFAULT_MAX_TOKENS_LOCAL,
        "filename": row.get("filename"), "include": row.get("include"),
        "port": row.get("port"), "host": row.get("host"),
        # SERVEABLE = at least one advertised task has a runner (k61). It used to
        # mean "EVERY task has one", which made a partially-servable row read as
        # dead: an inpaint pipeline advertises image-to-image (runner: yes) AND
        # image-inpainting (runner: no), and the old rule hid the img2img it can
        # actually serve. `unserveable_tasks` still names the ones that can't.
        "serveable": bool([t for t in tasks if (framework, t) in RUNNER_PAIRS])
                     and not peft_reason,
        "unserveable_tasks": no_runner or (list(tasks) if peft_reason else []),
        # Classification verdicts, carried by name so every reader (UI picker,
        # serve refusal, re-stamp) sees the same fact the derive saw.
        **({"adapter": True} if tasks == [ADAPTER_TASK] else {}),
        **({"needs_classification": True}
           if tasks == [NEEDS_CLASSIFICATION_TASK] else {}),
        # Why it can't serve, in words, when it can't. Named so the UI and the
        # serve refusal can both quote a CAUSE + FIX instead of leaving the
        # caller with a bare boolean (or, before this, "Unrecognized model").
        **({"unserveable_reason": peft_reason} if peft_reason else {}),
        # Provenance DECORATION (civitai sidecar via the comfy sweep) — this
        # derive rebuilds rows with a fixed schema, so decoration must be
        # passed through by name or it silently dies here. No serving
        # semantics; civitai_base_model deliberately NOT base_model (that
        # field means PEFT adapter and trips the base_present gate above).
        **{k: row[k] for k in ("display_name", "civitai_id",
                               "civitai_version_id", "civitai_base_model")
           if row.get(k) is not None},
        # LICENSE — the row's own declaration (discovery reads it off the HF
        # card). Dropped here, it was unreadable downstream, and the oracle
        # catalog's license gate had to report "not recorded on the registry
        # row" for a model whose card plainly says MIT (k98). Passed through by
        # NAME like the provenance decoration above; absent stays absent, which
        # is still reported as UNKNOWN and never assumed permissive.
        **({"license": row["license"]} if row.get("license") else {}),
    }, None

def _absorb_disk(staple, disc):
    """Disk facts from a discovered row override a staple's hand-written guesses."""
    for k in ("dir", "folder", "filename"):
        if disc.get(k):
            staple[k] = disc[k]

def merge_discovery_into_models(discovery, base=None):
    base = base if base is not None else MODELS
    merged = {k: dict(v) for k, v in base.items()}
    hub_to_key = {_clean_repo_id(v.get("hub_id")): k for k, v in base.items()}
    dropped = []
    for name, row in (discovery or {}).items():
        hub = _clean_repo_id(row.get("hub_id") or row.get("folder") or name)
        if name in merged:                       # same key as a staple
            _absorb_disk(merged[name], row); continue
        if hub in hub_to_key:                     # same hub_id as a staple
            _absorb_disk(merged[hub_to_key[hub]], row)
            dropped.append((name, f"merged into staple {hub_to_key[hub]} (same hub_id)"))
            continue
        cfg, why = derive_model_config_row(name, row)
        if cfg is None:
            dropped.append((name, why)); continue
        merged[name] = row if "dir" in row else dict(row)
        merged[name].setdefault("model_key", name)
        hub_to_key[hub] = name
    # derive every merged row (staples now carry absorbed disk facts)
    out, drops2 = {}, []
    for name, row in merged.items():
        cfg, why = derive_model_config_row(name, row)
        (out.__setitem__(name, cfg) if cfg else drops2.append((name, why)))
    return out, dropped + drops2


# ===========================================================================
# ModelConfig assembly — identical validation path as before.
# ===========================================================================
def assess_config(cls, values):
    """Build cls if values can form a valid instance, else False. Never raises."""
    flds = {f.name: f for f in fields(cls)}
    for f in flds.values():
        required = f.default is MISSING and f.default_factory is MISSING
        if required and values.get(f.name) in (None, "", []):
            return False
    out = {}
    for name, f in flds.items():
        if name in values:
            out[name] = values[name]
        elif f.default is not MISSING:
            out[name] = f.default
        else:
            out[name] = f.default_factory()
    # Non-field keys ride through to cls(**...) so ModelConfig's leftover
    # catcher ("extra: everything it wasn't expecting — kept not dropped")
    # actually receives them. Without this, decoration like the civitai
    # provenance keys died HERE, defeating the class's own design.
    for name, v in values.items():
        if name not in flds:
            out[name] = v
    return cls(**out)


def get_model_values(config, dict_return=False, return_dict=False):
    if dict_return or return_dict:
        return config.to_dict()
    return config


def get_assessed_model_config(values, dict_return=False, return_dict=False):
    assessed = assess_config(ModelConfig, values)
    if assessed is False:
        return False
    return get_model_values(assessed, dict_return=dict_return, return_dict=return_dict)


def update_model_config_dict(model_key=None, values=None, dict_obj=None,
                             dict_return=False, return_dict=False, key=None):
    dict_obj = dict_obj if dict_obj is not None else {}
    model_key = model_key or key
    values = dict(values or {})
    values["model_key"] = model_key
    config = get_assessed_model_config(values, dict_return=dict_return, return_dict=return_dict)
    if config is False:
        logger.warning("registry: %s failed ModelConfig assessment, skipped", model_key)
        return dict_obj
    dict_obj[model_key] = config
    return dict_obj


def _load_discovery_report(path=None):
    """Read the discovery report — registry DB first (Phase 1, 2026-09-10:
    HUGPY_REGISTRY_DB=pg; see directions/REGISTRY-DB-INDEX.md), then the
    on-disk JSON. An explicit ``path`` bypasses the DB (the caller asked for a
    specific file). DB off/empty/unreachable -> the JSON path, byte-identical
    to before. Prefer the descriptive report; fall back to the registry-shaped
    manifest. Either shape works."""
    if path is None:
        try:
            from hugpy_engine.model_index import load_discovery as _idx_load
            data = _idx_load()
            if data:
                return data
        except Exception:  # noqa: BLE001 — the registry read path never breaks on the DB
            pass
    for candidate in (path, MODELS_DISCOVERY_PATH, MODELS_DICT_PATH):
        if candidate and os.path.isfile(candidate):
            data = safe_load_from_json(candidate)
            if data:
                return data
    return {}


# ---------------------------------------------------------------------------
# Pruned-models list — user-hidden "ghost" registry rows.
# ---------------------------------------------------------------------------
# A model can show as "missing" forever: a curated staple that was never
# downloaded, or a discovery row whose files were deleted. The UI lets an
# operator prune such a row. Pruning is persisted here (a plain JSON list of
# model_keys beside the discovery report) and applied as a final filter in
# get_models_dict, so it hides the row from EVERY listing path (curated or
# discovered) without mutating the curated MODELS source in code.
def _pruned_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "pruned_models.json")


def _load_pruned():
    p = _pruned_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, list):
            return set(data)
        if isinstance(data, dict):
            return set(data.get("pruned") or [])
    return set()


def _save_pruned(keys):
    safe_dump_to_file(data=sorted(keys), file_path=_pruned_path())


def prune_model(model_key):
    """Hide a not-installed model from the registry listing.

    NON-DESTRUCTIVE: it only adds ``model_key`` to the persisted prune-list, which
    get_models_dict filters out. It does NOT mutate the discovery report or the
    curated MODELS code — so the action is fully reversible via unprune_model
    (and works for a curated staple, which never lives in the report). Returns a
    small status dict."""
    pruned = _load_pruned()
    was_present = model_key in pruned
    pruned.add(model_key)
    _save_pruned(pruned)
    return {
        "pruned": True,
        "model_key": model_key,
        "already_pruned": was_present,
    }


def unprune_model(model_key):
    """Reverse a prune: drop ``model_key`` from the prune-list (the curated staple
    or a re-discovered row then reappears in listings)."""
    pruned = _load_pruned()
    existed = model_key in pruned
    pruned.discard(model_key)
    _save_pruned(pruned)
    return {"unpruned": existed, "model_key": model_key}


# ---------------------------------------------------------------------------
# Media-chat allow-flag — which models the media-intelligence chat dropdown offers.
# ---------------------------------------------------------------------------
# The admin console's Models tab has a "Media" checkbox per model; the media arm's
# chat model picker only offers media-flagged + chat-capable models. DEFAULT: a
# curated staple (a key in MODELS) starts ENABLED, everything else (discovered
# rows) starts disabled — so the picker ships sane without any curation. The store
# records only DEVIATIONS from that default (a staple the operator turned off, or a
# discovered model they turned on), keeping it minimal and self-healing if MODELS
# changes. Shape on disk: {"enabled": [...], "disabled": [...]}.
def _media_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "media_models.json")


def _load_media():
    p = _media_path()
    enabled, disabled = set(), set()
    if os.path.isfile(p):
        data = retry_on_emfile(lambda: safe_load_from_json(p))
        if isinstance(data, dict):
            enabled = set(data.get("enabled") or [])
            disabled = set(data.get("disabled") or [])
        elif isinstance(data, list):       # legacy plain allow-list
            enabled = set(data)
    return {"enabled": enabled, "disabled": disabled}


def _save_media(state):
    safe_dump_to_file(
        data={"enabled": sorted(state["enabled"]), "disabled": sorted(state["disabled"])},
        file_path=_media_path(),
    )


def _default_media(model_key):
    """Default media-flag for a model with no explicit override: curated staples
    (the MODELS defaults) start ON, discovered models start OFF."""
    return model_key in MODELS


def _media_state(ov, model_key):
    """Effective flag for one key against an ALREADY-LOADED override set."""
    if model_key in ov["disabled"]:
        return False
    if model_key in ov["enabled"]:
        return True
    return _default_media(model_key)


def media_state(model_key):
    """Effective media-chat flag for ``model_key`` (override wins over default).

    Single-model read: it loads the override store. Use :func:`media_states` in
    a loop — see why there."""
    return _media_state(_load_media(), model_key)


def media_states(model_keys):
    """Effective media-chat flags for MANY models in ONE store read.

    ``/models`` loops the whole manifest, and calling :func:`media_state` per
    model re-read media_models.json ~107 times — an isfile + open + read EACH,
    every one a virtiofs round-trip on central. That is the same per-model I/O
    the persisted physical state exists to remove, so the listing hoists the
    load out of its loop through here."""
    ov = _load_media()
    return {k: _media_state(ov, k) for k in model_keys}


def set_model_media(model_key, enabled):
    """Set the media-chat flag. Stores only a deviation from the default — toggling
    a model back to its default state clears the override (no dead entries)."""
    enabled = bool(enabled)
    ov = _load_media()
    ov["enabled"].discard(model_key)
    ov["disabled"].discard(model_key)
    if enabled != _default_media(model_key):
        (ov["enabled"] if enabled else ov["disabled"]).add(model_key)
    _save_media(ov)
    return {"model_key": model_key, "media": enabled}


# ---------------------------------------------------------------------------
# Default media-chat model — the ONE model the media-intelligence chat dropdown
# preselects. Single global value (a model_key), not per-model. Persisted beside
# the media allow-flag store (same mechanism: a plain JSON file next to the
# discovery report) so every client agrees deterministically and it survives a
# restart. Shape on disk: {"default": "<model_key>"} or {"default": null}.
# Kept in its OWN file (not merged into media_models.json) so the whole-file
# rewrites of _save_media / _save_media_default never clobber each other.
def _media_default_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "media_default.json")


def media_default_state():
    """The currently-stored default media model_key, or None if unset/cleared."""
    p = _media_default_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, dict):
            val = data.get("default")
            return val or None
        if isinstance(data, str):       # tolerate a bare key on disk
            return data or None
    return None


def set_media_default(model_key, enabled):
    """Set or clear the single default media model (single-default semantics).

    enabled True  -> make ``model_key`` the default, REPLACING any previous one.
    enabled False -> clear the default IFF ``model_key`` is the current default
                     (clearing a non-default key is a no-op, never disturbs the
                     standing default).

    NOTE: setting a default does NOT require the model to be media-enabled — the
    caller may flag a model as default independently of its media allow-flag.
    Returns the resulting state for that key."""
    enabled = bool(enabled)
    current = media_default_state()
    if enabled:
        new_default = model_key
    else:
        new_default = None if current == model_key else current
    safe_dump_to_file(data={"default": new_default}, file_path=_media_default_path())
    return {
        "model_key": model_key,
        "media_default": new_default == model_key,
        "default": new_default,
    }


# ---------------------------------------------------------------------------
# Per-worker KEEP-WARM STAR (boot_prewarm) — the ONE model a given worker keeps
# warm. Mirrors the media_default store above (same mechanism: a plain JSON file
# next to the discovery report), but is a DIFFERENT thing and MUST NOT be
# conflated with it — and is DIFFERENT from 🔒static too. The three levers:
#
#   * media_default (the /media star) = "first in the list + default-selected"
#     — a routing/UI PREFERENCE only. It does NOT load anything.
#   * ⭐ boot_prewarm (this per-worker star) = the operator's KEEP-WARM
#     designation: reconcile keeps it warm every beat, so a star evicted under
#     pressure returns next cycle. Evictable, NOT eviction-protected.
#   * 🔒 static = warm AND eviction-protected (a different, heavier tier).
#   * 📌 pin = routing persistence only, never warms.
#
# Scoped PER WORKER and many-valued: one star per worker id, keyed worker_id ->
# model_key. Shape on disk: {"prewarm": {"<worker_id>": "<model_key>", ...}}.
#
# SEMANTICS (operator RULINGS 2026-07-23):
#   RULING 1 — "the star is the ONLY warm source. nothing warms until starred
#               (or static)."
#   RULING 2 — "star = reconcile-kept-warm" (NOT boot-once): a starred model
#               evicted under load COMES BACK on the next reconcile beat. The
#               star IS the keep-warm designation.
# So the star is loaded once and then RECONCILE keeps it warm every beat — if a
# busy box evicts it, the next reconcile beat reloads it. It is a NORMALLY
# EVICTABLE (FIFO) on-demand resident: NOT eviction-protected (it just doesn't
# STAY cold). This is NOT the static tier (which IS eviction-protected +
# operator-doctrine); do NOT conflate. For "start here AND stay here" with
# eviction protection, the operator promotes the model to 🔒static themselves.
# The identifier stays ``boot_prewarm``/``prewarm`` (rename churn isn't worth it)
# but the meaning is keep-warm, not boot-once.
#
# Kept in its OWN file (worker_boot_prewarm.json) so the whole-file rewrites of
# the other stores (_save_media / _save_media_default) never clobber it.
def _worker_boot_prewarm_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "worker_boot_prewarm.json")


def worker_boot_prewarm_state():
    """The current per-worker keep-warm star map: {worker_id: model_key}.
    Empty dict when unset/cleared. Tolerates a legacy/bare shape (a top-level
    {wid: key} dict written without the wrapper)."""
    p = _worker_boot_prewarm_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, dict):
            prewarm = data.get("prewarm")
            if isinstance(prewarm, dict):
                return {str(k): v for k, v in prewarm.items() if v}
            # tolerate a bare {wid: key} map written without the wrapper
            if "prewarm" not in data:
                return {str(k): v for k, v in data.items() if v}
    return {}


def set_worker_boot_prewarm(worker_id, model_key, enabled):
    """Set or clear a worker's single KEEP-WARM STAR (one-star-per-worker).

    enabled True  -> make ``model_key`` this worker's keep-warm star,
                     REPLACING any previous star for that worker.
    enabled False -> clear this worker's star IFF ``model_key`` is the worker's
                     current star (clearing a non-current key is a no-op, so a
                     stale clear never disturbs the standing star). Passing
                     model_key=None with enabled False clears unconditionally.

    NOTE: the star = the operator's keep-warm designation (operator RULINGS
    2026-07-23; NOT the media_default preference, NOT the 🔒static tier) —
    reconcile keeps it warm every beat, so a star evicted under pressure returns
    next cycle. It does NOT mark the model static, does NOT protect it from
    eviction (evictable, but returns next reconcile beat), and does NOT require
    the model to be present/allocated. Returns the resulting state for that
    worker."""
    enabled = bool(enabled)
    worker_id = str(worker_id)
    state = worker_boot_prewarm_state()
    current = state.get(worker_id)
    if enabled:
        state[worker_id] = model_key
        new_star = model_key
    else:
        if model_key is None or current == model_key:
            state.pop(worker_id, None)
            new_star = None
        else:
            new_star = current
    safe_dump_to_file(data={"prewarm": state}, file_path=_worker_boot_prewarm_path())
    return {
        "worker_id": worker_id,
        "model_key": model_key,
        "boot_prewarm": new_star,
        "starred": new_star is not None,
    }


# ---------------------------------------------------------------------------
# Per-worker WILDCARD flag — the "take all comers" ROUTING opt-in (operator
# doctrine 2026-07-23). Worker designations are a HARD routing scope: they seal
# where designated models CAN route, and an UNDESIGNATED model "gets in where it
# fits in" ONLY on workers that opted in here ("a worker can be designated to
# take all comers while adhering to its allocated model list as priority, or it
# can not be selected as a wildcard and adhere only to its own allocated
# models"). A wildcard box also catches the OVERFLOW of a designated model whose
# home workers are all unavailable (ranking sorts home above wildcard — see
# workers.py); the request busts only when neither can serve.
#
# This flag changes ROUTING ELIGIBILITY only. Once a model is resident, NORMAL
# eviction rules apply — designation affects routing, not eviction ("you don't
# want random evictions simply because you have a verbose model registry" is
# exactly why all-comers is an explicit per-worker opt-in). DEFAULT FALSE for
# every worker: with no flags set, routing is identical to the pre-feature
# fleet (defaults are promises). Not a warm source (that's the ⭐ star), not
# eviction protection (that's 🔒static), not routing persistence (that's 📌pin).
#
# Shape on disk: {"wildcard": {"<worker_id>": true, ...}} — only opted-in
# workers are stored; an absent worker id reads False. Kept in its OWN file
# (worker_wildcard.json) beside the others so the whole-file rewrites of the
# sibling stores (_save_media / _save_media_default / set_worker_boot_prewarm)
# never clobber it. Never persisted onto the worker registry record — routes
# stamp it on the response copy only, exactly like boot_prewarm.
def _worker_wildcard_path():
    return os.path.join(os.path.dirname(MODELS_DISCOVERY_PATH), "worker_wildcard.json")


def worker_wildcard_state():
    """The current per-worker wildcard opt-in map: {worker_id: True}.

    Only opted-in workers appear — an ABSENT key reads False (the default-false
    promise: no flags set == today's sealed-designation routing). Empty dict
    when unset/cleared. Tolerates a legacy/bare shape (a top-level {wid: bool}
    dict written without the wrapper)."""
    p = _worker_wildcard_path()
    if os.path.isfile(p):
        data = safe_load_from_json(p)
        if isinstance(data, dict):
            wc = data.get("wildcard")
            if isinstance(wc, dict):
                return {str(k): True for k, v in wc.items() if v}
            # tolerate a bare {wid: bool} map written without the wrapper
            if "wildcard" not in data:
                return {str(k): True for k, v in data.items() if v}
    return {}


def set_worker_wildcard(worker_id, enabled):
    """Set or clear a worker's WILDCARD ("take all comers") routing opt-in.

    enabled True  -> the worker becomes a wildcard: eligible to catch
                     UNDESIGNATED models and the overflow of designated models
                     whose home workers can't serve (its own designated models
                     stay its priority — ranking, not this store, enforces
                     that). enabled False -> back to the default sealed scope:
                     the worker serves ONLY its own designated / resident /
                     granted models. Clearing an already-absent worker is a
                     no-op (idempotent both ways).

    ROUTING ONLY — never warms anything, never protects anything from
    eviction, never bypasses block/admission/pool/engine/task gates. Returns
    the resulting state for that worker."""
    enabled = bool(enabled)
    worker_id = str(worker_id)
    state = worker_wildcard_state()
    if enabled:
        state[worker_id] = True
    else:
        state.pop(worker_id, None)
    safe_dump_to_file(data={"wildcard": state}, file_path=_worker_wildcard_path())
    return {"worker_id": worker_id, "wildcard": enabled}


def _sweep_comfy_checkpoints(merged):
    """Drop-a-file model registration: every ``*.safetensors``/``*.ckpt`` in
    ``<root>/checkpoints`` becomes a comfy registry row automatically —
    symlinked into the manifest layout (so the file endpoints serve it to
    workers; symlinks everywhere, never copies) and synthesized as
    ``comfy-<slug>`` with text-to-image + image-to-image. Files already
    claimed by an existing comfy row (e.g. a curated staple) are skipped."""
    import re as _re
    from hugpy_platform.constants import DEFAULT_ROOT
    from hugpy_storage.model_paths import route_destination
    from hugpy_storage.hugpy_marker import write_hugpy_marker
    root = os.path.join(DEFAULT_ROOT, "checkpoints")
    if not os.path.isdir(root):
        return {}

    def _ensure_infra(row, fn):
        """Idempotent layout symlink + hugpy.json marker — the marker is what
        flips the row's status to 'installed' (and thus into /v1/models)."""
        try:
            dest_dir = route_destination(row)
            os.makedirs(dest_dir, exist_ok=True)
            link = os.path.join(dest_dir, fn)
            src = os.path.join(root, fn)
            if not os.path.exists(link) and os.path.exists(src):
                os.symlink(src, link)
            if not os.path.exists(os.path.join(dest_dir, "hugpy.json")):
                write_hugpy_marker(
                    dest_dir, hub_id=row.get("hub_id"), name=row.get("name"),
                    framework="comfy", tasks=row.get("tasks"),
                    primary_task=row.get("primary_task"), filename=fn,
                    source="comfy-sweep")
        except OSError as exc:
            logger.warning("comfy sweep: infra for %s failed: %s", fn, exc)

    files = {fn for fn in os.listdir(root)
             if fn.lower().endswith((".safetensors", ".ckpt"))}
    claimed = set()
    # Pass 1: curated/staple comfy rows whose checkpoint is in /checkpoints —
    # make sure THEIR layout link + marker exist too (status -> installed).
    for v in merged.values():
        if isinstance(v, dict) and v.get("framework") == "comfy":
            fn = v.get("filename")
            claimed.add(fn)
            if fn in files:
                _ensure_infra(v, fn)
    def _civitai_sidecar(fn):
        """Read the ``<file>.civitai.json`` provenance stamp written by
        /civitai/download, if any. Local read only; corrupt/absent -> None.
        Files predating the stamp have no sidecar and stay EXACTLY as today
        (no network for unstamped files, ever)."""
        try:
            with open(os.path.join(root, fn + ".civitai.json")) as fh:
                sc = json.load(fh)
            return sc if isinstance(sc, dict) else None
        except Exception:  # noqa: BLE001 — decoration only, never a gate
            return None

    # Pass 2: unclaimed files synthesize their own rows.
    rows = {}
    for fn in sorted(files - claimed):
        stem = _re.sub(r"[^A-Za-z0-9]+", "-", fn.rsplit(".", 1)[0]).strip("-").lower()
        key = f"comfy-{stem}"
        if key in merged:
            continue
        hub = f"comfy/{stem}"
        row = {"model_max_length": 77, "include": None, "name": key,
               "framework": "comfy", "hub_id": hub, "filename": fn,
               "folder": hub, "tasks": ["text-to-image", "image-to-image"],
               "primary_task": "text-to-image", "port": None}
        # Civitai provenance decoration (sidecar-gated): keys stay identity
        # (name/key = comfy-<stem> unchanged), display/base_model are
        # decoration. model_max_length stays 77 deliberately — no per-base
        # token-length mapping exists in the codebase to reuse, and inventing
        # SDXL dual-encoder handling here is out of scope; base_model rides
        # the row so a future consumer CAN branch on it.
        sidecar = _civitai_sidecar(fn)
        if sidecar:
            if sidecar.get("name"):
                row["display_name"] = sidecar["name"]
            if sidecar.get("civitai_id") is not None:
                row["civitai_id"] = sidecar["civitai_id"]
            if sidecar.get("version_id") is not None:
                row["civitai_version_id"] = sidecar["version_id"]
            if sidecar.get("base_model"):
                # NOT row["base_model"]: that field means "PEFT adapter on
                # <base>" and a truthy value trips the adapter gate
                # (base_present at ~line 323) — "SD 1.5" isn't on disk as a
                # transformers base, so the row would be DROPPED from the
                # registry. civitai_base_model is decoration riding
                # ModelConfig.extra, no serving semantics.
                row["civitai_base_model"] = sidecar["base_model"]
            # Warm the central civitai_meta table (fetch-once) — ONLY for
            # stamped files carrying an id, and offline-safe: any failure
            # (no network on a worker, DNS, store trouble) degrades to the
            # unenriched row above.
            try:
                from hugpy_storage.model_metadata import fetch_civitai_meta
                fetch_civitai_meta(stem,
                                   civitai_id=sidecar.get("civitai_id"),
                                   version_id=sidecar.get("version_id"),
                                   timeout=5.0)
            except Exception:  # noqa: BLE001 — enrichment never breaks a sweep
                pass
        _ensure_infra(row, fn)
        rows[key] = row
    return rows


def get_models_dict(models_dict_path=None, dict_return=False, return_dict=False,
                    discovery=None):
    """Build the registry: MODELS + discovery (test downloads).

    discovery=None -> read the report on disk. Pass a dict to merge an
    in-memory discovery result (e.g. straight from a fresh walk).

    Operator-pruned model_keys (pruned_models.json) are filtered out as a final
    step so a hidden "ghost" row never shows in any listing."""
    dict_return = dict_return or return_dict
    # Last-known cache (Keeper 2026-07-09): the default read path — model_key
    # membership guards and manifest lookups hit on every worker heartbeat —
    # must NOT rebuild the whole registry (disk report + comfy sweep + dedup
    # merge). That rebuild-per-call pegged the API workers and flooded the logs.
    # Serve the cached MODEL_REGISTRY(_DICT), which refresh_registry() advances
    # IN PLACE on real changes (download complete, discover POST) — so even a
    # caller that reads right after a refresh sees fresh data. Only an explicit
    # discovery/path (a fresh walk) forces a rebuild. During the import-time
    # build the globals aren't bound yet -> globals().get() is None -> build.
    if discovery is None and models_dict_path is None:
        cached = globals().get("MODEL_REGISTRY_DICT" if dict_return else "MODEL_REGISTRY")
        if cached:
            return cached
    report = discovery if discovery is not None else _load_discovery_report(models_dict_path)

    # Comfy checkpoint sweep FIRST, so its rows join the dedupe BASE of the
    # merge. The sweep's own layout dirs (models/misc/text-to-image/comfy/…)
    # are walked by discovery too; when sweep rows were appended AFTER the
    # merge, that discovered copy (hub comfy/<stem>) had nothing to merge
    # into and survived as a SECOND row — with default framework/task when
    # enrichment came up empty (the 2026-07-05 sd-turbo/sd-xl-turbo phantom
    # rows). In the base, same-hub discovery merges into the sweep row.
    base = {k: dict(v) for k, v in MODELS.items()}
    try:
        for k, v in _sweep_comfy_checkpoints(base).items():
            base.setdefault(k, v)
    except Exception as exc:  # noqa: BLE001 — the sweep must never break the registry
        logger.warning("comfy checkpoint sweep failed: %s", exc)

    merged, dropped = merge_discovery_into_models(report, base=base)

    # DEBUG, not INFO: dedup merges are routine registry bookkeeping re-emitted
    # on every 30s reconcile rebuild (~2k lines/pass across workers). At INFO
    # this floods the journal + host rsyslog and pegs vCPUs. Keeper 2026-07-09.
    for model_key, why in dropped:
        logger.debug("registry: dropped %s (%s)", model_key, why)

    pruned = _load_pruned()
    nudict = {}
    for model_key, values in merged.items():
        if model_key in pruned:
            continue
        nudict = update_model_config_dict(
            model_key=model_key, values=values, dict_obj=nudict, dict_return=dict_return
        )
    return nudict


# ===========================================================================
# Registry — built at import from MODELS + existing discovery report.
# ===========================================================================
MODEL_REGISTRY: Dict[str, ModelConfig] = get_models_dict()
MODEL_REGISTRY_DICT: Dict[str, dict] = get_models_dict(dict_return=True)


def get_model_registry(dict_return=False, return_dict=False):
    dict_return = dict_return or return_dict
    return MODEL_REGISTRY_DICT if dict_return else MODEL_REGISTRY


def refresh_registry(run_discovery=True):
    """Re-walk the model tree and rebuild MODEL_REGISTRY in place. Call this on
    hugpy startup. run_discovery=False just re-reads the existing report.

    Late import of discover_models avoids a circular import at module load.

    The update is IN PLACE (update-then-prune, never rebind): other modules
    import these dicts by reference (`from ... import MODEL_REGISTRY`), so
    rebinding the names here would leave every importer holding the stale
    dict. Update-then-prune also means a concurrent reader on the threaded
    server never catches the dict momentarily empty."""
    report = None
    if run_discovery:
        try:
            from hugpy_engine.apis.get_module import discover_models
            report = discover_models(save_json=True, verbose=False, use_hub=True)
        except Exception as exc:
            logger.warning("refresh_registry: discovery walk failed (%s); "
                           "falling back to on-disk report", exc)
    fresh = get_models_dict(discovery=report)
    fresh_dict = get_models_dict(dict_return=True, discovery=report)
    MODEL_REGISTRY.update(fresh)
    for stale in [k for k in MODEL_REGISTRY if k not in fresh]:
        MODEL_REGISTRY.pop(stale, None)
    MODEL_REGISTRY_DICT.update(fresh_dict)
    for stale in [k for k in MODEL_REGISTRY_DICT if k not in fresh_dict]:
        MODEL_REGISTRY_DICT.pop(stale, None)
    try:
        from hugpy_engine.config.models.models_default import refresh_task_registries
        refresh_task_registries()
    except Exception as exc:
        logger.warning("refresh_registry: task registry refresh failed (%s)", exc)
    # Self-heal serve overrides orphaned by collision-qualification of keys
    # (bare `name` -> `owner~name`). Runs at discovery so a re-walk re-homes them.
    try:
        from hugpy_engine.serve.overrides import migrate_overrides
        moved = migrate_overrides(fresh_dict)
        if moved:
            logger.info("refresh_registry: migrated %d orphaned serve override(s): %s",
                        len(moved), moved)
    except Exception as exc:
        logger.warning("refresh_registry: serve-override migration skipped (%s)", exc)
    # THE chokepoint for "the catalog/store moved": every caller that changes
    # what is on disk lands here (download completion, the discovery sweep,
    # reconcile's _persist_registry). Both stores of derived physical state are
    # told from one place rather than hoping every future call site remembers.
    # Guarded and lazy — comms is stdlib-only and imports nothing back, but a
    # registry refresh must never fail over a cache.
    #
    # The PERSISTED physical state (comms/model_physical.py) is dropped at a
    # granularity that matches what actually changed — over-dropping only costs
    # a re-derive, but it costs the WHOLE table's worth, and the point of
    # persisting was to stop paying that:
    #
    #   run_discovery=True  -> a real store walk just happened, so anything on
    #     disk may have moved: drop EVERYTHING. This is /models/discover, which
    #     immediately rebuilds the table in its own background thread, so the
    #     next listing is warm as well as correct.
    #   run_discovery=False -> only the on-disk report was re-read; no fresh
    #     evidence about the store. Drop exactly the rows the registry no longer
    #     backs or whose ROUTING IDENTITY moved (a re-keyed / re-routed model —
    #     its old record must never answer for the new row). The events that DID
    #     change the store on this path carry their own targeted invalidation:
    #     download completion names its model_key, and an applied reconcile
    #     drops the whole table from _persist_registry / the route.
    try:
        from hugpy_storage.model_physical import forget_all_physical, reconcile_physical_identities
        if run_discovery:
            forget_all_physical("refresh_registry:discovery")
        else:
            reconcile_physical_identities(MODEL_REGISTRY_DICT,
                                          "refresh_registry")
    except Exception:  # noqa: BLE001
        logger.debug("refresh_registry: physical-state invalidation skipped",
                     exc_info=True)
    # The central-holdings memo answers a different question ("can central
    # provide this model to a worker?") off the same presence facts, and can
    # only express a whole-memo drop, so it is flushed unconditionally.
    try:
        from hugpy_storage.model_status_cache import invalidate_model_status
        invalidate_model_status("refresh_registry")
    except Exception:  # noqa: BLE001
        logger.debug("refresh_registry: model-status cache invalidation skipped",
                     exc_info=True)
    return MODEL_REGISTRY
