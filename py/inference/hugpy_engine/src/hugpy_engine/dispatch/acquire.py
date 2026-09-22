# abstract_hugpy/managers/dispatch/acquire.py
import os, json, logging
from abstract_essentials import safe_load_from_json
from hugpy_engine.apis.get_module import build_resolver_chain, enrich
from hugpy_engine.categories import HF_TASK_TO_TASKS
from hugpy_engine.config.models.models_config import MODELS
from hugpy_platform.constants import MODELS_DICT_PATH
from hugpy_platform.utils import extract_gguf_filename, get_guffs_in_dir, infer_framework

logger = logging.getLogger(__name__)

def _servable():
    # Lazy: importing FRAMEWORK_RUNNERS drags in torch and runs validate_registry() at
    # import time. Keeping it out of module scope lets a downloader process
    # import `acquire` without loading the inference stack.
    from hugpy_engine.resolvers.categories.frameworks import FRAMEWORK_RUNNERS
    return set(FRAMEWORK_RUNNERS)


def acquire(hub_id, path, *, gguf_filename=None):
    key = hub_id.split("/")[-1]
    if key in MODELS:                                   # Option A: never shadow curated
        raise ValueError(f"{key} is curated; edit models_dict.py, not the overlay")

    meta, _ = enrich(path, hub_id, build_resolver_chain(use_hub=True))

    framework = infer_framework(path)                   # disk truth, returns None if no weights
    if framework is None:
        raise ValueError(f"{hub_id}: no recognizable weights on disk at {path}")

    tasks = list(HF_TASK_TO_TASKS.get(meta.pipeline_tag, []))
    if not tasks:
        raise ValueError(f"{hub_id}: can't derive task from pipeline_tag="
                         f"{meta.pipeline_tag!r}")
    primary_task = tasks[0]

    if (framework, primary_task) not in _servable():    # no runner — register anyway
        # A downloaded model is always made available (it shows on the models
        # tab, flagged unserveable by the registry). It just can't be served
        # until a runner for (framework, primary_task) is registered — the serve
        # path refuses cleanly. Do NOT block the download/registration here.
        logger.warning("acquire: %s (%s,%s) has no runner — registering as "
                       "unserveable (download kept, won't serve)",
                       hub_id, framework, primary_task)

    if framework == "gguf" and not gguf_filename:  # picks the canonical shard
        gguf_filename = extract_gguf_filename(get_guffs_in_dir(path), path)

    overlay = safe_load_from_json(MODELS_DICT_PATH) or {}
    overlay[key] = {
        "name": key, "hub_id": hub_id, "folder": hub_id,
        "framework": framework, "tasks": tasks, "primary_task": primary_task,
        "filename": gguf_filename, "include": None, "port": None,
        "model_max_length": meta.tokenizer_model_max_length
                            or meta.max_position_embeddings or 32768,
    }
    with open(MODELS_DICT_PATH, "w") as f:
        json.dump(overlay, f, indent=2)
    return key
