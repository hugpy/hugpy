import os
from typing import Dict
from abstract_essentials import make_list
from hugpy_engine.categories import MEDIA_DEFAULTS, TASK_DEFAULTS
from hugpy_engine.config.models.models_config import MODEL_REGISTRY
from hugpy_engine.schemas.model_schemas import ModelConfig
from hugpy_platform.constants import (
    DEFAULT_AGENT_BRAIN,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_MODEL,
    HUGPY_MARKER,
)
from hugpy_storage.model_paths import route_destination

def get_context_tokens():
    context_tokens = {}
    for module_key,values in MODEL_REGISTRY.items():
        context_tokens[module_key] = values.model_max_length
    return context_tokens

DEFAULT_CONTEXT_TOKENS_BY_MODEL: dict[str, int] = get_context_tokens()

def default_context_tokens_for_model(model_key: str) -> int:
    return DEFAULT_CONTEXT_TOKENS_BY_MODEL.get(model_key, 8192)


def get_models_dict_by_tasks(tasks=None):
    tasks = make_list(tasks or [])
    models = {}
    for module_key, values in MODEL_REGISTRY.items():
        # was: if values.task in tasks
        if any(t in tasks for t in values.tasks):
            models[module_key] = values
    return models
def get_models_dict_by_names(names=None):
    names = make_list(names or [])
    models = {}
    for module_key,values in MODEL_REGISTRY.items():
        for name in names:
            if name in values.name:
                models[module_key] = values
                break
    return models


def _on_disk(cfg: ModelConfig) -> bool:
    """Weights present at the model's route_destination (marker or any files).
    Same signal /v1/models uses to report status=installed."""
    try:
        dest = route_destination(cfg.to_dict())
    except Exception:
        return False
    if os.path.isfile(os.path.join(dest, HUGPY_MARKER)):
        return True
    try:
        return os.path.isdir(dest) and bool(os.listdir(dest))
    except OSError:
        return False


def _reconcile_default(configured: str, registry: Dict[str, ModelConfig],
                       prefer_task: str | None = None) -> str:
    """Keep the configured default when the registry has it; otherwise stand in
    a model whose weights are already on disk (preferring one that serves
    prefer_task, then key order for determinism). Never picks a model that
    would need a download — when nothing usable is installed, the configured
    name is returned unchanged so resolve_model_key's 'not in MODEL_REGISTRY'
    error stays accurate instead of surprise-downloading a stand-in.

    Operator BLOCK (central pool primitive): a blocked model is never resolved by
    this fallback ladder — the configured default/agent-brain is SKIPPED when
    blocked (with an honest log line) and a stand-in installed, non-blocked model
    is chosen instead. Blocked candidates are excluded from the stand-in set too.
    Guarded: any blocklist failure degrades to "nothing blocked" (fail-open)."""
    if not registry:
        return configured

    # Guarded read of the operator block set (blocklist lives in stdlib-only
    # comms; a failure here must never break config import).
    try:
        from hugpy_engine.placement import get_blocklist
        blocked = set(get_blocklist().blocked_keys())
    except Exception:  # noqa: BLE001
        blocked = set()

    if configured in registry and configured not in blocked:
        return configured
    if configured in blocked:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "default/agent-brain candidate %s is BLOCKED from the serving pool "
            "by the operator — skipping it and standing in an installed model",
            configured)

    installed = {k: cfg for k, cfg in registry.items()
                 if k not in blocked and _on_disk(cfg)}
    if prefer_task:
        preferred = {k: cfg for k, cfg in installed.items() if prefer_task in cfg.tasks}
        installed = preferred or installed
    if not installed:
        return configured
    return min(installed)


_CONFIGURED_DEFAULTS = {
    "chat": DEFAULT_CHAT_MODEL,
    "vision": DEFAULT_VISION_MODEL,
    "whisper": DEFAULT_WHISPER_MODEL,
    "embed": DEFAULT_EMBED_MODEL,
}

CHAT_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["text-generation","text-generation-inference","text2text-generation"])
DEFAULT_CHAT_MODEL = _reconcile_default(DEFAULT_CHAT_MODEL, CHAT_MODELS_REGISTRY,
                                        prefer_task="text-generation")
# (The bare alias `DEFAULT_MODEL = DEFAULT_CHAT_MODEL` was DELETED 2026-07-17:
# zero in-tree consumers, and the generic name collides with hugpy_agent's
# brain and the todo-keeper's model — same-named keys resolve odd in a shared
# .env. Say DEFAULT_CHAT_MODEL when you mean the chat default.)

# The AGENT BRAIN (see constants.py — dedicated default for agent nodes/loops,
# operator ask 2026-07-17). Reconciled like every other kind default: keep the
# configured brain when the registry knows it, else stand in an installed chat
# model rather than promising a download (defaults are promises).
DEFAULT_AGENT_BRAIN = _reconcile_default(DEFAULT_AGENT_BRAIN, CHAT_MODELS_REGISTRY,
                                         prefer_task="text-generation")


VISION_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["image-text-to-text","text-to-image"])
DEFAULT_VISION_MODEL = _reconcile_default(DEFAULT_VISION_MODEL, VISION_MODELS_REGISTRY,
                                          prefer_task="image-text-to-text")


WHISPER_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["automatic-speech-recognition","speech-recognition"])
DEFAULT_WHISPER_MODEL = _reconcile_default(DEFAULT_WHISPER_MODEL, WHISPER_MODELS_REGISTRY,
                                           prefer_task="automatic-speech-recognition")


EMBED_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["feature-extraction", "sentence-similarity","sentence-transformers"])
DEFAULT_EMBED_MODEL = _reconcile_default(DEFAULT_EMBED_MODEL, EMBED_MODELS_REGISTRY,
                                         prefer_task="feature-extraction")


# TASK_DEFAULTS / MEDIA_DEFAULTS were built in categories.py from the
# *configured* constants at import time. When a configured default isn't in the
# registry (curated default never downloaded), the entries above get a registry
# stand-in — but the dicts still hold the dangling name, and resolve_model_key
# raises on any task-only request that hits one. Heal them IN PLACE: other
# modules import these dicts by reference, so rebinding would not reach them.
_RECONCILED_DEFAULTS = {
    _CONFIGURED_DEFAULTS["chat"]: DEFAULT_CHAT_MODEL,
    _CONFIGURED_DEFAULTS["vision"]: DEFAULT_VISION_MODEL,
    _CONFIGURED_DEFAULTS["whisper"]: DEFAULT_WHISPER_MODEL,
    _CONFIGURED_DEFAULTS["embed"]: DEFAULT_EMBED_MODEL,
}
for _defaults, _keys_are_tasks in ((TASK_DEFAULTS, True), (MEDIA_DEFAULTS, False)):
    for _key, _model in list(_defaults.items()):
        replacement = _RECONCILED_DEFAULTS.get(_model)
        if not replacement or replacement == _model:
            continue
        if _keys_are_tasks and _key not in MODEL_REGISTRY[replacement].tasks:
            continue  # stand-in serves the group but not this exact task
        _defaults[_key] = replacement


# The task-filtered registries above are import-time SNAPSHOTS of MODEL_REGISTRY.
# A model learned at runtime (e.g. pulled/registered from central after startup)
# lands in MODEL_REGISTRY but NOT in these derived dicts — so a vision/whisper/
# embed model resolved after boot raises "Unknown <kind> model key ...
# Available: []". refresh_task_registries() rebuilds them from the current
# MODEL_REGISTRY. It updates IN PLACE (clear+update), which is required because
# other modules import these dicts by reference — rebinding the names here would
# not reach them.
_TASK_REGISTRY_GROUPS = (
    (CHAT_MODELS_REGISTRY,    ["text-generation", "text-generation-inference", "text2text-generation"]),
    (VISION_MODELS_REGISTRY,  ["image-text-to-text", "text-to-image"]),
    (WHISPER_MODELS_REGISTRY, ["automatic-speech-recognition", "speech-recognition"]),
    (EMBED_MODELS_REGISTRY,   ["feature-extraction", "sentence-similarity", "sentence-transformers"]),
)


def refresh_task_registries() -> None:
    """Rebuild the task-filtered registries in place from the live MODEL_REGISTRY.

    Update-then-prune (never clear()): the worker serves on a threaded server, so
    a concurrent reader must never catch the dict momentarily empty. update()
    adds/refreshes current models first, then stale keys are popped one by one —
    so any valid key (notably the one being resolved right now) stays present
    throughout.
    """
    for registry, tasks in _TASK_REGISTRY_GROUPS:
        fresh = get_models_dict_by_tasks(tasks=tasks)
        registry.update(fresh)
        for stale in [k for k in registry if k not in fresh]:
            registry.pop(stale, None)
