"""Standalone Hugpy inference boundary.

Prompt/messages -> model discovery and resolution -> RAM/VRAM allocation ->
engine/runner selection -> generation -> streamed events or a completed reply.

The package imports CPU-only: llama_cpp / torch / transformers / peft load
lazily inside the runner that needs them (``pip install hugpy-engine[gguf]``,
``[transformers]``, ``[finetune]``). Engine-owned contracts (task capability
map, catalog vocabulary, manifest helpers, wire schemas, the placement and
task-plugin seams) are exposed lazily via ``__getattr__`` so ``import
hugpy_engine`` stays light.
"""

from __future__ import annotations

import importlib

try:
    from importlib.metadata import version as _package_version

    __version__ = _package_version("hugpy-engine")
except Exception:
    __version__ = "0.0.0+unknown"  # source tree without metadata

from .allocation import (
    allocation_spill,
    choose_allocation,
    feasible_allocations,
    plan_admission,
    resource_status,
)
from .backend import InferenceBackend, backend_scope, configure_backend, get_backend
from .catalog import get_model, list_models, refresh_models, resolve_model, text_chat_task
from .contracts import ModelDescriptor, QueryError, QueryResult
from .engines import (
    build_native_engine,
    evict_llama_runner,
    install_native_engine,
    llama_runner,
    loaded_llama_runners,
    native_engine_binaries,
    native_engine_status,
)
from .query import query, query_result, query_result_sync, query_sync, stream_query
from .runtime import (
    clear_models,
    evict_model,
    execute_request,
    loaded_models,
    loading_models,
    resolve_request,
    runner_for,
    stream_chat_request,
    stream_request,
    supported_tasks,
)

# Engine-owned contracts and seams, resolved on first attribute access
# (PEP 562) so the package import never touches discovery or pydantic.
_LAZY = {
    # task capability map (consumed by fleet, oracle, server)
    "TASK_DEPS": ("hugpy_engine.task_deps", "TASK_DEPS"),
    "task_capabilities": ("hugpy_engine.task_deps", "task_capabilities"),
    # catalog vocabulary
    "HF_TASK_TO_TASKS": ("hugpy_engine.categories", "HF_TASK_TO_TASKS"),
    "RUNNER_PAIRS": ("hugpy_engine.categories", "RUNNER_PAIRS"),
    "TASK_DEFAULTS": ("hugpy_engine.categories", "TASK_DEFAULTS"),
    "MEDIA_DEFAULTS": ("hugpy_engine.categories", "MEDIA_DEFAULTS"),
    # manifest helpers
    "load_manifest": ("hugpy_engine.manifest", "load_manifest"),
    "load_manifest_or_empty": ("hugpy_engine.manifest", "load_manifest_or_empty"),
    "save_manifest": ("hugpy_engine.manifest", "save_manifest"),
    "key_for_hub_id": ("hugpy_engine.manifest", "key_for_hub_id"),
    "upsert_model": ("hugpy_engine.manifest", "upsert_model"),
    # seams
    "placement": ("hugpy_engine.placement", None),
    "tasks": ("hugpy_engine.tasks", None),
    "register_task": ("hugpy_engine.tasks", "register_task"),
    "runner_for_task": ("hugpy_engine.tasks", "runner_for_task"),
    "request_builder_for_task": ("hugpy_engine.tasks", "request_builder_for_task"),
    "catalog_bridge": ("hugpy_engine.catalog_bridge", None),
    "LocalBackend": ("hugpy_engine.backends.local", "LocalBackend"),
    # wire schema modules (pydantic bodies shared with the HTTP layer)
    "wire": ("hugpy_engine.wire", None),
    "schemas": ("hugpy_engine.schemas", None),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "HF_TASK_TO_TASKS",
    "InferenceBackend",
    "LocalBackend",
    "MEDIA_DEFAULTS",
    "ModelDescriptor",
    "QueryError",
    "QueryResult",
    "RUNNER_PAIRS",
    "TASK_DEFAULTS",
    "TASK_DEPS",
    "allocation_spill",
    "backend_scope",
    "build_native_engine",
    "catalog_bridge",
    "choose_allocation",
    "clear_models",
    "configure_backend",
    "evict_llama_runner",
    "evict_model",
    "execute_request",
    "feasible_allocations",
    "get_backend",
    "get_model",
    "install_native_engine",
    "key_for_hub_id",
    "list_models",
    "llama_runner",
    "load_manifest",
    "load_manifest_or_empty",
    "loaded_llama_runners",
    "loaded_models",
    "loading_models",
    "native_engine_binaries",
    "native_engine_status",
    "placement",
    "plan_admission",
    "query",
    "query_result",
    "query_result_sync",
    "query_sync",
    "refresh_models",
    "register_task",
    "request_builder_for_task",
    "resolve_model",
    "resolve_request",
    "resource_status",
    "runner_for",
    "runner_for_task",
    "save_manifest",
    "schemas",
    "stream_chat_request",
    "stream_query",
    "stream_request",
    "supported_tasks",
    "task_capabilities",
    "tasks",
    "text_chat_task",
    "upsert_model",
    "wire",
]
