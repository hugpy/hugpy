"""Explicit, lazy re-exports of the engine/storage helpers the worker uses.

This replaces the monolith's ``from abstract_hugpy_dev.imports import *``
hub. Every name resolves to its true defining module in an *allowed*
dependency (``hugpy_engine``, ``hugpy_storage``) and is imported on first
use, so ``import hugpy_fleet.worker.agent`` stays cheap and works on a box
without the model catalog materialised.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "execute_prompt",
    "execute_chat_stream",
    "describe",
    "get_model_config",
    "get_model_path",
    "get_models_dict",
    "route_destination",
    "runner_for",
    "models_config",
]


def execute_prompt(*args: Any, **kwargs: Any):
    from hugpy_engine.dispatch.dispatch import execute_prompt as _f
    return _f(*args, **kwargs)


def execute_chat_stream(*args: Any, **kwargs: Any):
    """Returns the engine's async generator (not awaited here)."""
    from hugpy_engine.dispatch.dispatch import execute_chat_stream as _f
    return _f(*args, **kwargs)


def describe() -> dict:
    from hugpy_engine.spill import describe as _f
    return _f()


def get_model_config(*args: Any, **kwargs: Any):
    from hugpy_engine.config.main import get_model_config as _f
    return _f(*args, **kwargs)


def get_model_path(*args: Any, **kwargs: Any):
    from hugpy_engine.config.main import get_model_path as _f
    return _f(*args, **kwargs)


def get_models_dict(*args: Any, **kwargs: Any):
    from hugpy_engine.config.models.models_config import get_models_dict as _f
    return _f(*args, **kwargs)


def runner_for(*args: Any, **kwargs: Any):
    from hugpy_engine.dispatch.dispatch import runner_for as _f
    return _f(*args, **kwargs)


def route_destination(*args: Any, **kwargs: Any):
    from hugpy_storage.model_paths import route_destination as _f
    return _f(*args, **kwargs)


def __getattr__(name: str):
    # ``from hugpy_fleet.worker.imports import models_config`` — the module
    # itself (MODEL_REGISTRY / refresh_registry), resolved lazily.
    if name == "models_config":
        import importlib
        return importlib.import_module("hugpy_engine.config.models.models_config")
    raise AttributeError(name)
