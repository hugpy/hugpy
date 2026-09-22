"""Model discovery and selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from .backend import get_backend
from .contracts import ModelDescriptor


def _descriptor(key: str, row: Mapping[str, Any]) -> ModelDescriptor:
    tasks = row.get("tasks") or ()
    if isinstance(tasks, str):
        tasks = (tasks,)
    return ModelDescriptor(
        key=key,
        name=str(row.get("name") or key),
        hub_id=row.get("hub_id"),
        framework=row.get("framework"),
        tasks=tuple(str(task) for task in tasks),
        primary_task=row.get("primary_task") or row.get("task"),
        context_length=row.get("model_max_length"),
        status=row.get("status"),
        raw=dict(row),
    )


def text_chat_task(config, requested_task=None, has_media: bool = False):
    """Choose the text runner for a text-only request to a multi-task model."""
    if requested_task or has_media or config is None:
        return requested_task
    if isinstance(config, Mapping):
        tasks = config.get("tasks") or ()
        primary = config.get("primary_task")
    else:
        tasks = getattr(config, "tasks", None) or ()
        primary = getattr(config, "primary_task", None)
    if primary != "text-generation" and "text-generation" in tasks:
        return "text-generation"
    return None


def list_models(
    *, task: Optional[str] = None, framework: Optional[str] = None
) -> tuple[ModelDescriptor, ...]:
    """Return the current discovered catalog, optionally filtered."""
    found = []
    for key, row in get_backend().catalog_rows().items():
        descriptor = _descriptor(key, row)
        if task is not None and task not in descriptor.tasks:
            continue
        if framework is not None and descriptor.framework != framework:
            continue
        found.append(descriptor)
    return tuple(sorted(found, key=lambda item: item.key.lower()))


def resolve_model(
    model_key: Optional[str] = None,
    *,
    task: Optional[str] = None,
    file: Optional[str] = None,
    media_type: Optional[str] = None,
) -> str:
    """Resolve a requested, task-default, media-default, or chat-default model."""
    return get_backend().resolve_model(
        model_key=model_key,
        task=task,
        file=file,
        media_type=media_type,
    )


def get_model(model_key: str) -> ModelDescriptor:
    """Resolve ``model_key`` and return its live catalog description."""
    key = resolve_model(model_key)
    rows = get_backend().catalog_rows()
    try:
        row = rows[key]
    except KeyError:
        raise KeyError(f"resolved model {key!r} is absent from the live catalog") from None
    return _descriptor(key, row)


def refresh_models(*, discover: bool = True) -> tuple[ModelDescriptor, ...]:
    """Refresh disk discovery and return the resulting live catalog."""
    get_backend().refresh_models(discover=discover)
    return list_models()


__all__ = [
    "ModelDescriptor",
    "get_model",
    "list_models",
    "refresh_models",
    "resolve_model",
    "text_chat_task",
]
