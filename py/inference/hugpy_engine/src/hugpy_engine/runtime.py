"""Request resolution, execution, and runner lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from .backend import get_backend


def resolve_request(request: Mapping[str, Any]):
    return get_backend().resolve_request(request)


def runner_for(model_key: Optional[str] = None, *, task: Optional[str] = None):
    return get_backend().runner_for(model_key, task=task)


def execute_request(*args: Any, **kwargs: Any):
    return get_backend().execute_request(*args, **kwargs)


async def stream_request(*args: Any, cancel_event=None, **kwargs: Any):
    _it = get_backend().stream_request(*args, cancel_event=cancel_event, **kwargs)
    try:
        async for event in _it:
            yield event
    finally:
        # Cascade a client disconnect into the backend stream (releases the
        # relay/runner/llama-server stream beneath it) instead of leaving it to GC.
        _ac = getattr(_it, "aclose", None)
        if _ac is not None:
            try:
                await _ac()
            except Exception:  # noqa: BLE001 — teardown must never raise
                pass


async def stream_chat_request(*args: Any, cancel_event=None, **kwargs: Any):
    _it = get_backend().stream_chat_request(*args, cancel_event=cancel_event, **kwargs)
    try:
        async for event in _it:
            yield event
    finally:
        _ac = getattr(_it, "aclose", None)
        if _ac is not None:
            try:
                await _ac()
            except Exception:  # noqa: BLE001 — teardown must never raise
                pass


def loaded_models() -> tuple[tuple[str, str], ...]:
    return tuple(get_backend().loaded_models())


def loading_models() -> tuple[str, ...]:
    return tuple(get_backend().loading_models())


def evict_model(model_key: str, *, task: Optional[str] = None) -> bool:
    return get_backend().evict_model(model_key, task=task)


def clear_models() -> None:
    get_backend().clear_models()


def supported_tasks() -> tuple[tuple[str, str], ...]:
    return tuple(get_backend().supported_tasks())


__all__ = [
    "clear_models",
    "evict_model",
    "execute_request",
    "loaded_models",
    "loading_models",
    "resolve_request",
    "runner_for",
    "stream_chat_request",
    "stream_request",
    "supported_tasks",
]
