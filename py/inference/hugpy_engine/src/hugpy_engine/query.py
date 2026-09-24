"""Prompt-to-response orchestration."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Optional

from .backend import get_backend
from .catalog import get_model, text_chat_task
from .contracts import QueryError, QueryResult
from .runtime import stream_chat_request


def _messages_have_media(messages) -> bool:
    for message in messages or ():
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") in (
                "image",
                "image_url",
                "input_image",
            ):
                return True
    return False


def _request(
    prompt: Optional[str],
    messages: Optional[Sequence[Mapping[str, Any]]],
    model_key: Optional[str],
    request_id: Optional[str],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    if prompt is None and messages is None:
        raise ValueError("query requires either prompt or messages")
    if prompt is not None and messages is not None:
        raise ValueError("query accepts prompt or messages, not both")
    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    payload = dict(options)
    payload["request_id"] = request_id or uuid.uuid4().hex
    if model_key is not None:
        payload["model_key"] = model_key
    if prompt is not None:
        payload["prompt"] = prompt
    else:
        normalized = []
        for message in messages or ():
            if not isinstance(message, Mapping):
                raise TypeError("each message must be a mapping")
            normalized.append(dict(message))
        if not normalized:
            raise ValueError("messages must contain at least one message")
        payload["messages"] = normalized

    if model_key is not None:
        has_media = bool(payload.get("file") or payload.get("images"))
        has_media = has_media or _messages_have_media(messages)
        try:
            selected = text_chat_task(
                get_model(model_key),
                requested_task=payload.get("task"),
                has_media=has_media,
            )
        except Exception:
            selected = None
        if selected:
            payload["task"] = selected
    return payload


async def stream_query(
    prompt: Optional[str] = None,
    *,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
    model_key: Optional[str] = None,
    request_id: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    **options: Any,
) -> AsyncIterator[Any]:
    payload = _request(prompt, messages, model_key, request_id, options)
    # Per-model output repair (hugpy.json["output_repair"], derived by
    # hugpy-model-audit) — identity for every model without the block. The one
    # place /v1 (stream + non-stream, incl. the benchmark grader) and the
    # console chat relay all pass through.
    from .output_repair import repair_stream
    async for event in repair_stream(
            stream_chat_request(cancel_event=cancel_event, **payload),
            payload.get("model_key")):
        yield event


async def query_result(
    prompt: Optional[str] = None,
    *,
    messages: Optional[Sequence[Mapping[str, Any]]] = None,
    model_key: Optional[str] = None,
    request_id: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    **options: Any,
) -> QueryResult:
    rid = request_id or uuid.uuid4().hex
    parts: list[str] = []
    finish_reason = "stop"
    usage = timings = None
    completed = False

    async for event in stream_query(
        prompt,
        messages=messages,
        model_key=model_key,
        request_id=rid,
        cancel_event=cancel_event,
        **options,
    ):
        event_type = getattr(event, "type", None)
        if event_type == "token":
            parts.append(getattr(event, "text", "") or "")
        elif event_type == "done":
            completed = True
            finish_reason = getattr(event, "finish_reason", None) or "stop"
            event_usage = getattr(event, "usage", None)
            event_timings = getattr(event, "timings", None)
            usage = dict(event_usage) if isinstance(event_usage, Mapping) else None
            timings = dict(event_timings) if isinstance(event_timings, Mapping) else None
        elif event_type == "error":
            raise QueryError(
                getattr(event, "message", None) or "model query failed",
                request_id=rid,
                partial_text="".join(parts),
            )

    if not completed:
        raise QueryError(
            "model query ended without a completion event",
            request_id=rid,
            partial_text="".join(parts),
        )
    return QueryResult(
        text="".join(parts),
        request_id=rid,
        finish_reason=finish_reason,
        usage=usage,
        timings=timings,
    )


async def query(*args: Any, **kwargs: Any) -> str:
    return (await query_result(*args, **kwargs)).text


def query_result_sync(*args: Any, **kwargs: Any) -> QueryResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "query_result_sync cannot run inside an event loop; "
            "await query_result instead"
        )
    return get_backend().run_sync(query_result(*args, **kwargs))


def query_sync(*args: Any, **kwargs: Any) -> str:
    return query_result_sync(*args, **kwargs).text


__all__ = [
    "query",
    "query_result",
    "query_result_sync",
    "query_sync",
    "stream_query",
]
