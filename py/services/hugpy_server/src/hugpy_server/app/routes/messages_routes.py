"""Anthropic Messages API shim (`POST /v1/messages`) so Claude Code and the
Claude Agent SDK can point straight at hugpy's fleet — no external proxy.

Claude Code speaks Anthropic's Messages format; hugpy's engine speaks OpenAI's
`/v1/chat/completions`. This blueprint TRANSLATES Anthropic<->OpenAI and REUSES
the existing serving path — it builds the SAME OpenAI-style `payload` dict that
`v1_chat_completions` builds and runs it through the identical
`_build_tools_preamble`/`_inject_tools_preamble` -> `_completion_kwargs` ->
`_v1_events` pipeline, then formats the OUTPUT as Anthropic (messages_helpers).

No new model plumbing. The only things new here are protocol translation and
the Anthropic error/SSE shapes.
"""
from __future__ import annotations

import json
import logging

from flask import Response, jsonify, request, stream_with_context

from abstract_flask import get_bp
from hugpy_server.app.functions.chat.streaming import chat_iter_sync
from hugpy_server.app.functions.imports.utils.api_keys import api_key_required, verify_api_key
# Pure OpenAI-side plumbing, reused verbatim from the /v1 seam.
from hugpy_server.app.routes.v1_helpers import (
    _build_tools_preamble,
    _completion_kwargs,
    _inject_tools_preamble,
    _fleet_status_text,
    _parse_tool_calls,
    _usage_block,
)
# The /v1 route owns the engine event stream and the terminal-error classifiers;
# reuse them so /v1/messages and /v1/chat/completions can never drift on how a
# completion is driven or how a busy/malformed failure is graded.
from hugpy_server.app.routes.v1_routes import (
    _v1_events,
    _finish_reason,
    _is_capacity_message,
    _is_request_shape_message,
    _capacity_retry_after,
)
from hugpy_server.app.routes import messages_helpers as mh

messages_bp, _bp_logger = get_bp("messages_bp", __name__)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# auth — same key policy as /v1, but ALSO accept Anthropic's x-api-key header.
# ──────────────────────────────────────────────────────────────────────────
def _messages_token() -> "str | None":
    """The presented credential.

    Claude Code sends ANTHROPIC_AUTH_TOKEN as `Authorization: Bearer <key>`;
    the Anthropic SDKs send `x-api-key: <key>`. Accept either (plus ?api_key=
    for a quick curl), mirroring the /v1 bearer gate but widened for Anthropic
    clients.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    xkey = request.headers.get("x-api-key")
    if xkey:
        return xkey.strip()
    return request.args.get("api_key")


def _anthropic_error(message: str, status: int, err_type: str = "invalid_request_error",
                     retry_after: "int | None" = None):
    """The Anthropic-shaped error body. Never a 500 traceback to the client."""
    body = jsonify({"type": "error",
                    "error": {"type": err_type, "message": message}})
    if retry_after is None:
        return body, status
    return body, status, {"Retry-After": str(int(retry_after))}


def _auth_ok() -> bool:
    # Same policy as v1_auth: open unless require_key is on; a key must carry
    # the "v1" scope (or "full"). Legacy keys read as full and still pass.
    if not api_key_required():
        return True
    return bool(verify_api_key(_messages_token(), required_scope="v1"))


# ──────────────────────────────────────────────────────────────────────────
# request -> OpenAI payload -> engine prompt_kwargs (reused pipeline)
# ──────────────────────────────────────────────────────────────────────────
def _prepare(body: dict):
    """(prompt_kwargs, tools_preamble, model_echo, input_tokens_estimate).

    Translates the Anthropic body to an OpenAI payload, then runs the exact
    tools-preamble + `_completion_kwargs` steps `v1_chat_completions` runs.
    Raises ValueError for a bad request (-> 400).
    """
    # Map claude-* ids to the AGENT BRAIN here (routes-side, where constants
    # are importable — messages_helpers stays stdlib-only and its own
    # fallback maps to "default"). "default" resolves to DEFAULT_CHAT_MODEL,
    # a small fast chat model whose slot ctx an agent-sized prompt can
    # exceed; Claude Code is an agent client, so it gets the agent brain.
    import os as _os
    requested = body.get("model")
    resolved = None

    def _mk(x):
        return x.get("model_key") if isinstance(x, dict) else x

    # (1) Explicit per-model override: `hcon connect claude-code` sets
    #     ANTHROPIC_CUSTOM_HEADERS: "X-Hugpy-Model: <fleet key>". Honor it if it
    #     resolves to a real fleet model — gives claude-code true per-model select.
    _hdr = request.headers.get("X-Hugpy-Model")
    if _hdr:
        try:
            from hugpy_engine.resolvers.assure_model_key import assure_model_key
            resolved = assure_model_key(_hdr.strip())
        except Exception:
            resolved = None

    # (2) Otherwise a claude-* name resolves through the model-GROUPS pipeline,
    #     seeded at the agent brain: the operator/derived group's best AVAILABLE
    #     member (fit + order), reusing the same resolver /llm/model-groups/resolve
    #     previews. Ungrouped or unavailable -> the seed itself (brain).
    if resolved is None and isinstance(requested, str) and requested.strip().lower().startswith("claude"):
        from hugpy_platform.constants import DEFAULT_AGENT_BRAIN
        seed = _os.environ.get("HUGPY_ANTHROPIC_SEED", DEFAULT_AGENT_BRAIN)
        resolved = DEFAULT_AGENT_BRAIN
        try:
            from hugpy_fleet.central.priority_group_settings import resolve as _grp_resolve
            _r = _grp_resolve(seed, None, None, full=True) or {}
            resolved = _mk(_r.get("route")) or _mk(_r.get("chosen")) or seed
        except Exception:
            resolved = seed

    if resolved:
        body = {**body, "model": resolved}

    payload = mh.anthropic_to_openai_payload(body)

    tools_preamble = _build_tools_preamble(payload.get("tools"),
                                           payload.get("tool_choice"))
    if tools_preamble and payload.get("messages"):
        payload = dict(payload)
        payload["messages"] = _inject_tools_preamble(payload["messages"],
                                                     tools_preamble)

    prompt_kwargs = _completion_kwargs(payload)
    # A tool call is one short bounded turn — never auto-continue it (same guard
    # as /v1). An explicit client max_chunks still wins.
    if tools_preamble and "max_chunks" not in prompt_kwargs:
        prompt_kwargs["max_chunks"] = 1

    model_echo = requested or body.get("model") or "default"
    input_tokens = _estimate_input_tokens(payload.get("messages"))
    return prompt_kwargs, tools_preamble, model_echo, input_tokens


def _estimate_input_tokens(messages) -> int:
    """Cheap best-effort input-token estimate (~4 chars/token) for usage +
    count_tokens. Never authoritative; the done-event usage overrides it when
    the runner reports real counts."""
    chars = 0
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chars += len(part["text"])
    return max(1, chars // 4) if chars else 0


# ──────────────────────────────────────────────────────────────────────────
# POST /v1/messages
# ──────────────────────────────────────────────────────────────────────────
@messages_bp.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    if not _auth_ok():
        return _anthropic_error(
            "Missing or invalid API key. Pass 'x-api-key: <key>' or "
            "'Authorization: Bearer <key>' (create keys in the console under "
            "API access).",
            401, "authentication_error",
        )

    body = request.get_json(silent=True) or {}
    try:
        prompt_kwargs, tools_preamble, model_echo, input_tokens = _prepare(body)
    except (ValueError, TypeError) as exc:
        return _anthropic_error(str(exc), 400, "invalid_request_error")

    if body.get("stream"):
        return _stream(prompt_kwargs, tools_preamble, model_echo, input_tokens)
    return _non_stream(prompt_kwargs, tools_preamble, model_echo, input_tokens)


def _non_stream(prompt_kwargs, tools_preamble, model_echo, input_tokens):
    """Drain the engine event stream and assemble one Anthropic message body."""
    text_parts: list = []
    finish = "stop"
    usage = None
    error_message = None
    try:
        for ev in chat_iter_sync(_v1_events(prompt_kwargs)):
            t = getattr(ev, "type", None)
            if t == "token":
                text_parts.append(ev.text)
            elif t == "done":
                finish = _finish_reason(ev.finish_reason)
                usage = getattr(ev, "usage", None)
            elif t == "error":
                error_message = ev.message
    except KeyError as exc:
        return _anthropic_error(str(exc).strip("'\""), 404, "not_found_error")
    except Exception as exc:  # noqa: BLE001
        if _is_capacity_message(f"{exc}"):
            return _anthropic_error(f"{exc}", 503, "overloaded_error",
                                    retry_after=_capacity_retry_after())
        logger.exception("/v1/messages completion failed")
        if _is_request_shape_message(f"{type(exc).__name__}: {exc}"):
            return _anthropic_error(f"{exc}", 400, "invalid_request_error")
        return _anthropic_error(f"{type(exc).__name__}: {exc}", 500, "api_error")

    if error_message and not text_parts:
        if _is_capacity_message(error_message):
            return _anthropic_error(error_message, 503, "overloaded_error",
                                    retry_after=_capacity_retry_after())
        if "worker_busy" in error_message or "model_busy" in error_message:
            return _anthropic_error(error_message, 503, "overloaded_error")
        if _is_request_shape_message(error_message):
            return _anthropic_error(error_message, 400, "invalid_request_error")
        status, etype = ((404, "not_found_error") if "Unknown model" in error_message
                         else (500, "api_error"))
        return _anthropic_error(error_message, status, etype)

    content = "".join(text_parts)
    reasoning = ""
    tool_calls = None
    if tools_preamble:
        clean_text, tool_calls = _parse_tool_calls(content)
        content = clean_text if tool_calls else content
    else:
        # Separate re-inlined <think> into a thinking block (Anthropic native).
        from hugpy_engine.utils.no_think import strip_think
        content, reasoning = strip_think(content)

    had_tools = bool(tool_calls)
    stop_reason = mh.finish_to_stop_reason(finish, had_tool_calls=had_tools)
    blocks = mh.build_content_blocks(reasoning, content, tool_calls)

    return jsonify({
        "id": mh.new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model_echo,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": mh.usage_to_anthropic(_usage_block(usage)),
    })


def _stream(prompt_kwargs, tools_preamble, model_echo, input_tokens):
    """Anthropic SSE. Text+thinking stream fully; tool-heavy turns buffer and
    emit a single tool_use block with input_json_delta at done (mirrors how /v1
    buffers tools). See the module/skill notes — nothing is silently dropped."""
    from hugpy_engine.utils.no_think import StreamingThinkSplitter

    message_id = mh.new_message_id()

    async def sse():
        # A thinking splitter routes tokens into thinking_delta vs text_delta;
        # for a tools turn we buffer the raw reply and parse calls at done, so
        # no splitter (the <tool_call> block must be seen intact).
        enc = mh.AnthropicStreamEncoder(
            message_id, model_echo, StreamingThinkSplitter(),
            input_tokens=input_tokens)
        buffered: list = []
        usage = None
        stop_reason = "end_turn"
        for frame in enc.start():
            yield frame
        try:
            async for ev in _v1_events(prompt_kwargs):
                t = getattr(ev, "type", None)
                if t == "token":
                    if tools_preamble:
                        buffered.append(ev.text)
                    else:
                        for frame in enc.feed(ev.text):
                            yield frame
                elif t == "done":
                    usage = getattr(ev, "usage", None)
                    finish = _finish_reason(ev.finish_reason)
                    if tools_preamble:
                        for frame in _finish_tools(enc, buffered, finish):
                            yield frame
                        return  # message_delta+stop already emitted
                    stop_reason = mh.finish_to_stop_reason(finish)
                    break
                elif t == "status":
                    status = ev.model_dump()
                    # Anthropic-native thinking keeps fleet preparation
                    # visible in Claude harnesses but separate from answer
                    # text and tool input.
                    yield mh.sse("hugpy_status", {"type": "hugpy_status", **status})
                    for frame in enc._emit(_fleet_status_text(status) + "\n", "thinking"):
                        yield frame
                elif t == "error":
                    # Close cleanly; surface the error as text so the stream
                    # never hangs (Anthropic has no resumable error SSE event).
                    if not tools_preamble and getattr(ev, "message", None):
                        for frame in enc.feed(f"\n[error: {ev.message}]"):
                            yield frame
                    stop_reason = "end_turn"
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("/v1/messages stream failed")
            if not tools_preamble:
                for frame in enc.feed(f"\n[error: {exc}]"):
                    yield frame
            stop_reason = "end_turn"

        out_tokens = mh.usage_to_anthropic(_usage_block(usage))["output_tokens"]
        for frame in enc.finish(stop_reason, out_tokens):
            yield frame

    return Response(
        stream_with_context(chat_iter_sync(sse(),
                                           heartbeat=mh.sse("ping", {"type": "ping"}))),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _finish_tools(enc, buffered, finish):
    """Emit buffered tool calls as tool_use content blocks, then close the
    message with stop_reason=tool_use (or end_turn when no call parsed)."""
    clean_text, tool_calls = _parse_tool_calls("".join(buffered))
    # If the model produced plain prose (no call), stream it as one text block.
    if not tool_calls:
        if clean_text:
            yield from enc._emit(clean_text, "text")
        yield from enc.finish(mh.finish_to_stop_reason(finish), 0)
        return
    for tc in tool_calls:
        fn = tc.get("function") or {}
        tuid = tc.get("id") or mh.new_tool_use_id()
        yield from enc.open_tool_use(tuid, fn.get("name") or "")
        # arguments is a JSON string already; ship it as one input_json_delta.
        yield from enc.tool_use_json(fn.get("arguments") or "{}")
    yield from enc._close_open()
    yield mh.sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield mh.sse("message_stop", {"type": "message_stop"})


# ──────────────────────────────────────────────────────────────────────────
# POST /v1/messages/count_tokens — cheap best-effort estimate (optional call).
# ──────────────────────────────────────────────────────────────────────────
@messages_bp.route("/v1/messages/count_tokens", methods=["POST"])
def anthropic_count_tokens():
    if not _auth_ok():
        return _anthropic_error(
            "Missing or invalid API key.", 401, "authentication_error")
    body = request.get_json(silent=True) or {}
    try:
        payload = mh.anthropic_to_openai_payload({**body, "max_tokens": 1})
    except (ValueError, TypeError) as exc:
        return _anthropic_error(str(exc), 400, "invalid_request_error")
    n = _estimate_input_tokens(payload.get("messages"))
    return jsonify({"input_tokens": n})
