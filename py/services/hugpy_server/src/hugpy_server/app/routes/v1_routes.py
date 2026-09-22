"""Public, OpenAI-compatible inference API (/v1) + API-key management.

The UI is just one client; this is the programmatic surface. Any OpenAI SDK
or plain curl works against it:

    client = OpenAI(base_url="https://dev.hugpy.ai/api/v1", api_key="hp_…")
    client.chat.completions.create(model="Qwen2.5-1.5B-Instruct",
                                   messages=[...], stream=True)

Auth is optional by design: a site-level `require_key` flag (toggled from
the UI) decides whether /v1 calls must present `Authorization: Bearer hp_…`.
Keys themselves are minted/revoked from the UI via the /keys routes below,
which are part of the site (same origin as the console), not part of /v1.

Model names accept any form assure_model_key resolves — registry key,
hub id (org/name), or manifest slug.
"""
from __future__ import annotations

import json
import time
import uuid
from functools import wraps

from abstract_flask import get_bp
from flask import Response, jsonify, request, stream_with_context

from hugpy_engine.config.models.models_config import get_models_dict
from hugpy_server.app.functions.chat.streaming import chat_iter_sync
from hugpy_storage.console.cancelable_downloads import update_model_status
from hugpy_server.app.functions.imports.utils.api_keys import (
    api_key_required,
    create_api_key,
    list_api_keys,
    prune_api_keys,
    revoke_api_key,
    set_api_key_required,
    verify_api_key,
)
# Default media-chat model_key (single global value); explicit import — not in the functions star-export.
from hugpy_engine.config.models.models_config import media_default_state
# k9 video-share key store (its OWN category/store — never the /v1 api_keys store).
from hugpy_server.app.functions.imports.utils.video_share_keys import (
    create_share_key,
    list_share_keys,
    revoke_share_key,
)
# Pure request/response plumbing lives in v1_helpers (stdlib-only, no Flask)
# so it unit-tests offline; see that module's docstring.
from hugpy_server.app.routes.v1_helpers import (
    _build_tools_preamble,
    _completion_kwargs,
    _inject_tools_preamble,
    _fleet_status_text,
    _parse_tool_calls,
    _usage_block,
)

v1_bp, logger = get_bp("v1_bp", __name__)


# ──────────────────────────────────────────────────────────────────────────
# auth
# ──────────────────────────────────────────────────────────────────────────
def _bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # OpenAI SDKs send Bearer; allow ?api_key= for quick curl tests too.
    return request.args.get("api_key")


def _openai_error(message: str, status: int, err_type: str = "invalid_request_error",
                  retry_after: "int | None" = None):
    """The OpenAI-shaped error body. ``retry_after`` adds the standard header so
    a 503 is an INSTRUCTION ("come back in N seconds") rather than a brush-off —
    every OpenAI SDK and every well-behaved batch client honours it."""
    body = jsonify({"error": {"message": message, "type": err_type, "code": status}})
    if retry_after is None:
        return body, status
    return body, status, {"Retry-After": str(int(retry_after))}


# How long a capacity-refused caller is told to wait. Read from the ONE knob the
# relay uses, so the header and the message can never disagree.
def _capacity_retry_after() -> int:
    try:
        from hugpy_engine.resolvers.remote import _cold_hold_retry_after_s
        return int(_cold_hold_retry_after_s())
    except Exception:   # noqa: BLE001 — a header default must never break a reply
        return 20


def _is_capacity_message(message: str) -> bool:
    """Is this terminal error the concurrent-load ADMISSION CAP refusing to start
    another cold load? Distinct from ``worker_busy`` (a worker at its in-process
    limit): this one is CENTRAL protecting its own request pool. Both are honest
    503s; only this one carries a Retry-After from the hold config."""
    try:
        from hugpy_engine.resolvers.remote import ColdHoldCapacityError
        return ColdHoldCapacityError.code in (message or "")
    except Exception:   # noqa: BLE001 — older core without the cap
        return "cold_load_capacity" in (message or "")


def v1_auth(fn):
    """Enforce the site's key policy: open unless require_key is on."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # required_scope="v1" (2026-07-23): a key must carry the "v1" scope or
        # "full" to pass this gate. Legacy keys (no scopes field) read as
        # ["full"], so every pre-scope key still passes — only deliberately
        # narrower keys (e.g. an "ml"-only install-link key) are refused here.
        if api_key_required() and not verify_api_key(_bearer_token(),
                                                     required_scope="v1"):
            return _openai_error(
                "Missing or invalid API key. Pass 'Authorization: Bearer <key>' "
                "(create keys in the console under API access).",
                401, "authentication_error",
            )
        return fn(*args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────────────────
# /v1/models
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/v1/models", methods=["GET"])
@v1_auth
def v1_models():
    manifest = get_models_dict(dict_return=True)
    media_default = media_default_state()
    # Operator BLOCK set (guarded — a listing must never 500 over the blocklist).
    try:
        from hugpy_fleet.central.blocklist import blocked_keys
        _blocked = blocked_keys()
    except Exception:  # noqa: BLE001
        _blocked = set()
    data = []
    for key, model in manifest.items():
        model = update_model_status(model)
        if model.get("status") != "installed":
            continue
        # Derived facts that aren't ModelConfig FIELDS ride in `extra` (the
        # class's leftover catcher), so read both — a top-level value wins.
        extra = model.get("extra") if isinstance(model.get("extra"), dict) else {}
        data.append({
            "id": key,
            "object": "model",
            "created": 0,
            "owned_by": "hugpy",
            "hub_id": model.get("hub_id"),
            "task": model.get("primary_task") or model.get("task"),
            # FULL capability list — `task` (primary) alone hid secondary
            # capabilities from every task-filtered UI (e.g. a dual
            # text-to-image + image-to-image model looked t2i-only).
            "tasks": model.get("tasks") or ([model.get("primary_task")] if model.get("primary_task") else []),
            "context_length": model.get("model_max_length"),
            # k61 — WHY a listed model cannot be picked for a task. Adapters and
            # pipeline components are real, present files; they are simply not
            # servable on their own. The task-filtered pickers show them greyed
            # with this reason rather than either offering them (they refuse) or
            # hiding them (the operator then can't see the file they downloaded).
            "adapter": bool(model.get("adapter", extra.get("adapter"))),
            "serveable": model.get("serveable", extra.get("serveable", True)),
            "unserveable_reason": model.get(
                "unserveable_reason", extra.get("unserveable_reason")),
            "media_default": (key == media_default),
            # Additive: ⛔ blocked from the serving pool by the operator. A call
            # naming a blocked model fails fast with the distinct blocked reason.
            "blocked": (key in _blocked),
        })
    return jsonify({"object": "list", "data": data})


# ──────────────────────────────────────────────────────────────────────────
# /v1/chat/completions
# (payload -> prompt_kwargs translation is _completion_kwargs in v1_helpers)
# ──────────────────────────────────────────────────────────────────────────
async def _v1_events(prompt_kwargs: dict):
    """Raw StreamEvents from the chat engine (late import dodges circulars).

    Registered in the live queue (same as the console /chat/stream path) so /v1
    (OpenAI-compatible) traffic shows in the activity view too. Best-effort —
    queue bookkeeping must never break a completion."""
    from hugpy_engine import stream_query
    from hugpy_engine.dispatch import activity
    rid = prompt_kwargs.get("request_id")
    mk = prompt_kwargs.get("model_key")
    name = mk
    try:
        from hugpy_engine.config.main import get_model_config
        if mk:
            name = getattr(get_model_config(mk), "name", None) or mk
    except Exception:
        pass
    activity.begin(rid, mk, name, kind="v1")
    try:
        async for event in stream_query(**prompt_kwargs):
            if getattr(event, "type", None) == "token":
                activity.on_token(rid)
            yield event
    finally:
        activity.end(rid)


def _finish_reason(reason: str | None) -> str:
    return {"max_tokens": "length"}.get(reason or "stop", reason or "stop")


def _is_request_shape_message(message: str) -> bool:
    """Is this terminal error a malformed-REQUEST error (chat template) ?

    Delegates to the ONE classifier (managers.resolvers.remote) so the route and
    the relay can never drift on what counts as a request-shape fault. Any
    import failure degrades to False — today's behaviour, never a new 500 path.
    """
    try:
        from hugpy_engine.resolvers.remote import _is_request_shape_error
        return bool(_is_request_shape_error(message))
    except Exception:  # noqa: BLE001 — classification must never break a reply
        return False


@v1_bp.route("/v1/chat/completions", methods=["POST"])
@v1_auth
def v1_chat_completions():
    payload = request.get_json(silent=True) or {}

    # Central-side tools shim (see v1_helpers): the frozen engine schema can't
    # carry `tools`, so tool-calling is prompt-injected here and parsed back
    # out of the reply — every GGUF model gains it with no engine change.
    # tool_choice "none" (or no usable tool entries) leaves tools_preamble
    # None and the request behaves exactly as today.
    tools_preamble = _build_tools_preamble(payload.get("tools"),
                                           payload.get("tool_choice"))
    if tools_preamble and payload.get("messages"):
        payload = dict(payload)
        payload["messages"] = _inject_tools_preamble(payload["messages"],
                                                     tools_preamble)

    try:
        prompt_kwargs = _completion_kwargs(payload)
    except (ValueError, TypeError) as exc:
        return _openai_error(str(exc), 400)

    # REJECT-AT-INTAKE (slice 9, defect 3): a request naming a model that resolves
    # to NOTHING (not in the registry/catalog, no worker designation) must be
    # rejected NOW with a 4xx + known-keys hint — never accepted into a
    # pending-forever job (the operator's immortal flux2-klein row). We reuse the
    # DISPATCHER'S OWN resolution (resolve_model_key -> assure_model_key), so the
    # boundary is exact: a model that IS known but merely not local/loaded still
    # RESOLVES (registry membership, not disk presence) and queues — lazy download
    # is the design. Only a truly-unresolvable explicit key rejects. A None/
    # "default" model_key (no preference) is left for the engine to default.
    _mk = prompt_kwargs.get("model_key")
    if _mk:
        try:
            from hugpy_engine.resolvers.model_resolver import resolve_model_key
            resolve_model_key(model_key=_mk)
        except (KeyError, ValueError) as exc:
            # The resolver raises with the "Unknown model_key=..." / "did you
            # mean" hint (ValueError from hugpy_engine; KeyError historically).
            return _openai_error(str(exc).strip("'\""), 400)
        except Exception:
            # A non-resolution error (registry probe failed) must NOT reject a
            # legitimate request — fall through and let the engine try.
            pass
        # DRAFT-MODEL-GATE-20260910: a draft head can never answer; say so instead of queueing.
        try:
            from hugpy_engine.draft_models import draft_model_reason
            _draft = draft_model_reason(_mk)
        except Exception:  # noqa: BLE001
            _draft = None
        if _draft:
            return _openai_error(_draft, 400)

    # A tool call is one short, bounded turn — never auto-continue it. A
    # continuation pass is exactly what rambled the captured 2026-07-14
    # incident and would splice "Continue…" text into the JSON block. An
    # explicit client max_chunks still wins.
    if tools_preamble and "max_chunks" not in prompt_kwargs:
        prompt_kwargs["max_chunks"] = 1

    model = payload.get("model") or "default"
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if payload.get("stream"):
        def chunk(delta: dict, finish=None) -> bytes:
            return (
                "data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")

        # OpenAI semantics: usage rides in ONE extra final chunk (choices: []),
        # and only when the client opted in via stream_options.include_usage.
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        def usage_chunk(usage) -> bytes:
            return (
                "data: " + json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": _usage_block(usage),
                }, ensure_ascii=False) + "\n\n"
            ).encode("utf-8")

        async def sse():
            usage = None
            # Re-separate re-inlined <think> into reasoning_content so a client
            # like OpenCode renders a collapsible reasoning panel instead of raw
            # <think> text (operator 2026-07-31). Only when NOT buffering for
            # tools — the tools path parses the whole buffer at done and its
            # <tool_call> block must be seen intact.
            from hugpy_engine.utils.no_think import StreamingThinkSplitter
            splitter = None if tools_preamble else StreamingThinkSplitter()

            def _emit_split(text: str):
                """Yield content / reasoning_content deltas for a token."""
                if splitter is None:
                    if text:
                        yield chunk({"content": text})
                    return
                ans, rea = splitter.feed(text)
                if rea:
                    yield chunk({"reasoning_content": rea})
                if ans:
                    yield chunk({"content": ans})

            yield chunk({"role": "assistant", "content": ""})
            # Streaming with tools buffers the whole reply and parses at done —
            # the simplest CORRECT behavior: a <tool_call> block is only
            # recognizable once complete, and OpenAI SDKs accept the final
            # burst. A fully-incremental tool-call stream (deltas per argument
            # fragment) is a later refinement. Non-tool requests stream
            # token-by-token exactly as before.
            buffered: list = []
            try:
                async for ev in _v1_events(prompt_kwargs):
                    t = getattr(ev, "type", None)
                    if t == "token":
                        if tools_preamble:
                            buffered.append(ev.text)
                        else:
                            for c in _emit_split(ev.text):
                                yield c
                    elif t == "done":
                        usage = getattr(ev, "usage", None)
                        finish = _finish_reason(ev.finish_reason)
                        if splitter is not None:
                            ans, rea = splitter.flush()
                            if rea:
                                yield chunk({"reasoning_content": rea})
                            if ans:
                                yield chunk({"content": ans})
                        if tools_preamble:
                            clean_text, tool_calls = _parse_tool_calls("".join(buffered))
                            buffered = []
                            if tool_calls:
                                yield chunk({"tool_calls": [
                                    {**tc, "index": i}
                                    for i, tc in enumerate(tool_calls)
                                ]})
                                finish = "tool_calls"
                            elif clean_text:
                                # No call — the buffered reply is plain content.
                                yield chunk({"content": clean_text})
                        yield chunk({}, finish=finish)
                    elif t == "status":
                        status = ev.model_dump()
                        # Structured field for Hugpy-aware clients plus the
                        # widely-supported reasoning delta so OpenCode/Qwen/
                        # Hermes/Aider can surface preparation without mixing
                        # it into the assistant's final answer text.
                        yield chunk({"reasoning_content": _fleet_status_text(status) + "\n",
                                     "hugpy_status": status})
                    elif t == "error":
                        if buffered:
                            yield chunk({"content": "".join(buffered)})
                            buffered = []
                        yield chunk({"content": f"\n[error: {ev.message}]"}, finish="stop")
            except Exception as exc:
                logger.exception("v1 stream failed")
                if buffered:
                    yield chunk({"content": "".join(buffered)})
                yield chunk({"content": f"\n[error: {exc}]"}, finish="stop")
            if include_usage:
                yield usage_chunk(usage)
            yield b"data: [DONE]\n\n"

        return Response(
            # heartbeat keeps a slow stream alive past an upstream proxy's read
            # timeout (the non-streaming drain below stays heartbeat-free).
            stream_with_context(chat_iter_sync(sse(), heartbeat=b": keepalive\n\n")),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # non-streaming: drain the same event stream and assemble one body
    text_parts: list[str] = []
    finish = "stop"
    error_message = None
    usage = None
    status_events = []
    try:
        for ev in chat_iter_sync(_v1_events(prompt_kwargs)):
            t = getattr(ev, "type", None)
            if t == "token":
                text_parts.append(ev.text)
            elif t == "done":
                finish = _finish_reason(ev.finish_reason)
                usage = getattr(ev, "usage", None)
            elif t == "status":
                status_events.append(ev.model_dump())
            elif t == "error":
                error_message = ev.message
    except KeyError as exc:
        # resolve() raises before the stream starts, e.g. unknown model
        return _openai_error(str(exc).strip("'\""), 404, "invalid_request_error")
    except Exception as exc:
        # The admission cap raised out of the one-shot relay (rather than being
        # yielded as an error event): a 503 + Retry-After, and NOT an exception
        # traceback — nothing failed, we declined to start another load.
        if _is_capacity_message(f"{exc}"):
            return _openai_error(f"{exc}", 503, "server_busy",
                                 retry_after=_capacity_retry_after())
        logger.exception("v1 completion failed")
        # Same classification for an exception that escaped the stream (the
        # one-shot relay raises rather than yielding): a request-shape fault is
        # a 400, never a 500 the client is invited to retry.
        if _is_request_shape_message(f"{type(exc).__name__}: {exc}"):
            return _openai_error(f"{exc}", 400, "invalid_request_error")
        return _openai_error(f"{type(exc).__name__}: {exc}", 500, "api_error")

    if error_message and not text_parts:
        # Cold-hold ADMISSION CAP: central is already holding its maximum number
        # of concurrent model loads and refused to start another rather than park
        # this request in one of the site's 24 slots for up to the hold ceiling.
        # Nothing is broken — 503 + Retry-After, the honest, actionable answer.
        if _is_capacity_message(error_message):
            return _openai_error(error_message, 503, "server_busy",
                                 retry_after=_capacity_retry_after())
        # Cap-aware relay gate (concurrency hardening): a busy in-process runner
        # is not a fault — every holder of the model is momentarily at its safe
        # concurrency limit. Answer 503 (retryable) so a batch client backs off
        # instead of treating it as a hard error.
        if "worker_busy" in error_message or "model_busy" in error_message:
            return _openai_error(error_message, 503, "server_busy")
        # A malformed request (the model's chat template refuses this message
        # sequence) is the CLIENT's fault, not ours: answer 400
        # invalid_request_error so an OpenAI SDK raises BadRequestError instead
        # of retrying a 500 — and so nothing blames the box's size.
        if _is_request_shape_message(error_message):
            return _openai_error(error_message, 400, "invalid_request_error")
        status = 404 if "Unknown model" in error_message else 500
        return _openai_error(error_message, status, "api_error")

    content = "".join(text_parts)
    message = {"role": "assistant", "content": content}
    # Non-streaming: hand reasoning back under reasoning_content (OpenAI reasoning
    # shape) so OpenCode collapses it, instead of leaving <think> inline in the
    # answer (operator 2026-07-31). Skipped for the tools path — the buffered
    # <tool_call> block is parsed out of `content` below. reasoning_content is
    # omitted when there is none, so a plain answer is byte-identical to before.
    if not tools_preamble:
        from hugpy_engine.utils.no_think import strip_think
        _ans, _reasoning = strip_think(content)
        # Rewrite when the reply CARRIED think tags at all — including an EMPTY
        # <think></think> a chat template pre-opened, which has no reasoning to
        # surface but must still not sit in the answer as literal tags.
        if _ans != content or _reasoning:
            message = {"role": "assistant", "content": _ans}
            if _reasoning:
                message["reasoning_content"] = _reasoning
    if tools_preamble:
        # Errors-as-data: _parse_tool_calls returns (original text, None) on
        # no/malformed calls, so the worst case is a plain content answer —
        # a shim parse failure can never 500 the route.
        clean_text, tool_calls = _parse_tool_calls(content)
        if tool_calls:
            message = {"role": "assistant", "content": clean_text or None,
                       "tool_calls": tool_calls}
            finish = "tool_calls"

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        # Real token accounting threaded up from the runner via the done
        # event; all-None only when genuinely unavailable (never a crash).
        "usage": _usage_block(usage),
    }
    if status_events:
        result["hugpy_status"] = status_events
    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────
# auth config (console-side) — tells the UI how to authenticate users.
# mode "external": delegate to a separate login service (HUGPY_AUTH_BASE).
# mode "open":     single-operator instance, no login wall (distribution
#                  default; the /v1 key system still gates programmatic use).
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/fleet/runbook", methods=["GET"])
def fleet_runbook():
    """Serve the SINGLE-SOURCE-OF-TRUTH fleet remediation runbook
    (``hugpy_fleet.doctrine.runbook``). Both the console Docs 'Fleet
    remediation runbook' page and the hugpy-agent read this one document — edit
    it there only. Public GET (no secrets)."""
    try:
        from hugpy_fleet.doctrine.runbook import load_runbook
        return jsonify(load_runbook())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "runbook unavailable", "detail": str(exc)}), 404


@v1_bp.route("/auth/config", methods=["GET"])
def auth_config():
    import os as _os
    try:
        from hugpy_platform.platform_facade import env_value as _gev
    except Exception:
        _gev = lambda *_a, **_k: None
    mode = (_os.environ.get("HUGPY_AUTH_MODE") or _gev("HUGPY_AUTH_MODE") or "external").lower()
    if mode not in ("open", "external"):
        mode = "external"
    if mode != "external":
        return jsonify({"mode": mode, "base": None})
    # external mode: by default advertise a SAME-ORIGIN base so the UI talks to
    # our auth proxy (auth_proxy_routes.py) instead of the upstream auth service
    # directly. That keeps the session cookie first-party → Safari/Firefox stop
    # dropping it (the cross-site third-party-cookie login loop). Set
    # HUGPY_AUTH_PROXY=0 to fall back to advertising the upstream directly.
    from hugpy_server.app.routes.auth_proxy_routes import proxy_enabled, public_base
    if proxy_enabled():
        base = _os.environ.get("HUGPY_AUTH_PUBLIC_BASE") or public_base()
    else:
        base = (_os.environ.get("HUGPY_AUTH_BASE") or _gev("HUGPY_AUTH_BASE")
                or "https://api.abstractendeavors.com")
    return jsonify({"mode": mode, "base": base})


# ──────────────────────────────────────────────────────────────────────────
# key management (console-side, same-origin; not part of the /v1 surface)
# ──────────────────────────────────────────────────────────────────────────
@v1_bp.route("/keys", methods=["GET"])
def keys_list():
    return jsonify({"require_key": api_key_required(), "keys": list_api_keys()})


@v1_bp.route("/keys", methods=["POST"])
def keys_create():
    body = request.get_json(silent=True) or {}
    # Optional `pool` binds the key to a dedicated worker pool so the app's
    # requests route to its reserved workers from the key alone.
    # Optional `label` / `scopes` (2026-07-23): scoped/labeled keys; omitted =>
    # full-scope, exactly the pre-scope mint. Unknown scopes 400 (never a typo
    # silently minting the wrong key).
    scopes = body.get("scopes")
    if scopes is not None and not isinstance(scopes, list):
        return jsonify({"error": "'scopes' must be a list"}), 400
    try:
        return jsonify(create_api_key(
            body.get("name", ""), pool=body.get("pool", ""),
            label=body.get("label", ""), scopes=scopes))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@v1_bp.route("/keys/<key_id>", methods=["DELETE"])
def keys_revoke(key_id):
    if not revoke_api_key(key_id):
        return jsonify({"ok": False, "error": "unknown key id"}), 404
    return jsonify({"ok": True})


@v1_bp.route("/keys/prune", methods=["POST"])
def keys_prune():
    """Bulk-revoke 'dated' keys. Body (all optional):
      {"expired": true,            # revoke keys whose expires_at has passed (default)
       "older_than_days": 90,      # also revoke keys minted > N days ago
       "unused_only": true}        # ...but among the age sweep, only never-used ones
    Returns {ok, pruned: [{id,name,prefix,reason}], count}."""
    body = request.get_json(silent=True) or {}
    older = body.get("older_than_days")
    try:
        older = float(older) if older not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "'older_than_days' must be a number"}), 400
    pruned = prune_api_keys(
        expired=bool(body.get("expired", True)),
        older_than_days=older,
        unused_only=bool(body.get("unused_only", False)))
    return jsonify({"ok": True, "pruned": pruned, "count": len(pruned)})


@v1_bp.route("/keys/require", methods=["PUT"])
def keys_require():
    body = request.get_json(silent=True) or {}
    return jsonify({"require_key": set_api_key_required(bool(body.get("require")))})


# ──────────────────────────────────────────────────────────────────────────
# k9 — VIDEO-SHARE links. Mint a video-scoped share credential (a NEW key
# category, its own store — see functions/.../video_share_keys.py) and hand its
# link to an outside party so they can drive the /video features WITHOUT a
# console login. These routes are OPERATOR-ONLY (operator_auth._SENSITIVE gates
# ^/keys/video-share) and are deliberately NOT on the /video surface, so a
# share principal (which can pass the /video gate) can never reach them —
# structurally "no key-minting-by-key". The GET doubles as the SPA's auth probe.
# ──────────────────────────────────────────────────────────────────────────
def _public_base() -> str:
    """Best-effort public origin for building a share URL. Prefers an explicit
    HUGPY_PUBLIC_BASE; else reconstructs from the forwarded host/proto (the SPA
    also rebuilds the link from window.location.origin, so this is the
    curl/programmatic path)."""
    import os as _os
    base = (_os.environ.get("HUGPY_PUBLIC_BASE") or "").strip()
    if base:
        return base.rstrip("/")
    proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "https").split(",")[0].strip()
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").split(",")[0].strip()
    return f"{proto}://{host}".rstrip("/") if host else ""


@v1_bp.route("/keys/video-share", methods=["GET"])
def video_share_list():
    return jsonify({"keys": list_share_keys()})


@v1_bp.route("/keys/video-share", methods=["POST"])
def video_share_create():
    body = request.get_json(silent=True) or {}
    ttl = body.get("ttl_days", None)
    minted = create_share_key(label=body.get("label", ""),
                              ttl_days=(ttl if ttl is not None else 30))
    base = _public_base()
    minted["url"] = f"{base}/video/?share={minted['key']}" if base else \
        f"/video/?share={minted['key']}"
    return jsonify(minted)


@v1_bp.route("/keys/video-share/<key_id>", methods=["DELETE"])
def video_share_revoke(key_id):
    if not revoke_share_key(key_id):
        return jsonify({"ok": False, "error": "unknown key id"}), 404
    return jsonify({"ok": True})
