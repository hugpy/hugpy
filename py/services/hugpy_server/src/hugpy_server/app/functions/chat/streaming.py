import asyncio
import json
import logging
import os
import uuid

from flask import Response, stream_with_context, request

from hugpy_platform import async_runtime
from hugpy_control.jobs import job_store
from hugpy_engine.config.main import get_model_config
from hugpy_platform.constants import DEFAULT_MAX_TOKENS
from hugpy_platform.utils import messages_to_dicts

from hugpy_engine.wire.chat_schemas import ChatBody


logger = logging.getLogger(__name__)

# SSE keepalive comment (lines starting with ':' are ignored by ES/OpenAI
# clients). Emitted during long gaps so an upstream proxy's read timeout can't
# cut a stream whose first token (or continuation pass) is slow.
SSE_KEEPALIVE = b": keepalive\n\n"
_HEARTBEAT_SECS = float(os.environ.get("HUGPY_SSE_HEARTBEAT_SECS", "15") or 15)


def sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# Stages a status event may carry that mean "the model is being made ready"
# (the cold-load HOLD, t36, and the worker's own provisioning). Any of them
# moves the job to `processing`/`awaiting-load` so it reads honestly and its
# progressed_at is fed (never orphan-expired mid-load).
_LOADING_STAGES = ("awaiting-load", "loading", "provision")


def _feed_job_from_status(rid, event) -> None:
    """Reflect a chat status event into the shared job (F5). Best-effort — job
    bookkeeping must never break a chat.

    * dispatch banner (``served_by``): stamp the SELECTED worker onto the job.
      The pending-orphan sweep only expires WORKER-LESS pending rows, so a call
      that IS placed on a worker and loading can no longer be auto-expired.
    * awaiting-load / provisioning progress: move the job to ``processing`` +
      stage ``awaiting-load`` and mirror progress; ``update`` bumps
      ``progressed_at`` on a status/stage/progress advance, feeding the honest
      stall clock while the load churns."""
    from hugpy_control.jobs import job_store
    try:
        if getattr(event, "served_by", None) is not None:
            if getattr(event, "served_by", None) == "worker":
                wname = (getattr(event, "worker_name", None)
                         or getattr(event, "worker_id", None))
                if wname:
                    job_store.update(rid, worker=str(wname))
            return
        stage = getattr(event, "stage", None)
        if stage in _LOADING_STAGES:
            changes = {"status": "processing", "stage": stage}
            prog = getattr(event, "progress", None)
            if isinstance(prog, (int, float)):
                changes["progress"] = float(prog)
            msg = getattr(event, "message", None)
            if msg:
                changes["message"] = str(msg)
            job_store.update(rid, **changes)
    except Exception:
        pass


def event_to_sse(ev) -> bytes:
    """Serialize a dispatch StreamEvent to the browser's SSE wire shape.

    token/done/error get their minimal browser payloads; everything else
    (status / provisioning progress / continuation markers — including events
    relayed from a GPU worker) rides through verbatim via model_dump().
    """
    t = getattr(ev, "type", None)
    if t == "token":
        return sse_event({"type": "token", "text": ev.text})
    if t == "done":
        return sse_event({"type": "done", "finish_reason": ev.finish_reason})
    if t == "error":
        return sse_event({"type": "error", "message": ev.message})
    return sse_event(ev.model_dump())


def chat_iter_sync(agen, heartbeat: "bytes | None" = None,
                   heartbeat_secs: float = _HEARTBEAT_SECS):
    """Drive an async generator from Flask's synchronous WSGI context.

    When ``heartbeat`` bytes are given (streaming callers), each step waits at
    most ``heartbeat_secs`` for the next event; on timeout it yields the
    keepalive and keeps waiting on the SAME pending step. This keeps an upstream
    proxy from timing out a slow stream, and — because every keepalive is a
    write — lets the WSGI server notice a dead client quickly and trigger the
    teardown below. ``heartbeat=None`` (the non-streaming drain) keeps the plain
    blocking behavior so internal callers see only real events.

    Drives on the process-wide async runtime (one long-lived loop), NOT a fresh
    per-request loop — so cached asyncio primitives never hit "bound to a
    different event loop", and many streams interleave on one loop instead of
    pinning a loop each. Teardown (cancel in-flight step → aclose the chain so
    GeneratorExit releases a relayed worker's httpx stream) lives in the runtime.
    """
    from hugpy_platform import async_runtime
    yield from async_runtime.iter_sync(agen, heartbeat=heartbeat,
                                       heartbeat_secs=heartbeat_secs)


def _resolve_max_new_tokens(body: ChatBody) -> int:
    """Default to the model's full context when the client didn't cap it.

    A tool, not a service — so when max_new_tokens is omitted we give the model
    as much room as it has. The engine auto-continues past this per-call cap, so
    this is the per-pass budget, not a hard ceiling on total output.
    """
    if body.max_new_tokens:
        return body.max_new_tokens
    try:
        cfg = get_model_config(body.model_key) if body.model_key else None
        ctx = getattr(cfg, "model_max_length", None)
        if ctx and int(ctx) > 0:
            return int(ctx)
    except Exception:
        pass
    # Fall back to the global default cap.
    try:
        return int(DEFAULT_MAX_TOKENS)
    except Exception:
        return 4096


async def stream_events(body: ChatBody):
    """Build prompt_kwargs and stream the unified chat engine to SSE.

    The route is deliberately dumb: it does NOT decide local vs worker. It hands
    prompt_kwargs to hugpy_engine.stream_query, whose dispatch engine drives
    resolve() — the single place that picks in-process / placement-peer /
    live-GPU-worker and falls back to local. So local and worker chat stream
    identically (token-by-token, with auto-continuation past the cap), and there
    is no separate worker-offload path in this route anymore.
    """
    from hugpy_engine import stream_query

    prompt_kwargs = {}
    if body.unbounded is not None:
        prompt_kwargs["unbounded"] = body.unbounded
        prompt_kwargs["max_new_tokens"] = body.max_new_tokens or _resolve_max_new_tokens(body)
    elif body.max_new_tokens:
        # Explicit cap from the client -> honor it (bounded, per-call).
        prompt_kwargs["max_new_tokens"] = body.max_new_tokens
    else:
        # No cap requested -> run unbounded: the runner generates chunk-by-chunk
        # until the model naturally stops, so the response is never truncated by
        # a token limit. (Per-chunk size uses the model's context.)
        prompt_kwargs["unbounded"] = True
        prompt_kwargs["max_new_tokens"] = _resolve_max_new_tokens(body)

    if body.model_key:
        prompt_kwargs["model_key"] = body.model_key

    # Route to a dedicated worker pool when set (resolved in chat_stream from the
    # API key + override). Threads through to DelegatingRunner._select.
    if getattr(body, "pool", None):
        prompt_kwargs["pool"] = body.pool

    # Per-request allocation triggers (4-bit / MoE / alloc mode / budgets) —
    # ride prompt_kwargs to the relay, which whitelists + merges them over the
    # designation spill and strips the key from the worker wire.
    if getattr(body, "alloc", None):
        prompt_kwargs["alloc"] = body.alloc

    if body.temperature is not None:
        prompt_kwargs["temperature"] = body.temperature

    if body.top_p is not None:
        prompt_kwargs["top_p"] = body.top_p

    if body.do_sample is not None:
        prompt_kwargs["do_sample"] = body.do_sample

    if body.task:
        prompt_kwargs["task"] = body.task

    if body.messages:
        prompt_kwargs["messages"] = messages_to_dicts(body.messages)
    else:
        prompt_kwargs["prompt"] = body.prompt

    if body.file:
        prompt_kwargs["file"] = body.file
    if body.images:
        prompt_kwargs["images"] = body.images
    # Stable id the engine threads through every continuation pass; also lets
    # the browser correlate the stream. Minted here when the client didn't send
    # one, so EVERY chat is registered in the shared job store and cancellable —
    # an untracked stream can't be stopped.
    rid = body.request_id or uuid.uuid4().hex
    prompt_kwargs["request_id"] = rid

    logger.info("prompt_kwargs == %s", prompt_kwargs)

    # Register this request in the shared job store (F5): pending -> streaming
    # on first token, terminal on stream end/error/disconnect via finally. The
    # same record backs the console queue view, /llm/jobs, and the cancel
    # plane. Best-effort: job bookkeeping must never break a chat.
    try:
        _name = None
        try:
            _cfg = get_model_config(body.model_key) if body.model_key else None
            _name = getattr(_cfg, "name", None) or body.model_key
        except Exception:
            _name = body.model_key
        _existing = job_store.get(rid)
        if _existing is None or _existing.terminal:
            job_store.create(body.model_key or "", id=rid, kind="chat",
                             transport=body.transport or "web",
                             channel=body.channel,
                             principal=body.principal,
                             model_name=_name)
    except Exception:
        pass

    # A real cancel path (F1.3): the job carries a handle that sets this
    # event on the shared runtime loop; hugpy_engine.stream_query honors it
    # between and during passes. Anything — the cancel route, a bus control
    # message — stops this stream through job_store.cancel(rid).
    cancel_event = asyncio.Event()
    try:
        job_store.attach_cancel(
            rid, lambda: async_runtime.call_soon_threadsafe(cancel_event.set))
    except Exception:
        pass

    import time as _time
    _t0 = _time.monotonic()
    try:
        async for event in stream_query(cancel_event=cancel_event,
                                        **prompt_kwargs):
            etype = getattr(event, "type", None)
            if etype == "token":
                job_store.on_output(rid)
            elif etype == "status":
                # A HELD cold call (t36) surfaces its dispatch worker + load
                # progress as status events. Reflect them into the job so the row
                # reads honestly (stage `awaiting-load`, progress) AND its
                # progressed_at stays fed — a placed, loading call must never be
                # retired by the pending-orphan sweep while its load churns.
                _feed_job_from_status(rid, event)
            elif etype == "error":
                # Surface an honest failure ON the job (first-terminal-wins; the
                # finally's finish() then no-ops). Covers the held-load-gave-up
                # and honest-refusal cases so /llm/jobs shows failed, not done.
                try:
                    job_store.finish(rid, error=getattr(event, "message", "error"))
                except Exception:
                    pass
            yield event_to_sse(event)
    except Exception as exc:
        logger.exception("stream_events failed")
        try:
            job_store.finish(rid, error=exc)
        except Exception:
            pass
        _lf = None
        try:
            from hugpy_engine.serve.load_failure import load_failure_of
            _lf = load_failure_of(exc)
        except Exception:  # noqa: BLE001
            pass
        yield sse_event({"type": "error",
                         "message": _friendly_stream_error(
                             exc, model_key=body.model_key, request_id=rid,
                             elapsed_s=_time.monotonic() - _t0),
                         "error_class": type(exc).__name__, "request_id": rid,
                         "model": body.model_key, "load_failure": _lf})
    finally:
        # Resolves to done, or cancelled if a cancel was requested; no-op when
        # the except above already marked it failed.
        try:
            job_store.finish(rid)
        except Exception:
            pass


def _friendly_stream_error(exc: Exception, *, model_key=None, request_id=None,
                           elapsed_s=None) -> str:
    """The error line shown for a failed call, built from what happened: the
    model, the request id, the seconds spent, and the real exception chain
    (type + text of each distinct link) — never a canned remedy."""
    parts, seen, e = [], set(), exc
    while e is not None and id(e) not in seen and len(parts) < 4:
        seen.add(id(e))
        text = f"{type(e).__name__}: {e}".strip()
        if text not in parts:
            parts.append(text)
        e = e.__cause__ or e.__context__
    ctx = [f"model={model_key}" if model_key else None,
           f"request={request_id}" if request_id else None,
           f"after {elapsed_s:.1f}s" if elapsed_s is not None else None]
    ctx = ", ".join(c for c in ctx if c)
    return " <- ".join(parts) + (f" ({ctx})" if ctx else "")


def _request_bearer():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return (request.args.get("api_key") or "").strip() or None


def _resolve_request_pool(explicit):
    """Effective dedicated pool: the API key's bound pool is the default; an
    explicit ``pool`` overrides it only when the key permits (the key may use its
    own pool; a keyless/open request may set any). Must run in request context."""
    try:
        from hugpy_server.app.functions.imports.utils.api_keys import pool_for_key
    except Exception:
        pool_for_key = lambda _t: None
    key_pool = pool_for_key(_request_bearer())
    explicit = (explicit or "").strip() or None
    if explicit:
        if key_pool is None or explicit == key_pool:
            return explicit
        return key_pool
    return key_pool


def _resolve_request_principal():
    """Attribution (F2): who is asking, from whatever credential they sent.
    Must run in request context — the SSE generator has none. Best-effort;
    a chat never fails over attribution."""
    try:
        from hugpy_control.principals import principal_store
        bearer = _request_bearer()
        if bearer and bearer.startswith("hpp_"):
            p = principal_store.resolve_token(bearer)
            if p is not None:
                return p.id
        if bearer:
            from hugpy_server.app.functions.imports.utils.api_keys import key_id_for_token
            kid = key_id_for_token(bearer)
            if kid:
                return f"apikey:{kid}"
        from hugpy_server.app.operator_auth import operator_authenticated
        if operator_authenticated():
            return "operator"
    except Exception:
        pass
    return None


def chat_stream(mimetype=None, headers=None, **kwargs):
    logger.info(kwargs)
    body = ChatBody(**kwargs)
    # DRAFT-MODEL-GATE-20260910: refuse a draft head up front with a readable 400 (the console
    # shows the response text) rather than a stream that ends in an engine crash.
    try:
        from hugpy_engine.draft_models import draft_model_reason
        _draft = draft_model_reason(getattr(body, "model_key", None))
    except Exception:  # noqa: BLE001
        _draft = None
    if _draft:
        return Response(_draft, status=400, mimetype="text/plain")
    # Resolve the dedicated pool + principal HERE — the SSE generator runs on
    # the shared async runtime loop, which has no flask.request context. The
    # principal always overwrites whatever the client sent (never trusted).
    eff_pool = _resolve_request_pool(getattr(body, "pool", None))
    updates = {"principal": _resolve_request_principal()}
    if eff_pool:
        updates["pool"] = eff_pool
    body = body.model_copy(update=updates)

    return Response(
        stream_with_context(chat_iter_sync(stream_events(body), heartbeat=SSE_KEEPALIVE)),
        mimetype=mimetype or "text/event-stream",
        headers=headers or {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
        direct_passthrough=True,
    )
