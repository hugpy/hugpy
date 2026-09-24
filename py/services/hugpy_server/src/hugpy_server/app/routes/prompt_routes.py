"""Generic execute_prompt passthrough — the whole dispatch surface over HTTP.

/chat/stream covers streamed text generation; this route covers everything
else with one verb. The JSON body is the same kwargs surface execute_prompt
takes in-process (task, model_key, prompt/messages/text/texts/other_texts,
file/image_b64, generation params, task-specific params), and the response is
the TaskResult dumped to JSON — so an HTTP client can drive every registered
task category (embeddings, similarity, summarize-with-presets, whisper with
language/translate, vision, text-to-image) without a dedicated route each.

    POST /prompt        {"task": "sentence-similarity", "texts": [...], "other_texts": [...]}
    GET  /prompt/tasks  -> {"tasks": [...], "defaults": {task: model_key}}

Explicit values win; anything omitted falls to resolve()'s default chain
(model_key > task > media-type-of-file > chat default).
"""
from __future__ import annotations


from abstract_flask import get_bp
from flask import jsonify, request

prompt_bp, logger = get_bp("prompt_bp", __name__)


from hugpy_platform.async_runtime import await_sync as _await_sync


def _capacity_refusal(exc):
    """A 503 + Retry-After when ``exc`` is the cold-hold admission cap, else None.

    The cap is central declining to START another concurrent model load so the
    rest of the site keeps its request slots — an honest, retryable answer, never
    a 500. Guarded so an older core without the cap changes nothing.
    """
    try:
        from hugpy_engine.resolvers.remote import ColdHoldCapacityError
    except Exception:                       # noqa: BLE001 — older core
        return None
    if not isinstance(exc, ColdHoldCapacityError):
        return None
    body = dict(exc.as_error()["error"])
    return (jsonify({"ok": False, "error": exc.stream_message(), **body}), 503,
            {"Retry-After": str(exc.retry_after_s)})


def _client_gone(exc) -> bool:
    """Did this request end because the CALLER disconnected (not a failure)?"""
    try:
        from hugpy_platform.client_liveness import ClientGone
    except Exception:                       # noqa: BLE001
        return False
    return isinstance(exc, ClientGone)


from hugpy_platform.results import result_payload as _result_payload


@prompt_bp.route("/prompt", methods=["POST"])
def prompt_execute():
    body = request.get_json(silent=True) or {}
    # Underscore-prefixed keys are internal routing controls (_force_local);
    # never client-settable.
    kwargs = {
        k: v for k, v in body.items()
        if not k.startswith("_") and v is not None
    }
    if not kwargs:
        return jsonify({"ok": False, "error": "empty request body"}), 400

    # Resolve the dedicated worker pool (API key's bound pool + explicit override)
    # in request context, then thread it so non-chat tasks route to it too.
    try:
        from hugpy_server.app.functions.chat.streaming import _resolve_request_pool
        eff_pool = _resolve_request_pool(kwargs.get("pool"))
        if eff_pool:
            kwargs["pool"] = eff_pool
        else:
            kwargs.pop("pool", None)
    except Exception:
        pass

    from hugpy_engine.dispatch.dispatch import execute_prompt

    try:
        result = _await_sync(execute_prompt(**kwargs))
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        # resolve()/builder validation errors — the caller's to fix.
        return jsonify({"ok": False, "error": str(exc).strip("'\"")}), 400
    except Exception as exc:
        # Cold-hold ADMISSION CAP: central is already holding its maximum number
        # of concurrent model loads. Not a fault and not a traceback — an honest
        # 503 + Retry-After so the caller (very often a script iterating models,
        # which is exactly what took the site down on 2026-07-27) backs off
        # instead of parking another of the site's 24 request slots in a hold.
        cap = _capacity_refusal(exc)
        if cap is not None:
            return cap
        # The caller hung up while we were working; the work was cancelled and
        # the slot released. There is nobody to answer — log it as the routine
        # event it is (never a 500 traceback) and close.
        if _client_gone(exc):
            logger.info("prompt abandoned: client disconnected")
            return jsonify({"ok": False, "error": "client disconnected"}), 499
        logger.exception("execute_prompt failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    payload = _result_payload(result)
    payload.setdefault("ok", not payload.get("error"))
    return jsonify(payload)


@prompt_bp.route("/prompt/tasks", methods=["GET"])
def prompt_tasks():
    from hugpy_engine.resolvers.categories.frameworks import KNOWN_TASKS_REGISTRY
    from hugpy_engine.categories import TASK_DEFAULTS
    return jsonify({
        "tasks": sorted(KNOWN_TASKS_REGISTRY),
        "defaults": dict(TASK_DEFAULTS),
    })
