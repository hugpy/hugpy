"""KNOWN-GOOD CONTRACT — response_format reaches the engine or is refused.

2026-09-29: the B-reducer's ``response_format: {"type":"json_object"}`` was
dropped at /v1 intake and never reached llama-server. Now: ChatRequest carries
it, the runner forwards it (llama-server honours json_object/json_schema; the
in-process runner maps it or REJECTS with a clear error), and a worker whose
schema predates the field rejects at its intake — classified as a request-shape
verdict (terminal, never retried, never held). Deterministic, no network.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import types

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")
from hugpy_engine.llama.runners.src import base_runner as br  # noqa: E402
from hugpy_engine.llama.runners.src import ccp_runner, python_runner  # noqa: E402
from hugpy_engine.schemas.chat_schemas import ChatRequest  # noqa: E402

RF = {"type": "json_object"}


def test_chat_request_carries_response_format_and_engine_extras_forward_it():
    """INVARIANT: ChatRequest accepts response_format; _engine_extras includes
    it beside chat_template_kwargs/logit_bias; ccp_runner merges extras into
    the llama-server body (source pin). Established: 2026-09-29."""
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], response_format=RF)
    assert req.response_format == RF
    assert br.LlamaCppBaseRunner._engine_extras(req) == {"response_format": RF}
    assert br.LlamaCppBaseRunner._engine_extras(
        ChatRequest(messages=[{"role": "user", "content": "hi"}])) == {}
    assert "payload.update(extras)" in inspect.getsource(ccp_runner.LlamaCppRunner._iter_stream)


def test_in_process_runner_maps_response_format_or_rejects_clearly():
    """INVARIANT: with a llama_cpp build whose create_chat_completion accepts
    response_format, json_object / json_schema map to llama_cpp's
    {"type":"json_object"[,"schema"]}; a build without it raises a RuntimeError
    naming the reason — never a silent drop. Established: 2026-09-29."""
    def _accepting(messages=None, response_format=None, **kw):  # noqa: D401
        pass

    def _legacy(messages=None, **kw):
        pass
    R = python_runner.LlamaCppPythonRunner
    me = types.SimpleNamespace(model_key="m", llm=types.SimpleNamespace(create_chat_completion=_accepting))
    me._map_response_format = types.MethodType(R._map_response_format, me)
    m = R._map_response_format
    assert m(me, {"type": "json_object"}) == {"type": "json_object"}
    assert m(me, {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}) == \
        {"type": "json_object", "schema": {"type": "object"}}
    with pytest.raises(ValueError):
        m(me, {"type": "json_schema"})
    kw = python_runner.LlamaCppPythonRunner._llm_extras_kwargs(me, {"response_format": RF})
    assert kw == {"response_format": {"type": "json_object"}}
    legacy = types.SimpleNamespace(model_key="m", llm=types.SimpleNamespace(create_chat_completion=_legacy))
    legacy._map_response_format = types.MethodType(R._map_response_format, legacy)
    with pytest.raises(RuntimeError, match="does not accept response_format"):
        python_runner.LlamaCppPythonRunner._llm_extras_kwargs(legacy, {"response_format": RF})


def test_old_worker_intake_rejection_is_a_terminal_request_shape_verdict(monkeypatch):
    """INVARIANT: a worker whose frozen ChatRequest predates response_format
    answers 'Extra inputs are not permitted' at its intake; the relay classifies
    that as a request-shape failure and surfaces ONE clear ErrorEvent naming
    response_format — no hold, no retry, no size speculation.
    Established: 2026-09-29."""
    msg = ("ValidationError: 1 validation error for ChatRequest\nresponse_format\n"
           "  Extra inputs are not permitted [type=extra_forbidden]")
    assert remote._is_request_shape_error(msg)
    text = remote._request_shape_message("m", {"name": "old-worker"}, msg)
    assert "cannot honour response_format" in text and "NOT silently dropped" in text
    # Through the relay attempt: the worker's error frame -> _LoadFailed -> one error.
    monkeypatch.setenv("HUGPY_CENTRAL_GATE", "off")
    monkeypatch.setenv("HUGPY_COMMS_DB", "off")
    for k in ("HUGPY_LOCAL_FALLBACK", "HUGPY_NO_LOCAL_SERVING", "HUGPY_COLD_HOLD"):
        monkeypatch.delenv(k, raising=False)
    worker = {"id": "w1", "name": "old-worker", "url": "http://w1:9200", "slots": []}
    monkeypatch.setattr(remote, "_select", lambda mk, pool=None, task=None, **kw: (worker, None))
    remote.set_load_state_provider(None)
    calls = {"n": 0}

    async def ws(w, payload, rid):
        calls["n"] += 1
        from hugpy_engine.schemas.event_schemas import ErrorEvent
        yield ErrorEvent(request_id=rid, message=msg)
    monkeypatch.setattr(remote, "_worker_stream", ws)
    fw, tk = next(p for p in remote.FRAMEWORK_RUNNERS if p[1] != "image-text-to-text")
    runner = remote.make_delegating_runner(fw, tk)(types.SimpleNamespace(model_key="m"))
    req = types.SimpleNamespace(request_id="r", pool=None, reference_images=None,
                                reference_images_b64=None,
                                model_dump=lambda: {"messages": [{"role": "user", "content": "hi"}],
                                                    "response_format": RF})

    async def _run():
        return [e async for e in runner.stream(req)]
    evs = asyncio.run(_run())
    errs = [e for e in evs if getattr(e, "type", None) == "error"]
    assert len(errs) == 1 and "cannot honour response_format" in errs[0].message
    assert calls["n"] == 1
