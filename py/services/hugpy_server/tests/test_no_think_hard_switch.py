"""t74 — the HARD no-think switch: chat_template_kwargs / logit_bias, reachable.

Some models ignore the /no_think soft directive outright (wazimondo~Qwen3.6-
35B-A3B-Uncensored-Wasserstein-GGUF burns its whole budget thinking anyway).
The fix is the Qwen3 chat-template idiom — ``chat_template_kwargs``
``{"enable_thinking": false}`` emits a PRE-CLOSED <think> block, enforced by
the TEMPLATE rather than model obedience — plus an optional per-request
``logit_bias`` (<think>-token-ban fallback). This file covers the whole seam:

  * the wire: ChatRequest carries both keys, OMITS them from model_dump()
    when unset (released workers are extra="forbid" — the wire must stay
    byte-identical for everyone not using the feature);
  * the intake: _completion_kwargs (/v1) and _build_chat_request forward them;
  * the VERSION GATE (alloc_modes.CHAT_EXTRAS_MIN_PKG_VERSION, the
    NEW_SPILL_KEYS pattern): _worker_payload ships the keys verbatim to a
    worker that honors them and STRIPS them — never a silent no-op, always a
    logged downgrade — for one that predates them;
  * the runners: base_runner._engine_extras reads them off the request, the
    HTTP runner puts them verbatim on the llama-server body, the in-process
    runner honors logit_bias when its llama_cpp build can and names what it
    can't.

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_no_think_hard_switch.py -q
    venv/bin/python tests/test_no_think_hard_switch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_engine.alloc_modes import (
    CHAT_EXTRAS_MIN_PKG_VERSION,
    CHAT_EXTRAS_WIRE_KEYS,
    gate_chat_extras_for_worker,
    worker_honors_chat_extras,
)

CTK = {"enable_thinking": False}
LB = {"151667": -100.0}


# --------------------------------------------------------------------------- #
# the version gate (NEW_SPILL_KEYS pattern)
# --------------------------------------------------------------------------- #

def test_gate_keys_are_exactly_the_two_wire_keys():
    assert CHAT_EXTRAS_WIRE_KEYS == {"chat_template_kwargs", "logit_bias"}


def test_worker_honors_chat_extras_matrix():
    assert worker_honors_chat_extras(CHAT_EXTRAS_MIN_PKG_VERSION)
    assert worker_honors_chat_extras("0.2.0")
    assert not worker_honors_chat_extras("0.1.228")   # the released cut
    assert not worker_honors_chat_extras(None)        # unknown -> fail SAFE
    assert not worker_honors_chat_extras("garbage")


def test_gate_passes_through_for_a_worker_that_honors():
    p = {"task": "text-generation", "chat_template_kwargs": CTK,
         "logit_bias": LB}
    out, note = gate_chat_extras_for_worker(p, CHAT_EXTRAS_MIN_PKG_VERSION, "w")
    assert out["chat_template_kwargs"] == CTK
    assert out["logit_bias"] == LB
    assert note is None


def test_gate_strips_both_keys_and_says_so_for_an_old_worker():
    p = {"task": "text-generation", "chat_template_kwargs": CTK,
         "logit_bias": LB, "messages": [{"role": "user", "content": "q"}]}
    out, note = gate_chat_extras_for_worker(p, "0.1.228", "oldbox")
    assert "chat_template_kwargs" not in out
    assert "logit_bias" not in out
    assert out["messages"] == p["messages"]           # the rest untouched
    assert note and "oldbox" in note and CHAT_EXTRAS_MIN_PKG_VERSION in note
    assert "chat_template_kwargs" in note and "logit_bias" in note


def test_gate_is_a_no_op_without_the_keys_regardless_of_version():
    p = {"task": "text-generation", "messages": []}
    for ver in (None, "0.1.150", "0.2.0"):
        out, note = gate_chat_extras_for_worker(p, ver, "w")
        assert out == p and note is None


# --------------------------------------------------------------------------- #
# the wire: ChatRequest carries the keys, and OMITS them when unset
# --------------------------------------------------------------------------- #

def test_chat_request_omits_unset_extras_from_the_dump():
    # Released workers re-validate the relay with extra="forbid" — a None key
    # on the wire is the 2026-07-17 incident class. Unset MUST mean absent.
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    d = ChatRequest(messages=[{"role": "user", "content": "hi"}]).model_dump()
    assert "chat_template_kwargs" not in d
    assert "logit_bias" not in d


def test_chat_request_carries_set_extras_verbatim():
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    d = ChatRequest(messages=[{"role": "user", "content": "hi"}],
                    chat_template_kwargs=CTK, logit_bias=LB).model_dump()
    assert d["chat_template_kwargs"] == CTK
    assert d["logit_bias"] == LB
    # the messages wire is untouched by the serializer
    assert d["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# intake: /v1 translation + the chat builder forward both keys
# --------------------------------------------------------------------------- #

def test_v1_completion_kwargs_forwards_the_extras():
    from hugpy_server.app.routes.v1_helpers import _completion_kwargs
    kw = _completion_kwargs({
        "messages": [{"role": "user", "content": "q"}],
        "chat_template_kwargs": CTK, "logit_bias": LB,
    })
    assert kw["chat_template_kwargs"] == CTK
    assert kw["logit_bias"] == LB


def test_v1_completion_kwargs_drops_non_dict_and_empty_extras():
    from hugpy_server.app.routes.v1_helpers import _completion_kwargs
    kw = _completion_kwargs({
        "messages": [{"role": "user", "content": "q"}],
        "chat_template_kwargs": "enable_thinking=false",   # wrong shape
        "logit_bias": {},                                  # empty
    })
    assert "chat_template_kwargs" not in kw
    assert "logit_bias" not in kw


def test_chat_builder_forwards_the_extras_onto_the_request():
    from hugpy_engine.resolvers.categories.builders import _build_chat_request
    req = _build_chat_request(
        {"messages": [{"role": "user", "content": "q"}],
         "chat_template_kwargs": CTK, "logit_bias": LB}, "m")
    assert req.chat_template_kwargs == CTK
    assert req.logit_bias == LB


def test_chat_builder_without_extras_builds_a_clean_request():
    from hugpy_engine.resolvers.categories.builders import _build_chat_request
    req = _build_chat_request(
        {"messages": [{"role": "user", "content": "q"}]}, "m")
    assert req.chat_template_kwargs is None
    assert req.logit_bias is None


# --------------------------------------------------------------------------- #
# the relay: _worker_payload version-gates per worker
# --------------------------------------------------------------------------- #

def _relay_payload_for(worker):
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    from hugpy_engine.resolvers.remote import _worker_payload
    req = ChatRequest(messages=[{"role": "user", "content": "q /no_think"}],
                      chat_template_kwargs=CTK, logit_bias=LB)
    return _worker_payload("text-generation", req, "m",
                           (worker or {}).get("id"), worker=worker)


def test_relay_ships_extras_to_a_worker_that_honors_them():
    payload = _relay_payload_for({"id": "w1", "name": "newbox",
                                  "pkg_version": CHAT_EXTRAS_MIN_PKG_VERSION})
    assert payload["chat_template_kwargs"] == CTK
    assert payload["logit_bias"] == LB


def test_relay_strips_extras_for_a_worker_that_predates_them():
    payload = _relay_payload_for({"id": "w2", "name": "oldbox",
                                  "pkg_version": "0.1.228"})
    assert "chat_template_kwargs" not in payload
    assert "logit_bias" not in payload
    # the soft switch still rides the messages — degrade, never break
    assert "/no_think" in payload["messages"][-1]["content"]


def test_relay_strips_extras_when_no_worker_row_is_supplied():
    # An older internal call shape can't prove the version — fail SAFE.
    payload = _relay_payload_for(None)
    assert "chat_template_kwargs" not in payload
    assert "logit_bias" not in payload


# --------------------------------------------------------------------------- #
# the runners
# --------------------------------------------------------------------------- #

def test_engine_extras_reads_the_request_fields():
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    from hugpy_engine.llama.runners.src.base_runner import LlamaCppBaseRunner
    req = ChatRequest(messages=[{"role": "user", "content": "q"}],
                      chat_template_kwargs=CTK, logit_bias=LB)
    assert LlamaCppBaseRunner._engine_extras(req) == {
        "chat_template_kwargs": CTK, "logit_bias": LB}
    plain = ChatRequest(messages=[{"role": "user", "content": "q"}])
    assert LlamaCppBaseRunner._engine_extras(plain) == {}
    # getattr-tolerant: an older/foreign request shape yields {}
    assert LlamaCppBaseRunner._engine_extras(object()) == {}


def test_http_runner_puts_extras_on_the_llama_server_body(monkeypatch):
    """The slot path: extras land verbatim on the /v1/chat/completions body."""
    from hugpy_engine.llama.runners.src import ccp_runner

    seen = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            seen["url"] = url
            seen["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(ccp_runner.httpx, "Client", _FakeClient)

    runner = ccp_runner.LlamaCppRunner("m", base_url="http://x")
    msgs = [{"role": "user", "content": "q"}]
    text, finish = runner._chat_complete(
        msgs, 64, 0.0, 1.0, None,
        extras={"chat_template_kwargs": CTK, "logit_bias": LB})
    assert text == "ok" and finish == "stop"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["json"]["chat_template_kwargs"] == CTK
    assert seen["json"]["logit_bias"] == LB
    assert seen["json"]["messages"] == msgs

    # without extras the body is byte-identical to before
    seen.clear()
    runner._chat_complete(msgs, 64, 0.0, 1.0, None)
    assert "chat_template_kwargs" not in seen["json"]
    assert "logit_bias" not in seen["json"]


def test_in_process_runner_honors_what_llama_cpp_can():
    """logit_bias passes when create_chat_completion accepts it;
    chat_template_kwargs is named-and-skipped (no in-process analogue)."""
    from hugpy_engine.llama.runners.src.python_runner import LlamaCppPythonRunner

    runner = object.__new__(LlamaCppPythonRunner)
    runner.model_key = "m"

    class _AcceptingLlm:
        def create_chat_completion(self, messages=None, logit_bias=None, **kw):
            pass

    class _OldLlm:
        def create_chat_completion(self, messages=None, **kw):
            pass

    runner.llm = _AcceptingLlm()
    kw = runner._llm_extras_kwargs({"chat_template_kwargs": CTK,
                                    "logit_bias": LB})
    assert kw == {"logit_bias": LB}     # ctk has no analogue; lb honored

    runner.llm = _OldLlm()
    kw = runner._llm_extras_kwargs({"logit_bias": LB})
    assert kw == {}                     # named in the log, never a TypeError

    assert runner._llm_extras_kwargs(None) == {}
    assert runner._llm_extras_kwargs({}) == {}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
