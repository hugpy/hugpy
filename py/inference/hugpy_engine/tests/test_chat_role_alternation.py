"""ROLE-ALTERNATION NORMALISATION + honest request-shape diagnostics.

The operator had NEVER received a reply from
``DavidAU~MN-GRAND-23.5B-Gutenberg-UNCENSORED-V2-GLM4.7-Thinking`` on ae. Two
defects stacked into an unrecoverable loop:

 1. POISON LOOP. That model's chat template demands strict
    user/assistant/user/assistant alternation. The chat UI posts the WHOLE
    history; a failed turn appends no assistant reply, so the next attempt posts
    ``[user, user]``, then ``[user, user, user]`` — every retry more malformed
    than the last. Once it failed once it could never succeed.

 2. MISATTRIBUTING DIAGNOSTIC. The TemplateError is a request-shape error, but
    it was classified TRANSIENT, held+retried for the full cold-hold ceiling
    (the reproduced HTTP 000 hang — the client gave up before central did), and
    then reported as "the model may be too large for the box or the load
    stalled". Hours of VRAM investigation for a malformed message list.

This test is central-side only (no worker / no GPU / no model):

  * ``merge_consecutive_messages`` / ``ChatRequest`` — the choke point. Adjacent
    same-role turns merge, content-preserving; already-alternating histories are
    byte-identical; a leading system message is untouched; tool-loop turns are
    barriers.
  * The relay payload (``model_dump``) is normalised at CENTRAL, so a released
    worker on the frozen role+content wire receives a well-formed history with
    no wire change.
  * ``remote._is_request_shape_error`` / ``_request_shape_message`` and the
    DelegatingRunner stream()/run() paths — fail FAST, once, naming the real
    fault, never blaming box size.

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_chat_role_alternation.py -q
    venv/bin/python tests/test_chat_role_alternation.py
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# In-process only — no cross-process comms DB side effects during tests.
os.environ.setdefault("HUGPY_COMMS_DB", "off")

ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# The exact worker the mocked _select always returns (mirrors test_cold_hold).
WORKER = {"id": "w1", "name": "ae", "url": "http://w1:9100"}

# The literal error the operator saw, verbatim.
TEMPLATE_ERR = ("TemplateError: After the optional system message, conversation "
                "roles must alternate user/assistant/user/assistant/...")


def _dumps(req):
    return [m.model_dump() for m in req.messages]


# ---------------------------------------------------------------------------
# 1. the merge primitive
# ---------------------------------------------------------------------------
def _merge_checks(schemas):
    merge = schemas.merge_consecutive_messages

    # -- consecutive user turns are merged, content-preserving ---------------
    out = merge([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    check("two consecutive user turns merge into one",
          out == [{"role": "user", "content": "a\n\nb"}])

    # -- the operator's transcript shape: four user turns, no assistant ------
    four = [{"role": "user", "content": f"turn{i}"} for i in range(1, 5)]
    out = merge(four)
    check("the 4-user-turn poison-loop shape collapses to ONE user turn",
          out == [{"role": "user", "content": "turn1\n\nturn2\n\nturn3\n\nturn4"}])
    check("merging never drops content", all(f"turn{i}" in out[0]["content"]
                                             for i in range(1, 5)))
    check("merging never reorders content",
          out[0]["content"].index("turn1") < out[0]["content"].index("turn4"))

    # -- already-alternating: byte-identical, SAME object --------------------
    alternating = [{"role": "system", "content": "S"},
                   {"role": "user", "content": "a"},
                   {"role": "assistant", "content": "x"},
                   {"role": "user", "content": "b"}]
    before = copy.deepcopy(alternating)
    out = merge(alternating)
    check("an already-alternating history is returned as the SAME list object",
          out is alternating)
    check("an already-alternating history is byte-identical", alternating == before)

    # -- a leading system message is preserved as-is -------------------------
    out = merge([{"role": "system", "content": "S"},
                 {"role": "user", "content": "a"},
                 {"role": "user", "content": "b"}])
    check("a leading system message is never merged into the user turn",
          out[0] == {"role": "system", "content": "S"} and len(out) == 2)
    check("the user run after the system message is merged",
          out[1] == {"role": "user", "content": "a\n\nb"})

    # -- assistant runs merge too (double-send from a continuation loop) -----
    out = merge([{"role": "user", "content": "a"},
                 {"role": "assistant", "content": "x"},
                 {"role": "assistant", "content": "y"},
                 {"role": "user", "content": "b"}])
    check("consecutive assistant turns merge as well",
          out == [{"role": "user", "content": "a"},
                  {"role": "assistant", "content": "x\n\ny"},
                  {"role": "user", "content": "b"}])

    # -- empty contents contribute nothing (no stray blank lines) ------------
    out = merge([{"role": "user", "content": ""}, {"role": "user", "content": "b"}])
    check("an empty content contributes no leading blank line",
          out == [{"role": "user", "content": "b"}])
    out = merge([{"role": "user", "content": None}, {"role": "user", "content": None}])
    check("all-empty merge yields empty content, not 'None'",
          out == [{"role": "user", "content": ""}])

    # -- degenerate inputs ---------------------------------------------------
    check("a single message is untouched", merge([{"role": "user", "content": "a"}])
          == [{"role": "user", "content": "a"}])
    check("an empty list is untouched", merge([]) == [])
    check("a non-list is passed straight through", merge("hi") == "hi")


# ---------------------------------------------------------------------------
# 2. tool-loop turns are barriers — the working /v1 loop must not move
# ---------------------------------------------------------------------------
_ECHO = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": '{"city": "Berlin"}'}}],
}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "72F, sunny"}


def _tool_checks(schemas):
    merge = schemas.merge_consecutive_messages

    loop = [{"role": "user", "content": "weather in Berlin?"}, _ECHO, _RESULT]
    before = copy.deepcopy(loop)
    out = merge(loop)
    check("the OpenAI tool loop (user/assistant+tool_calls/tool) is untouched",
          out is loop and loop == before)

    # role="tool" is a BARRIER: it never merges, and it never merges ACROSS.
    out = merge([{"role": "user", "content": "a"}, _RESULT,
                 {"role": "user", "content": "b"}])
    check("a role='tool' result never merges with a neighbouring user turn",
          len(out) == 3 and out[1] == _RESULT)

    # Two tool results in a row stay two messages (never folded together).
    r2 = {"role": "tool", "tool_call_id": "call_2", "content": "rain"}
    out = merge([_RESULT, r2])
    check("two consecutive tool results are never folded together",
          len(out) == 2 and out[0] is _RESULT and out[1] is r2)

    # An assistant echo carrying tool_calls is a barrier on the assistant side.
    out = merge([_ECHO, {"role": "assistant", "content": "prose"}])
    check("an assistant tool_calls echo is never merged with plain assistant text",
          len(out) == 2 and out[0] is _ECHO)

    # ...and the full request still validates, unchanged.
    req = schemas.ChatRequest(messages=[{"role": "user", "content": "weather?"},
                                        _ECHO, _RESULT])
    check("ChatRequest still validates the 3-message tool loop unchanged",
          len(req.messages) == 3 and req.messages[1].role == "assistant"
          and req.messages[2].role == "tool")
    check("tool_calls survive validation intact",
          req.messages[1].tool_calls[0]["function"]["name"] == "get_weather")
    check("the tool result's tool_call_id survives",
          req.messages[2].tool_call_id == "call_1")


# ---------------------------------------------------------------------------
# 3. the choke point — ChatRequest, and therefore the relay wire
# ---------------------------------------------------------------------------
def _chat_request_checks(schemas):
    CR = schemas.ChatRequest

    req = CR(messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    check("ChatRequest normalises consecutive user turns at validation",
          _dumps(req) == [{"role": "user", "content": "a\n\nb"}])

    # The operator's exact shape, through the schema.
    req = CR(messages=[{"role": "user", "content": f"t{i}"} for i in range(4)])
    check("ChatRequest collapses the 4-user-turn transcript to one alternating turn",
          _dumps(req) == [{"role": "user", "content": "t0\n\nt1\n\nt2\n\nt3"}])

    # WIRE: this is what _worker_payload() relays to a released 0.1.217 worker.
    dumped = req.model_dump()["messages"]
    check("the RELAY payload is normalised at central (no worker change needed)",
          dumped == [{"role": "user", "content": "t0\n\nt1\n\nt2\n\nt3"}])
    check("the relay payload stays the frozen role+content wire (extra=forbid safe)",
          all(set(m) == {"role", "content"} for m in dumped))

    # Already-alternating: byte-identical dump.
    good = [{"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "b"}]
    check("an already-alternating history dumps byte-identically",
          CR(messages=copy.deepcopy(good)).model_dump()["messages"] == good)

    # System message preserved through the schema.
    req = CR(messages=[{"role": "system", "content": "You are terse."},
                       {"role": "user", "content": "a"},
                       {"role": "user", "content": "b"}])
    check("the system message survives normalisation unchanged",
          _dumps(req)[0] == {"role": "system", "content": "You are terse."})
    check("normalisation leaves an alternating [system, user] history",
          [m["role"] for m in _dumps(req)] == ["system", "user"])

    # Multimodal flattening still runs FIRST (merge sees flat text).
    req = CR(messages=[
        {"role": "user", "content": [{"type": "text", "text": "look"},
                                     {"type": "image_url",
                                      "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "user", "content": "and this"},
    ])
    check("multimodal parts are still hoisted to images",
          req.images == ["data:image/png;base64,AA"])
    check("multimodal text is flattened THEN merged with the next user turn",
          _dumps(req) == [{"role": "user", "content": "look\n\nand this"}])

    # A bare-string prompt still works (field validator path).
    check("a bare string prompt still builds a single user turn",
          len(CR(messages="hello").messages) == 1)

    # Idempotent: re-validating a normalised request changes nothing.
    once = CR(model_key="m", messages=[{"role": "user", "content": "a"},
                                       {"role": "user", "content": "b"}])
    twice = CR(**once.model_dump())
    check("normalisation is idempotent (central then worker re-validation)",
          twice.model_dump()["messages"] == once.model_dump()["messages"])


# ---------------------------------------------------------------------------
# 4. the /v1 seam: tool messages render, THEN alternation normalises
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. request-shape classification — FIX 2
# ---------------------------------------------------------------------------
def _classification_checks(remote):
    shape = remote._is_request_shape_error
    perm = remote._is_permanent_load_error

    check("the operator's verbatim TemplateError is a REQUEST-SHAPE error",
          shape(TEMPLATE_ERR))
    check("a bare jinja TemplateError is request-shape",
          shape("jinja2.exceptions.TemplateError: bad roles"))
    check("llama.cpp's 'failed to apply chat template' is request-shape",
          shape("Failed to apply chat template: roles must alternate"))
    check("'only user and assistant roles are supported' is request-shape",
          shape("Only user and assistant roles are supported!"))
    check("classifier reads .message off an exception too",
          shape(remote._LoadFailed(TEMPLATE_ERR)))

    # ...and it must NOT swallow the real capacity/load classes.
    check("won't fit is NOT request-shape", not shape("LoadRefusal: won't fit on GPU"))
    check("CUDA OOM is NOT request-shape", not shape("CUDA out of memory"))
    check("a server disconnect is NOT request-shape",
          not shape("RemoteProtocolError: Server disconnected"))
    check("won't fit is still a permanent LOAD error (unchanged)",
          perm("LoadRefusal: won't fit on GPU"))
    check("a template error is not smuggled into the permanent-LOAD class",
          not perm(TEMPLATE_ERR))

    # -- the message itself ---------------------------------------------------
    msg = remote._request_shape_message("MN-GRAND-23.5B", WORKER, TEMPLATE_ERR)
    low = msg.lower()
    check("the message names the REQUEST SHAPE as the fault", "request shape" in low)
    check("the message quotes the underlying engine error",
          "roles must alternate" in low)
    check("the message NEVER speculates about box size",
          "too large for the box" not in low and "too large" not in low)
    check("the message NEVER speculates about a stalled load",
          "load stalled" not in low and "did not finish loading" not in low)
    check("the message says it is not a capacity problem",
          "not a capacity" in low)

    # -- the cold-timeout wrapper stops misattributing -------------------------
    late = remote._cold_timeout_message("MN-GRAND-23.5B", WORKER, TEMPLATE_ERR)
    check("a template error reaching the timeout wrapper is NOT blamed on box size",
          "too large for the box" not in late.lower())
    check("a template error reaching the timeout wrapper reports request shape",
          "request shape" in late.lower())
    genuine = remote._cold_timeout_message("Big-GGUF", WORKER, "connection reset")
    # OPERATOR DIRECTIVE 2026-07-29 (specificity): "why is it unsure of what the
    # actual problem was? … this needs to be specific". The old wording hedged —
    # "the model may be too large for the box or the load stalled" — which is a
    # GUESS between two unrelated causes, and it buried the one actionable line
    # (the worker's own error). It is gone on purpose; asserting it here
    # contradicted this very file's "NEVER speculates" checks 15 lines above.
    # What must hold now is that the message is SPECIFIC and ATTRIBUTED.
    gl = genuine.lower()
    check("a genuine load timeout names the model and the worker",
          "big-gguf" in gl and "ae" in gl)
    check("a genuine load timeout quotes the WORKER'S OWN error as the cause "
          "(the one actionable line), not a guess",
          "connection reset" in gl and "cause" in gl)
    check("a genuine load timeout does NOT hedge between box size and a stall",
          "too large for the box" not in gl and "load stalled" not in gl
          and " may be " not in gl)


# ---------------------------------------------------------------------------
# 6. the relay: fail FAST + clean, never hang
# ---------------------------------------------------------------------------
def _text_framework(remote):
    for (fw, tk) in remote.FRAMEWORK_RUNNERS:
        if tk != "image-text-to-text":
            return fw, tk
    return next(iter(remote.FRAMEWORK_RUNNERS))


def _req(rid="rid-1"):
    return types.SimpleNamespace(
        request_id=rid, pool=None,
        reference_images=None, reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "a"},
                                         {"role": "user", "content": "b"}]},
    )


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def _etypes(evs):
    return [getattr(e, "type", None) for e in evs]


def _relay_checks(remote):
    from hugpy_engine.schemas.event_schemas import TokenEvent, DoneEvent

    fw, tk = _text_framework(remote)
    Runner = remote.make_delegating_runner(fw, tk)
    runner = Runner(types.SimpleNamespace(model_key="MN-GRAND-23.5B"))

    orig_select = remote._select
    orig_ws = remote._worker_stream
    orig_run = remote._worker_run_once
    orig_ls = remote._load_state_provider
    os.environ["HUGPY_CENTRAL_GATE"] = "off"
    os.environ["HUGPY_COLD_HOLD_POLL_S"] = "0.01"
    # A generous ceiling: if the fix regresses, the call HOLDS and the elapsed
    # assertion below fails instead of the test silently passing fast.
    os.environ["HUGPY_COLD_HOLD_MAX_S"] = "30"
    os.environ["HUGPY_COLD_HOLD_STALL_S"] = "30"
    os.environ.pop("HUGPY_LOCAL_FALLBACK", None)
    os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)
    remote._select = lambda mk, pool=None, task=None, **kw: (dict(WORKER), None)
    # Worker says the model is HEALTHY and loading is moving — exactly the live
    # condition that fed the stall clock and held the poison call for the full
    # ceiling. Nothing here may keep the request alive.
    remote.set_load_state_provider(
        lambda mk, wid, since=0.0: {"healthy": True, "in_progress": True,
                                    "progress": 0.9, "message": "loading"})

    try:
        # -- stream(): a TemplateError SSE error event ------------------------
        calls = {"n": 0}

        async def ws_template_error(worker, payload, rid):
            calls["n"] += 1
            from hugpy_engine.schemas.event_schemas import ErrorEvent
            yield ErrorEvent(request_id=rid, message=TEMPLATE_ERR)

        remote._worker_stream = ws_template_error
        t0 = time.time()
        evs = asyncio.run(_collect(runner.stream(_req("shape-1"))))
        elapsed = time.time() - t0
        check("stream(): a template error surfaces ONE error event",
              _etypes(evs).count("error") == 1 and "token" not in _etypes(evs))
        check("stream(): the malformed request is NOT held/retried (one attempt)",
              calls["n"] == 1)
        check("stream(): it fails FAST — no hang (this was the HTTP 000)",
              elapsed < 2.0)
        check("stream(): no awaiting-load status was emitted for a malformed request",
              not any(getattr(e, "stage", None) == "awaiting-load" for e in evs))
        err = [e for e in evs if getattr(e, "type", None) == "error"][0]
        check("stream(): the error names the request shape",
              "request shape" in err.message.lower())
        check("stream(): the error does NOT blame the box size",
              "too large for the box" not in err.message.lower())
        check("stream(): the underlying template error is preserved verbatim",
              "roles must alternate" in err.message)

        # -- stream(): a raised TemplateError (not an SSE event) --------------
        calls2 = {"n": 0}

        async def ws_raise_template(worker, payload, rid):
            calls2["n"] += 1
            raise RuntimeError(TEMPLATE_ERR)
            yield  # pragma: no cover — marks this a generator

        remote._worker_stream = ws_raise_template
        t0 = time.time()
        evs = asyncio.run(_collect(runner.stream(_req("shape-2"))))
        elapsed = time.time() - t0
        check("stream(): a RAISED template error also fails fast, once",
              calls2["n"] == 1 and elapsed < 2.0
              and _etypes(evs).count("error") == 1)
        check("stream(): the raised-path message is the request-shape line",
              "request shape" in evs[-1].message.lower()
              and "too large for the box" not in evs[-1].message.lower())

        # -- even with local fallback opted in, a malformed request fails fast:
        #    the local runner renders the SAME template and would fail the same
        #    way, so falling back only burns a load of a 23.5B model.
        os.environ["HUGPY_LOCAL_FALLBACK"] = "always"
        try:
            callsf = {"n": 0}

            async def ws_shape_fb(worker, payload, rid):
                callsf["n"] += 1
                from hugpy_engine.schemas.event_schemas import ErrorEvent
                yield ErrorEvent(request_id=rid, message=TEMPLATE_ERR)

            remote._worker_stream = ws_shape_fb
            evs = asyncio.run(_collect(runner.stream(_req("shape-fb"))))
            check("local fallback does NOT swallow a request-shape error",
                  callsf["n"] == 1 and _etypes(evs).count("error") == 1
                  and "request shape" in evs[-1].message.lower())
        finally:
            os.environ.pop("HUGPY_LOCAL_FALLBACK", None)

        # -- a TRANSIENT failure is still held (no regression) ----------------
        calls3 = {"n": 0}

        async def ws_transient(worker, payload, rid):
            calls3["n"] += 1
            if calls3["n"] <= 2:
                raise RuntimeError("RemoteProtocolError: Server disconnected")
            yield TokenEvent(request_id=rid, text="Hello")
            yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=1,
                            finish_reason="stop")

        remote._worker_stream = ws_transient
        evs = asyncio.run(_collect(runner.stream(_req("hold-1"))))
        check("a genuinely transient cold failure is STILL held and retried",
              calls3["n"] == 3 and "token" in _etypes(evs)
              and "error" not in _etypes(evs))

        # -- run(): the one-shot twin -----------------------------------------
        calls4 = {"n": 0}

        async def run_template_error(worker, payload, result_type, request_id,
                                     model_key):
            calls4["n"] += 1
            raise RuntimeError(TEMPLATE_ERR)

        remote._worker_run_once = run_template_error
        raised = None
        t0 = time.time()
        try:
            asyncio.run(runner.run(_req("shape-run")))
        except RuntimeError as exc:
            raised = str(exc)
        elapsed = time.time() - t0
        check("run(): a template error raises immediately (one attempt, no hold)",
              raised is not None and calls4["n"] == 1 and elapsed < 2.0)
        check("run(): the raised message names the request shape",
              "request shape" in raised.lower())
        check("run(): the raised message does NOT blame the box size",
              "too large for the box" not in raised.lower())

        # -- run(): a transient failure is still held (no regression) ---------
        calls5 = {"n": 0}

        async def run_transient(worker, payload, result_type, request_id, model_key):
            calls5["n"] += 1
            if calls5["n"] <= 2:
                raise RuntimeError("RemoteProtocolError: Server disconnected")
            return {"ok": True, "text": "done", "request_id": request_id,
                    "model_key": model_key}

        remote._worker_run_once = run_transient
        res = asyncio.run(runner.run(_req("hold-run")))
        check("run(): a transient cold failure is STILL held and retried",
              res.get("ok") is True and calls5["n"] == 3)
    finally:
        remote._select = orig_select
        remote._worker_stream = orig_ws
        remote._worker_run_once = orig_run
        remote.set_load_state_provider(orig_ls)
        for k in ("HUGPY_CENTRAL_GATE", "HUGPY_COLD_HOLD_POLL_S",
                  "HUGPY_COLD_HOLD_MAX_S", "HUGPY_COLD_HOLD_STALL_S"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# 7. /v1 answers 4xx, not 500 and not a hang
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def test_chat_role_alternation():
    global ok
    ok = 0
    schemas = importlib.import_module(
        "hugpy_engine.schemas.chat_schemas")
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    _merge_checks(schemas)
    _tool_checks(schemas)
    _chat_request_checks(schemas)
    _classification_checks(remote)
    _relay_checks(remote)
    print(f"\nall {ok} checks passed")


if __name__ == "__main__":
    test_chat_role_alternation()
