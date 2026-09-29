"""KNOWN-GOOD CONTRACT — the relay's silence budgets and live cancel.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "call queue / relay", 2026-09-29
Coder-Next "stuck answering" incident). Source under test:
hugpy_engine/resolvers/remote.py — _pump_bounded, _relay_silence_budget_s,
_estimate_prompt_tokens and the _relay_attempt closure of
DelegatingRunner.stream().

Deterministic: the hold loop's clock is faked as in test_call_queue_contract;
the pump uses REAL asyncio with sub-second budgets (env knobs), and the worker
SSE stream is a fake async generator whose teardown is observable. No worker,
no network, no GPU.
"""
from __future__ import annotations

import asyncio
import importlib
import time as _real_time
import types

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")
from hugpy_engine.schemas.event_schemas import DoneEvent, StatusEvent, TokenEvent  # noqa: E402

MODEL = "known-good-model"
WORKER_ID = "w1"


def _req(rid="rid-1", chars=8):
    return types.SimpleNamespace(
        request_id=rid, pool=None,
        reference_images=None, reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "x" * chars}]},
    )


def _text_framework():
    for (fw, tk) in remote.FRAMEWORK_RUNNERS:
        if tk != "image-text-to-text":
            return fw, tk
    return next(iter(remote.FRAMEWORK_RUNNERS))


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def _types(evs):
    return [getattr(e, "type", None) for e in evs]


class _Clock:
    def __init__(self, t0=1_000_000.0):
        self.t = t0

    def time(self):
        return self.t

    async def sleep(self, s):
        self.t += float(s)
        await asyncio.sleep(0)


@pytest.fixture
def rig(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(remote, "time", types.SimpleNamespace(
        time=clock.time, monotonic=_real_time.monotonic, sleep=_real_time.sleep,
        perf_counter=_real_time.perf_counter))
    monkeypatch.setattr(remote, "asyncio", types.SimpleNamespace(
        sleep=clock.sleep, Semaphore=asyncio.Semaphore,
        get_running_loop=asyncio.get_running_loop, Event=asyncio.Event,
        Lock=asyncio.Lock, gather=asyncio.gather, CancelledError=asyncio.CancelledError))
    monkeypatch.setenv("HUGPY_CENTRAL_GATE", "off")
    monkeypatch.setenv("HUGPY_COLD_HOLD_POLL_S", "10")
    monkeypatch.setenv("HUGPY_COLD_HOLD_MAX_S", "50")
    monkeypatch.setenv("HUGPY_COLD_HOLD_STALL_S", "30")
    monkeypatch.setenv("HUGPY_COMMS_DB", "off")
    # Sub-second silence budgets so a stall test finishes in well under a second.
    monkeypatch.setenv("HUGPY_RELAY_FIRST_TOKEN_BASE_S", "0.25")
    monkeypatch.setenv("HUGPY_RELAY_PREFILL_MIN_TOK_S", "1000000")
    monkeypatch.setenv("HUGPY_RELAY_PREFILL_STALL_S", "0.4")
    monkeypatch.setenv("HUGPY_RELAY_STREAM_STALL_S", "0.3")
    for k in ("HUGPY_LOCAL_FALLBACK", "HUGPY_NO_LOCAL_SERVING", "HUGPY_COLD_HOLD"):
        monkeypatch.delenv(k, raising=False)
    worker = {"id": WORKER_ID, "name": "ae-worker", "url": "http://w1:9200", "slots": []}
    monkeypatch.setattr(remote, "_select",
                        lambda mk, pool=None, task=None, **kw: (worker, None))
    cancels = []

    async def _wc(w, rid):
        cancels.append((w.get("id"), rid))
    monkeypatch.setattr(remote, "_worker_cancel_best_effort", _wc)
    orig_ls = remote._load_state_provider
    remote.set_load_state_provider(None)
    remote._COLD_KICKING.clear()
    fw, tk = _text_framework()
    Runner = remote.make_delegating_runner(fw, tk)
    runner = Runner(types.SimpleNamespace(model_key=MODEL))
    state = {"attempts": 0, "closed": 0}
    yield types.SimpleNamespace(worker=worker, runner=runner, state=state,
                                cancels=cancels, monkeypatch=monkeypatch)
    remote.set_load_state_provider(orig_ls)
    remote._COLD_KICKING.clear()


def _install_stream(rig, script):
    """``script(rid)`` is an async generator of worker SSE dicts; a ``None``
    item means 'hang forever'. Teardown (our aclose / a cancelled __anext__)
    is counted in rig.state['closed']."""
    async def ws(worker, payload, rid):
        rig.state["attempts"] += 1
        try:
            async for item in script(rid):
                if item is None:
                    await asyncio.Event().wait()           # silence
                else:
                    yield item
        finally:
            rig.state["closed"] += 1
    rig.monkeypatch.setattr(remote, "_worker_stream", ws)


def _request_frame(rid):
    return StatusEvent(type="request", request_id=rid)


# ---------------------------------------------------------------------------
# (2) streaming stall: no token for N s after the first token -> terminal
# ---------------------------------------------------------------------------
def test_stream_stall_after_first_token_is_terminal_and_closes_the_relay(rig):
    """INVARIANT: once tokens have flowed, a silence longer than
    HUGPY_RELAY_STREAM_STALL_S ends the call with ONE ErrorEvent naming the
    stall, the upstream worker stream is torn down (its finally ran), and the
    request is never re-dispatched (one attempt). Established: 2026-09-29."""
    async def script(rid):
        yield _request_frame(rid)
        yield TokenEvent(request_id=rid, text="hel")
        yield None
    _install_stream(rig, script)
    t0 = _real_time.monotonic()
    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert _types(evs)[-1] == "error" and _types(evs).count("token") == 1
    assert "stalled" in evs[-1].message and "no token for 0s" in evs[-1].message
    assert "not retried" in evs[-1].message
    assert rig.state == {"attempts": 1, "closed": 1}
    assert 0.25 < _real_time.monotonic() - t0 < 3.0


# ---------------------------------------------------------------------------
# prefill: the pre-token budget scales with the prompt and is terminal
# ---------------------------------------------------------------------------
def test_prefill_silence_is_bounded_by_a_prompt_scaled_budget_and_not_retried(rig):
    """INVARIANT: with no token and no engine progress, the relay waits
    base + prompt_tokens_est / min_prefill_rate (never a flat 90 s) and then
    fails terminally — never re-dispatches (a re-dispatch would re-prefill the
    same prompt behind itself). The worker's first frame is announced as
    stage "prefill" with the ESTIMATE labelled as such. Established: 2026-09-29
    (29,451-token prompt, 104 s prefill, read as 'stalled' at 90 s)."""
    async def script(rid):
        yield _request_frame(rid)
        yield None
    _install_stream(rig, script)
    evs = asyncio.run(_collect(rig.runner.stream(_req(chars=4000))))
    assert _types(evs)[-1] == "error"
    assert "no first token or prefill progress within 0s for ~1000 prompt tokens" in evs[-1].message
    assert rig.state == {"attempts": 1, "closed": 1}
    pre = [e for e in evs if getattr(e, "stage", None) == "prefill"]
    assert len(pre) == 1 and pre[0].n_prompt_est == 1000 and "estimate" in pre[0].message
    assert pre[0].worker_name == "ae-worker"


def test_silence_budget_math(monkeypatch):
    """INVARIANT: budget = base + est/min_rate before progress; the prefill
    stall bound once engine progress was seen; the stream stall bound after
    the first token; all capped by the ceiling; defaults 120 / 25 tok/s /
    180 / 90 / 1800. Established: 2026-09-29."""
    for k in ("HUGPY_RELAY_FIRST_TOKEN_BASE_S", "HUGPY_RELAY_PREFILL_MIN_TOK_S",
              "HUGPY_RELAY_PREFILL_STALL_S", "HUGPY_RELAY_STREAM_STALL_S",
              "HUGPY_RELAY_SILENCE_CEILING_S"):
        monkeypatch.delenv(k, raising=False)
    b = remote._relay_silence_budget_s
    assert b(produced_tokens=False, progress_seen=False, prompt_tokens_est=0) == 120.0
    assert b(produced_tokens=False, progress_seen=False, prompt_tokens_est=29451) == pytest.approx(120 + 29451 / 25)
    assert b(produced_tokens=False, progress_seen=True, prompt_tokens_est=29451) == 180.0
    assert b(produced_tokens=True, progress_seen=True, prompt_tokens_est=29451) == 90.0
    assert b(produced_tokens=False, progress_seen=False, prompt_tokens_est=10**9) == 1800.0
    assert remote._estimate_prompt_tokens({"messages": [
        {"role": "system", "content": "a" * 400},
        {"role": "user", "content": [{"type": "text", "text": "b" * 400}]}]}) == 200
    assert remote._estimate_prompt_tokens(None) == 0


def test_engine_prefill_progress_resets_the_silence_clock(rig):
    """INVARIANT: a worker that forwards llama-server's prompt_progress as
    stage="prefill" status events keeps the call alive as long as progress
    keeps arriving (each event resets the clock; the pre-token budget no
    longer applies), the events reach the caller tagged with the worker, and
    the call completes normally. Established: 2026-09-29."""
    async def script(rid):
        yield _request_frame(rid)
        for past in (2048, 4096, 6144, 8192):
            await asyncio.sleep(0.15)          # 4 x 0.15 > base 0.25, < stall 0.4
            yield StatusEvent(request_id=rid, stage="prefill", n_prompt=8192,
                              n_past=past, progress=past / 8192)
        yield TokenEvent(request_id=rid, text="ok")
        yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1, finish_reason="stop")
    _install_stream(rig, script)
    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert _types(evs)[-2:] == ["token", "done"], _types(evs)
    pre = [e for e in evs if getattr(e, "stage", None) == "prefill"]
    # First the relay's labelled estimate (on the worker's first frame), then
    # the engine's real figures replace it.
    assert getattr(pre[0], "n_prompt_est", None) == 2 and "estimate" in pre[0].message
    assert [e.n_past for e in pre[1:]] == [2048, 4096, 6144, 8192]
    assert all(e.worker_name == "ae-worker" for e in pre)
    assert rig.state == {"attempts": 1, "closed": 1}


# ---------------------------------------------------------------------------
# (3)/(5) cancel during prefill: upstream closed NOW, worker told, no error
# ---------------------------------------------------------------------------
def test_cancel_during_prefill_closes_upstream_immediately_and_tells_the_worker(rig):
    """INVARIANT: cancel_event fired while the worker stream is silent
    (prefill) ends the relay at once — the upstream stream's teardown runs,
    the worker is asked to abort the request (POST /infer/cancel/<id>), no
    error/token is emitted, and nothing is retried. Before 2026-09-29 the
    cancel was only read between hold iterations, so a cancel during a 100 s
    prefill did nothing until the first token. Established: 2026-09-29."""
    async def script(rid):
        yield _request_frame(rid)
        yield None
    _install_stream(rig, script)
    rig.monkeypatch.setenv("HUGPY_RELAY_FIRST_TOKEN_BASE_S", "30")

    async def _drive():
        cancel = asyncio.Event()
        seen = []

        async def _later():
            await asyncio.sleep(0.1)
            cancel.set()
        asyncio.get_running_loop().create_task(_later())
        async for ev in rig.runner.stream(_req("rid-cancel"), cancel_event=cancel):
            seen.append(ev)
        return seen

    t0 = _real_time.monotonic()
    seen = asyncio.run(_drive())
    assert _real_time.monotonic() - t0 < 3.0
    assert "error" not in _types(seen) and "token" not in _types(seen)
    assert rig.state == {"attempts": 1, "closed": 1}
    assert rig.cancels == [(WORKER_ID, "rid-cancel")]
    assert (WORKER_ID, MODEL) not in remote._COLD_KICKING


def test_client_disconnect_mid_stream_closes_upstream(rig):
    """INVARIANT: the consumer going away (aclose on the relay generator —
    what a GeneratorExit from a dropped SSE client cascades into) closes the
    upstream worker stream deterministically. Established: incident
    2026-09-25 (bound aclose); re-pinned through the bounded pump 2026-09-29."""
    async def script(rid):
        yield _request_frame(rid)
        yield TokenEvent(request_id=rid, text="a")
        yield None
    _install_stream(rig, script)

    async def _drive():
        agen = rig.runner.stream(_req("rid-dc"))
        async for ev in agen:
            if getattr(ev, "type", None) == "token":
                break
        await agen.aclose()
    asyncio.run(_drive())
    assert rig.state == {"attempts": 1, "closed": 1}
