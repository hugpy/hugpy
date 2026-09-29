"""KNOWN-GOOD CONTRACT — call queue / worker relay hold loop.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "call queue / relay").
Source under test: hugpy_engine/resolvers/remote.py — DelegatingRunner.stream()
hold loop, _cold_progress (heartbeat slot reading), _worker_busy_max_s,
_cold_hold_max_s, _is_worker_busy_signal.

Deterministic: the relay's ``time`` and ``asyncio.sleep`` are replaced by a
fake clock (sleep ADVANCES the clock instead of waiting), worker selection and
the worker SSE stream are fakes, the relay gate is off, and no worker/network
is touched. Each test names the invariant it pins and the handoff/date that
established it (frontier handoff 2026-09-28 and toolserver handoff h25).
"""
from __future__ import annotations

import asyncio
import importlib
import time as _real_time
import types

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")
from hugpy_engine.schemas.event_schemas import DoneEvent, TokenEvent  # noqa: E402

MODEL = "known-good-model"
WORKER_ID = "w1"


def _req(rid="rid-1"):
    return types.SimpleNamespace(
        request_id=rid, pool=None,
        reference_images=None, reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "hi"}]},
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


def _stages(evs):
    return [getattr(e, "stage", None) for e in evs if getattr(e, "type", None) == "status"]


class _Clock:
    """A settable wall clock; ``sleep`` advances it instead of waiting."""

    def __init__(self, t0=1_000_000.0):
        self.t = t0
        self.t0 = t0
        self.sleeps: list[float] = []

    def time(self):
        return self.t

    def elapsed(self):
        return self.t - self.t0

    async def sleep(self, s):
        self.sleeps.append(float(s))
        self.t += float(s)
        await asyncio.sleep(0)


@pytest.fixture
def rig(monkeypatch):
    """Fake clock + fake worker selection; the worker dict is mutable so a test
    can shape the heartbeat ``slots`` the relay reads."""
    clock = _Clock()
    monkeypatch.setattr(remote, "time", types.SimpleNamespace(
        time=clock.time, monotonic=_real_time.monotonic, sleep=_real_time.sleep,
        perf_counter=_real_time.perf_counter))
    monkeypatch.setattr(remote, "asyncio", types.SimpleNamespace(
        sleep=clock.sleep, Semaphore=asyncio.Semaphore,
        get_running_loop=asyncio.get_running_loop, Event=asyncio.Event,
        Lock=asyncio.Lock, gather=asyncio.gather, CancelledError=asyncio.CancelledError))
    # Relay gate off (no cap math), tight deterministic hold clocks.
    monkeypatch.setenv("HUGPY_CENTRAL_GATE", "off")
    monkeypatch.setenv("HUGPY_COLD_HOLD_POLL_S", "10")
    monkeypatch.setenv("HUGPY_COLD_HOLD_MAX_S", "50")
    monkeypatch.setenv("HUGPY_COLD_HOLD_STALL_S", "30")
    monkeypatch.setenv("HUGPY_WORKER_BUSY_MAX_S", "120")
    monkeypatch.setenv("HUGPY_COMMS_DB", "off")
    for k in ("HUGPY_LOCAL_FALLBACK", "HUGPY_NO_LOCAL_SERVING", "HUGPY_COLD_HOLD"):
        monkeypatch.delenv(k, raising=False)
    worker = {"id": WORKER_ID, "name": "ae-worker", "url": "http://w1:9200", "slots": []}
    monkeypatch.setattr(remote, "_select",
                        lambda mk, pool=None, task=None, **kw: (worker, None))
    orig_ls = remote._load_state_provider
    remote.set_load_state_provider(None)
    remote._COLD_KICKING.clear()
    with remote._LOAD_VERDICTS_LOCK:
        remote._LOAD_VERDICTS.clear()
    calls = {"stream": 0}
    fw, tk = _text_framework()
    Runner = remote.make_delegating_runner(fw, tk)
    runner = Runner(types.SimpleNamespace(model_key=MODEL))
    yield types.SimpleNamespace(clock=clock, worker=worker, runner=runner,
                                calls=calls, monkeypatch=monkeypatch)
    remote.set_load_state_provider(orig_ls)
    remote._COLD_KICKING.clear()
    with remote._LOAD_VERDICTS_LOCK:
        remote._LOAD_VERDICTS.clear()


def _busy_slot():
    return {"model_key": MODEL, "healthy": True, "serving": True, "busy": True}


def _idle_slot():
    return {"model_key": MODEL, "healthy": True, "serving": True, "busy": False}


def _stream_always_busy(rig, message=None):
    # The worker agent's real ModelBusy text (hugpy_fleet.worker.gen_gate.ModelBusy):
    # "model '<key>' is busy: N request(s) already in the runner …" — the prose
    # marker "is busy:" is what the flattened _ColdRetry message keeps.
    message = message or (f"model '{MODEL}' is busy: 1 request(s) already in the "
                          "runner; waited 30.0s")

    async def ws(worker, payload, rid):
        rig.calls["stream"] += 1
        raise remote._WorkerHTTPError(503, {"ok": False, "error": {
            "code": "model_busy", "message": message}})
        yield  # pragma: no cover — generator marker
    rig.monkeypatch.setattr(remote, "_worker_stream", ws)


# ---------------------------------------------------------------------------
# Warm slot busy → bounded, structured worker_busy
# ---------------------------------------------------------------------------
def test_warm_slot_busy_is_released_as_worker_busy_after_busy_max(rig):
    """INVARIANT: a request queued behind an already-WARM slot that is busy with
    another generation is held at most HUGPY_WORKER_BUSY_MAX_S (default 120s)
    and then released with a structured ``worker_busy`` ErrorEvent — it is NOT
    held for the cold-load ceiling and NOT treated as forward progress forever.
    The busy clock is separate from the cold-hold clocks: here the cold ceiling
    is 50s and the busy release still happens at 120s.
    Established: frontier handoff 2026-09-28 (_worker_busy_max_s / busy_since)."""
    rig.worker["slots"] = [_busy_slot()]
    _stream_always_busy(rig)
    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert _types(evs)[-1] == "error", _types(evs)
    msg = evs[-1].message
    assert msg.startswith("worker_busy:"), msg
    assert "remained occupied for 120s" in msg, msg
    assert "hard ceiling" not in msg and "did not finish loading" not in msg
    assert 120 <= rig.clock.elapsed() < 140, rig.clock.elapsed()
    # 13 relay attempts: t=0,10,…,120 — every poll retried against the warm slot.
    assert rig.calls["stream"] == 13
    # While waiting the job reads 'awaiting-capacity' (loaded, occupied) — never
    # 'awaiting-load' (that is the cold path's stage).
    assert set(_stages(evs)) == {"dispatch", "awaiting-capacity"}, _stages(evs)
    assert all(getattr(e, "worker_state", None) == "loaded"
               for e in evs if getattr(e, "stage", None) == "awaiting-capacity")
    assert (WORKER_ID, MODEL) not in remote._COLD_KICKING


def test_structured_busy_code_alone_is_enough_to_hold(rig):
    """INVARIANT (intended): the structured busy CODE from the worker is the
    authority; the bounded worker_busy hold must not depend on the wording of
    the message. Established: 2026-07-28 entry-path fix (structured first)."""
    rig.worker["slots"] = [_busy_slot()]
    _stream_always_busy(rig, message="slot is serving another request")
    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert evs[-1].message.startswith("worker_busy:"), evs[-1].message


def test_worker_busy_max_env_knob_and_default(monkeypatch):
    """INVARIANT: HUGPY_WORKER_BUSY_MAX_S defaults to 120s; garbage or <=0
    degrades to the default (a knob can misconfigure, never break).
    Established: frontier handoff 2026-09-28."""
    monkeypatch.delenv("HUGPY_WORKER_BUSY_MAX_S", raising=False)
    assert remote._worker_busy_max_s() == 120.0
    monkeypatch.setenv("HUGPY_WORKER_BUSY_MAX_S", "45")
    assert remote._worker_busy_max_s() == 45.0
    monkeypatch.setenv("HUGPY_WORKER_BUSY_MAX_S", "nope")
    assert remote._worker_busy_max_s() == 120.0
    monkeypatch.setenv("HUGPY_WORKER_BUSY_MAX_S", "0")
    assert remote._worker_busy_max_s() == 120.0


# ---------------------------------------------------------------------------
# Cold load keeps the long cold-hold ceiling
# ---------------------------------------------------------------------------
def test_cold_load_keeps_cold_hold_ceiling_not_the_busy_clock(rig):
    """INVARIANT: a model that is genuinely LOADING (heartbeat load-state
    in_progress, no healthy slot yet) is held under HUGPY_COLD_HOLD_MAX_S /
    HUGPY_COLD_HOLD_STALL_S — the busy clock never applies — and gives up with
    the honest ceiling message, not ``worker_busy``. Default ceiling is 900s.
    Established: t36 cold hold (2026-07-28) — re-pinned 2026-09-28 when the
    busy clock was split off."""
    remote.set_load_state_provider(
        lambda mk, wid, since=0.0: {"healthy": False, "in_progress": True,
                                    "progress": 0.3, "message": "loading weights"})

    async def ws(worker, payload, rid):
        rig.calls["stream"] += 1
        raise RuntimeError("RemoteProtocolError: Server disconnected without sending a response.")
        yield  # pragma: no cover
    rig.monkeypatch.setattr(remote, "_worker_stream", ws)

    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert _types(evs)[-1] == "error"
    msg = evs[-1].message
    assert "worker_busy" not in msg
    assert "hard ceiling" in msg, msg
    assert 50 < rig.clock.elapsed() <= 70, rig.clock.elapsed()
    assert set(_stages(evs)) == {"dispatch", "awaiting-load"}, _stages(evs)
    assert rig.calls["stream"] >= 6


def test_cold_hold_default_ceiling_is_900s(monkeypatch):
    """INVARIANT: HUGPY_COLD_HOLD_MAX_S default 900s (raised from 300 on
    2026-07-28); stall window default 90s."""
    monkeypatch.delenv("HUGPY_COLD_HOLD_MAX_S", raising=False)
    monkeypatch.delenv("HUGPY_COLD_HOLD_STALL_S", raising=False)
    assert remote._cold_hold_max_s() == 900.0
    assert remote._cold_hold_stall_s() == 90.0


# ---------------------------------------------------------------------------
# Loaded + idle heartbeat → immediate dispatch (h25)
# ---------------------------------------------------------------------------
def test_loaded_idle_heartbeat_dispatches_immediately_without_retry_spam(rig):
    """INVARIANT: when the worker heartbeat shows the model's native slot
    loaded AND idle, a transient relay failure is retried IMMEDIATELY — no
    'loaded and idle; retrying' status is published and no poll sleep happens.
    A stale ``_COLD_KICKING`` marker for that (worker, model) is discarded
    instead of turning the idle slot into a poll loop; the worker's generation
    gate remains the concurrency authority.
    Established: toolserver handoff h25 (2026-09-28), remote.py ~3885 / ~4023."""
    rig.worker["slots"] = [_idle_slot()]
    remote._COLD_KICKING.add((WORKER_ID, MODEL))     # stale marker from a dead kick

    async def ws(worker, payload, rid):
        rig.calls["stream"] += 1
        if rig.calls["stream"] == 1:
            # The worker's gen-gate held us for a beat (structured 503 busy) —
            # the slot heartbeat already says loaded+idle: retry NOW.
            raise remote._WorkerHTTPError(503, {"error": {
                "code": "model_busy",
                "message": f"model '{MODEL}' is busy: 1 request(s) already in the runner"}})
        yield TokenEvent(request_id=rid, text="ok")
        yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=1, finish_reason="stop")
    rig.monkeypatch.setattr(remote, "_worker_stream", ws)

    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert "token" in _types(evs) and "done" in _types(evs) and "error" not in _types(evs)
    assert rig.calls["stream"] == 2
    assert rig.clock.sleeps == [], "no poll sleep for a loaded+idle slot"
    assert set(_stages(evs)) == {"dispatch"}, _stages(evs)
    assert (WORKER_ID, MODEL) not in remote._COLD_KICKING


def test_loaded_idle_slot_with_a_real_failure_is_surfaced_not_held(rig):
    """INVARIANT: when the heartbeat says loaded+idle and the request still
    fails with a NON-busy error, there is nothing to wait for — the failure is
    surfaced at once naming the worker and the error (no hold, no retry storm).
    Established: h25 (2026-09-28), remote.py ~3996."""
    rig.worker["slots"] = [_idle_slot()]

    async def ws(worker, payload, rid):
        rig.calls["stream"] += 1
        raise RuntimeError("RemoteProtocolError: Server disconnected")
        yield  # pragma: no cover
    rig.monkeypatch.setattr(remote, "_worker_stream", ws)

    evs = asyncio.run(_collect(rig.runner.stream(_req())))
    assert _types(evs) == ["status", "error"], _types(evs)
    assert "loaded and idle, but the request failed" in evs[-1].message
    assert "Server disconnected" in evs[-1].message
    assert rig.calls["stream"] == 1 and rig.clock.sleeps == []


def test_cold_progress_reads_slot_state_from_heartbeat(rig):
    """INVARIANT: _cold_progress trusts the live heartbeat slot over the
    central load-state provider: healthy+busy → 'waiting for its current
    request', healthy+idle → 'loaded and idle', both ``ready``. Slot keys are
    alias-tolerant (bare vs ``owner~`` spelling name the same slot).
    Established: h25 (2026-09-28); alias tolerance 2026-07-23."""
    rig.worker["slots"] = [_busy_slot()]
    moved, prog, msg, honest, ready = remote._cold_progress(MODEL, rig.worker, 0.0)
    assert (moved, honest, ready) == (True, None, True)
    assert "waiting for its current request" in msg
    rig.worker["slots"] = [_idle_slot()]
    moved, prog, msg, honest, ready = remote._cold_progress(MODEL, rig.worker, 0.0)
    assert ready and "loaded and idle" in msg
    rig.worker["slots"] = [{**_idle_slot(), "model_key": "Qwen3-Coder-Next-GGUF"}]
    _, _, msg, _, ready = remote._cold_progress("Qwen~Qwen3-Coder-Next-GGUF", rig.worker, 0.0)
    assert ready and msg, "owner-qualified request matches the bare slot key"
    # No slot and no provider → nothing known (the hold degrades to blind retry).
    rig.worker["slots"] = []
    assert remote._cold_progress(MODEL, rig.worker, 0.0) == (False, None, None, None, False)


# ---------------------------------------------------------------------------
# Cancel while held is clean and frees the kick key
# ---------------------------------------------------------------------------
def test_cancel_while_waiting_on_busy_slot_is_clean(rig):
    """INVARIANT: cancelling a request that is waiting on a busy warm slot
    ends the stream with NO error/token event, releases the cold-kick key and
    leaves no load verdict — a user pulling out is not a load failure.
    Established: t36 cancel-while-held; re-pinned for the busy wait 2026-09-28."""
    rig.worker["slots"] = [_busy_slot()]
    _stream_always_busy(rig)

    async def _drive():
        cancel = asyncio.Event()
        seen = []
        async for ev in rig.runner.stream(_req("rid-cancel"), cancel_event=cancel):
            seen.append(ev)
            if getattr(ev, "stage", None) == "awaiting-capacity":
                cancel.set()
        return seen

    seen = asyncio.run(_drive())
    assert "error" not in _types(seen) and "token" not in _types(seen)
    assert any(getattr(e, "stage", None) == "awaiting-capacity" for e in seen)
    assert (WORKER_ID, MODEL) not in remote._COLD_KICKING
    assert remote._active_load_verdict(WORKER_ID, MODEL) is None
    assert rig.clock.elapsed() < 120


# ---------------------------------------------------------------------------
# Busy-signal classification (structured first, prose last)
# ---------------------------------------------------------------------------
def test_worker_busy_signal_classification():
    """INVARIANT: a structured 503 / busy code from the worker is a HOLD
    signal; a 507 budget refusal or a permanent marker is a verdict, never
    'still working'.
    Established: 2026-07-28 entry-path fix (_WorkerHTTPError + _BUSY_CODES)."""
    busy = remote._WorkerHTTPError(503, {"error": {"code": "model_busy", "message": "warming"}})
    assert remote._is_worker_busy_signal(busy) is True
    loading = remote._WorkerHTTPError(503, {"error": {"code": "loading", "message": "…"}})
    assert remote._is_worker_busy_signal(loading) is True
    refused = remote._WorkerHTTPError(507, {"error": {"code": "budget_refusal",
                                                     "message": "won't fit on GPU"}})
    assert remote._is_worker_busy_signal(refused) is False
    assert remote._is_worker_busy_signal("model_busy: still warming") is True
    assert remote._is_worker_busy_signal("LoadRefusal: won't fit on GPU") is False
    assert remote._is_permanent_load_error("CUDA out of memory") is True
