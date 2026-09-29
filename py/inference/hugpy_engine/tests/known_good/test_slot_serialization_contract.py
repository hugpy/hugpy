"""KNOWN-GOOD CONTRACT — one in-flight relay per native slot, FIFO, bounded.

Keeper's decision on the operator's behalf, 2026-09-29: central serialises to
ONE in-flight request per native slot per model by default (two 30k-token
prefills co-running on one seat halved each other), FIFO for the rest, with a
bounded wait that returns a structured worker_busy; HUGPY_CENTRAL_SLOT_CONCURRENCY
raises it. Source: hugpy_engine/resolvers/remote.py — _slot_concurrency,
_slot_count, _effective_cap, _gate_wait_s, _gate_idle_grace_s,
_acquire_relay_slot_async, DelegatingRunner.stream().

Deterministic: REAL clock and asyncio (sub-second fake generations), fake
worker selection + fake worker SSE stream, no network.
"""
from __future__ import annotations

import asyncio
import importlib
import time
import types

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")
from hugpy_engine.schemas.event_schemas import DoneEvent, StatusEvent, TokenEvent  # noqa: E402

MODEL = "known-good-model"
WORKER_ID = "ws"


def _req(rid):
    return types.SimpleNamespace(
        request_id=rid, pool=None, reference_images=None, reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "hi"}]})


def _text_framework():
    for (fw, tk) in remote.FRAMEWORK_RUNNERS:
        if tk != "image-text-to-text":
            return fw, tk
    return next(iter(remote.FRAMEWORK_RUNNERS))


@pytest.fixture
def rig(monkeypatch):
    """Gate ON, one healthy native slot for MODEL, real clocks."""
    for k in ("HUGPY_CENTRAL_GATE", "HUGPY_CENTRAL_GATE_WAIT_S",
              "HUGPY_CENTRAL_SLOT_CONCURRENCY", "HUGPY_LOCAL_FALLBACK",
              "HUGPY_NO_LOCAL_SERVING", "HUGPY_COLD_HOLD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HUGPY_COMMS_DB", "off")
    monkeypatch.setenv("HUGPY_RELAY_FIRST_TOKEN_BASE_S", "5")
    worker = {"id": WORKER_ID, "name": "ae-worker", "url": "http://ws:9200",
              "slots": [{"model_key": MODEL, "healthy": True, "serving": True, "busy": False}]}
    monkeypatch.setattr(remote, "_select", lambda mk, pool=None, task=None, **kw: (worker, None))
    remote.set_worker_candidates_provider(lambda mk, pool=None, task=None: [worker])
    orig_ls = remote._load_state_provider
    remote.set_load_state_provider(None)
    remote._COLD_KICKING.clear()
    with remote._INFLIGHT_LOCK:
        remote._INFLIGHT.clear(); remote._INFLIGHT_TS.clear()
    spans = {}

    async def ws(w, payload, rid):
        spans[rid] = [time.monotonic(), None]
        yield StatusEvent(type="request", request_id=rid)
        await asyncio.sleep(0.3)                     # the "generation"
        yield TokenEvent(request_id=rid, text="ok")
        spans[rid][1] = time.monotonic()
        yield DoneEvent(request_id=rid, input_tokens=0, output_chunks=1, finish_reason="stop")
    monkeypatch.setattr(remote, "_worker_stream", ws)
    fw, tk = _text_framework()
    Runner = remote.make_delegating_runner(fw, tk)
    yield types.SimpleNamespace(runner=Runner(types.SimpleNamespace(model_key=MODEL)),
                                worker=worker, spans=spans, monkeypatch=monkeypatch)
    remote.set_load_state_provider(orig_ls)
    remote.set_worker_candidates_provider(None)
    remote._COLD_KICKING.clear()
    with remote._INFLIGHT_LOCK:
        remote._INFLIGHT.clear(); remote._INFLIGHT_TS.clear()


async def _collect(agen):
    return [e async for e in agen]


def _types(evs):
    return [getattr(e, "type", None) for e in evs]


def test_two_concurrent_submissions_serialise_on_one_native_slot(rig):
    """INVARIANT: two concurrent calls for a slot-served model on one seat run
    one AFTER the other (second starts after the first's done), both complete;
    the seat is free afterwards. Established: 2026-09-29."""
    async def _both():
        return await asyncio.gather(_collect(rig.runner.stream(_req("a"))),
                                    _collect(rig.runner.stream(_req("b"))))
    ea, eb = asyncio.run(_both())
    assert _types(ea)[-1] == "done" and _types(eb)[-1] == "done"
    a, b = rig.spans["a"], rig.spans["b"]
    first, second = (a, b) if a[0] <= b[0] else (b, a)
    assert second[0] >= first[1], "second call co-ran with the first on the same seat"
    assert remote._inflight_count(WORKER_ID, MODEL) == 0


def test_bounded_wait_returns_structured_worker_busy(rig):
    """INVARIANT: a queued call waits at most HUGPY_CENTRAL_GATE_WAIT_S and is
    then released with ONE error event carrying the structured worker_busy
    verdict; the running call is unaffected. Default wait = HUGPY_WORKER_BUSY_MAX_S
    (120 s), explicit 0 = wait until capacity. Established: 2026-09-29."""
    rig.monkeypatch.setenv("HUGPY_CENTRAL_GATE_WAIT_S", "0.05")

    async def _both():
        return await asyncio.gather(_collect(rig.runner.stream(_req("a"))),
                                    _collect(rig.runner.stream(_req("b"))))
    ea, eb = asyncio.run(_both())
    outcomes = sorted((_types(ea)[-1], _types(eb)[-1]))
    assert outcomes == ["done", "error"], outcomes
    err = (ea if _types(ea)[-1] == "error" else eb)[-1]
    assert "worker_busy" in err.message
    assert remote._inflight_count(WORKER_ID, MODEL) == 0
    rig.monkeypatch.delenv("HUGPY_CENTRAL_GATE_WAIT_S", raising=False)
    rig.monkeypatch.delenv("HUGPY_WORKER_BUSY_MAX_S", raising=False)
    assert remote._gate_wait_s() == remote._worker_busy_max_s() == 120.0
    rig.monkeypatch.setenv("HUGPY_CENTRAL_GATE_WAIT_S", "0")
    assert remote._gate_wait_s() == 0.0


def test_slot_concurrency_knob_raises_the_cap_and_seats_multiply(rig):
    """INVARIANT: HUGPY_CENTRAL_SLOT_CONCURRENCY=2 lets two calls co-run on one
    seat; two seats for the model double the cap; the cap is per (worker,
    model). Established: 2026-09-29."""
    rig.monkeypatch.setenv("HUGPY_CENTRAL_SLOT_CONCURRENCY", "2")

    async def _both():
        return await asyncio.gather(_collect(rig.runner.stream(_req("a"))),
                                    _collect(rig.runner.stream(_req("b"))))
    ea, eb = asyncio.run(_both())
    assert _types(ea)[-1] == "done" and _types(eb)[-1] == "done"
    a, b = rig.spans["a"], rig.spans["b"]
    assert max(a[0], b[0]) < min(a[1], b[1]), "with cap 2 the calls should overlap"
    rig.monkeypatch.delenv("HUGPY_CENTRAL_SLOT_CONCURRENCY", raising=False)
    two_seats = {"id": "w2", "slots": [
        {"model_key": MODEL, "healthy": True, "slot_id": "1"},
        {"model_key": MODEL, "healthy": True, "slot_id": "2"}]}
    assert remote._effective_cap(two_seats, MODEL) == 2
    assert remote._effective_cap(rig.worker, MODEL) == 1


def test_fresh_permit_is_not_reconciled_as_a_leak_by_a_stale_idle_heartbeat(rig, monkeypatch):
    """INVARIANT: the worker heartbeat still saying "idle" a moment after a
    permit was taken (the beat has not fired yet) must NOT reset the count —
    only a saturated count older than HUGPY_CENTRAL_GATE_IDLE_GRACE_S (30 s)
    may be reconciled by an idle heartbeat. Established: 2026-09-29."""
    monkeypatch.delenv("HUGPY_CENTRAL_GATE_IDLE_GRACE_S", raising=False)
    assert remote._inflight_try_acquire(WORKER_ID, MODEL, 1, worker_idle=True)
    assert not remote._inflight_try_acquire(WORKER_ID, MODEL, 1, worker_idle=True)
    with remote._INFLIGHT_LOCK:
        remote._INFLIGHT_TS[(WORKER_ID, MODEL)] -= 31          # older than the grace
    assert remote._inflight_try_acquire(WORKER_ID, MODEL, 1, worker_idle=True)
