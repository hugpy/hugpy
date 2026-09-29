"""KNOWN-GOOD CONTRACT — relay admission is a PER-MODEL FIFO, not one global line.

Catalogue: notes/KNOWN-GOOD-QUEUE.md (pillar 4, "session call allocations over
strict fifo").
Source under test: hugpy_engine/resolvers/remote.py — the FIFO admission tickets
(_GATE_WAITERS / _gate_waiter_join / _gate_waiter_is_head / _gate_waiter_leave)
and the in-flight cap accounting (_inflight_try_acquire / _inflight_release /
_effective_cap) that a reserved relay slot rides on.

This is the "over strict FIFO" pillar: admission order is FIFO *within one
model's* queue, but each model_key has its OWN queue, so a request for model B
is never head-of-line-blocked behind a saturated model A — the allocation is
per-(model) call session, not one global arrival-order line.

Deterministic: pure in-process data-structure calls; no clock, no worker, no
network, no GPU. Adds a NEW file; touches no existing case.
Established: reviewer pass 2026-09-29 against remote.py:971-1010, :1204-1258.
"""
from __future__ import annotations

import importlib

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")

A = "modelA"
B = "modelB"


@pytest.fixture(autouse=True)
def _clean_gate():
    """Isolate the process-global admission structures around each test."""
    with remote._GATE_WAITERS_LOCK:
        remote._GATE_WAITERS.clear()
    with remote._INFLIGHT_LOCK:
        remote._INFLIGHT.clear()
        remote._INFLIGHT_TS.clear()
    yield
    with remote._GATE_WAITERS_LOCK:
        remote._GATE_WAITERS.clear()
    with remote._INFLIGHT_LOCK:
        remote._INFLIGHT.clear()
        remote._INFLIGHT_TS.clear()


def test_admission_is_strict_fifo_within_a_model_but_per_model_independent():
    """INVARIANT: within one model_key the admission tickets are strict FIFO —
    only the head ticket may reserve; a later ticket waits. But a DIFFERENT
    model's head is independently ready — model B is never blocked behind
    model A's queue. Releasing the head advances the next waiter.
    Established: _gate_waiter_join/is_head/leave (remote.py:980)."""
    a1 = remote._gate_waiter_join(A)
    a2 = remote._gate_waiter_join(A)
    b1 = remote._gate_waiter_join(B)

    # Strict FIFO within model A: only the first ticket is head.
    assert remote._gate_waiter_is_head(A, a1) is True
    assert remote._gate_waiter_is_head(A, a2) is False
    # Per-model allocation: model B's own head is ready despite A's backlog.
    assert remote._gate_waiter_is_head(B, b1) is True

    # The head leaving hands the front of A's line to the next waiter.
    remote._gate_waiter_leave(A, a1)
    assert remote._gate_waiter_is_head(A, a2) is True
    # An emptied queue is cleaned up (no unbounded growth of dead model keys).
    remote._gate_waiter_leave(B, b1)
    with remote._GATE_WAITERS_LOCK:
        assert B not in remote._GATE_WAITERS


def test_inflight_cap_admits_up_to_cap_then_refuses_and_release_frees_a_slot():
    """INVARIANT: the per-(worker, model) in-flight counter admits up to the
    worker's advertised cap and then refuses (the crash-safe backstop that keeps
    central from firing a relay into a busy non-reentrant runner); a release
    frees exactly one slot. worker_idle=False so the stale-leak self-heal does
    NOT fire here (that path is the leaked-release reconcile, tested by its own
    heartbeat contract).
    Established: _inflight_try_acquire / _inflight_release (remote.py:1204)."""
    wid, mk, cap = "w1", A, 2
    assert remote._inflight_try_acquire(wid, mk, cap) is True   # 1/2
    assert remote._inflight_try_acquire(wid, mk, cap) is True   # 2/2
    assert remote._inflight_count(wid, mk) == 2
    # At cap, no worker-idle proof, fresh timestamp -> honest refusal.
    assert remote._inflight_try_acquire(wid, mk, cap, worker_idle=False) is False
    # A release frees one permit; the next acquire is admitted again.
    remote._inflight_release(wid, mk)
    assert remote._inflight_count(wid, mk) == 1
    assert remote._inflight_try_acquire(wid, mk, cap, worker_idle=False) is True


def test_effective_cap_is_none_for_a_slot_served_model_and_advertised_otherwise():
    """INVARIANT: a model seated in a healthy SLOT child is NOT centrally capped
    (its llama-server schedules its own concurrency -> _effective_cap None); an
    in-process model uses the worker's advertised in_process_max_concurrency,
    defaulting to the crash-safe 1 when the field is absent.
    Established: _effective_cap / _advertised_cap / _model_slot_served
    (remote.py:1077-1153)."""
    slot_worker = {"id": "w1", "slots": [{"model_key": A, "healthy": True}]}
    assert remote._effective_cap(slot_worker, A) is None

    inproc = {"id": "w2", "serving_limits": {"in_process_max_concurrency": 3}}
    assert remote._effective_cap(inproc, A) == 3

    legacy = {"id": "w3"}                       # no serving_limits advertised
    assert remote._effective_cap(legacy, A) == 1
