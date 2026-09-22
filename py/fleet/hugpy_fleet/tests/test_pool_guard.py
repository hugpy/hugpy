"""k59 — the fast-read reserve: SSE streams may not take the last threads.

Central is `gunicorn --workers 1 --threads N`. A long-lived stream holds one of
those N threads for its whole life, so enough open feeds starve every ordinary
request — the operator's "blips over just a few calls". STREAM_MAX_S bounds how
LONG one stream holds a thread; this reserve bounds how MANY do.

The property under test: past the cap a NEW stream is refused immediately and
honestly (503 + Retry-After, a message that says why), the slot is returned when
the stream ends OR when the client walks away mid-stream, and a refusal never
blocks — a stream waiting for a slot would still be holding the thread it was
trying not to monopolize.

Run: venv/bin/python -m pytest tests/test_pool_guard.py -q
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_fleet.central import pool_guard


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HUGPY_STREAM_SLOTS", "HUGPY_GUNICORN_THREADS"):
        monkeypatch.delenv(var, raising=False)
    pool_guard.reset()
    yield
    pool_guard.reset()


def test_reserve_leaves_threads_for_fast_reads(monkeypatch):
    """The sizing IS the property: with the shipped 8-thread pool, streams may
    hold at most 4 — so /llm/workers always has somewhere to run."""
    monkeypatch.setenv("HUGPY_GUNICORN_THREADS", "8")
    pool_guard.reset()
    assert pool_guard.stream_slots() == 4
    monkeypatch.setenv("HUGPY_GUNICORN_THREADS", "32")
    pool_guard.reset()
    assert pool_guard.stream_slots() == 19
    assert pool_guard.stream_slots() < 32, "the reserve must never be the pool"


def test_slots_are_explicitly_overridable(monkeypatch):
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "3")
    pool_guard.reset()
    assert pool_guard.stream_slots() == 3


def test_the_first_stream_is_never_refused(monkeypatch):
    """A reserve that could refuse the only open feed would be a broken feature,
    not a protected one — hence the floor of 2."""
    monkeypatch.setenv("HUGPY_GUNICORN_THREADS", "1")
    pool_guard.reset()
    assert pool_guard.stream_slots() >= 2


def test_streams_are_admitted_up_to_the_cap_then_refused(monkeypatch):
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "2")
    pool_guard.reset()
    a = pool_guard.stream_slot(); a.__enter__()
    b = pool_guard.stream_slot(); b.__enter__()
    assert pool_guard.snapshot() == {"held": 2, "limit": 2, "refused": 0}
    with pytest.raises(pool_guard.StreamCapacityExceeded):
        with pool_guard.stream_slot():
            pass
    assert pool_guard.snapshot()["refused"] == 1
    a.__exit__(None, None, None)
    b.__exit__(None, None, None)


def test_refusal_is_immediate_not_a_wait(monkeypatch):
    """Blocking for a slot would defeat the whole point: the waiter is still
    sitting on the thread it was trying not to monopolize."""
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    pool_guard.reset()
    held = pool_guard.stream_slot(); held.__enter__()
    t0 = time.monotonic()
    with pytest.raises(pool_guard.StreamCapacityExceeded):
        with pool_guard.stream_slot():
            pass
    assert time.monotonic() - t0 < 0.1
    held.__exit__(None, None, None)


def test_refusal_says_why_and_when_to_come_back(monkeypatch):
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    pool_guard.reset()
    held = pool_guard.stream_slot(); held.__enter__()
    try:
        with pool_guard.stream_slot():
            pass
    except pool_guard.StreamCapacityExceeded as exc:
        env = exc.as_error()
        assert exc.retry_after > 0
    assert env["ok"] is False
    assert env["error"]["code"] == "StreamCapacity"
    assert "reconnect" in env["error"]["message"]
    assert env["error"]["retry_after_s"] > 0
    held.__exit__(None, None, None)


def test_ending_a_stream_returns_its_slot(monkeypatch):
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    pool_guard.reset()
    with pool_guard.stream_slot():
        assert pool_guard.snapshot()["held"] == 1
    assert pool_guard.snapshot()["held"] == 0
    with pool_guard.stream_slot():       # capacity really is back
        pass


def test_a_stream_that_raises_still_returns_its_slot(monkeypatch):
    """A walked-away tab reaches the generator as GeneratorExit. If that leaked
    the slot, the reserve would shrink one abandoned tab at a time."""
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    pool_guard.reset()
    with pytest.raises(GeneratorExit):
        with pool_guard.stream_slot():
            raise GeneratorExit()
    assert pool_guard.snapshot()["held"] == 0


def test_the_gate_is_thread_safe(monkeypatch):
    """N threads racing the last slot: exactly `limit` may win."""
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "5")
    pool_guard.reset()
    won, lost = [], []
    start = threading.Barrier(20)

    def go():
        start.wait()
        try:
            with pool_guard.stream_slot():
                won.append(1)
                time.sleep(0.05)
        except pool_guard.StreamCapacityExceeded:
            lost.append(1)

    ts = [threading.Thread(target=go) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(won) == 5
    assert len(lost) == 15
    assert pool_guard.snapshot()["held"] == 0


# ── the route actually uses it ─────────────────────────────────────────────

