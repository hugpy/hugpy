"""EMFILE burst hardening (incident 2026-07-23).

Restarting hugpy-api-dev spawns all gunicorn workers at once; they open
store-backed files on the virtiofs mount simultaneously and the mount
momentarily returns EMFILE ("Too many open files") — the mount degrading under
load, NOT a real fd-limit breach. SQLite surfaces the same transient as
OperationalError("unable to open database file"). Both settle sub-second.

These tests cover ``comms.shared.retry_on_emfile`` (the shared helper) and an
integration check that ``AgentNodeStore._connect`` — the SOURCE-OF-TRUTH store
whose failure 500'd worker registration — retries a simulated EMFILE and hands
back a usable connection.

Run: cd .../abstract_hugpy_dev
     venv/bin/python -m pytest tests/test_emfile_retry.py -q
"""
import errno
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Point every comms-db singleton at a scratch file (never a real store).
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-emfile-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", os.path.join(
    tempfile.mkdtemp(prefix="hugpy-emfile-comms-"), "comms.db"))

import pytest

from hugpy_control.shared import retry_on_emfile, _is_emfile
from hugpy_fleet.central import agent_nodes


# ── helpers ──────────────────────────────────────────────────────────────────
def _emfile() -> OSError:
    """The exact transient the mount throws under an open burst."""
    return OSError(errno.EMFILE, "Too many open files")


def _sqlite_open_fault() -> sqlite3.OperationalError:
    """The same root cause surfaced through sqlite's open path."""
    return sqlite3.OperationalError("unable to open database file")


class _FakeSleep:
    """Records every backoff sleep so tests stay instant and can assert on the
    backoff schedule (count + boundedness)."""

    def __init__(self):
        self.calls = []

    def __call__(self, delay):
        self.calls.append(delay)


# ── _is_emfile classifier ────────────────────────────────────────────────────
def test_is_emfile_classifies_only_the_transients():
    assert _is_emfile(_emfile()) is True
    assert _is_emfile(_sqlite_open_fault()) is True
    # A DIFFERENT sqlite error is NOT the open-burst transient.
    assert _is_emfile(sqlite3.OperationalError("database is locked")) is False
    # A different OSError errno is not EMFILE.
    assert _is_emfile(OSError(errno.EACCES, "permission denied")) is False
    assert _is_emfile(PermissionError("nope")) is False
    assert _is_emfile(ValueError("unrelated")) is False


# ── retry_on_emfile: success after N transient failures ──────────────────────
def test_retries_emfile_then_returns_value():
    sleep = _FakeSleep()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= 3:        # fail EMFILE 3×, succeed on the 4th
            raise _emfile()
        return "conn"

    out = retry_on_emfile(fn, attempts=5, sleep=sleep, rng=None)
    assert out == "conn"
    assert calls["n"] == 4
    assert len(sleep.calls) == 3   # one sleep before each of the 3 retries


def test_sqlite_unable_to_open_is_retryable():
    sleep = _FakeSleep()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sqlite_open_fault()
        return 42

    assert retry_on_emfile(fn, attempts=5, sleep=sleep, rng=None) == 42
    assert calls["n"] == 2
    assert len(sleep.calls) == 1


# ── retry_on_emfile: non-matching errors propagate immediately ───────────────
def test_permission_error_not_retried():
    sleep = _FakeSleep()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise PermissionError("denied")

    with pytest.raises(PermissionError):
        retry_on_emfile(fn, attempts=5, sleep=sleep, rng=None)
    assert calls["n"] == 1         # NOT retried
    assert sleep.calls == []       # no backoff for a non-transient


def test_different_sqlite_error_not_retried():
    sleep = _FakeSleep()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        retry_on_emfile(fn, attempts=5, sleep=sleep, rng=None)
    assert calls["n"] == 1
    assert sleep.calls == []


# ── retry_on_emfile: attempts exhausted re-raises the LAST error ─────────────
def test_exhausted_reraises_last_error():
    sleep = _FakeSleep()
    calls = {"n": 0}
    last_exc = _emfile()

    def fn():
        calls["n"] += 1
        raise last_exc

    with pytest.raises(OSError) as ei:
        retry_on_emfile(fn, attempts=4, sleep=sleep, rng=None)
    assert ei.value is last_exc            # the LAST error, re-raised
    assert calls["n"] == 4                 # exactly `attempts` tries
    assert len(sleep.calls) == 3           # attempts-1 sleeps, then give up


# ── retry_on_emfile: backoff is bounded ──────────────────────────────────────
def test_backoff_is_bounded_and_monotone_capped():
    sleep = _FakeSleep()

    def fn():
        raise _emfile()

    with pytest.raises(OSError):
        retry_on_emfile(fn, attempts=8, base_delay=0.05, max_delay=0.2,
                        sleep=sleep, rng=None)
    # 7 sleeps (attempts-1), each within [0, max_delay], never exceeding the cap.
    assert len(sleep.calls) == 7
    assert all(0.0 <= d <= 0.2 for d in sleep.calls), sleep.calls
    # Exponential until the cap, then pinned at the cap.
    assert sleep.calls[0] == pytest.approx(0.05)
    assert sleep.calls[-1] == pytest.approx(0.2)


def test_jitter_stays_within_one_base_delay():
    sleep = _FakeSleep()

    def fn():
        raise _emfile()

    # rng always returns its max (→1.0) so jitter is at its ceiling: each delay
    # is min(max_delay, base*2**i) + base, and must still never exceed max_delay
    # + base_delay. Confirms jitter is bounded, not unbounded.
    with pytest.raises(OSError):
        retry_on_emfile(fn, attempts=6, base_delay=0.05, max_delay=0.2,
                        sleep=sleep, rng=lambda: 1.0)
    assert all(d <= 0.2 + 0.05 + 1e-9 for d in sleep.calls), sleep.calls


def test_single_attempt_never_sleeps_and_raises():
    sleep = _FakeSleep()

    def fn():
        raise _emfile()

    with pytest.raises(OSError):
        retry_on_emfile(fn, attempts=1, sleep=sleep, rng=None)
    assert sleep.calls == []               # no retry, so no backoff


# ── integration: AgentNodeStore._connect retries a simulated EMFILE ──────────
def test_agent_node_store_connect_retries_emfile(tmp_path, monkeypatch):
    """The registration-critical store: monkeypatch sqlite3.connect to fail
    EMFILE twice then succeed, and confirm _connect returns a USABLE connection
    (PRAGMAs run, a round-trip query works). Sleeps are mocked to stay fast."""
    db = str(tmp_path / "nodes.db")
    real_connect = sqlite3.connect
    state = {"n": 0}

    # Patching sqlite3.connect is process-global, so scope the fault to OUR db
    # path — ambient singleton stores in the same process must pass through
    # untouched (otherwise their connects pollute the counter).
    def flaky_connect(target, *a, **k):
        if target != db:
            return real_connect(target, *a, **k)
        state["n"] += 1
        if state["n"] <= 2:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_connect(target, *a, **k)

    monkeypatch.setattr(agent_nodes.sqlite3, "connect", flaky_connect)
    # Mock the backoff sleep (the helper lives in comms.shared).
    monkeypatch.setattr("hugpy_control.shared.time.sleep",
                        lambda *_: None)

    store = agent_nodes.AgentNodeStore(path=db)
    conn = store._connect()
    try:
        assert state["n"] == 3                 # 2 EMFILE + 1 success
        # A usable connection: PRAGMAs applied, round-trip works.
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (7)")
        assert conn.execute("SELECT x FROM t").fetchone()[0] == 7
    finally:
        conn.close()


def test_agent_node_store_enroll_survives_emfile_burst(tmp_path, monkeypatch):
    """End-to-end store op through the retry: a two-EMFILE burst on the FIRST
    connect must not break enrollment — the node registers and can be fetched."""
    db = str(tmp_path / "nodes2.db")
    real_connect = sqlite3.connect
    state = {"n": 0}

    # Scope the fault to OUR db (see the sibling test) so ambient singleton
    # connects in the same process pass through untouched.
    def flaky_connect(target, *a, **k):
        if target != db:
            return real_connect(target, *a, **k)
        state["n"] += 1
        if state["n"] <= 2:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_connect(target, *a, **k)

    monkeypatch.setattr(agent_nodes.sqlite3, "connect", flaky_connect)
    monkeypatch.setattr("hugpy_control.shared.time.sleep",
                        lambda *_: None)

    store = agent_nodes.AgentNodeStore(path=db)
    rec = store.register(name="ae", host="10.0.0.5", capabilities=["chat"])
    assert rec.get("id")
    # The row is real and retrievable (no ghost from a swallowed failure).
    node = store.get(rec["id"])
    assert node is not None
    assert node["name"] == "ae"
