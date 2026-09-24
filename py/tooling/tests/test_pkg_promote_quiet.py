"""pkg_promote: a pin that restarts central waits for a quiet central (no
running benchmark / admission job), bounded; proceeds after the window with a
checkpoint and a recorded note; ``--now`` skips the wait. No DB, no network:
a fake fleet and a fake psycopg connection that records the UPDATEs."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pkg_promote as P  # noqa: E402

BUSY = {"reason": "benchmark r1 running (5/15)", "benchmark": "r1"}


class FakeFleet:
    def __init__(self, busy=None, restart=True):
        self.busy_value, self.restart = busy, restart
        self.pins, self.checkpoints = [], []

    def busy(self):
        return self.busy_value

    def restart_needed(self, version):
        return self.restart

    def checkpoint(self, reason):
        self.checkpoints.append(reason)
        return {"checkpointed": True, "run_id": "r1"}

    def workers(self):
        return []

    def smoke(self, worker, model):
        return {"ok": True}

    def pin(self, version, index_dir):
        self.pins.append(version)
        return {"ok": True, "restarted": True}


class FakeCursor:
    def __init__(self, db):
        self.db, self.description, self._one = db, None, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params=()):
        text = query.as_string(None) if hasattr(query, "as_string") else str(query)
        if text.startswith("SELECT * FROM"):
            row = self.db.row
            self.description = [type("D", (), {"name": k})() for k in row]
            self._one = tuple(row.values())
        elif text.startswith("UPDATE"):
            self.db.updates.append((text, params))
            vals = [getattr(p, "obj", p) for p in params]
            if "status = 'rolling_out'" in text:
                self.db.row.update(status="rolling_out", pin_log=vals[2], note=vals[3])
            else:
                self.db.row.update(pin_log=vals[0], note=vals[1])

    def fetchone(self):
        return self._one


class FakeConn:
    def __init__(self):
        self.row = {"id": 7, "version": "0.2.1.post5", "status": "published", "pin_log": None,
                    "note": None, "created_at": datetime(2026, 9, 23)}
        self.updates = []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass


def test_pin_waits_while_a_benchmark_runs(monkeypatch):
    monkeypatch.delenv(P.NOW_ENV, raising=False)
    conn, fleet = FakeConn(), FakeFleet(busy=BUSY)
    out = P.start_rollout(conn, 7, fleet, Path("/nonexistent"))
    assert out["pinned"] is False and fleet.pins == []          # central NOT restarted
    assert conn.row["status"] == "published"
    assert conn.row["note"].startswith("waiting: benchmark r1 running (5/15)")
    assert conn.row["pin_log"]["quiet_wait"]["since"] > 0
    # the next tick retries the same row; still inside the window -> still waiting
    out = P.start_rollout(conn, 7, fleet, Path("/nonexistent"))
    assert out["pinned"] is False and conn.row["pin_log"]["quiet_wait"]["checks"] == 2


def test_pin_proceeds_after_the_quiet_wait_expires_with_checkpoint(monkeypatch):
    monkeypatch.delenv(P.NOW_ENV, raising=False)
    conn, fleet = FakeConn(), FakeFleet(busy=BUSY)
    monkeypatch.setattr(P, "QUIET_WAIT_S", 60.0)
    P.start_rollout(conn, 7, fleet, Path("/nonexistent"))
    conn.row["pin_log"]["quiet_wait"]["since"] -= 61           # the window has passed
    out = P.start_rollout(conn, 7, fleet, Path("/nonexistent"))
    assert out["pinned"] is True and fleet.pins == ["0.2.1.post5"]
    assert fleet.checkpoints and "quiet-wait 60s expired" in fleet.checkpoints[0]
    assert conn.row["status"] == "rolling_out"
    assert conn.row["note"] == "proceeded after quiet-wait expiry; benchmark r1 checkpointed"
    assert conn.row["pin_log"]["quiet_wait"]["expired_at"] > 0


def test_now_flag_and_env_skip_the_wait(monkeypatch):
    monkeypatch.delenv(P.NOW_ENV, raising=False)
    conn, fleet = FakeConn(), FakeFleet(busy=BUSY)
    out = P.start_rollout(conn, 7, fleet, Path("/nonexistent"), now=True)
    assert out["pinned"] is True and fleet.pins == ["0.2.1.post5"] and fleet.checkpoints == []
    conn, fleet = FakeConn(), FakeFleet(busy=BUSY)
    monkeypatch.setenv(P.NOW_ENV, "1")
    assert P.start_rollout(conn, 7, fleet, Path("/nonexistent"))["pinned"] is True


def test_quiet_central_or_no_restart_never_waits(monkeypatch):
    monkeypatch.delenv(P.NOW_ENV, raising=False)
    conn, fleet = FakeConn(), FakeFleet(busy=None)
    assert P.start_rollout(conn, 7, fleet, Path("/x"))["pinned"] is True
    assert conn.row["note"] is None
    conn, fleet = FakeConn(), FakeFleet(busy=BUSY, restart=False)   # central already runs V
    assert P.start_rollout(conn, 7, fleet, Path("/x"))["pinned"] is True


def test_admission_only_busy_expiry_note(monkeypatch):
    monkeypatch.delenv(P.NOW_ENV, raising=False)
    busy = {"reason": "admission 1 running / 0 queued", "admission": ["j1"]}
    row = {"id": 7, "version": "V", "pin_log": {"quiet_wait": {"since": 0.0}}}
    gate = P.quiet_gate(row, FakeFleet(busy=busy), wait_s=10, clock=lambda: 100.0)
    assert gate["go"] and gate["note"] == ("proceeded after quiet-wait expiry; "
                                           "admission jobs j1 re-queue on restart")


def test_live_fleet_busy_reads_benchmark_and_admission(monkeypatch):
    answers = {
        "/api/llm/benchmark/status": (200, {"status": "running", "run_id": "r9",
                                            "progress": {"completed": 3, "total": 15}}),
        "/api/llm/admission?status=pending": (200, {"queue": [{"id": "j1", "status": "running",
                                                               "model_key": "M"}]}),
    }
    monkeypatch.setattr(P, "_http", lambda m, url, *a, **k: answers[url.split("7002", 1)[1]])
    monkeypatch.setattr(P, "api_key", lambda: None)
    busy = P.LiveFleet("http://127.0.0.1:7002").busy()
    assert busy["benchmark"] == "r9" and busy["admission"] == ["j1"]
    assert busy["reason"].startswith("benchmark r9 running (3/15); admission 1 running / 0 queued")
    answers["/api/llm/benchmark/status"] = (200, {"status": "complete"})
    answers["/api/llm/admission?status=pending"] = (200, {"queue": []})
    assert P.LiveFleet("http://127.0.0.1:7002").busy() is None
