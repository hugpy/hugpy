"""Grading trigger (operator 2026-10-02): first successful load of a never-
attempted model queues it; the drain starts one benchmark at a time."""
from __future__ import annotations

import importlib

gt = importlib.import_module("hugpy_server.app.grading_trigger")
rr = importlib.import_module("hugpy_server.app.routes.review_routes")


class FakeDB:
    def __init__(self, known=()):
        self.rows = {}
        self.known = set(known)


def _wire(monkeypatch, db):
    monkeypatch.setattr(gt, "_known", set(db.known))
    monkeypatch.setenv("HUGPY_AUTO_GRADING", "1")

    def run(fn):
        class Cur:
            rowcount = 0
            def execute(self, sql, params=()):
                self.sql, self.params = sql, params
                if sql.startswith("INSERT"):
                    mk, wid, _s, _t = params[0], params[1], None, params[2]
                    if mk in db.rows:
                        self.rowcount = 0
                    else:
                        db.rows[mk] = {"worker_id": wid, "status": "queued", "queued_at": params[2], "run_id": None}
                        self.rowcount = 1
                elif "WHERE status='queued'" in sql:
                    q = sorted((r["queued_at"], k) for k, r in db.rows.items() if r["status"] == "queued")
                    self._one = (q[0][1], db.rows[q[0][1]]["worker_id"]) if q else None
                elif "WHERE status='running'" in sql:
                    self._all = [(k, r["run_id"]) for k, r in db.rows.items() if r["status"] == "running"]
                elif sql.startswith("UPDATE grading_attempts SET status='running'"):
                    db.rows[params[2]].update(status="running", run_id=params[0])
                elif sql.startswith("UPDATE grading_attempts SET status=%s"):
                    db.rows[params[2]]["status"] = params[0]
            def fetchone(self):
                return self._one
            def fetchall(self):
                return self._all
        return fn(Cur())
    monkeypatch.setattr(gt, "_run", run)


def test_first_success_queues_once_and_known_models_never(monkeypatch):
    db = FakeDB(known={"graded-before"})
    _wire(monkeypatch, db)
    assert gt.on_call_success("w1", "M", {}) is True
    assert gt.on_call_success("w1", "M", {}) is False          # already attempted
    assert gt.on_call_success("w1", "graded-before", {}) is False
    assert db.rows["M"]["status"] == "queued"


def test_disabled_and_db_down_queue_nothing(monkeypatch):
    db = FakeDB()
    _wire(monkeypatch, db)
    monkeypatch.setenv("HUGPY_AUTO_GRADING", "0")
    assert gt.on_call_success("w1", "M", {}) is False
    monkeypatch.setenv("HUGPY_AUTO_GRADING", "1")
    monkeypatch.setattr(gt, "_known", None)
    monkeypatch.setattr(gt, "_load_known", lambda: None)
    assert gt.on_call_success("w1", "M", {}) is False


def test_drain_starts_one_run_when_idle_and_settles_it(monkeypatch):
    db = FakeDB()
    _wire(monkeypatch, db)
    gt.on_call_success("w1", "A", {})
    gt.on_call_success("w2", "B", {})
    state = {"run_id": None, "status": "idle"}
    started = []
    monkeypatch.setattr(rr, "benchmark_state", lambda: dict(state))
    monkeypatch.setattr(rr, "benchmark_active", lambda: state["status"] in ("running",))

    def start(params):
        started.append((params["models"], params["workers"]))
        state.update(run_id=f"r{len(started)}", status="running")
        return True, {"run_id": state["run_id"]}
    monkeypatch.setattr(rr, "start_benchmark_run", start)
    assert gt.drain_once() == "A" and started == [(["A"], ["w1"])]
    assert gt.drain_once() is None                              # busy: nothing new
    state["status"] = "complete"
    assert gt.drain_once() == "B"
    assert db.rows["A"]["status"] == "done" and db.rows["B"]["status"] == "running"
    state["status"] = "error"
    gt.drain_once()
    assert db.rows["B"]["status"] == "failed"


def test_central_hook_seam_calls_on_success_only(monkeypatch):
    W = importlib.import_module("hugpy_fleet.central.workers")
    seen = []
    monkeypatch.setattr(W, "_CALL_SUCCESS_HOOKS", [lambda wid, mk, meta: seen.append((wid, mk))])
    monkeypatch.setattr(W.worker_store, "record_serve_metrics", lambda *a, **k: True)
    W.record_serve_metrics("w", "M", 10.0, outcome={"ok": True})
    W.record_serve_metrics("w", "N", None, outcome={"ok": False})
    assert seen == [("w", "M")]
