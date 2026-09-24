"""A central restart (package promotion) must not end a running capacity
benchmark: the run state is persisted on every result row and a restarted
central resumes it — remaining lanes only, the in-flight lane restarted with a
recorded reason — unless the operator cancelled it first."""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

flask = pytest.importorskip("flask")

from hugpy_curation.review import fleet_grading as fg  # noqa: E402
from hugpy_server.app.routes import review_routes as rr  # noqa: E402

LANES = [{"model": "org/m", "worker": "aeb", "worker_id": "w1", "quant": q, "config": "standard",
          "alloc_mode": "gpu_only"} for q in ("A.gguf", "B.gguf", "C.gguf")]


class FakeLoop:
    """Stands in for run_capacity_benchmark: one complete row per lane not in
    ``done``; ``gate`` (optional) blocks after the first row it reports."""

    def __init__(self, gate=None):
        self.gate, self.calls, self.finished = gate, [], threading.Event()

    def __call__(self, client, workers, tokens, stop, report, done=None, cold_store=None, **kw):
        self.calls.append({"done": set(done or ()), **kw})
        report("plan", {"rows": LANES, "total": 3, "runnable": 3})
        for i, lane in enumerate(LANES):
            if fg.result_key(lane) in (done or ()):
                continue
            report("call", {**lane, "task": "factual (easy)"})
            report("result", {**lane, "status": "complete", "grade": "10/15", "score": 10, "max": 15})
            if self.gate is not None and i == 0:
                self.gate.wait(10)
        report("notice", "HugPy-native benchmark complete")
        self.finished.set()


class FakeClient:
    def __init__(self, *a, **k):
        pass

    def request(self, path, *a, **k):
        return {"workers": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = tmp_path / "benchmark_state.json"
    monkeypatch.setattr(rr, "_BENCHMARK_STATE", str(state))
    monkeypatch.setattr(rr, "_BENCHMARK_LEGACY_STATE", str(tmp_path / "legacy.json"))
    monkeypatch.setattr(rr, "_persist_benchmark_result", lambda row: True)
    monkeypatch.setattr(rr, "_persist_benchmark_call", lambda row: True)
    monkeypatch.setattr(fg, "Client", FakeClient)
    saved = dict(rr._BENCHMARK)
    rr._BENCHMARK.clear()
    rr._BENCHMARK.update({"status": "idle", "events": [], "results": [], "calls": []})
    app = flask.Flask(__name__)
    app.register_blueprint(rr.review_bp)
    yield state, app.test_client(), monkeypatch
    rr._BENCHMARK.clear(); rr._BENCHMARK.update(saved)


def _wait(pred, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _orphan(state, **extra):
    """The state file a killed central leaves: owner process gone, lane A done,
    lane B mid-flight (a call recorded, no result)."""
    body = {"run_id": "r1", "status": "running", "executor": "hugpy-central",
            "owner": {"host": socket.gethostname(), "pid": 2**22 + 7, "start": 1},
            "heartbeat": 1000.0, "updated": 1000.0, "started": 900.0, "tokens": 64,
            "params": {"tokens": 64, "models": ["org/m"], "workers": [], "suite": None,
                       "budgets": {"model_s": 99.0}, "resume": True, "force": False, "force_cold": False},
            "plan": {"rows": LANES, "total": 3, "runnable": 3},
            "progress": {"completed": 1, "total": 3, "percent": 33.3},
            "results": [{**LANES[0], "status": "complete", "grade": "12/15"}],
            "calls": [{**LANES[0], "task": "x"}, {**LANES[1], "task": "factual (easy)"}],
            "events": [], "summary": {}, **extra}
    state.write_text(json.dumps(body))
    rr._BENCHMARK.clear()
    rr._BENCHMARK.update({"status": "idle"})


def test_state_is_persisted_on_every_result_row(env):
    state, client, mp = env
    gate = threading.Event(); loop = FakeLoop(gate)
    mp.setattr(fg, "run_capacity_benchmark", loop)
    r = client.post("/llm/benchmark/run", json={"models": ["org/m"], "suite": None,
                                                "budgets": {"model_s": 99}, "resume": True,
                                                "force_cold": True})
    assert r.status_code == 202
    assert _wait(lambda: state.exists() and json.loads(state.read_text()).get("results"))
    saved = json.loads(state.read_text())
    assert saved["status"] == "running" and len(saved["results"]) == 1
    assert saved["plan"]["rows"] == LANES and saved["calls"]
    assert saved["params"]["budgets"] == {"model_s": 99} and saved["params"]["resume"] is True
    assert saved["params"]["force_cold"] is True and saved["owner"]["pid"]
    assert (state.stat().st_mode & 0o777) == 0o600
    gate.set()
    assert loop.finished.wait(10)
    assert _wait(lambda: json.loads(state.read_text())["status"] == "complete")
    assert len(json.loads(state.read_text())["results"]) == 3
    assert loop.calls[0]["force_cold"] is True


def test_resume_on_start_continues_remaining_lanes_with_restart_reason(env):
    state, client, mp = env
    loop = FakeLoop(); mp.setattr(fg, "run_capacity_benchmark", loop)
    _orphan(state)
    assert rr.start_benchmark_resume(base="http://127.0.0.1:9", wait_s=0.1) is True
    assert loop.finished.wait(10)
    assert _wait(lambda: rr._BENCHMARK.get("status") == "complete")
    assert loop.calls[0]["done"] == {fg.result_key(LANES[0])}
    assert loop.calls[0]["budgets"] == {"model_s": 99.0} and loop.calls[0]["resume"] is True
    body = client.get("/llm/benchmark/status").get_json()
    assert body["run_id"] == "r1" and body["status"] == "complete"
    assert body["interrupted_at"] == 1000.0
    assert body["resumed_from"]["run_id"] == "r1" and body["resumed_from"]["completed_rows"] == 1
    assert body["resumed_from"]["restarted_lane"][1] == "B.gguf"
    rows = {r["quant"]: r for r in body["results"]}
    assert set(rows) == {"A.gguf", "B.gguf", "C.gguf"}
    assert rows["B.gguf"]["restart_reason"] == "restarted after central restart"
    assert "restart_reason" not in rows["C.gguf"] and rows["A.gguf"]["grade"] == "12/15"
    assert any(e.get("kind") == "resume" for e in body["events"])


def test_status_poll_resumes_an_orphaned_run(env):
    state, client, mp = env
    loop = FakeLoop(); mp.setattr(fg, "run_capacity_benchmark", loop)
    _orphan(state)
    body = client.get("/llm/benchmark/status").get_json()
    assert body["status"] in ("resuming", "running", "complete") and body["resumed_from"]["run_id"] == "r1"
    assert loop.finished.wait(10)
    assert _wait(lambda: rr._BENCHMARK.get("status") == "complete")


def test_cancelled_run_is_not_resumed(env):
    state, client, mp = env
    loop = FakeLoop(); mp.setattr(fg, "run_capacity_benchmark", loop)
    _orphan(state, cancel_requested=True, cancelled_at=990.0)
    assert rr.benchmark_resume_orphaned() == "cancelled"
    saved = json.loads(state.read_text())
    assert saved["status"] == "cancelled" and "not resumed" in saved["error"] and loop.calls == []


def test_resume_is_bounded(env):
    state, client, mp = env
    loop = FakeLoop(); mp.setattr(fg, "run_capacity_benchmark", loop)
    _orphan(state, resume_count=rr._BENCHMARK_RESUME_MAX)
    assert rr.benchmark_resume_orphaned() == "interrupted"
    assert "not resumed: interrupted by" in json.loads(state.read_text())["error"] and loop.calls == []


def test_live_run_is_not_resumed_and_cancel_is_persisted(env):
    state, client, mp = env
    gate = threading.Event(); loop = FakeLoop(gate)
    mp.setattr(fg, "run_capacity_benchmark", loop)
    client.post("/llm/benchmark/run", json={"models": ["org/m"]})
    assert _wait(lambda: rr._BENCHMARK.get("control") is not None and rr._BENCHMARK.get("results"))
    assert rr.benchmark_resume_orphaned() is None           # owner (this process) is alive
    ck = client.post("/llm/benchmark/checkpoint", json={"reason": "promotion 11"}).get_json()
    assert ck["checkpointed"] is True and ck["rows"] == 1
    assert json.loads(state.read_text())["checkpoint"]["reason"] == "promotion 11"
    client.post("/llm/benchmark/cancel", json={"scope": "execution"})
    assert json.loads(state.read_text())["cancel_requested"] is True
    gate.set(); loop.finished.wait(10)
    assert _wait(lambda: rr._BENCHMARK.get("status") not in rr._BENCHMARK_ACTIVE)
