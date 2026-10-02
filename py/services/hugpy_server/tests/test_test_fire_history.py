"""Test-fire history (2026-10-02): runs/results persisted through
test_fire_store (fail-open), history/analysis routes, and the per-model
inference (verdict_for)."""
from __future__ import annotations

import importlib

from flask import Flask

tf = importlib.import_module("hugpy_server.app.routes.test_fire_routes")
store = importlib.import_module("hugpy_server.app.test_fire_store")


class FakeStore:
    def __init__(self):
        self.runs, self.results = {}, []

    def save_run(self, job):
        self.runs[job.job_id] = {"job_id": job.job_id, "worker_id": job.worker_id, "ok": job.ok,
                                 "failed": job.failed, "finished": job.finished,
                                 "stop_reason": job.stop_reason, "created": job.created}

    def save_result(self, job, result):
        self.results.append(dict(result, job_id=job.job_id))

    def list_runs(self, worker_id, limit):
        return [dict(r) for r in self.runs.values() if r["worker_id"] == worker_id][:limit]

    def get_run(self, job_id):
        r = self.runs.get(job_id)
        return dict(r, results=[x for x in self.results if x["job_id"] == job_id]) if r else None

    def analyze(self, worker_id, runs):
        return {"worker_id": worker_id, "runs": runs}


def _client(monkeypatch, fake):
    monkeypatch.setattr(tf, "_history", lambda: fake)
    tf.reset_registry()
    app = Flask(__name__)
    app.register_blueprint(tf.test_fire_bp)
    return app.test_client()


def test_a_run_and_its_results_are_persisted(monkeypatch):
    fake = FakeStore()
    client = _client(monkeypatch, fake)
    calls = lambda key, prompt, max_tokens, job: {"model_key": key, "ok": key != "B", "tokens": 3,
                                                  "error": None if key != "B" else "boom"}
    job = tf.start_job("w1", "box", ["A", "B"], call_fn=calls)
    job.thread.join(5)
    stored = fake.runs[job.job_id]
    assert stored["ok"] == 1 and stored["failed"] == 1 and stored["finished"]
    assert sorted(r["model_key"] for r in fake.results) == ["A", "B"]
    h = client.get("/llm/workers/w1/test-fire/history").get_json()
    assert h["runs"][0]["job_id"] == job.job_id and h["runs"][0]["state"] == "complete"
    run = client.get(f"/llm/workers/w1/test-fire/history/{job.job_id}").get_json()
    assert len(run["results"]) == 2
    assert client.get(f"/llm/workers/other/test-fire/history/{job.job_id}").status_code == 404
    assert client.get("/llm/workers/w1/test-fire/analysis?runs=5").get_json()["runs"] == 5


def test_unfinished_stored_run_reads_interrupted(monkeypatch):
    fake = FakeStore()
    fake.runs["old"] = {"job_id": "old", "worker_id": "w1", "finished": None, "stop_reason": None, "created": 1}
    client = _client(monkeypatch, fake)
    assert client.get("/llm/workers/w1/test-fire/history").get_json()["runs"][0]["state"] == "interrupted"


def test_history_failure_never_breaks_a_run(monkeypatch):
    class Broken(FakeStore):
        def save_result(self, job, result):
            raise RuntimeError("db down")
    _client(monkeypatch, Broken())
    job = tf.start_job("w1", "box", ["A"], call_fn=lambda k, p, m, j: {"model_key": k, "ok": True})
    job.thread.join(5)
    assert job.ok == 1 and job.stop_reason == "complete"


def _c(job, ok, lat=1.0, tps=50.0, kind=None):
    return {"job_id": job, "ok": ok, "latency_s": lat, "tok_s": tps if ok else None,
            "error_kind": kind, "started": 0}


def test_verdicts():
    assert store.verdict_for([], None)["verdict"] == "untested"
    healthy = [_c("j1", True)] * 4 + [_c("j2", True)]
    assert store.verdict_for(healthy, "j2")["verdict"] == "healthy"
    failing = [_c("j1", True)] * 3 + [_c("j2", False, kind="oom")] * 3
    v = store.verdict_for(failing, "j2")
    assert v["verdict"] == "failing" and v["error_kinds"] == {"oom": 3}
    regressed = [_c("j1", True, tps=50)] * 4 + [_c("j2", True, tps=20)]
    v = store.verdict_for(regressed, "j2")
    assert v["verdict"] == "regressed" and v["baseline"]["tok_s"] == 50 and v["latest"]["tok_s"] == 20
    slow = [_c("j1", True, lat=1.0)] * 4 + [_c("j2", True, lat=3.0)]
    assert store.verdict_for(slow, "j2")["verdict"] == "regressed"
    flaky = [_c("j1", True)] * 8 + [_c("j1", False)] + [_c("j2", True)]
    assert store.verdict_for(flaky, "j2")["verdict"] == "flaky"


def test_store_fails_open_without_dsn(monkeypatch):
    monkeypatch.setattr(store, "_dsn", lambda: None)
    assert store.list_runs("w1") == [] and store.get_run("x") is None
    assert store.analyze("w1")["models"] == {}
