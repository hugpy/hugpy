"""Every failed model load is ONE compute_actions row (2026-09-23).

action="load", outcome="fail", model, variant (the GGUF the loader opened),
worker_card, duration_s, detail = the worker's structured load_failure
({class, loader_stderr, path}) + message + phase/source. Fed by the /v1 relay
error path (per run) and by set_load_report (catch-all, deduped by report ts).
Queried with GET /llm/compute-actions?action=load&outcome=fail&model=<key>.
Sqlite backend throughout.
"""
import importlib
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MM = importlib.import_module("hugpy_fleet.central.model_metrics")
FP = importlib.import_module("hugpy_fleet.central.placement")
W = importlib.import_module("hugpy_fleet.central.workers")
remote = importlib.import_module("hugpy_engine.resolvers.remote")

KEY = "Echo-Mini"
PATH = "/mnt/llm/Echo-Mini/Echo-Mini.Q4_K_M.gguf"
STDERR = ("0.00.132.623 E llama_model_load: error loading model: check_tensor_dims: "
          "tensor 'token_embd.weight' has wrong shape; expected 4096, 32005, got 384, 32000, 1, 1")
LF = {"class": "hard_load_failure", "loader_stderr": STDERR, "path": PATH}
ERR = (f"HardLoadFailure: {KEY}: hard_load_failure — the native loader rejected {PATH} "
       f"(hard load failure; retrying cannot fix it, in-process fallback refused). "
       f"Loader stderr: {STDERR}")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    s = MM.ModelMetricsStore(path=str(tmp_path / "model_metrics.db"))
    monkeypatch.setattr(MM, "model_metrics_store", s)
    monkeypatch.setattr(MM, "_LOAD_FAIL_SEEN", set())
    return s


def _fails(store):
    return MM.recent_actions(store, action="load", outcome="fail", model=KEY)


def test_relay_worker_error_records_one_fail_row(store, monkeypatch):
    monkeypatch.setattr(remote._placement, "get_model_metrics",
                        lambda: FP.FleetModelMetrics(store=store))
    worker = {"id": "wid-ae", "name": "ae", "loaded_models": []}
    body = {"ok": False, "error": ERR, "load_failure": LF,
            "worker": {"id": "wid-ae", "name": "ae"}}
    remote._record_relay_load_failure(worker, KEY, body, time.monotonic() - 1.2)
    # A plain generation error (no load_failure) is NOT a load failure.
    remote._record_relay_load_failure(worker, KEY, {"ok": False, "error": "boom"}, None)
    rows = _fails(store)
    assert len(rows) == 1
    r = rows[0]
    assert (r["action"], r["outcome"], r["model"], r["worker_card"]) == \
        ("load", "fail", KEY, "ae:0")
    assert r["variant"] == "Echo-Mini.Q4_K_M.gguf"
    assert r["duration_s"] is not None and r["duration_s"] >= 1.2
    d = r["detail"]
    assert d["class"] == "hard_load_failure" and d["loader_stderr"] == STDERR
    assert d["path"] == PATH and d["message"] == ERR
    assert d["phase"] == "cold" and d["source"] == "relay"


def test_stream_error_event_records(store, monkeypatch):
    monkeypatch.setattr(remote._placement, "get_model_metrics",
                        lambda: FP.FleetModelMetrics(store=store))
    ev = {"type": "error", "message": ERR, "load_failure": LF}
    remote._record_relay_load_failure({"id": "c1", "name": "computron",
                                       "loaded_models": [KEY]}, KEY, ev, None)
    rows = _fails(store)
    assert len(rows) == 1 and rows[0]["worker_card"] == "computron:0"
    assert rows[0]["detail"]["phase"] == "hot"


def test_load_report_recorded_once_per_ts(store, tmp_path):
    ws = W.WorkerStore(path=str(tmp_path / "workers.json"))
    ws.register(name="ae", url="http://ae:9100", worker_id="wid-ae")
    report = {"ts": time.time() - 3.0, "ok": False, "fit": False,
              "error": ERR, "load_failure": LF}
    ws.set_load_report("wid-ae", KEY, dict(report))
    ws.set_load_report("wid-ae", KEY, dict(report))          # unchanged beat
    MM._LOAD_FAIL_SEEN.clear()                                # "restart"
    ws.set_load_report("wid-ae", KEY, dict(report))
    rows = _fails(store)
    assert len(rows) == 1
    d = rows[0]["detail"]
    assert d["class"] == "hard_load_failure" and d["source"] == "load_report"
    assert d["report_ts"] == pytest.approx(report["ts"])
    assert rows[0]["duration_s"] >= 3.0
    # A NEW attempt (new ts) is a new row; a success adds no fail row.
    ws.set_load_report("wid-ae", KEY, dict(report, ts=time.time()))
    ws.set_load_report("wid-ae", KEY, {"ts": time.time(), "ok": True})
    assert len(_fails(store)) == 2


def test_older_worker_text_is_classified(store):
    legacy = ("RuntimeError: slot load failed: RuntimeError: slot 1: Echo-Mini hard load "
              "failure: the llama-server child exited (code 1) after 1.0s without ever "
              f"serving — the loader rejected the load, not stalled; retrying cannot fix "
              f"it. Loader stderr: {STDERR} Attempt 1, backing off 30s")
    assert MM.record_load_report_failure({"name": "ae"}, KEY,
                                         {"ts": 1.0, "ok": False, "error": legacy})
    assert not MM.record_load_report_failure(
        {"name": "ae"}, KEY, {"ts": 2.0, "ok": False, "error": "not local — probe ..."})
    d = _fails(store)[0]["detail"]
    assert d["class"] == "hard_load_failure" and d["loader_stderr"] == STDERR


def test_outcome_filter_store_route_and_pg_shape(store, monkeypatch):
    store.append_action("load", model=KEY, worker_card="ae:0", outcome="loaded")
    MM.record_load_failure(KEY, "ae:0", load_failure=LF, message=ERR)
    assert [r["outcome"] for r in store.recent_actions(outcome="fail")] == ["fail"]

    class _PgLike:                     # a reader that predates ``outcome``
        def recent_actions(self, limit=200, *, since_ts=None, action=None,
                           model=None, worker=None):
            return store.recent_actions(limit=limit, since_ts=since_ts,
                                        action=action, model=model, worker=worker)
    got = MM.recent_actions(_PgLike(), action="load", outcome="fail", model=KEY)
    assert len(got) == 1 and got[0]["detail"]["class"] == "hard_load_failure"

    flask = pytest.importorskip("flask")
    from hugpy_server.app.routes import metrics_routes as mr
    app = flask.Flask(__name__)
    app.register_blueprint(mr.metrics_bp)
    c = app.test_client()
    body = c.get(f"/llm/compute-actions?action=load&outcome=fail&model={KEY}").get_json()
    assert body["count"] == 1 and body["actions"][0]["detail"]["loader_stderr"] == STDERR
    assert c.get(f"/llm/compute-actions?action=load&model={KEY}").get_json()["count"] == 2
