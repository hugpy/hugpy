"""COMPUTE-ACTIVITY METRICS (2026-09-01) — the durable compute-action log,
the eviction-emit tap that feeds it, and the two read routes the Metrics panel
draws from.

The invariant every one of these guards is the model_metrics store's own:
the durable-log write is on the HOT COMPUTE PATH (tapped from the total
eviction emit and from record_load/record_call), so a store fault must be
INVISIBLE — never raise, never block, never change a compute outcome — and the
read routes must never 500 (empty payload + an error field on a fault).

    ./venv/bin/pytest tests/test_compute_actions.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MM = importlib.import_module("hugpy_fleet.central.model_metrics")
EV = importlib.import_module("hugpy_fleet.central.evictions")


@pytest.fixture()
def store(tmp_path):
    return MM.ModelMetricsStore(path=str(tmp_path / "model_metrics.db"))


# --------------------------------------------------------------------------- #
# append_action / recent_actions round-trip
# --------------------------------------------------------------------------- #

def test_append_then_read_roundtrip(store):
    assert store.append_action(
        "load", model="ModelA", variant="moe", worker_card="ae:0",
        duration_s=5.0, tok_per_s=25.0, tokens=400, outcome="loaded",
        detail={"engine": "llama.cpp"})
    rows = store.recent_actions(limit=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["action"] == "load"
    assert r["model"] == "ModelA"
    assert r["variant"] == "moe"
    assert r["worker_card"] == "ae:0"
    assert r["duration_s"] == 5.0
    assert r["tok_per_s"] == 25.0
    assert r["tokens"] == 400
    assert r["outcome"] == "loaded"
    assert r["detail"] == {"engine": "llama.cpp"}
    assert r["id"] > 0


def test_recent_actions_is_newest_first(store):
    for i in range(5):
        store.append_action("call", model=f"m{i}")
    rows = store.recent_actions(limit=10)
    assert [r["model"] for r in rows] == ["m4", "m3", "m2", "m1", "m0"]


def test_recent_actions_limit_returns_the_latest(store):
    for i in range(10):
        store.append_action("call", model=f"m{i}")
    rows = store.recent_actions(limit=3)
    assert [r["model"] for r in rows] == ["m9", "m8", "m7"]


def test_recent_actions_filters(store):
    store.append_action("load", model="A", worker_card="ae:0")
    store.append_action("evict", model="B", worker_card="computron:0")
    store.append_action("call", model="A", worker_card="ae:0")
    assert [r["action"] for r in store.recent_actions(action="load")] == ["load"]
    assert [r["model"] for r in store.recent_actions(model="A")] == ["A", "A"]
    # worker filter matches the NAME (part before ':') as well as the full card.
    assert len(store.recent_actions(worker="ae")) == 2
    assert len(store.recent_actions(worker="ae:0")) == 2
    assert len(store.recent_actions(worker="computron")) == 1


def test_recent_actions_since_ts_filters_by_time(store):
    import time
    store.append_action("call", model="old", ts=time.time() - 3600)
    store.append_action("call", model="new")
    got = store.recent_actions(since_ts=time.time() - 60)
    assert [r["model"] for r in got] == ["new"]


def test_append_requires_an_action(store):
    assert store.append_action("") is False
    assert store.append_action("   ") is False
    assert store.recent_actions() == []


# --------------------------------------------------------------------------- #
# THE hot-path contract: a log fault must never raise or change a record
# --------------------------------------------------------------------------- #

def test_append_never_raises_on_a_disabled_store():
    s = MM.ModelMetricsStore(path="/proc/definitely/not/writable/x.db")
    # A disabled store answers False, never raises, and reads empty.
    for _ in range(5):
        assert s.append_action("load", model="m") is False
    assert s.recent_actions() == []


def test_append_swallows_an_unserializable_detail(store):
    class Weird:
        def __repr__(self):
            raise RuntimeError("even repr is broken")

    # A bad detail must not drop the row, let alone raise: the row lands with a
    # null detail rather than failing a real compute action.
    assert store.append_action("evict", model="m", detail={"x": Weird()})
    rows = store.recent_actions()
    assert len(rows) == 1
    assert rows[0]["detail"] is None


def test_record_load_also_logs_a_load_action(store):
    assert store.record_load("m", "moe", "ae:0", "loaded",
                             upload_time_s=6.0, tok_per_s=20.0)
    rows = store.recent_actions(action="load")
    assert len(rows) == 1
    assert rows[0]["model"] == "m"
    assert rows[0]["duration_s"] == 6.0
    assert rows[0]["tok_per_s"] == 20.0
    # The EMA row is still written — the log is ADDITIVE, not a replacement.
    assert store.get_load("m", "moe", "ae:0", "loaded")["n_samples"] == 1


def test_record_call_also_logs_a_call_action(store):
    assert store.record_call("m", 400.0)
    rows = store.recent_actions(action="call")
    assert len(rows) == 1
    assert rows[0]["model"] == "m"
    assert rows[0]["tokens"] == 400


# --------------------------------------------------------------------------- #
# the eviction-emit tap — an emit produces a compute_actions row
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tapped_store(store, monkeypatch):
    """Point the module-level singleton the tap writes through at a tmp store,
    and clear the eviction ring/sinks around the test."""
    monkeypatch.setattr(MM, "model_metrics_store", store)
    EV.reset_for_tests()
    yield store
    EV.reset_for_tests()


def test_an_evict_emit_produces_a_compute_action_row(tapped_store):
    got = EV.emit_eviction_event("evict.done", model_key="m1",
                                 tier="in-process", freed_bytes=999,
                                 duration_ms=1500)
    # The emit itself is unaffected — it still returns the event.
    assert got["stage"] == "evict.done"
    rows = tapped_store.recent_actions()
    assert len(rows) == 1
    r = rows[0]
    assert r["action"] == "evict"
    assert r["model"] == "m1"
    assert r["duration_s"] == 1.5          # duration_ms -> seconds
    assert r["outcome"] == "ok"
    assert r["detail"]["tier"] == "in-process"
    assert r["detail"]["freed_bytes"] == 999


def test_a_load_fail_emit_maps_to_a_failed_load_action(tapped_store):
    EV.emit_eviction_event("load.fail", model_key="m2", engine="X",
                           error="SIGILL")
    r = tapped_store.recent_actions()[0]
    assert r["action"] == "load"
    assert r["outcome"] == "fail"


def test_stage_to_action_maps_the_charted_stages(tapped_store):
    for stage, action in [("load.done", "load"), ("evict.done", "evict"),
                          ("provision.done", "provision"),
                          ("fit.fail", "fit_fail"),
                          ("member.select", "member_select")]:
        tapped_store.append_action  # keep the store warm
        EV.emit_eviction_event(stage, model_key="k")
    seen = {r["action"] for r in tapped_store.recent_actions(limit=100)}
    assert {"load", "evict", "provision", "fit_fail", "member_select"} <= seen


def test_unmapped_stages_do_not_log(tapped_store):
    # start/heartbeat/verdict noise the panel doesn't chart must NOT log.
    for stage in ("load.start", "headroom.start", "reclaim.done",
                  "candidate.skip"):
        EV.emit_eviction_event(stage, model_key="k")
    assert tapped_store.recent_actions() == []


def test_the_emit_is_unaffected_by_a_broken_log(monkeypatch):
    """THE contract: a log fault is invisible to the compute path. A tap that
    raises must not stop the emit from returning its event."""
    def boom(*_a, **_k):
        raise RuntimeError("log exploded")

    monkeypatch.setattr(MM.model_metrics_store, "append_action", boom)
    EV.reset_for_tests()
    try:
        got = EV.emit_eviction_event("evict.done", model_key="m1")
        assert got is not None and got["stage"] == "evict.done"
    finally:
        EV.reset_for_tests()


# --------------------------------------------------------------------------- #
# read routes — response shapes, never 500
# --------------------------------------------------------------------------- #

@pytest.fixture()
def client(store, monkeypatch):
    flask = pytest.importorskip("flask")
    from hugpy_server.app.routes import metrics_routes as mr
    monkeypatch.setattr(MM, "model_metrics_store", store)
    app = flask.Flask(__name__)
    app.register_blueprint(mr.metrics_bp)
    return app.test_client()


def test_model_metrics_route_shape(client, store):
    store.record_load("ModelA", "moe", "ae:0", "loaded",
                      upload_time_s=5.0, tok_per_s=25.0)
    store.record_call("ModelA", 400.0)
    r = client.get("/llm/model-metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {"load_metrics", "call_metrics", "generated_at"}
    assert len(body["load_metrics"]) == 1
    assert len(body["call_metrics"]) == 1
    row = body["load_metrics"][0]
    assert set(row) == {"model", "variant", "worker_card", "temperature",
                        "upload_time_s", "tok_per_s", "n_samples", "updated_at"}
    assert row["model"] == "ModelA"
    assert body["call_metrics"][0]["n_calls"] == 1


def test_compute_actions_route_shape(client, store):
    store.append_action("evict", model="ModelB", worker_card="computron:0",
                        bytes=999, outcome="ok")
    store.append_action("load", model="ModelA", worker_card="ae:0",
                        duration_s=5.0, tok_per_s=25.0, outcome="loaded")
    r = client.get("/llm/compute-actions?limit=50")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {"actions", "count"}
    assert body["count"] == 2
    # Newest first.
    assert [a["action"] for a in body["actions"]] == ["load", "evict"]
    a0 = body["actions"][0]
    assert set(a0) >= {"id", "ts", "action", "model", "worker_card",
                       "duration_s", "tok_per_s", "outcome", "detail"}


def test_compute_actions_route_honors_filters(client, store):
    store.append_action("load", model="A", worker_card="ae:0")
    store.append_action("evict", model="B", worker_card="computron:0")
    assert client.get("/llm/compute-actions?action=load").get_json()["count"] == 1
    assert client.get("/llm/compute-actions?worker=computron").get_json()["count"] == 1
    assert client.get("/llm/compute-actions?model=A").get_json()["count"] == 1


def test_routes_never_500_on_a_store_fault(monkeypatch):
    """A broken store shows an empty panel with the reason attached, never a
    500 — a broken metrics PAGE must not look like a broken metrics FEATURE."""
    flask = pytest.importorskip("flask")
    from hugpy_server.app.routes import metrics_routes as mr

    class Boom:
        def _ensure(self):
            raise RuntimeError("store is on fire")

        def recent_actions(self, *a, **k):
            raise RuntimeError("store is on fire")

    monkeypatch.setattr(MM, "model_metrics_store", Boom())
    app = flask.Flask(__name__)
    app.register_blueprint(mr.metrics_bp)
    c = app.test_client()

    r1 = c.get("/llm/model-metrics")
    assert r1.status_code == 200
    b1 = r1.get_json()
    assert b1["load_metrics"] == [] and b1["call_metrics"] == []
    assert "error" in b1

    r2 = c.get("/llm/compute-actions")
    assert r2.status_code == 200
    b2 = r2.get_json()
    assert b2["actions"] == [] and b2["count"] == 0
    assert "error" in b2


def test_compute_actions_bad_query_params_degrade_gracefully(client, store):
    store.append_action("call", model="m")
    # Junk limit/since fall back to defaults rather than 400/500.
    r = client.get("/llm/compute-actions?limit=notanumber&since=nope")
    assert r.status_code == 200
    assert r.get_json()["count"] == 1
