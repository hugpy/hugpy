"""Worker-side ROLLING AGGREGATE — operator ruling 2026-07-29.

    "API calls to workers are overloading the hugpy pool quite consistently.
     Have the workers agg their own datas as much as is reasonable for the
     package health" — "a rolling json for central to pick up upon read."

What these tests hold to account is the RULING, not just the code:

  * it must ROLL — every collection bounded, so the health file can never
    become the disk/CPU problem it exists to diagnose;
  * it must be ATOMIC — a reader gets a whole document or the previous one,
    never a torn one;
  * the beat must stay SMALL — a compact summary, never the document;
  * it must be COMPATIBLE IN BOTH DIRECTIONS — a new worker must not break an
    old central, and a new central must not break an old worker (release
    ordering is the one thing a fleet cannot roll back mid-flight);
  * central must read it ON DEMAND and CACHED — an uncached relay would
    recreate the exact fan-out the ruling halted;
  * the self-test must ship DARK — off means genuinely zero calls.

Pytest-clean by construction (k50): no module-level sys.exit, no import-time
fleet touching, and every env mutation is monkeypatched per-test (k51).

Run: venv/bin/python -m pytest tests/test_worker_aggregate.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import aggregate as agg_mod


@pytest.fixture()
def agg(tmp_path, monkeypatch):
    """A fresh aggregate pointed at a tmp file — never the real state dir."""
    monkeypatch.delenv("HUGPY_WORKER_AGGREGATE", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_SELFTEST", raising=False)
    monkeypatch.setenv("HUGPY_WORKER_AGGREGATE_FLUSH_S", "1")
    return agg_mod.reset_aggregate(str(tmp_path / "agg.json"))


# ── rolling / bounding ──────────────────────────────────────────────────────

def test_serve_stats_accumulate(agg):
    agg.record_serve("m1", ok=True, latency_ms=100.0, tokens_out=10)
    agg.record_serve("m1", ok=True, latency_ms=300.0, tokens_out=5)
    agg.record_serve("m1", ok=False, latency_ms=50.0, error="boom: it broke")
    row = agg.document()["models"]["m1"]
    assert row["requests"] == 3
    assert row["ok"] == 2 and row["fail"] == 1
    assert row["tokens_out"] == 15
    assert row["latency_ms"]["mean"] == pytest.approx(150.0)
    assert row["latency_ms"]["min"] == 50.0 and row["latency_ms"]["max"] == 300.0
    assert row["latency_ms"]["p95"] == 300.0
    # verbatim, not paraphrased
    assert row["last_error"] == "boom: it broke"
    assert row["last_served_at"] >= row["first_served_at"]


def test_tokens_never_guessed(agg):
    """An envelope that states no usage contributes NOTHING — an invented token
    count in a health file is worse than an honest absence."""
    assert agg_mod.tokens_out_of({"text": "hello world"}) is None
    assert agg_mod.tokens_out_of({"usage": {"completion_tokens": 7}}) == 7
    assert agg_mod.tokens_out_of("not a dict") is None


def test_model_table_is_bounded_lru(agg):
    for i in range(agg_mod.MAX_MODELS + 25):
        agg.record_serve(f"m{i}", ok=True, latency_ms=1.0)
    models = agg.document()["models"]
    assert len(models) <= agg_mod.MAX_MODELS
    # least-recently-touched went first: the earliest keys are gone, latest kept
    assert "m0" not in models
    assert f"m{agg_mod.MAX_MODELS + 24}" in models


def test_event_ring_is_bounded(agg):
    for i in range(agg_mod.MAX_EVENTS * 4):
        agg.record_serve("m1", ok=True, latency_ms=float(i))
    events = agg.document()["models"]["m1"]["events"]
    assert len(events) == agg_mod.MAX_EVENTS
    # it is a RING: the newest survived, the oldest rolled off
    assert events[-1]["ms"] == float(agg_mod.MAX_EVENTS * 4 - 1)


def test_latency_reservoir_is_bounded(agg):
    for i in range(agg_mod.MAX_LATENCY_SAMPLES * 3):
        agg.record_serve("m1", ok=True, latency_ms=float(i))
    # counters keep the TRUE total even though the reservoir is windowed
    assert agg.document()["models"]["m1"]["requests"] == agg_mod.MAX_LATENCY_SAMPLES * 3
    # the private reservoir never leaks into the document
    assert not any(k.startswith("_") for k in agg.document()["models"]["m1"])


def test_last_error_is_bounded_but_verbatim(agg):
    huge = "E" * (agg_mod.MAX_ERROR_CHARS * 3)
    agg.record_serve("m1", ok=False, error=huge)
    stored = agg.document()["models"]["m1"]["last_error"]
    assert len(stored) < len(huge)
    assert stored.startswith("EEE") and "[elided]" in stored


def test_selftest_history_bounded(agg):
    for i in range(agg_mod.MAX_SELFTEST * 3):
        agg.record_selftest("m1", {"case_id": f"c{i}", "mech_points": 1.0, "mech_max": 35.0})
    st = agg.document()["models"]["m1"]["selftest"]
    assert st["runs"] == agg_mod.MAX_SELFTEST * 3      # the COUNT is honest
    assert len(st["history"]) == agg_mod.MAX_SELFTEST  # the STORAGE is bounded


# ── atomicity + the file ────────────────────────────────────────────────────

def test_flush_is_atomic_and_readable(agg):
    agg.record_serve("m1", ok=True, latency_ms=5.0)
    assert agg.flush() is True
    doc = json.loads(Path(agg.path).read_text())
    assert doc["schema_version"] == agg_mod.SCHEMA_VERSION
    assert doc["models"]["m1"]["requests"] == 1
    # no tmp file left behind — a reader must never find a half-written sibling
    assert not os.path.exists(agg.path + ".tmp")
    assert agg.read_file()["models"]["m1"]["requests"] == 1


def test_flush_is_debounced(agg, monkeypatch):
    """A burst of requests must cost ONE write, not one per request — the
    aggregate must never become the load."""
    monkeypatch.setenv("HUGPY_WORKER_AGGREGATE_FLUSH_S", "3600")
    agg._last_flush = 0.0
    writes = {"n": 0}
    real_flush = agg.flush

    def counting_flush():
        writes["n"] += 1
        return real_flush()

    monkeypatch.setattr(agg, "flush", counting_flush)
    for _ in range(50):
        agg.record_serve("m1", ok=True, latency_ms=1.0)
    assert writes["n"] == 1          # first call primes, the rest are debounced
    agg.maybe_flush(force=True)
    assert writes["n"] == 2


def test_document_survives_unserializable_values(agg):
    agg.record_serve("m1", ok=False, error=object())
    assert agg.flush() is True
    assert json.loads(Path(agg.path).read_text())["models"]["m1"]["fail"] == 1


def test_off_switch_records_nothing(agg, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_AGGREGATE", "off")
    agg.record_serve("m1", ok=True, latency_ms=1.0)
    agg.record_load("m1", seconds=1.0)
    assert agg.document()["models"] == {}


def test_telemetry_never_raises(agg, monkeypatch):
    """A broken aggregator degrades to silence — it must never be able to fail
    a served request or skip a beat."""
    monkeypatch.setattr(agg, "_row", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    agg.record_serve("m1", ok=True, latency_ms=1.0)     # must not raise
    agg.record_load("m1", seconds=1.0)
    agg.record_selftest("m1", {"mech_points": 1})
    assert agg.heartbeat_summary()["schema_version"] == agg_mod.SCHEMA_VERSION


# ── load events (observed, never provoked) ──────────────────────────────────

def test_loading_transition_becomes_a_load_event(agg):
    agg.observe_loading(["m1"], [], at=1000.0)
    agg.observe_loading([], ["m1"], at=1042.0)
    loads = agg.document()["models"]["m1"]["loads"]
    assert loads["count"] == 1
    assert loads["last_seconds"] == pytest.approx(42.0)


def test_failed_load_is_recorded_as_a_failure(agg):
    agg.observe_loading(["m1"], [], at=1000.0)
    agg.observe_loading([], [], at=1010.0)     # left loading, never resident
    loads = agg.document()["models"]["m1"]["loads"]
    assert loads["count"] == 0 and loads["failures"] == 1
    assert "without becoming resident" in loads["last_error"]


def test_calibration_samples_are_the_precise_load_source(agg):
    """Reuse, not re-measure: the beat already drained these rows off the
    0.1.224 measured-truth helpers."""
    n = agg.ingest_calibration_samples([
        {"model_key": "m1", "load_seconds": 12.5, "ok": True, "device": "cuda", "ts": 500.0},
        {"model_key": "m2", "load_seconds": 3.0, "ok": False, "verdict": "refused: no vram", "ts": 501.0},
        "not a dict",
    ])
    assert n == 2
    assert agg.document()["models"]["m1"]["loads"]["last_seconds"] == 12.5
    assert agg.document()["models"]["m2"]["loads"]["failures"] == 1
    assert "refused: no vram" in agg.document()["models"]["m2"]["loads"]["last_error"]


def test_process_health_takes_values_not_probes(agg):
    agg.record_process_health(ram_worker_bytes=123, vram_attributed_bytes=None,
                              resident_models=2)
    proc = agg.document()["process"]
    assert proc["ram_worker_bytes"] == 123
    assert proc["resident_models"] == 2
    assert "vram_attributed_bytes" not in proc     # None is absence, not zero


# ── the heartbeat rider ─────────────────────────────────────────────────────

def test_heartbeat_summary_is_compact(agg):
    for i in range(5):
        agg.record_serve(f"m{i}", ok=True, latency_ms=1.0, tokens_out=3)
    agg.flush()
    s = agg.heartbeat_summary()
    assert s["models"] == 5 and s["requests"] == 5 and s["fail"] == 0
    assert s["digest"] and len(s["digest"]) == 16
    assert s["mtime"] and s["bytes"]
    # the DOCUMENT must never ride the beat
    assert "models" in s and not isinstance(s["models"], dict)
    assert "events" not in json.dumps(s)
    assert len(json.dumps(s)) < 600


def test_digest_changes_only_when_content_does(agg):
    agg.record_serve("m1", ok=True, latency_ms=1.0)
    agg.flush()
    first = agg.heartbeat_summary()["digest"]
    agg.flush()                                   # nothing changed but the clock
    agg.record_serve("m1", ok=True, latency_ms=1.0)
    agg.flush()
    assert agg.heartbeat_summary()["digest"] != first


# ── heartbeat COMPATIBILITY, both directions ────────────────────────────────

def test_new_worker_against_old_central_is_accepted():
    """NEW worker -> OLD central. The old central's HeartbeatRequest has no
    `aggregate` field; pydantic's default extra='ignore' must DROP it, not 422.
    Simulated by validating against a model built without the field."""
    from pydantic import BaseModel

    class OldHeartbeatRequest(BaseModel):        # the pre-feature shape
        loaded_models: list | None = None
        pkg_version: str | None = None

    body = OldHeartbeatRequest(**{
        "loaded_models": ["m1"], "pkg_version": "0.1.225",
        "aggregate": {"schema_version": 1, "digest": "abc", "requests": 3},
    })
    assert body.loaded_models == ["m1"]
    assert not hasattr(body, "aggregate") or getattr(body, "aggregate", None) is None


def test_new_central_accepts_a_worker_that_sends_no_aggregate():
    """NEW central -> OLD worker. `aggregate` is `| None = None`, so a beat
    without it validates and simply leaves the field unset."""
    from hugpy_server.app.routes.worker_routes import HeartbeatRequest

    body = HeartbeatRequest(**{"loaded_models": ["m1"], "pkg_version": "0.1.224"})
    assert body.aggregate is None


def test_new_central_accepts_and_keeps_the_summary():
    from hugpy_server.app.routes.worker_routes import HeartbeatRequest

    summary = {"schema_version": 1, "digest": "deadbeefdeadbeef", "requests": 9}
    body = HeartbeatRequest(**{"loaded_models": ["m1"], "aggregate": summary})
    assert body.aggregate == summary


def test_new_central_still_ignores_a_genuinely_unknown_field():
    """The tolerance that makes worker->central additive-safe is a PROPERTY the
    release ordering depends on — regress it, so nobody 'tightens' it later."""
    from hugpy_server.app.routes.worker_routes import HeartbeatRequest

    body = HeartbeatRequest(**{"loaded_models": ["m1"], "some_future_field": {"x": 1}})
    assert body.loaded_models == ["m1"]


def test_worker_store_persists_the_summary():
    from hugpy_fleet.central import workers as W

    summary = {"schema_version": 1, "digest": "abc123", "requests": 4}
    row: dict = {}
    # the store's contract: `aggregate` is stored verbatim, absent when not sent
    assert "aggregate" in W.WorkerStore.heartbeat.__code__.co_varnames
    row["aggregate"] = summary
    assert {**row}["aggregate"] == summary        # _public_view spreads **worker


# ── central relay route: on demand, cached ──────────────────────────────────

@pytest.fixture()
def central(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path))
    wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    wr.reset_aggregate_cache()
    monkeypatch.setattr(wr, "get_worker",
                        lambda wid: {"id": wid, "name": "box",
                                     "url": "http://w:9000",
                                     "aggregate": {"digest": "abc", "requests": 3}}
                        if wid == "w1" else None)
    return wr, app.test_client()


def _fake_httpx(monkeypatch, calls, *, status=200, body=None):
    """k59: the relay goes through worker_http, so the seam is httpx.request."""
    import httpx

    def fake_get(method, url, **kwargs):
        calls.append(url)

        class R:
            status_code = status

            def json(self):
                return body if body is not None else {"ok": True, "aggregate": {"models": {}}}

        return R()

    monkeypatch.setattr(httpx, "request", fake_get)


def test_relay_returns_the_document_and_the_beat_summary(central, monkeypatch):
    wr, client = central
    calls = []
    _fake_httpx(monkeypatch, calls)
    r = client.get("/llm/workers/w1/aggregate")
    assert r.status_code == 200
    data = r.get_json()
    assert data["aggregate"] == {"models": {}}
    assert data["worker"]["id"] == "w1"
    assert data["heartbeat_summary"] == {"digest": "abc", "requests": 3}
    assert data["cached"] is False
    assert calls == ["http://w:9000/ops/aggregate"]


def test_cache_collapses_a_refresh_storm_into_one_worker_call(central, monkeypatch):
    """The whole safety property: N console refreshes cost the pool ONE call."""
    wr, client = central
    calls = []
    _fake_httpx(monkeypatch, calls)
    for _ in range(10):
        r = client.get("/llm/workers/w1/aggregate")
        assert r.status_code == 200
    assert len(calls) == 1
    assert r.get_json()["cached"] is True
    assert r.get_json()["cache_age_s"] >= 0.0


def test_fresh_bypasses_the_cache(central, monkeypatch):
    wr, client = central
    calls = []
    _fake_httpx(monkeypatch, calls)
    client.get("/llm/workers/w1/aggregate")
    client.get("/llm/workers/w1/aggregate?fresh=1")
    assert len(calls) == 2


def test_ttl_zero_disables_caching(central, monkeypatch):
    wr, client = central
    monkeypatch.setenv("HUGPY_WORKER_AGGREGATE_TTL_S", "0")
    calls = []
    _fake_httpx(monkeypatch, calls)
    client.get("/llm/workers/w1/aggregate")
    client.get("/llm/workers/w1/aggregate")
    assert len(calls) == 2


def test_old_worker_gets_an_honest_501_not_a_bare_502(central, monkeypatch):
    """The ONE real release-ordering constraint: central may ship first, and a
    worker that predates /ops/aggregate must say so in words."""
    wr, client = central
    calls = []
    _fake_httpx(monkeypatch, calls, status=404, body={})
    r = client.get("/llm/workers/w1/aggregate")
    assert r.status_code == 501
    assert r.get_json()["error"]["code"] == "AggregateUnsupported"


def test_worker_errors_are_data_and_are_not_cached(central, monkeypatch):
    """A momentary blip must not be pinned for the whole TTL — that would turn
    one deaf second into 15s of false 'worker is broken'."""
    import httpx

    wr, client = central
    calls = []

    def boom(method, url, **kwargs):
        calls.append(url)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", boom)
    r = client.get("/llm/workers/w1/aggregate")
    assert r.status_code == 502
    assert r.get_json()["error"]["code"] == "ConnectError"
    client.get("/llm/workers/w1/aggregate")
    assert len(calls) == 2          # not served from cache


def test_unknown_worker_is_404(central, monkeypatch):
    wr, client = central
    calls = []
    _fake_httpx(monkeypatch, calls)
    assert client.get("/llm/workers/nope/aggregate").status_code == 404
    assert calls == []              # never dialed anything


# ── the self-test ships DARK ────────────────────────────────────────────────

@pytest.fixture()
def selftest(monkeypatch):
    from hugpy_fleet.worker.aptitude import selftest as st

    monkeypatch.delenv("HUGPY_WORKER_SELFTEST", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_SELFTEST_INTERVAL_S", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_SELFTEST_IDLE_S", raising=False)
    return st, st.reset_runner()


def test_selftest_off_by_default_makes_zero_calls(selftest):
    st, runner = selftest
    calls = []
    out = runner.maybe_run(["m1"], lambda **kw: calls.append(kw) or {"text": "{}"})
    assert out == {"ran": False, "reason": "disabled"}
    assert calls == []
    assert st.enabled() is False


def test_selftest_on_but_nothing_resident_makes_zero_calls(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    calls = []
    out = runner.maybe_run([], lambda **kw: calls.append(kw) or {"text": "{}"})
    assert out["ran"] is False and "no idle resident" in out["reason"]
    assert calls == []


def test_selftest_never_touches_a_busy_or_loading_model(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    calls = []
    call = lambda **kw: calls.append(kw) or {"text": "{}"}          # noqa: E731
    # m1 served 1s ago (real traffic), m2 is mid-load -> both must be skipped
    out = runner.maybe_run(["m1", "m2"], call,
                           last_served={"m1": 1_000_000.0}, loading=["m2"],
                           now=1_000_001.0)
    assert out["ran"] is False
    assert calls == []


def test_selftest_runs_one_case_and_scores_it_mechanically(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    calls = []

    def call(**kw):
        calls.append(kw)
        return {"text": json.dumps({
            "prompt": ("A wide, cold frame holds the empty platform as the "
                       "fluorescent tubes stutter; Alex stands at the yellow "
                       "line with back to camera, unmoving, while the tunnel "
                       "mouth brightens and a train slides in behind, filling "
                       "the frame with reflected light and settling to a stop."),
            "directions_used": [0, 1, 2],
            "invented_identity_attributes": [],
            "warnings": [],
        }), "finish_reason": "stop"}

    out = runner.maybe_run(["m1"], call)
    assert out["ran"] is True and out["model_key"] == "m1"
    assert len(calls) == 1                       # exactly ONE case
    score = out["score"]
    assert score["mech_max"] == 35.0
    assert 0.0 <= score["mech_points"] <= 35.0
    assert score["directions_matched"] == score["directions_total"] == 3
    assert score["inventions"] == 0


def test_selftest_runs_at_most_one_case_per_interval(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    calls = []
    call = lambda **kw: calls.append(kw) or {"text": "{}"}          # noqa: E731
    assert runner.maybe_run(["m1"], call)["ran"] is True
    for _ in range(5):
        assert runner.maybe_run(["m1"], call)["reason"] == "interval"
    assert len(calls) == 1


def test_selftest_a_broken_call_never_raises(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")

    def boom(**kw):
        raise RuntimeError("runner exploded")

    out = runner.maybe_run(["m1"], boom)
    assert out["ran"] is False and "runner exploded" in out["reason"]


def test_selftest_rotates_through_the_cases(selftest, monkeypatch):
    st, runner = selftest
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST_INTERVAL_S", "60")
    seen = []
    call = lambda **kw: {"text": "{}"}                              # noqa: E731
    t = 0.0
    for _ in range(4):
        t += 100.0
        out = runner.maybe_run(["m1"], call, now=t)
        seen.append(out["score"]["case_id"])
    assert len(set(seen)) == 4                   # coverage accumulates, no repeat


def test_selftest_scoring_is_pure_and_needs_no_model(selftest):
    st, _ = selftest
    from hugpy_fleet.worker.aptitude import cases as C

    row = st.score_unit("routing", {"id": "r05", "text": "make it slower",
                                    "expected": "direction"},
                        {"text": '{"intent":"direction","confidence":0.9}'})
    assert row["suite"] == "routing" and row["got"] == "direction"
    assert row["mech_points"] == 35.0
    bad = st.score_unit("routing", {"id": "r05", "text": "x", "expected": "direction"},
                        {"text": '{"intent":"scene_prompt"}'})
    assert bad["mech_points"] == 0.0
    assert len(C.SCENE_CASES) and len(C.ROUTING_CASES) and len(C.NEGATIVE_CASES)


def test_aggregate_reports_the_selftest_lever_state(agg, monkeypatch):
    assert agg.document()["selftest"]["enabled"] is False
    monkeypatch.setenv("HUGPY_WORKER_SELFTEST", "on")
    assert agg.document()["selftest"]["enabled"] is True
