"""model_liveness + pkg_promote: liveness is DERIVED (calls record, then GPU
state); the watcher/timer path never spends inference; a probe is an explicit
action, max_tokens 1, recorded as a 'grade' call, never repeated inside the
window, never while the queue holds the model. No network: a scripted http."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_liveness as ML  # noqa: E402
import pkg_promote as P  # noqa: E402

CENTRAL = "http://127.0.0.1:7002"
NOW = 1_800_000_000.0
W, M = "ae-worker", "Qwen3-Coder-Next-GGUF"


def worker_row(*, resident=True, busy=False, healthy=True, last_seen=NOW - 5, status="online",
               last_call=None):
    slots = [{"model_key": M, "busy": busy, "healthy": healthy, "child_pid": 1}] if resident else []
    return {"name": W, "status": status, "last_seen": last_seen, "slots": slots,
            "loaded_models": [M] if resident else [],
            "model_call_stats": {M: {"last_call": last_call, "tok_s_last": 26.2}} if last_call else {}}


class Http:
    """Scripted central. Records every POST; ``calls`` grows when a probe lands
    (central appends the graded row to the calls record)."""

    def __init__(self, calls=(), workers=(), queue=None, probe_status=200):
        self.calls, self.workers = list(calls), list(workers)
        self.queue = {"active": [], "waiting": [], "counts": {}} if queue is None else queue
        self.posts, self.probe_status = [], probe_status

    def __call__(self, method, url, body=None, headers=None, timeout=15):
        path = url.split("7002", 1)[1]
        if method == "POST":
            self.posts.append({"path": path, "body": body, "headers": headers or {}})
            if self.probe_status == 200:
                self.calls.insert(0, {"id": "v1-graded", "model": body["model"], "status": "done",
                                      "worker": body["alloc"]["worker"], "ts": NOW, "ended_ts": NOW + 1,
                                      "duration_ms": 900, "ua": headers.get("User-Agent")})
                return 200, {"choices": [{"message": {"content": "OK"}}]}
            return self.probe_status, {"error": "slot unavailable"}
        if path.startswith("/api/llm/calls"):
            return 200, {"calls": self.calls, "count": len(self.calls)}
        if path == "/api/llm/workers":
            return 200, self.workers
        if path == "/api/llm/queue":
            return 200, self.queue
        raise AssertionError(f"unexpected GET {path}")


def done_call(ts=NOW - 120, worker=W, model=M, status="done", **kw):
    return {"id": f"v1-{int(ts)}", "model": model, "status": status, "worker": worker, "ts": ts,
            "ended_ts": ts, "duration_ms": 11352, "ua": "someone-else", **kw}


# ------------------------------------------------------------------ the three sources

def test_processed_call_in_window_means_alive_no_probe():
    http = Http(calls=[done_call()], workers=[worker_row(resident=False)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is True and r["source"] == "calls"
    assert r["last_call_ts"] == NOW - 120 and r["latency_ms"] == 11352
    assert http.posts == []                                    # no inference spent


def test_pending_failed_old_or_other_worker_calls_do_not_count():
    rows = [done_call(status="pending"), done_call(status="failed"),
            done_call(ts=NOW - 2 * ML.LIVENESS_WINDOW_S), done_call(worker="computron")]
    assert ML.latest_processed_call(rows, M, W, NOW - ML.LIVENESS_WINDOW_S) is None
    legacy = done_call(worker=None)                            # old log line without a worker
    assert ML.latest_processed_call([legacy], M, W, NOW - ML.LIVENESS_WINDOW_S) is legacy


def test_resident_on_fresh_heartbeat_means_alive_no_probe():
    http = Http(workers=[worker_row(resident=True, busy=True)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is True and r["source"] == "gpu"
    assert r["resident"] is True and r["slot_state"] == "generating"
    assert http.posts == []


def test_worker_record_of_a_processed_call_counts_as_gpu_evidence():
    http = Http(workers=[worker_row(resident=False, last_call=NOW - 30)])
    r = ML.model_liveness(http, CENTRAL, W, M, now=NOW)
    assert r["alive"] is True and r["source"] == "gpu" and r["last_call_ts"] == NOW - 30


def test_stale_heartbeat_or_unhealthy_slot_is_not_alive_and_never_probed():
    http = Http(workers=[worker_row(last_seen=NOW - 999)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is False and r["source"] == "gpu" and "heartbeat" in r["why"]
    http = Http(workers=[worker_row(healthy=False)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is False and r["slot_state"] == "unhealthy"
    assert http.posts == []


def test_contract_fields_always_present():
    for http in (Http(), Http(calls=[done_call()]), Http(workers=[worker_row()])):
        r = ML.model_liveness(http, CENTRAL, W, M, now=NOW)
        assert set(r) >= {"alive", "source", "last_call_ts", "latency_ms", "resident",
                          "slot_state", "worker", "model", "why"}
        assert r["source"] in ("calls", "gpu", "probe", "unknown")


# ------------------------------------------------------------------ the probe, last resort

def test_nothing_on_record_and_no_permission_is_unknown_without_inference():
    http = Http(workers=[worker_row(resident=False)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=False, now=NOW)
    assert r["alive"] is None and r["source"] == "gpu" and r["resident"] is False
    assert "probe not allowed" in r["why"] and http.posts == []


def test_explicit_probe_is_one_max_tokens_1_call_recorded_as_grade():
    http = Http(workers=[worker_row(resident=False)])
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is True and r["source"] == "probe" and r["latency_ms"] is not None
    assert len(http.posts) == 1
    post = http.posts[0]
    assert post["path"] == "/v1/chat/completions"
    assert post["body"]["max_tokens"] == 1 and post["body"]["alloc"] == {"worker": W}
    assert post["headers"]["User-Agent"] == "hugpy-grade/1"
    assert post["headers"]["X-Hugpy-Client-Process"] == "grade"
    assert post["headers"]["X-Hugpy-Client-Task"] == f"grade:{W}:{M}"


def test_probe_never_runs_while_the_queue_holds_the_model():
    busy = {"active": [{"model": M, "id": "j1"}], "waiting": [], "counts": {}}
    http = Http(workers=[worker_row(resident=False)], queue=busy)
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is None and "probe deferred" in r["why"] and http.posts == []


def test_a_graded_row_is_the_persisted_verdict_no_second_sender_reprobes():
    # sender A must probe (nothing on record) …
    http = Http(workers=[worker_row(resident=False)])
    a = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert a["source"] == "probe" and len(http.posts) == 1
    # … sender B, inside the window, reads the graded call and never re-probes
    b = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW + 60)
    assert b["alive"] is True and b["source"] == "calls" and len(http.posts) == 1
    # repeat probing of a model with ANY successful call in the window is a bug
    for _ in range(5):
        ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW + 120)
    assert len(http.posts) == 1


def test_failed_probe_is_reported_not_retried_by_the_caller():
    http = Http(workers=[worker_row(resident=False)], probe_status=503)
    r = ML.model_liveness(http, CENTRAL, W, M, allow_probe=True, now=NOW)
    assert r["alive"] is False and r["source"] == "probe" and r["transient"] is True


# ------------------------------------------------------------------ pkg_promote's timer path

def guarded_fleet(monkeypatch, http):
    monkeypatch.setattr(P, "_http", http)
    monkeypatch.setattr(P, "api_key", lambda: "hp_test")
    return P.LiveFleet(CENTRAL)


def test_timer_path_baseline_and_judge_make_no_inference_call(monkeypatch):
    # The watcher's two call sites (start_rollout -> baseline, rollout_tick -> judge_worker):
    # a hot model is resident by definition, so the GPU state answers and NO generation
    # is ever sent — even though an API key is present.
    http = Http(workers=[worker_row(resident=True)])
    fleet = guarded_fleet(monkeypatch, http)
    base = P.baseline(fleet)
    assert base[W]["smoke_model"] == M and base[W]["pre_pin_smoke"][0]["source"] == "gpu"
    live = {"name": W, "status": "online", "last_seen": NOW - 1, "pkg_version": "V"}
    monkeypatch.setattr(P.time, "time", lambda: NOW)
    v = P.judge_worker(fleet, W, "V", base[W], live, {}, None, NOW, False)
    assert v["verdict"] == "pass" and v["smoke"]["source"] == "gpu"
    assert http.posts == []


def test_timer_path_with_a_recent_call_makes_no_inference_call(monkeypatch):
    http = Http(calls=[done_call()], workers=[worker_row(resident=True)])
    fleet = guarded_fleet(monkeypatch, http)
    base = P.baseline(fleet)
    live = {"name": W, "status": "online", "last_seen": NOW - 1, "pkg_version": "V"}
    monkeypatch.setattr(P.time, "time", lambda: NOW)
    v = P.judge_worker(fleet, W, "V", base[W], live, {}, None, NOW, False)
    assert v["verdict"] == "pass" and v["smoke"]["source"] == "calls"
    assert http.posts == []


def test_timer_path_with_nothing_on_record_stays_unjudged_without_inference(monkeypatch):
    # hot at pin time, evicted after the restart, no call in the window: the
    # watcher records 'unjudged' and never generates; the verdict is kept.
    http = Http(workers=[worker_row(resident=True)])
    fleet = guarded_fleet(monkeypatch, http)
    base = P.baseline(fleet)
    http.workers = [worker_row(resident=False)]
    live = {"name": W, "status": "online", "last_seen": NOW - 1, "pkg_version": "V"}
    monkeypatch.setattr(P.time, "time", lambda: NOW)
    v = P.judge_worker(fleet, W, "V", base[W], live, {}, None, NOW, False)
    assert v["verdict"] == "unjudged" and "not retried" in v["why"]
    again = P.judge_worker(fleet, W, "V", base[W], live, v, None, NOW + 60, False)
    assert again["verdict"] == "unjudged" and http.posts == []


def test_explicit_probe_on_judge_is_one_graded_call(monkeypatch):
    http = Http(workers=[worker_row(resident=False)])
    fleet = guarded_fleet(monkeypatch, http)
    base = {"online": True, "smoke_model": M}
    live = {"name": W, "status": "online", "last_seen": NOW - 1, "pkg_version": "V"}
    v = P.judge_worker(fleet, W, "V", base, live, {}, None, NOW, False, probe=True)
    assert v["verdict"] == "pass" and v["smoke"]["source"] == "probe"
    assert len(http.posts) == 1 and http.posts[0]["body"]["max_tokens"] == 1
    assert http.posts[0]["headers"]["Authorization"] == "Bearer hp_test"


def test_no_api_key_never_probes_even_when_asked(monkeypatch):
    http = Http(workers=[worker_row(resident=False)])
    monkeypatch.setattr(P, "_http", http)
    monkeypatch.setattr(P, "api_key", lambda: None)
    r = P.LiveFleet(CENTRAL).liveness(W, M, allow_probe=True)
    assert r["ok"] is False and r["alive"] is None
    assert http.posts == []


def test_no_chat_payload_is_built_in_the_watcher_module():
    # The old smoke built its own /v1/chat/completions payload here; the module
    # doc may recount the incident, but no code path may carry a prompt again.
    src = Path(P.__file__).read_text()
    assert '"content": "Reply with' not in src and "/v1/chat/completions" not in src
