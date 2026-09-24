"""Cold grading prepares state but lets ordinary inference provision it."""
from unittest.mock import Mock, call

from hugpy_curation.review import fleet_grading
from hugpy_platform.constants import DEFAULT_AGENT_BRAIN


def test_cold_reset_only_evicts_memory_and_worker_cache():
    client = Mock()
    client.request.side_effect = [{"evicted": True}, {"removed": True}]
    lane = {"worker_id": "w/1", "model": "org/model"}

    result = fleet_grading._cold_reset(client, lane)

    assert client.request.call_args_list == [
        call("/llm/workers/w%2F1/evict", "POST", {"model_key": "org/model", "force": True}),
        call("/llm/workers/w%2F1/cache-evict", "POST", {"model_key": "org/model"}),
    ]
    assert result == {"evict": {"evicted": True},
                      "cache_evict": {"removed": True}}


# ── recorded cold loads (operator 2026-09-23: reset only to MEASURE, once) ──
import threading  # noqa: E402
import time  # noqa: E402

FAST = {"call_s": 5.0, "cold_load_cap_s": 5.0, "model_s": 30.0}


def _worker():
    return {"id": "w1", "name": "aeb", "status": "online", "admission": "approved", "serve_mode": "on",
            "max_vram_bytes": 10**11, "max_ram_bytes": 10**11, "loaded_models": []}


def _catalog(model="org/m"):
    return [{"model_key": model, "primary_task": "text-generation", "tasks": ["text-generation"],
             "framework": "gguf", "size_bytes": 10**9, "workers": [{"worker_id": "w1", "designated": True}]}]


class Store:
    def __init__(self, rec=None, rate=None):
        self.rec, self.rate, self.asked = rec, rate, []

    def get(self, model, quant, worker):
        self.asked.append((model, quant, worker))
        return self.rec

    def worker_rate(self, worker):
        return self.rate


def _run(store, **kw):
    calls, events = [], []

    def request(path, method="GET", body=None, timeout=None):
        calls.append((path, method, body))
        if path.startswith("/models"): return {"models": _catalog()}
        if path.startswith("/llm/serving/"): return {}
        if path.startswith("/llm/evictions"):
            now = time.time()
            return {"events": [
                {"stage": "provision.done", "model_key": "org/m", "worker_id": "w1", "ts": now,
                 "bytes": 4 * 10**9, "duration_ms": 40000},
                {"stage": "load.done", "model_key": "org/m", "worker_id": "w1", "ts": now, "duration_ms": 9000},
                {"stage": "provision.done", "model_key": "org/other", "worker_id": "w1", "bytes": 1, "duration_ms": 1}]}
        if path.startswith("/llm/workers/"): return {"ok": True}
        return {"choices": [{"message": {"content": "37"}}]}
    client = Mock(); client.request.side_effect = request
    fleet_grading.run_capacity_benchmark(client, [_worker()], 16, threading.Event(),
                                         lambda k, v: events.append((k, v)), budgets=FAST,
                                         cold_store=store, **kw)
    results = [v for k, v in events if k == "result" and v.get("status") == "complete"]
    plan = next(v for k, v in events if k == "plan")
    resets = [p for p, _m, _b in calls if p.endswith("/evict") or p.endswith("/cache-evict")]
    return results, plan, resets, events


def test_first_run_measures_and_carries_the_split_for_recording():
    results, plan, resets, _ = _run(Store())
    assert resets == ["/llm/workers/w1/evict", "/llm/workers/w1/cache-evict"]
    assert {r["cold"] for r in plan["rows"]} == {"will measure"}
    first = results[0]
    assert first["cold_source"] == "measured" and first["cold_measured"] is True
    assert (first["transfer_s"], first["transfer_bytes"], first["load_s"]) == (40.0, 4 * 10**9, 9.0)
    assert first["bytes_per_s"] == 10**8 and isinstance(first["cold_s"], float)


def test_second_run_reuses_the_recorded_cold_load_and_skips_the_reset():
    at = time.mktime((2026, 9, 23, 12, 0, 0, 0, 0, -1))
    store = Store({"cold_load_s": 174.2, "measured_at": at, "transfer_s": 40.0, "bytes_per_s": 1e8})
    results, plan, resets, events = _run(store)
    assert resets == []                                     # no evict, no cache-evict
    assert {r["cold"] for r in plan["rows"]} == {"recorded 174 s (2026-09-23)"}
    assert all(r["cold_s"] == 174.2 and r["cold_measured"] is False for r in results)
    assert results[0]["cold_source"] == "recorded 174 s (2026-09-23)" and results[0]["transfer_s"] == 40.0
    notices = [v for k, v in events if k == "notice" and isinstance(v, dict) and v.get("phase") == "cold-reset"]
    assert notices and notices[0]["skipped"].startswith("cold load recorded 174 s")
    assert store.asked == [("org/m", "runtime default", "aeb")]


def test_force_cold_re_measures():
    store = Store({"cold_load_s": 174.2, "measured_at": time.time()})
    results, plan, resets, _ = _run(store, force_cold=True)
    assert resets and {r["cold"] for r in plan["rows"]} == {"will measure (force_cold)"}
    assert results[0]["cold_source"] == "measured"


def test_cold_budget_derives_from_the_recorded_worker_rate(monkeypatch):
    b = fleet_grading.bench_budgets({"cold_load_cap_s": 1000.0})
    lane = {"worker_record": {}, "size_bytes": 10**10}
    assert fleet_grading._cold_budget(lane, b) == 1000.0                    # no rate: the cap
    assert fleet_grading._cold_budget(lane, b, recorded_rate=10**8) == 230.0  # 2x10e9/1e8 + 30
    assert fleet_grading._cold_budget({"worker_record": {}, "size_bytes": 10**8}, b, 10**8) == 60.0  # floor
    # heartbeat rate (worker's own load_bytes_per_s) wins over the recorded one
    assert fleet_grading._cold_budget({"worker_record": {"load_bytes_per_s": 2 * 10**8}, "size_bytes": 10**10},
                                      b, recorded_rate=10**8) == 130.0
    seen = []
    real = fleet_grading._cold_budget
    monkeypatch.setattr(fleet_grading, "_cold_budget",
                        lambda lane, budgets, rate=None: seen.append(rate) or real(lane, budgets, rate))
    _run(Store(rate=123456.0))
    assert seen and set(seen) == {123456.0}               # plan budget + the lane's seat budget


def test_done_rows_are_skipped_on_resume():
    results, plan, resets, events = _run(Store())
    done = {fleet_grading.result_key(r) for r in results[:1]}
    again, _plan, _resets, events2 = _run(Store(), done=done)
    keys = {fleet_grading.result_key(r) for r in again}
    assert keys and not keys & done
    last = [v for k, v in events2 if k == "progress"][-1]
    assert last["completed"] == last["total"]              # skipped rows still count as completed


def test_legacy_ema_cold_value_is_not_a_measurement():
    # no measured_at (the row has no cold_measured_at): legacy EMA value -> re-measure
    results, plan, resets, _ = _run(Store({"cold_load_s": 7.85}))
    assert resets and {r["cold"] for r in plan["rows"]} == {"will measure"}
    assert results[0]["cold_source"] == "measured"


def test_implausibly_fast_recorded_cold_is_discarded(caplog):
    # 10**9 bytes in 0.1 s = 10 GB/s > 5 GB/s: bogus, logged, re-measured
    store = Store({"cold_load_s": 0.1, "measured_at": time.time()})
    with caplog.at_level("WARNING"):
        results, plan, resets, _ = _run(store)
    assert resets and results[0]["cold_source"] == "measured"
    assert any("implausible" in r.message for r in caplog.records)


def test_known_rate_budget_is_not_held_to_the_300s_cap():
    b = fleet_grading.bench_budgets()                       # cap 300, max 1800
    big = {"worker_record": {}, "size_bytes": 5 * 10**10}
    assert fleet_grading._cold_budget(big, b) == 300.0      # no rate: the fixed cap
    assert fleet_grading._cold_budget(big, b, recorded_rate=10**8) == 1030.0
    assert fleet_grading._cold_budget({"worker_record": {"load_bytes_per_s": 10**8}, "size_bytes": 5 * 10**10},
                                      b) == 1030.0
    assert fleet_grading._cold_budget({"worker_record": {}, "size_bytes": 10**12}, b, 10**8) == 1800.0
    assert fleet_grading.bench_budgets({"cold_load_max_s": 900})["cold_load_max_s"] == 900.0


def test_model_budget_never_undercuts_the_computed_need():
    b = fleet_grading.bench_budgets()                        # model_s 600, call_s 120, max 3600
    cold = fleet_grading._cold_budget({"worker_record": {}, "size_bytes": 5 * 10**10}, b, recorded_rate=10**8)
    assert cold == 1030.0
    assert fleet_grading.model_budget(b, cold, 18) == 3220.0  # 1030 + 18x120 + 30, not 600
    assert fleet_grading.model_budget(b, 60.0, 1) == 600.0    # model_s stays the floor
    assert fleet_grading.model_budget(b, 1800.0, 30) == 3600.0  # capped by model_max_s


def test_plan_rows_carry_the_effective_model_budget():
    _results, plan, _resets, _ = _run(Store(rate=10**8))      # 1 GB, 18 text items, FAST budgets
    b = fleet_grading.bench_budgets(FAST)
    from hugpy_curation.review.suites import suite_for_model
    items = sum(len(t) for t in suite_for_model(_catalog()[0]).tasks.values())
    want = fleet_grading.model_budget(b, fleet_grading._cold_budget({"worker_record": {}, "size_bytes": 10**9}, b, 10**8), items)
    assert {r["budget_model_s"] for r in plan["rows"]} == {round(want, 1)} and want > FAST["model_s"]


def test_cold_split_prefers_the_transfer_ledger_then_events():
    lane = {"model": "org/m", "worker": "aeb", "worker_id": "w1"}
    since = 1000.0
    events = {"events": [
        {"stage": "provision.done", "model_key": "org/m", "worker_id": "w1", "bytes": 4 * 10**9, "duration_ms": 40000},
        {"stage": "load.done", "model_key": "org/m", "worker_id": "w1", "duration_ms": 9000}]}
    ledger = {"transfers": [
        {"model_key": "org/m", "worker": "aeb", "worker_id": "x", "status": "complete",
         "started_at": 1002.0, "finished_at": 1022.0, "bytes_served": 4 * 10**9, "total_bytes": 4 * 10**9},
        {"model_key": "org/m", "worker": "aeb", "status": "complete",          # older than the seat
         "started_at": 500.0, "finished_at": 900.0, "bytes_served": 1},
        {"model_key": "org/other", "worker": "aeb", "status": "complete", "started_at": 1003.0,
         "finished_at": 1004.0, "bytes_served": 1}]}
    seen = []

    def req(transfers):
        def request(path, *a, **k):
            seen.append(path)
            return transfers if path.startswith("/llm/transfers") else events
        return Mock(request=Mock(side_effect=request))
    out = fleet_grading._cold_split(req(ledger), lane, since)
    assert out["transfer_source"] == "ledger" and out["transfer_s"] == 20.0
    assert out["transfer_bytes"] == 4 * 10**9 and out["bytes_per_s"] == 2 * 10**8 and out["load_s"] == 9.0
    assert seen[0] == "/llm/transfers?model=org%2Fm&worker=w1"
    out = fleet_grading._cold_split(req({"transfers": []}), lane, since)   # no ledger entry
    assert out["transfer_source"] == "events" and out["transfer_s"] == 40.0 and out["bytes_per_s"] == 10**8


def test_history_and_call_rows_carry_expected_actual_why():
    answers = {"Compute 15 + 22": "37", "Compute 17 * 23": "I think 390"}

    def request(path, method="GET", body=None, timeout=None):
        if path.startswith("/models"): return {"models": _catalog()}
        if path.startswith(("/llm/serving/", "/llm/evictions", "/llm/transfers")): return {}
        if path.startswith("/llm/workers/"): return {"ok": True}
        text = body["messages"][0]["content"]
        # The fallback (factual) reply must exceed ACTUAL_MAX so the truncation
        # branch in _actual fires; ACTUAL_MAX was raised this round (whole-reply
        # ceiling), so size the fixture off the live constant, not a fixed 5000.
        long_reply = "x" * (fleet_grading.ACTUAL_MAX + 500)
        return {"choices": [{"message": {"content": next((v for k, v in answers.items() if k in text), long_reply)}}]}
    events = []
    fleet_grading.run_capacity_benchmark(Mock(request=Mock(side_effect=request)), [_worker()], 16,
                                         threading.Event(), lambda k, v: events.append((k, v)), budgets=FAST)
    row = next(v for k, v in events if k == "result" and v.get("status") == "complete")
    easy, medium = row["detail"]["math"]["history"][:2]
    assert {k: v for k, v in easy.items() if k != "judge"} == {
                    "tier": "easy", "pass": True, "check_pass": True, "format": True,
                    "revised": False, "prompt": "Compute 15 + 22. Reply with only the final number.",
                    "expected": "last number == 37", "expected_answer": "37", "actual": "37"}
    assert easy["judge"]["model"] == DEFAULT_AGENT_BRAIN and easy["judge"]["correct"] is None
    assert medium["pass"] is False and medium["expected"] == "last number == 391"
    assert medium["actual"] == "I think 390" and medium["why"] == "expected last number == 391; got 'I think 390'"
    long_ = row["detail"]["factual"]["history"][0]["actual"]
    assert len(long_) <= fleet_grading.ACTUAL_MAX + 20 and long_.endswith("[truncated]")
    call = next(v for k, v in events if k == "call" and v.get("task") == "math (medium)")
    assert call["expected"] == "last number == 391" and call["why"].startswith("expected") and call["actual"] == "I think 390"
    from hugpy_curation.review.suites import text
    assert all(fleet_grading.describe(c) for tiers in text.TASKS.values() for _t, _p, c in tiers)
