"""The two-phase grading split (operator ruling 2026-09-24): PHASE 1 collects
and persists every raw output with the deterministic check and makes NO brain-
judge call; PHASE 2 (``judge_collected``) loads the judge once, after the whole
lot, grades the stored outputs, is resumable/idempotent, and leaves an item
unjudged WITH the real reason when the judge cannot be served (no silent
substitution)."""
import threading

import pytest

from hugpy_curation.review import fleet_grading
from hugpy_platform.constants import DEFAULT_AGENT_BRAIN


def _worker(ram=10**11):
    return {"id": "w1", "name": "aeb", "status": "online", "admission": "approved",
            "serve_mode": "on", "max_vram_bytes": 10**11, "max_ram_bytes": ram,
            "loaded_models": []}


def _catalog(model="org/txt", task="text-generation"):
    return {"model_key": model, "primary_task": task, "tasks": [task], "framework": "gguf",
            "size_bytes": 10**9, "workers": [{"worker_id": "w1", "designated": True}]}


class FakeClient:
    def __init__(self, responder):
        self.calls, self.responder = [], responder

    def request(self, path, method="GET", body=None, timeout=None):
        self.calls.append((path, method, body))
        return self.responder(path, body)


def _collect(model="org/txt", answer="37", defer_judge=True, worker=None):
    events = []

    def route(path, body):
        if path.startswith("/models"):
            return {"models": [_catalog(model)]}
        if path.startswith("/llm/serving/"):
            return {}
        if path.startswith("/llm/workers/"):
            return {"ok": True}
        if path.startswith("/llm/model-metrics2"):
            return {"rows": []}
        return {"choices": [{"message": {"content": answer}}]}

    client = FakeClient(route)
    fleet_grading.run_capacity_benchmark(client, [worker or _worker()], 16, threading.Event(),
                                         lambda k, v: events.append((k, v)),
                                         budgets={"call_s": 5, "cold_load_cap_s": 5, "model_s": 60},
                                         defer_judge=defer_judge)
    results = [v for k, v in events if k == "result" and v.get("status") == "complete"]
    return client, events, results


def _judge_bodies(client):
    return [b for p, _m, b in client.calls if p == "/v1/chat/completions"
            and any(isinstance(m, dict) and m.get("role") == "system"
                    and m.get("content") == fleet_grading.JUDGE_SYSTEM
                    for m in (b or {}).get("messages", []))]


# ------------------------------------------------------ phase 1: collect ----
def test_collection_makes_no_inline_judge_call_and_leaves_items_pending():
    client, _events, results = _collect(defer_judge=True)
    assert results, "collection produced no completed rows"
    # NOT ONE brain-judge inference happened during collection.
    assert _judge_bodies(client) == []
    item = results[0]["detail"]["math"]["history"][0]
    assert item["judge"]["status"] == "pending"
    assert item["judge"]["model"] == DEFAULT_AGENT_BRAIN and item["judge"]["correct"] is None
    # The deterministic check is still recorded in phase 1.
    assert item["check_pass"] is True and item["pass"] is True
    # The row's judge summary says pending, and pending items are countable.
    assert results[0]["judge"]["status"] == "pending"
    assert len(fleet_grading._judge_pending(results)) > 0


def test_inline_default_still_judges_when_not_deferred():
    client, _events, results = _collect(defer_judge=False)
    assert results and _judge_bodies(client)          # inline judge DID run
    item = results[0]["detail"]["math"]["history"][0]
    assert item["judge"]["status"] in ("judged", "error")   # never left pending


# ------------------------------------------------------ phase 2: judging ----
def _verdict(correct=True, fmt=True, reason="ok"):
    body = '{"correct": %s, "format_ok": %s, "reason": "%s"}' % (
        str(correct).lower(), str(fmt).lower(), reason)
    return {"choices": [{"message": {"content": body}}]}


def test_judge_collected_grades_pending_updates_grade_and_is_idempotent():
    _c, _e, results = _collect(answer="37", defer_judge=True)
    before = {id(r): r["grade"] for r in results}
    jclient = FakeClient(lambda p, b: _verdict(correct=True))
    reports = []
    summary = fleet_grading.judge_collected(jclient, results, lambda k, v: reports.append((k, v)),
                                            sleep=lambda s: None)
    item = results[0]["detail"]["math"]["history"][0]
    assert item["judge"]["status"] == "judged" and item["judge"]["correct"] is True
    assert summary["pending"] == 0 and summary["total"] > 0
    # every item now passes -> the row's grade is the full max
    assert results[0]["grade"] == f"{results[0]['max']}/{results[0]['max']}"
    assert results[0]["judge"]["status"] == "judged"
    # a judge-result was reported for the row (drives the DB grade update)
    assert any(k == "judge-result" for k, _v in reports)
    # RESUMABLE / IDEMPOTENT: a second pass judges nothing (all done).
    jclient.calls.clear()
    again = fleet_grading.judge_collected(jclient, results, lambda k, v: None, sleep=lambda s: None)
    assert jclient.calls == [] and again["pending"] == 0


def test_judge_refusal_leaves_item_unjudged_with_reason_then_retries():
    # A single-lane worker (RAM too small for ram_only) so the graded ``detail``
    # is not shared across alloc variations — the first-round refusal genuinely
    # carries into a second round.
    _c, _e, results = _collect(answer="37", defer_judge=True, worker=_worker(ram=1))
    state = {"n": 0}

    def responder(path, body):
        state["n"] += 1
        if state["n"] == 1:                    # first judge call is a refusal
            raise fleet_grading.FleetError(
                'HTTP 503 at /v1/chat/completions: {"error": {"message": "gate=no_worker"}}')
        return _verdict(correct=True)

    jclient = FakeClient(responder)
    waited = []
    summary = fleet_grading.judge_collected(jclient, results, lambda k, v: None,
                                            rounds=2, sleep=lambda s: waited.append(s))
    # the refused item was retried on the next round and is now judged; the real
    # refusal reason was recorded on it in the meantime (never a silent fallback).
    assert summary["refused"] >= 1 and summary["pending"] == 0
    assert len(waited) == 1                     # exactly one backoff between the two rounds
    assert all((i.get("judge") or {}).get("status") == "judged"
               for r in results for cat, e in r["detail"].items()
               if isinstance(e, dict) and not cat.startswith("judge_")
               for i in e.get("history") or [])


def test_judge_refusal_persists_when_never_served():
    _c, _e, results = _collect(answer="37", defer_judge=True)

    def responder(path, body):
        raise fleet_grading.FleetError(
            'HTTP 503 at /v1/chat/completions: {"error": {"message": "gate=no_worker"}}')

    jclient = FakeClient(responder)
    summary = fleet_grading.judge_collected(jclient, results, lambda k, v: None,
                                            rounds=2, sleep=lambda s: None)
    assert summary["pending"] == summary["total"] and summary["pending"] > 0
    # unjudged rows carry the REAL reason, and no verdict was invented.
    reasons = {u["reason"] for u in summary["unjudged"]}
    assert reasons and all("gate=no_worker" in (r or "") for r in reasons)
    item = results[0]["detail"]["math"]["history"][0]
    assert item["judge"]["status"] == "pending" and item["judge"]["correct"] is None
    assert item["judge"]["attempts"] >= 1


def test_ladder_exhausted_when_model_under_test_is_the_brain():
    # The judge = the model under test -> no judge exists; recorded unavailable at
    # collection, and phase 2 has nothing pending to serve.
    _c, _e, results = _collect(model=DEFAULT_AGENT_BRAIN, defer_judge=True)
    assert results
    item = results[0]["detail"]["math"]["history"][0]
    assert item["judge"]["status"] == "unavailable"
    assert item["judge"]["model"] is None and fleet_grading.JUDGE_LADDER_EXHAUSTED in item["judge"]["error"]
    assert fleet_grading._judge_pending(results) == []
