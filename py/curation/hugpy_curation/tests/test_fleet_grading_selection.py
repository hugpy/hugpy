"""Operator ruling 2026-09-24: the benchmark tests every SELECTED eligible
worker, designated or not — the designation gate is gone."""
import threading

from unittest.mock import Mock

from hugpy_curation.review import fleet_grading


def _worker(wid, name):
    return {"id": wid, "name": name, "status": "online", "admission": "approved",
            "unreachable": False, "serve_mode": "on",
            "max_vram_bytes": 10**11, "max_ram_bytes": 10**11, "loaded_models": []}


def _serving_client():
    client = Mock()
    client.request.side_effect = lambda path, *a, **k: {} if path.startswith("/llm/serving/") else {}
    return client


def test_lanes_built_for_every_eligible_worker_without_designation():
    # A model designated on NEITHER worker (no join rows at all).
    model = {"model_key": "org/m", "framework": "gguf", "size_bytes": 10**9, "workers": []}
    workers = [_worker("w1", "aeb"), _worker("w2", "computron")]
    lanes = fleet_grading._lanes(_serving_client(), workers, [model])
    assert {l["worker"] for l in lanes} == {"aeb", "computron"}
    assert all(l["quant"] == "runtime default" for l in lanes)
    # Undesignated worker still gets a lane with an empty join row.
    assert all(l["joined"].get("designated") in (None, False) for l in lanes)


def test_lanes_use_the_join_row_when_the_worker_has_one():
    model = {"model_key": "org/m", "framework": "gguf", "size_bytes": 10**9,
             "workers": [{"worker_id": "w1", "designated": True, "bnb_4bit": True}]}
    workers = [_worker("w1", "aeb"), _worker("w2", "computron")]
    lanes = fleet_grading._lanes(_serving_client(), workers, [model])
    aeb = next(l for l in lanes if l["worker"] == "aeb")
    computron = next(l for l in lanes if l["worker"] == "computron")
    assert aeb["joined"].get("bnb_4bit") is True           # its own row rode in
    assert computron["joined"].get("bnb_4bit") is None      # no row -> empty join


def test_ineligible_workers_get_no_lane():
    model = {"model_key": "org/m", "framework": "gguf", "size_bytes": 10**9, "workers": []}
    offline = _worker("w2", "computron"); offline["status"] = "offline"
    lanes = fleet_grading._lanes(_serving_client(), [_worker("w1", "aeb"), offline], [model])
    assert {l["worker"] for l in lanes} == {"aeb"}


# ── a full run grades a worker the model is NOT designated on ────────────────
FAST = {"call_s": 5.0, "cold_load_cap_s": 5.0, "model_s": 30.0}


def test_run_grades_an_undesignated_worker():
    # model has NO designation anywhere; two eligible workers selected.
    catalog = [{"model_key": "org/m", "primary_task": "text-generation",
                "tasks": ["text-generation"], "framework": "gguf", "size_bytes": 10**9,
                "workers": []}]
    workers = [_worker("w1", "aeb"), _worker("w2", "computron")]

    def request(path, method="GET", body=None, timeout=None):
        if path.startswith("/models"): return {"models": catalog}
        if path.startswith("/llm/serving/"): return {}
        if path.startswith(("/llm/evictions", "/llm/transfers")): return {}
        if path.startswith("/llm/workers/"): return {"ok": True}
        return {"choices": [{"message": {"content": "37"}}]}
    events = []
    fleet_grading.run_capacity_benchmark(Mock(request=Mock(side_effect=request)), workers, 16,
                                         threading.Event(), lambda k, v: events.append((k, v)),
                                         budgets=FAST)
    completed = [v for k, v in events if k == "result" and v.get("status") == "complete"]
    # Both undesignated workers were graded, and no row cites a designation reason.
    assert {r["worker"] for r in completed} == {"aeb", "computron"}
    no_lane = [v for k, v in events if k == "result" and v.get("failure_class") == "no_lane"]
    assert not no_lane
    for _k, v in events:
        text = str(v)
        assert "has no designation" not in text and "never designates" not in text
