"""chaos schema: observation completeness + verdict inference + verbatim
refusal capture (the predicted-vs-measured contract the learner joins on)."""
from __future__ import annotations

import json

import pytest
from chaos_fakes import GIB, FakeClient

from hugpy_ops.chaos import observe
from hugpy_ops.chaos.schema import (
    REQUIRED_ADMISSION_KEYS,
    REQUIRED_MEASURED_KEYS,
    REQUIRED_TOP_KEYS,
    SCHEMA_VERSION,
    blank_observation,
    validate_observation,
)


def test_blank_observation_is_schema_complete():
    obs = blank_observation()
    assert validate_observation(obs) == []
    assert obs["schema_version"] == SCHEMA_VERSION
    assert all(k in obs for k in REQUIRED_TOP_KEYS)
    assert all(k in obs["measured"] for k in REQUIRED_MEASURED_KEYS)
    assert all(k in obs["measured"]["admission"] for k in REQUIRED_ADMISSION_KEYS)


def test_validator_flags_dropped_keys_and_bad_skips():
    broken = blank_observation()
    del broken["measured"]["allocation"]
    assert any("allocation" in p for p in validate_observation(broken))
    broken2 = blank_observation()
    broken2["kind"] = "skip"
    assert any("skip_reason" in p for p in validate_observation(broken2))
    broken3 = blank_observation()
    broken3["skip_reason"] = "made-up-reason"
    assert any("unknown skip_reason" in p for p in validate_observation(broken3))


def _measured(term, workers, model_key="small-gguf", ev_before=None, jobs=None):
    return observe.build_measured(term, workers, model_key, ev_before or {}, jobs)


W_PROCEED = [{"name": "ae", "vram_evictions": 3, "vram_total": 24 * GIB,
              "gpus": [{"memory_free": 6 * GIB}],
              "allocations": [{"model_key": "small-gguf", "kind": "slot",
                               "vram_bytes": 2 * GIB, "rss_bytes": 3 * GIB,
                               "n_gpu_layers": -1, "total_layers": 29,
                               "ctx": 16384, "serving": True}]}]


def test_verdict_proceed_from_full_gpu_slot():
    m = _measured({"outcome": "done", "served_worker": "ae"}, W_PROCEED, ev_before={"ae": 3})
    assert m["admission"]["verdict"] == "proceed"
    assert m["allocation"]["vram_bytes"] == 2 * GIB
    assert m["admission"]["vram_evictions_delta"] == 0


@pytest.mark.parametrize("workers, ev_before, verdict", [
    ([{"name": "ae", "vram_evictions": 3,
       "allocations": [{"model_key": "small-gguf", "kind": "slot",
                        "vram_bytes": GIB, "rss_bytes": 5 * GIB,
                        "n_gpu_layers": 12, "total_layers": 29, "ctx": 8192}]}],
     {"ae": 3}, "partial"),
    ([{"name": "ae", "vram_evictions": 5,
       "allocations": [{"model_key": "small-gguf", "kind": "slot",
                        "vram_bytes": 2 * GIB, "n_gpu_layers": -1, "total_layers": 29}]}],
     {"ae": 3}, "evicted"),
])
def test_verdict_partial_and_evicted(workers, ev_before, verdict):
    m = _measured({"outcome": "done", "served_worker": "ae"}, workers, ev_before=ev_before)
    assert m["admission"]["verdict"] == verdict
    if verdict == "evicted":
        assert m["admission"]["vram_evictions_delta"] == 2


def test_verdict_cpu_from_ram_allocation():
    w = [{"name": "op", "vram_evictions": 0,
          "allocations": [{"model_key": "small-gguf", "kind": "ram", "rss_bytes": 4 * GIB}]}]
    m = _measured({"outcome": "done", "served_worker": "op"}, w, ev_before={"op": 0})
    assert m["admission"]["verdict"] == "cpu"


def test_refusal_captured_verbatim():
    refusal = {"state": "refused", "model_key": "huge-gguf",
               "needs_bytes": 400 * GIB, "needs_weights_bytes": 380 * GIB,
               "needs_kv_bytes": 20 * GIB, "ctx_pct": 50,
               "partial_offload_considered": {"admit": False, "reject_reason": "CPU remainder OOM"},
               "protected": [{"model_key": "sd-turbo", "why": "actively replying"}],
               "evicted": []}
    w = [{"name": "ae", "vram_evictions": 3, "last_load_error": refusal, "allocations": []}]
    m = _measured({"outcome": "refused", "served_worker": "ae",
                   "error": "won't fit on GPU: needs 400.0G"}, w, ev_before={"ae": 3})
    assert m["admission"]["verdict"] == "refuse"
    assert m["admission"]["refusal_reason"]["needs_weights_bytes"] == 380 * GIB
    assert m["admission"]["refusal_reason"]["needs_kv_bytes"] == 20 * GIB
    assert m["admission"]["partial_offload_considered"]["reject_reason"] == "CPU remainder OOM"
    assert "won't fit" in m["error"]


def test_refusal_json_embedded_in_error_string_is_parsed():
    err = "load failed: " + json.dumps({"state": "refused", "needs_bytes": 123})
    w = [{"name": "ae", "allocations": [], "last_load_error": None}]
    m = _measured({"outcome": "error", "served_worker": "ae", "error": err}, w, ev_before={})
    assert m["admission"]["refusal_reason"]["needs_bytes"] == 123


def test_served_worker_falls_back_to_job_row():
    m = _measured({"outcome": "done", "served_worker": None}, W_PROCEED, ev_before={"ae": 3},
                  jobs={"jobs": [{"id": "x", "worker": "ae", "status": "done"}]})
    assert m["served_worker"] == "ae"


def test_predicted_side_prices_from_meta():
    c = FakeClient()
    combo = {"model_key": "small-gguf", "framework": "gguf",
             "effective_bytes": 2 * GIB, "alloc_mode": "budget",
             "spill": {"gpu_mem_gib": 4.0, "ctx_pct": 50}, "ctx_pct": 50,
             "target_workers": ["computron", "ae"]}
    pred = observe.build_predicted(c, combo, c.workers())
    assert isinstance(pred["need_bytes"], int)
    assert pred["needs_weights_bytes"] == 2 * GIB and pred["needs_kv_bytes"] > 0
    assert "n_gpu_layers" in pred["per_worker"]["computron"]["advice"]
    assert pred["feasible"] is True


def test_filled_skip_observation_validates():
    skip = blank_observation()
    skip.update({"run_id": "r", "trial_id": "t", "seed": 1, "round": 0,
                 "ts_start": 1.0, "ts_end": 2.0, "duration_s": 1.0,
                 "kind": "skip", "skip_reason": "predicted-infeasible"})
    assert validate_observation(skip) == []
