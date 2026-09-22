"""Per-(model, task) call metrics — avg output tokens AND avg compute time.

Central both routes and record-keeps, so `task` and the call's wall-clock
(`compute_s`) are in scope at the record site. These get folded into a
per-(model, task) EMA (`call_metrics_by_task`) that feeds the Metrics panel's
per-task columns and the time-aware allocator. The model-level `call_metrics`
row must stay byte-identical when a task is absent (backward-compat).

Also pins the derive_variant fix: `moe_capable` arrives as a {model_key: bool}
MAP at the record site; bool(map) was truthy for any non-empty map, stamping
every load 'moe'. The per-model flag must be read instead.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MM = importlib.import_module("hugpy_fleet.central.model_metrics")


@pytest.fixture()
def store(tmp_path):
    return MM.ModelMetricsStore(path=str(tmp_path / "model_metrics.db"))


def test_record_call_with_task_writes_a_per_task_row(store):
    assert store.record_call("m", 100.0, task="text-generation",
                             compute_s=0.5) is True
    rows = store.all_calls_by_task()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "m" and r["task"] == "text-generation"
    assert r["avg_tok_output_per_call"] == 100.0
    assert r["avg_compute_s"] == 0.5
    assert r["n_calls"] == 1


def test_task_and_compute_are_ema_blended_across_calls(store):
    store.record_call("m", 100.0, task="text-generation", compute_s=1.0)
    store.record_call("m", 200.0, task="text-generation", compute_s=2.0)
    r = store.all_calls_by_task()[0]
    assert r["n_calls"] == 2
    # first sample is the estimate; second blends at EMA_ALPHA
    a = MM.EMA_ALPHA
    assert r["avg_tok_output_per_call"] == pytest.approx((1 - a) * 100 + a * 200)
    assert r["avg_compute_s"] == pytest.approx((1 - a) * 1.0 + a * 2.0)


def test_distinct_tasks_are_separate_rows(store):
    store.record_call("m", 100.0, task="text-generation", compute_s=1.0)
    store.record_call("m", 4.0, task="image-text-to-text", compute_s=0.1)
    rows = {r["task"]: r for r in store.all_calls_by_task()}
    assert set(rows) == {"text-generation", "image-text-to-text"}
    assert rows["image-text-to-text"]["avg_compute_s"] == 0.1


def test_no_task_leaves_the_per_task_table_empty(store):
    # backward-compat: a task-less call still moves the model-level row only
    assert store.record_call("m", 100.0) is True
    assert store.all_calls_by_task() == []
    assert store.get_call("m")["n_calls"] == 1


def test_compute_s_optional_when_task_present(store):
    store.record_call("m", 100.0, task="text-generation")  # no compute_s
    r = store.all_calls_by_task()[0]
    assert r["avg_tok_output_per_call"] == 100.0
    assert r["avg_compute_s"] is None
    # a later call carrying compute_s starts the compute EMA
    store.record_call("m", 100.0, task="text-generation", compute_s=0.7)
    r = store.all_calls_by_task()[0]
    assert r["avg_compute_s"] == 0.7


def test_all_calls_by_task_never_raises_on_disabled_store():
    s = MM.ModelMetricsStore(path="/proc/definitely/not/writable/x.db")
    assert s.all_calls_by_task() == []


# --- the moe_capable map bug (fixed in managers/resolvers/remote.py) -------- #

def test_derive_variant_reads_per_model_flag_not_a_truthy_map():
    """derive_variant itself takes a bool; the bug was the CALLER passing a map.
    This pins the intended shape: a per-model False must derive a NON-moe variant
    for a full-GPU (n_gpu_layers=-1) dense load."""
    # dense model, all layers on GPU -> gpu_only_4bit (NOT moe)
    assert MM.derive_variant(-1, 48, moe_capable=False) == "gpu_only_4bit"
    # a MoE-capable model is moe regardless of layer split
    assert MM.derive_variant(-1, 48, moe_capable=True) == "moe"
    # the caller's fix: bool((map or {}).get(key)) — verify the idiom's outcome
    moe_map = {"other-model": True}   # non-empty map, but THIS model absent
    per_model = bool(moe_map.get("this-model"))
    assert per_model is False
    assert MM.derive_variant(-1, 48, moe_capable=per_model) == "gpu_only_4bit"
