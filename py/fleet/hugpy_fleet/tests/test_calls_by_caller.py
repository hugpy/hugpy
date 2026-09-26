"""Per-CALLER call aggregate (2026-09-24) — harness attribution in metrics.

Every completed call already lands one durable compute_actions row carrying the
call's own numbers plus detail.caller (the harness/client identity central
stamped). This aggregates that ONE source by caller: n and the means over ALL
recorded call rows, so the Metrics panel can show which harness made how many
calls. Rows with no caller collect under "(unattributed)".
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


def _call(store, model, tok, caller, dur):
    return store.record_call(
        model, float(tok), task="text-generation", compute_s=dur,
        call={"worker_card": "ae:0", "duration_s": dur, "tok_per_s": 1.0,
              "detail": {"caller": caller}})


def test_aggregates_n_and_means_per_caller(store):
    assert _call(store, "m", 100, "hermes", 1.0)
    assert _call(store, "m", 200, "hermes", 3.0)
    assert _call(store, "m", 50, "aider", 0.5)
    rows = {r["caller"]: r for r in store.all_calls_by_caller()}
    assert rows["hermes"]["n_calls"] == 2
    assert rows["hermes"]["avg_tok_output_per_call"] == 150.0  # (100+200)/2
    assert rows["hermes"]["avg_compute_s"] == 2.0              # (1.0+3.0)/2
    assert rows["aider"]["n_calls"] == 1


def test_rows_without_a_caller_are_unattributed(store):
    # A call with no caller detail (pre-attribution history) -> "(unattributed)".
    store.record_call("m", 10.0, task="text-generation", compute_s=0.1,
                      call={"worker_card": "ae:0", "duration_s": 0.1})
    rows = {r["caller"]: r for r in store.all_calls_by_caller()}
    assert rows["(unattributed)"]["n_calls"] == 1


def test_module_helper_prefers_store_method(store):
    assert _call(store, "m", 100, "codex", 1.0)
    rows = {r["caller"]: r for r in MM.calls_by_caller(store)}
    assert rows["codex"]["n_calls"] == 1


def test_empty_store_is_empty_list(store):
    assert store.all_calls_by_caller() == []
    assert MM.calls_by_caller(store) == []
