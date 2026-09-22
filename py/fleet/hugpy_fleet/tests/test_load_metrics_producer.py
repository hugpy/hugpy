"""LOAD-METRICS PRODUCER (2026-08-28) — the cold-load half's single producer.

Calibration samples carrying a worker-measured ``load_seconds`` feed the
cold-load EMA via ``record_loads_from_calibration`` (the merge decision: the
calibration wire is THE producer — no second table, no second reporter).

    ./venv/bin/pytest tests/test_load_metrics_producer.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MM = importlib.import_module("hugpy_fleet.central.model_metrics")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    st = MM.ModelMetricsStore(path=str(tmp_path / "model_metrics.db"))
    monkeypatch.setattr(MM, "model_metrics_store", st)
    return st


def _sample(**over):
    s = {"model_key": "m", "n_gpu_layers": -1, "total_layers": 28,
         "load_seconds": 42.5, "ok": True}
    s.update(over)
    return s


def test_sample_with_load_seconds_feeds_cold_ema(store):
    assert MM.record_loads_from_calibration("ae", [_sample()]) == 1
    variant = MM.derive_variant(-1, 28)
    row = store.get_load("m", variant, "ae:0", "unloaded")
    assert row["upload_time_s"] == pytest.approx(42.5)
    assert row["n_samples"] == 1


def test_moe_capable_sample_lands_in_moe_variant(store):
    assert MM.record_loads_from_calibration(
        "ae", [_sample(moe_capable=True)]) == 1
    assert store.get_load("m", "moe", "ae:0", "unloaded")["upload_time_s"] \
        == pytest.approx(42.5)


def test_skips_never_guesses(store):
    samples = [
        _sample(load_seconds=None),        # unmeasured — the common case
        _sample(ok=False),                 # failed load — never loaded
        _sample(n_gpu_layers=None),        # unknowable variant — unrecorded
        _sample(load_seconds=0),           # degenerate measurement
        "not-a-dict",                      # hostile wire — skipped, not raised
    ]
    assert MM.record_loads_from_calibration("ae", samples) == 0


def test_empty_inputs_are_no_ops(store):
    assert MM.record_loads_from_calibration("", [_sample()]) == 0
    assert MM.record_loads_from_calibration("ae", []) == 0


def test_one_bad_sample_does_not_drop_the_rest(store):
    assert MM.record_loads_from_calibration(
        "computron", [{"ok": True}, _sample()]) == 1
    variant = MM.derive_variant(-1, 28)
    assert store.get_load("m", variant, "computron:0", "unloaded") is not None
