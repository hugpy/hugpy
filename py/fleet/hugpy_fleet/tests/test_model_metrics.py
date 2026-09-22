"""MODEL METRICS (2026-08-28) — EMA store behind group-member ranking.

Covers the record layer and the total_time_to_output estimate; the
decision-time terms (download presence, evict-to-fit dry run) are scheduler
inputs and exercised on dev.

    ./venv/bin/pytest tests/test_model_metrics.py -q
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


def test_first_sample_is_the_estimate(store):
    assert store.record_load("m", "moe", "ae:0", "unloaded",
                             upload_time_s=40.0, tok_per_s=12.0)
    row = store.get_load("m", "moe", "ae:0", "unloaded")
    assert row["upload_time_s"] == 40.0
    assert row["tok_per_s"] == 12.0
    assert row["n_samples"] == 1


def test_ema_blend(store):
    store.record_load("m", "moe", "ae:0", "loaded", upload_time_s=10.0)
    store.record_load("m", "moe", "ae:0", "loaded", upload_time_s=20.0)
    row = store.get_load("m", "moe", "ae:0", "loaded")
    expected = (1.0 - MM.EMA_ALPHA) * 10.0 + MM.EMA_ALPHA * 20.0
    assert row["upload_time_s"] == pytest.approx(expected)
    assert row["n_samples"] == 2


def test_absent_measurement_keeps_previous(store):
    store.record_load("m", "split", "computron:0", "loaded",
                      upload_time_s=8.0, tok_per_s=30.0)
    # A load that never generated has no tok/s — upload blends, tok/s stays.
    store.record_load("m", "split", "computron:0", "loaded", upload_time_s=12.0)
    row = store.get_load("m", "split", "computron:0", "loaded")
    assert row["tok_per_s"] == 30.0
    assert row["upload_time_s"] == pytest.approx(
        (1.0 - MM.EMA_ALPHA) * 8.0 + MM.EMA_ALPHA * 12.0)


def test_keys_are_typed_failures(store):
    assert not store.record_load("m", "fp16", "ae:0", "loaded", upload_time_s=1.0)
    assert not store.record_load("m", "moe", "ae:0", "lukewarm",
                                 upload_time_s=1.0)
    assert not store.record_load("", "moe", "ae:0", "loaded", upload_time_s=1.0)
    assert not store.record_load("m", "moe", "", "loaded", upload_time_s=1.0)


def test_call_metrics_ema(store):
    store.record_call("m", 400.0)
    store.record_call("m", 800.0)
    row = store.get_call("m")
    assert row["avg_tok_output_per_call"] == pytest.approx(
        (1.0 - MM.EMA_ALPHA) * 400.0 + MM.EMA_ALPHA * 800.0)
    assert row["n_calls"] == 2


def test_estimate_total_time(store):
    store.record_load("m", "moe", "ae:0", "loaded",
                      upload_time_s=6.0, tok_per_s=20.0)
    store.record_call("m", 500.0)
    # download 3 + evict 4 + upload 6 + 500/20 + displaced re-upload 5 = 43
    assert store.estimate_total_time(
        "m", "moe", "ae:0", temperature="loaded",
        time_to_download_s=3.0, time_to_evict_s=4.0,
        displaced_reupload_s=5.0) == pytest.approx(43.0)


def test_derive_variant_from_serving_contract():
    # MoE is its own regime and wins first.
    assert MM.derive_variant(-1, 48, moe_capable=True) == "moe"
    # llama.cpp vocabulary: -1 = all on GPU, 0 / chaos "off" = none.
    assert MM.derive_variant(-1) == "gpu_only_4bit"
    assert MM.derive_variant(48, 48) == "gpu_only_4bit"
    assert MM.derive_variant(0) == "ram_only_4bit"
    assert MM.derive_variant("off") == "ram_only_4bit"
    assert MM.derive_variant(26, 48) == "split"
    # Unknowable splits are unrecorded, not guessed.
    assert MM.derive_variant(None) is None
    assert MM.derive_variant(26, None) is None
    assert MM.derive_variant("garbage", 48) is None


def test_unmeasured_pair_is_unranked_not_free(store):
    assert store.estimate_total_time("m", "moe", "ae:0") is None
    store.record_load("m", "moe", "ae:0", "loaded", upload_time_s=6.0,
                      tok_per_s=20.0)
    # Still no call metrics — still unranked.
    assert store.estimate_total_time("m", "moe", "ae:0") is None
