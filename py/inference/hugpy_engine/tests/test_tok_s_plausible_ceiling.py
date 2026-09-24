"""tok_s_from_timings rejects an IMPLAUSIBLE decode rate.

The bug: a vision / very-short reply reports a sub-millisecond predicted_ms for a
handful of tokens, so n*1000/ms (or the engine's own predicted_per_second) came
out at ~240k tok/s — physically impossible on this fleet's cards. That value was
accepted, recorded, and EMA-poisoned the VL-3B estimate. A rate above
``_MAX_PLAUSIBLE_TOK_S`` is a measurement error; return None so it is never
recorded (absent beats wrong).
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EV = importlib.import_module("hugpy_engine.eviction")


def test_engine_rate_within_bound_is_kept():
    assert EV.tok_s_from_timings({"timings": {"predicted_per_second": 250.0}}) == 250.0


def test_engine_rate_above_ceiling_is_dropped():
    assert EV.tok_s_from_timings(
        {"timings": {"predicted_per_second": 240000.0}}) is None


def test_derived_rate_within_bound_is_kept():
    # 100 predicted tokens over a 1000 ms window that timed the (n-1) decode
    # steps (llama-server starts the generation clock after the first token):
    # the decode rate is (100 - 1) * 1000 / 1000 = 99.0 tok/s, well within the
    # ceiling, so it is kept.
    assert EV.tok_s_from_timings(
        {"timings": {"predicted_n": 100, "predicted_ms": 1000.0}}) == 99.0


def test_derived_rate_from_degenerate_window_is_dropped():
    # 5 tokens in 0.02 ms -> 250_000 tok/s: the exact VL-3B artifact
    assert EV.tok_s_from_timings(
        {"timings": {"predicted_n": 5, "predicted_ms": 0.02}}) is None


def test_zero_and_missing_still_none():
    assert EV.tok_s_from_timings({"timings": {"predicted_n": 0, "predicted_ms": 5}}) is None
    assert EV.tok_s_from_timings({"timings": {}}) is None
    assert EV.tok_s_from_timings({}) is None


def test_ceiling_is_generous_enough_for_a_fast_card():
    # a genuinely fast small model near the fleet's real ceiling is still kept
    assert EV.tok_s_from_timings({"timings": {"predicted_per_second": 800.0}}) == 800.0
