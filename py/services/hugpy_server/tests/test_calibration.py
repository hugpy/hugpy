"""t28 load-and-learn — the calibration layer that closes the predict/measure loop.

Covers all four moving parts with the seams stubbed (no GPU, no live central):

  1. Central store + aggregation — median measured/predicted ratio, the spread
     gate, the sample floor, the clamp band, partial/refuse exclusion, and the
     HUGPY_CALIBRATION kill-switch.
  2. Worker capture — sample SHAPE (omit-when-None), verdict classification,
     dedup-per-residency + re-arm, the load-fail (refuse) sample, and the drain.
  3. Application — need-pricing consults the learned correction ELSE the static
     x1.15, defensive clamp, kill-switch.
  4. End-to-end — real store -> real aggregate -> adopt -> real _incoming_need_
     detail applies the learned number.
  5. Wire-landmine proof — HeartbeatRequest is additive-safe (extra ignored), and
     the heartbeat reply is consumed as a plain dict (unknown key tolerated).
"""
from __future__ import annotations

import os

import pytest

from hugpy_fleet.central.calibration import CalibrationStore
from hugpy_fleet.worker import agent as A

ROW = {"model_key": "m", "device": "cuda", "n_gpu_layers": 32,
       "total_layers": 32, "vram_bytes": 900, "rss_bytes": 500}


@pytest.fixture(autouse=True)
def _calib_env(monkeypatch):
    """Deterministic env for the store gates (no ambient tuning / kill-switch)."""
    for k in ("HUGPY_CALIBRATION", "HUGPY_CALIBRATION_MIN_SAMPLES",
              "HUGPY_CALIBRATION_MAX_SPREAD", "HUGPY_CALIBRATION_CLAMP_LO",
              "HUGPY_CALIBRATION_CLAMP_HI"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def store(tmp_path_factory):
    def _make():
        return CalibrationStore(str(tmp_path_factory.mktemp("calib") / "c.db"))
    return _make


def _sample(model="m", need=1000, vram=900, verdict="full", ok_=True,
            ngl=None, total_layers=None, device="cuda"):
    s = {"model_key": model, "engine": "gguf", "needs_weights_bytes": int(need * 0.9),
         "needs_kv_bytes": int(need * 0.1), "ctx_pct": 50, "need_total_bytes": need,
         "verdict": verdict, "vram_bytes": vram, "device": device, "ok": ok_,
         "ts": 1000.0}
    if ngl is not None:
        s["n_gpu_layers"] = ngl
    if total_layers is not None:
        s["total_layers"] = total_layers
    return s


@pytest.fixture
def worker(monkeypatch):
    """Worker-side capture with the prediction stubbed (deterministic without a
    real model/GPU) and the calibration buffers reset around the test."""
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: 1000)
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")

    def _reset():
        with A._CALIB_LOCK:
            A._CALIB_BUFFER.clear()
            A._CALIB_SAMPLED.clear()
            A._CALIB_CORRECTIONS.clear()

    _reset()
    yield A
    _reset()


# ── 1) central store + aggregation ──────────────────────────────────────────
def test_aggregate_median_spread_gate_correction(store):
    st = store()
    for _ in range(3):
        st.record(None, _sample(need=1000, vram=900))          # ratio 0.9 x3
    agg = st.aggregate("m")
    assert agg["usable_count"] == 3, "counts only full/ok/measured rows"
    assert agg["median_ratio"] == 0.9
    assert agg["spread"] == 0.0
    assert agg["gated"] is True, "gated at >= floor with tight spread"
    assert agg["correction"] == 0.9, "correction == median (in band)"


def test_sample_floor_ungates(store):
    """2 samples < default floor 3 -> not gated, no correction, median still reported."""
    st = store()
    for _ in range(2):
        st.record(None, _sample(need=1000, vram=950))
    agg = st.aggregate("m")
    assert agg["gated"] is False
    assert agg["correction"] is None, "static stands"
    assert agg["median_ratio"] == 0.95, "still reports median for the table"


def test_spread_gate_rejects_inconsistent_ratios(store):
    st = store()
    for v in (500, 1500, 500, 1500):                            # ratios .5/1.5
        st.record(None, _sample(need=1000, vram=v))
    agg = st.aggregate("m")
    assert agg["gated"] is False and agg["correction"] is None
    assert agg["spread"] and agg["spread"] > 0.35


def test_clamp_band(store):
    """median below 0.8 clamps up to 0.8; above 1.5 clamps down to 1.5."""
    st_lo = store()
    for _ in range(3):
        st_lo.record(None, _sample(need=1000, vram=500))        # ratio 0.5
    assert st_lo.aggregate("m")["correction"] == 0.8
    st_hi = store()
    for _ in range(3):
        st_hi.record(None, _sample(need=1000, vram=2000))       # ratio 2.0
    assert st_hi.aggregate("m")["correction"] == 1.5


def test_partial_cpu_refuse_rows_excluded_from_ratio(store):
    st = store()
    for _ in range(3):
        st.record(None, _sample(need=1000, vram=900))         # full, usable
    st.record(None, _sample(need=1000, vram=300, verdict="partial"))
    st.record(None, _sample(need=1000, vram=0, verdict="cpu", device="cpu"))
    st.record(None, _sample(need=1000, vram=None, verdict="refuse", ok_=False))
    agg = st.aggregate("m")
    assert agg["usable_count"] == 3
    assert agg["sample_count"] == 6, "total sample_count counts ALL rows"
    assert agg["correction"] == 0.9, "unaffected by non-full rows"


def _hot_cold_store(store):
    st = store()
    for _ in range(3):
        st.record(None, _sample(model="hot", need=1000, vram=1200))   # ratio 1.2
    st.record(None, _sample(model="cold", need=1000, vram=900))       # 1 sample -> ungated
    return st


def test_corrections_publishes_only_gated_models(store):
    st = _hot_cold_store(store)
    corr = st.corrections()
    assert corr.get("hot", {}).get("correction") == 1.2
    assert "cold" not in corr
    assert st.correction_for("hot") == 1.2
    assert st.correction_for("cold") is None
    assert {r["model_key"] for r in st.table()} == {"hot", "cold"}


def test_kill_switch_makes_corrections_inert(store, monkeypatch):
    st = _hot_cold_store(store)
    monkeypatch.setenv("HUGPY_CALIBRATION", "off")
    assert st.corrections() == {}
    assert st.correction_for("hot") is None
    assert any(r["model_key"] == "hot" for r in st.table()), \
        "table still shows what WOULD be learned"


def test_env_tunable_floor_ungates(store, monkeypatch):
    st = store()
    for _ in range(3):
        st.record(None, _sample(need=1000, vram=900))
    assert st.aggregate("m")["gated"] is True
    monkeypatch.setenv("HUGPY_CALIBRATION_MIN_SAMPLES", "5")
    assert st.aggregate("m")["gated"] is False


# ── 2) worker capture ───────────────────────────────────────────────────────
def test_success_sample_shape(worker):
    s = worker._build_calibration_success("m", ROW)
    assert s["need_total_bytes"] == 1000, "base (uncorrected) prediction"
    assert s["verdict"] == "full"
    assert s["vram_bytes"] == 900
    assert s["ok"] is True
    assert all(v is not None for v in s.values()), "omit-when-None: no null keys on the wire"


@pytest.mark.parametrize("device,ngl,total,verdict", [
    ("cuda", 10, 32, "partial"),
    ("cuda", 0, 32, "cpu"),
    ("cpu", None, None, "cpu"),
    ("cuda", None, None, "full"),    # in-process cuda (ngl None)
    ("cuda", -1, 32, "full"),        # max-gpu
])
def test_verdict_classification(device, ngl, total, verdict):
    assert A._calib_verdict(device, ngl, total) == verdict


def test_dedup_per_residency_and_rearm(worker):
    worker._collect_calibration_from_allocations([ROW])
    assert len(worker._CALIB_BUFFER) == 1, "first residency -> one sample"
    worker._collect_calibration_from_allocations([ROW])
    assert len(worker._CALIB_BUFFER) == 1, "same residency -> no duplicate"
    worker._collect_calibration_from_allocations([{"model_key": "m", "vram_bytes": None}])
    assert len(worker._CALIB_BUFFER) == 1, "unmeasured row -> not sampled"
    worker._collect_calibration_from_allocations([])       # model left residency
    worker._collect_calibration_from_allocations([ROW])    # reload -> re-sampled
    assert len(worker._CALIB_BUFFER) == 2, "re-arm on departure"


def test_refuse_sample_and_drain(worker):
    worker._record_calibration_refuse(
        "m", {"weights": 900, "kv": 100, "base_total": 1000, "ctx_pct": 50})
    assert len(worker._CALIB_BUFFER) == 1
    assert worker._CALIB_BUFFER[0]["ok"] is False
    assert worker._CALIB_BUFFER[0]["verdict"] == "refuse"
    drained = worker._drain_calibration_samples()
    assert len(drained) == 1
    assert len(worker._CALIB_BUFFER) == 0, "drain clears the buffer"


def test_kill_switch_stops_capture(worker, monkeypatch):
    monkeypatch.setenv("HUGPY_CALIBRATION", "off")
    worker._collect_calibration_from_allocations([ROW])
    worker._record_calibration_refuse("m", {"base_total": 1000})
    assert len(worker._CALIB_BUFFER) == 0


# ── 3) application: consults-learned-ELSE-static ────────────────────────────
def test_learned_correction_applied_else_static(worker, monkeypatch):
    assert worker._incoming_need_detail("m")["total"] == 1000, "no correction -> static"
    worker._adopt_calibration({"calibration": {"m": {"correction": 1.2}}})
    det = worker._incoming_need_detail("m")
    assert det["total"] == 1200
    assert det["base_total"] == 1000, "base_total stays the UNcorrected prediction"
    assert det["calibration_correction"] == 1.2, "provenance"
    worker._adopt_calibration({"calibration": {"m": {"correction": 5.0}}})
    assert worker._incoming_need_detail("m")["total"] == 1500, "defensive re-clamp (5.0 -> 1.5)"
    monkeypatch.setenv("HUGPY_CALIBRATION", "off")
    assert worker._incoming_need_detail("m")["total"] == 1000, "kill-switch"


def test_absent_reply_clears_corrections(worker):
    """adopting an empty/absent reply clears prior corrections (older/off central)."""
    worker._adopt_calibration({"calibration": {"m": {"correction": 1.2}}})
    worker._adopt_calibration({})
    assert worker._incoming_need_detail("m")["total"] == 1000


# ── 4) end-to-end: store -> aggregate -> adopt -> apply ─────────────────────
def test_end_to_end_store_to_worker(worker, store):
    e2e = store()
    for _ in range(4):
        e2e.record("worker-1", _sample(model="m", need=1000, vram=1150))   # ratio 1.15
    pub = e2e.corrections(["m"])
    assert pub.get("m", {}).get("correction") == 1.15
    worker._adopt_calibration({"calibration": pub})
    assert worker._incoming_need_detail("m")["total"] == 1150


# ── 5) wire-landmine proof ──────────────────────────────────────────────────
def test_heartbeat_request_is_additive_safe():
    from hugpy_server.app.routes.worker_routes import HeartbeatRequest
    hb = HeartbeatRequest(**{"calibration_samples": [_sample()],
                             "some_future_field_old_central_never_saw": 123})
    assert hb.calibration_samples and len(hb.calibration_samples) == 1
    assert not hasattr(hb, "some_future_field_old_central_never_saw"), \
        "extra='ignore', not forbid"


def test_heartbeat_reply_without_calibration_key_is_tolerated(worker):
    """a heartbeat reply is a plain dict the worker reads with .get() — a reply
    WITHOUT the calibration key clears cleanly (no exception)."""
    worker._adopt_calibration({"calibration": {"m": {"correction": 1.2}}})
    worker._adopt_calibration({"limits": {}, "required_pkg_version": "0.1.191"})
    assert worker._incoming_need_detail("m")["total"] == 1000
