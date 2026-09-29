"""KNOWN-GOOD CONTRACT — the weights term is priced at a MEASURED margin;
the x1.15 prior only for a file nobody has loaded ("measured beats assumed").

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (measured weights margin).
Source under test: hugpy_fleet/worker/agent.py (_weights_margin_for,
_incoming_need_detail, _margin_note_admission, _margin_measure,
_collect_weights_margins, _margin_row_fields, _load_weights_margins /
_persist_weights_margins, _need_split_str, _vram_evict_to_fit refusal),
hugpy_engine/fit/plan.py (need_split), hugpy_engine/fit/types.py (FitFailure).

LIVE CASE (ae, 2026-09-29): "needs 23.1 GB = 23.1 GB weights (20.1 GB on disk
x 1.15 weights headroom)" refused a 21.3 GB-free card for a file whose real
load takes ~1.04x.

Deterministic: the agent's device / resident / eviction seams are stubbed on
the module exactly as tests/test_vram_evict_to_fit.py does; no GPU, no worker.
"""
from __future__ import annotations

import importlib
import json
import types

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
gen_gate = importlib.import_module("hugpy_fleet.worker.gen_gate")
fit_plan_mod = importlib.import_module("hugpy_engine.fit.plan")
fit_types = importlib.import_module("hugpy_engine.fit.types")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30
MIB = 1 << 20
MK = "Qwen3-Coder-Next-GGUF"
FILE = "Qwen3-Coder-Next-Q4_K_M.gguf"
WFILE = int(20.1 * GIB)


class _State:
    pass


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Empty margin state per test; the size/geometry seams stubbed so no
    registry, disk or CUDA is consulted."""
    for d in (A._WEIGHTS_MARGINS, A._REMOTE_MARGINS, A._MARGIN_BASELINES):
        d.clear()
    A._MARGIN_SAMPLED.clear()
    A._ACTIVITY_EPOCH[0] = 0
    A._MARGIN_ARGS["args"] = None
    A._MOE_SPLIT.clear()
    monkeypatch.setattr(A, "_incoming_weights_file", lambda mk: (WFILE, FILE))
    monkeypatch.setattr(A, "_kv_need_bytes",
                        lambda mk, cfg=None: (0, {"ctx_pct": None, "ctx_resolved": None,
                                                  "ctx_max": None, "ctx_effective": None,
                                                  "ctx_source": None, "geometry_source": None,
                                                  "kv_bytes": 0}))
    monkeypatch.setattr(A, "_calib_correction", lambda mk: None)
    monkeypatch.setattr(A, "_moe_plan_for", lambda mk: None)
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(A, "_kv_bytes_at_ctx", lambda mk, ctx, cfg=None: 0)
    monkeypatch.setattr(A, "detect_gpus", lambda: [{"index": 0, "name": "NVIDIA GeForce RTX 3090"}])
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    yield
    for d in (A._WEIGHTS_MARGINS, A._REMOTE_MARGINS, A._MARGIN_BASELINES):
        d.clear()
    A._MARGIN_SAMPLED.clear()
    A._MARGIN_ARGS["args"] = None


def _measured(margin=1.04, samples=3, file=FILE, device="NVIDIA GeForce RTX 3090"):
    return {"model_key": MK, "file": file, "file_bytes": WFILE, "backend": "gguf",
            "device_class": device, "margin": margin, "last_ratio": margin,
            "weights_measured_bytes": int(WFILE * margin), "kv_measured_bytes": 0,
            "measured_ctx": 32768, "delta_bytes": int(WFILE * margin),
            "measured_at": 1000.0, "samples": samples}


# ── pricing rule ────────────────────────────────────────────────────────────

def test_pricing_chooses_the_measured_margin_over_the_prior():
    """INVARIANT: with a measurement for (key, served file) on this box the
    weights term is file x measured margin (source `measured`, n samples);
    the total is measured weights + KV with NO calibration correction stacked;
    the prior figure is kept beside it for the calibration sample's base.
    Established: 2026-09-29."""
    A._WEIGHTS_MARGINS[MK] = _measured(1.04, 3)
    det = A._incoming_need_detail(MK)
    assert det["weights"] == int(WFILE * 1.04)
    assert det["total"] == int(WFILE * 1.04)
    assert det["weights_file_bytes"] == WFILE and det["weights_file"] == FILE
    assert det["weights_headroom"] == det["weights_margin"] == 1.04
    assert det["weights_margin_source"] == "measured"
    assert det["weights_margin_samples"] == 3
    assert det["weights_margin_origin"] == "local"
    assert det["weights_prior_bytes"] == int(WFILE * A._WEIGHTS_HEADROOM)
    assert det["base_total"] == int(WFILE * A._WEIGHTS_HEADROOM)   # the prior basis
    assert det["calibration_correction"] == 1.0
    assert A._need_split_str(det) == (
        f" = {A._human_bytes(det['weights'])} weights ({A._human_bytes(WFILE)} on disk "
        f"x 1.04 measured (3 samples on NVIDIA GeForce RTX 3090))")


def test_pricing_falls_to_the_prior_when_nothing_measured():
    """INVARIANT: no measurement anywhere -> file x 1.15, source `prior`,
    0 samples, and the line says "prior (never loaded)". Established: 2026-09-29."""
    det = A._incoming_need_detail(MK)
    assert det["weights"] == int(WFILE * A._WEIGHTS_HEADROOM)
    assert det["weights_margin_source"] == "prior" and det["weights_margin_samples"] == 0
    assert det["weights_margin"] == A._WEIGHTS_HEADROOM
    assert A._need_split_str(det).endswith("on disk x 1.15 prior (never loaded))")
    # the pure need_split carries the provenance for FitFailure
    split = fit_plan_mod.need_split(det, det["total"])
    assert split["weights_margin_source"] == "prior" and split["weights_margin"] == 1.15


def test_a_measurement_for_a_different_served_file_does_not_price_this_one():
    """INVARIANT: the margin is per (key, quant/file): a q8_0 measurement never
    prices the q4_k_m load. Established: 2026-09-29."""
    A._WEIGHTS_MARGINS[MK] = _measured(1.04, 3, file="Qwen3-Coder-Next-Q8_0.gguf")
    det = A._incoming_need_detail(MK)
    assert det["weights_margin_source"] == "prior"


def test_a_peer_measurement_relayed_by_central_prices_the_file_same_device_first():
    """INVARIANT: with no local record, a peer's record for the same file
    (adopted from the heartbeat reply) is used — the same device class
    preferred over more samples elsewhere; the origin says `central`.
    Established: 2026-09-29."""
    A._adopt_weights_margins({"weights_margins": {MK: [
        dict(_measured(1.30, 9, device="NVIDIA GeForce RTX 4060"), worker="computron"),
        dict(_measured(1.05, 2, device="NVIDIA GeForce RTX 3090"), worker="ae-b"),
        {"margin": 5.0, "file": FILE, "samples": 99},      # implausible: never adopted
    ]}})
    mg = A._weights_margin_for(MK, FILE)
    assert mg["source"] == "measured" and mg["origin"] == "central"
    assert mg["margin"] == 1.05 and mg["samples"] == 2 and mg["worker"] == "ae-b"
    # a local record outranks every peer
    A._WEIGHTS_MARGINS[MK] = _measured(1.02, 1)
    assert A._weights_margin_for(MK, FILE)["origin"] == "local"


# ── measurement ─────────────────────────────────────────────────────────────

def _row(**kw):
    row = {"kind": "slot", "model_key": MK, "vram_bytes": int(21 * GIB), "device": "cuda",
           "materialized": True, "n_gpu_layers": -1, "total_layers": 48, "ctx": 32768,
           "gpu_index": 0}
    row.update(kw)
    return row


def test_measurement_is_the_free_delta_net_of_kv_and_is_recorded_with_raw_bytes(monkeypatch):
    """INVARIANT: baseline (free right after admission) - free once the
    resident is measured healthy - KV at the served ctx = measured weights;
    ratio = that / file bytes; the record carries the raw bytes, the ctx, the
    device class, the sample count, and the next pricing uses it.
    Established: 2026-09-29."""
    monkeypatch.setattr(A, "_kv_bytes_at_ctx", lambda mk, ctx, cfg=None: 2 * GIB if ctx == 32768 else 0)
    A._margin_note_admission(MK, free_before=int(23 * GIB), device_index=0)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23 * GIB) - int(WFILE * 1.04) - 2 * GIB)
    rec = A._margin_measure(MK, _row(), loading=[MK])
    assert rec is not None
    assert rec["kv_measured_bytes"] == 2 * GIB and rec["measured_ctx"] == 32768
    assert rec["weights_measured_bytes"] == int(WFILE * 1.04)
    assert abs(rec["margin"] - 1.04) < 1e-3 and rec["samples"] == 1
    assert rec["device_class"] == "NVIDIA GeForce RTX 3090" and rec["file"] == FILE
    assert A._incoming_need_detail(MK)["weights_margin_source"] == "measured"
    # allocation rows carry it (omit-when-unset for anything unmeasured)
    fields = A._margin_row_fields(MK)
    assert fields["weights_margin_source"] == "measured" and fields["weights_margin_samples"] == 1
    assert fields["weights_measured_bytes"] == int(WFILE * 1.04)
    assert A._margin_row_fields("never-measured") == {}
    # a second sample averages in; a baseline is consumed once
    A._margin_note_admission(MK, free_before=int(23 * GIB), device_index=0)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23 * GIB) - int(WFILE * 1.06) - 2 * GIB)
    rec2 = A._margin_measure(MK, _row(), loading=[])
    assert rec2["samples"] == 2 and abs(rec2["margin"] - 1.05) < 1e-3
    assert A._margin_measure(MK, _row(), loading=[]) is None          # no baseline left


def test_implausible_measurements_are_discarded_not_averaged(monkeypatch):
    """INVARIANT: a ratio outside [0.9, 2.0] is discarded (the record stays
    as it was, samples unchanged). Established: 2026-09-29."""
    A._WEIGHTS_MARGINS[MK] = _measured(1.04, 3)
    for after in (int(23 * GIB) - int(WFILE * 0.5),        # 0.5x: half the file "landed"
                  int(23 * GIB) - int(WFILE * 2.5)):       # 2.5x: something else moved
        A._margin_note_admission(MK, free_before=int(23 * GIB), device_index=0)
        monkeypatch.setattr(A, "_free_vram_bytes", lambda a=after: a)
        assert A._margin_measure(MK, _row(), loading=[]) is None
    assert A._WEIGHTS_MARGINS[MK]["samples"] == 3 and A._WEIGHTS_MARGINS[MK]["margin"] == 1.04


def test_a_measurement_with_another_load_or_evict_in_flight_is_discarded(monkeypatch):
    """INVARIANT: a baseline that saw an eviction or another admission (the
    activity epoch moved), or a beat where another model is still loading,
    yields no sample. Established: 2026-09-29."""
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23 * GIB) - int(WFILE * 1.04))
    # another model loading at measurement time
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    assert A._margin_measure(MK, _row(), loading=[MK, "other"]) is None
    # an eviction between baseline and measurement
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    A._bump_activity_epoch("evict")
    assert A._margin_measure(MK, _row(), loading=[]) is None
    # another admission on the box between baseline and measurement
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    A._margin_note_admission("other", free_before=int(2 * GIB))
    assert A._margin_measure(MK, _row(), loading=[]) is None
    # a load that was already in flight when THIS baseline was armed
    monkeypatch.setattr(A, "_loading_model_keys", lambda: ["other"])
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    assert A._margin_measure(MK, _row(), loading=[]) is None
    assert MK not in A._WEIGHTS_MARGINS
    # a partial / MoE-split residency is never a weights measurement
    monkeypatch.setattr(A, "_loading_model_keys", lambda: [])
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    assert A._margin_measure(MK, _row(n_gpu_layers=20), loading=[]) is None


def test_the_heartbeat_collector_measures_once_per_residency_episode(monkeypatch):
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: int(23 * GIB) - int(WFILE * 1.04))
    A._margin_note_admission(MK, free_before=int(23 * GIB))
    A._collect_weights_margins([_row()], loading=[])
    assert A._WEIGHTS_MARGINS[MK]["samples"] == 1
    A._collect_weights_margins([_row()], loading=[])           # same episode: no re-sample
    assert A._WEIGHTS_MARGINS[MK]["samples"] == 1
    A._collect_weights_margins([], loading=[])                 # left residency: re-armed
    assert MK not in A._MARGIN_SAMPLED


# ── persistence ─────────────────────────────────────────────────────────────

def test_persistence_round_trip_through_the_settings_file(tmp_path):
    """INVARIANT: the records live in the worker's settings file
    (``weights_margins``) beside the operator's runtime settings, survive a
    restart, and an implausible/incomplete row on disk is dropped on load.
    Established: 2026-09-29."""
    args = types.SimpleNamespace(id_file=str(tmp_path / "worker.id"))
    with open(A._settings_path(args), "w", encoding="utf-8") as fh:
        json.dump({"slot_count": 2}, fh)
    A._MARGIN_ARGS["args"] = args
    A._WEIGHTS_MARGINS[MK] = _measured(1.04, 3)
    assert A._persist_weights_margins() is True
    on_disk = A._load_settings(args)
    assert on_disk["slot_count"] == 2                          # untouched
    assert on_disk["weights_margins"][MK]["margin"] == 1.04
    # a "restart"
    A._WEIGHTS_MARGINS.clear()
    on_disk["weights_margins"]["junk"] = {"margin": 7.0, "file_bytes": 1}
    on_disk["weights_margins"]["hollow"] = {"margin": 1.1}
    A._save_settings(args, on_disk)
    assert A._load_weights_margins(args) == 1
    assert A._WEIGHTS_MARGINS == {MK: _measured(1.04, 3)}
    assert A._weights_margin_for(MK, FILE)["source"] == "measured"


# ── the refusal names the source ────────────────────────────────────────────

@pytest.fixture
def rig(monkeypatch):
    for leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                 "HUGPY_LENIENCY_PCT", "HUGPY_PRIORITY_DEVICE", "HUGPY_BNB_4BIT",
                 "HUGPY_N_GPU_LAYERS", "HUGPY_VRAM_CEILING_FRAC",
                 "HUGPY_VRAM_RESERVE_GIB", "HUGPY_VRAM_CEILING_CUSHION_GIB",
                 "HUGPY_EVICT_LEAST_REAPING", "HUGPY_NO_EVICT"):
        monkeypatch.delenv(leak, raising=False)
    card = {"total": int(23.6 * GIB), "free": int(21.3 * GIB)}
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_vram_residents", lambda s: [])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: None)
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subject: set())
    monkeypatch.setattr(A, "_target_device_index", lambda: None)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    monkeypatch.setattr(A, "_subject_resident_vram_bytes", lambda s, mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(A, "_vram_occupancy_attribution", lambda: {})
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: {})
    monkeypatch.setattr(A, "_evict_model", lambda state, mk, force=False: {"evicted": False})
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    return card


def test_the_live_ae_case_admits_at_the_measured_margin_and_refuses_naming_the_prior(rig):
    """INVARIANT (the live case): a 20.1 GiB file on a card with 21.3 GiB free.
    Priced at the x1.15 PRIOR it is refused, and the refusal / fit_failure say
    "x 1.15 prior (never loaded)" with weights_margin_source=prior; priced at a
    MEASURED 1.02 it is admitted (proceed) and the admission arms the margin
    baseline. Established: 2026-09-29."""
    v = A._vram_evict_to_fit(_State(), MK)
    assert v["action"] == "refuse"
    r = v["reason"]
    assert "on disk x 1.15 prior (never loaded)" in r["reason"]
    assert r["weights_margin_source"] == "prior" and r["weights_margin_samples"] == 0
    assert r["needs_weights_headroom"] == 1.15
    ff = r["fit_failure"]
    assert ff["weights_margin_source"] == "prior" and ff["weights_margin"] == 1.15
    assert ff["weights_margin_samples"] == 0
    assert isinstance(fit_types.FitFailure(kind="vram_fit", code="wont_fit", reason="x",
                                           weights_margin=1.04, weights_margin_source="measured",
                                           weights_margin_samples=3).as_dict()["weights_margin"], float)
    assert MK not in A._MARGIN_BASELINES                     # a refusal arms nothing

    # 20.1 GiB x 1.02 = 20.5 GiB against 21.3 GiB free - the 512 MiB cushion:
    # admitted; the SAME card refused it at the prior.
    A._WEIGHTS_MARGINS[MK] = _measured(1.02, 3)
    v = A._vram_evict_to_fit(_State(), MK)
    assert v["action"] == "proceed"
    assert A._MARGIN_BASELINES[MK]["free_before"] == rig["free"]
