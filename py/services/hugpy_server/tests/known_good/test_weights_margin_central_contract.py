"""KNOWN-GOOD CONTRACT — central prices a model's weights at the margin a
worker MEASURED (heartbeat-borne), the x1.15 prior only when none has.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_fleet/central/workers.py (WorkerStore.heartbeat
``weights_margins``, weights_margin_for, peer_weights_margins),
hugpy_server/app/routes/worker_routes.py (_worker_fit / _measured_weights_margin,
_context_preview, the heartbeat reply relay).

Deterministic: a private WorkerStore under tmp_path; the model size seams are
stubbed on the routes module; a flask test_request_context for the preview.
"""
from __future__ import annotations

import importlib

import pytest

flask = pytest.importorskip("flask")
W = importlib.import_module("hugpy_fleet.central.workers")
WR = importlib.import_module("hugpy_server.app.routes.worker_routes")

GIB = 1 << 30
MK = "Qwen3-Coder-Next-GGUF"
FILE = "Qwen3-Coder-Next-Q4_K_M.gguf"
WFILE = int(20.1 * GIB)


def _rec(margin, samples, device, file=FILE):
    return {"model_key": MK, "file": file, "file_bytes": WFILE, "backend": "gguf",
            "device_class": device, "margin": margin, "samples": samples,
            "weights_measured_bytes": int(WFILE * margin), "kv_measured_bytes": 0,
            "measured_at": 1000.0}


@pytest.fixture
def store(monkeypatch, tmp_path):
    s = W.WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    ae = s.register(name="ae", url="http://ae:9100")
    comp = s.register(name="computron", url="http://computron:9100")
    for w in (ae, comp):
        s.set_admission(w["id"], "approved")
    return s, ae["id"], comp["id"]


def test_heartbeat_stores_margins_and_central_prefers_the_same_device_class(store):
    """INVARIANT: ``weights_margins`` on the beat is stored verbatim on the
    worker record; weights_margin_for(model, device_class) returns the record
    from the same device class first, else the most samples; implausible
    ratios are ignored; a record only on an allocation row also counts; the
    reply relays PEER records only. Established: 2026-09-29."""
    s, ae, comp = store
    s.heartbeat(ae, gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_total": 24 * GIB,
                          "memory_free": 21 * GIB}],
                weights_margins={MK: _rec(1.04, 2, "NVIDIA GeForce RTX 3090")})
    s.heartbeat(comp, gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 4060", "memory_total": 8 * GIB,
                            "memory_free": 6 * GIB}],
                weights_margins={MK: _rec(1.30, 9, "NVIDIA GeForce RTX 4060"),
                                 "Other": {"margin": 9.0, "samples": 50, "file": None}},
                allocations=[{"kind": "slot", "model_key": "Row-Only", "vram_bytes": 1,
                              "weights_margin": 1.10, "weights_margin_samples": 4,
                              "weights_margin_device": "NVIDIA GeForce RTX 4060"}])
    assert s.get(ae)["weights_margins"][MK]["margin"] == 1.04
    best = W.weights_margin_for(MK, device_class="NVIDIA GeForce RTX 3090")
    assert best["margin"] == 1.04 and best["worker"] == "ae" and best["source"] == "measured"
    best = W.weights_margin_for(MK, device_class="NVIDIA GeForce RTX 5090")   # no same class
    assert best["margin"] == 1.30 and best["samples"] == 9
    assert W.weights_margin_for("Other") is None                            # implausible
    assert W.weights_margin_for("Row-Only")["margin"] == 1.10               # from the row
    assert W.weights_margin_for("Never") is None
    peers = W.peer_weights_margins(ae, [MK])
    assert [r["worker"] for r in peers[MK]] == ["computron"]                # never itself
    assert "Other" not in peers                                             # not relevant / implausible


def test_worker_fit_prices_measured_over_prior_and_never_stacks_calibration(monkeypatch):
    """INVARIANT: _worker_fit's need = raw x measured margin when any worker
    measured this file (headroom = that margin, source measured, n samples,
    no calibration correction applied); else raw x HUGPY_VRAM_HEADROOM x
    calibration, source prior. Established: 2026-09-29."""
    monkeypatch.setattr(WR, "_model_gguf_bytes", lambda mk: WFILE)
    monkeypatch.setattr(WR, "_model_moe_fit", lambda mk: None)
    worker = {"id": "w1", "name": "ae", "gpu": "RTX 3090", "vram_free": int(21.3 * GIB),
              "free_ram": 64 * GIB, "gpus": [{"name": "NVIDIA GeForce RTX 3090",
                                             "memory_total": 24 * GIB, "memory_free": int(21.3 * GIB)}]}
    monkeypatch.setattr(WR, "_measured_weights_margin",
                        lambda mk, w: {"margin": 1.04, "samples": 3, "worker": "ae-b",
                                       "device_class": "NVIDIA GeForce RTX 3090", "source": "measured"})
    v = WR._worker_fit(MK, worker)
    assert v["need"] == int(WFILE * 1.04) and v["headroom"] == 1.04
    assert v["weights_margin_source"] == "measured" and v["weights_margin_samples"] == 3
    assert v["weights_margin_worker"] == "ae-b"
    assert v["calibration_correction"] is None
    assert v["gpu_resident"] is True                          # 20.9 GiB <= 21.3 GiB free
    monkeypatch.setattr(WR, "_measured_weights_margin", lambda mk, w: None)
    v = WR._worker_fit(MK, worker)
    assert v["need"] == int(WFILE * WR.VRAM_HEADROOM)
    assert v["weights_margin_source"] == "prior" and v["weights_margin_samples"] == 0
    assert v["gpu_resident"] is False                         # 23.1 GiB > 21.3 GiB: the live refusal


def test_context_preview_carries_the_same_margin_source(monkeypatch):
    """INVARIANT: GET /llm/workers/<id>/context-preview prices weights off
    _worker_fit's figure and names its source (measured n samples | prior).
    Established: 2026-09-29."""
    worker = {"id": "w1", "name": "ae", "gpu": "RTX 3090"}
    monkeypatch.setattr(WR, "get_worker", lambda wid: worker if wid == "w1" else None)
    monkeypatch.setattr(WR, "_worker_fit", lambda mk, w: {
        "need": int(WFILE * 1.04), "need_raw": WFILE, "headroom": 1.04,
        "weights_margin": 1.04, "weights_margin_source": "measured",
        "weights_margin_samples": 3, "weights_margin_worker": "ae-b",
        "gpu_vram_free": int(21.3 * GIB), "calibration_correction": None, "reason": None})
    monkeypatch.setattr(WR, "_model_ctx_geometry", lambda mk: {
        "geometry": {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128, "ctx_train": 262144},
        "ctx_max": 262144, "dtype_bytes": 2.0, "source": "gguf-header"})
    app = flask.Flask(__name__)
    with app.test_request_context(f"/llm/workers/w1/context-preview?model_key={MK}&pct=5"):
        body = WR.workers_context_preview("w1").get_json()
    assert body["weights_bytes"] == int(WFILE * 1.04) and body["headroom"] == 1.04
    assert body["weights_margin_source"] == "measured" and body["weights_margin_samples"] == 3
    assert body["weights_margin_worker"] == "ae-b"
