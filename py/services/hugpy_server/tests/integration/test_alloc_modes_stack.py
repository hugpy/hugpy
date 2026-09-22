"""k37 Slice B — the engine/fleet/server half of the monolith's
``tests/test_alloc_modes.py`` (moved here by the ops agent, 2026-09-22; the
chaos half lives in ``py/operations/hugpy_ops/tests/test_alloc_modes.py``).

Needs hugpy_engine (overrides, spill), hugpy_fleet (flex, WorkerStore) and
hugpy_server (worker_routes) together — an integration test by the guide's
definition. Converted from the script style into pytest functions, intent kept.
"""
from __future__ import annotations

import importlib
import os
import tempfile

import pytest

from hugpy_engine.alloc_modes import (
    MODE_MIN_PKG_VERSION,
    derive_alloc_mode,
    gate_spill_for_worker,
    mode_to_spill,
    normalize_spill,
    worker_honors_mode_keys,
)

GIB = 2 ** 30


@pytest.fixture(autouse=True)
def _projects_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path))
    yield


def test_mode_to_spill_materialisation():
    assert mode_to_spill("gpu-only") == {"n_gpu_layers": -1}
    assert mode_to_spill("ram-only") == {"n_gpu_layers": "off"}
    assert mode_to_spill("max-gpu") == {}
    assert mode_to_spill("max-gpu", explicit_pick=True) == {"alloc_mode": "max-gpu"}
    assert mode_to_spill("gpu-only", explicit_pick=True) == {"n_gpu_layers": -1}
    assert mode_to_spill("ram-only", explicit_pick=True) == {"n_gpu_layers": "off"}
    assert mode_to_spill("max-ram", explicit_pick=True) == {"alloc_mode": "max-ram"}
    assert mode_to_spill("max-ram") == {"alloc_mode": "max-ram"}
    assert mode_to_spill("explicit", gpu_mem_gib=8, leniency_pct=30, priority=1) == {
        "alloc_mode": "explicit", "gpu_mem_gib": 8.0, "leniency_pct": 30.0, "priority": 1}
    assert mode_to_spill("warp-drive") == {}


def test_normalize_spill():
    assert normalize_spill({"alloc_mode": "autofit"})[0] == {"alloc_mode": "max-gpu"}
    assert normalize_spill({"alloc_mode": "max-gpu"})[0] == {"alloc_mode": "max-gpu"}
    assert normalize_spill({"alloc_mode": "gpu-only"})[0] == {"n_gpu_layers": -1}
    assert normalize_spill({"alloc_mode": "cpu-only"})[0] == {"n_gpu_layers": "off"}
    assert normalize_spill({"alloc_mode": "max-ram"})[0] == {"alloc_mode": "max-ram"}
    s, note = normalize_spill({"alloc_mode": "bogus", "ctx_pct": 25})
    assert s == {"ctx_pct": 25} and "bogus" in note


def test_derivation_matrix_default_is_max_gpu():
    assert derive_alloc_mode({"n_gpu_layers": -1}) == "gpu-only"
    assert derive_alloc_mode({"n_gpu_layers": 0}) == "ram-only"
    assert derive_alloc_mode({"n_gpu_layers": "off"}) == "ram-only"
    assert derive_alloc_mode({}) == "max-gpu" == derive_alloc_mode(None)
    assert derive_alloc_mode({"n_gpu_layers": 17}) == "max-gpu"
    assert derive_alloc_mode({"gpu_mem_gib": 6.0}) == "explicit"
    assert derive_alloc_mode({"gpu_mem_gib": 6.0, "gpu_mem_gib_deviation_pct": 25}) == "explicit"
    assert derive_alloc_mode({"alloc_mode": "max-ram", "n_gpu_layers": -1}) == "max-ram"
    assert derive_alloc_mode({"alloc_mode": "autofit"}) == "max-gpu"


def test_persisted_overrides_resolve_on_write_and_derive_on_read():
    from hugpy_engine.serve import overrides as OV
    OV.set_override("m-legacy", {"alloc_mode": "cpu-only"})
    assert OV.get_override("m-legacy").get("alloc_mode") == "ram-only"
    OV.set_override("m-typo", {"alloc_mode": "warp-drive"})
    assert "alloc_mode" not in OV.get_override("m-typo")
    OV.set_override("m-explicit", {"alloc_mode": "explicit", "leniency_pct": "30",
                                   "priority": "2", "priority_device": "ram"})
    assert OV.get_override("m-explicit") == {"alloc_mode": "explicit", "leniency_pct": 30.0,
                                             "priority": 2, "priority_device": "ram"}
    OV.set_override("m-old-wire", {"n_gpu_layers": -1})
    assert OV.effective_alloc_mode("m-old-wire") == "gpu-only"
    assert OV.effective_alloc_mode("m-never-touched") == "max-gpu"


def test_leniency_floor_math_against_flex_band_engine():
    from hugpy_fleet.worker.flex import band_floor, leniency_floor_pct, plan_explicit_offload
    assert leniency_floor_pct(100, 30) == 70.0
    assert leniency_floor_pct(50, 80) == 0.0
    model = 10 * GIB
    assert band_floor(model, 30.0, model) == 7 * GIB
    p = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                              vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB,
                              mode="explicit", priority_device="gpu", leniency_pct=30.0)
    assert p is not None and p.admit and 23 <= p.n_gpu_layers < 32
    assert p.ram_need_bytes > 0 and p.vram_need_bytes <= 8 * GIB
    p2 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=5 * GIB, ram_free_bytes=64 * GIB,
                               mode="explicit", priority_device="gpu", leniency_pct=30.0)
    assert p2 is not None and not p2.admit
    assert "explicit" in p2.reject_reason and "70%" in p2.reject_reason and "floor" in p2.reject_reason
    p3 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=12 * GIB, ram_free_bytes=64 * GIB, leniency_pct=30.0)
    assert p3.admit and p3.n_gpu_layers == 32
    p4 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB, leniency_pct=0.0)
    assert not p4.admit
    p5 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=8 * GIB, ram_free_bytes=int(6.5 * GIB),
                               mode="max-ram", priority_device="ram", leniency_pct=100.0)
    assert p5.admit and 0 < p5.n_gpu_layers < 32
    p6 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=2 * GIB, ram_free_bytes=2 * GIB,
                               mode="max-ram", priority_device="ram", leniency_pct=100.0)
    assert not p6.admit and "max-ram" in p6.reject_reason and "can't satisfy" in p6.reject_reason
    p7 = plan_explicit_offload(weights_bytes=model, kv_bytes=0, total_layers=32,
                               vram_budget_bytes=8 * GIB, ram_free_bytes=64 * GIB,
                               mode="max-ram", priority_device="ram", leniency_pct=100.0)
    assert p7.admit and p7.n_gpu_layers == 0


def test_worker_loader_maxram_is_autofit_inverted(monkeypatch):
    from hugpy_engine import spill as SP
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"x" * (1 << 20))
        gguf_path = f.name
    try:
        assert SP.maxram_gpu_layers(gguf_path, free_ram=4 * (1 << 20)) == 0
        n_over = SP.maxram_gpu_layers(gguf_path, free_ram=512 * 1024)
        assert 0 < n_over < 32
        monkeypatch.setenv("HUGPY_ALLOC_MODE", "max-ram")
        monkeypatch.setenv("HUGPY_CPU_MEM_GIB", str(512 * 1024 / GIB))
        assert SP.gguf_gpu_layers(gguf_path) == n_over
        monkeypatch.delenv("HUGPY_CPU_MEM_GIB")
        mm = SP.transformers_max_memory(model_need_bytes=20 * GIB)
        assert mm is not None and mm["cpu"] != "0.00GiB"
        mm2 = SP.transformers_max_memory()
        assert mm2 is not None and mm2[0] == "0.00GiB"
    finally:
        os.unlink(gguf_path)


def test_version_gate_no_dead_knobs():
    assert worker_honors_mode_keys(MODE_MIN_PKG_VERSION)
    assert worker_honors_mode_keys("0.2.0")
    assert not worker_honors_mode_keys("0.1.202")
    assert not worker_honors_mode_keys(None)
    assert not worker_honors_mode_keys("garbage")
    gated, note = gate_spill_for_worker({"alloc_mode": "max-ram"}, "0.1.202", "op")
    assert gated == {} and "max-ram" in note and "0.1.202" in note
    gated2, note2 = gate_spill_for_worker({"alloc_mode": "explicit", "leniency_pct": 30}, "0.1.203", "ae")
    assert gated2 == {"alloc_mode": "explicit", "leniency_pct": 30} and note2 is None
    gated3, note3 = gate_spill_for_worker({"n_gpu_layers": -1}, "0.1.150", "op")
    assert gated3 == {"n_gpu_layers": -1} and note3 is None


def test_worker_store_spill_for_gates_per_worker_version(tmp_path):
    from hugpy_fleet.central.workers import WorkerStore
    store = WorkerStore(path=str(tmp_path / "wk.json"))
    old = store.register(name="old-box", url="http://x:9100", pkg_version="0.1.202")
    new = store.register(name="new-box", url="http://y:9100", pkg_version="0.1.203")
    store.assign_model(old["id"], "m1", spill={"alloc_mode": "max-ram"})
    store.assign_model(new["id"], "m1", spill={"alloc_mode": "max-ram"})
    assert store.spill_for(old["id"], "m1") == {}
    assert store.spill_for(new["id"], "m1") == {"alloc_mode": "max-ram"}
    raw = (store._load().get(old["id"]) or {}).get("spill_by_model", {}).get("m1")
    assert raw == {"alloc_mode": "max-ram"}


def test_engine_gating_and_validation_at_the_assign_seam(monkeypatch):
    wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
    monkeypatch.setattr(wr, "_model_framework",
                        lambda mk: "transformers" if mk.startswith("tf") else "gguf")
    assert wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "tf-m") == (True, None)
    ok2, r2 = wr._alloc_spill_ok_for_engine({"alloc_mode": "explicit", "leniency_pct": 30}, "tf-m")
    assert ok2 is False and "explicit" in r2 and "GGUF-only" in r2 and "analogue" in r2
    assert wr._alloc_spill_ok_for_engine({}, "tf-m") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"n_gpu_layers": -1}, "tf-m") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"n_gpu_layers": "off"}, "tf-m") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"alloc_mode": "max-ram"}, "g-m") == (True, None)
    assert wr._alloc_spill_ok_for_engine({"alloc_mode": "explicit", "leniency_pct": 10}, "g-m") == (True, None)

    assert wr._validate_alloc_spill({"alloc_mode": "autofit"}) == ({"alloc_mode": "max-gpu"}, None)
    assert wr._validate_alloc_spill({"alloc_mode": "max-gpu"}) == ({"alloc_mode": "max-gpu"}, None)
    assert wr._validate_alloc_spill(None) == ({}, None)
    assert wr._validate_alloc_spill({"alloc_mode": "max-ram", "ctx_pct": 50}) == (
        {"alloc_mode": "max-ram", "ctx_pct": 50}, None)
    assert wr._validate_alloc_spill({"alloc_mode": "warp"})[1] is not None
    assert wr._validate_alloc_spill({"leniency_pct": 130})[1] is not None
    assert wr._validate_alloc_spill({"alloc_mode": "explicit", "leniency_pct": 30})[1] is None
    assert wr._validate_alloc_spill({"priority_device": "tpu"})[1] is not None

    assert wr._alloc_label({}) == "max-gpu"
    assert wr._alloc_label({"n_gpu_layers": -1}) == "gpu-only"
    assert wr._alloc_label({"n_gpu_layers": "off"}) == "ram-only"
    assert wr._alloc_label({"alloc_mode": "max-ram"}) == "max-ram"
