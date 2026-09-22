"""chaos alloc: snapshot / apply / restore round-trip + verification, the
engine-gate (non-200) abort, and partial-apply restore."""
from __future__ import annotations

from chaos_fakes import FakeClient

from hugpy_ops.chaos import alloc
from hugpy_ops.chaos.assortment import worker_index


def _snap(client, model, workers):
    return alloc.snapshot(client, model, workers, worker_index(client.workers()))


def test_snapshot_captures_current_spill_including_autofit():
    c = FakeClient()
    snap = _snap(c, "small-gguf", ["computron", "ae"])
    assert snap["computron"]["before"] == {"n_gpu_layers": -1}
    assert snap["ae"]["before"] is None          # absent spill == autofit
    assert snap["computron"]["worker_id"] == "wid-comp"


def test_apply_then_restore_round_trips_and_verifies():
    c = FakeClient()
    snap = _snap(c, "small-gguf", ["computron", "ae"])
    chaos_spill = {"gpu_mem_gib": 4.0, "ctx_pct": 50}
    res = alloc.apply(c, "small-gguf", chaos_spill, snap)
    assert res["ok"] is True
    w = {x["name"]: x for x in c.workers()}
    assert w["computron"]["spill_by_model"]["small-gguf"] == chaos_spill
    assert w["ae"]["spill_by_model"]["small-gguf"] == chaos_spill

    rest = alloc.restore(c, "small-gguf", snap)
    assert rest["ok"] is True
    w2 = {x["name"]: x for x in c.workers()}
    assert w2["computron"]["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1}
    assert "small-gguf" not in w2["ae"]["spill_by_model"]   # None -> {} clears
    assert all(v["matches"] for v in rest["per_worker"].values())


def test_engine_gate_refusal_surfaces_409_and_restore_is_clean():
    c = FakeClient()
    snap = _snap(c, "tf-model", ["computron"])
    bad = alloc.apply(c, "tf-model", {"n_gpu_layers": "off"}, snap)
    assert bad["ok"] is False
    assert bad["results"]["computron"]["status"] == 409
    assert "not GGUF" in bad["results"]["computron"]["error"]
    rest = alloc.restore(c, "tf-model", snap)
    assert rest["ok"] is True
    w = {x["name"]: x for x in c.workers()}
    assert "tf-model" not in w["computron"].get("spill_by_model", {})


def test_partial_apply_still_restores_the_worker_that_changed():
    c = FakeClient()
    snap = _snap(c, "small-gguf", ["computron", "ae"])
    snap["ae"]["worker_id"] = "wid-does-not-exist"     # force a failed write
    res = alloc.apply(c, "small-gguf", {"gpu_mem_gib": 2.0, "ctx_pct": 25}, snap)
    assert res["ok"] is False
    assert c.workers()[0]["spill_by_model"]["small-gguf"] == {"gpu_mem_gib": 2.0, "ctx_pct": 25}
    alloc.restore(c, "small-gguf", snap)
    w = {x["name"]: x for x in c.workers()}
    assert w["computron"]["spill_by_model"]["small-gguf"] == {"n_gpu_layers": -1}
