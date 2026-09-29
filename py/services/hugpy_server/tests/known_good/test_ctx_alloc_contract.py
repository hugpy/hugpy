"""KNOWN-GOOD CONTRACT — ctx_pct allocation contract at the central API.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_server/app/routes/worker_routes.py
(_validate_alloc_spill, _validate_band_values, _alloc_is_gguf_only,
_alloc_spill_ok_for_engine, POST /llm/workers/<id>/alloc-all) and
hugpy_engine/alloc_modes.py (normalize_spill, mode_to_spill).

Deterministic: the worker registry, model framework lookup, assign_model and
the worker relay are faked on the routes module (same harness shape as
tests/test_bulk_alloc_route.py). No worker, no restart.
"""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
AM = importlib.import_module("hugpy_engine.alloc_modes")

MIXED = {"id": "wid", "name": "box", "models": ["g1", "t1"], "models_local": []}
MIXED_FW = {"g1": "gguf", "t1": "transformers"}


@pytest.fixture
def harness(monkeypatch):
    calls: list = []
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    monkeypatch.setattr(wr, "get_worker", lambda wid: dict(MIXED) if wid == "wid" else None)
    monkeypatch.setattr(wr, "_model_framework", lambda mk: MIXED_FW.get(mk))

    def _assign(worker_id, model_key, spill=None, source=None, retag=True):
        calls.append((model_key, spill))
        return dict(MIXED)
    monkeypatch.setattr(wr, "assign_model", _assign)
    monkeypatch.setattr(wr, "_kick_warm", lambda w, keys, source: list(keys), raising=False)

    def _tripwire(*a, **k):
        raise AssertionError("alloc must never relay to the worker (no restart)")
    monkeypatch.setattr(wr, "_relay_worker_op", _tripwire)
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    return app.test_client(), calls


# ---------------------------------------------------------------------------
# ctx_pct is NOT a GGUF-only knob
# ---------------------------------------------------------------------------
def test_ctx_pct_is_engine_agnostic_not_gguf_only(monkeypatch):
    """INVARIANT: a spill carrying only ctx_pct / ctx_deviation_pct is NOT
    classified GGUF-only, so a context target is accepted for a Transformers
    (or any) model; explicit GPU/RAM budgets, MoE split and alloc_mode
    'explicit' remain GGUF-only; max-ram is engine-agnostic.
    Established: frontier handoff 2026-09-28 (worker_routes excludes ctx_pct
    from the GGUF-only budget rejection); max-ram ruling 2026-07-24."""
    assert wr._alloc_is_gguf_only({"ctx_pct": 40}) is False
    assert wr._alloc_is_gguf_only({"ctx_pct": 40, "ctx_deviation_pct": 10}) is False
    assert wr._alloc_is_gguf_only({"n_gpu_layers": -1, "ctx_pct": 40}) is False
    assert wr._alloc_is_gguf_only({"alloc_mode": "max-ram", "ctx_pct": 40}) is False
    assert wr._alloc_is_gguf_only({"gpu_mem_gib": 4}) is True
    assert wr._alloc_is_gguf_only({"n_cpu_moe": 39}) is True
    assert wr._alloc_is_gguf_only({"alloc_mode": "explicit"}) is True
    assert "ctx_pct" not in wr._GGUF_EXPLICIT_BUDGET_KEYS

    monkeypatch.setattr(wr, "_model_framework", lambda mk: "transformers")
    assert wr._alloc_spill_ok_for_engine({"ctx_pct": 40}, "t1") == (True, None)
    ok, reason = wr._alloc_spill_ok_for_engine({"gpu_mem_gib": 4}, "t1")
    assert ok is False and "GGUF-only" in reason


def test_ctx_pct_range_validation_at_the_door():
    """INVARIANT: ctx_pct must be an integer 1..100 (percent of max context);
    ctx_deviation_pct 0..100; unknown keys are rejected; None/{} mean autofit.
    Established: t21 band validation (shared by /assign and /alloc-all)."""
    clean, err = wr._validate_alloc_spill({"ctx_pct": 40})
    assert (clean, err) == ({"ctx_pct": 40}, None)
    assert wr._validate_alloc_spill(None) == ({}, None)
    assert wr._validate_alloc_spill({}) == ({}, None)
    for bad in ({"ctx_pct": 0}, {"ctx_pct": 101}, {"ctx_pct": "abc"},
                {"ctx_deviation_pct": 150}, {"ctx_pct": 40, "bogus": 1}):
        clean, err = wr._validate_alloc_spill(bad)
        assert clean is None and err, bad


def test_alloc_all_applies_ctx_pct_to_gguf_and_transformers_members(harness):
    """INVARIANT: a bulk allocation carrying a context target is applied to
    EVERY selected member regardless of engine (no 'GGUF-only' skip), while a
    GGUF-only explicit budget still skips the non-GGUF member honestly.
    Nothing is relayed to the worker (a context edit is a contract for the
    next load/reseat, not a restart).
    Established: frontier handoff 2026-09-28."""
    client, calls = harness
    r = client.post("/llm/workers/wid/alloc-all",
                    json={"model_keys": ["g1", "t1"], "spill": {"ctx_pct": 40}})
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["results"] == {"g1": "ok", "t1": "ok"}
    assert body["counts"] == {"ok": 2, "error": 0, "skipped": 0, "total": 2}
    assert body["restarting"] is False
    assert [c for c in calls] == [("g1", {"ctx_pct": 40}), ("t1", {"ctx_pct": 40})]

    calls.clear()
    body = client.post("/llm/workers/wid/alloc-all",
                       json={"model_keys": ["g1", "t1"], "spill": {"gpu_mem_gib": 4}}).get_json()
    assert body["results"]["g1"] == "ok"
    assert "skipped" in body["results"]["t1"] and "GGUF-only" in body["results"]["t1"]
    assert [c[0] for c in calls] == ["g1"]


# ---------------------------------------------------------------------------
# Apply merges the current allocation; Auto removes ctx_pct
# ---------------------------------------------------------------------------
def test_context_apply_merges_placement_fields_and_auto_removes_ctx_pct():
    """INVARIANT (the backend half of the Ctx cell contract): the persisted
    spill is the client's merged dict — placement fields (n_gpu_layers,
    alloc_mode max-ram/explicit, budgets) survive next to ctx_pct, and the
    same dict with ctx_pct removed ('Auto') is a valid spill that keeps the
    placement. normalize_spill never strips ctx_pct and never invents one.
    The UI half (WorkerRow.jsx applyContext: ``next = {...currentSpill};
    pct == null ? delete next.ctx_pct : next.ctx_pct = pct``) is pinned in the
    catalogue as a gap (no JS unit test).
    Established: frontier handoff 2026-09-28 ('Apply merges the current
    allocation spill, preserving placement fields. Auto removes ctx_pct')."""
    applied = {"n_gpu_layers": -1, "ctx_pct": 40}
    clean, err = wr._validate_alloc_spill(applied)
    assert err is None and clean == applied
    auto = {k: v for k, v in applied.items() if k != "ctx_pct"}
    assert wr._validate_alloc_spill(auto) == ({"n_gpu_layers": -1}, None)

    merged = {"alloc_mode": "max-ram", "ctx_pct": 40, "ctx_deviation_pct": 10}
    clean, _ = AM.normalize_spill(merged)
    assert clean == merged
    clean, _ = AM.normalize_spill({"alloc_mode": "gpu-only", "ctx_pct": 40})
    assert clean == {"n_gpu_layers": -1, "ctx_pct": 40}
    clean, _ = AM.normalize_spill({"alloc_mode": "explicit", "gpu_mem_gib": 6.0, "ctx_pct": 25})
    assert clean == {"alloc_mode": "explicit", "gpu_mem_gib": 6.0, "ctx_pct": 25}
    assert AM.mode_to_spill("gpu-only", ctx_pct=40) == {"n_gpu_layers": -1, "ctx_pct": 40}
    assert AM.mode_to_spill("max-ram", ctx_pct=40) == {"alloc_mode": "max-ram", "ctx_pct": 40}
