"""chaos assortment: enumeration, blocked-skip, framework-gated modes,
seeded-draw determinism, and hybrid feasibility (predicted-infeasible)."""
from __future__ import annotations

import random

import pytest
from chaos_fakes import GIB, make_models, make_workers

from hugpy_ops.chaos import assortment as A
from hugpy_ops.chaos.schema import SPILL_KEYS


@pytest.fixture
def fleet():
    return make_models(), make_workers()


def test_enumeration_skips_blocked_and_non_chat(fleet):
    models, workers = fleet
    enum = A.enumerate_assortment(models, workers)
    keys = [r["model_key"] for r in enum["models"]]
    assert "blocked-model" not in keys
    assert enum["blocked_excluded"] == ["blocked-model"]
    assert "image-only" not in keys
    assert enum["n_servable"] == 4   # small-gguf, tf-model, huge-gguf, unassigned-gguf
    assert any(r["model_key"] == "unassigned-gguf" and not r["exercisable"]
               for r in enum["models"])
    assert enum["n_exercisable"] == 3


def test_framework_gated_alloc_modes(fleet):
    models, _ = fleet
    sm = {m["model_key"]: m for m in A.servable_models(models)}
    assert set(A.modes_for(sm["small-gguf"]["framework"])) == set(A.ALLOC_MODES)
    assert A.modes_for(sm["tf-model"]["framework"]) == ("gpu-only", "ram-only", "max-gpu", "max-ram")


def test_candidate_workers_are_already_assigned_online(fleet):
    _, workers = fleet
    widx = A.worker_index(workers)
    assert A.candidate_workers("small-gguf", widx) == ["ae", "computron"]
    assert A.candidate_workers("huge-gguf", widx) == ["ae"]
    assert A.candidate_workers("unassigned-gguf", widx) == []


def test_seeded_draws_are_reproducible_and_safe(fleet):
    models, workers = fleet
    widx = A.worker_index(workers)
    draws_a = [A.draw_combo(random.Random(42), models, workers) for _ in range(1)]
    r_a, r_b = random.Random(42), random.Random(42)
    draws_a = [A.draw_combo(r_a, models, workers) for _ in range(20)]
    draws_b = [A.draw_combo(r_b, models, workers) for _ in range(20)]
    assert [d["model_key"] for d in draws_a] == [d["model_key"] for d in draws_b]
    assert [d["spill"] for d in draws_a] == [d["spill"] for d in draws_b]
    draws_c = [A.draw_combo(random.Random(43), models, workers) for _ in range(20)]
    assert [d["model_key"] for d in draws_a] != [d["model_key"] for d in draws_c]
    for d in draws_a:
        assert d["target_workers"]
        assert set(d["target_workers"]) <= set(A.candidate_workers(d["model_key"], widx))
        assert d["ctx_pct"] == d["spill"].get("ctx_pct")
        if d["alloc_mode"] == "ram-only":
            assert d["ctx_pct"] is None
    assert all(d["model_key"] != "blocked-model" for d in draws_a + draws_c)
    assert all(set(d["spill"]) <= SPILL_KEYS for d in draws_a + draws_c)


def test_hybrid_feasibility(fleet):
    _, workers = fleet
    huge = A.feasibility(400 * GIB, ["ae"], workers)
    assert huge["feasible"] is False and huge["infeasible_reason"]
    small = A.feasibility(2 * GIB, ["ae", "computron"], workers)
    assert small["feasible"] is True and small["infeasible_reason"] is None
    assert small["per_worker"]["ae"]["hybrid_total"] == (24 + 128) * GIB
    assert A.feasibility(None, ["ae"], workers)["feasible"] is True   # fails open
