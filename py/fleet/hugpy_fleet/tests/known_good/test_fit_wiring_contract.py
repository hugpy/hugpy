"""KNOWN-GOOD CONTRACT — the worker's VRAM admission is wired through the pure
core (fleet side).

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/CORE-ISOLATION-DESIGN.md Part A.
Source under test: hugpy_fleet/worker/agent.py (_vram_evict_to_fit gathers a
VramSnapshot + FitPolicy + Resident rows + FitRequest ONCE, calls
hugpy_engine.fit.plan_fit, then executes the FitPlan), hugpy_fleet/worker/
flex.py (re-export shim over hugpy_engine.fit.flex).

Deterministic: the agent's device / resident / eviction seams are stubbed on
the module exactly as tests/test_vram_evict_to_fit.py does; no GPU, no worker.
"""
from __future__ import annotations

import importlib

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
gen_gate = importlib.import_module("hugpy_fleet.worker.gen_gate")
fit = importlib.import_module("hugpy_engine.fit")
fit_plan_mod = importlib.import_module("hugpy_engine.fit.plan")
fleet_flex = importlib.import_module("hugpy_fleet.worker.flex")
engine_flex = importlib.import_module("hugpy_engine.fit.flex")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


class _State:
    pass


@pytest.fixture
def rig(monkeypatch):
    """A 24 GiB card with a mutable free cell and a resident set; evicting a
    resident removes it and returns its bytes to free (the reclaim)."""
    for leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                 "HUGPY_LENIENCY_PCT", "HUGPY_PRIORITY_DEVICE", "HUGPY_BNB_4BIT",
                 "HUGPY_N_GPU_LAYERS", "HUGPY_VRAM_CEILING_FRAC",
                 "HUGPY_VRAM_RESERVE_GIB", "HUGPY_VRAM_CEILING_CUSHION_GIB",
                 "HUGPY_EVICT_LEAST_REAPING", "HUGPY_NO_EVICT"):
        monkeypatch.delenv(leak, raising=False)
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents: dict = {}
    lru: dict = {}
    evicted: list = []
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_incoming_need_detail",
                        lambda mk: {"total": card["need"], "weights": card["need"], "kv": 0,
                                    "ctx_pct": None, "ctx_resolved": None, "ctx_max": None,
                                    "geometry_source": None})
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v, "host_mode": "slot",
                                    "alive": True} for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: None)
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subject: set())
    monkeypatch.setattr(A, "detect_gpus", lambda: [])
    monkeypatch.setattr(A, "_target_device_index", lambda: None)
    monkeypatch.setattr(A, "_served_gguf_geometry", lambda mk: (None, None))
    monkeypatch.setattr(A, "_subject_resident_vram_bytes", lambda s, mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(D, "last_used_snapshot", lambda: dict(lru))

    def _fake_evict(state, mk, force=False):
        evicted.append(mk)
        vb = residents.pop(mk, None)
        if vb:
            card["free"] += vb
        return {"model_key": mk, "evicted": bool(vb), "vram_freed": vb, "host_mode": "slot"}
    monkeypatch.setattr(A, "_evict_model", _fake_evict)
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    return type("Rig", (), {"card": card, "residents": residents, "lru": lru,
                            "evicted": evicted})()


def test_admission_gathers_once_and_decides_through_plan_fit(rig, monkeypatch):
    """INVARIANT: _vram_evict_to_fit builds ONE VramSnapshot / FitPolicy /
    FitRequest / Resident set and consults hugpy_engine.fit.plan_fit for the
    decision; the executor then performs exactly the evictions the plan chose
    (coldest first) and re-proves fit against the live card. The verdict dict
    keeps its historical shape. Established: core isolation step 1 (2026-09-29)."""
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        plan = real(request, snapshot, residents, policy)
        calls.append((request, snapshot, tuple(residents), policy, plan))
        return plan
    monkeypatch.setattr(fit, "plan_fit", spy)

    rig.card["free"] = 1 * GIB
    rig.card["need"] = 4 * GIB
    rig.residents.update({"warm": 6 * GIB, "cold": 6 * GIB})
    rig.lru.update({"warm": 900.0, "cold": 100.0})

    verdict = A._vram_evict_to_fit(_State(), "subject")

    assert verdict["action"] == "evicted"
    assert verdict["evicted"] == ["cold"] and rig.evicted == ["cold"]
    assert verdict["freed_bytes"] == 6 * GIB and verdict["reason"] is None
    assert len(calls) == 1, "the happy eviction path plans exactly once"
    request, snapshot, residents, policy, plan = calls[0]
    assert isinstance(request, fit.FitRequest) and request.model_key == "subject"
    assert isinstance(snapshot, fit.VramSnapshot)
    assert snapshot.total_bytes == 24 * GIB and snapshot.free_bytes == 1 * GIB
    assert isinstance(policy, fit.FitPolicy)
    assert sorted(r.model_key for r in residents) == ["cold", "warm"]
    assert all(isinstance(r, fit.Resident) for r in residents)
    assert plan.action == "evict" and plan.evicted_keys == ["cold"]


def test_refusal_re_plans_the_tail_from_a_fresh_snapshot_with_no_candidates(rig, monkeypatch):
    """GUARD (step-2 slot): when the planned evictions leave the card short,
    today's contract re-plans ONLY the offload/refusal tail from a FRESH free
    read with NO remaining candidates; the refusal carries the plan's own
    structured reason alongside the operator sentence. Making the first plan's
    predicted budget authoritative ('stable budget') is the step-2 change and
    will retire this second call. Established: core isolation step 1."""
    calls: list = []
    real = fit_plan_mod.plan_fit

    def spy(request, snapshot, residents, policy):
        calls.append((snapshot, tuple(residents)))
        return real(request, snapshot, residents, policy)
    monkeypatch.setattr(fit, "plan_fit", spy)

    rig.card["free"] = 1 * GIB
    rig.card["need"] = 20 * GIB
    rig.residents.update({"idle": 2 * GIB})
    rig.lru.update({"idle": 100.0})

    verdict = A._vram_evict_to_fit(_State(), "subject")

    assert verdict["action"] == "refuse"
    assert verdict["evicted"] == ["idle"]
    assert len(calls) == 2
    assert calls[1][1] == ()                              # nothing left evictable
    assert calls[1][0].free_bytes == 3 * GIB              # the fresh read (1G + 2G freed)
    reason = verdict["reason"]
    assert reason["state"] == "refused" and "won't fit on GPU" in reason["reason"]
    assert reason["evicted_freed_bytes"] == 2 * GIB
    assert reason["plan_refuse_reason"].startswith("won't fit on GPU: needs 21474836480 B")


def test_fleet_flex_is_a_shim_over_the_engine_core():
    """GUARD: hugpy_fleet.worker.flex re-exports the engine's fit.flex objects
    (same function objects, not copies), so every existing fleet / server
    caller and every band/offload test keeps pinning ONE implementation.
    Established: core isolation step 1 (flex.py moved into hugpy_engine.fit)."""
    for name in ("plan_flex", "plan_partial_offload", "plan_explicit_offload",
                 "band_bounds", "band_ceiling", "band_floor", "ctx_band_bounds",
                 "kv_at_ctx_pct", "flex_priority_key", "leniency_floor_pct",
                 "FlexPlan", "PartialPlan"):
        assert getattr(fleet_flex, name) is getattr(engine_flex, name), name
    assert A._shared_evict_order.__module__ == A.__name__
    assert fit.evict_order.__module__ == "hugpy_engine.fit.plan"


def test_refusal_logs_one_structured_verdict_and_names_both_free_bases(rig, monkeypatch, caplog):
    """LIVE CASE (computron 2026-09-29): a 15.7 MiB model refused on a card whose
    BUDGETABLE free is 12451840 B behind a static resident. INVARIANT: the
    worker (a) emits exactly ONE `plan_fit verdict:` line per admission carrying
    action/kind/code and every figure in BYTES, and (b) the refusal reason names
    BOTH free bases — `free_vram_bytes` (budgetable, floor already out) and
    `free_vram_device_bytes` (= budgetable + `external_floor_bytes`) — plus the
    `fit_budget_bytes` the need was compared against, so the console can show
    why a tiny need failed on a card the heartbeat still reports ~1 GiB free.
    Established: fit-hotfix 2026-09-29."""
    import logging
    MIB = 1 << 20
    rig.card["total"] = 8_585_740_288
    rig.card["free"] = 12_451_840                 # budgetable (floor already out)
    rig.card["need"] = int(18.0 * MIB)
    rig.residents.update({"Qwen2.5-VL-7B-Instruct-GGUF": 6_578_765_824})
    rig.lru.update({"Qwen2.5-VL-7B-Instruct-GGUF": 100.0})
    monkeypatch.setattr(A, "_residency",
                        lambda mk: "static" if mk == "Qwen2.5-VL-7B-Instruct-GGUF" else "on-demand")
    monkeypatch.setattr(A, "_external_vram_floor_bytes", lambda: 1 * GIB)
    monkeypatch.setattr(A, "_vram_ceiling_reserve_bytes", lambda total: 0)

    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "test-save-tiny-random-llama3-smashed-pro")

    assert verdict["action"] == "refuse" and verdict["evicted"] == []
    assert rig.evicted == []                                    # static: never touched
    reason = verdict["reason"]
    assert reason["state"] == "refused"
    assert reason["needs_bytes"] == 18_874_368
    assert reason["free_vram_bytes"] == 12_451_840
    assert reason["external_floor_bytes"] == 1 * GIB
    assert reason["free_vram_device_bytes"] == 12_451_840 + 1 * GIB
    assert reason["fit_budget_bytes"] == 12_451_840
    assert reason["ceiling_reserve_bytes"] == 0
    assert reason["fit_failure"]["kind"] == "vram_fit"
    assert reason["fit_failure"]["code"] == "wont_fit"
    assert reason["fit_failure"]["need_bytes"] == 18_874_368
    assert reason["fit_failure"]["budget_bytes"] == 12_451_840
    assert "1 protected resident(s) still hold the card" in reason["reason"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) >= 1, "the admission must log a structured verdict"
    first = lines[0]
    for token in ("action=refuse", "fits_now=False", "kind=vram_fit", "code=wont_fit",
                  "need_bytes=18874368", "free_bytes=12451840",
                  f"device_free_bytes={12_451_840 + GIB}", "ceiling_reserve_bytes=0",
                  f"external_floor_bytes={GIB}", "total_bytes=8585740288", "protected=1"):
        assert token in first, (token, first)


def test_tiny_model_against_a_gib_of_budgetable_free_proceeds_and_logs_fits_now(rig, caplog):
    """LIVE CASE (ae-worker 2026-09-29 13:26:59): the same 15.7 MiB model with
    1185284096 B budgetable free proceeds — no eviction, no refusal — and the
    structured verdict line says so (action=proceed fits_now=True kind=None).
    Established: fit-hotfix 2026-09-29."""
    import logging
    MIB = 1 << 20
    rig.card["free"] = 1_185_284_096
    rig.card["need"] = int(18.0 * MIB)
    rig.residents.update({"Qwen3-Coder-Next-GGUF": 18_820_000_000})
    rig.lru.update({"Qwen3-Coder-Next-GGUF": 900.0})
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        verdict = A._vram_evict_to_fit(_State(), "test-save-tiny-random-llama3-smashed-pro")
    assert verdict["action"] == "proceed" and verdict["evicted"] == [] and verdict["reason"] is None
    assert rig.evicted == []
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan_fit verdict:")]
    assert len(lines) == 1
    for token in ("action=proceed", "fits_now=True", "kind=None", "need_bytes=18874368",
                  "free_bytes=1185284096"):
        assert token in lines[0], (token, lines[0])
