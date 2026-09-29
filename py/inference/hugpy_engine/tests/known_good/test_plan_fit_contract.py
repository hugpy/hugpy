"""KNOWN-GOOD CONTRACT — plan_fit, the pure evict-to-fit core (engine side).

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/CORE-ISOLATION-DESIGN.md Part A (core isolation, step 1).
Source under test: hugpy_engine/fit/plan.py (plan_fit, evict_order),
hugpy_engine/fit/types.py (VramSnapshot, FitPolicy, Resident, FitRequest,
FitPlan), hugpy_engine/fit/flex.py (moved from hugpy_fleet/worker/flex.py).

Deterministic by construction: plan_fit takes every fact as an argument — no
GPU, no env, no clock, no worker. The standalone-import test runs a child
interpreter so the module's import weight is measured, not assumed.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
import sys

import pytest

fit = importlib.import_module("hugpy_engine.fit")
ev = importlib.import_module("hugpy_engine.eviction")

GIB = 1 << 30
NOW = 1_000_000.0


def _snap(free, total=24 * GIB, **kw):
    return fit.VramSnapshot(total_bytes=total, free_bytes=free, now=NOW, **kw)


def _req(need, **kw):
    det = {"total": need, "weights": need, "kv": 0, "ctx_pct": None}
    det.update(kw.pop("detail", {}))
    return fit.FitRequest(model_key="subject", need_bytes=need,
                          need_detail=det, **kw)


def _res(mk, vram, last_call=None, **kw):
    return fit.Resident(model_key=mk, vram_bytes=vram, host_mode="slot",
                        last_call=last_call, **kw)


POLICY = fit.FitPolicy(ceiling_reserve_bytes=1 * GIB)


# ---------------------------------------------------------------------------
# fits without eviction
# ---------------------------------------------------------------------------
def test_fits_without_eviction_leaves_every_resident_alone():
    """INVARIANT: when free VRAM (less the ceiling reserve) already holds the
    need, plan_fit proceeds with NO evictions, reports fits_now, and leaves the
    eviction-aware size-up available to the executor. Uncontended == target:
    no band, no eviction, no partial plan is consulted.
    Established: core isolation step 1 (2026-09-29), pinning the orchestrator's
    'ok' branch."""
    plan = fit.plan_fit(_req(4 * GIB), _snap(10 * GIB),
                        [_res("idle", 6 * GIB, last_call=100.0)], POLICY)
    assert plan.action == "proceed"
    assert plan.fits_now is True
    assert plan.evictions == () and plan.evicted_keys == []
    assert plan.size_up_eligible is True
    assert plan.partial is None and plan.refuse_reason is None
    assert plan.need_bytes == 4 * GIB


# ---------------------------------------------------------------------------
# one eviction, LRU / policy order
# ---------------------------------------------------------------------------
def test_needs_one_eviction_takes_the_coldest_resident_only():
    """INVARIANT: over the ceiling, plan_fit hands back the MINIMUM eviction
    set via the shared eviction function: the coldest (longest-idle) evictable
    resident goes first and the walk stops once the deficit is covered; the
    warmer resident is untouched. predicted_fits says the set suffices.
    Established: core isolation step 1; the order is eviction.sort_key (key 2:
    oldest idle anchor first), unchanged."""
    residents = [_res("warm", 6 * GIB, last_call=900.0),
                 _res("cold", 6 * GIB, last_call=100.0)]
    plan = fit.plan_fit(_req(4 * GIB), _snap(1 * GIB), residents, POLICY)
    assert plan.action == "evict"
    assert plan.evicted_keys == ["cold"]
    assert plan.eviction_need_bytes == 4 * GIB          # reserve 1G - (1G - 4G)
    assert plan.predicted_freed_bytes == 6 * GIB
    assert plan.predicted_fits is True


def test_flex_priority_is_the_outer_eviction_key():
    """INVARIANT: the operator's explicit per-model priority is the OUTER key —
    a lower-priority resident yields before a higher-priority one even when
    the higher-priority one is colder. Bands are planned against the need
    that remains, so least reaping never inverts the operator's order.
    Established: evict-order priority bands (agent._shared_evict_order),
    re-pinned on the pure evict_order."""
    residents = [_res("important-cold", 6 * GIB, last_call=100.0, priority=5),
                 _res("plain-warm", 6 * GIB, last_call=900.0, priority=0)]
    plan = fit.plan_fit(_req(4 * GIB), _snap(1 * GIB), residents, POLICY)
    assert plan.evicted_keys == ["plain-warm"]


def test_evict_order_degrades_to_the_full_pool_when_need_is_unknown():
    """INVARIANT: evict_order(need=None) returns the whole pool in shared-key
    order (degrade-not-guess) and never invents a byte figure to plan a drop
    pass against. Established: agent._shared_evict_order contract, now pure."""
    units = [ev.EvictUnit("b", bytes=GIB, last_call=500.0),
             ev.EvictUnit("a", bytes=GIB, last_call=100.0)]
    assert fit.evict_order(units, None, {}, now=NOW) == ["a", "b"]


# ---------------------------------------------------------------------------
# nothing evictable -> structured, honest refusal
# ---------------------------------------------------------------------------
def test_nothing_evictable_is_a_structured_refusal_naming_the_numbers():
    """INVARIANT: when every resident is protected and no hybrid is possible
    (no GGUF geometry), plan_fit REFUSES with no evictions and a deterministic
    refuse_reason that names need, free, total, reserve and the protected
    count — never admit-then-OOM, never a silent proceed.
    Established: core isolation step 1 (design A.4 invariant 6)."""
    residents = [_res("locked", 20 * GIB, protected=True, why="static (locked residency)"),
                 _res("busy", 2 * GIB, protected=True, why="actively replying (in-flight/busy)")]
    plan = fit.plan_fit(_req(10 * GIB), _snap(1 * GIB), residents, POLICY)
    assert plan.action == "refuse"
    assert plan.evictions == () and plan.predicted_fits is False
    assert plan.partial is None
    r = plan.refuse_reason
    assert r.startswith("won't fit on GPU: needs 10737418240 B, 1073741824 B free of")
    assert "1073741824 B ceiling reserve" in r
    assert "2 protected resident(s)" in r
    assert "no partial offload possible" in r
    assert plan.need_bytes == 10 * GIB and plan.free_bytes == 1 * GIB
    assert plan.ceiling_reserve_bytes == 1 * GIB


def test_protection_outranks_subject_priority():
    """INVARIANT: a protected (static / replying / queued-ahead) resident is
    never planned for eviction regardless of the subject's priority, while an
    idle on-demand resident is; the plan still refuses honestly when the
    evictable set cannot cover the need. Established: two-classes-only
    protection ruling; design A.4 invariant 4."""
    residents = [_res("locked", 18 * GIB, protected=True, why="static (locked residency)"),
                 _res("idle", 2 * GIB, last_call=100.0)]
    plan = fit.plan_fit(_req(10 * GIB, priority=9), _snap(1 * GIB), residents, POLICY)
    assert plan.evicted_keys == ["idle"]          # walked, but not enough
    assert plan.action == "refuse"
    assert "locked" not in plan.evicted_keys


# ---------------------------------------------------------------------------
# reserve + subject credit
# ---------------------------------------------------------------------------
def test_ceiling_reserve_is_respected_to_the_byte():
    """INVARIANT: the fit test is (free + credit - need) >= ceiling_reserve.
    One byte short of the reserve is over the ceiling; exactly the reserve
    fits. Established: _vram_ceiling_reserve_bytes is THE one binding every
    stage prices against."""
    need = 4 * GIB
    short = fit.plan_fit(_req(need), _snap(need + GIB - 1), [], POLICY)
    exact = fit.plan_fit(_req(need), _snap(need + GIB), [], POLICY)
    assert short.action == "refuse" and short.fits_now is False
    assert exact.action == "proceed" and exact.fits_now is True


def test_subject_credit_is_headroom_for_the_subject_only():
    """INVARIANT: an already-resident subject's own footprint is credited on
    the FREE side (never subtracted from need): a re-seat that fits only with
    its own bytes back proceeds, and the refusal figures still report the raw
    free plus the credited effective figure. Established: subject-credit fix
    2026-07-27; design A.4 invariant 5."""
    need = 8 * GIB
    without = fit.plan_fit(_req(need), _snap(2 * GIB), [], POLICY)
    with_credit = fit.plan_fit(_req(need, subject_held_bytes=7 * GIB), _snap(2 * GIB), [], POLICY)
    assert without.action == "refuse"
    assert with_credit.action == "proceed"
    assert with_credit.need_bytes == need                 # need is untouched
    assert with_credit.free_effective_bytes == 9 * GIB
    # A credit that still leaves the DELTA short refuses, and says so.
    still_short = fit.plan_fit(_req(need, subject_held_bytes=5 * GIB), _snap(2 * GIB), [], POLICY)
    assert still_short.action == "refuse"
    assert "held by the subject, credited" in still_short.refuse_reason


# ---------------------------------------------------------------------------
# polite load, flex, partial offload, MoE re-target, fail-open
# ---------------------------------------------------------------------------
def test_polite_load_spares_every_candidate_and_names_them():
    """INVARIANT: no_evict spends only free headroom — no evictions are
    planned, the evictable residents are listed as spared, and comfy reclaim
    is not offered to the executor. Established: k56 polite load."""
    residents = [_res("idle-a", 6 * GIB, last_call=100.0), _res("idle-b", 6 * GIB, last_call=200.0)]
    plan = fit.plan_fit(_req(4 * GIB, polite=True), _snap(1 * GIB), residents, POLICY)
    assert plan.evictions == ()
    assert sorted(e.model_key for e in plan.polite_spared) == ["idle-a", "idle-b"]
    assert plan.comfy_reclaim_eligible is False
    assert plan.action == "refuse" and "spared by the polite load" in plan.refuse_reason


def test_self_ctx_flex_clears_the_ceiling_before_any_eviction():
    """INVARIANT: ctx is the cheapest flex — a subject with a ctx band is
    compressed to its band floor, its KV re-priced linearly and the need
    lowered, and when that alone clears the ceiling the plan is 'flex' with no
    evictions even though an idle resident was available.
    Established: t21 tolerance bands; flex.plan_flex order of operations."""
    det = {"total": 9 * GIB, "weights": 4 * GIB, "kv": 5 * GIB, "ctx_pct": 50}
    req = _req(9 * GIB, detail=det, ctx_deviation_pct=20)      # band floor 30%
    plan = fit.plan_fit(req, _snap(9 * GIB), [_res("idle", 6 * GIB, last_call=100.0)], POLICY)
    assert plan.action == "flex"
    assert plan.self_ctx_pct == 30
    assert plan.need_bytes == 4 * GIB + 3 * GIB               # kv 5G * 30/50
    assert plan.evictions == ()


def test_gguf_that_cannot_fit_whole_is_offloaded_not_refused():
    """INVARIANT (feasibility preference): a GGUF whose full offload does not
    fit even after the eviction walk is admitted as an honest partial offload
    (layers that fit under the reserve, remainder to host RAM) — 'partial',
    never 'refuse' — when the hybrid is admissible. Established: autofit's
    hybrid contract (stage 2.5); design A.4 invariant 3."""
    det = {"total": 16 * GIB, "weights": 16 * GIB, "kv": 0, "ctx_pct": None}
    req = _req(16 * GIB, detail=det, gguf_path="/m/q.gguf", total_layers=32)
    plan = fit.plan_fit(req, _snap(6 * GIB, ram_free_bytes=64 * GIB), [], POLICY)
    assert plan.action == "partial"
    assert plan.partial_kind == "dense"
    assert 0 < plan.n_gpu_layers < 32
    assert plan.partial["admit"] is True
    assert plan.budget_bytes == 5 * GIB                      # free 6G - reserve 1G


def test_moe_split_retargets_the_gpu_need_unless_experts_exceed_host_ram():
    """INVARIANT: when the need detail carries a governing MoE split, admission
    prices the split's GPU share (moe_commit set) — unless the expert tensors
    exceed 95% of the box's RAM, in which case the full need stands and the
    skip is attributed. Established: MoE re-target default 2026-07-25 + the
    mmap RAM guard (coder-next/ae 2026-08-28)."""
    det = {"total": 50 * GIB, "weights": 50 * GIB, "kv": 0, "ctx_pct": None,
           "moe_split": {"path": "/m/moe.gguf", "n_cpu_moe": 40,
                         "gpu_total": 4 * GIB, "cpu_bytes": 44 * GIB}}
    req = _req(50 * GIB, detail=det)
    big_box = fit.plan_fit(req, _snap(10 * GIB, ram_total_bytes=128 * GIB), [], POLICY)
    assert big_box.action == "proceed" and big_box.need_bytes == 4 * GIB
    assert big_box.moe_commit["n_cpu_moe"] == 40
    small_box = fit.plan_fit(req, _snap(10 * GIB, ram_total_bytes=32 * GIB), [], POLICY)
    assert small_box.moe_commit is None and small_box.need_bytes == 50 * GIB
    assert any("MoE split skipped" in r for r in small_box.reasons)


def test_unmeasurable_inputs_fail_open_never_closed():
    """INVARIANT: no GPU / unknown need / unreadable free VRAM each yield
    'proceed' with a note and fits_now None — an unmeasurable load is never
    blocked because we couldn't measure. Established: the orchestrator's
    fail-open guards, unchanged."""
    assert fit.plan_fit(_req(4 * GIB), _snap(1 * GIB, total=None), [], POLICY).note.startswith("no GPU")
    assert fit.plan_fit(_req(None), _snap(1 * GIB), [], POLICY).note.startswith("unknown weight size")
    p = fit.plan_fit(_req(4 * GIB), _snap(None), [], POLICY)
    assert p.action == "proceed" and p.fits_now is None and "fail open" in p.note
    zero = fit.plan_fit(_req(4 * GIB, planned_gpu_bytes=0), _snap(1 * GIB), [], POLICY)
    assert zero.action == "proceed" and "0 B on the GPU" in zero.note


# ---------------------------------------------------------------------------
# determinism + immutability
# ---------------------------------------------------------------------------
def test_same_inputs_yield_an_identical_plan_every_time():
    """INVARIANT: plan_fit is a pure function — same snapshot, policy,
    residents and request produce an equal FitPlan (and as_dict) across
    repeated calls and across resident input order, and the inputs are not
    mutated. This is the stable-reserve invariant: nothing is re-read
    mid-decision, so the verdict cannot drift. Established: core isolation
    step 1 (design A.4 invariants 1-2)."""
    det = {"total": 9 * GIB, "weights": 6 * GIB, "kv": 3 * GIB, "ctx_pct": 60,
           "moe_split": None}
    req = _req(9 * GIB, detail=det, ctx_deviation_pct=10, subject_held_bytes=GIB,
               gguf_path="/m/q.gguf", total_layers=40)
    residents = [_res("c", 3 * GIB, last_call=300.0, calls=3),
                 _res("a", 5 * GIB, last_call=100.0, calls=1),
                 _res("p", 9 * GIB, protected=True, why="static (locked residency)"),
                 _res("b", 4 * GIB, last_call=200.0, calls=2)]
    snap = _snap(1 * GIB, ram_free_bytes=64 * GIB, ram_total_bytes=128 * GIB,
                 external_floor_bytes=GIB)
    det_before = dict(det)
    first = fit.plan_fit(req, snap, residents, POLICY)
    for _ in range(5):
        assert fit.plan_fit(req, snap, residents, POLICY) == first
    # Resident INPUT ORDER is not part of the decision: the eviction set and
    # every priced figure are identical (plan_flex's tie-ordered
    # `priority_order` hint is the one input-order-dependent diagnostic).
    shuffled = fit.plan_fit(req, snap, list(reversed(residents)), POLICY)
    for name in ("action", "need_bytes", "self_ctx_pct", "evictions",
                 "eviction_need_bytes", "predicted_freed_bytes", "predicted_fits",
                 "partial", "partial_kind", "failure", "refuse_reason", "split"):
        assert getattr(shuffled, name) == getattr(first, name), name
    assert fit.plan_fit(req, snap, residents, POLICY).as_dict() == first.as_dict()
    assert req.need_detail == det_before                  # inputs untouched
    assert first.evicted_keys == ["a", "b", "c"][:len(first.evictions)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.action = "refuse"                           # type: ignore[misc]


# ---------------------------------------------------------------------------
# standalone import: the core is importable with engine deps only
# ---------------------------------------------------------------------------
def test_fit_core_imports_standalone_without_torch_nvml_or_fleet():
    """INVARIANT: `import hugpy_engine.fit` succeeds in a fresh interpreter
    with only hugpy_engine's declared dependencies on the path and pulls in
    neither torch / pynvml nor any hugpy_fleet / hugpy_media / hugpy_server
    module — the evict-to-fit core is a headless, standalone unit.
    Established: operator direction 2026-09-29 (core isolation step 1)."""
    code = (
        "import sys, hugpy_engine.fit as F\n"
        "bad = sorted({m.split('.')[0] for m in sys.modules} & "
        "{'torch', 'pynvml', 'hugpy_fleet', 'hugpy_media', 'hugpy_server', "
        "'hugpy_video', 'hugpy_ops'})\n"
        "assert not bad, bad\n"
        "assert callable(F.plan_fit) and F.FitPlan and F.VramSnapshot\n"
        "print('ok')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# the diagnosed case: coder-next "loaded and idle" (2026-09-29)
# ---------------------------------------------------------------------------
def _coder_next_moe_detail():
    """Qwen3-Coder-Next-GGUF as gguf_moe_detail sees it: a 1.49 GiB dense
    backbone + 43.59 GiB of expert tensors over 48 blocks (~0.908 GiB each)."""
    per_layer = 975_000_000
    return {"is_moe": True, "expert_count": 512, "expert_used": 10,
            "non_expert_bytes": 1_600_000_000,
            "expert_bytes": 48 * per_layer,
            "expert_bytes_by_layer": {i: per_layer for i in range(48)}}


def test_polite_moe_admission_over_the_ram_budget_fails_at_plan_time_without_evicting():
    """INVARIANT (diagnosis notes/coder-next-loaded-idle-diagnosis-2026-09-29.md
    (b).3/(b).4): a POLITE admission of a 48 GB MoE GGUF on a card with ~6 GiB
    free (slot budget ~1.5 GB after the ctx reserve) re-derives the split to
    --n-cpu-moe 999 (all experts to RAM), which puts ~43.6 GiB in host RAM —
    over the contract's 35.1 GiB RAM budget that was priced at --n-cpu-moe 39.
    plan_fit returns a structured FitFailure(kind=ram_budget, code=over_budget)
    with need and budget in the SAME basis and both N recorded, evicts NOTHING
    (polite; the pinned resident is spared, not evicted), and never lets the
    caller fall through to an in-process runner. Established: core isolation
    step 1 (2026-09-29), reproducing the 12:24/12:31/12:32 CDT cycles."""
    weights = 48_410_992_032
    det = {"total": weights, "weights": weights, "kv": 0, "ctx_pct": None}
    req = fit.FitRequest(model_key="Qwen3-Coder-Next-GGUF", need_bytes=weights,
                         need_detail=det, polite=True,
                         gguf_path="/models/coder-next.gguf", total_layers=48,
                         ngl_intent="gpu", ngl_requested=None,
                         moe_detail=_coder_next_moe_detail(),
                         moe_auto_gpu_budget_bytes=1_700_000_000,     # "~1.5 GB"
                         contract_n_cpu_moe=39)
    snap = fit.ResourceSnapshot(total_bytes=25_769_803_776, free_bytes=6_023_413_760,
                                ram_free_bytes=104_906_166_272,
                                ram_total_bytes=134_112_841_728, now=NOW)
    policy = fit.FitPolicy(ceiling_reserve_bytes=0, alloc_mode="explicit",
                           ram_target_bytes=37_686_190_538)            # 35.098 GiB
    residents = [
        fit.Resident("Qwen3.5-9B-DeepSeek-V4-Flash-GGUF", vram_bytes=14_535_557_120,
                     host_mode="slot", pinned=True, materialized=True, last_call=NOW - 60),
        fit.Resident("Qwythos-9B-Claude-Mythos-5-1M", vram_bytes=1_374_389_534,
                     host_mode="in_process", materialized=True, last_call=NOW - 900),
        fit.Resident("Surogate-3.5-2B", vram_bytes=1_374_389_534,
                     host_mode="in_process", materialized=True, last_call=NOW - 1800),
    ]
    plan = fit.plan_fit(req, snap, residents, policy)
    assert plan.action == "refuse"
    f = plan.failure
    assert f is not None and f.kind == "ram_budget" and f.code == "over_budget"
    assert f.need_bytes == 48 * 975_000_000                # ~43.6 GiB at N=999
    assert abs(f.need_bytes / GIB - 43.6) < 0.1
    assert f.budget_bytes == 37_686_190_538 and abs(f.budget_bytes / GIB - 35.1) < 0.01
    assert f.plan_n_cpu_moe == 999 and f.contract_n_cpu_moe == 39
    assert f.state_dependent is True and f.permanent is False
    assert plan.split.n_cpu_moe == 999 and plan.split.cpu_bytes == f.need_bytes
    assert plan.ram_need_bytes == f.need_bytes and plan.ram_budget_bytes == f.budget_bytes
    assert plan.evictions == ()                             # polite: nothing evicted
    assert "Qwen3.5-9B-DeepSeek-V4-Flash-GGUF" not in plan.evicted_keys
    assert sorted(e.model_key for e in plan.polite_spared) == sorted(r.model_key for r in residents)
    assert "over this model's RAM budget" in plan.refuse_reason
    assert "--n-cpu-moe 39" in plan.refuse_reason          # the basis mismatch is named
    # Same inputs, same failure — the verdict is a function of the snapshot.
    assert fit.plan_fit(req, snap, residents, policy) == plan


def test_split_within_the_ram_budget_is_admitted_with_need_and_budget_in_one_basis():
    """INVARIANT: the same MoE load whose contract RAM budget covers the expert
    bytes the plan actually spills is ADMITTED as the split, and the plan
    carries n_cpu_moe with the RAM need it implies at that N beside the budget
    it was checked against. Established: core isolation step 1 ((b).3)."""
    weights = 48_410_992_032
    det = {"total": weights, "weights": weights, "kv": 0, "ctx_pct": None}
    req = fit.FitRequest(model_key="Qwen3-Coder-Next-GGUF", need_bytes=weights,
                         need_detail=det, polite=True,
                         gguf_path="/models/coder-next.gguf", total_layers=48,
                         ngl_intent="gpu", moe_detail=_coder_next_moe_detail(),
                         moe_auto_gpu_budget_bytes=1_700_000_000, contract_n_cpu_moe=39)
    snap = fit.ResourceSnapshot(total_bytes=25_769_803_776, free_bytes=6_023_413_760,
                                ram_free_bytes=104_906_166_272,
                                ram_total_bytes=134_112_841_728, now=NOW)
    generous = fit.FitPolicy(ceiling_reserve_bytes=0, alloc_mode="explicit",
                             ram_target_bytes=int(46 * GIB))
    plan = fit.plan_fit(req, snap, [], generous)
    assert plan.action == "partial" and plan.partial_kind == "moe-first"
    assert plan.n_gpu_layers == -1 and plan.n_cpu_moe == 999
    assert plan.split == fit.MoeSplit(n_cpu_moe=999, gpu_bytes=1_600_000_000,
                                      cpu_bytes=48 * 975_000_000, basis="moe-first",
                                      contract_n_cpu_moe=39)
    assert plan.ram_need_bytes == 48 * 975_000_000 and plan.ram_budget_bytes == int(46 * GIB)
    assert plan.failure is None


def test_inexpressible_split_is_a_plan_time_failure_not_a_fallback():
    """INVARIANT ((b).5): when the caller states that no engine can express a
    MoE split (split_expressible=False), a split plan is refused at PLAN time
    with FitFailure(kind=split_not_expressible); unknown (None) is not checked.
    Established: core isolation step 1."""
    weights = 48_410_992_032
    det = {"total": weights, "weights": weights, "kv": 0, "ctx_pct": None}
    base = dict(model_key="m", need_bytes=weights, need_detail=det, polite=True,
                gguf_path="/m.gguf", total_layers=48, ngl_intent="gpu",
                moe_detail=_coder_next_moe_detail(), moe_auto_gpu_budget_bytes=1_700_000_000)
    snap = fit.ResourceSnapshot(total_bytes=24 * GIB, free_bytes=6 * GIB,
                                ram_total_bytes=128 * GIB, now=NOW)
    policy = fit.FitPolicy(ceiling_reserve_bytes=0)
    no_engine = fit.plan_fit(fit.FitRequest(split_expressible=False, **base), snap, [], policy)
    assert no_engine.action == "refuse" and no_engine.failure.kind == "split_not_expressible"
    unknown = fit.plan_fit(fit.FitRequest(split_expressible=None, **base), snap, [], policy)
    assert unknown.action == "partial" and unknown.failure is None


def test_snapshot_and_resident_carry_measured_facts_for_step_two():
    """GUARD: the records can carry the step-2 inputs — ResourceSnapshot holds
    per-device VRAM + host RAM + reserves (VramSnapshot is its alias), Resident
    carries a MEASURED `materialized` (True/False/None) that plan_fit never
    infers, and FitPlan carries the split with its RAM need. Established: core
    isolation step 1 (diagnosis (b).1/(b).4)."""
    assert fit.VramSnapshot is fit.ResourceSnapshot
    snap = fit.ResourceSnapshot(total_bytes=24 * GIB, free_bytes=8 * GIB,
                                devices=(fit.DeviceVram(0, 24 * GIB, 8 * GIB),
                                         fit.DeviceVram(1, 24 * GIB, 24 * GIB)),
                                target_device=0, ram_free_bytes=64 * GIB,
                                ram_total_bytes=128 * GIB, ram_reserve_bytes=4 * GIB, now=NOW)
    hollow = fit.Resident("hollow", vram_bytes=None, host_mode="in_process", materialized=False)
    live = fit.Resident("live", vram_bytes=2 * GIB, host_mode="slot", materialized=True,
                        last_call=100.0)
    plan = fit.plan_fit(_req(4 * GIB), snap, [hollow, live], POLICY)
    assert plan.action == "proceed" and plan.split is None and plan.failure is None
    assert {r.materialized for r in (hollow, live)} == {False, True}
    assert fit.Resident("x").materialized is None
    d = plan.as_dict()
    assert d["ram_budget_bytes"] is None and d["split"] is None


# ---------------------------------------------------------------------------
# fit-hotfix 2026-09-29: the operator's "15.7 MiB failed vram_fit" case, pinned
# ---------------------------------------------------------------------------
MIB = 1 << 20
EXTERNAL_FLOOR = 1 * GIB                     # the HUGPY_VRAM_RESERVE_GIB in force
                                             # that day (default 0 since 2026-09-29)


def _default_reserve(total):
    """agent._vram_ceiling_reserve_bytes default arithmetic, in bytes:
    max(0, min(512 MiB cushion, 10% of the card) - external floor)."""
    return max(0, min(512 * MIB, int(total * 0.10)) - EXTERNAL_FLOOR)


def _tiny_request(need=int(18.0 * MIB)):
    weights = int(15.7 * MIB)
    return fit.FitRequest(model_key="test-save-tiny-random-llama3-smashed-pro",
                          need_bytes=need,
                          need_detail={"total": need, "weights": weights,
                                       "kv": need - weights, "ctx_pct": 100},
                          planned_gpu_bytes=need)


def test_live_computron_2026_09_29_tiny_model_refused_on_a_full_card_is_honest():
    """LIVE CASE (computron, 2026-09-29 13:26:35, RTX 4060 Laptop 8 GiB): the
    operator's 15.7 MiB transformers model was refused 'vram_fit'. The worker
    gathered total=8585740288 B, BUDGETABLE free=12451840 B (device free
    1086324736 B less the 1 GiB external floor), ceiling reserve 0 B, one
    PROTECTED static max-gpu resident (Qwen2.5-VL-7B, 6578765824 B). INVARIANT:
    plan_fit reproduces that verdict byte-for-byte — refuse, kind=vram_fit,
    code=wont_fit, need 18874368 B vs budget 12451840 B — because the card
    genuinely had 11.9 MiB of budgetable room; the refusal is honest, not a
    units error, and the protected resident is never proposed for eviction.
    Established: fit-hotfix 2026-09-29 (evidence in load_reports on central)."""
    total, free = 8_585_740_288, 12_451_840
    snap = fit.ResourceSnapshot(total_bytes=total, free_bytes=free,
                                external_floor_bytes=EXTERNAL_FLOOR,
                                ram_free_bytes=6_306_209_792, ram_total_bytes=16 * GIB,
                                devices=(fit.DeviceVram(0, total, free),),
                                target_device=0, now=NOW)
    policy = fit.FitPolicy(ceiling_reserve_bytes=_default_reserve(total),
                           empty_card_budget_bytes=total - EXTERNAL_FLOOR)
    assert policy.ceiling_reserve_bytes == 0                # "0 B ceiling reserve"
    static_vl7b = fit.Resident("Qwen2.5-VL-7B-Instruct-GGUF", vram_bytes=6_578_765_824,
                               host_mode="slot", protected=True, why="static residency",
                               materialized=True)
    plan = fit.plan_fit(_tiny_request(), snap, [static_vl7b], policy)
    assert plan.action == "refuse" and plan.fits_now is False
    assert plan.failure is not None
    assert plan.failure.kind == "vram_fit" and plan.failure.code == "wont_fit"
    assert plan.failure.need_bytes == 18_874_368
    assert plan.failure.budget_bytes == 12_451_840
    assert plan.free_bytes == free and plan.free_effective_bytes == free
    assert plan.subject_held_bytes == 0 and plan.ceiling_reserve_bytes == 0
    assert plan.evictions == ()                             # protected: never proposed
    assert "12451840 B free of 8585740288 B" in plan.refuse_reason
    assert "1073741824 B external floor" in plan.refuse_reason
    assert "1 protected resident(s)" in plan.refuse_reason


def test_live_ae_worker_2026_09_29_same_tiny_model_proceeds_with_1_gib_budgetable():
    """LIVE CASE (ae-worker, 2026-09-29 13:26:59, RTX 3090): the SAME 15.7 MiB
    model against 1185284096 B budgetable free (Coder-Next 18.8 GB resident)
    proceeds without touching any resident — which is what happened (probe
    ok=True, fit=True, 315949056 B used). INVARIANT: a MiB-scale need against a
    GiB-scale free figure is 'proceed' with fits_now=True and no evictions.
    Established: fit-hotfix 2026-09-29."""
    total, free = 25_769_803_776, 1_185_284_096
    snap = fit.ResourceSnapshot(total_bytes=total, free_bytes=free,
                                external_floor_bytes=EXTERNAL_FLOOR, now=NOW)
    policy = fit.FitPolicy(ceiling_reserve_bytes=_default_reserve(total))
    coder_next = fit.Resident("Qwen3-Coder-Next-GGUF", vram_bytes=18_820_000_000,
                              host_mode="slot", last_call=NOW - 10.0, materialized=True)
    plan = fit.plan_fit(_tiny_request(), snap, [coder_next], policy)
    assert plan.action == "proceed" and plan.fits_now is True
    assert plan.evictions == () and plan.failure is None
    assert plan.need_bytes == 18_874_368 and plan.free_bytes == free


@pytest.mark.parametrize("free_gib", [1, 2, 8, 24])
def test_units_sanity_every_field_is_bytes_and_15_7_mib_fits_1_gib_free(free_gib):
    """UNITS SANITY: ResourceSnapshot/FitRequest/FitPolicy/FitPlan carry BYTES
    (plain ints, no MiB/GiB mixing). A 15.7 MiB need against >= 1 GiB budgetable
    free must proceed with fits_now=True under the default reserve, and the
    plan echoes the inputs unchanged (no rescaling anywhere in the core).
    Established: fit-hotfix 2026-09-29."""
    total, free = 24 * GIB, free_gib * GIB
    snap = fit.ResourceSnapshot(total_bytes=total, free_bytes=free,
                                external_floor_bytes=EXTERNAL_FLOOR, now=NOW)
    req = _tiny_request()
    policy = fit.FitPolicy(ceiling_reserve_bytes=_default_reserve(total))
    for v in (snap.total_bytes, snap.free_bytes, snap.external_floor_bytes,
              req.need_bytes, req.planned_gpu_bytes, policy.ceiling_reserve_bytes):
        assert type(v) is int, (v, type(v))
    assert req.need_bytes < 32 * MIB < GIB <= snap.free_bytes
    plan = fit.plan_fit(req, snap, [], policy)
    assert plan.action == "proceed" and plan.fits_now is True
    assert plan.need_bytes == req.need_bytes == 18_874_368
    assert plan.free_bytes == free and plan.total_bytes == total
    assert plan.free_effective_bytes == free           # no subject credit
    assert plan.ceiling_reserve_bytes == policy.ceiling_reserve_bytes
    assert plan.failure is None and plan.evictions == ()


def test_load_refusal_reaches_the_wire_structured_not_as_prose_only():
    """INVARIANT (fit-hotfix 2026-09-29): when the worker refuses through
    plan_fit, the LoadRefusal it raises carries the typed verdict, and
    ``serve.load_failure.load_failure_of`` surfaces it STRUCTURED — class
    'vram_fit' plus ``fit_failure`` (kind/code/need/budget in one basis) and
    ``refusal`` (budgetable free, device free, external floor, reserve,
    protected residents) — so load_reports / the inference error can show WHY a
    15.7 MiB need failed on a card the heartbeat still reports ~1 GiB free.
    The prose sentence is kept in ``message``. Established: fit-hotfix
    2026-09-29 (computron live case)."""
    LF = importlib.import_module("hugpy_engine.serve.load_failure")
    D = importlib.import_module("hugpy_engine.dispatch.dispatch")
    total, free = 8_585_740_288, 12_451_840
    plan = fit.plan_fit(
        _tiny_request(),
        fit.ResourceSnapshot(total_bytes=total, free_bytes=free,
                             external_floor_bytes=EXTERNAL_FLOOR, now=NOW),
        [fit.Resident("Qwen2.5-VL-7B-Instruct-GGUF", vram_bytes=6_578_765_824,
                      host_mode="slot", protected=True, why="static residency")],
        fit.FitPolicy(ceiling_reserve_bytes=0))
    assert plan.action == "refuse"
    # The worker's verdict dict, as _execute_fit_plan builds it (subset).
    reason = {"state": "refused", "model_key": "test-save-tiny-random-llama3-smashed-pro",
              "reason": "won't fit on GPU: needs 18.0 MB, 11.9 MB free of 7.6 GB (...)",
              "needs_bytes": plan.need_bytes, "free_vram_bytes": free,
              "external_floor_bytes": EXTERNAL_FLOOR,
              "free_vram_device_bytes": free + EXTERNAL_FLOOR,
              "fit_budget_bytes": free, "ceiling_reserve_bytes": 0,
              "total_vram_bytes": total, "evicted": [], "evicted_freed_bytes": 0,
              "protected": [{"model_key": "Qwen2.5-VL-7B-Instruct-GGUF",
                             "vram_bytes": 6_578_765_824, "host_mode": "slot",
                             "why": "static residency"}],
              "fit_failure": plan.failure.as_dict()}
    exc = D.LoadRefusal(reason)
    out = LF.load_failure_of(exc)
    assert out["class"] == "vram_fit"
    assert out["fit_failure"]["kind"] == "vram_fit" and out["fit_failure"]["code"] == "wont_fit"
    assert out["fit_failure"]["need_bytes"] == 18_874_368
    assert out["fit_failure"]["budget_bytes"] == 12_451_840
    ref = out["refusal"]
    assert ref["free_vram_bytes"] == 12_451_840
    assert ref["free_vram_device_bytes"] == 12_451_840 + EXTERNAL_FLOOR
    assert ref["external_floor_bytes"] == EXTERNAL_FLOOR and ref["fit_budget_bytes"] == 12_451_840
    assert ref["protected_count"] == 1
    assert ref["protected"][0]["model_key"] == "Qwen2.5-VL-7B-Instruct-GGUF"
    assert out["message"].startswith("LoadRefusal: won't fit on GPU")
    # A refusal WITHOUT a typed verdict (slot-path prose) is unchanged: class only.
    plain = LF.load_failure_of(D.LoadRefusal({"reason": "won't fit on a slot: ...",
                                              "model_key": "m", "no_makeroom": True}))
    assert plain["class"] == "vram_fit" and "fit_failure" not in plain and "refusal" not in plain
    # Wrapped one level (the probe's RuntimeError) still resolves via the chain.
    try:
        try:
            raise exc
        except D.LoadRefusal as inner:
            raise RuntimeError("m: in-process load failed") from inner
    except RuntimeError as wrapped:
        chained = LF.load_failure_of(wrapped, classify=True)
    assert chained["class"] == "vram_fit" and chained["fit_failure"]["kind"] == "vram_fit"


# ---------------------------------------------------------------------------
# step 2 (F7): explicit weights-vs-KV split + the ctx-cap-before-evict proposal
# ---------------------------------------------------------------------------
def test_need_detail_carries_the_explicit_weights_vs_kv_split():
    """INVARIANT (step 2, F7): every plan's need_detail["need_split"] states
    weights_bytes / kv_bytes / kv_share_pct / ctx_pct so a 4B seat priced at
    max ctx (KV dwarfing the weights) is legible from the plan alone; a
    self-flex re-prices the split at the flexed ctx. Established: core
    isolation step 2 (2026-09-29)."""
    det = {"total": 8 * GIB, "weights": 4 * GIB, "kv": 4 * GIB, "ctx_pct": 100,
           "ctx_resolved": 262144, "ctx_max": 262144}
    plan = fit.plan_fit(_req(8 * GIB, detail=det), _snap(2 * GIB),
                        [_res("cold", 8 * GIB, last_call=100.0)], POLICY)
    split = plan.need_detail["need_split"]
    assert split["weights_bytes"] == 4 * GIB and split["kv_bytes"] == 4 * GIB
    assert split["kv_share_pct"] == 50.0 and split["ctx_pct"] == 100
    assert split["ctx_resolved"] == 262144 and split["ctx_max"] == 262144
    assert plan.weights_bytes == 4 * GIB and plan.kv_bytes == 4 * GIB
    assert fit.need_split(det, 8 * GIB, kv_bytes=1 * GIB, ctx_pct=25)["kv_share_pct"] == 12.5


def test_ctx_cap_knob_is_off_by_default_and_proposes_a_reduced_ctx_seat_when_on():
    """INVARIANT (step 2, F7): with FitPolicy.ctx_cap_on_evict_pct unset the plan
    is byte-identical to before (the eviction). With the knob set, a subject
    whose weights + KV@cap fit the free room gets a `partial` of kind
    `ctx-cap` (self_ctx_pct = cap, need re-priced, NO evictions, the spared
    residents counted in the note) BEFORE any eviction is planned — a
    plan-side proposal the executor may ignore. Established: step 2."""
    det = {"total": 8 * GIB, "weights": 4 * GIB, "kv": 4 * GIB, "ctx_pct": 100,
           "ctx_resolved": 262144, "ctx_max": 262144}
    residents = [_res("cold", 8 * GIB, last_call=100.0)]
    off = fit.plan_fit(_req(8 * GIB, detail=det), _snap(6 * GIB), residents, POLICY)
    assert off.action == "evict" and off.evicted_keys == ["cold"]
    assert POLICY.ctx_cap_on_evict_pct is None

    on_policy = dataclasses.replace(POLICY, ctx_cap_on_evict_pct=25)
    on = fit.plan_fit(_req(8 * GIB, detail=det), _snap(6 * GIB), residents, on_policy)
    assert on.action == "partial" and on.partial_kind == "ctx-cap"
    assert on.self_ctx_pct == 25
    assert on.need_bytes == 5 * GIB                       # 4 GiB weights + KV@25% = 1 GiB
    assert on.evictions == () and on.predicted_fits is True
    assert on.partial["admit"] is True and on.partial["kv_bytes"] == 1 * GIB
    assert on.partial["ctx_pct_target"] == 100 and on.partial["weights_bytes"] == 4 * GIB
    assert on.need_detail["need_split"]["kv_bytes"] == 1 * GIB
    assert on.need_detail["need_split"]["ctx_pct"] == 25
    assert "1 evictable resident(s) spared" in on.note
    # The cap never widens a ctx: a subject already at/below the cap is unaffected.
    low = fit.plan_fit(_req(8 * GIB, detail=dict(det, ctx_pct=20)), _snap(6 * GIB),
                       residents, on_policy)
    assert low.action == "evict"
    # When even the capped seat does not fit, the proposal is skipped and the
    # eviction stands.
    tight = fit.plan_fit(_req(8 * GIB, detail=det), _snap(3 * GIB), residents, on_policy)
    assert tight.action == "evict" and tight.evicted_keys == ["cold"]


# ---------------------------------------------------------------------------
# Resident identity is the canonical key (operator rule, board 2026-09-29)
# ---------------------------------------------------------------------------
def test_resident_identity_is_the_canonical_key_and_same_file_residents_stay_distinct():
    """INVARIANT: ``Resident.identity`` is the key with only ``-GGUF`` stripped.
    ``Qwen3.8-9B-Distill-GGUF`` and ``Qwen3.8-9B-GGUF`` are DISTINCT residents
    (same file on ae's drive notwithstanding): a load of the plain key is a new
    admission that may evict the Distill seat; it is never "already resident".
    ``Qwen3.8-9B`` and ``Qwen3.8-9B-GGUF`` are ONE identity: the subject is never
    a victim of its own admission.
    Established: eviction test S3 / F7 + operator rule 2026-09-29."""
    distill = _res("Qwen3.8-9B-Distill-GGUF", 10 * GIB, last_call=100.0)
    plain = _res("Qwen3.8-9B-GGUF", 10 * GIB, last_call=200.0)
    assert distill.identity == "Qwen3.8-9B-Distill"
    assert plain.identity == "Qwen3.8-9B" == _res("Qwen3.8-9B", 1).identity
    assert fit.key_equivalent(plain.model_key, "Qwen3.8-9B")
    assert not fit.key_equivalent(plain.model_key, distill.model_key)

    # S3 shape: resident Distill, request the plain key -> a NEW load that
    # evicts the distinct Distill seat (not served by it, not credited by it).
    req = dataclasses.replace(_req(11 * GIB), model_key="Qwen3.8-9B-GGUF")
    plan = fit.plan_fit(req, _snap(2 * GIB), [distill], POLICY)
    assert plan.action == "evict" and plan.evicted_keys == ["Qwen3.8-9B-Distill-GGUF"]
    assert plan.model_key == "Qwen3.8-9B-GGUF"

    # Same identity under another spelling: the subject itself, never evicted.
    own = _res("Qwen3.8-9B", 10 * GIB, last_call=100.0)
    plan = fit.plan_fit(req, _snap(2 * GIB), [own], POLICY)
    assert plan.action == "refuse" and plan.evicted_keys == []
    assert any("subject identity" in r for r in plan.reasons)

    # Two same-file siblings are two residents in one plan: both may be named.
    heretic = _res("Qwen3.8-9B-heretic-uncensored-GGUF", 10 * GIB, last_call=150.0)
    plan = fit.plan_fit(dataclasses.replace(req, need_bytes=20 * GIB,
                                            need_detail={"total": 20 * GIB, "weights": 20 * GIB,
                                                         "kv": 0, "ctx_pct": None}),
                        _snap(1 * GIB), [distill, heretic], POLICY)
    assert sorted(plan.evicted_keys) == ["Qwen3.8-9B-Distill-GGUF",
                                         "Qwen3.8-9B-heretic-uncensored-GGUF"]


# ---------------------------------------------------------------------------
# 2026-09-29: the per-GPU 1 GiB hold-back is gone; occupancy is MEASURED
# ---------------------------------------------------------------------------
def test_vram_reserve_knob_defaults_to_zero_and_still_honours_an_explicit_value(monkeypatch):
    """INVARIANT: with HUGPY_VRAM_RESERVE_GIB unset, spill.vram_reserve_bytes()
    is 0 — the budgetable free figure IS the device free read; no fixed slice
    is held back "for out-of-band GPU consumers" (their bytes are already out
    of the device read, and the context need is priced explicitly). An
    explicit value is still honoured verbatim, so the knob remains an operator
    lever for a known co-tenant that grows after the load. GUARD: the
    whole-fit un-stack in autofit_gpu_layers un-stacks NOTHING by default.
    Established: operator 2026-09-29 (the ae refusal "21.3 GB free of 23.6 GB
    (0 B ceiling reserve + 1.0 GB already held back ...)" with 0 B measured
    unattributed on the card)."""
    spill = importlib.import_module("hugpy_engine.spill")
    monkeypatch.delenv("HUGPY_VRAM_RESERVE_GIB", raising=False)
    assert spill.vram_reserve_bytes() == 0
    monkeypatch.setenv("HUGPY_VRAM_RESERVE_GIB", "1.5")
    assert spill.vram_reserve_bytes() == int(1.5 * GIB)
    monkeypatch.setenv("HUGPY_VRAM_RESERVE_GIB", "0")
    assert spill.vram_reserve_bytes() == 0


def test_snapshot_carries_measured_occupancy_as_reporting_only():
    """INVARIANT: ResourceSnapshot carries the MEASURED occupancy split
    (attributed_vram_bytes / foreign_vram_bytes) beside the budgetable free
    figure, defaulting to None (unmeasured, never 0), and plan_fit prices
    against free_bytes ALONE — the measured foreign figure is occupancy that is
    already out of the device read and is never subtracted a second time.
    Established: 2026-09-29 (double-count fix)."""
    need = 4 * GIB
    plain = _snap(need + 512 * (1 << 20))
    assert plain.attributed_vram_bytes is None and plain.foreign_vram_bytes is None
    assert plain.external_floor_bytes == 0
    measured = _snap(need + 512 * (1 << 20),
                     attributed_vram_bytes=3 * GIB, foreign_vram_bytes=2 * GIB)
    policy = fit.FitPolicy(ceiling_reserve_bytes=512 * (1 << 20))
    a = fit.plan_fit(_req(need), plain, [], policy)
    b = fit.plan_fit(_req(need), measured, [], policy)
    assert a.action == "proceed" and b.action == "proceed"
    assert a.free_bytes == b.free_bytes == need + 512 * (1 << 20)
    assert b.fits_now is True                    # 2 GiB "foreign" changed nothing
