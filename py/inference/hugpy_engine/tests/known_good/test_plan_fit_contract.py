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
