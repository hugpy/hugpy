"""``plan_fit`` — the VRAM evict-to-fit decision as a PURE function.

CORE ISOLATION, step 1 (notes/CORE-ISOLATION-DESIGN.md, Part A.3). This is the
decision half of the worker's ``_vram_evict_to_fit``; the worker gathers a
:class:`VramSnapshot` + :class:`FitPolicy` + :class:`Resident` rows + a
:class:`FitRequest` ONCE, calls :func:`plan_fit`, then EXECUTES the returned
:class:`FitPlan` (the only place eviction / CUDA / telemetry happen).

Inside this module there is NO I/O, NO env read, NO clock, NO torch/nvml, NO
logging that matters. Same inputs -> byte-identical plan, always. It composes
the pure pieces that already existed, in the orchestrator's order:

    fit test -> flex.plan_flex (self-ctx compression) -> eviction.evict_plan
    (priority bands, least reaping) -> flex.plan_explicit_offload /
    plan_partial_offload (+ spill.moe_dense_first_plan) -> honest refuse

Behaviour is the orchestrator's, stage for stage — this is a refactor. The
places where the known step-2 fixes belong are marked ``STEP 2 SLOT``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from hugpy_engine import eviction as _ev
from hugpy_engine.fit import flex as _flex
from hugpy_engine.fit.types import (
    Eviction, FitFailure, FitPlan, FitPolicy, FitRequest, MoeSplit, Resident,
    ResourceSnapshot, key_equivalent,
)
from hugpy_engine.spill import moe_dense_first_plan as _moe_dense_first_plan


# ── the shared eviction ORDER (was agent._shared_evict_order) ────────────────
def evict_order(rows: Iterable[_ev.EvictUnit], need: Optional[int],
                priority: Optional[Mapping[str, int]] = None, *,
                now: float, least_reaping: bool = _ev.DEFAULT_LEAST_REAPING,
                ) -> "list[str]":
    """Model keys to evict, in order — via THE shared function
    (``eviction.evict_plan``), with the operator's flex priority as the OUTER
    key. PURE: ``now`` and ``least_reaping`` are passed in.

    ``need`` None (unmeasurable free VRAM) -> DEGRADE-NOT-GUESS: the full pool
    in shared-key order; the caller's incremental fit loop stops it.

    Priority grouping keeps least-reaping honest under an override: the drop
    pass runs WITHIN a priority band, so a high-priority resident is never
    spared by a low-priority one covering the need — bands are walked in order
    and each is planned against the need that REMAINS."""
    rows = list(rows)
    pri = dict(priority or {})
    if need is None:
        ordered = sorted(rows, key=lambda r: (pri.get(r.model_key, 0),
                                              _ev.sort_key(r, _ev.VRAM, now)))
        return [r.model_key for r in ordered]
    out: list[str] = []
    remaining = int(need)
    for band in sorted({pri.get(r.model_key, 0) for r in rows}):
        if remaining <= 0:
            break
        members = [r for r in rows if pri.get(r.model_key, 0) == band]
        plan = _ev.evict_plan(_ev.VRAM, remaining, members, now=now,
                              least_reaping=least_reaping)
        out.extend(plan.victims)
        remaining -= plan.freed
    return out


def _unit(r: Resident) -> _ev.EvictUnit:
    """A Resident as the shared eviction function describes it."""
    return _ev.EvictUnit(
        model_key=r.model_key,
        bytes=(int(r.vram_bytes or 0) or None),
        pref=r.pref or _ev.VRAM,
        last_call=r.last_call, calls=int(r.calls or 0),
        mid_generation=bool(r.mid_generation),
        resident_since=r.resident_since)


def _eviction(r: Resident, rank: int) -> Eviction:
    return Eviction(model_key=r.model_key, vram_bytes=r.vram_bytes,
                    host_mode=r.host_mode, rank=rank)


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def need_split(det: Mapping[str, Any], need: Optional[int],
               kv_bytes: Optional[int] = None,
               ctx_pct: Optional[int] = None) -> dict:
    """The explicit weights-vs-KV split of a priced need (step 2, F7). ``kv``
    and ``ctx_pct`` default to the detail's own; a flex / ctx-cap passes the
    re-priced figures. ``kv_share_pct`` says how much of the need is context
    — at a model's max ctx it dwarfs the weights (4B -> 21.2 GB seat)."""
    w = _int_or_none(det.get("weights"))
    kv = _int_or_none(det.get("kv")) if kv_bytes is None else int(kv_bytes)
    pct = det.get("ctx_pct") if ctx_pct is None else ctx_pct
    out = {"weights_bytes": w, "kv_bytes": kv, "ctx_pct": pct,
           "ctx_resolved": det.get("ctx_resolved"), "ctx_max": det.get("ctx_max")}
    n = _int_or_none(need)
    if n and kv is not None:
        out["kv_share_pct"] = round(100.0 * kv / n, 1)
    return out


def _split_of(n_cpu_moe: Any, gpu_bytes: Any, cpu_bytes: Any, basis: str,
              contract_n: Optional[int]) -> MoeSplit:
    return MoeSplit(n_cpu_moe=int(n_cpu_moe), gpu_bytes=_int_or_none(gpu_bytes),
                    cpu_bytes=_int_or_none(cpu_bytes), basis=basis,
                    contract_n_cpu_moe=contract_n)


def _split_failure(split: MoeSplit, request: FitRequest,
                   policy: FitPolicy) -> Optional[FitFailure]:
    """PLAN-TIME predicates for a MoE split (diagnosis (b).3 / (b).5), priced
    in ONE basis: the RAM the split puts in host memory AT THE PLAN'S N is
    checked against the contract's RAM budget (the slot preflight's
    ``cpu_mem_gib`` check, moved to plan time so need and budget are never
    priced at different N), and a split that no available engine can express
    is refused here rather than falling through to an in-process runner."""
    if request.split_expressible is False:
        return FitFailure(
            kind="split_not_expressible", code="no_native_engine",
            reason=(f"MoE split --n-cpu-moe {split.n_cpu_moe} cannot be expressed "
                    "by any available engine (no native slot) — refusing at plan "
                    "time instead of falling back in-process"),
            plan_n_cpu_moe=split.n_cpu_moe,
            contract_n_cpu_moe=split.contract_n_cpu_moe,
            permanent=False, state_dependent=True)
    budget = policy.ram_target_bytes
    cpu = split.cpu_bytes
    if budget is not None and cpu is not None and split.n_cpu_moe and int(cpu) > int(budget):
        return FitFailure(
            kind="ram_budget", code="over_budget",
            reason=(f"MoE split --n-cpu-moe {split.n_cpu_moe} puts {int(cpu)} B "
                    f"({int(cpu) / 2 ** 30:.1f} GiB) of expert tensors in host RAM, "
                    f"over this model's RAM budget of {int(budget)} B "
                    f"({int(budget) / 2 ** 30:.1f} GiB)"
                    + (f" (budget derived at --n-cpu-moe {split.contract_n_cpu_moe})"
                       if split.contract_n_cpu_moe is not None
                       and split.contract_n_cpu_moe != split.n_cpu_moe else "")),
            need_bytes=int(cpu), budget_bytes=int(budget),
            plan_n_cpu_moe=split.n_cpu_moe,
            contract_n_cpu_moe=split.contract_n_cpu_moe,
            permanent=False, state_dependent=True)
    return None


# ── THE decision ─────────────────────────────────────────────────────────────
def plan_fit(request: FitRequest, snapshot: ResourceSnapshot,
             residents: Iterable[Resident], policy: FitPolicy) -> FitPlan:
    """Decide how ``request`` lands on the card described by ``snapshot`` with
    ``residents`` on it, under ``policy``. PURE — see the module docstring.

    Fails OPEN (``proceed`` with a ``note``) whenever the card or the need is
    unmeasurable, exactly as the orchestrator did: an unmeasurable load
    proceeds, never blocked because we couldn't measure."""
    residents = tuple(residents)
    reasons: list[str] = []
    mk = request.model_key

    def _proceed(note: str, **kw) -> FitPlan:
        return FitPlan(action="proceed", model_key=mk, note=note,
                       reasons=tuple(reasons), **kw)

    # ── stage 0: measurability + pricing ────────────────────────────────────
    total = _int_or_none(snapshot.total_bytes)
    if not total:
        return _proceed("no GPU / unmeasurable — gate is a no-op")
    det: dict = dict(request.need_detail or {})
    need = _int_or_none(request.need_bytes)
    if not need:
        return _proceed("unknown weight size — fail open", total_bytes=total,
                        need_detail=det)
    # PLACEMENT-INTENT RE-PRICE: admission prices what will ACTUALLY land on
    # the card (a RAM-only designation puts 0 B there; max-ram only the
    # remainder over the CPU budget). The caller derived the figure with the
    # loader's own function so gate and loader cannot disagree.
    planned = _int_or_none(request.planned_gpu_bytes)
    if planned is not None and planned < need:
        if planned <= 0:
            return _proceed("placement intent puts 0 B on the GPU "
                            "(CPU/RAM-only) — VRAM admission is a no-op",
                            total_bytes=total, need_detail=det)
        need = int(planned)
        det["intent_gpu_remainder"] = need
        reasons.append(f"placement intent re-priced the GPU need to {need} B")
    reserve = int(policy.ceiling_reserve_bytes or 0)

    # ── MoE re-target: typed bytes over the opaque byte-bag ─────────────────
    # When a MoE split governs this model the plan that ACTUALLY loads is the
    # split, so admission prices ITS GPU need (non-expert + KV). RAM-guarded:
    # the experts must fit the BOX (total RAM — mmap doctrine; momentary free
    # is the fallback). Only when `need` still equals the authoritative total
    # (an explicit / re-priced need stands as given).
    moe_commit: Optional[dict] = None
    if need == det.get("total"):
        ms = det.get("moe_split")
        if ms:
            exp_bytes = int(ms.get("cpu_bytes") or 0)
            ecb = policy.empty_card_budget_bytes
            impossible_full = ecb is not None and int(need) > int(ecb)
            ram_ceiling = snapshot.ram_total_bytes or snapshot.ram_free_bytes
            if ram_ceiling is not None and exp_bytes > ram_ceiling * policy.ram_safety_frac:
                reasons.append(
                    f"MoE split skipped: expert tensors (~{exp_bytes} B) exceed "
                    f"host RAM (~{int(ram_ceiling)} B) — keeping full-need admission")
            elif ms.get("gpu_total"):
                need = int(ms["gpu_total"])
                moe_commit = dict(ms)
                reasons.append(
                    "MoE re-target: "
                    + ("full weights can never fit this card" if impossible_full
                       else "the expert split is the default placement")
                    + f" (GPU need {need} B, experts ~{exp_bytes} B to CPU)")

    contract_n = _int_or_none(request.contract_n_cpu_moe)
    ram_budget = _int_or_none(policy.ram_target_bytes)

    def _commit_split() -> Optional[MoeSplit]:
        if moe_commit is None:
            return None
        return _split_of(moe_commit.get("n_cpu_moe") or 0, moe_commit.get("gpu_total"),
                         moe_commit.get("cpu_bytes"), "contract", contract_n)

    def _refuse_split(split: MoeSplit, failure: FitFailure, **kw) -> FitPlan:
        reasons.append(failure.reason)
        return FitPlan(action="refuse", failure=failure, refuse_reason=failure.reason,
                       split=split, n_cpu_moe=split.n_cpu_moe,
                       ram_need_bytes=split.cpu_bytes,
                       reasons=tuple(reasons), note="refuse", **kw)

    # ── SUBJECT CREDIT: the subject's own footprint is headroom for itself ──
    subject_held = int(request.subject_held_bytes or 0)
    if subject_held:
        reasons.append(f"subject already resident holding {subject_held} B — "
                       "credited against its own need")
    free = _int_or_none(snapshot.free_bytes)
    if free is None:
        return _proceed("can't read free VRAM — fail open", total_bytes=total,
                        need_bytes=need, need_detail=det, moe_commit=moe_commit,
                        subject_held_bytes=subject_held,
                        ceiling_reserve_bytes=reserve)
    free_eff = int(free) + subject_held

    def _fits(n: int, extra_free: int = 0) -> bool:
        return (free_eff + extra_free - n) >= reserve

    det["need_split"] = need_split(det, need)
    common = dict(model_key=mk, need_detail=det, total_bytes=total,
                  free_bytes=free, free_effective_bytes=free_eff,
                  subject_held_bytes=subject_held, ceiling_reserve_bytes=reserve,
                  weights_bytes=_int_or_none(det.get("weights")),
                  kv_bytes=_int_or_none(det.get("kv")), ram_budget_bytes=ram_budget)

    if _fits(need):
        split = _commit_split()
        if split is not None:
            bad = _split_failure(split, request, policy)
            if bad is not None:
                return _refuse_split(split, bad, need_bytes=need, fits_now=True,
                                     moe_commit=moe_commit, **common)
        # Fits under the ceiling: nothing MUST be evicted. The eviction-aware
        # autofit size-up (a bonus that buys a better layer count BY evicting)
        # stays the executor's — it is skipped for a MoE commit and for a
        # polite load ("never evict" outranks "seat it better").
        return FitPlan(action="proceed", fits_now=True, need_bytes=need,
                       moe_commit=moe_commit, split=split,
                       n_cpu_moe=(split.n_cpu_moe if split else None),
                       ram_need_bytes=(split.cpu_bytes if split else None),
                       size_up_eligible=(moe_commit is None and not request.polite),
                       reasons=tuple(reasons),
                       note=("polite load (no_evict): admitted into free room; "
                             "the eviction-aware size-up was skipped"
                             if request.polite and moe_commit is None else ""),
                       **common)

    # Over the ceiling.
    # SUBJECT IDENTITY (operator rule 2026-09-29): a resident whose canonical
    # identity equals the subject's (``X`` vs ``X-GGUF``: one key) IS the
    # subject — it is never a victim of its own admission. A resident that
    # differs by any other token (``X-Distill-GGUF``) is a DISTINCT resident
    # even when it mmaps the same file: it stays a candidate and the subject
    # is admitted as a new load, never served from that seat.
    same_seat = [r for r in residents if key_equivalent(r.model_key, mk)]
    if same_seat:
        reasons.append("subject identity: %d resident(s) are the subject itself "
                       "(%s) — excluded from eviction"
                       % (len(same_seat), ", ".join(r.model_key for r in same_seat)))
        residents = tuple(r for r in residents if not key_equivalent(r.model_key, mk))
    candidates = [r for r in residents if not r.protected]
    protected = [r for r in residents if r.protected]

    # ── stage 1: tolerance-band FLEX before evict ───────────────────────────
    # ctx is the CHEAPEST flex: compress the SUBJECT's own ctx toward its band
    # floor (lowers `need`); a higher-priority subject's neighbour compression
    # manifests as the priority-ordered eviction below. Only unprotected rows
    # are offered as flex-eligible neighbours — protection is absolute.
    deficit = reserve - (free_eff - need)
    subj_weights = det.get("weights")
    if moe_commit is not None:
        subj_weights = max(0, int(moe_commit.get("gpu_total") or 0)
                           - int(det.get("kv") or 0))
    subject = {"weights_bytes": subj_weights, "kv_bytes": det.get("kv"),
               "ctx_pct": det.get("ctx_pct"),
               "ctx_deviation_pct": request.ctx_deviation_pct,
               "priority": request.priority}
    resident_rows = [{
        "model_key": r.model_key, "kv_bytes": int(r.kv_bytes or 0),
        "ctx_pct": r.ctx_pct, "ctx_deviation_pct": r.ctx_deviation_pct,
        "vram_bytes": int(r.vram_bytes or 0), "protected": False,
        "pinned": bool(r.pinned), "alloc": {"priority": r.priority}}
        for r in candidates]
    fplan = _flex.plan_flex(subject, resident_rows, deficit)
    self_ctx_pct: Optional[int] = None
    if fplan.self_ctx_pct is not None and det.get("kv"):
        # Commit the subject to its compressed ctx (the executor records the
        # floor so the SERVED -c and the KV reserved agree) and re-price `need`
        # at that floor — every later fit test uses the flexed figure.
        self_ctx_pct = int(fplan.self_ctx_pct)
        new_kv = _flex.kv_at_ctx_pct(det.get("kv"), det.get("ctx_pct"),
                                     fplan.self_ctx_pct)
        need = int(subj_weights or 0) + int(new_kv or 0)
        if moe_commit is not None:
            moe_commit["gpu_total"] = int(need)
        det["need_split"] = need_split(det, need, kv_bytes=int(new_kv or 0),
                                       ctx_pct=self_ctx_pct)
        reasons.append(f"self-flex ctx -> {self_ctx_pct}%: need re-priced to {need} B")
    common["fits_now"] = False
    split = _commit_split()
    if fplan.action == "flex" and _fits(need):
        bad = _split_failure(split, request, policy) if split is not None else None
        if bad is not None:
            return _refuse_split(split, bad, need_bytes=need, self_ctx_pct=self_ctx_pct,
                                 flex=fplan.as_dict(), flex_note=fplan.note,
                                 moe_commit=moe_commit, **common)
        return FitPlan(action="flex", need_bytes=need, self_ctx_pct=self_ctx_pct,
                       flex=fplan.as_dict(), flex_note=fplan.note,
                       moe_commit=moe_commit, split=split,
                       n_cpu_moe=(split.n_cpu_moe if split else None),
                       ram_need_bytes=(split.cpu_bytes if split else None),
                       reasons=tuple(reasons), note=f"flex: {fplan.note}", **common)
    flex_note = fplan.note

    # ── stage 1.5 (step 2, F7): ctx-cap PROPOSAL before any eviction ────────
    # DEFAULT OFF (policy.ctx_cap_on_evict_pct None). When on: the KV term is
    # the part of the need that can shrink without touching anyone else, so
    # before an eviction is planned the subject is offered a seat at
    # min(cap, its ctx_pct). If weights + KV@cap fits the free room the plan
    # is a `partial` of kind `ctx-cap` (self_ctx_pct = the cap, no evictions).
    # A PROPOSAL: the executor may ignore it and evict instead.
    cap = _int_or_none(policy.ctx_cap_on_evict_pct)
    kv_target = _int_or_none(det.get("kv"))
    pct_target = _int_or_none(det.get("ctx_pct"))
    if cap and kv_target and pct_target and 0 < cap < (self_ctx_pct or pct_target):
        kv_cap = _flex.kv_at_ctx_pct(kv_target, pct_target, cap)
        need_cap = int(subj_weights or 0) + int(kv_cap or 0)
        if _fits(need_cap):
            det_cap = dict(det)
            det_cap["need_split"] = need_split(det, need_cap, kv_bytes=int(kv_cap or 0),
                                               ctx_pct=cap)
            common_cap = dict(common, need_detail=det_cap)
            if moe_commit is not None:
                moe_commit["gpu_total"] = int(need_cap)
            note = (f"ctx cap: {self_ctx_pct or pct_target}% -> {cap}% ctx re-prices "
                    f"KV {kv_target} B -> {kv_cap} B; need {need_cap} B fits without "
                    f"eviction (proposal; {len(candidates)} evictable resident(s) spared)")
            reasons.append(note)
            return FitPlan(action="partial", partial_kind="ctx-cap",
                           need_bytes=need_cap, self_ctx_pct=int(cap),
                           flex=fplan.as_dict(), flex_note=flex_note,
                           evictions=(), eviction_need_bytes=0,
                           predicted_freed_bytes=0, predicted_fits=True,
                           partial={"kind": "ctx-cap", "admit": True,
                                    "ctx_pct": int(cap), "ctx_pct_target": pct_target,
                                    "kv_bytes": int(kv_cap or 0),
                                    "kv_bytes_target": kv_target,
                                    "weights_bytes": int(subj_weights or 0),
                                    "vram_need_bytes": need_cap, "note": note},
                           moe_commit=moe_commit, split=split,
                           n_cpu_moe=(split.n_cpu_moe if split else None),
                           ram_need_bytes=(split.cpu_bytes if split else None),
                           reasons=tuple(reasons), note=note, **common_cap)

    # ── stage 2: EVICT — the SHARED function, priority bands outermost ─────
    # A POLITE load never reaches the walk: the candidates are SPARED and the
    # plan falls through to the offload / refusal tail sized from free VRAM.
    polite_spared: tuple = ()
    if request.polite:
        polite_spared = tuple(_eviction(r, i) for i, r in enumerate(candidates))
        candidates = []
        reasons.append(f"polite load (no_evict): {len(polite_spared)} evictable "
                       "resident(s) spared")
    ev_need = max(0, reserve - (free_eff - need))    # only the REMAINING deficit
    order = evict_order([_unit(r) for r in candidates], ev_need,
                        {r.model_key: int(r.priority or 0) for r in candidates},
                        now=snapshot.now, least_reaping=policy.least_reaping)
    by_mk = {r.model_key: r for r in candidates}
    evictions = tuple(_eviction(by_mk[k], i) for i, k in enumerate(order) if k in by_mk)
    predicted_freed = sum(int(e.vram_bytes or 0) for e in evictions)
    predicted_fits = _fits(need, predicted_freed)
    evict_common = dict(need_bytes=need, self_ctx_pct=self_ctx_pct,
                        flex=fplan.as_dict(), flex_note=flex_note,
                        evictions=evictions, eviction_need_bytes=ev_need,
                        predicted_freed_bytes=predicted_freed,
                        predicted_fits=predicted_fits,
                        polite_spared=polite_spared, moe_commit=moe_commit,
                        comfy_reclaim_eligible=not request.polite)
    if predicted_fits:
        bad = _split_failure(split, request, policy) if split is not None else None
        if bad is not None:
            return _refuse_split(split, bad, **evict_common, **common)
        reasons.append(f"eviction of {len(evictions)} resident(s) frees ~{predicted_freed} B")
        return FitPlan(action="evict", reasons=tuple(reasons), split=split,
                       n_cpu_moe=(split.n_cpu_moe if split else None),
                       ram_need_bytes=(split.cpu_bytes if split else None),
                       note=f"evict {len(evictions)} resident(s)", **evict_common, **common)

    # ── stage 2.5: honest GGUF PARTIAL offload — autofit's hybrid contract ──
    # Full offload still short after flex + (all) evictions. Priced from the
    # PREDICTED post-eviction free figure (raw + subject credit, less the
    # ceiling reserve). STEP 2 SLOT ("stable budget"): the executor today
    # re-plans this tail from a FRESH free read after executing the evictions;
    # making this predicted budget authoritative is the step-2 change.
    fv = free + predicted_freed
    fv_eff = fv + subject_held
    partial = None
    partial_kind: Optional[str] = None
    budget: Optional[int] = None
    mode = policy.alloc_mode
    total_layers = _int_or_none(request.total_layers)
    if total_layers:
        weights = int(det.get("weights") or 0)
        kv_eff = max(0, int(need) - weights)       # honors any committed ctx flex
        budget = max(0, fv_eff - reserve)
        # Cap by the model's explicit VRAM band CEILING when a gpu_mem_gib
        # budget is set — stretchable to the band ceiling under its own need.
        if policy.gpu_target_bytes is not None:
            cap = int(_flex.band_ceiling(policy.gpu_target_bytes,
                                         request.vram_deviation_pct, total))
            budget = min(budget, cap)
        intent, requested = request.ngl_intent, request.ngl_requested
        # MoE FIRST: every partial-offload entry on a detected-MoE GGUF defers
        # to the dense-backbone-first split BEFORE any dense layer math. An
        # operator-stated layer count and a cpu intent are obeyed verbatim; the
        # backbone must fit the budget; the experts must fit the BOX.
        if requested is None and (intent or "auto") != "cpu" and request.moe_detail:
            det_moe = dict(request.moe_detail)
            mbudget = request.moe_auto_gpu_budget_bytes or budget
            try:
                mplan = _moe_dense_first_plan(det_moe, mbudget)
            except Exception:  # noqa: BLE001 — fall through to dense math
                mplan = None
            if mplan and int(mplan.get("cpu_bytes") or 0) and mplan.get("dense_fits"):
                exp_b = int(det_moe.get("expert_bytes") or 0)
                ram_ceiling = snapshot.ram_total_bytes or snapshot.ram_free_bytes
                if not ram_ceiling or exp_b <= ram_ceiling * policy.ram_safety_frac:
                    msplit = _split_of(mplan["n_cpu_moe"], mplan.get("gpu_bytes"),
                                       mplan.get("cpu_bytes"), "moe-first", contract_n)
                    bad = _split_failure(msplit, request, policy)
                    if bad is not None:
                        return _refuse_split(msplit, bad, moe_plan=dict(mplan),
                                             budget_bytes=mbudget, **evict_common, **common)
                    reasons.append(f"MoE-first partial admit: --n-cpu-moe {mplan['n_cpu_moe']}")
                    return FitPlan(action="partial", partial_kind="moe-first",
                                   n_gpu_layers=-1, n_cpu_moe=int(mplan["n_cpu_moe"]),
                                   split=msplit, ram_need_bytes=msplit.cpu_bytes,
                                   moe_plan=dict(mplan), budget_bytes=mbudget,
                                   reasons=tuple(reasons),
                                   note=(f"MoE dense-first split (--n-cpu-moe "
                                         f"{mplan['n_cpu_moe']}): all layers on GPU, "
                                         f"~{int(mplan.get('cpu_bytes') or 0)} B "
                                         f"expert tensors on CPU"),
                                   **evict_common, **common)
        # k37: max-ram / explicit route to the leniency-band engine; gpu-only /
        # ram-only / max-gpu keep the plan_partial_offload path.
        if mode in ("max-ram", "explicit"):
            partial = _flex.plan_explicit_offload(
                weights_bytes=weights, kv_bytes=kv_eff,
                total_layers=total_layers, vram_budget_bytes=budget,
                ram_free_bytes=snapshot.ram_free_bytes, mode=mode,
                priority_device=("ram" if mode == "max-ram" else policy.priority_device),
                gpu_target_bytes=policy.gpu_target_bytes,
                ram_target_bytes=policy.ram_target_bytes,
                leniency_pct=(100.0 if mode == "max-ram" else (policy.leniency_pct or 0.0)),
                ram_safety_frac=policy.ram_safety_frac)
        else:
            partial = _flex.plan_partial_offload(
                weights_bytes=weights, kv_bytes=kv_eff, total_layers=total_layers,
                vram_budget_bytes=budget, ram_free_bytes=snapshot.ram_free_bytes,
                intent=intent, requested_layers=requested,
                min_offload_frac=policy.min_offload_frac,
                ram_safety_frac=policy.ram_safety_frac)

    if partial is not None and partial.admit:
        # A MODE-ENGINE admit on a detected-MoE GGUF rides the dense-first
        # split, not a raw layer count (a stated positive n_gpu_layers would
        # disable the split). Operator-stated counts never reach this branch.
        if mode in ("max-ram", "explicit") and partial.n_gpu_layers > 0 and request.moe_detail:
            det_moe = dict(request.moe_detail)
            mbudget = request.moe_auto_gpu_budget_bytes
            if not mbudget:
                mbudget = int(partial.vram_budget_bytes or 0)
            try:
                mplan = _moe_dense_first_plan(det_moe, mbudget)
            except Exception:  # noqa: BLE001 — fall back to the layer count
                mplan = None
            if mplan and int(mplan.get("cpu_bytes") or 0):
                msplit = _split_of(mplan["n_cpu_moe"], mplan.get("gpu_bytes"),
                                   mplan.get("cpu_bytes"), "mode-moe", contract_n)
                bad = _split_failure(msplit, request, policy)
                if bad is not None:
                    return _refuse_split(msplit, bad, moe_plan=dict(mplan),
                                         partial=partial.as_dict(), budget_bytes=mbudget,
                                         **evict_common, **common)
                reasons.append(f"{mode} admit -> MoE dense-first split "
                               f"(--n-cpu-moe {mplan['n_cpu_moe']})")
                return FitPlan(action="partial", partial_kind="mode-moe",
                               n_gpu_layers=-1, n_cpu_moe=int(mplan["n_cpu_moe"]),
                               split=msplit, ram_need_bytes=msplit.cpu_bytes,
                               moe_plan=dict(mplan), partial=partial.as_dict(),
                               budget_bytes=mbudget, reasons=tuple(reasons),
                               note=(f"{mode} MoE split (--n-cpu-moe "
                                     f"{mplan['n_cpu_moe']}): dense backbone first, "
                                     f"~{int(mplan.get('cpu_bytes') or 0)} B "
                                     f"expert tensors to CPU"),
                               **evict_common, **common)
        reasons.append(f"partial offload: {partial.n_gpu_layers}/{partial.total_layers} "
                       f"layers on GPU ({partial.gpu_pct}%)")
        return FitPlan(action="partial", partial_kind="dense",
                       n_gpu_layers=partial.n_gpu_layers, partial=partial.as_dict(),
                       budget_bytes=budget, reasons=tuple(reasons),
                       note=f"partial GPU offload: {partial.note}",
                       **evict_common, **common)

    # ── honest REFUSE (never admit-then-OOM) ────────────────────────────────
    # STEP 2 SLOT ("offload fallback"): a feasible max-ram / partial placement
    # must never reach here as a refusal; that preference rule lands above.
    why = (f"won't fit on GPU: needs {need} B, {fv} B free of {total} B"
           + (f" (+{subject_held} B held by the subject, credited -> {fv_eff} B)"
              if subject_held else "")
           + f", {reserve} B ceiling reserve"
           + (f" + {int(snapshot.external_floor_bytes or 0)} B external floor"
              if snapshot.external_floor_bytes else "")
           + f"; {len(evictions)} eviction(s) planned freeing ~{predicted_freed} B"
           + f"; {len(protected)} protected resident(s)"
           + (f"; {len(polite_spared)} spared by the polite load" if polite_spared else "")
           + ("; partial offload not admissible: "
              + str(partial.reject_reason or "rejected")
              if partial is not None else
              ("; no partial offload possible (no GGUF geometry)"
               if not total_layers else "")))
    reasons.append(why)
    failure = FitFailure(kind="vram_fit", code="wont_fit", reason=why,
                         need_bytes=int(need), budget_bytes=max(0, fv_eff - reserve),
                         plan_n_cpu_moe=(split.n_cpu_moe if split else None),
                         contract_n_cpu_moe=contract_n,
                         permanent=False, state_dependent=True)
    return FitPlan(action="refuse", partial=(partial.as_dict() if partial is not None else None),
                   partial_kind=None, budget_bytes=budget, refuse_reason=why,
                   failure=failure, split=split,
                   n_cpu_moe=(split.n_cpu_moe if split else None),
                   ram_need_bytes=(split.cpu_bytes if split else None),
                   reasons=tuple(reasons), note="refuse", **evict_common, **common)


__all__ = ["plan_fit", "evict_order", "need_split"]
