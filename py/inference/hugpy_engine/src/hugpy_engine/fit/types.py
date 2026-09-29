"""Frozen inputs and output of :func:`hugpy_engine.fit.plan_fit`.

CORE ISOLATION, step 1 (notes/CORE-ISOLATION-DESIGN.md Part A;
notes/coder-next-loaded-idle-diagnosis-2026-09-29.md section (b)). The VRAM
evict-to-fit decision used to READ its facts live, mid-decision (free VRAM,
free RAM, the clock, env levers), so identical logical state produced
different verdicts and nothing could be pinned by a test. Everything the
planner consults now arrives in these records, captured ONCE by the caller:

  * :class:`ResourceSnapshot` — the device(s) and the host as measured at
                                capture time (``VramSnapshot`` is its alias).
  * :class:`FitPolicy`        — the admission knobs the caller resolved from
                                env / settings (reserve, alloc mode, leniency,
                                the contract's VRAM/RAM targets).
  * :class:`Resident`         — one measured GPU resident, with the protection
                                verdict, MEASURED materialisation, the flex
                                inputs and the eviction-ledger inputs attached.
  * :class:`FitRequest`       — the incoming load: its priced need (weights +
                                KV split), the subject credit, geometry, MoE
                                facts, the contract's split.
  * :class:`FitPlan`          — the ordered decision: evictions, offload /
                                split choice (with the RAM need it implies, in
                                the SAME basis as the budget), final need,
                                a structured :class:`FitFailure` on refusal,
                                attribution strings.

All records are ``frozen`` dataclasses. Mapping-valued fields (``need_detail``,
``moe_detail``, ``moe_commit``…) are plain dicts by convention treated as
immutable; ``plan_fit`` never mutates its inputs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


# ── model-key identity (operator rule, board 2026-09-29) ─────────────────────
# ``X-GGUF`` and ``X`` are NECESSARILY the same model: the ``-GGUF`` suffix is a
# FORMAT variant of ONE key. Any other differing name token (``-Distill``, a
# quant marker, a finetune tag, an ``-i1``, anything) makes a DIFFERENT model
# that must not be loaded or served unless that exact key was called — even
# when both keys are backed by the same file on the worker's drive. Same-file
# residency never implies key equivalence. This is THE single definition; the
# resolver, central routing, the worker's resident match, the slot seat
# match and plan_fit's Resident identity all consult it (S3 / F7 eviction
# test 2026-09-29: a request for ``Qwen3.8-9B-GGUF`` was served by the
# resident ``Qwen3.8-9B-Distill-GGUF`` because both name the same GGUF file).
#
# Case is not a name token (the routing alias union already folds it); the
# separator before the suffix may be ``-`` or ``_`` (``Qwen3.8_4B_Distilled_GGUF``
# is a live key) — nothing else is folded.
_FORMAT_SUFFIX_RE = re.compile(r"[-_]gguf$", re.IGNORECASE)


def canonical_key(model_key: Any) -> str:
    """``model_key`` with ONLY the trailing format suffix (``-GGUF``) removed.
    Every other token is identity and is kept verbatim."""
    s = str(model_key or "").strip()
    return _FORMAT_SUFFIX_RE.sub("", s, count=1)


def key_equivalent(a: Any, b: Any) -> bool:
    """True iff ``a`` and ``b`` name the same model: equal after stripping only
    the ``-GGUF`` format suffix (case-insensitive). Empty never matches."""
    ca = canonical_key(a)
    return bool(ca) and ca.lower() == canonical_key(b).lower()


@dataclass(frozen=True)
class DeviceVram:
    """One GPU as measured: ``index`` (None when unknown / single card)."""
    index: Optional[int]
    total_bytes: Optional[int]
    free_bytes: Optional[int]


@dataclass(frozen=True)
class ResourceSnapshot:
    """The card(s) and the host, measured ONCE.

    ``total_bytes`` / ``free_bytes`` describe the TARGET card the load lands on
    (``devices`` carries every card when the caller knows them; ``target_device``
    names the target's index). ``free_bytes`` is the BUDGETABLE free VRAM —
    the device free read less ``external_floor_bytes``, the operator hold-back
    ``HUGPY_VRAM_RESERVE_GIB`` (DEFAULT 0 since 2026-09-29; it was a flat
    1.0 GiB "for out-of-band GPU consumers", retired because a foreign
    process's bytes are already out of the device free read and the context
    need is priced explicitly) — exactly what the worker's
    ``_free_vram_bytes()`` reports. ``external_floor_bytes`` is carried so a
    report can put the device figure back (``free + floor``).
    ``attributed_vram_bytes`` / ``foreign_vram_bytes`` are the MEASURED
    occupancy split (this worker's model rows + its own CUDA context / other
    compute processes) from the same per-pid accounting the heartbeat ships;
    reporting only — both are occupancy, already OUT of ``free_bytes``, and
    must never be subtracted from it again. ``ram_free_bytes`` is budgetable
    MemAvailable (reserve-adjusted), ``ram_total_bytes`` the box, and
    ``ram_reserve_bytes`` the host reserve that was held out. ``now`` is the
    clock the eviction ranking measures idleness against (never read inside
    the planner)."""
    total_bytes: Optional[int]
    free_bytes: Optional[int]
    external_floor_bytes: int = 0
    ram_free_bytes: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    ram_reserve_bytes: int = 0
    devices: tuple = ()
    target_device: Optional[int] = None
    now: float = 0.0
    attributed_vram_bytes: Optional[int] = None
    foreign_vram_bytes: Optional[int] = None


# The name the design note (and the step-1 brief) uses; the general record is
# ResourceSnapshot (VRAM per device + host RAM + reserves, taken once).
VramSnapshot = ResourceSnapshot


@dataclass(frozen=True)
class FitPolicy:
    """Admission knobs, resolved by the caller (env / settings / heartbeat).

    ``ceiling_reserve_bytes``   budgetable free VRAM that must REMAIN after the
                                need lands (``_vram_ceiling_reserve_bytes``).
    ``empty_card_budget_bytes`` the largest need this card could ever admit
                                (report only: "the full weights can never fit").
    ``least_reaping``           the fleet-wide drop-pass switch for
                                ``eviction.evict_plan``.
    ``alloc_mode``              k37 mode from the spill wire: ``"max-ram"`` |
                                ``"explicit"`` | None (legacy / autofit).
    ``gpu_target_bytes`` / ``ram_target_bytes``  the allocation CONTRACT's
                                ``HUGPY_GPU_MEM_GIB`` / ``HUGPY_CPU_MEM_GIB`` in
                                bytes, None when unset. ``ram_target_bytes`` is
                                the per-model RAM budget a MoE split's CPU
                                share is checked against (the slot preflight's
                                ``cpu_mem_gib``), now at PLAN time and at the
                                plan's own N.
    ``ctx_cap_on_evict_pct``    step 2 (F7), DEFAULT OFF (None). When set
                                (1..100), a subject that would otherwise
                                EVICT is first offered a reduced-ctx seat: if
                                weights + KV at ``min(cap, ctx_pct)`` fits the
                                free room, plan_fit returns a ``partial`` of
                                kind ``ctx-cap`` (``self_ctx_pct`` = the cap,
                                no evictions) instead of the eviction. A
                                PROPOSAL: the executor may honour or ignore
                                it. Motivation: at a model's max ctx the KV
                                term dwarfs the weights (4B -> 21.2 GB seat).
    """
    ceiling_reserve_bytes: int = 0
    empty_card_budget_bytes: Optional[int] = None
    least_reaping: bool = True
    alloc_mode: Optional[str] = None
    leniency_pct: Optional[float] = None
    priority_device: str = "gpu"
    gpu_target_bytes: Optional[int] = None
    ram_target_bytes: Optional[int] = None
    ram_safety_frac: float = 0.95
    min_offload_frac: float = 0.05
    ctx_cap_on_evict_pct: Optional[int] = None


@dataclass(frozen=True)
class Resident:
    """One measured GPU resident as the planner needs it.

    ``protected``/``why`` is THE protection verdict (static / actively replying /
    queued ahead / comfy busy / other card) — decided by the caller's single
    definition (``_partition_residents``); the planner never re-derives it and
    never touches a protected row.

    ``materialized`` is MEASURED residency — a live child / a probed in-process
    handle (True), proven absent (False), or unknown (None). It is NEVER
    inferred from dispatch ``_INSTANCES`` membership (a hollow runner object is
    not a resident — the "loaded and idle" diagnosis). Carried for the step-2
    residency rule; step 1 does not decide on it.

    The remaining fields are the flex inputs (``kv_bytes``, ``ctx_pct``,
    ``ctx_deviation_pct``, ``pinned``, ``priority``) and the shared-eviction
    ledger inputs (``pref``, ``last_call``, ``calls``, ``mid_generation``,
    ``resident_since`` — see ``eviction.EvictUnit``)."""
    model_key: str
    vram_bytes: Optional[int] = None
    host_mode: Optional[str] = None
    protected: bool = False
    why: Optional[str] = None
    materialized: Optional[bool] = None
    pinned: bool = False
    priority: int = 0
    kv_bytes: int = 0
    ctx_pct: Optional[int] = None
    ctx_deviation_pct: Optional[float] = None
    pref: str = "vram"
    last_call: Optional[float] = None
    calls: int = 0
    mid_generation: bool = False
    resident_since: Optional[float] = None
    gpu_index: Optional[int] = None

    @property
    def identity(self) -> str:
        """The resident's CANONICAL identity (``canonical_key``): the key with
        only the ``-GGUF`` format suffix removed. Two residents are the same
        model iff their identities are equal — a ``…-Distill-GGUF`` seat is a
        distinct resident from a ``…-GGUF`` seat even when both mmap the same
        file; a load of the plain key is a separate admission / seat."""
        return canonical_key(self.model_key)


@dataclass(frozen=True)
class FitRequest:
    """The incoming load.

    ``need_bytes``          the caller-PRICED total need: an explicit figure, or
                            ``need_detail["total"]``, already 4-bit re-priced
                            when that lever is on (pricing levers that read the
                            request env are the caller's; the planner decides).
    ``need_detail``         the authoritative split (``total``, ``weights``,
                            ``kv``, ``ctx_pct``, ``moe_split``…).
    ``planned_gpu_bytes``   what the placement intent will actually put on the
                            card (``spill.planned_gpu_need_bytes(need_bytes)``),
                            None when unknown.
    ``subject_held_bytes``  VRAM the subject ALREADY holds (the subject credit).
    ``polite``              k56 ``no_evict``: spend only genuinely free room.
    ``gguf_path`` / ``total_layers`` / ``ngl_intent`` / ``ngl_requested``
                            served-quant geometry and the n_gpu_layers intent
                            (partial-offload inputs; ``total_layers`` None means
                            non-GGUF → no hybrid is possible).
    ``moe_detail``          ``spill.gguf_moe_detail`` of the served quant, or None.
    ``moe_auto_gpu_budget_bytes``  the slot's MoE budget as the worker prices it
                            (``_moe_auto_gpu_budget``), None when unmeasurable.
    ``contract_n_cpu_moe``  the allocation contract's ``--n-cpu-moe`` (the
                            ``HUGPY_N_CPU_MOE`` wire), None when the contract
                            states none — recorded on the plan's split so a
                            plan that re-derives N is never silently compared
                            against a budget priced at the contract's N.
    ``split_expressible``   whether an engine that can express a MoE split
                            (native slot ``--n-cpu-moe``) is available: True /
                            False / None (unknown → not checked). A split plan
                            with ``False`` is a plan-time ``FitFailure``
                            (``split_not_expressible``), never an in-process
                            fallback.
    """
    model_key: str
    need_bytes: Optional[int]
    need_detail: Mapping[str, Any] = field(default_factory=dict)
    planned_gpu_bytes: Optional[int] = None
    subject_held_bytes: int = 0
    ctx_deviation_pct: Optional[float] = None
    vram_deviation_pct: Optional[float] = None
    priority: int = 0
    polite: bool = False
    gguf_path: Optional[str] = None
    total_layers: Optional[int] = None
    ngl_intent: str = "auto"
    ngl_requested: Optional[int] = None
    moe_detail: Optional[Mapping[str, Any]] = None
    moe_auto_gpu_budget_bytes: Optional[int] = None
    contract_n_cpu_moe: Optional[int] = None
    split_expressible: Optional[bool] = None


@dataclass(frozen=True)
class Eviction:
    """One planned (or, under a polite load, SPARED) eviction, in plan order."""
    model_key: str
    vram_bytes: Optional[int] = None
    host_mode: Optional[str] = None
    rank: int = 0


@dataclass(frozen=True)
class MoeSplit:
    """The expert split a plan commits to, priced in ONE basis.

    ``n_cpu_moe``   the ``--n-cpu-moe`` the child launches with (999 = all).
    ``gpu_bytes``   what lands on the card (dense backbone + kept experts [+ KV
                    when the source carried it]).
    ``cpu_bytes``   the expert bytes that land in host RAM AT THIS N — the RAM
                    need the plan implies; the budget is checked against THIS,
                    never against a figure priced at another N.
    ``basis``       where the numbers came from: ``"contract"`` (the need
                    detail's governing split), ``"moe-first"`` / ``"mode-moe"``
                    (re-derived here from the snapshot budget).
    ``contract_n_cpu_moe``  the contract's N for comparison (None = none)."""
    n_cpu_moe: int
    gpu_bytes: Optional[int] = None
    cpu_bytes: Optional[int] = None
    basis: str = "plan"
    contract_n_cpu_moe: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


# FitFailure kinds (diagnosis (b).6): capacity failures a plan can name at
# PLAN time. ``permanent`` = no state change would admit it; ``state_dependent``
# = an eviction / a freed card / a changed contract could.
FAILURE_KINDS = ("vram_fit", "ram_budget", "split_not_expressible",
                 "engine_unavailable", "hard_load")


@dataclass(frozen=True)
class FitFailure:
    """A structured refusal: WHAT failed, in numbers, in one basis.

    ``kind``          one of :data:`FAILURE_KINDS`.
    ``code``          a finer code (``wont_fit`` | ``over_budget`` |
                      ``no_native_engine`` …).
    ``reason``        the deterministic sentence (bytes, with units).
    ``need_bytes`` / ``budget_bytes``  the two figures that disagreed, priced in
                      the same basis (VRAM need vs VRAM room for ``vram_fit``;
                      RAM need at the plan's N vs the contract's RAM budget for
                      ``ram_budget``).
    ``plan_n_cpu_moe`` / ``contract_n_cpu_moe``  the split N the plan derived
                      vs the contract's, so a basis mismatch is visible.
    ``ctx_effective`` / ``kv_bytes`` / ``ctx_pct``  (2026-09-29) the context
                      the need was priced at and the KV term it carried — the
                      fit never prices KV at zero, so a refusal always says
                      which ctx it refused.
    """
    kind: str
    code: str
    reason: str
    need_bytes: Optional[int] = None
    budget_bytes: Optional[int] = None
    plan_n_cpu_moe: Optional[int] = None
    contract_n_cpu_moe: Optional[int] = None
    permanent: bool = False
    state_dependent: bool = True
    ctx_effective: Optional[int] = None
    kv_bytes: Optional[int] = None
    ctx_pct: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FitPlan:
    """The decision. ``action``:

      ``proceed``  fits now (or the gate is a no-op / fails open — see ``note``);
                   no eviction. ``size_up_eligible`` tells the executor whether
                   the eviction-aware autofit size-up may still run.
      ``flex``     fits after compressing the subject's own ctx to
                   ``self_ctx_pct``; no eviction.
      ``evict``    ``evictions`` (ordered, minimum set) are predicted to clear
                   the ceiling.
      ``partial``  even after the planned evictions the full offload is short;
                   admit the hybrid described by ``partial`` / ``n_gpu_layers`` /
                   ``split`` (``partial_kind``: ``moe-first`` | ``mode-moe`` |
                   ``dense``) — or, under ``FitPolicy.ctx_cap_on_evict_pct``,
                   the reduced-ctx seat ``ctx-cap`` (``self_ctx_pct`` = the
                   cap, NO evictions; a proposal the executor may ignore).

    ``need_detail["need_split"]`` (step 2, F7) is the explicit weights-vs-KV
    split the need was priced from: ``weights_bytes``, ``kv_bytes``,
    ``kv_share_pct``, ``ctx_pct``, ``ctx_resolved``, ``ctx_max`` — re-priced
    whenever a flex / ctx-cap changes the KV term.
      ``refuse``   honest refusal; ``failure`` is the structured
                   :class:`FitFailure`, ``refuse_reason`` its sentence.

    ``need_bytes`` is the FINAL priced need (after intent / MoE re-target /
    self-flex) — the figure the executor's per-victim fit re-check uses.
    ``split`` is the MoE split that governs the load when set, and
    ``ram_need_bytes`` the host RAM it implies AT THAT N; ``ram_budget_bytes``
    is the contract's RAM budget it was checked against (None = no contract).
    ``moe_commit`` keeps the need detail's governing split dict for the
    executor's commit. ``reasons`` is the attribution trail.
    """
    action: str
    model_key: str
    need_bytes: Optional[int] = None
    need_detail: Mapping[str, Any] = field(default_factory=dict)
    weights_bytes: Optional[int] = None
    kv_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    free_effective_bytes: Optional[int] = None
    subject_held_bytes: int = 0
    ceiling_reserve_bytes: int = 0
    fits_now: Optional[bool] = None
    self_ctx_pct: Optional[int] = None
    flex: Optional[Mapping[str, Any]] = None
    flex_note: Optional[str] = None
    evictions: tuple = ()
    eviction_need_bytes: Optional[int] = None
    predicted_freed_bytes: int = 0
    predicted_fits: Optional[bool] = None
    polite_spared: tuple = ()
    moe_commit: Optional[Mapping[str, Any]] = None
    split: Optional[MoeSplit] = None
    n_gpu_layers: Optional[int] = None
    n_cpu_moe: Optional[int] = None
    ram_need_bytes: Optional[int] = None
    ram_budget_bytes: Optional[int] = None
    moe_plan: Optional[Mapping[str, Any]] = None
    partial: Optional[Mapping[str, Any]] = None
    partial_kind: Optional[str] = None
    budget_bytes: Optional[int] = None
    size_up_eligible: bool = False
    comfy_reclaim_eligible: bool = False
    failure: Optional[FitFailure] = None
    refuse_reason: Optional[str] = None
    reasons: tuple = ()
    note: str = ""

    @property
    def evicted_keys(self) -> list:
        return [e.model_key for e in self.evictions]

    def as_dict(self) -> dict:
        return asdict(self)


__all__ = ["DeviceVram", "ResourceSnapshot", "VramSnapshot", "FitPolicy",
           "Resident", "FitRequest", "Eviction", "MoeSplit", "FitFailure",
           "FAILURE_KINDS", "FitPlan"]
