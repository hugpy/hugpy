"""Tolerance-band secondary allocation layer (t21) — the pure band math + the
flex-before-evict decision, and the ONE priority comparator seam.

WHY THIS EXISTS (operator spec, recorded 2026-07-17, verbatim intent):
  "a threshold should be made as a secondary allocation layer, in which the
   percentage can deviate within a certain percent total from the desired
   explicit allocation for edge cases and explicit priorities."
Extended the same day: context (ctx%) joins VRAM% and RAM% as the THIRD banded
variable — the SAME system, one engine.

── THE MODEL ────────────────────────────────────────────────────────────────
Every EXPLICIT allocation (the spill contract — GGUF-only per 98f5056/363d0ed)
gains, per banded variable, a ``target`` and a ``deviation_pct``. The deviation
is PERCENT-OF-TOTAL (of the honest whole — the encroachment-model denominators
from the honest-bars work, commit e0bd9e4), so a band is the HARD interval

    [ target − deviation_pct%·whole , target + deviation_pct%·whole ]

clamped to the variable's own domain ([0, whole] for bytes; [1, 100] for ctx%).

THREE banded variables, ONE engine:
  * VRAM%  — target = spill ``gpu_mem_gib`` (bytes); whole = the box's honest
             VRAM denominator.
  * RAM%   — target = spill ``cpu_mem_gib`` (bytes); whole = the honest RAM
             denominator.
  * CTX%   — target = ``ctx_pct`` (already a percent 1..100 of the model's max
             context); the band is ± deviation percentage-points of that scale
             (i.e. "whole" is the 100-point ctx scale, so a percent-of-total
             deviation is just ± that many points).

── RESOLUTION ORDER AT CONTENTION (the operator's three stages) ─────────────
  (1) FLEX  — an edge-case load may stretch WITHIN its own band; a higher-
              priority load may compress neighbors WITHIN THEIR bands (never
              below the band FLOOR). ctx-KV is the CHEAPEST flex, so this module
              prefers compressing ctx before touching weight placement.
  (2) EVICT — only when flexing cannot fit (idle on-demand first, LRU, with the
              admission-doctrine protections — static/replying/queue-ahead never
              yield). This module does NOT evict; it hands the caller a plan and
              the caller runs the existing eviction engine (agent._vram_evict_to
              _fit) for stage (2).
  (3) REFUSE honestly — the existing honest-refusal path, unchanged.

UNCONTENDED == everyone at target. Bands ONLY matter under contention: if the
subject already fits at its target, no band is consulted and behaviour is
byte-identical to before this module existed.

── PROTECTION OUTRANKS PRIORITY ─────────────────────────────────────────────
The admission-doctrine protected tiers (🔒static / actively-replying / queued-
ahead) are NEVER flex-compressed by ANY priority. Protection is absolute;
priority only orders the UNPROTECTED, flex-eligible neighbours.

This module is PURE (no I/O, no globals, no eviction). That is what lets the
tests assert floors/ceilings, the compress-by-priority ordering, protection
immunity, uncontended==target, and the ctx-first ordering without a GPU.
"""
from __future__ import annotations

from typing import Optional


# ── THE priority comparator seam ─────────────────────────────────────────────
# Keeper decision (confirm-pending with the operator, 2026-07-17): priority is an
# optional explicit per-model integer on the explicit allocation (unset == 0 ==
# normal; higher compresses lower within bands). Ties are broken by pin (the
# existing pin-as-tiebreak doctrine) — the caller supplies that tiebreak; this
# function answers ONLY the primary priority key. Protected tiers are handled by
# the caller and never reach a priority comparison.
#
# THE OPERATOR MAY ADJUST THE SOURCE of priority (e.g. residency tier, a queue
# signal, an SLA class). Keep that change to THIS ONE function: everything else
# consumes ``flex_priority_key`` and never re-reads ``alloc['priority']`` itself,
# so a different answer plugs in here without re-plumbing the engine.
def flex_priority_key(alloc: Optional[dict]) -> int:
    """The primary priority key for an explicit allocation (higher == more
    important, i.e. compresses/evicts lower-priority neighbours first). Unset,
    non-dict, or unparseable == 0 == normal. THE single source of the priority
    number — see the module note before changing where it reads from."""
    if not isinstance(alloc, dict):
        return 0
    try:
        return int(alloc.get("priority") or 0)
    except (TypeError, ValueError):
        return 0


# ── band math (bytes-domain: VRAM / RAM) ─────────────────────────────────────
def band_bounds(target: float, deviation_pct: Optional[float],
                whole: Optional[float]) -> "tuple[float, float]":
    """The hard band interval ``(floor, ceiling)`` for a bytes-domain variable.

    ``deviation_pct`` is percent-of-``whole`` (the honest denominator). No band
    (deviation unset/<=0 or ``whole`` unknown) collapses to the point
    ``(target, target)`` — a model with no declared tolerance can neither be
    stretched nor compressed, so it behaves exactly as it did pre-t21.

    Clamped to ``[0, whole]`` when ``whole`` is known (never propose a negative
    or over-total allocation). ``target`` itself is clamped into the same domain
    first so a mis-entered target can't produce a floor above the ceiling."""
    t = max(0.0, float(target or 0.0))
    if whole is not None and whole > 0:
        t = min(t, float(whole))
    dev = float(deviation_pct or 0.0)
    if dev <= 0 or not whole or whole <= 0:
        return t, t
    span = (dev / 100.0) * float(whole)
    floor = max(0.0, t - span)
    ceil = min(float(whole), t + span)
    if ceil < floor:                      # degenerate (target clamped past ceil)
        floor = ceil = t
    return floor, ceil


def band_floor(target: float, deviation_pct: Optional[float],
               whole: Optional[float]) -> float:
    """The most-compressed admissible allocation for a bytes-domain band — the
    minimum a higher-priority neighbour may squeeze this one to (stage 1). Never
    below the floor (operator: 'never below band floor')."""
    return band_bounds(target, deviation_pct, whole)[0]


def band_ceiling(target: float, deviation_pct: Optional[float],
                 whole: Optional[float]) -> float:
    """The most-stretched admissible allocation for a bytes-domain band — the
    maximum an edge-case load may stretch ITSELF to (stage 1, self-flex up)."""
    return band_bounds(target, deviation_pct, whole)[1]


# ── band math (ctx%: the 1..100 point scale) ─────────────────────────────────
def ctx_band_bounds(ctx_pct: Optional[int],
                    deviation_pct: Optional[float]) -> "tuple[int, int] | None":
    """The hard ctx band ``(floor_pct, ceil_pct)`` in whole percent points, or
    None when there is no ctx target (ctx band is opt-in — no target == today's
    default ctx, byte-identical).

    ctx% is ALREADY a percent of the model's max context, so a percent-of-total
    deviation is simply ± that many points, clamped to the ctx domain [1, 100].
    ctx is the CHEAPEST flex (KV bytes are linear in ctx tokens), so the engine
    reaches for this band first."""
    if ctx_pct is None:
        return None
    try:
        c = int(ctx_pct)
    except (TypeError, ValueError):
        return None
    c = max(1, min(100, c))
    dev = float(deviation_pct or 0.0)
    if dev <= 0:
        return c, c
    floor = max(1, int(round(c - dev)))
    ceil = min(100, int(round(c + dev)))
    if ceil < floor:
        floor = ceil = c
    return floor, ceil


def kv_at_ctx_pct(kv_at_target: Optional[int], target_pct: Optional[int],
                  new_pct: Optional[int]) -> int:
    """Scale a KV-cache byte figure measured at ``target_pct`` to ``new_pct``.

    KV bytes are LINEAR in resolved ctx tokens and ctx tokens are linear in the
    percent, so ``kv(new) = kv(target) · new_pct / target_pct``. Used to price a
    ctx flex (compress the subject's ctx toward its band floor) WITHOUT a re-load
    or re-measure. Degenerate inputs (missing/zero) return the input unchanged so
    a pricing gap never fabricates savings."""
    if not kv_at_target or not target_pct or not new_pct:
        return int(kv_at_target or 0)
    try:
        return max(0, int(kv_at_target * float(new_pct) / float(target_pct)))
    except (TypeError, ValueError, ZeroDivisionError):
        return int(kv_at_target or 0)


# ── stage (2.5): honest GGUF partial-offload (autofit's contract) ────────────
# When the FULL GGUF does not fit the GPU even after flex + evict, autofit's
# PROMISE is a hybrid: offload as many layers as safely fit and stream the rest
# from disk to CPU RAM — NOT a hard refusal. This is the layers-that-fit math,
# kept PURE so the tests can assert the split, the KV proportionality, the RAM
# guard, the degenerate floor, and the intent modes without a GPU. The agent
# calls it AFTER its eviction loop, BEFORE building the honest refusal.
#
# WHY per-layer VRAM folds KV in: llama.cpp allocates the KV cache per layer, so
# a layer offloaded to the GPU also holds its KV slice on the GPU. Pricing each
# offloaded layer at (weights_share + kv_share) means the plan NEVER over-admits
# the VRAM budget — the incident we're fixing was an admit-then-OOM, so the
# budget must hold weights AND their KV, not weights alone.
class PartialPlan:
    """The result of :func:`plan_partial_offload`.

    ``admit`` True  -> serve as a hybrid: ``n_gpu_layers`` offloaded to the GPU
                       (holding ``vram_need_bytes``), the remainder streamed to
                       host RAM (``ram_need_bytes``). ``gpu_pct`` is the offloaded
                       fraction as whole percent.
    ``admit`` False -> the hybrid was rejected (``reject_reason`` says why: a
                       degenerate offload below the floor, or a CPU remainder that
                       would OOM host RAM). The caller REFUSES honestly, extended
                       with this plan so the operator sees what was considered."""

    __slots__ = ("admit", "n_gpu_layers", "gpu_pct", "total_layers",
                 "vram_need_bytes", "ram_need_bytes",
                 "weights_gpu_bytes", "weights_cpu_bytes",
                 "kv_gpu_bytes", "kv_cpu_bytes",
                 "vram_budget_bytes", "ram_free_bytes",
                 "floor_layers", "reject_reason", "note")

    def __init__(self, admit, n_gpu_layers=0, gpu_pct=0, total_layers=0,
                 vram_need_bytes=0, ram_need_bytes=0, weights_gpu_bytes=0,
                 weights_cpu_bytes=0, kv_gpu_bytes=0, kv_cpu_bytes=0,
                 vram_budget_bytes=0, ram_free_bytes=0, floor_layers=0,
                 reject_reason=None, note=""):
        self.admit = bool(admit)
        self.n_gpu_layers = int(n_gpu_layers)
        self.gpu_pct = int(gpu_pct)
        self.total_layers = int(total_layers)
        self.vram_need_bytes = int(vram_need_bytes)
        self.ram_need_bytes = int(ram_need_bytes)
        self.weights_gpu_bytes = int(weights_gpu_bytes)
        self.weights_cpu_bytes = int(weights_cpu_bytes)
        self.kv_gpu_bytes = int(kv_gpu_bytes)
        self.kv_cpu_bytes = int(kv_cpu_bytes)
        self.vram_budget_bytes = int(vram_budget_bytes)
        self.ram_free_bytes = int(ram_free_bytes or 0)
        self.floor_layers = int(floor_layers)
        self.reject_reason = reject_reason
        self.note = note

    def as_dict(self) -> dict:
        return {"admit": self.admit, "n_gpu_layers": self.n_gpu_layers,
                "gpu_pct": self.gpu_pct, "total_layers": self.total_layers,
                "vram_need_bytes": self.vram_need_bytes,
                "ram_need_bytes": self.ram_need_bytes,
                "weights_gpu_bytes": self.weights_gpu_bytes,
                "weights_cpu_bytes": self.weights_cpu_bytes,
                "kv_gpu_bytes": self.kv_gpu_bytes,
                "kv_cpu_bytes": self.kv_cpu_bytes,
                "vram_budget_bytes": self.vram_budget_bytes,
                "ram_free_bytes": self.ram_free_bytes,
                "floor_layers": self.floor_layers,
                "reject_reason": self.reject_reason, "note": self.note}


def plan_partial_offload(*, weights_bytes: Optional[int], kv_bytes: Optional[int],
                         total_layers: Optional[int], vram_budget_bytes: Optional[int],
                         ram_free_bytes: Optional[int], intent: str = "auto",
                         requested_layers: Optional[int] = None,
                         min_offload_frac: float = 0.05,
                         ram_safety_frac: float = 0.95) -> Optional["PartialPlan"]:
    """Layers-that-fit plan for an oversize GGUF — autofit's hybrid contract. PURE.

    Inputs (all VRAM/RAM figures in bytes, from the caller's honest, shard-aware
    need pricing — NOT the shard-blind on-disk file size):
      * ``weights_bytes``    honest effective weights of the WHOLE model.
      * ``kv_bytes``         KV-cache bytes at the resolved (possibly ctx-flexed)
                             context — 0 when no ctx allocation is set.
      * ``total_layers``     the GGUF's ``.block_count`` (transformer blocks).
      * ``vram_budget_bytes``the VRAM the offloaded layers may consume = free
                             VRAM after the ceiling reserve AND after flex+evict,
                             already capped by any explicit VRAM band ceiling.
      * ``ram_free_bytes``   budgetable free host RAM for the CPU remainder.
      * ``intent``           placement intent (t26): ``"auto"``/``"gpu"`` fit as
                             many layers as safely fit (identical for an oversize
                             model — see below); ``"cpu"`` forces 0 GPU layers.
      * ``requested_layers`` an explicit positive n_gpu_layers count (GGUF layer
                             count) to honor, capped to what fits.

    Returns a :class:`PartialPlan`, or ``None`` when the plan is NOT computable
    (missing/degenerate geometry or weight size) so the caller keeps its existing
    honest refusal unchanged.

    ── "Max GPU" vs autofit (documented choice) ──────────────────────────────
    For an OVERSIZE model both converge on the same answer: as many layers as fit
    UNDER the ceiling reserve. "Max GPU" deliberately does NOT squeeze the ceiling
    reserve here — doing so reintroduces the exact admit-then-OOM this fix closes.
    The only place the two ever differ (a model that FULLY fits) never reaches
    this function; it is admitted whole (-1) upstream.

    ── The math ──────────────────────────────────────────────────────────────
      per_layer_vram = (weights + kv) / total_layers   (KV rides its GPU layer)
      n_gpu          = floor(budget / per_layer_vram)   (0..total_layers)
      vram_need      = n_gpu · per_layer_vram
      ram_need       = (total_layers − n_gpu)/total_layers · (weights + kv)

    ── Floors (never admit dross, never admit-then-OOM) ──────────────────────
      * degenerate: a NON-cpu plan whose ``n_gpu`` is below
        ``ceil(min_offload_frac · total_layers)`` (min 1) buys negligible speed
        for real VRAM — reject (serve CPU-only or a smaller quant instead).
      * RAM: a CPU remainder above ``ram_free · ram_safety_frac`` would OOM host
        RAM — reject. (``cpu`` intent is still subject to this — a CPU-only model
        that can't fit RAM must refuse, not OOM.)
    """
    # Geometry / weight guards — not computable -> caller keeps its refusal.
    try:
        total_layers = int(total_layers)
        weights = int(weights_bytes)
    except (TypeError, ValueError):
        return None
    if total_layers <= 0 or weights <= 0:
        return None
    kv = max(0, int(kv_bytes or 0))
    budget = max(0, int(vram_budget_bytes or 0))
    ram_free = max(0, int(ram_free_bytes or 0))
    whole = weights + kv
    per_layer_vram = whole / float(total_layers)          # weights + KV per layer

    mode = (intent or "auto").strip().lower()
    if mode == "cpu":
        n_gpu = 0
    elif per_layer_vram <= 0:
        n_gpu = 0
    else:
        fit = int(budget // per_layer_vram)
        if requested_layers is not None:                  # explicit count, capped
            try:
                fit = min(fit, max(0, int(requested_layers)))
            except (TypeError, ValueError):
                pass
        n_gpu = max(0, min(fit, total_layers))

    frac = n_gpu / float(total_layers)
    weights_gpu = int(round(weights * frac))
    weights_cpu = max(0, weights - weights_gpu)
    kv_gpu = int(round(kv * frac))
    kv_cpu = max(0, kv - kv_gpu)
    vram_need = weights_gpu + kv_gpu
    ram_need = weights_cpu + kv_cpu
    gpu_pct = int(round(100 * frac))
    floor_layers = max(1, int(-(-min_offload_frac * total_layers // 1)))  # ceil

    plan = PartialPlan(
        admit=True, n_gpu_layers=n_gpu, gpu_pct=gpu_pct, total_layers=total_layers,
        vram_need_bytes=vram_need, ram_need_bytes=ram_need,
        weights_gpu_bytes=weights_gpu, weights_cpu_bytes=weights_cpu,
        kv_gpu_bytes=kv_gpu, kv_cpu_bytes=kv_cpu,
        vram_budget_bytes=budget, ram_free_bytes=ram_free, floor_layers=floor_layers,
        note=f"{n_gpu}/{total_layers} layers on GPU ({gpu_pct}%)")

    # Degenerate floor — a non-cpu hybrid that offloads almost nothing.
    if mode != "cpu" and n_gpu < floor_layers:
        plan.admit = False
        plan.reject_reason = (
            f"partial GPU offload degenerate: only {n_gpu}/{total_layers} layers "
            f"({gpu_pct}%) fit the {budget / 2**30:.1f} GiB VRAM budget — below the "
            f"{floor_layers}-layer floor ({int(min_offload_frac*100)}% of layers); "
            f"serve CPU-only or pick a smaller quant")
        return plan

    # RAM guard — never admit a hybrid whose CPU remainder OOMs host RAM.
    if ram_need > ram_free * ram_safety_frac:
        plan.admit = False
        plan.reject_reason = (
            f"partial GPU offload would need ~{ram_need / 2**30:.1f} GiB host RAM "
            f"for the {total_layers - n_gpu} CPU-resident layer(s) but only "
            f"~{ram_free / 2**30:.1f} GiB is budgetable — would OOM system RAM")
        return plan

    return plan


# ── k37: the explicit / max-ram admission engine ─────────────────────────────
# LENIENCY (operator-confirmed math): N% leniency = up to N% OF THE MODEL may
# land off its ideal device before bust. The ideal share on the priority device
# slides from its target down to (target − N% of the model) — the FLOOR — and
# the engine DEGRADES within that band before busting past it. 100% GPU target
# + 30% leniency -> floor 70% GPU / 30% RAM. The conversion onto the band
# machinery above: whole = the MODEL's bytes, deviation_pct = leniency_pct, so
# band_floor(target_bytes, leniency_pct, model_bytes) IS the leniency floor.
def leniency_floor_pct(target_pct: float, leniency_pct: Optional[float]) -> float:
    """The floor SHARE (percent of the model) the priority device may degrade
    to: max(0, target − leniency). 100/30 -> 70."""
    try:
        t = max(0.0, min(100.0, float(target_pct)))
    except (TypeError, ValueError):
        t = 100.0
    lo = float(leniency_pct or 0.0)
    return max(0.0, t - max(0.0, lo))


def plan_explicit_offload(*, weights_bytes: Optional[int], kv_bytes: Optional[int],
                          total_layers: Optional[int],
                          vram_budget_bytes: Optional[int],
                          ram_free_bytes: Optional[int],
                          mode: str = "explicit",
                          priority_device: str = "gpu",
                          gpu_target_bytes: Optional[int] = None,
                          ram_target_bytes: Optional[int] = None,
                          leniency_pct: Optional[float] = None,
                          ram_safety_frac: float = 0.95) -> Optional["PartialPlan"]:
    """Layers-that-fit plan for the k37 **explicit** and **max-ram** modes —
    degrade WITHIN the leniency band toward the floor, bust past it with an
    HONEST refusal that names the mode and the floor. PURE (no I/O), like
    plan_partial_offload, and returns the same :class:`PartialPlan` so the
    agent's admit/refuse plumbing is shared.

    max-ram IS explicit(ram priority, 100% target, 100% leniency) internally —
    the caller passes ``mode="max-ram"`` so refusals still name the flat mode
    the operator chose (the 5 flat names are a cognitive-load boundary; the
    shared engine is an implementation detail).

    Semantics per priority device:
      * ``gpu``: ideal = the VRAM target (bytes; default the whole model).
        Place as many layers as the VRAM budget holds, capped at the target
        (degrade = accept less GPU than ideal, down to the leniency floor).
        Below the floor -> bust. The RAM remainder must fit budgetable RAM.
      * ``ram``: mirror — fill RAM toward its target first; only the OVERFLOW
        layers go to the GPU (autofit's per-layer pricing, inverted). RAM
        share below the floor -> bust; overflow the GPU can't hold -> bust
        ("can't satisfy").

    Returns None when not computable (missing geometry/size) so the caller
    keeps its existing honest-refusal path unchanged."""
    try:
        total_layers = int(total_layers)
        weights = int(weights_bytes)
    except (TypeError, ValueError):
        return None
    if total_layers <= 0 or weights <= 0:
        return None
    kv = max(0, int(kv_bytes or 0))
    vram_budget = max(0, int(vram_budget_bytes or 0))
    ram_free = max(0, int(ram_free_bytes or 0))
    ram_budget = int(ram_free * ram_safety_frac)
    whole = weights + kv                       # the MODEL — leniency's denominator
    dev = "ram" if str(priority_device).strip().lower() == "ram" else "gpu"

    # Integer layer math (never float-divide whole/total then floor — a 1-ulp
    # error would drop a layer at exact fits): layers(x) = x·total // whole,
    # ceil for the floor requirement.
    def _layers_for(x: float) -> int:
        return max(0, min(int((int(x) * total_layers) // whole), total_layers))

    def _layers_ceil(x: float) -> int:
        return max(0, min(int(-((-int(x) * total_layers) // whole)), total_layers))

    # Layer count on the priority device: as many layers as the device budget's
    # BYTES actually hold (never over-admit the budget), stretched up to the
    # CEIL of the achieved share when the budget has room past the target.
    # ceil(achievable) alone would over-admit an exact-fit budget by a partial
    # layer; floor(achievable) alone loses a layer the budget can pay for.
    def _layers_within(achieved: int, dev_budget: int) -> int:
        return min(_layers_ceil(achieved), _layers_for(dev_budget))

    if dev == "gpu":
        ideal = min(whole, int(gpu_target_bytes)) if gpu_target_bytes else whole
        floor = band_floor(ideal, leniency_pct, whole)
        achievable = min(ideal, vram_budget)
        n_gpu = _layers_within(achievable, vram_budget)
        floor_layers = _layers_ceil(floor)
    else:
        ideal = min(whole, int(ram_target_bytes)) if ram_target_bytes else whole
        floor = band_floor(ideal, leniency_pct, whole)
        achievable = min(ideal, ram_budget)
        in_ram = _layers_within(achievable, ram_budget)
        n_gpu = total_layers - in_ram          # ONLY the overflow rides the GPU
        floor_layers = _layers_ceil(floor)
    floor_pct = int(round(100.0 * floor / whole)) if whole else 0
    ideal_pct = int(round(100.0 * ideal / whole)) if whole else 0

    frac = n_gpu / float(total_layers)
    weights_gpu = int(round(weights * frac))
    weights_cpu = max(0, weights - weights_gpu)
    kv_gpu = int(round(kv * frac))
    kv_cpu = max(0, kv - kv_gpu)
    vram_need = weights_gpu + kv_gpu
    ram_need = weights_cpu + kv_cpu
    gpu_pct = int(round(100 * frac))

    plan = PartialPlan(
        admit=True, n_gpu_layers=n_gpu, gpu_pct=gpu_pct, total_layers=total_layers,
        vram_need_bytes=vram_need, ram_need_bytes=ram_need,
        weights_gpu_bytes=weights_gpu, weights_cpu_bytes=weights_cpu,
        kv_gpu_bytes=kv_gpu, kv_cpu_bytes=kv_cpu,
        vram_budget_bytes=vram_budget, ram_free_bytes=ram_free,
        floor_layers=floor_layers,
        note=(f"{mode} ({dev}-priority): {n_gpu}/{total_layers} layers on GPU "
              f"({gpu_pct}%), floor {floor_pct}% on {dev}"))

    # Bust past the floor: the priority device could not be given even the
    # LOOSEST allowed share (target − leniency). Compared BYTES-to-BYTES, not
    # layers-to-layers: n_gpu rounds the achieved share DOWN to whole layers
    # while floor_layers rounds the floor UP, so a layer comparison busts an
    # EXACT-fit budget (budget == floor, 0% leniency) against its own target —
    # the coder-next/ae 2026-08-28 refusal. If the budget's bytes cover the
    # floor's bytes, layer quantization must never turn that into a bust.
    # Integer bytes (ceil the float floor by whole bytes) — same 1-ulp doctrine
    # as the layer helpers above. Name mode + floor honestly when it DOES bust.
    floor_bytes = int(-(-floor // 1))          # ceil to a whole byte
    if int(achievable) < floor_bytes:
        plan.admit = False
        lo = float(leniency_pct or 0.0)
        plan.reject_reason = (
            f"{mode}: even the loosened floor won't fit — target {ideal_pct}% of "
            f"the model on {dev.upper()} with {lo:g}% leniency gives a floor of "
            f"{floor_pct}% ({floor_layers}/{total_layers} layers, "
            f"~{floor / 2**30:.1f} GiB) but only "
            f"{(vram_budget if dev == 'gpu' else ram_budget) / 2**30:.1f} GiB of "
            f"{dev.upper()} is budgetable — bust past the leniency floor")
        return plan

    if dev == "gpu":
        # RAM guard on the spilled remainder — degrading within the band must
        # never OOM host RAM.
        if ram_need > ram_budget:
            plan.admit = False
            plan.reject_reason = (
                f"{mode}: the {total_layers - n_gpu} off-GPU layer(s) need "
                f"~{ram_need / 2**30:.1f} GiB host RAM but only "
                f"~{ram_budget / 2**30:.1f} GiB is budgetable — would OOM system RAM")
            return plan
    else:
        # GPU guard on the overflow — "can't satisfy -> bust" (max-ram) /
        # explicit ram-priority overflow that the card can't hold.
        if vram_need > vram_budget:
            plan.admit = False
            plan.reject_reason = (
                f"{mode}: RAM holds {total_layers - n_gpu}/{total_layers} layers; "
                f"the {n_gpu}-layer overflow needs ~{vram_need / 2**30:.1f} GiB "
                f"VRAM but only ~{vram_budget / 2**30:.1f} GiB is budgetable — "
                f"can't satisfy")
            return plan

    return plan


# ── the flex-before-evict decision ───────────────────────────────────────────
class FlexPlan:
    """The result of :func:`plan_flex`.

    ``action``:
      * ``"proceed"`` — fits at target; bands untouched (uncontended path).
      * ``"flex"``    — fits AFTER flexing; ``self_ctx_pct`` is the compressed
                        ctx the subject should serve at (or None if unchanged),
                        ``compress`` is the ordered neighbour-compression plan
                        (list of ``{"model_key","from_ctx_pct","to_ctx_pct",
                        "kv_freed"}``), and ``freed_bytes`` is the total the flex
                        yields. The caller applies the self-ctx compression and
                        (worker-enforcement, next cut) the neighbour compression.
      * ``"evict"``   — flexing cannot fit; hand off to the eviction engine.
                        ``priority_order`` is the flex-priority-ascending +
                        pin-tiebreak order the evictor should yield candidates in
                        (lowest priority / most-compressible first).
    ``deficit_bytes`` is what still had to be found after self-flex; ``note`` is
    a short human trace of which stage resolved it."""

    __slots__ = ("action", "self_ctx_pct", "compress", "freed_bytes",
                 "priority_order", "deficit_bytes", "note")

    def __init__(self, action, self_ctx_pct=None, compress=None, freed_bytes=0,
                 priority_order=None, deficit_bytes=0, note=""):
        self.action = action
        self.self_ctx_pct = self_ctx_pct
        self.compress = compress or []
        self.freed_bytes = int(freed_bytes or 0)
        self.priority_order = priority_order or []
        self.deficit_bytes = int(deficit_bytes or 0)
        self.note = note

    def as_dict(self) -> dict:
        return {"action": self.action, "self_ctx_pct": self.self_ctx_pct,
                "compress": self.compress, "freed_bytes": self.freed_bytes,
                "priority_order": self.priority_order,
                "deficit_bytes": self.deficit_bytes, "note": self.note}


def _neighbour_sort_key(n: dict) -> tuple:
    """Yield/compress order for a flex-eligible neighbour: LOWEST priority first
    (a higher-priority subject compresses lower-priority neighbours), then
    UNPINNED before pinned (the existing pin-as-tiebreak doctrine — pin buys a
    hair of precedence at an exact priority tie, nothing more), then the largest
    compressible headroom first so the fewest neighbours are disturbed."""
    return (flex_priority_key(n.get("alloc")),
            bool(n.get("pinned")),
            -int(n.get("flex_headroom_bytes") or 0))


def plan_flex(subject: dict, residents: list, deficit_bytes: int) -> FlexPlan:
    """Decide whether ``deficit_bytes`` of VRAM pressure for ``subject`` can be
    absorbed by FLEXING within bands instead of evicting — stage (1).

    PURE. Never evicts, never mutates. The caller (agent._vram_evict_to_fit)
    runs this BEFORE its LRU eviction planner: a ``flex`` verdict means the load
    fits without evicting anyone; an ``evict`` verdict hands back the priority
    order the existing evictor should use.

    ``subject`` = ``{"weights_bytes", "kv_bytes", "ctx_pct", "ctx_deviation_pct",
        "priority"}`` — the incoming load's need split + its ctx band + priority.
    ``residents`` = list of ``{"model_key", "kv_bytes", "ctx_pct",
        "ctx_deviation_pct", "vram_bytes", "protected"(bool), "pinned"(bool),
        "alloc": {"priority": int}}`` — the current GPU residents.
    ``deficit_bytes`` = free_needed − free_have (>0; the shortfall to clear the
    ceiling). Callers only invoke this when the target does NOT fit, so a caller
    that passes deficit<=0 gets an immediate ``proceed``.

    Order of operations (ctx-first, self before neighbours, floor-respecting):
      1. SELF-FLEX: compress the subject's OWN ctx toward its band floor,
         re-pricing KV linearly. This shrinks the deficit at zero disruption.
      2. NEIGHBOUR-FLEX: only a higher-priority subject may reclaim resident KV
         by compressing lower-priority, UNPROTECTED neighbours toward THEIR ctx
         floors (priority-ascending, pin-tiebroken). Sum their headroom until the
         (post-self) deficit is covered.
      3. If self+neighbour flex covers the deficit -> ``flex``. Else -> ``evict``
         with the priority order for the eviction engine.
    """
    if deficit_bytes <= 0:
        return FlexPlan("proceed", note="fits at target — bands untouched")

    remaining = int(deficit_bytes)
    note_parts = []

    # ── stage 1: SELF-FLEX (compress the subject's own ctx to its band floor) ──
    self_ctx_pct = None
    subj_ctx = subject.get("ctx_pct")
    subj_dev = subject.get("ctx_deviation_pct")
    subj_kv = int(subject.get("kv_bytes") or 0)
    band = ctx_band_bounds(subj_ctx, subj_dev)
    if band is not None and subj_kv > 0:
        floor_pct, _ceil = band
        if floor_pct < int(subj_ctx):
            kv_floor = kv_at_ctx_pct(subj_kv, int(subj_ctx), floor_pct)
            saved = max(0, subj_kv - kv_floor)
            if saved > 0:
                self_ctx_pct = floor_pct
                remaining -= saved
                note_parts.append(
                    f"self ctx {subj_ctx}%->{floor_pct}% freed {saved}B")

    if remaining <= 0:
        return FlexPlan("flex", self_ctx_pct=self_ctx_pct,
                        freed_bytes=deficit_bytes - remaining,
                        deficit_bytes=0,
                        note="; ".join(note_parts) or "self-flex fit")

    # ── stage 2: NEIGHBOUR-FLEX (higher-priority subject compresses lowers) ────
    subj_prio = flex_priority_key(
        {"priority": subject.get("priority")})
    eligible = []
    for r in residents:
        if r.get("protected"):
            continue                       # protection outranks priority, always
        rb = ctx_band_bounds(r.get("ctx_pct"), r.get("ctx_deviation_pct"))
        rkv = int(r.get("kv_bytes") or 0)
        if rb is None or rkv <= 0:
            continue                       # nothing to compress
        r_floor, _ = rb
        if r_floor >= int(r.get("ctx_pct")):
            continue                       # no compressible headroom
        # ONLY a strictly-higher-priority subject may compress a neighbour.
        if subj_prio <= flex_priority_key(r.get("alloc")):
            continue
        kv_floor = kv_at_ctx_pct(rkv, int(r["ctx_pct"]), r_floor)
        headroom = max(0, rkv - kv_floor)
        if headroom <= 0:
            continue
        eligible.append({**r, "flex_headroom_bytes": headroom,
                         "to_ctx_pct": r_floor})

    eligible.sort(key=_neighbour_sort_key)

    compress = []
    for n in eligible:
        if remaining <= 0:
            break
        compress.append({"model_key": n["model_key"],
                         "from_ctx_pct": int(n["ctx_pct"]),
                         "to_ctx_pct": int(n["to_ctx_pct"]),
                         "kv_freed": int(n["flex_headroom_bytes"])})
        remaining -= int(n["flex_headroom_bytes"])

    if remaining <= 0:
        if compress:
            note_parts.append(
                f"compressed {len(compress)} lower-priority neighbour(s)")
        return FlexPlan("flex", self_ctx_pct=self_ctx_pct, compress=compress,
                        freed_bytes=deficit_bytes - remaining,
                        deficit_bytes=0,
                        note="; ".join(note_parts) or "neighbour-flex fit")

    # ── stage 3: flex cannot fit -> hand the evictor a priority order ──────────
    # Lowest priority / most-compressible first (unprotected only; the evictor
    # re-applies protection itself). This is the bridge until in-place resident
    # KV-shrink is enforced worker-side (next cut): a higher-priority subject's
    # eviction still prefers the lower-priority neighbours first.
    order = [r["model_key"] for r in
             sorted((r for r in residents if not r.get("protected")),
                    key=_neighbour_sort_key)]
    if note_parts:
        note_parts.append("flex insufficient — evict")
    return FlexPlan("evict", self_ctx_pct=self_ctx_pct, compress=compress,
                    freed_bytes=deficit_bytes - remaining,
                    priority_order=order, deficit_bytes=remaining,
                    note="; ".join(note_parts) or "no flex headroom — evict")
