"""THE shared eviction function — one implementation, every call site.

Operator spec: ``assets/evictionflow.html`` ("Allocation & Eviction Flow",
2026-07-25). This module is the executable form of its box 2, and nothing else
in the tree may re-implement the ordering.

── THE CORE IDEA ────────────────────────────────────────────────────────────
``max-*`` is a DEVICE PREFERENCE, and the SAME preference that decides where an
incoming model lands decides what gets pushed out to make room. The incoming
model fills its designated device to the fullest available; the preference
decides who leaves.

    admit M (size Z, preference P ∈ {max-gpu, max-ram})
      Z > X + Y (both devices combined)?  -> reject, infeasible on this card
      D := P's device (max-gpu->VRAM, max-ram->RAM);  O := the other device
      D_free >= Z?                        -> place all of Z on D
      else EVICT(D, need = Z - D_free); freed enough?
          -> place all of Z on D
          else place what fits on D; R := Z - placed
               O_free >= R?               -> place R on O (split residency)
               else EVICT(O, need = R - O_free); freed enough?
                   -> place R on O
                   else refuse, REPORTING THE BLOCKING RESIDENTS

``plan_admission`` below is that flowchart; ``evict_plan`` is the blue
subroutine both of its call sites run.

── THE SORT (box 2) ─────────────────────────────────────────────────────────
    pool := residents on d, minus 🔒static
    sort lexicographic:
      ① pref == other device first  (mismatched residents go first)
      ② time since last call, longest first (never-called anchors at load time)
      ③ total calls, fewest
      ④ model_key — stable final tiebreak
    WALK: accumulate victims in that order until freed >= n or pool exhausted
    DROP PASS, same order: remove any victim the remaining set already covers
    fully unload each remaining victim; return freed

WHAT THIS REPLACES. Every eviction site in the tree previously sorted
``(last_picked, -bytes, model_key)`` — oldest-first then LARGEST-FIRST. The
largest-first term is exactly what the spec's walk-then-drop replaces: it made
the *biggest* cold model the preferred victim, which clears a budget in the
fewest deletes but has no relationship to what the admission actually needs.
The spec orders by COST TO THE FLEET (cliff order, then idleness, then call
count) and then removes the surplus, which is a different and better answer.

── THE THREE INVARIANTS (spec's own words, and how they are enforced here) ──

**Parity.** Central's preview and the worker's auto-evict run THE SAME
function. That is why this module is PURE — no I/O, no globals, no clock reads,
no environment. Every input (sizes, free bytes, idle times, call counts,
preferences) is passed in by the caller, so a central preview and a worker
auto-evict over the same fixture produce byte-identical victim sets. Idle times
come from ONE ledger — central's call log, shipped to the worker at emission
(``model_last_picked`` / ``model_call_stats`` on the heartbeat reply) — never
from each side's own clock. Divergent victim sets are the bug this prevents;
``tests/test_eviction_parity.py`` asserts it directly.

**Least reaping.** ``_walk`` then ``_drop``. If y1 is first by order but y2
must go anyway and y2 alone covers the need, y1 is spared. The drop pass ONLY
REMOVES — it iterates the walked set and never consults the pool beyond the
frontier — so a hot model past the walk frontier is never taken just for being
conveniently sized. This asymmetry is load-bearing; a "pick the best-fitting
subset" optimiser would violate it.

**Full unload.** A victim is unloaded ENTIRELY, never spill-chained onto the
other device. Its contribution to ``freed`` is its own resident size, never a
function of recursive state elsewhere. That is what keeps the choice externally
derivable: you can recompute any decision from the inputs alone.

**Cliff order** is the rationale for key ①. A resident whose preference names
the OTHER device is already off the cliff by design — it asked to live
elsewhere and is only here opportunistically. A resident whose preference
matches this device loses its measured 135->36 tok/s when it goes, so it sorts
LAST. Mismatched first is not a tiebreak; it is the point.

── THE THREE OPEN ITEMS ─────────────────────────────────────────────────────
The spec marks three things "not yet decided" and states a PROPOSAL for each.
This module ENACTS those proposals so the behaviour is testable, and names them
here so the operator can rule and change exactly one place:

  1. IN-FLIGHT GUARD (``EvictUnit.mid_generation``). The spec's pool excludes only
     🔒static, which makes a model mid-generation a legal victim, and a long
     stream look idle if "last call" means request START. ENACTED PROPOSAL:
     last-activity = ``max(request start, last token emitted)`` (the caller
     supplies it as ``last_call``), and ``mid_generation`` removes the resident from
     the pool REGARDLESS of rank. Rationale for unevictable-not-deprioritised:
     a rank penalty still evicts it when it is the only candidate, which is the
     failure it exists to prevent.
  2. THRASH FLOOR — RETIRED 2026-07-27 (operator: "is there still some timeblock
     on a model being evicted? if so eliminate it"). A ``min_residency_s`` floor
     briefly removed freshly-loaded residents from the pool for 300s. It was a
     THIRD protection class, and a clock-driven one, which contradicts both
     standing rulings: protection is exactly ``static`` OR ``mid_generation``
     (2026-07-23), and "when designing anything that unloads/evicts/expires a
     model, ask what demanded the resources — if the answer is a timer, it's
     wrong by default" (minimize-loading doctrine).

     What it was actually defending against still stands — a fresh load has zero
     calls and anchors its idle clock at load time, so it sorts high in the
     never-called bucket — but the answer is RANK, not a veto. ``sort_key``
     already orders by (calls, last_call); a zero-call fresh load only wins the
     victim lottery when nothing better exists, and in that case evicting it is
     correct, because the alternative is refusing the admission outright. A veto
     turns "this is the least-bad victim" into "no victim at all".
  3. DROPDOWN DISAGREEMENT is not here — it is a defect in
     ``alloc_modes.feasible_default_mode`` and is fixed there.

── DEGRADE-NOT-GUESS ────────────────────────────────────────────────────────
An unmeasurable input never produces a guessed eviction. ``EvictUnit.bytes`` of
None (an occupant we could not size) makes the resident UNEVICTABLE-BY-PLAN: it
stays in the pool report as blocking, but is never walked, because evicting it
would free an unknown amount and the caller could not verify the plan. A None
``free`` or ``size`` at the admission level short-circuits to
``action="degrade"`` and the caller keeps today's behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# The two devices this vocabulary knows. "vram" is the GPU, "ram" is host RAM.
VRAM = "vram"
RAM = "ram"
DEVICES = (VRAM, RAM)

# Mode -> the device that mode PREFERS. The spec's P ∈ (max-gpu, max-ram) is
# the interesting pair; the "-only" modes are prohibitions rather than
# preferences but still name a device, and `explicit` names one via its
# priority_device (the caller resolves that before calling in).
#
# NOTE (per the brief): BOTH `{}` (derived max-gpu) and
# `{"alloc_mode": "max-gpu"}` (explicit max-gpu, fixed 2026-07-25 b0e02ff) mean
# max-gpu for preference purposes. They differ in PROVENANCE, not preference,
# so `preferred_device` is fed the resolved mode NAME and never the raw spill.
_MODE_DEVICE = {
    "max-gpu": VRAM,
    "gpu-only": VRAM,
    "max-ram": RAM,
    "ram-only": RAM,
}

# Least reaping (the DROP PASS) — the spec read literally: drop any walked
# victim the remaining set already covers, so ONE unload of 35 GiB satisfies a
# 15 GiB need instead of two smaller ones. That is fewer disturbances but LESS
# HEADROOM: the surplus 20 GiB is freed either way under the old greedy walk,
# and on a tight disk the operator may want it.
#
# DEFAULT True == today's shipped behaviour, unchanged. False restores the
# pre-drop-pass greedy walk EXACTLY: `kept = walked`, nothing spared. This is a
# FLEET-WIDE policy, never per-worker — see `evict_plan`'s note on why.
DEFAULT_LEAST_REAPING = True


def preferred_device(mode: Any, *, default: str = VRAM) -> str:
    """The device an allocation mode PREFERS — the spec's P -> D mapping.

    ``explicit`` has no fixed answer (its device comes from priority_device),
    so the caller resolves it to a name before calling; an unknown/unset mode
    degrades to ``default`` (VRAM), which is the blank max-gpu default and
    therefore today's behaviour."""
    m = str(mode or "").strip().lower()
    return _MODE_DEVICE.get(m, default)


def other_device(device: str) -> str:
    """O := the other device. The spec uses this in both admission branches."""
    return RAM if str(device).strip().lower() == VRAM else VRAM


@dataclass(frozen=True)
class EvictUnit:
    """One occupant of one device, as BOTH sides describe it.

    This is the parity contract: central builds these from its worker record
    and its call log; the worker builds them from its measured residents and
    the ledger central shipped it. Identical fields in -> identical victims out.

      model_key       stable identity, and the spec's final tiebreak ④.
      bytes           MEASURED resident footprint on this device. None = an
                      occupant of unknown size: reported as blocking, never
                      walked (degrade-not-guess).
      pref            this resident's preferred device (see preferred_device).
                      A pref naming the OTHER device sorts FIRST — key ①.
      last_call       last-activity epoch, from the ONE ledger. Per enacted
                      proposal 1 the caller passes
                      ``max(request start, last token emitted)``, never bare
                      request start. None/0 = never called -> the caller passes
                      ``resident_since`` in its place (the spec: "never-called
                      anchors at load time").
      calls           total call count from the same ledger — key ③.
      static          🔒static residency: THE only lock. The spec's pool is
                      "residents on d, minus static".
      mid_generation  enacted proposal 1: mid-generation -> not a candidate.
      resident_since  load-time epoch; the thrash floor's clock (proposal 2)
                      and the never-called idle anchor.
      why             free-text carried into the blocking report.
    """
    model_key: str
    bytes: Optional[int] = None
    pref: str = VRAM
    last_call: Optional[float] = None
    calls: int = 0
    static: bool = False
    mid_generation: bool = False
    resident_since: Optional[float] = None
    why: str = ""

    def idle_anchor(self) -> float:
        """The epoch key ② measures from. A called model anchors at its last
        call; a NEVER-called one anchors at load time, exactly as the spec says
        ("never-called = since load"). Neither known -> 0.0, the coldest
        possible anchor, which is right for a leftover nobody can account for.
        """
        for v in (self.last_call, self.resident_since):
            try:
                f = float(v)          # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if f > 0:
                return f
        return 0.0


@dataclass
class EvictPlan:
    """The result of EVICT(d, n) — a PLAN, never an action. Callers execute it.

      victims   model_keys to FULLY unload, in the order they were chosen.
      freed     bytes the plan frees (sum of the victims' own sizes — full
                unload, never a spill-chained function of the other device).
      need      what was asked for.
      enough    freed >= need. False = the caller must refuse and REPORT
                ``blocking``.
      blocking  residents that could not be walked, each with a ``why``:
                the static locks, the in-flight, the thrash-floored, the
                unmeasurable. This is the spec's "refuse — report blocking
                residents", pre-assembled.
      spared    walked-then-dropped by the least-reaping pass. Diagnostic:
                naming who was SAVED is how the invariant is auditable in a log.
    """
    device: str
    need: int
    victims: list = field(default_factory=list)
    freed: int = 0
    enough: bool = False
    blocking: list = field(default_factory=list)
    spared: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"device": self.device, "need": self.need,
                "victims": list(self.victims), "freed": self.freed,
                "enough": self.enough, "blocking": list(self.blocking),
                "spared": list(self.spared)}


def sort_key(r: EvictUnit, device: str, now: float) -> tuple:
    """THE lexicographic sort key — box 2, verbatim, and the ONE definition.

    Every eviction site imports this rather than spelling a tuple, because
    three hand-written copies of one key is precisely how Parity was lost
    before. Do not inline it.

      ① pref == other device first  -> 0 for mismatched, 1 for matched. The
         cliff order: a mismatched resident is already off the cliff by design;
         a matched one loses 135->36 tok/s, so it sorts LAST.
      ② time since last call, LONGEST first -> negated idle-anchor epoch, so
         the oldest anchor (smallest epoch) sorts first. ``now`` is passed in,
         never read, so both sides measure from the same instant.
      ③ total calls, FEWEST first.
      ④ model_key — stable, total, and deterministic across processes.

    NOTE what is deliberately ABSENT: size. The old key's ``-bytes``
    (largest-first) is replaced by the walk-then-drop pass, which is what makes
    least-reaping possible without ever reaching past the frontier.
    """
    matched = 1 if str(r.pref or "").strip().lower() == str(device).strip().lower() else 0
    return (matched, r.idle_anchor(), int(r.calls or 0), str(r.model_key))


# ── MEASURED DECODE RATE (operator, 2026-07-25) ──────────────────────────────
# "in the end it is about maximizing tok/s ... lets start recording this".
#
# ⚠ READ THIS BEFORE ASSUMING IT IS PART OF THE SORT: it is NOT. ``sort_key``
# above is unchanged and takes no tok/s input. This parser exists only so the
# ONE ledger can ACCUMULATE a decode-rate history that a future policy might
# rank on. Recording changes no victim, no placement, no order.
#
# It lives in THIS module — rather than beside the central-side writers — for
# the same reason ``sort_key`` does: eviction.py is the parity substrate both
# central's preview and the worker's auto-evict already import. A parser that
# only central could reach would be the first crack in "one ledger, one
# definition", which is precisely the failure this module was created to end.

# Upper sanity bound on a decode rate. Real GGUF decode on this fleet's cards
# tops out in the low hundreds of tok/s (a tiny model on a 3090 ~250); 5000
# leaves generous headroom for a faster card while still rejecting the
# degenerate sub-ms-window artifact (the VL-3B 240k-tok/s poison). A rate above
# this is a measurement error, not a fast model.
_MAX_PLAUSIBLE_TOK_S: float = float(
    os.environ.get("HUGPY_MAX_PLAUSIBLE_TOK_S", "5000") or 5000)


def tok_s_from_timings(payload: Any) -> Optional[float]:
    """The engine's OWN decode rate from a llama-server reply, or None.

    llama-server returns a ``timings`` block on EVERY completion, and
    ``predicted_per_second`` is authoritative decode tok/s measured inside the
    engine. It costs nothing to keep: no instrumentation in the serving path, no
    wall-clock arithmetic of ours (which would fold in queueing, network and
    prompt-eval time and would not be decode rate at all). It was simply being
    thrown away.

    Verified against llama.cpp ``tools/server/server-task.cpp``: present on the
    one-shot response body AND pushed onto the FINAL streaming chunk
    unconditionally, so both transports can be read the same way.

    Falls back to ``predicted_n / predicted_ms`` for a build that reports the
    raw counters but not the derived rate.

    Returns None for ANYTHING it does not fully recognize — a missing block, a
    non-dict, a non-numeric, a zero-token generation, or an IMPLAUSIBLE rate. The
    serving path is live; degrade-not-guess, and never raise.

    Implausible-rate guard: a degenerate ``predicted_ms`` (a vision/short reply
    that reports a sub-millisecond decode window for a handful of tokens) yields
    a physically-impossible rate — the VL-3B row that read 240k tok/s and then
    EMA-poisoned the estimate. No GGUF on this fleet decodes faster than
    ``_MAX_PLAUSIBLE_TOK_S``; anything above it is a bad measurement, dropped
    (None) rather than recorded, so it never enters the EMA. Absent beats wrong.
    """
    if not isinstance(payload, dict):
        return None
    t = payload.get("timings")
    if not isinstance(t, dict):
        return None
    try:
        rate = t.get("predicted_per_second")
        if rate is not None:
            f = float(rate)
            if not (f > 0.0 and f == f and f != float("inf")):
                return None
            return f if f <= _MAX_PLAUSIBLE_TOK_S else None
        n, ms = t.get("predicted_n"), t.get("predicted_ms")
        if n is None or ms is None:
            return None
        n_f, ms_f = float(n), float(ms)
        # predicted_n == 0 is a generation that produced no tokens: it reports
        # nothing about decode speed, so it is absent rather than 0 tok/s.
        if n_f <= 0 or ms_f <= 0:
            return None
        derived = (n_f * 1000.0) / ms_f
        return derived if derived <= _MAX_PLAUSIBLE_TOK_S else None
    except (TypeError, ValueError):
        return None


def _partition(pool: "Iterable[EvictUnit]", *, now: float) -> "tuple[list, list]":
    """Split residents into (walkable, blocking-with-a-why).

    The spec's pool is "residents on d, minus 🔒static". Exactly two further
    exclusions survive, each named in the blocking report so a refusal explains
    itself:
      * mid_generation  — enacted proposal 1
      * unmeasurable    — degrade-not-guess (never free an unknown amount)

    There is deliberately NO time-based exclusion (the 300s thrash floor was
    retired 2026-07-27 — see the module docstring). ``now`` is still taken
    because the ranking downstream is time-aware; nothing here vetoes on age.
    """
    walkable: list = []
    blocking: list = []
    for r in pool:
        if r.static:
            blocking.append({"model_key": r.model_key, "bytes": r.bytes,
                             "why": r.why or "static (locked residency)"})
            continue
        if r.mid_generation:
            # ENACTED PROPOSAL 1 (spec "Open"): unevictable regardless of rank.
            blocking.append({"model_key": r.model_key, "bytes": r.bytes,
                             "why": "in flight (mid-generation)"})
            continue
        try:
            b = int(r.bytes)          # type: ignore[arg-type]
        except (TypeError, ValueError):
            b = 0
        if b <= 0:
            # DEGRADE-NOT-GUESS: an occupant we cannot size. Evicting it would
            # free an unknown amount, so the plan cannot be verified against
            # `need`. Report it as blocking; never walk it.
            blocking.append({"model_key": r.model_key, "bytes": r.bytes,
                             "why": "unmeasurable footprint — not planned"})
            continue
        walkable.append(r)
    return walkable, blocking


def evict_plan(device: str, need: int, residents: "Iterable[EvictUnit]", *,
               now: float,
               least_reaping: bool = DEFAULT_LEAST_REAPING
               ) -> EvictPlan:
    """EVICT(device d, need n) — the ONE shared function. PURE.

    Both admission call sites (D and O) run this identically; so do central's
    preview and the worker's auto-evict. See the module docstring for the three
    invariants this upholds.

    ``now`` is REQUIRED and never defaulted to ``time.time()``: a default clock
    read is exactly how central and the worker would drift apart, and the spec
    names that as the failure mode Parity exists to prevent.

    ``least_reaping`` is PASSED IN, never read from the environment here,
    because this module is pure and a self-read would be a second clock. It
    changes THE DROP PASS, which every site runs — including central's
    ``storage_proposal``. If worker A ran it ON and central previewed it OFF,
    their victim lists would differ on the very fixture
    ``tests/test_eviction_parity.py`` exists to police. It is therefore
    FLEET-WIDE: one central setting, shipped to every worker on the heartbeat
    (the ``blocked_models`` idiom), never a per-worker key.

    There is no residency-floor argument: nothing here vetoes a victim on age
    (retired 2026-07-27, see the module docstring)."""
    need = max(0, int(need or 0))
    plan = EvictPlan(device=device, need=need)
    walkable, blocking = _partition(residents, now=now)
    plan.blocking = blocking
    if need <= 0:
        plan.enough = True
        return plan

    walkable.sort(key=lambda r: sort_key(r, device, now))

    # ── WALK: accumulate victims IN ORDER until freed >= n or pool exhausted ─
    walked: list = []
    freed = 0
    for r in walkable:
        if freed >= need:
            break
        walked.append(r)
        freed += int(r.bytes or 0)

    # ── DROP PASS, same order: remove any victim the REMAINING set covers ────
    # Least reaping. This ONLY removes — it never looks past `walked`, so the
    # frontier rule holds by construction: a hot resident beyond where the walk
    # reached cannot be pulled in, however conveniently sized it is.
    #
    # GATED (operator switch, 2026-07-25). least_reaping=False skips the pass
    # entirely — `kept` stays exactly `walked` and `spared` stays empty, which
    # is byte-identical to the pre-drop-pass greedy walk. The switch is the
    # ONLY difference; the WALK and the sort are untouched in both states, so
    # OFF is a strict superset of ON's victims and never a different ordering.
    kept = list(walked)
    if least_reaping:
        for r in list(walked):        # same order the walk produced
            remaining = sum(int(k.bytes or 0) for k in kept if k is not r)
            if remaining >= need:
                kept.remove(r)
                plan.spared.append(r.model_key)

    plan.victims = [r.model_key for r in kept]
    plan.freed = sum(int(r.bytes or 0) for r in kept)
    plan.enough = plan.freed >= need
    # NOT enough means everything walkable went and it still fell short. No
    # extra work is needed here: `blocking` was already populated by
    # `_partition` with the residents that could not be walked AND their
    # reasons, which is exactly what the spec's "refuse — report blocking
    # residents" terminal needs.
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Box 1 — admission & placement. The flowchart, executable.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Placement:
    """The verdict of ``plan_admission`` — box 1's terminal states.

      action    "place"    -> resident, honored (all of Z on D)
                "split"    -> split residency (what fits on D, remainder on O)
                "reject"   -> infeasible on this card (Z > X + Y)
                "refuse"   -> the devices could not be cleared; `blocking` says
                              who stood in the way
                "degrade"  -> an input was unmeasurable; the CALLER KEEPS
                              TODAY'S BEHAVIOUR (never a guessed eviction)
      on_device / on_other  bytes placed on D and on O.
      evict     the per-device EvictPlan(s) actually run, in call order.
    """
    action: str
    device: str = VRAM
    on_device: int = 0
    on_other: int = 0
    evict: list = field(default_factory=list)
    blocking: list = field(default_factory=list)
    note: str = ""

    @property
    def victims(self) -> list:
        """Every model this admission would unload, in the order chosen."""
        out: list = []
        for p in self.evict:
            out.extend(p.victims)
        return out

    def as_dict(self) -> dict:
        return {"action": self.action, "device": self.device,
                "on_device": self.on_device, "on_other": self.on_other,
                "evict": [p.as_dict() for p in self.evict],
                "victims": self.victims, "blocking": list(self.blocking),
                "note": self.note}


def plan_admission(size: "Optional[int]", mode: Any, *,
                   vram_free: "Optional[int]", ram_free: "Optional[int]",
                   vram_total: "Optional[int]" = None,
                   ram_total: "Optional[int]" = None,
                   residents: "Optional[Iterable[EvictUnit]]" = None,
                   now: float = 0.0,
                   ) -> Placement:
    """Box 1 of the spec — the SINGLE-POOL convenience form. PURE.

    Device occupancy is not a property of a EvictUnit (a model's ``pref`` says
    where it WANTS to live, not where it currently is), so the real entry point
    ``plan_admission_split`` takes the two device pools separately. This wrapper
    is for the common caller that only knows about ONE pool — the residents of
    the PREFERRED device — and it assigns ``residents`` to that pool, leaving
    the other empty.

    Consequence worth stating plainly: with an empty O pool, the second
    EVICT call site can free nothing, so an admission that needs room on the
    other device lands on ``split`` (if O has free space) or ``refuse``. That is
    correct for a caller who genuinely has no O-side inventory; a caller that
    does have one MUST use ``plan_admission_split`` or it will under-report what
    could be reclaimed.

    DEGRADE-NOT-GUESS: unknown size, or unknown free on the device we must
    place on, returns ``action="degrade"``. The caller then does exactly what
    it does today — never a guessed eviction."""
    return plan_admission_split(
        size, mode, vram_free=vram_free, ram_free=ram_free,
        vram_total=vram_total, ram_total=ram_total,
        vram_residents=(residents if preferred_device(mode) == VRAM else None),
        ram_residents=(residents if preferred_device(mode) == RAM else None),
        now=now)


def plan_admission_split(size: "Optional[int]", mode: Any, *,
                         vram_free: "Optional[int]", ram_free: "Optional[int]",
                         vram_total: "Optional[int]" = None,
                         ram_total: "Optional[int]" = None,
                         vram_residents: "Optional[Iterable[EvictUnit]]" = None,
                         ram_residents: "Optional[Iterable[EvictUnit]]" = None,
                         now: float = 0.0,
                         ) -> Placement:
    """Box 1 with the two device pools passed explicitly. THE real entry point.

    Walks the flowchart exactly:

        Z > X + Y ?                       -> reject
        D := P's device;  O := the other
        D_free >= Z ?                     -> place all of Z on D
        else EVICT(D, Z - D_free); enough? -> place all of Z on D
             else place what fits on D; R := Z - placed
                  O_free >= R ?            -> place R on O (split)
                  else EVICT(O, R - O_free); enough? -> place R on O
                       else refuse, reporting the blocking residents
    """
    def _i(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    Z = _i(size)
    device = preferred_device(mode)
    other = other_device(device)
    if Z is None or Z <= 0:
        return Placement(action="degrade", device=device,
                         note="unknown model size — keeping today's behaviour "
                              "(degrade-not-guess)")

    free = {VRAM: _i(vram_free), RAM: _i(ram_free)}
    total = {VRAM: _i(vram_total), RAM: _i(ram_total)}
    pools = {VRAM: list(vram_residents or []), RAM: list(ram_residents or [])}

    # Z > X + Y ? -> reject, infeasible on this card. Only when BOTH totals are
    # measured: a single unknown total cannot support a rejection (a rejection
    # from a guess takes a working model out of the pool, which is strictly
    # worse than a late honest refusal).
    if total[VRAM] is not None and total[RAM] is not None:
        if Z > total[VRAM] + total[RAM]:
            return Placement(action="reject", device=device,
                             note=(f"infeasible on this card: {Z} bytes exceeds "
                                   f"VRAM+RAM ({total[VRAM]}+{total[RAM]})"))

    if free[device] is None:
        return Placement(action="degrade", device=device,
                         note=f"unknown free {device} — keeping today's "
                              "behaviour (degrade-not-guess)")

    # D_free >= Z ? -> place all of Z on D, resident · honored.
    if free[device] >= Z:
        return Placement(action="place", device=device, on_device=Z,
                         note=f"fits {device} free")

    out = Placement(action="place", device=device)

    # EVICT(D, need = Z - D_free)
    p1 = evict_plan(device, Z - free[device], pools[device], now=now)
    out.evict.append(p1)
    if p1.enough:
        out.on_device = Z
        out.note = f"evicted {len(p1.victims)} on {device} to seat all of Z"
        return out

    # Place what fits on D; R := Z - placed. FULL UNLOAD means `freed` is the
    # victims' own sizes, so `placed` is derivable from the inputs alone.
    placed = free[device] + p1.freed
    R = Z - placed
    out.on_device = placed

    if free[other] is None:
        out.action = "degrade"
        out.note = (f"unknown free {other} — cannot plan the remainder; "
                    "keeping today's behaviour")
        return out

    # O_free >= R ? -> place R on O (split residency).
    if free[other] >= R:
        out.action = "split"
        out.on_other = R
        out.note = (f"split residency: {placed} on {device}, {R} on {other}")
        return out

    # EVICT(O, need = R - O_free)
    p2 = evict_plan(other, R - free[other], pools[other], now=now)
    out.evict.append(p2)
    if p2.enough:
        out.action = "split"
        out.on_other = R
        out.note = (f"split residency after evicting {len(p2.victims)} on "
                    f"{other}: {placed} on {device}, {R} on {other}")
        return out

    # Refuse — REPORT THE BLOCKING RESIDENTS. Both devices' blockers, because
    # the operator's question is "what is holding my card", not "which of the
    # two sub-steps failed".
    out.action = "refuse"
    out.on_device = 0
    out.on_other = 0
    out.blocking = list(p1.blocking) + list(p2.blocking)
    out.note = (f"refused: needed {Z}, could seat {placed} on {device} and "
                f"{free[other] + p2.freed} on {other}")
    return out
