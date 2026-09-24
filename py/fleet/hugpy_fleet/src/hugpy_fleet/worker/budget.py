"""Call-driven storage budget for the worker: evict-to-fit, or REFUSE the pull.

WHY THIS EXISTS (incident, 2026-07-16): the operator's workstation ("op") filled
to 0 bytes free. ``provision.py`` had NO disk checks at all — it downloaded until
the kernel returned [Errno 28], leaving a wedged partial pull on a full disk.

THE OPERATOR'S DESIGN (verbatim):
  1. "the total space from those modules, if the total space of allocated drive
     space from central is taken up, then fifo the models"
  2. "yes fifo it, remove an existing model and install the one that is being
     called"
  3. "and yes, refuse it if it wont, show it as missing, hover info why"

So: **the model being CALLED always wins.** On a pull, if the worker's model
total would exceed its central-allocated budget, evict oldest-first (FIFO) to
make room, then install the caller. ONLY if a full FIFO of every reclaimable
candidate still can't free enough do we REFUSE — before a single byte is
downloaded — and report the model as MISSING with an honest reason.

── HOW THIS RELATES TO THE OPERATOR-GATED REAPER (read before changing) ────────
This is a SEPARATE, NARROW path. It is NOT a second eviction policy and NOT a
background sweep:

  * ``/reap`` + ``/reap-approve`` (flask_app/.../worker_routes.py) remain the
    BULK, operator-in-the-loop path: a human reads the proposal and approves a
    subset. That route's guarantee — "nothing deletes except through this
    explicit, operator-gated, audited call" — was scoped to THAT route's own
    bulk-reclaim flow, and it still holds for it: this module never calls it,
    never auto-approves it, and never widens what it may delete.
  * THIS path fires ONLY to make room for a model ACTIVELY BEING PROVISIONED
    (call-driven), never on a timer, never as a sweep, and it evicts the MINIMUM
    prefix of the FIFO order needed to seat the caller. No call -> no delete.

Both paths funnel into the SAME single delete choke point (``wipe_model``, which
is path-jailed and re-proves the shared/central-store gate), so neither can
delete something the other wouldn't.

── ORDERING + GUARDS ARE REUSED, NOT REINVENTED ────────────────────────────────
The FIFO order and the candidate domain deliberately mirror the two proven
implementations already in the tree, so a third divergent policy never exists:

  * ``utils/workers.py:storage_proposal`` — central's read-only preview: the
    candidate domain is UNPROTECTED models only, sorted ASCENDING by
    ``last_picked`` (oldest-first), greedily accumulated until ``need`` is
    covered. This module produces the SAME order over the SAME domain, so what
    the console previews is what an auto-evict would actually take.
  * ``managers/serve/model_cache.py:evict_for(need_bytes, keep_dir)`` — the hot
    cache's LRU-evict-until-fits loop, including the ``keep_dir`` exclusion that
    stops a warm from evicting the very entry it is warming.

We MIRROR rather than import ``evict_for``: it is bound to the hot-cache tier
(its own CACHE_DIR/CACHE_MAX_BYTES globals, mtime as the LRU key, whole-dir
rmtree). This tier is different in every one of those inputs — the model root,
a central-allocated budget, central's ``last_picked`` as the FIFO key, and
``wipe_model`` as the only permitted delete. Reusing it would mean rewiring the
hot cache around parameters it doesn't have. The SEMANTICS are copied exactly:
oldest-first, stop as soon as it fits, never touch the keep target.

── FIFO KEY ────────────────────────────────────────────────────────────────────
``last_picked`` (when central was last asked to serve this model on this box) is
the operator's "oldest" — a model nobody has called in weeks is the right thing
to drop for one being called RIGHT NOW. Central owns that clock and ships it in
the assignment payload; a model central has never served has no entry and sorts
as 0 = coldest = evicted first (exactly right for never-served test leftovers).
When central hasn't shipped the map at all, we fall back to on-disk mtime so the
order is still oldest-first rather than arbitrary.

── 📌 PIN + ALLOCATION HAVE NO BEARING ON EVICTION (operator, 2026-07-17) ───────
The canonical statement (verbatim): "the pins only should designate that the
model allocation survives restarts. the allocation only stipulates the routing
for that model (to that worker). neither of those should have any bearing on the
pull or eviction, unless its to do with priority, then a pinned model should
take higher precidence than unpinned, but even that is trivial".

So in THIS module: a pinned or assigned model is a normal eviction CANDIDATE
(see _is_protected). Evicting its files leaves the pin + allocation untouched —
routing survives and the bytes re-pull on the next call. Pin's ONLY eviction
role is the trivial FIFO tiebreak in fit_plan: among equally-stale candidates,
unpinned evict first. Only 🔒static promises local presence and is protected;
loaded/loading/provisioning are protected as live-use guards. This is the
day-one tripwire the operator called out — conflating attribution/routing with a
disk shield filled his workstation to 0 bytes free on 2026-07-16.
"""
from __future__ import annotations

import os
import time as _time
import logging

logger = logging.getLogger("hugpy_fleet.worker.budget")

# Protection reasons that make a model INELIGIBLE for auto-eviction. Mirrors
# storage_proposal's chain (utils/workers.py) minus `assigned` AND `pinned` —
# see _is_protected for why NEITHER protects here.
#   * `assigned` = attribution/routing only (lazy-download doctrine).
#   * `pinned`   = the ALLOCATION survives restarts, nothing else (operator,
#     2026-07-17). Pin has NO bearing on eviction — a pinned model's files are a
#     normal LRU candidate; evicting them leaves pin + routing untouched and the
#     bytes re-pull on next call.
# Only 🔒static (durable local-presence promise) and the live-use guards
# (loaded/loading/provisioning — deleting under a live pull corrupts the fetch)
# stay protected.
_PROTECTED_REASONS = ("static", "loaded", "loading", "provisioning")


class BudgetRefusal(Exception):
    """A pull that cannot fit even after a full permissible FIFO.

    Raised BEFORE any bytes are downloaded. Carries the machine-readable
    ``reason`` dict the heartbeat/console render on hover, so the model reads as
    MISSING-with-a-reason rather than a stalled pull.
    """

    def __init__(self, reason: dict):
        self.reason = reason
        super().__init__(reason.get("reason") or "won't fit")


from hugpy_platform.formatting import human_bytes as _human


def cap_bytes(limits: dict | None) -> int | None:
    """The worker's EXPLICIT storage allocation in bytes, or None if unset.

    ``limits['disk_cache_gib']`` is central's per-worker allocation (the
    operator sets it; the box may tighten it via HUGPY_DISK_CACHE_MAX_GIB, which
    central already clamps against in _clamp_limits). Returns None when it is
    absent/blank/unparseable/non-positive — i.e. "no allocation declared".

    D (budget must be real): None is a FIRST-CLASS answer, not a zero. An unset
    allocation means this box has no declared model-cache ceiling, and the
    auto-evict path treats that as "don't evict" rather than inventing one. See
    fit_plan for why the free-disk reserve is NOT used as a fallback budget here.
    """
    if not isinstance(limits, dict):
        return None
    raw = limits.get("disk_cache_gib")
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    return int(val * (1 << 30))


def _drive_id(path: str):
    """Filesystem device id (st_dev) of the drive holding ``path`` (or its
    nearest existing parent), or None if it can't be resolved.

    st_dev is THE reliable same-drive signal (slice 4): a symlinked or
    bind-mounted hot root that points at the store root's own filesystem shares
    its st_dev, where a realpath-prefix compare would wrongly call them separate
    drives (or miss a symlink entirely). Two roots with the same st_dev are the
    same physical drive → their byte budgets contend for the same space and the
    stricter one must govern. A different st_dev is a genuinely different drive
    whose budget must NOT drag into the store drive's min."""
    p = path or ""
    for _ in range(64):
        try:
            return os.stat(p).st_dev
        except OSError:
            parent = os.path.dirname(p)
            if not parent or parent == p:
                return None
            p = parent
    return None


def worker_declared_caps(store_root: str) -> list[tuple[str, int]]:
    """Every WORKER-declared byte budget that governs the same drive as
    ``store_root``, as ``[(source_name, cap_bytes), ...]``.

    The operator's min-wins ruling (2026-07-17): "worker designation beats out
    central, UNLESS central designation for that worker's drive is lower than the
    worker designation" → the effective per-drive cap is the MIN across central's
    disk_cache_gib and the worker's own declarations that sit on the SAME drive.
    "the real issue for the workers is overcrowding of the HDD."

    The worker's own tiers (knob map, slice 4):
      * hot_cache (managers/serve/hot_cache.py): root HUGPY_HOT_CACHE_ROOT,
        budget HUGPY_HOT_CACHE_GIB (default 225). ae's ACTIVE tier — 1500 GiB.
      * model_cache (managers/serve/model_cache.py): dir HUGPY_MODEL_CACHE
        (default /var/cache/hugpy-models), budget HUGPY_MODEL_CACHE_MAX_GIB
        (default 450) → CACHE_MAX_BYTES.
    (HUGPY_DISK_CACHE_MAX_GIB is NOT here — it is projected into central's
    disk_cache_gib by the worker's own _clamp_limits, so it already rides
    cap_bytes; adding it here would double-count.)

    A tier contributes ONLY when its root shares the store root's drive (same
    st_dev): a hot tier on a genuinely different NVMe has its own space and must
    not constrain the store drive. Missing/unset knobs contribute nothing.
    Best-effort: any resolution failure just omits that tier."""
    out: list[tuple[str, int]] = []
    store_dev = _drive_id(store_root) if store_root else None
    if store_dev is None:
        return out
    GIB = 1 << 30

    def _consider(source: str, root: str, gib_env: str, default_gib: float | None):
        if not root:
            return
        try:
            if _drive_id(root) != store_dev:
                return                       # different physical drive — skip
        except Exception:  # noqa: BLE001
            return
        raw = os.environ.get(gib_env)
        if raw in (None, ""):
            if default_gib is None:
                return                       # tier active but no explicit cap
            gib = default_gib
        else:
            try:
                gib = float(raw)
            except (TypeError, ValueError):
                return
        if gib <= 0:
            return
        out.append((source, int(gib * GIB)))

    # hot_cache: only when a root is configured (unset root == tier disabled).
    hot_root = (os.environ.get("HUGPY_HOT_CACHE_ROOT") or "").strip()
    _consider("worker_hot_cache_gib", hot_root, "HUGPY_HOT_CACHE_GIB", 225.0)
    # model_cache: "on" only when its dir is a real, WRITABLE dir (mirrors
    # model_cache.enabled() — an unwritable/absent CACHE_DIR disables the tier).
    # Its default cap (450) then applies when it shares the store drive.
    mc_dir = (os.environ.get("HUGPY_MODEL_CACHE") or "/var/cache/hugpy-models").strip()
    try:
        mc_on = bool(mc_dir) and os.path.isdir(mc_dir) and os.access(mc_dir, os.W_OK)
    except OSError:
        mc_on = False
    if mc_on:
        _consider("worker_model_cache_gib", mc_dir, "HUGPY_MODEL_CACHE_MAX_GIB", 450.0)
    return out


def resolve_effective_cap(limits: dict | None,
                          store_root: str = "") -> tuple[int | None, dict]:
    """The EFFECTIVE per-drive cap (bytes) and a machine-readable source map.

    Min-wins over {central disk_cache_gib, worker same-drive declarations}. Any
    term optional; NONE declared → (None, sources) meaning unmanaged — decision D
    is unchanged. ``sources`` always names every contributing term (in GiB) so
    the operator can see WHY a number governs, e.g.
    ``{"central_gib": 400, "worker_hot_cache_gib": 1500}`` → effective 400.

    Pure w.r.t. its inputs EXCEPT it reads env + stats the drive (that is what
    makes it the impure resolver the pure fit_plan is handed the RESULT of)."""
    GIB = 1 << 30
    terms: list[tuple[str, int]] = []
    sources: dict = {}
    central = cap_bytes(limits)              # central disk_cache_gib -> bytes|None
    if central is not None:
        terms.append(("central_gib", central))
        sources["central_gib"] = round(central / GIB, 3)
    try:
        for name, cap in worker_declared_caps(store_root):
            terms.append((name, cap))
            sources[name] = round(cap / GIB, 3)
    except Exception:  # noqa: BLE001 — a knob probe must never break the pull
        pass
    if not terms:
        return None, sources
    name, alloc = min(terms, key=lambda t: t[1])
    # THE ALLOCATION IS THE TOTAL LIMIT (operator rule, 2026-08-22): the
    # inference headroom/reserve is portioned OUT OF the allocation, not added
    # beside it. effective = allocation - reserve is the ceiling the cache may
    # actually reach; "available space + headroom = the limit".
    reserve = disk_reserve_bytes(limits)
    eff = max(0, alloc - reserve)
    sources["allocation_gib"] = round(alloc / GIB, 3)
    sources["allocation_source"] = name
    sources["reserve_gib"] = round(reserve / GIB, 3)
    sources["effective_gib"] = round(eff / GIB, 3)
    sources["effective_source"] = name
    return eff, sources


def disk_reserve_bytes(limits: dict | None = None) -> int:
    """Free-space reserve (bytes) to keep on the model-root volume after a pull.

    Resolution order: per-worker ``limits.disk_reserve_gib`` (central-set, same
    dict that carries disk_cache_gib) -> ``HUGPY_WORKER_DISK_RESERVE_GIB`` env ->
    50. The reserve is carved OUT of the disk_cache_gib allocation (see
    resolve_effective_cap), so it is also the inference headroom.

    Central imports this function as ``_disk_reserve_bytes`` — the same env var
    and default give the worker's disk-free floor the same value as central's
    display budget. Sized to comfortably exceed the
    largest single pull (~45 GiB) so provisioning never drives a volume to
    [Errno 28]. Override with ``HUGPY_WORKER_DISK_RESERVE_GIB`` (default 50).

    This is the ONLY floor that applies on a shared/central store: the per-worker
    CAP is a category error there (that volume is centrally managed), but a full
    volume still corrupts everyone's pull — the op incident — so the disk-free
    check stays. Do NOT invent a second reserve constant; this is the tree's one.
    """
    gib = None
    v = (limits or {}).get("disk_reserve_gib") if isinstance(limits, dict) else None
    if v not in (None, ""):
        try:
            gib = float(v)
        except (TypeError, ValueError):
            gib = None
    if gib is None:
        try:
            gib = float(os.environ.get("HUGPY_WORKER_DISK_RESERVE_GIB", "50"))
        except (TypeError, ValueError):
            gib = 50.0
    if gib < 0:
        gib = 0.0
    return int(gib * (1 << 30))


def _is_protected(row: dict) -> str:
    """The reason ``row`` may NOT be auto-evicted, or "" if it is a candidate.

    Domain mirrors storage_proposal's guard chain with TWO deliberate
    differences: NEITHER ``assigned`` NOR ``pinned`` protects here.

    CANONICAL STATEMENT (operator ruling, 2026-07-17): "the pins only should
    designate that the model allocation survives restarts. the allocation only
    stipulates the routing for that model (to that worker). neither of those
    should have any bearing on the pull or eviction".
      * 📌 pin = the model's ALLOCATION survives restarts (and unassign
        attempts). Nothing else.
      * Allocation = ROUTING: which worker answers for that model.
      * NEITHER has any bearing on the pull (already true — lazy download,
        7f0e6e8/2a3baeb) NOR on eviction (this rule). Evicting a pinned or
        assigned model's FILES never touches its pin or allocation — routing
        survives, bytes re-pull on next call. A row whose ONLY claim is pinned
        (or assigned) is a CANDIDATE.

    This closes the day-one tripwire the operator called out: assignment and pin
    are attribution/routing, not a disk shield. On a box whose models are all
    assigned (the normal case, and op's actual case) they must all be reclaimable
    or the disk fills — exactly the 2026-07-16 incident.

    Only the DURABLE local-presence promise and the live-use guards protect:
    🔒static (the ONE tier that promises the files stay local), plus
    loaded/loading/provisioning (deleting under a live pull corrupts the fetch;
    deleting a loaded model breaks serving). A pinned/assigned model that is ALSO
    static/loaded keeps protection through THAT flag — never through pin.
    """
    if row.get("protected") and (row.get("why") or "") not in ("assigned", ""):
        # Worker-side flag already decided (e.g. "shared/central storage —
        # never reaped", "model store not marked reapable"). Trust it, except
        # for a bare `assigned`, which is attribution only (see above).
        return str(row.get("why") or "protected")
    for flag in _PROTECTED_REASONS:
        if row.get(flag):
            return flag
    return ""


def _allocation_clause(allocated: dict | None, cap: int) -> tuple[str, dict]:
    """The ALLOCATION-LEVEL half of a refusal: is the ASSIGNED SET itself too big?

    OPERATOR (2026-07-16): "it should also show how much is needed based on the
    total size of all models allocated". The per-pull numbers answer "can THIS
    pull fit". This answers the more useful question: the deficit is STRUCTURAL
    — the assignment set cannot fit at ANY eviction order, so no call will ever
    be lucky. Without it the operator reads a refusal as this-pull-was-unlucky
    and re-tries forever.

    Returns ``(text, fields)``. Central sizes the set (it owns the manifest) and
    ships the totals in the heartbeat reply; this is a pure read of that answer.

    HONESTY RULES (a silent 0 makes an over-subscribed set look fine):
      * no totals yet (pre-first-beat / older central) -> ("", {}). SAY NOTHING
        rather than claim a 0 GiB allocation.
      * unknown-size models are COUNTED and NAMED in the text ("N unknown"), and
        the total is then a FLOOR — "≥" says so rather than implying precision.
    """
    if not isinstance(allocated, dict) or allocated.get("allocated_count") is None:
        return "", {}
    total = int(allocated.get("allocated_total_bytes") or 0)
    count = int(allocated.get("allocated_count") or 0)
    unknown = int(allocated.get("allocated_unknown_count") or 0)
    over = max(0, total - cap)
    fields = {
        "allocated_total_bytes": total,
        "allocated_count": count,
        "allocated_unknown_count": unknown,
        "allocated_over_budget_bytes": over,
    }
    if not count:
        return "", fields
    approx = "≥" if unknown else ""
    text = f" — assigned set ({count}) totals {approx}{_human(total)}"
    if unknown:
        text += f", {unknown} unknown"
    if over:
        text += f" ({_human(over)} over budget)"
    return text, fields


def _least_reaping() -> bool:
    """The fleet drop-pass policy as adopted onto this worker's env.

    Mirrors ``agent._evict_least_reaping`` exactly (same env, same parsing) but
    is defined here so budget.py never imports agent.py. The env is the shared
    contract between them: ``agent._adopt_least_reaping`` writes it from the
    heartbeat, ``_apply_settings_env`` projects any local setting onto it, and
    both readers agree on what an absent/blank value means."""
    from hugpy_engine.eviction import DEFAULT_LEAST_REAPING
    raw = os.environ.get("HUGPY_EVICT_LEAST_REAPING")
    if raw in (None, ""):
        return DEFAULT_LEAST_REAPING
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def fit_plan(model_key: str, need_bytes: int, storage: dict,
             limits: dict | None, last_picked: dict | None = None,
             allocated: dict | None = None, shared_store: bool = False,
             effective_cap: int | None = None,
             budget_sources: dict | None = None,
             call_stats: dict | None = None,
             model_modes: dict | None = None,
             now: float | None = None) -> dict:
    """Decide how to seat ``need_bytes`` of ``model_key`` under the budget.

    PURE — computes, never deletes. The caller (evict_to_fit) executes it. Being
    pure is what lets the tests assert the ORDER and the REFUSAL without a disk.

    ``effective_cap`` / ``budget_sources`` (slice 4, min-wins) — the EFFECTIVE
    per-drive cap in bytes and its source map, resolved by the impure caller
    (resolve_effective_cap) as ``min`` over {central disk_cache_gib, worker
    same-drive declarations}. When ``effective_cap`` is None the cap falls back
    to ``cap_bytes(limits)`` (central only) — so every existing caller/test is
    byte-identical. ``budget_sources`` is reported verbatim in the plan/refusal so
    the operator can see WHY a number governs (e.g. central 400 wins over hot
    1500). The verdict is a function of this ONE effective number; passing it in
    (rather than reading env here) keeps fit_plan pure.

    ``allocated`` — central's ALLOCATION-LEVEL totals for this worker's
    assignment set (``{allocated_total_bytes, allocated_count,
    allocated_unknown_count}``, adopted from the heartbeat reply). Optional and
    purely ADDITIVE: it changes no decision, only what a refusal REPORTS. The
    fit verdict stays a function of real bytes on real disk.

    ``shared_store`` — True when this box's model root IS the shared/central
    catalog (ae on the NAS: DEFAULT_ROOT == the 13T fleet volume). The caller
    (evict_to_fit) resolves this via provision._on_shared_model_store. On such a
    box the per-worker CAP does NOT apply: the cap exists to stop a WORKER'S OWN
    drive filling up, but a shared/central volume is centrally managed (central's
    own transfer/budget gates + the operator-gated reaper govern it), and its
    measured cache_used is the WHOLE FLEET'S resident catalog, not this box's
    consumption — so ``used > cap`` is permanently true and every pull would
    refuse forever (the ae defect). Skip cap logic entirely, evict NOTHING (those
    files are the fleet's source of truth — never deletable from here), and keep
    only the DISK-FREE floor so a genuinely full volume still refuses honestly
    (the op [Errno 28] incident) rather than being written to failure.

    Returns::

        {"action": "proceed"|"evict"|"refuse",
         "evict": [model_key, ...],        # FIFO order, oldest-first
         "reason": {...} | None,           # machine-readable, only on refuse
         "note": "..."?}                   # why a gate was/ wasn't applied

    Decision table:
      * shared/central store       -> "proceed" (disk-free permitting), no evict
      * no explicit cap            -> "proceed" (D: unset != evict everything)
      * fits under the cap as-is   -> "proceed", nothing evicted
      * fits after evicting a FIFO prefix of reclaimable candidates -> "evict"
      * even a FULL FIFO can't free enough -> "refuse" (+ an honest reason)
    """
    # EFFECTIVE cap (slice 4): the min-wins number the impure caller resolved.
    # Fall back to central-only cap_bytes when the caller didn't supply one, so
    # every legacy caller/test is unchanged. budget_sources is carried through to
    # the plan/refusal verbatim for operator visibility.
    # ``effective_cap`` from resolve_effective_cap is ALREADY allocation-minus-
    # reserve. The central-only fallback applies the same carve-out here so the
    # two paths (and central's over_budget) agree on one ceiling (Parity).
    cap = effective_cap
    srcs = dict(budget_sources or {})
    if cap is None:
        alloc = cap_bytes(limits)
        if alloc is not None:
            reserve = disk_reserve_bytes(limits)
            cap = max(0, alloc - reserve)
            srcs.setdefault("allocation_gib", round(alloc / (1 << 30), 3))
            srcs.setdefault("reserve_gib", round(reserve / (1 << 30), 3))
            srcs.setdefault("effective_gib", round(cap / (1 << 30), 3))
    rows = [r for r in (storage.get("models") or [])
            if isinstance(r, dict) and r.get("model_key")]
    used = int(storage.get("cache_used_bytes") or 0)
    need = max(0, int(need_bytes or 0))

    # The model being provisioned is the KEEP TARGET (model_cache.evict_for's
    # keep_dir exclusion, by key rather than by path): never evict the thing we
    # are making room for. Bytes it ALREADY has on disk (a resumed/partial pull)
    # are counted as headroom it doesn't need to re-take.
    have = 0
    for r in rows:
        if r["model_key"] == model_key:
            # k60: only bytes on a REAPABLE store are headroom this pull already
            # holds. A read-through copy on the shared catalog is not — the pull
            # still has to land its own copy on the worker's own drive, and
            # `used` no longer counts the shared bytes either, so crediting them
            # here would under-state the delta by the same amount.
            if r.get("counts_toward_budget", True):
                have = int(r.get("bytes") or 0)
            break
    delta = max(0, need - have)          # NEW bytes this pull will add

    # ── SHARED/CENTRAL STORE: the worker cap is not applicable ──────────────
    # Skip cap+FIFO entirely. Never evict (shared files are the fleet SoT). Keep
    # ONLY the disk-free floor: if the NEW bytes won't fit under the reserve on
    # the real volume, refuse honestly — a full shared volume must not be driven
    # to [Errno 28] (the op incident), and central/console read the reason.
    if shared_store:
        free = int(storage.get("disk_free") or 0)
        reserve = disk_reserve_bytes(limits)
        note = ("store is shared/central — worker cap gate not applicable; "
                "volume is centrally managed")
        if delta and free and (free - delta) < reserve:
            reason = {
                "state": "refused",
                "model_key": model_key,
                "reason": (
                    f"won't fit on shared/central volume: needs {_human(delta)}, "
                    f"{_human(free)} free, {_human(reserve)} reserve kept — "
                    f"the worker cap does not apply here (centrally managed), but "
                    f"the volume is too full to land this pull safely"
                ),
                "needs_bytes": delta,
                "disk_free_bytes": free,
                "disk_reserve_bytes": reserve,
                "shared_store": True,
                "note": note,
            }
            return {"action": "refuse", "evict": [], "reason": reason}
        return {"action": "proceed", "evict": [], "reason": None, "note": note}

    # ── D: no explicit allocation -> no auto-eviction ───────────────────────
    # A worker with no disk_cache_gib has no declared ceiling, so there is no
    # honest way to say what is "over". Deliberately NOT falling back to the
    # free-disk reserve (which storage_proposal uses for its DISPLAY budget):
    # on a 100%-full drive, disk_free < reserve makes EVERYTHING read as
    # over-budget, and an auto path on that basis would evict model after model
    # on every call — thrash, and exactly the damage this fix exists to stop.
    # Unset therefore means "don't manage this box's storage": pull as before.
    # The operator sets disk_cache_gib to turn self-maintenance ON.
    if cap is None:
        return {"action": "proceed", "evict": [], "reason": None,
                "note": "no disk_cache_gib allocation set — budget unmanaged"}

    # ── TWO deficits, ONE eviction target: the CALLED MODEL WINS ────────────
    # (operator, 2026-08-31: "it NEEDS to be able to serve what's being called.
    # period" / "never refuse a call if the model at least exists in central").
    # A pull can be blocked two independent ways, and BOTH must be relieved by
    # FIFO eviction so the caller is SERVED rather than refused:
    #   * CAP deficit       — the cache would exceed its allocated ceiling.
    #   * DISK-FREE deficit — landing `delta` new bytes would drop the real
    #     volume below the reserve floor (the op 2026-07-16 [Errno 28] guard).
    # A model on THIS same volume frees its bytes from BOTH `used` and the disk
    # at once when evicted, so one `freed` total pays down whichever deficit
    # governs → the eviction target is the MAX of the two.
    #
    # THE BUG THIS FIXES: historically only the cap deficit triggered FIFO; a
    # disk-free block hard-REFUSED with no eviction attempt. So a drive whose
    # cache sat just under its ceiling could never self-heal — exactly ae's case
    # (cap 1030 + reserve 150 == the whole 1180 drive, zero slack): the cache
    # near its ceiling means free is permanently below the reserve, and because
    # `used+delta <= cap` stayed true, no oldest-model eviction ever fired and
    # every call refused forever. Routing the disk-free deficit into the SAME
    # FIFO makes the called model win: evict the oldest until it fits.
    free = int(storage.get("disk_free") or 0)
    reserve = disk_reserve_bytes(limits)
    cap_deficit = max(0, used + delta - cap)
    # Disk-free deficit only when free is KNOWN (>0): unknown free can't be
    # assessed, so invent no deficit (mirrors the old guard's `and free`).
    disk_deficit = max(0, (reserve + delta) - free) if (delta and free) else 0
    must_free = max(cap_deficit, disk_deficit)

    if must_free <= 0:
        return {"action": "proceed", "evict": [], "reason": None,
                "budget_effective_bytes": cap, "budget_sources": srcs}

    # ── over cap OR under the disk-free floor: FIFO the reclaimable ──────────
    # candidates, oldest first, until ``must_free`` (computed above as the max of
    # the cap and disk-free deficits) is covered.
    lp_map = last_picked or {}

    def _lp(mk, row):
        v = lp_map.get(mk)
        try:
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
        # No central clock for this key: fall back to the worker's own on-disk
        # mtime so the order stays oldest-first instead of arbitrary. 0.0 (never
        # served, no mtime) sorts coldest — evicted first, which is right for
        # never-called leftovers.
        try:
            v2 = row.get("mtime")
            return float(v2) if v2 is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    candidates = []
    blocked: dict[str, int] = {}
    for r in rows:
        mk = r["model_key"]
        if mk == model_key:
            continue                       # the keep target — never a candidate
        why = _is_protected(r)
        if why:
            blocked[why] = blocked.get(why, 0) + 1
            continue
        candidates.append((_lp(mk, r), int(r.get("bytes") or 0), mk))

    # ── THE SHARED EVICT FUNCTION (operator spec assets/evictionflow.html) ───
    # This used to sort `(last_picked, -bytes, model_key)` — oldest-first then
    # LARGEST-first — and greedily walk it. Both halves are replaced by
    # ``managers.eviction.evict_plan``: the spec's lexicographic key (pref
    # mismatch, idle, calls, key) plus walk-then-drop least-reaping. The key is
    # NOT spelled here; three hand-written copies of one tuple is exactly how
    # Parity was lost, so this site, storage_proposal, and hot_cache all import
    # the SAME function.
    #
    # THE DEVICE FOR THIS SITE IS THE DISK. Key ① ("pref == other device
    # first") is a residency concept; a model's alloc-mode preference names
    # VRAM or RAM, and NEITHER is the disk, so on this site every candidate is
    # equally "mismatched" and key ① is a constant — the order collapses to
    # ②/③/④ (idle, calls, key), which is the correct storage semantics. Passing
    # the disk as the device is what makes that degeneracy explicit and
    # DERIVED, rather than a special-cased branch that could drift.
    #
    # 📌pin is NOT in the key (operator ruling, 2026-07-25: "pin routing has
    # nothing to do with eviction... the only eviction protection is 'static'
    # residency"). Protection is decided solely by _is_protected() (🔒static),
    # whose verdict is folded in as ``static=True`` below.
    from hugpy_engine import eviction as _ev

    _stats = call_stats or {}
    _modes = model_modes or {}
    _now = float(now) if now is not None else _time.time()

    def _calls_for(mk: str) -> int:
        try:
            return int((_stats.get(mk) or {}).get("calls") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0

    # ``must_free`` was resolved above as max(cap_deficit, disk_deficit) — the
    # single eviction target that satisfies whichever floor (allocated cap or
    # physical disk-free reserve) is binding. Do NOT recompute it as the cap
    # deficit alone: that was the pre-2026-08-31 bug that let a disk-free block
    # refuse without evicting.
    reclaimable_total = sum(b for _lp_, b, _mk in candidates)

    # No residency floor here or anywhere else — the 300s anti-thrash veto was
    # retired 2026-07-27 (operator). This path never had one regardless: rows
    # carry mtime, not a load time, and a floor derived from mtime would protect
    # exactly the cold leftovers this budget exists to clear.
    _plan = _ev.evict_plan(
        "disk", must_free,
        [_ev.EvictUnit(model_key=mk, bytes=b,
                      pref=_ev.preferred_device(_modes.get(mk)),
                      last_call=(lp or None), calls=_calls_for(mk))
         for lp, b, mk in candidates],
        now=_now,
        # FLEET-WIDE drop-pass policy, adopted from central on the heartbeat
        # (agent._adopt_least_reaping projects it onto this env). Read from the
        # env rather than imported from agent.py to keep budget.py free of that
        # import cycle; the env IS the mechanism, exactly as it is for the
        # anti-thrash floor. Central's storage_proposal reads the same policy
        # from its own store, so this preview/execute pair stays in Parity.
        least_reaping=_least_reaping())
    evict: list[str] = list(_plan.victims)
    freed = int(_plan.freed)

    if freed < must_free:
        # ── B: REFUSE. Not even a full FIFO can seat this model. ────────────
        # Return BEFORE any download starts — the whole point: no more
        # 7%-wedged pulls that fill a disk.
        blocked_str = ", ".join(f"{n} {why}" for why, n in
                                sorted(blocked.items(), key=lambda kv: -kv[1]))
        # The ALLOCATION-LEVEL clause: per-pull numbers say "this pull won't
        # fit"; this says WHY it never will if the assigned set is itself
        # over-subscribed. Additive — it never changes the verdict above.
        alloc_text, alloc_fields = _allocation_clause(allocated, cap)
        # When the effective cap comes from a WORKER declaration beating central
        # (or vice-versa), name the winning source so the refusal is honest about
        # WHICH number governs — the operator's min-wins visibility requirement.
        eff_text = ""
        eff_src = srcs.get("effective_source")
        if eff_src and len(srcs) > 2:            # >1 real term contributed
            eff_text = f" [effective cap {_human(cap)} = min via {eff_src}]"
        # HONESTY about WHICH floor is binding: when the disk-free reserve is the
        # governing deficit (a physically full volume, not an over-cap cache),
        # say so — otherwise a "budget" line reads as a cap problem on a drive
        # that is simply out of physical room even after evicting everything.
        disk_text = ""
        if disk_deficit >= cap_deficit and disk_deficit > 0:
            disk_text = (f" — volume physically full: {_human(free)} free, "
                         f"{_human(reserve)} reserve kept")
        reason = {
            "state": "refused",
            "model_key": model_key,
            "reason": (
                f"won't fit: needs {_human(delta)}, budget {_human(cap)}, "
                f"{_human(reclaimable_total)} reclaimable"
                + (f" ({blocked_str})" if blocked_str else "")
                + alloc_text + eff_text + disk_text
            ),
            **alloc_fields,
            "needs_bytes": delta,
            "budget_bytes": cap,
            "budget_effective_bytes": cap,
            "budget_sources": srcs,
            "used_bytes": used,
            "disk_free_bytes": free,
            "disk_reserve_bytes": reserve,
            "cap_deficit_bytes": cap_deficit,
            "disk_deficit_bytes": disk_deficit,
            "must_free_bytes": must_free,
            "reclaimable_bytes": reclaimable_total,
            "reclaimable_count": len(candidates),
            "blocked": blocked,
            "shortfall_bytes": must_free - reclaimable_total,
        }
        return {"action": "refuse", "evict": [], "reason": reason}

    return {"action": "evict", "evict": evict, "reason": None,
            "freed_bytes": freed, "must_free_bytes": must_free,
            "budget_effective_bytes": cap, "budget_sources": srcs}


def _store_is_shared() -> bool:
    """True when this box's model store root lives on the SHARED/central catalog.

    Resolves the store root the SAME way the reaper/heartbeat resolve it
    (agent._models_store_root) and asks provision._on_shared_model_store about
    its realpath — the identical signal (sentinel file or HUGPY_SHARED_MODEL_STORE
    env) that already makes _model_store_reapable False on such a box. So the cap
    skip and the delete protection can never disagree about what "shared" means.

    Fail SAFE: any resolution failure returns False, so the per-worker cap keeps
    applying — a probe error must never silently disable a real worker's cap.
    """
    try:
        from hugpy_fleet.worker.agent import _models_store_root
        from hugpy_storage.provision import _on_shared_model_store
        root = _models_store_root()
        if not root:
            return False
        return bool(_on_shared_model_store(os.path.realpath(root)))
    except Exception:  # noqa: BLE001
        return False


def evict_to_fit(state, model_key: str, need_bytes: int) -> None:
    """Make room under the budget for ``model_key``, or raise BudgetRefusal.

    The IMPURE bookend to fit_plan: gathers live inputs, runs the plan, and
    executes any evictions through the worker's single guarded delete path
    (``_reap_reclaim`` — which re-proves EVERY guard per key at delete time and
    whose ``wipe_model`` is path-jailed and refuses shared/central storage).

    Called from provision.ensure_model_present BEFORE a pull starts. Best-effort
    by construction: any failure to COMPUTE a plan lets the pull proceed exactly
    as it did before this feature (never break a working pull on a bookkeeping
    error) — but a REFUSAL is a decision, not a failure, and always propagates.
    """
    try:
        from hugpy_fleet.worker.agent import _worker_storage, _reap_reclaim
        storage = _worker_storage(state)
        limits = getattr(state, "limits", None) or {}
        last_picked = getattr(state, "model_last_picked", None) or {}
        # Central-computed allocation totals (heartbeat reply). Absent before the
        # first beat -> the refusal simply omits the structural clause.
        allocated = getattr(state, "allocated", None) or {}
        # Is this box's model root the SHARED/central catalog (ae on the NAS)?
        # Resolved the SAME way the reap paths resolve the root: the store root
        # realpath through provision._on_shared_model_store (sentinel + env). On
        # a shared store the per-worker cap is a category error — fit_plan skips
        # it (see there). Fail SAFE: if the probe raises, treat as NOT shared so
        # the cap still applies (never accidentally disable a real worker's cap).
        shared_store = _store_is_shared()
        # EFFECTIVE cap (slice 4, min-wins): resolve min over {central
        # disk_cache_gib, worker same-drive declarations} against THIS box's
        # store root, and hand fit_plan the number + source map. Not consulted on
        # a shared store (fit_plan skips the cap there). Best-effort — a resolve
        # failure yields (None, {}) and fit_plan falls back to central-only.
        effective_cap, budget_sources = None, {}
        if not shared_store:
            try:
                from hugpy_fleet.worker.agent import _models_store_root
                store_root = _models_store_root() or ""
            except Exception:  # noqa: BLE001
                store_root = ""
            effective_cap, budget_sources = resolve_effective_cap(limits, store_root)
        plan = fit_plan(model_key, need_bytes, storage, limits, last_picked,
                        allocated, shared_store=shared_store,
                        effective_cap=effective_cap, budget_sources=budget_sources)
    except BudgetRefusal:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("budget check for %s failed (%s) — proceeding with the "
                       "pull as before", model_key, exc)
        return

    if plan["action"] == "refuse":
        logger.error("REFUSING pull of %s — %s", model_key,
                     plan["reason"]["reason"])
        raise BudgetRefusal(plan["reason"])

    if plan["action"] != "evict" or not plan["evict"]:
        return

    logger.info("budget: %s needs room — FIFO-evicting %d model(s) oldest-first: %s",
                model_key, len(plan["evict"]), ", ".join(plan["evict"]))
    try:
        result = _reap_reclaim(state, plan["evict"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("budget: eviction for %s failed (%s) — proceeding; the "
                       "pull may still hit a full disk", model_key, exc)
        return
    freed = result.get("freed_bytes", 0) if isinstance(result, dict) else 0
    logger.info("budget: freed %s for %s", _human(freed), model_key)
    # Force a fresh storage walk so the next check sees the deletions (the
    # 60s _STORAGE_CACHE would otherwise re-report the evicted models).
    try:
        from hugpy_fleet.worker.agent import _STORAGE_CACHE
        _STORAGE_CACHE["at"] = 0.0
    except Exception:  # noqa: BLE001
        pass
