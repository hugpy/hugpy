"""MODEL GROUPS — the CENTRAL-SIDE provider. Everything impure lives here.

Spec: ``dev/MODEL-GROUPS-SPEC.md``.

The split with ``managers/resolvers/groups.py`` is deliberate and load-bearing:

    groups.py (core)   PURE. Derivation + the member-selection stage pipeline.
                       No settings, no catalog, no worker registry, no clock.
                       Unit-testable against real catalog fixtures with no fleet.

    this module (web)  Reads the settings store, the model catalog, the persisted
                       physical records and the live worker registry; shapes them
                       into the pure pipeline's inputs; emits telemetry; registers
                       the seam.

⚠ THE KILL SWITCH LIVES HERE, ON THE FIRST LINE OF THE SEAM (operator directive
2026-07-28). ``member_for_model`` returns None — meaning "change nothing" —
before touching anything else unless an operator has explicitly turned groups
on. Default OFF. ``HUGPY_MODEL_GROUPS=off`` is a HARD off that outranks the
setting. With the flag off this module reads nothing, allocates nothing, and
costs one dict lookup on the serve path.

``describe_groups`` (the ``GET /llm/groups`` body) deliberately does NOT check
the flag: it is a pure read over the catalog that routes nothing, so an operator
can inspect what groups WOULD do before enabling them. It reports the flag's
state so the UI can say "advisory only".
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SETTINGS_NS = "model_groups"
ENABLED_KEY = "enabled"
ENV_FLAG = "HUGPY_MODEL_GROUPS"

# The derived registry is a pure function of the catalog + the override map, and
# the catalog changes only on a discovery sweep. Re-deriving it per request would
# put a 109-entry regex walk on the serve path for no new information. Short TTL
# rather than event invalidation: a stale group for a few seconds is harmless
# (worst case an operator's brand-new tick takes 5s to bite), and the settings
# bus already exists for anything that needs to be instant.
_CACHE_TTL_S = 5.0
_cache: Dict[str, Any] = {"at": 0.0, "groups": None, "sig": None}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# The kill switch
# ---------------------------------------------------------------------------
def enabled_state() -> tuple:
    """``(enabled: bool, source: str)``. Default OFF.

    ``source`` is one of ``env-off`` / ``settings`` / ``default`` so the UI and
    ``GET /llm/groups`` can say WHERE the answer came from — the same
    provenance idiom SettingsPanel already renders for evict-policy."""
    env = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if env in ("off", "0", "false", "no"):
        return False, "env-off"           # HARD off, outranks the setting
    try:
        from hugpy_control.settings import settings_store
        raw = settings_store.get(SETTINGS_NS, ENABLED_KEY, None)
    except Exception:  # noqa: BLE001 — an unreadable store is an OFF store
        return False, "default"
    if raw is None:
        return False, "default"
    if isinstance(raw, dict):             # tolerate {"value": true}
        raw = raw.get("value")
    return bool(raw), "settings"


def is_enabled() -> bool:
    return enabled_state()[0]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _overrides() -> dict:
    try:
        from hugpy_control.settings import settings_store
        return settings_store.all(SETTINGS_NS) or {}
    except Exception:  # noqa: BLE001
        return {}


def _catalog() -> dict:
    try:
        from hugpy_engine.config.models.models_config import get_models_dict
        return get_models_dict(dict_return=True) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("model groups: catalog unavailable (%s) — no groups", exc)
        return {}


def registry(force: bool = False) -> dict:
    """The derived group registry, briefly cached. Never raises."""
    ov = _overrides()
    # Signature = (catalog size, override map) — cheap, and it catches the two
    # things that actually change membership. A full catalog hash would cost
    # more than the derivation it is trying to avoid.
    try:
        cat = _catalog()
        sig = (len(cat), repr(sorted(ov.items())))
    except Exception:  # noqa: BLE001
        return {}
    with _cache_lock:
        fresh = (not force and _cache["groups"] is not None
                 and _cache["sig"] == sig
                 and (time.time() - _cache["at"]) < _CACHE_TTL_S)
        if fresh:
            return _cache["groups"]
    try:
        from hugpy_engine.resolvers.groups import derive_groups
        groups = derive_groups(cat, ov)
    except Exception as exc:  # noqa: BLE001 — a bad derivation must not route
        logger.warning("model groups: derivation failed (%s) — no groups", exc)
        groups = {}
    with _cache_lock:
        _cache.update({"at": time.time(), "groups": groups, "sig": sig})
    return groups


def _physical(model_key: str) -> dict:
    """The model's persisted physical record — ``gguf_variants`` (with the
    completeness flags gguf_election produces) and ``size_bytes``.

    Reuses ``workers._model_physical``, which is a DICT LOOKUP against the
    record central already wrote, not a store walk. That distinction is the
    2026-07-27 /llm/workers fix and it applies here for the same reason: this
    runs on the serve path."""
    try:
        from hugpy_fleet.central.workers import _model_physical
        return _model_physical(model_key) or {}
    except Exception:  # noqa: BLE001
        return {}


def _boxes_for(model_key: str, pool: Optional[str], task: Optional[str]) -> list:
    """The workers that would serve ``model_key`` — the EXISTING selector's
    answer, verbatim.

    Model groups do not re-implement worker eligibility and do not touch the
    gate loop: admission, engine, pool, designation/wildcard, liveness, env
    tier, task capability and id_lock are all decided exactly where they are
    decided today. A group only chooses among the members those gates already
    approved."""
    try:
        from hugpy_fleet.central.workers import worker_store
        return worker_store.workers_for_model(model_key, pool=pool, task=task)
    except Exception:  # noqa: BLE001
        return []


def _alloc_mode(worker: dict, model_key: str) -> tuple:
    """``(mode, explicit)`` for this (worker, model).

    ``explicit`` is the operator-lever test: a persisted spill carrying
    ``alloc_mode`` (or a legacy ``n_gpu_layers``) is an operator statement and
    OUTRANKS a group tick; an EMPTY spill is derived and does not. Same
    provenance distinction the 2026-07-25 max-gpu bugfix turns on — read, never
    re-derived, so groups and the console always agree."""
    try:
        from hugpy_engine.alloc_modes import derive_alloc_mode
        spill = (worker.get("spill_by_model") or {}).get(model_key) or {}
        explicit = bool(spill)
        return derive_alloc_mode(spill), explicit
    except Exception:  # noqa: BLE001
        return "max-gpu", False


def _candidate(member: dict, worker: dict) -> dict:
    """One (member x worker) candidate for the pure pipeline."""
    mk = member.get("model_key")
    phys = _physical(mk)
    mode, explicit = _alloc_mode(worker, mk)
    gpus = worker.get("gpus") or []
    return {
        "model_key": mk,
        "framework": member.get("framework"),
        "hub_id": member.get("hub_id"),
        "bnb": bool((worker.get("bnb_by_model") or {}).get(mk)),
        "variants": phys.get("gguf_variants") or [],
        "bytes": phys.get("effective_bytes") or phys.get("size_bytes") or 0,
        "worker": {
            "id": worker.get("id"),
            "name": worker.get("name") or worker.get("id"),
            # MEASURED, and FREE rather than total — the pipeline's fit
            # predicate is a "does it fit right now" question. vram_free /
            # free_ram come off _vram_summary / _ram_summary, i.e. the
            # heartbeat, never an estimate.
            "vram_total": worker.get("vram_total"),
            "vram_free": worker.get("vram_free"),
            "ram_total": worker.get("ram_total"),
            "ram_free": worker.get("free_ram"),
            "has_gpu": bool(gpus),
            "alloc_mode": mode,
            "alloc_explicit": explicit,
        },
    }


# ---------------------------------------------------------------------------
# THE SEAM
# ---------------------------------------------------------------------------
def member_for_model(model_key: str, pool: Optional[str] = None,
                     task: Optional[str] = None) -> Optional[str]:
    """The group member to route instead of ``model_key``, or None.

    None means CHANGE NOTHING and is the answer whenever groups are off (the
    default), the key is in no group, the group chose the key already named, or
    anything at all went wrong. This function is on the serve path of every
    chat request, so it is total: it cannot raise and it cannot block.

    EXPLICIT PRIORITY GROUPS COME FIRST (operator directive 2026-08-06). An
    operator-written, ORDERED group is a ruling; the derivation below is a
    heuristic. When a priority group claims this key, its ordered walk IS the
    answer and the derived pipeline is never consulted for it — otherwise the
    ranking stages (quality/speed/ladder/fit) would re-decide an order the
    operator just wrote down by hand. This runs BEFORE ``is_enabled()`` on
    purpose: ``model_groups.enabled`` is the kill switch for the DERIVED
    feature, and explicit groups are not that feature. They need no kill switch
    of their own because they do nothing until an operator creates one, and
    each group carries its own ``enabled`` flag.
    """
    try:
        from hugpy_fleet.central.priority_group_settings import covers as _pg_covers
        from hugpy_fleet.central.priority_group_settings import member_for_model as _pg_member
        if _pg_covers(model_key):
            return _pg_member(model_key, pool, task)
    except Exception as exc:  # noqa: BLE001 — never break routing over policy
        logger.warning("priority groups: consult failed for %s (%s) — falling "
                       "through to the derived grouping", model_key, exc)

    if not is_enabled():
        return None                       # THE OFF-PATH. One dict lookup, done.
    try:
        return _member_for_model(model_key, pool, task)
    except Exception as exc:  # noqa: BLE001 — routing never dies of a policy bug
        logger.warning("model groups: selection failed for %s (%s) — routing "
                       "the requested key unchanged", model_key, exc)
        return None


def _member_for_model(model_key: str, pool: Optional[str],
                      task: Optional[str]) -> Optional[str]:
    from hugpy_engine.resolvers.groups import group_for_model, select_member

    groups = registry()
    group = group_for_model(groups, model_key)
    if not group:
        return None
    members = group.get("members") or []
    if len(members) <= 1 and not any((group.get("ticks") or {}).values()):
        # A single-member group with no ticks has nothing to decide. Skip the
        # whole pipeline rather than emit a member.select that says "we chose
        # the only option" on every request — a feed nobody can read is worse
        # than no feed.
        return None

    candidates: List[dict] = []
    for m in members:
        for w in _boxes_for(m.get("model_key"), pool, task):
            candidates.append(_candidate(m, w))
    if not candidates:
        return None

    sel = select_member(group, candidates)
    _emit(sel, model_key)

    chosen = sel.get("model_key")
    if not chosen or chosen == model_key:
        return None

    if sel.get("verdict") == "need":
        # PRIORITY: the ticked standards cannot be met by anything that fits
        # right now. Declare the need and open the ambient group scope so the
        # headroom pass this provokes is TAGGED with the group and the tick that
        # demanded it. We do NOT evict here and we do not size the eviction —
        # declare-need-then-evict is the admission path's job, and this module
        # is not allowed to become a second one.
        #
        # ⚠ The scope is opened, not held: phase 1 routes and returns, and the
        # worker runs its own admission in its own process. See
        # MODEL-GROUPS-SPEC §8 — putting the declared need on the wire is phase 2.
        _declare_need(sel)
    return chosen


def _declare_need(sel: dict) -> None:
    try:
        from hugpy_fleet.central import evictions as ev
        ev.emit_eviction_event(
            "member.select", group_key=sel.get("group_key"),
            model_key=sel.get("model_key"), reason=sel.get("why"),
            need_bytes=sel.get("need_bytes"),
            demanded_by=sel.get("demanded_by"), verdict="need")
    except Exception:  # noqa: BLE001 — telemetry is observation only
        pass


def _emit(sel: dict, requested: str) -> None:
    """member.select once, member.skip per losing iteration. Never raises."""
    try:
        from hugpy_fleet.central import evictions as ev
        ev.emit_member_select(
            group_key=sel.get("group_key"), model_key=sel.get("model_key"),
            reason=sel.get("why"), as_=sel.get("as"), ticks=sel.get("ticks"),
            requested=requested, verdict=sel.get("verdict"))
        for s in (sel.get("skipped") or ()):
            ev.emit_member_skip(group_key=sel.get("group_key"),
                                model_key=s.get("model_key"),
                                reason=s.get("reason") or "passed over")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# GET /llm/groups
# ---------------------------------------------------------------------------
def describe_groups(pool: Optional[str] = None) -> dict:
    """The groups report: members, ticks, and a PER-WORKER verdict with the WHY.

    The why is the point. "excluded" is not a status, it is a sentence — one
    line per exclusion, the same sentence the feed shows. An operator reading
    this should never have to ask why an iteration lost.

    Runs regardless of the kill switch (a pure read that routes nothing) and
    reports the switch's state so the UI can mark the verdicts advisory."""
    enabled, source = enabled_state()
    out = {"enabled": enabled, "source": source, "groups": []}
    try:
        groups = registry()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model groups: describe failed (%s)", exc)
        return out

    from hugpy_engine.resolvers.groups import ladder, select_member

    for gk in sorted(groups):
        g = groups[gk]
        members = []
        for m in g.get("members") or ():
            phys = _physical(m.get("model_key"))
            members.append({
                "model_key": m.get("model_key"),
                "framework": m.get("framework"),
                "hub_id": m.get("hub_id"),
                "bytes": phys.get("effective_bytes") or phys.get("size_bytes"),
                "ladder": [{"quant": r["quant"], "bytes": r["bytes"],
                            "bits": r["bits"]}
                           for r in ladder(phys.get("gguf_variants") or [])],
            })

        # One verdict per worker that could serve ANY member — computed by the
        # same pipeline that routes, so the report can never disagree with the
        # decision. (Running the real thing beats describing it.)
        by_worker: Dict[str, list] = {}
        for m in g.get("members") or ():
            for w in _boxes_for(m.get("model_key"), pool, None):
                by_worker.setdefault(w.get("id") or "", []).append(
                    _candidate(m, w))
        verdicts = {}
        for wid, cands in by_worker.items():
            if not cands:
                continue
            name = (cands[0].get("worker") or {}).get("name") or wid
            sel = select_member(g, cands)
            verdicts[name] = {
                "worker_id": wid,
                "preferred": sel.get("model_key"),
                "as": sel.get("as"),
                "verdict": sel.get("verdict"),
                "why": sel.get("why"),
                "excluded": list(sel.get("skipped") or ()),
            }
        out["groups"].append({
            "group_key": gk,
            "derived": bool(g.get("derived")),
            "ticks": g.get("ticks"),
            "members": members,
            "verdicts": verdicts,
        })
    return out
