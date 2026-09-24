"""MODEL PRIORITY GROUPS — the CENTRAL-SIDE resolver. Everything impure is here.

The split mirrors the derived-groups feature next door, for the same reason:

    comms/priority_groups.py   the STORE + key matching. No flask, no catalog,
                               no worker registry. One validated write path.

    this module                reads the model catalog, the blocklist and the
                               live worker registry, and answers "which member
                               of the operator's ordered list is usable RIGHT
                               NOW".

RESOLUTION, EXACTLY
-------------------
1. Find the ENABLED priority group that lists the requested key (alias-tolerant:
   "Qwen2.5-VL-7B-Instruct" and "Qwen~Qwen2.5-VL-7B-Instruct" are one model).
   No group -> ``None`` -> change nothing -> today's behaviour, byte for byte.
2. Walk ``group["members"]`` IN THE OPERATOR'S ORDER. For each, in this order:
       blocked?   -> skip, reason "blocked from the serving pool"
       in catalog?-> if not, skip, reason "not in the model catalog"
       servable?  -> ``worker_store.workers_for_model`` non-empty
   The FIRST member that clears all three is the answer.
3. Nothing clears them -> ``None`` -> change nothing. The originally requested
   key then fails EXACTLY as it fails today. We do not synthesize an error, we
   do not mask one, and we never route to a model outside the group.

THE BLOCK IS ABSOLUTE
---------------------
A blocked member is skipped at step 2 BEFORE anything else is asked about it,
and it is skipped whichever spelling the blocklist holds (the member name and
its canonical catalog key are both checked). The group reorders which candidate
is TRIED; it can never make a blocked model reachable. Note that
``workers_for_model`` independently returns [] for a blocked key — the explicit
check exists so the operator gets the honest WORD "blocked" in the preview
instead of an indistinguishable "no worker".

ORDER OUTRANKS THE DERIVATION
-----------------------------
When a priority group claims the requested key, ``member_for_model`` returns
this module's answer and never consults ``model_groups`` — the derived,
suffix-stripping grouping whose ranking pipeline would otherwise re-decide the
order the operator just wrote down. That derived feature also remains OFF by
default (``model_groups.enabled``); this one needs no kill switch because it
does nothing at all until an operator creates a group.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Outcome vocabulary — closed, and rendered verbatim by the console.
ST_CHOSEN = "chosen"          # first usable member, in the operator's order
ST_BLOCKED = "blocked"        # operator block — never laundered
ST_ARCHIVED = "archived"      # operator archive mark — never laundered either
ST_MISSING = "missing"        # not in the model catalog
ST_NO_WORKER = "no-worker"    # in the catalog, but nothing can serve it now
ST_LOWER = "lower-priority"   # usable, but a higher-ranked member won


def _catalog() -> dict:
    try:
        from hugpy_engine.config.models.models_config import get_models_dict
        return get_models_dict(dict_return=True) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("priority groups: catalog unavailable (%s)", exc)
        return {}


def _is_blocked(model_key: Any) -> bool:
    """Guarded block read. Fail-open (= not blocked) for the same reason
    ``workers._model_blocked`` does: a routing gate that raises is worse than a
    momentarily-unblocked model, and the REAL gate in ``workers_for_model``
    still stands behind this one."""
    try:
        from hugpy_fleet.central.blocklist import is_blocked
        return bool(is_blocked(model_key))
    except Exception:  # noqa: BLE001
        return False


def _boxes_for(model_key: str, pool: Optional[str], task: Optional[str]) -> list:
    """The workers that would serve ``model_key`` — the EXISTING selector's
    answer, narrowed by the SAME placement fence the picker applies. Priority
    groups do not re-implement worker eligibility: admission, pool, engine,
    designation/wildcard, liveness, tier, task capability and the block gate
    are all decided where they are decided today. A group only chooses WHICH
    KEY to hand that machinery.

    The ``_prefs_scope`` narrowing (2026-08-25) keeps the walk HONEST: a model
    carrying an ordered worker preference (its own, or its group's ``workers``
    allocation) is only pickable on listed workers — so a member whose fence
    excludes every eligible box must read ``no-worker`` HERE and let the walk
    fall to the next member, instead of being chosen and then refused by the
    picker (a failure the next member could have served)."""
    try:
        from hugpy_fleet.central.workers import worker_store
        boxes = worker_store.workers_for_model(model_key, pool=pool, task=task)
    except Exception as exc:  # noqa: BLE001
        logger.warning("priority groups: worker lookup failed for %s (%s)",
                       model_key, exc)
        return []
    try:
        from hugpy_engine.serve.overrides import placement_policy
        prefs, _polite, _by = placement_policy(model_key)
    except Exception:  # noqa: BLE001 — no policy read, no narrowing
        return boxes
    if not prefs:
        return boxes
    want = {str(p).strip().lower() for p in prefs if str(p).strip()}
    return [w for w in boxes
            if {str(w.get("id") or "").strip().lower(),
                str(w.get("name") or "").strip().lower()} & want]


def canonical_key(name: Any, catalog: Optional[dict] = None) -> Optional[str]:
    """The catalog key a member name refers to, or None when it is not there.

    Exact match wins; otherwise the shortest alias match, so a bare
    "Qwen2.5-VL-7B-Instruct" resolves to "Qwen~Qwen2.5-VL-7B-Instruct" rather
    than to a longer duplicate row like "...~2"."""
    from hugpy_fleet.central.priority_groups import keys_match

    cat = _catalog() if catalog is None else catalog
    nm = str(name or "").strip()
    if not nm:
        return None
    if nm in cat:
        return nm
    hits = [k for k in cat if keys_match(k, nm)]
    if not hits:
        return None
    hits.sort(key=lambda k: (len(str(k)), str(k)))
    return hits[0]


def _evaluate(name: str, pool: Optional[str], task: Optional[str],
              catalog: dict) -> dict:
    """One member's honest state. Never raises."""
    ck = canonical_key(name, catalog)
    row = {"model_key": name, "catalog_key": ck, "usable": False}
    # BLOCK FIRST, and against both spellings — the blocklist is keyed by
    # whatever the operator clicked, which may be the bare name or the
    # ~-qualified registry key.
    if _is_blocked(name) or (ck and _is_blocked(ck)):
        row["status"] = ST_BLOCKED
        row["reason"] = ("blocked from the serving pool by the operator — a "
                         "priority group never launders a block")
        return row
    if not ck:
        row["status"] = ST_MISSING
        row["reason"] = "not in the model catalog"
        return row
    # ARCHIVE MARK (2026-09-23): a member the operator marked for archive is
    # skipped like a block (never laundered through a group), with the
    # recorded mark as the reason.
    try:
        from hugpy_fleet.central.archive_gate import archive_reason
        _arch = archive_reason(ck)
    except Exception:  # noqa: BLE001 — fail open
        _arch = None
    if _arch:
        row["status"] = ST_ARCHIVED
        row["reason"] = _arch
        return row
    boxes = _boxes_for(ck, pool, task)
    if not boxes:
        row["status"] = ST_NO_WORKER
        row["reason"] = ("no worker can serve it right now (not assigned, not "
                         "loaded, or no eligible box)")
        return row
    row["usable"] = True
    row["status"] = ST_CHOSEN          # provisional; demoted below if it lost
    row["workers"] = [w.get("name") or w.get("id") for w in boxes]
    row["reason"] = f"servable now on {len(boxes)} worker(s)"
    return row


def resolve(model_key: str, pool: Optional[str] = None,
            task: Optional[str] = None, *, full: bool = False) -> dict:
    """The ordered walk. Returns the preview payload; ``route`` is the seam's
    answer (None = change nothing).

    ``full=False`` (the serve path) stops at the first usable member.
    ``full=True`` (the preview endpoint) evaluates every member so the operator
    sees why each one lost, then marks the ones after the winner
    ``lower-priority`` — the winner is still decided by the SAME first-usable
    walk, so the preview can never disagree with what routing does.
    """
    from hugpy_fleet.central.priority_groups import group_for_key, keys_match

    out: Dict[str, Any] = {"requested": model_key, "group": None,
                           "candidates": [], "chosen": None, "route": None,
                           "why": ""}
    group = group_for_key(model_key)
    if not group:
        out["why"] = ("not a member of any enabled priority group — resolution "
                      "is unchanged")
        return out
    out["group"] = {"id": group["id"], "name": group.get("name"),
                    "enabled": bool(group.get("enabled"))}

    catalog = _catalog()
    chosen: Optional[str] = None
    rows: List[dict] = []
    # Nested groups expand IN PLACE (comms.expand_members): a "group:<id>"
    # member contributes its own ordered members at that position, and each
    # expanded row carries ``via`` so the preview says which module it came
    # through instead of silently flattening the operator's structure.
    from hugpy_fleet.central.priority_groups import expand_members
    walk = expand_members(group)
    for pos, (name, via) in enumerate(walk, start=1):
        if chosen is not None and not full:
            rows.append({"model_key": name, "catalog_key": None, "position": pos,
                         "usable": None, "status": ST_LOWER, "via": via,
                         "reason": f"not tried — {chosen} is higher priority"})
            continue
        row = _evaluate(name, pool, task, catalog)
        row["position"] = pos
        row["via"] = via
        if row["usable"] and chosen is None:
            chosen = row["catalog_key"] or name
        elif row["usable"]:
            row["status"] = ST_LOWER
            row["reason"] = f"usable, but {chosen} is higher priority"
        rows.append(row)
    out["candidates"] = rows

    if chosen is None:
        out["why"] = (f"no member of {group['id']!r} is usable right now — "
                      f"routing {model_key} unchanged, so it fails exactly as "
                      f"it would without the group")
        return out

    out["chosen"] = chosen
    if keys_match(chosen, model_key):
        out["why"] = (f"{chosen} is the highest-priority usable member and is "
                      f"the key already requested — nothing to rewrite")
        return out                       # route stays None: change nothing
    out["route"] = chosen
    out["why"] = (f"{chosen} is the first usable member of {group['id']!r} in "
                  f"the operator's order")
    return out


def member_for_model(model_key: str, pool: Optional[str] = None,
                     task: Optional[str] = None) -> Optional[str]:
    """THE SEAM. The member to route instead of ``model_key``, or None.

    Total by construction — this sits on the serve path of every chat request,
    so it cannot raise and it cannot block. Every failure mode (no group, no
    usable member, a bug in here) yields None, which means "change nothing"."""
    try:
        return resolve(model_key, pool, task).get("route")
    except Exception as exc:  # noqa: BLE001 — routing never dies of a policy bug
        logger.warning("priority groups: resolution failed for %s (%s) — "
                       "routing the requested key unchanged", model_key, exc)
        return None


def covers(model_key: str) -> bool:
    """True when an ENABLED priority group claims ``model_key``. The derived
    grouping must not be consulted for a key the operator has ruled on."""
    try:
        from hugpy_fleet.central.priority_groups import group_for_key
        return group_for_key(model_key) is not None
    except Exception:  # noqa: BLE001
        return False
