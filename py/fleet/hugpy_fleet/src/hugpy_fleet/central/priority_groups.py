"""MODEL PRIORITY GROUPS — the operator's EXPLICIT, ORDERED fallback list.

WHAT THIS IS, AND WHAT IT IS NOT
-------------------------------
A *priority group* is a hand-written, ORDERED list of model keys. When a request
names ANY member of an enabled group, resolution walks that list IN THE
OPERATOR'S ORDER and routes the FIRST member that is actually usable right now.
Nothing is inferred, nothing is derived, nothing is guessed.

Contrast with ``managers/resolvers/groups.py`` (*model groups*, 2026-07-28),
which DERIVES membership by stripping packaging/quant suffixes off a repo name
("Qwen2.5-VL-7B-Instruct-GGUF" -> group "qwen2.5-vl-7b-instruct") and then picks
a member by a ranking pipeline (quality/speed/priority ticks, ladder walk, fit).
That feature is heuristic BOTH in who is in a group AND in which member wins.
It is off by default (settings ``model_groups.enabled``, default FALSE) and it
stays off unless an operator turns it on.

    THIS module is the EXPLICIT one, and it OUTRANKS the derived one. If a
    requested key belongs to an enabled priority group, the operator's ordering
    is the answer and the derivation pipeline is never consulted for that key.
    (Operator directive 2026-08-06: "not grouped by default" — an ordering the
    operator did not write is not an ordering they consented to.)

A BLOCK IS NEVER LAUNDERED
--------------------------
A blocked member is SKIPPED, never served. A group can only reorder which
candidate is tried; it can never make a blocked model reachable. When no member
of the group is usable, resolution returns "change nothing" and the originally
requested key fails exactly as it fails today — the group must not invent a new
error, and must not mask the old one.

PERSISTENCE
-----------
The F4 runtime settings store (``comms.settings.settings_store``), namespace
``model_priority_groups``, keyed by group id. That store is
``$PROJECTS_HOME/settings.json``: fcntl-locked read-modify-write, atomic
``os.replace``, thread-safe (an RLock), with a short read cache — the same
mechanism ``comms/blocklist.py`` uses for ``models.blocked``. No new storage
mechanism is introduced, and there is exactly ONE write path (this module), so
the operator gate on ``/llm/model-groups`` mutations is the only gate to get
right.

RECORD SHAPE
------------
    {"id": str,             # slug, the settings key
     "name": str,           # operator-facing label
     "members": [str, ...], # ORDERED model keys — the whole point
     "workers": [str, ...], # ORDERED worker ids/names — the group's ALLOCATION
                            # (2026-08-25): where the group's models live, in
                            # priority order. [] = no placement statement, and
                            # routing is byte-identical to pre-feature. Feeds
                            # placement_policy as the fallback when the model
                            # has no per-model worker_prefs (which outrank it).
     "enabled": bool,
     "created_at": float, "updated_at": float, "by": str}

MATCHING is alias-tolerant, mirroring ``workers._match_keys``: a registry key
qualifies a base name with its owner via "~" ("Qwen~Qwen2.5-VL-7B-Instruct")
while the operator, the hub and the workers all routinely say the bare name
("Qwen2.5-VL-7B-Instruct"). Those spellings must intersect or a group written in
the obvious way would silently never fire. Suffixes are NEVER stripped here:
"Qwen2.5-VL-7B-Instruct" and "Qwen2.5-VL-7B-Instruct-GGUF" are two DIFFERENT
members that the operator listed in a deliberate order, and conflating them is
precisely the implicit behaviour this module exists to replace.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from hugpy_control.settings import settings_store

logger = logging.getLogger(__name__)

# Settings namespace for the explicit group registry (group_id -> record).
NS = "model_priority_groups"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Key matching
# ---------------------------------------------------------------------------
def key_forms(model_key: Any) -> set:
    """The spellings ``model_key`` may legitimately be named by.

    Deliberately the SAME rule as ``workers._match_keys``: raw, lowercased, the
    "/"-tail (hub_id -> repo name) and the "~"-tail (registry key -> bare base).
    Deliberately NOT ``resolvers.groups.base_name``: that one strips ``-GGUF``
    and quant tokens, which would merge the operator's two ordered members into
    one and destroy the ordering they asked for."""
    s = str(model_key or "").strip()
    if not s:
        return set()
    forms = {s, s.lower()}
    tail = s.split("/")[-1]
    forms.add(tail)
    forms.add(tail.lower())
    if "~" in s:
        base = s.split("~", 1)[1]
        if base:
            forms.add(base)
            forms.add(base.lower())
    return forms


def keys_match(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` are two spellings of the same model."""
    fa, fb = key_forms(a), key_forms(b)
    return bool(fa and fb and (fa & fb))


def slugify(text: Any, fallback: str = "group") -> str:
    s = _SLUG_RE.sub("-", str(text or "").strip().lower()).strip("-")
    return s or fallback


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _normalize(gid: str, raw: Any) -> Optional[dict]:
    """A stored value -> a well-formed record, or None when unusable.

    Tolerant on read (a hand-edited settings.json must not take the feature
    down) and strict on write (``validate``)."""
    if not isinstance(raw, dict):
        return None
    members: List[str] = []
    for m in (raw.get("members") or []):
        m = str(m or "").strip()
        # Order-preserving de-dup: a repeated member is a typo, not a second
        # chance — walking it twice would just print the same skip reason twice.
        if m and not any(keys_match(m, seen) for seen in members):
            members.append(m)
    workers: List[str] = []
    for w in (raw.get("workers") or []):
        w = str(w or "").strip()
        if w and w.lower() not in {seen.lower() for seen in workers}:
            workers.append(w)
    return {
        "id": str(gid),
        "name": str(raw.get("name") or gid),
        "members": members,
        "workers": workers,
        "enabled": bool(raw.get("enabled", True)),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "by": raw.get("by") or "operator",
    }


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def all_groups() -> List[dict]:
    """Every stored group, normalized, sorted by name then id. Never raises."""
    try:
        raw = settings_store.all(NS) or {}
    except Exception as exc:  # noqa: BLE001 — a read must never break a caller
        logger.warning("priority groups: read failed (%s) — no groups", exc)
        return []
    out = []
    for gid, rec in raw.items():
        norm = _normalize(str(gid), rec)
        if norm is not None:
            out.append(norm)
    out.sort(key=lambda g: (str(g.get("name") or "").lower(), g["id"]))
    return out


def get_group(group_id: Any) -> Optional[dict]:
    gid = str(group_id or "").strip()
    if not gid:
        return None
    try:
        return _normalize(gid, settings_store.get(NS, gid))
    except Exception as exc:  # noqa: BLE001
        logger.warning("priority groups: read failed for %s (%s)", gid, exc)
        return None


def enabled_groups() -> List[dict]:
    return [g for g in all_groups() if g.get("enabled")]


def group_for_key(model_key: Any, groups: Optional[List[dict]] = None
                  ) -> Optional[dict]:
    """The ENABLED group that lists ``model_key`` (alias-tolerant), or None.

    "At most one enabled group per key" is enforced on write (``validate``), so
    a first-match walk here is deterministic. If a hand-edited store ever breaks
    that invariant this returns the first by the stable sort order and logs it,
    rather than refusing to route."""
    groups = enabled_groups() if groups is None else [
        g for g in groups if g.get("enabled")]
    hits = [g for g in groups
            if any(keys_match(model_key, m) for m in (g.get("members") or []))]
    if len(hits) > 1:
        logger.warning(
            "priority groups: %s is claimed by %d enabled groups (%s) — using "
            "%s; fix the overlap, membership is meant to be exclusive",
            model_key, len(hits), [g["id"] for g in hits], hits[0]["id"])
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Nesting (2026-08-25): a member may be a REFERENCE to another group
# ---------------------------------------------------------------------------
# Spelled "group:<id>". The referenced group functions as a MODULE inside the
# parent: at evaluation its own ordered members expand IN PLACE (recursively,
# cycle-guarded), and for placement it inherits the parent's ``workers`` when
# it carries none of its own. A dangling reference (id not in the registry) is
# tolerated on read and skipped at expansion — same posture as a missing model
# member, which the walk reports rather than trips over.
GROUP_REF = "group:"


def is_group_ref(member: Any) -> bool:
    return str(member or "").strip().lower().startswith(GROUP_REF)


def ref_id(member: Any) -> str:
    return str(member or "").strip()[len(GROUP_REF):].strip()


def expand_members(group: dict, groups: Optional[List[dict]] = None,
                   _seen: Optional[set] = None) -> List[Tuple[str, Optional[str]]]:
    """The group's ordered members with every nested group expanded in place.

    Returns ``[(model_key, via), ...]`` where ``via`` is the id of the nested
    group a key arrived through (None for direct members) — the walk reports
    provenance, never hides it. Cycles are impossible to expand and are simply
    not followed; a repeated model key keeps its FIRST (highest-priority)
    position. Never raises."""
    groups = all_groups() if groups is None else groups
    by_id = {g["id"]: g for g in groups}
    seen_groups = set(_seen or ())
    seen_groups.add(group.get("id"))
    out: List[Tuple[str, Optional[str]]] = []
    have: List[str] = []
    for m in (group.get("members") or []):
        if is_group_ref(m):
            child = by_id.get(ref_id(m))
            if child is None or child["id"] in seen_groups:
                continue
            for key, _via in expand_members(child, groups, seen_groups):
                if not any(keys_match(key, h) for h in have):
                    have.append(key)
                    out.append((key, child["id"]))
        else:
            if not any(keys_match(m, h) for h in have):
                have.append(m)
                out.append((m, None))
    return out


def effective_workers(group: dict, groups: Optional[List[dict]] = None,
                      _seen: Optional[set] = None) -> List[str]:
    """The ordered worker allocation that GOVERNS this group: its own
    ``workers`` when set, else the nearest enabled PARENT's (a module inherits
    the placement of the group it is mounted in). ``[]`` when nobody up the
    chain says anything. Cycle-guarded, never raises."""
    own = [str(w) for w in (group.get("workers") or []) if str(w).strip()]
    if own:
        return own
    groups = all_groups() if groups is None else groups
    seen = set(_seen or ())
    seen.add(group.get("id"))
    for parent in groups:
        if not parent.get("enabled") or parent["id"] in seen:
            continue
        if any(is_group_ref(m) and ref_id(m) == group.get("id")
               for m in (parent.get("members") or [])):
            got = effective_workers(parent, groups, seen)
            if got:
                return got
    return []


def workers_for_key(model_key: Any) -> List[str]:
    """The ORDERED worker allocation of the enabled group claiming ``model_key``,
    or ``[]`` — the group half of placement.

    ``[]`` is the byte-identical no-op: a model in no group, in a disabled
    group, or in a group whose ``workers`` list was never filled routes exactly
    as it does without this feature. Consumed by
    ``managers.serve.overrides.placement_policy`` as the FALLBACK when the model
    carries no per-model ``worker_prefs`` — an order written on the model itself
    is more specific than the group's and outranks it, the same
    explicit-outranks-implicit rule this module already applies to the derived
    grouping. Never raises."""
    try:
        groups = all_groups()
        group = group_for_key(model_key, groups)
        if not group:
            return []
        # Nesting: a group with no workers of its own inherits the allocation
        # of the enabled parent it is mounted in (effective_workers).
        return effective_workers(group, groups)
    except Exception as exc:  # noqa: BLE001 — placement must never break here
        logger.warning("priority groups: workers_for_key(%s) failed (%s)",
                       model_key, exc)
        return []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(group_id: str, name: Any, members: Any, enabled: Any,
             groups: Optional[List[dict]] = None,
             workers: Any = None) -> Tuple[list, list, list]:
    """``(clean_members, clean_workers, errors)``. ``errors`` empty means the
    write is legal.

    THE ONE STRUCTURAL RULE: a model key may appear in AT MOST ONE ENABLED
    group. Two enabled groups claiming the same key would make resolution depend
    on iteration order, i.e. on nothing an operator can see. Conflicts are
    REPORTED with the offending key and the group that already holds it — never
    silently resolved by dropping a member (that would quietly rewrite the
    operator's list).

    A DISABLED group may overlap freely: it routes nothing, and staging a
    replacement group next to the live one is a legitimate thing to want."""
    errors: List[str] = []
    gid = str(group_id or "").strip()
    if not gid:
        errors.append("id is required")
    if not str(name or "").strip():
        errors.append("name is required")

    clean: List[str] = []
    if not isinstance(members, (list, tuple)):
        errors.append("members must be a list of model keys, in priority order")
        members = []
    for m in members:
        m = str(m or "").strip()
        if not m:
            continue
        if any(keys_match(m, seen) for seen in clean):
            errors.append(f"duplicate member {m!r} — a member may be listed once")
            continue
        clean.append(m)
    if not clean:
        errors.append("members must contain at least one model key")
    for m in clean:
        if is_group_ref(m) and ref_id(m) == gid:
            errors.append(f"{m!r} — a group cannot contain itself")

    # ``workers`` is the group's ORDERED worker allocation — optional, and an
    # empty list is legal (a group may order models without placing them). A
    # duplicate is a typo like a duplicate member; matching is case-insensitive
    # because ``_pref_index`` matches id-or-name case-insensitively downstream.
    clean_workers: List[str] = []
    if workers is None:
        workers = []
    if not isinstance(workers, (list, tuple)):
        errors.append("workers must be a list of worker ids or names, "
                      "in priority order")
        workers = []
    for w in workers:
        w = str(w or "").strip()
        if not w:
            continue
        if w.lower() in {seen.lower() for seen in clean_workers}:
            errors.append(f"duplicate worker {w!r} — a worker may be listed once")
            continue
        clean_workers.append(w)

    if bool(enabled):
        others = [g for g in (all_groups() if groups is None else groups)
                  if g.get("enabled") and g["id"] != gid]
        for m in clean:
            for og in others:
                if any(keys_match(m, om) for om in (og.get("members") or [])):
                    errors.append(
                        f"{m!r} is already a member of the enabled group "
                        f"{og['id']!r} ({og.get('name')}) — a model key may be "
                        f"in at most one enabled group")
    return clean, clean_workers, errors


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def put_group(group_id: Any, *, name: Any, members: Any, enabled: Any = True,
              workers: Any = None,
              by: Optional[str] = None) -> Tuple[Optional[dict], list]:
    """Create or REPLACE a group. ``(record, errors)``; record is None on error.

    Replace rather than merge: ``members`` and ``workers`` are each an ORDER,
    and a partial update of an order is not a thing that has a meaning.
    ``PATCH`` in the route layer reads the current record, applies the patch,
    and calls this — so there is still exactly one validated write path."""
    gid = slugify(group_id or name)
    clean, clean_workers, errors = validate(gid, name, members, enabled,
                                            workers=workers)
    if errors:
        return None, errors
    prior = get_group(gid) or {}
    now = time.time()
    rec = {
        "id": gid,
        "name": str(name).strip(),
        "members": clean,
        "workers": clean_workers,
        "enabled": bool(enabled),
        "created_at": prior.get("created_at") or now,
        "updated_at": now,
        "by": by or "operator",
    }
    settings_store.set(NS, gid, rec)
    logger.info("priority group %s: members=%s workers=%s enabled=%s (by=%s)",
                gid, clean, clean_workers, rec["enabled"], rec["by"])
    return rec, []


def migrate_worker_tokens(resolve) -> dict:
    """Rewrite STALE worker-name references in every group's ``workers`` ORDER to
    a stable worker id (operator incident 2026-09-25).

    A group's ``workers`` is the ordered allocation, stored by NAME, so a worker
    rename ("aeb" -> "ae-worker", same id) strands the token exactly as it does
    for a per-model ``worker_prefs``. ``resolve(token) -> worker_id | None``
    returns the id ONLY for a token that does not already resolve to a live worker
    but matches a former name; a None leaves the token in place (already-valid or
    genuinely orphaned). Writes through ``put_group`` — the one validated path —
    so ordering and dedupe stay intact. Returns ``{group_id: [(old, new)]}``.
    Idempotent: a rewritten token is an id, which never matches a former name."""
    changed: dict = {}
    for g in all_groups():
        workers = g.get("workers") or []
        rows, new_workers = [], []
        for tok in workers:
            new = resolve(tok)
            new_workers.append(new or tok)
            if new:
                rows.append((str(tok), new))
        if not rows:
            continue
        rec, errs = put_group(g["id"], name=g["name"], members=g["members"],
                              enabled=g.get("enabled", True), workers=new_workers,
                              by="worker-rename-migration")
        if errs:
            logger.warning("priority group %s: worker-token migration skipped (%s)",
                           g["id"], errs)
            continue
        changed[g["id"]] = rows
        for old, new in rows:
            logger.info("priority group %s: worker %r -> id %r", g["id"], old, new)
    return changed


def move_member(model_key: Any, group_id: Any = None,
                by: Optional[str] = None) -> Tuple[List[dict], list]:
    """Move ``model_key`` INTO ``group_id`` (appended LAST — joining a group is
    not an opinion about its order) and OUT of every other group; a falsy
    ``group_id`` just removes it everywhere. ``(changed_records, errors)``.

    The one-select gesture behind the model table's Group column. Removals run
    BEFORE the add so the at-most-one-enabled-group invariant never trips on
    the transition. A removal that would leave a group EMPTY is refused with
    "delete the group instead" — an empty members list is invalid by
    ``validate``, and silently deleting a group the operator wrote is exactly
    the kind of surprise this module refuses. Every write goes through
    ``put_group`` — still the one validated path."""
    key = str(model_key or "").strip()
    if not key:
        return [], ["model_key is required"]
    target_gid = str(group_id or "").strip() or None
    groups = all_groups()
    target = None
    if target_gid:
        target = next((g for g in groups if g["id"] == target_gid), None)
        if target is None:
            return [], [f"no such priority group: {target_gid!r}"]

    changed: List[dict] = []
    errors: List[str] = []
    for g in groups:
        if target is not None and g["id"] == target["id"]:
            continue
        if not any(keys_match(key, m) for m in (g.get("members") or [])):
            continue
        members = [m for m in g["members"] if not keys_match(key, m)]
        if not members:
            errors.append(
                f"removing {key!r} would leave group {g['id']!r} empty — "
                "delete the group instead")
            continue
        rec, errs = put_group(g["id"], name=g["name"], members=members,
                              enabled=g["enabled"], workers=g.get("workers"),
                              by=by)
        errors += errs
        if rec:
            changed.append(rec)

    if target is not None and not any(
            keys_match(key, m) for m in (target.get("members") or [])):
        rec, errs = put_group(
            target["id"], name=target["name"],
            members=list(target.get("members") or []) + [key],
            enabled=target["enabled"], workers=target.get("workers"), by=by)
        errors += errs
        if rec:
            changed.append(rec)
    return changed, errors


def delete_group(group_id: Any) -> bool:
    gid = str(group_id or "").strip()
    if not gid:
        return False
    existed = settings_store.delete(NS, gid)
    if existed:
        logger.info("priority group deleted: %s", gid)
    return bool(existed)
