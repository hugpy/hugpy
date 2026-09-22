"""Fleet template schema, storage, snapshot + dry-run diff (Slice 0).

Grounded on FLEET-TEMPLATES-DESIGN.md:
  * §1 — the versioned template document (``schema_version`` 1).
  * §2 — definitions live in the namespaced settings store (``fleet.templates``,
          one key per name); a pointer ``fleet.active`` records the last apply.
  * §3a — DRY-RUN DIFF: resolve selectors against the live ``/llm/workers`` view,
          compute a per-worker ORDERED plan of the concrete calls that would move
          live -> desired. Slice 0 computes the plan; it NEVER executes it.
  * §3b — AUTO-SNAPSHOT: capture the live fleet as a valid template whose per-worker
          selectors are resolved to concrete ids, so it ROUND-TRIPS (feeding it back
          through ``compute_diff`` against the same unchanged fleet yields no plan).
  * §9.1 — composition: several sections may match one worker; LISTS union across
          layers, SCALARS override by specificity (``*`` < group < name < id).
  * §9.6 — ``serving_mode`` vocabulary: ``static`` (residency held) / ``on_demand``
          (default, absence of a residency entry) / ``off`` (== the ``absent`` list).

Deliberately dependency-injected: ``validate_template`` / ``build_snapshot`` /
``compute_diff`` are pure and take the live worker view as an argument. Only the
thin storage wrappers touch the settings store, and they accept an injected store
so tests can point at a temp file. NOTHING here imports flask_app or mutates a
worker — Slice 0 is read + blob-write only.
"""
from __future__ import annotations

import fnmatch
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# §9.6 — the canonical three hard serving states.
SERVING_MODES = ("static", "on_demand", "off")

# §2 — storage coordinates in the namespaced settings store.
NS_TEMPLATES = "fleet.templates"     # key = template name, value = the doc
NS_FLEET = "fleet"                   # key "active" = the applied-template pointer
KEY_ACTIVE = "active"

# Selector specificity (§9.1): most-specific wins for scalar knobs.
_SELECTOR_RANK = {"all": 0, "group": 1, "name": 2, "id": 3}

# limits knobs that map to the real set_limits route (workers.WorkerStore._LIMIT_KEYS).
_LIMIT_KEYS = ("ram_max_gib", "gpu_mem_gib", "threads", "disk_cache_gib",
               "disk_reserve_gib")


class TemplateError(ValueError):
    """A template document failed schema validation (with a clear message)."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TemplateError(f"{where} must be a non-empty string")
    return value


def _check_unknown(d: Dict[str, Any], allowed: Tuple[str, ...], where: str) -> None:
    extra = [k for k in d if k not in allowed]
    if extra:
        raise TemplateError(
            f"{where}: unknown field(s) {sorted(extra)}; "
            f"allowed: {sorted(allowed)}")


def parse_selector(select: Any) -> Tuple[str, str]:
    """Normalize a §1 selector to ``(kind, value)``.

    Accepted forms (kind in id | name | group | all):
      * ``"*"``                       -> ("all", "*")
      * ``{"all": true}``             -> ("all", "*")
      * ``{"id": "w-abc"}``           -> ("id",   "w-abc")
      * ``{"name": "studio-box"}``    -> ("name", "studio-box")
      * ``{"group": "studio"}``       -> ("group","studio")
    Exactly one addressing key is allowed. Raises TemplateError otherwise.
    """
    if select == "*":
        return ("all", "*")
    if isinstance(select, dict):
        if list(select.keys()) == ["all"]:
            if select["all"] is True:
                return ("all", "*")
            raise TemplateError('selector {"all": ...} must be true')
        keys = [k for k in ("id", "name", "group") if k in select]
        if len(keys) == 1 and len(select) == 1:
            k = keys[0]
            return (k, _require_str(select[k], f"selector .{k}"))
    raise TemplateError(
        'select must be "*" or a dict with exactly one of '
        '{"id"|"name"|"group": <str>} or {"all": true}, got: %r' % (select,))


def _worker_groups(worker: Dict[str, Any]) -> List[str]:
    """A worker's group memberships (§9.5). Prefers a ``groups`` list; falls
    back to the legacy single ``pool`` string read as a one-element list."""
    groups = worker.get("groups")
    if isinstance(groups, list):
        return [str(g) for g in groups if str(g).strip()]
    pool = (worker.get("pool") or "").strip()
    return [pool] if pool else []


def _selector_matches(kind: str, value: str, worker: Dict[str, Any]) -> bool:
    if kind == "all":
        return True
    if kind == "id":
        return worker.get("id") == value
    if kind == "name":
        return worker.get("name") == value
    if kind == "group":
        return value in _worker_groups(worker)
    return False


# ---------------------------------------------------------------------------
# §1 — schema validation
# ---------------------------------------------------------------------------
_DOC_KEYS = ("schema_version", "name", "description", "created_at", "updated_at",
             "author", "revision", "fleet", "workers")
_FLEET_KEYS = ("download_budget_per_worker", "required_pkg_version")
_SECTION_KEYS = ("select", "slot_count", "assignments", "absent", "limits",
                 "serving_mode")
_ASSIGN_KEYS = ("model", "serving_mode", "pin", "spill")


def _validate_assignment(a: Any, where: str) -> None:
    if not isinstance(a, dict):
        raise TemplateError(f"{where} must be an object")
    _check_unknown(a, _ASSIGN_KEYS, where)
    _require_str(a.get("model"), f"{where}.model")
    sm = a.get("serving_mode")
    if sm is not None and sm not in SERVING_MODES:
        raise TemplateError(
            f"{where}.serving_mode must be one of {list(SERVING_MODES)}, got {sm!r}")
    if "pin" in a and not isinstance(a["pin"], bool):
        raise TemplateError(f"{where}.pin must be a boolean")
    if "spill" in a and not isinstance(a["spill"], dict):
        raise TemplateError(f"{where}.spill must be an object")


def _validate_section(sec: Any, idx: int) -> None:
    where = f"workers[{idx}]"
    if not isinstance(sec, dict):
        raise TemplateError(f"{where} must be an object")
    _check_unknown(sec, _SECTION_KEYS, where)
    if "select" not in sec:
        raise TemplateError(f"{where}.select is required")
    parse_selector(sec["select"])  # raises on malformed selector
    if "slot_count" in sec:
        n = sec["slot_count"]
        if not isinstance(n, int) or isinstance(n, bool) or not (0 <= n <= 16):
            raise TemplateError(f"{where}.slot_count must be an int in 0..16")
    pinned_models: set = set()
    if "assignments" in sec:
        if not isinstance(sec["assignments"], list):
            raise TemplateError(f"{where}.assignments must be a list")
        for j, a in enumerate(sec["assignments"]):
            _validate_assignment(a, f"{where}.assignments[{j}]")
            if a.get("pin") is True:
                pinned_models.add(a["model"])
    absent = sec.get("absent")
    if absent is not None:
        if not isinstance(absent, list) or not all(isinstance(x, str) for x in absent):
            raise TemplateError(f"{where}.absent must be a list of strings")
        # §8.8 — a model that is both pinned and absent is a contradiction.
        for mk in sorted(pinned_models):
            for pat in absent:
                if mk == pat or fnmatch.fnmatch(mk, pat):
                    raise TemplateError(
                        f"{where}: model {mk!r} is pinned AND matched by "
                        f"absent pattern {pat!r} — a template cannot both pin "
                        f"and park the same model (§8.8)")
    if "limits" in sec:
        lim = sec["limits"]
        if not isinstance(lim, dict):
            raise TemplateError(f"{where}.limits must be an object")
        _check_unknown(lim, _LIMIT_KEYS, f"{where}.limits")
        for k, v in lim.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TemplateError(f"{where}.limits.{k} must be numeric")
    sm = sec.get("serving_mode")
    if sm is not None and sm not in SERVING_MODES:
        raise TemplateError(
            f"{where}.serving_mode must be one of {list(SERVING_MODES)}, got {sm!r}")


def validate_template(doc: Any) -> Dict[str, Any]:
    """Validate a template document against schema_version 1 (§1).

    Returns the doc unchanged on success; raises TemplateError with a clear,
    field-anchored message on any malformation.
    """
    if not isinstance(doc, dict):
        raise TemplateError("template must be a JSON object")
    _check_unknown(doc, _DOC_KEYS, "template")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise TemplateError(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{doc.get('schema_version')!r}")
    _require_str(doc.get("name"), "name")
    for opt in ("description", "created_at", "updated_at", "author"):
        if opt in doc and not isinstance(doc[opt], str):
            raise TemplateError(f"{opt} must be a string")
    if "revision" in doc:
        r = doc["revision"]
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise TemplateError("revision must be a non-negative integer")
    fleet = doc.get("fleet")
    if fleet is not None:
        if not isinstance(fleet, dict):
            raise TemplateError("fleet must be an object")
        _check_unknown(fleet, _FLEET_KEYS, "fleet")
        b = fleet.get("download_budget_per_worker")
        if b is not None and (not isinstance(b, int) or isinstance(b, bool) or b < 0):
            raise TemplateError(
                "fleet.download_budget_per_worker must be a non-negative integer")
        v = fleet.get("required_pkg_version")
        if v is not None and not isinstance(v, str):
            raise TemplateError("fleet.required_pkg_version must be a string")
    workers = doc.get("workers")
    if workers is not None:
        if not isinstance(workers, list):
            raise TemplateError("workers must be a list")
        for idx, sec in enumerate(workers):
            _validate_section(sec, idx)
    return doc


# ---------------------------------------------------------------------------
# §3b — snapshot: live fleet -> a valid, round-tripping template
# ---------------------------------------------------------------------------
def _serving_mode_of(residency: Dict[str, Any], mk: str) -> str:
    """Live residency -> template serving_mode (§9.6). ``static`` if the agent
    holds a static residency entry, else the default ``on_demand`` (absence)."""
    return "static" if residency.get(mk) == "static" else "on_demand"


def build_snapshot(name: str, description: Optional[str],
                   workers_view: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Capture the CURRENT live fleet as a valid template document (§3b).

    ``workers_view`` is the live ``/llm/workers`` list (each row as list_workers
    returns it). Per worker we emit one section with ``select={"id": <id>}`` and
    the fields the diff reads back — so the snapshot round-trips: feeding it to
    ``compute_diff`` against the SAME unchanged view yields an empty plan.

    Fields captured per worker:
      * assignments — from the worker's assigned ``models`` set; each model's
        ``serving_mode`` from ``config.residency`` (static|on_demand), ``pin``
        from ``config.pinned``, ``spill`` from ``spill_by_model``.
      * slot_count — from ``config.slot_count`` (omitted when the agent hasn't
        reported one, so the diff won't try to force a value).
      * limits — the worker's operator-set ``limits`` (omitted when unset).
    """
    sections: List[Dict[str, Any]] = []
    for w in workers_view:
        cfg = w.get("config") or {}
        residency = cfg.get("residency") or {}
        pinned = cfg.get("pinned") or {}
        spill_by = w.get("spill_by_model") or {}
        assignments: List[Dict[str, Any]] = []
        for mk in sorted(set(w.get("models") or [])):
            a: Dict[str, Any] = {"model": mk,
                                 "serving_mode": _serving_mode_of(residency, mk)}
            if pinned.get(mk):
                a["pin"] = True
            if isinstance(spill_by.get(mk), dict) and spill_by[mk]:
                a["spill"] = dict(spill_by[mk])
            assignments.append(a)
        section: Dict[str, Any] = {"select": {"id": w.get("id")},
                                   "assignments": assignments}
        sc = cfg.get("slot_count")
        if isinstance(sc, int) and not isinstance(sc, bool):
            section["slot_count"] = sc
        limits = w.get("limits")
        if isinstance(limits, dict) and limits:
            section["limits"] = {k: v for k, v in limits.items()
                                 if k in _LIMIT_KEYS}
        sections.append(section)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description or "",
        "created_at": _now_iso(),
        "author": "snapshot",
        "workers": sections,
    }
    return validate_template(doc)


# ---------------------------------------------------------------------------
# §9.1 — composition: resolve the sections matching one worker to a flat
# desired-state (lists union, scalars override by specificity).
# ---------------------------------------------------------------------------
def _resolve_desired(doc: Dict[str, Any],
                     worker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Flatten every template section that matches ``worker`` into one desired
    per-worker state, or None when no section matches (worker out of scope).

    LISTS union across layers (assignments merged by model, most-specific wins
    for a model's per-model scalars; absent patterns unioned). SCALAR knobs
    (slot_count, serving_mode, each limits key) take the most-specific set value;
    two sections at the SAME specificity tier that set the same scalar to
    different values raise TemplateError (§9.1 same-tier conflict).
    """
    matches: List[Tuple[int, Dict[str, Any]]] = []
    for sec in (doc.get("workers") or []):
        kind, value = parse_selector(sec["select"])
        if _selector_matches(kind, value, worker):
            matches.append((_SELECTOR_RANK[kind], sec))
    if not matches:
        return None
    matches.sort(key=lambda t: t[0])  # least specific first; most specific last

    assignments: Dict[str, Dict[str, Any]] = {}
    absent: set = set()
    scalars: Dict[str, Any] = {}
    scalar_rank: Dict[str, int] = {}
    limits: Dict[str, Any] = {}
    limits_rank: Dict[str, int] = {}

    def _set_scalar(field: str, val: Any, rank: int) -> None:
        if field in scalars and scalar_rank[field] == rank and scalars[field] != val:
            raise TemplateError(
                f"same-tier conflict on {field!r}: two selectors at specificity "
                f"tier {rank} set it to {scalars[field]!r} and {val!r} (§9.1)")
        if field not in scalars or rank >= scalar_rank[field]:
            scalars[field] = val
            scalar_rank[field] = rank

    for rank, sec in matches:
        for a in (sec.get("assignments") or []):
            mk = a["model"]
            cur = assignments.setdefault(mk, {"model": mk})
            # more-specific (later) layer overrides this model's per-model scalars
            if "serving_mode" in a:
                cur["serving_mode"] = a["serving_mode"]
            if "pin" in a:
                cur["pin"] = a["pin"]
            if "spill" in a:
                cur["spill"] = a["spill"]
        for pat in (sec.get("absent") or []):
            absent.add(pat)
        if "slot_count" in sec:
            _set_scalar("slot_count", sec["slot_count"], rank)
        if "serving_mode" in sec:
            _set_scalar("serving_mode", sec["serving_mode"], rank)
        for k, v in (sec.get("limits") or {}).items():
            if k in limits and limits_rank[k] == rank and limits[k] != v:
                raise TemplateError(
                    f"same-tier conflict on limits.{k}: tier {rank} sets it to "
                    f"{limits[k]!r} and {v!r} (§9.1)")
            if k not in limits or rank >= limits_rank[k]:
                limits[k] = v
                limits_rank[k] = rank

    return {
        "assignments": assignments,          # {model: {serving_mode?, pin?, spill?}}
        "absent": sorted(absent),
        "slot_count": scalars.get("slot_count"),
        "worker_serving_mode": scalars.get("serving_mode"),
        "limits": limits,
    }


# ---------------------------------------------------------------------------
# §3a — dry-run diff: desired vs live -> an ordered per-worker plan
# ---------------------------------------------------------------------------
def _matches_any(mk: str, patterns: List[str]) -> bool:
    return any(mk == p or fnmatch.fnmatch(mk, p) for p in patterns)


def _plan_for_worker(desired: Dict[str, Any],
                     worker: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered list of the concrete calls that WOULD move this worker
    live -> desired (§3a/§3c order: config knobs, then unassigns, then assigns).
    Slice 0 emits these as plan LINES only; nothing is executed."""
    cfg = worker.get("config") or {}
    live_slot = cfg.get("slot_count")
    live_res = cfg.get("residency") or {}
    live_pin = cfg.get("pinned") or {}
    live_models = set(worker.get("models") or [])
    live_limits = worker.get("limits") or {}
    wid = worker.get("id")
    wname = worker.get("name")

    plan: List[Dict[str, Any]] = []

    def line(op: str, **kw: Any) -> None:
        plan.append({"op": op, "worker_id": wid, "worker_name": wname, **kw})

    # 1) config knob: slot_count (/ops/config)
    dslot = desired.get("slot_count")
    if dslot is not None and dslot != live_slot:
        line("config", field="slot_count", **{"from": live_slot, "to": dslot})

    # 2) config knob: limits (set_limits)
    for k, v in (desired.get("limits") or {}).items():
        if live_limits.get(k) != v:
            line("limits", field=k, **{"from": live_limits.get(k), "to": v})

    # desired assigned set (§8.2 UNION/ADD: omission never unassigns; only the
    # explicit `absent` list removes) — but a model flagged serving_mode "off"
    # in-line is treated as a park too.
    desired_assign = {mk: a for mk, a in (desired.get("assignments") or {}).items()
                      if a.get("serving_mode") != "off"}
    off_inline = [mk for mk, a in (desired.get("assignments") or {}).items()
                  if a.get("serving_mode") == "off"]
    absent_patterns = list(desired.get("absent") or [])

    # 3) config knobs per already-resident model: residency + pin
    for mk in sorted(set(desired_assign) & live_models):
        a = desired_assign[mk]
        want_static = a.get("serving_mode") == "static"
        live_static = live_res.get(mk) == "static"
        if want_static != live_static:
            line("config", field="residency", model=mk,
                 **{"from": "static" if live_static else "on_demand",
                    "to": "static" if want_static else "on_demand"})
        want_pin = bool(a.get("pin"))
        has_pin = bool(live_pin.get(mk))
        if want_pin != has_pin:
            line("config", field="pinned", model=mk,
                 **{"from": has_pin, "to": want_pin},
                 destructive=(has_pin and not want_pin))  # unpin is destructive (§4)

    # 4) unassigns: models matched by the `absent` list (or in-line "off") that
    #    are currently assigned. Blocked by pin permanence -> flagged (§4).
    for mk in sorted(live_models):
        if _matches_any(mk, absent_patterns) or mk in off_inline:
            line("unassign", model=mk, destructive=True,
                 blocked_by_pin=bool(live_pin.get(mk)))

    # 5) assigns: desired models not currently assigned (warm follows assign).
    for mk in sorted(set(desired_assign) - live_models):
        a = desired_assign[mk]
        line("assign", model=mk, serving_mode=a.get("serving_mode", "on_demand"),
             pin=bool(a.get("pin")), spill=a.get("spill"))

    return plan


def compute_diff(doc: Dict[str, Any],
                 workers_view: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DRY-RUN diff of a template against the live fleet (§3a). Side-effect free.

    Resolves selectors against ``workers_view``, flattens composition (§9.1),
    and returns a structured, ordered plan per in-scope worker. Never writes,
    never relays. ``empty`` is True iff no worker needs any change and every
    selector matched at least one worker (the round-trip invariant checks this).
    """
    validate_template(doc)
    workers_out: List[Dict[str, Any]] = []
    out_of_scope: List[Dict[str, Any]] = []
    matched_workers: set = set()
    duplicate_counts: Dict[str, int] = {}

    for w in workers_view:
        desired = _resolve_desired(doc, w)
        if desired is None:
            out_of_scope.append({"worker_id": w.get("id"),
                                 "worker_name": w.get("name")})
            continue
        matched_workers.add(w.get("id"))
        for mk in (desired.get("assignments") or {}):
            duplicate_counts[mk] = duplicate_counts.get(mk, 0) + 1
        plan = _plan_for_worker(desired, w)
        workers_out.append({
            "worker_id": w.get("id"),
            "worker_name": w.get("name"),
            "plan": plan,
        })

    # Selectors that matched no live worker (id/name typos, offline boxes).
    unmatched: List[Dict[str, str]] = []
    for sec in (doc.get("workers") or []):
        kind, value = parse_selector(sec["select"])
        if kind in ("id", "name") and not any(
                _selector_matches(kind, value, w) for w in workers_view):
            unmatched.append({"kind": kind, "value": value})

    # §9.2 — deliberate double-booking; report the fleet-wide duplication so the
    # operator sees the cost they chose (a model on N workers is not de-duped).
    multi_homed = {mk: n for mk, n in duplicate_counts.items() if n > 1}

    empty = (not unmatched) and all(not wp["plan"] for wp in workers_out)

    return {
        "template": doc.get("name"),
        "revision": doc.get("revision"),
        "empty": empty,
        "workers": workers_out,
        "out_of_scope_workers": out_of_scope,
        "unmatched_selectors": unmatched,
        "multi_homed": multi_homed,
    }


# ---------------------------------------------------------------------------
# §2 — storage wrappers over the namespaced settings store
# ---------------------------------------------------------------------------
def _store(store: Any = None) -> Any:
    if store is not None:
        return store
    from hugpy_control.settings import settings_store
    return settings_store


def list_templates(store: Any = None) -> List[Dict[str, Any]]:
    """Names + revisions + descriptions of every stored template (§6 list)."""
    s = _store(store)
    out = []
    for name, doc in (s.all(NS_TEMPLATES) or {}).items():
        doc = doc if isinstance(doc, dict) else {}
        out.append({"name": name,
                    "revision": doc.get("revision"),
                    "description": doc.get("description", "")})
    return sorted(out, key=lambda d: str(d["name"]))


def get_template(name: str, store: Any = None) -> Optional[Dict[str, Any]]:
    doc = _store(store).get(NS_TEMPLATES, name)
    return doc if isinstance(doc, dict) else None


def save_template(doc: Dict[str, Any], store: Any = None) -> Dict[str, Any]:
    """Validate -> bump ``revision`` (per-save) -> store; return the stored doc.

    The revision is server-authoritative: it is (prior stored revision + 1),
    so a first save is revision 1 and every subsequent save increments — any
    ``revision`` the caller supplied is ignored. ``created_at`` is preserved from
    the first save; ``updated_at`` is stamped each time.
    """
    validate_template(doc)
    s = _store(store)
    name = doc["name"]
    prior = s.get(NS_TEMPLATES, name)
    prior = prior if isinstance(prior, dict) else {}
    prior_rev = prior.get("revision")
    prior_rev = prior_rev if isinstance(prior_rev, int) and not isinstance(prior_rev, bool) else 0
    stored = dict(doc)
    stored["revision"] = prior_rev + 1
    stored["created_at"] = prior.get("created_at") or doc.get("created_at") or _now_iso()
    stored["updated_at"] = _now_iso()
    s.set(NS_TEMPLATES, name, stored)
    return stored


def delete_template(name: str, store: Any = None) -> bool:
    return bool(_store(store).delete(NS_TEMPLATES, name))


def get_active(store: Any = None) -> Optional[Dict[str, Any]]:
    """The ``fleet.active`` pointer (last applied template), or None. Slice 0
    never sets this — APPLY is Slice 1 — so it stays null until then."""
    ptr = _store(store).get(NS_FLEET, KEY_ACTIVE)
    return ptr if isinstance(ptr, dict) else None
