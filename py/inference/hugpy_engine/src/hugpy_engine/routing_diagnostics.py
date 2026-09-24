"""Routing DIAGNOSTICS — the structured record behind every routing refusal.

Operator, 2026-09-23: "THIS 'worker_busy: … retry shortly' IS NEVER ACCURATE
AND NEVER USEFUL. THIS IS DEV, i need info, not user facing placation."

Every refusal the resolver returns (operator block, admission hold, worker
busy, cold-load capacity, no worker) is built HERE first as one dict:

    {"request_id", "model": {"requested", "resolved"}, "gate", "worker",
     "predicate", "alloc", "rule", "candidates": [per-worker row], "ts", "pid",
     "log_ref"}

``worker`` is the real name(s) of the candidate worker(s) the request was tried
on, or the vocabulary word ``unrouted`` when routing refused before any worker
was selected — never the placeholder ``?``.

per-worker row: name, id, status, online, last_seen (+age), holds
{on_disk, loaded, designated}, slots for the model (slot id, healthy/busy),
central in-flight count, the concurrency limit AND where it comes from,
vram free/total, the last load failure for the model on that worker
(class + first stderr line + ts), and ``skipped`` — why it did not take the
request.

The record is stored as a ``compute_actions`` row (action=call,
outcome=refused, detail=the dict) through the metrics seam — the same store
the load failures live in — and kept in a bounded in-process map, so the
route that answers the request and ``GET /llm/diagnostics/<request_id>`` read
the same thing. The refusal MESSAGE is rendered from the record
(:func:`render`): facts only — the gate, the failed predicate, the
candidates, the rule and its timer when one applies, and the ``log_ref``.
Never raises: a diagnostic that fails still yields a message.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_RECENT: "OrderedDict[str, dict]" = OrderedDict()
_RECENT_MAX = 512
_LOCK = threading.Lock()
_MISS: Dict[str, str] = {}


def _forms(model_key: str) -> set:
    try:
        from hugpy_engine import placement as _placement
        return set(_placement.get_worker_registry().key_forms(model_key))
    except Exception:  # noqa: BLE001
        k = str(model_key or "")
        return {k, k.lower(), k.split("~")[-1], k.split("/")[-1]}


def _first_line(text: Any, limit: int = 300) -> Optional[str]:
    s = str(text or "").strip()
    if not s:
        return None
    return s.splitlines()[0][:limit]


def cap_and_source(worker: dict, model_key: str, slot_served: bool) -> tuple:
    """``(limit, source)`` for (worker, model) exactly as the relay gate decides it."""
    if slot_served:
        return None, "slot-served: the worker's llama-server child schedules concurrency (central does not gate)"
    lim = (worker.get("serving_limits") or {}).get("in_process_max_concurrency")
    try:
        n = int(lim)
    except (TypeError, ValueError):
        return 1, "serving_limits.in_process_max_concurrency absent -> legacy in-process default 1"
    return max(1, n), f"serving_limits.in_process_max_concurrency={lim}" + (" (floored to 1)" if n < 1 else "")


def last_load_failure(worker: dict, forms: set) -> Optional[dict]:
    for k, rep in (worker.get("load_reports") or {}).items():
        if k in forms and isinstance(rep, dict) and not rep.get("ok", True):
            lf = rep.get("load_failure") or {}
            return {"ts": rep.get("ts"), "class": lf.get("class"),
                    "error": _first_line(rep.get("error")),
                    "loader_stderr": _first_line(lf.get("loader_stderr"))}
    return None


def worker_row(worker: dict, model_key: str, *, forms: Optional[set] = None,
               in_flight: Optional[int] = None, slot_served: bool = False,
               candidate: bool = False, skipped: Optional[str] = None,
               now: Optional[float] = None) -> dict:
    forms = forms if forms is not None else _forms(model_key)
    now = now if now is not None else time.time()
    seen = worker.get("last_seen")
    try:
        age = round(now - float(seen), 1) if seen else None
    except (TypeError, ValueError):
        age = None
    slots = [{"slot_id": s.get("slot_id") or s.get("control_port"), "healthy": s.get("healthy"),
              "busy": s.get("busy"), "model_key": s.get("model_key")}
             for s in (worker.get("slots") or [])
             if isinstance(s, dict) and s.get("model_key") in forms]
    limit, source = cap_and_source(worker, model_key, slot_served)
    return {
        "name": worker.get("name"), "id": worker.get("id"), "status": worker.get("status"),
        "online": worker.get("status") == "online", "last_seen": seen, "last_seen_age_s": age,
        "admission": worker.get("admission"),
        "holds": {"on_disk": bool(forms & set(worker.get("models_local") or [])),
                  "loaded": bool(forms & set(worker.get("loaded_models") or [])),
                  "designated": bool(forms & set(worker.get("models") or []))},
        "candidate": candidate, "slots": slots, "in_flight_central": in_flight,
        "concurrency_limit": limit, "limit_source": source,
        "vram_free": worker.get("vram_free"),
        "vram_total": worker.get("vram_total") or worker.get("gpu_total_bytes_known"),
        "last_load_failure": last_load_failure(worker, forms),
        "skipped": skipped,
    }


def default_skip(row: dict) -> str:
    if not row["online"]:
        return f"status={row['status']} (last_seen {row['last_seen_age_s']}s ago)"
    if row.get("admission") not in (None, "approved"):
        return f"worker admission={row.get('admission')}"
    if not row["candidate"]:
        h = row["holds"]
        return ("not a routing candidate for this model (on_disk=%s loaded=%s designated=%s)"
                % (h["on_disk"], h["loaded"], h["designated"]))
    if row["concurrency_limit"] is not None and row["in_flight_central"] is not None \
            and row["in_flight_central"] >= row["concurrency_limit"]:
        return (f"in_flight_central={row['in_flight_central']} >= limit={row['concurrency_limit']} "
                f"({row['limit_source']})")
    return (f"candidate with room at refusal time (in_flight_central={row['in_flight_central']}, "
            f"limit={row['concurrency_limit']} from {row['limit_source']})")


def build(*, request_id: Optional[str], requested: Optional[str], resolved: str, gate: str,
          predicate: str, alloc: Any = None, rule: Optional[str] = None,
          workers: Iterable[dict] = (), candidate_ids: Iterable[str] = (),
          in_flight: Optional[Callable[[str], Optional[int]]] = None,
          slot_served: Optional[Callable[[dict], bool]] = None,
          skips: Optional[Dict[str, str]] = None, extra: Optional[dict] = None) -> dict:
    forms = _forms(resolved)
    cids = {c for c in candidate_ids if c}
    rows: List[dict] = []
    for w in workers or ():
        if not isinstance(w, dict):
            continue
        wid = w.get("id") or ""
        try:
            served = bool(slot_served(w)) if slot_served else False
        except Exception:  # noqa: BLE001
            served = False
        row = worker_row(w, resolved, forms=forms, candidate=wid in cids, slot_served=served,
                         in_flight=in_flight(wid) if in_flight else None)
        row["skipped"] = (skips or {}).get(wid) or default_skip(row)
        rows.append(row)
    cand_names = [r.get("name") for r in rows if r.get("candidate") and r.get("name")]
    worker = ", ".join(cand_names) if cand_names else "unrouted"
    return {"request_id": request_id, "model": {"requested": requested or resolved, "resolved": resolved},
            "gate": gate, "worker": worker, "predicate": predicate, "rule": rule,
            "alloc": alloc if isinstance(alloc, dict) else None,
            "candidates": rows, "ts": time.time(), "pid": os.getpid(),
            "log_ref": None, **(extra or {})}


def _worker_brief(r: dict) -> str:
    bits = [f"status={r['status']}"]
    h = r["holds"]
    bits.append("holds=" + ("+".join(k for k in ("loaded", "on_disk", "designated") if h[k]) or "none"))
    if r["slots"]:
        bits.append("slots=" + ",".join(f"{s['slot_id']}:{'busy' if s['busy'] else 'idle'}"
                                        f"{'' if s['healthy'] else '/unhealthy'}" for s in r["slots"]))
    if r["in_flight_central"] is not None:
        bits.append(f"in_flight={r['in_flight_central']}/{r['concurrency_limit']}")
    if r["vram_free"] is not None and r["vram_total"]:
        bits.append(f"vram_free={int(r['vram_free']) / 2**30:.1f}/{int(r['vram_total']) / 2**30:.1f}GiB")
    if r["last_load_failure"]:
        lf = r["last_load_failure"]
        bits.append(f"last_load_fail={lf.get('class') or ''}:{lf.get('loader_stderr') or lf.get('error')}")
    bits.append(f"skipped: {r['skipped']}")
    return f"{r['name']}[" + "; ".join(bits) + "]"


def render(diag: dict, code: Optional[str] = None) -> str:
    """The refusal message, from the record: facts only."""
    m = diag.get("model") or {}
    head = f"{code}: " if code else ""
    parts = [f"{head}gate={diag.get('gate')}", f"predicate: {diag.get('predicate')}",
             f"model={m.get('resolved')}" + (f" (requested {m.get('requested')})"
                                            if m.get("requested") != m.get("resolved") else "")]
    if diag.get("alloc"):
        parts.append(f"alloc={diag['alloc']}")
    if diag.get("rule"):
        parts.append(f"rule: {diag['rule']}")
    if diag.get("candidates"):
        parts.append("workers: " + " | ".join(_worker_brief(r) for r in diag["candidates"]))
    else:
        parts.append("workers: 0 worker rows in the registry snapshot at refusal time")
    parts.append(f"request={diag.get('request_id')} log_ref={diag.get('log_ref')}")
    return "; ".join(parts)


def record(diag: dict) -> dict:
    """Store the record (compute_actions row + in-process map); sets log_ref."""
    rid = diag.get("request_id") or f"anon-{time.time():.6f}"
    try:
        from hugpy_engine import placement as _placement
        fn = getattr(_placement.get_model_metrics(), "record_refusal", None)
        ref = fn(diag) if fn is not None else None
        if ref is not None:
            diag["log_ref"] = f"compute_actions#{ref}" if isinstance(ref, int) else str(ref)
    except Exception as exc:  # noqa: BLE001 — a diagnostic store fault never breaks the refusal
        logger.debug("refusal record failed for %s", rid, exc_info=True)
        diag["log_ref_error"] = f"compute_actions write failed: {type(exc).__name__}: {exc}"
    if diag.get("log_ref") is None:
        diag.setdefault("log_ref_error", "record_refusal returned no row id (metrics seam has no store)")
        diag["log_ref"] = f"memory:{os.getpid()}:{rid}"
    with _LOCK:
        _RECENT[rid] = diag
        _RECENT.move_to_end(rid)
        while len(_RECENT) > _RECENT_MAX:
            _RECENT.popitem(last=False)
    logger.warning("routing refusal %s: %s", rid, render(diag))
    return diag


def lookup(request_id: Optional[str]) -> Optional[dict]:
    """The stored record for ``request_id``: this process first, then the store."""
    if not request_id:
        return None
    with _LOCK:
        hit = _RECENT.get(request_id)
    if hit is not None:
        return hit
    if len(_MISS) > _RECENT_MAX:
        _MISS.clear()
    try:
        from hugpy_engine import placement as _placement
        fn = getattr(_placement.get_model_metrics(), "find_refusal", None)
        if fn is None:
            _MISS[request_id] = "the metrics seam has no find_refusal (no persistent store)"
            return None
        hit = fn(request_id)
        if hit is None:
            _MISS[request_id] = "find_refusal returned no compute_actions row"
        return hit
    except Exception as exc:  # noqa: BLE001
        _MISS[request_id] = f"compute_actions lookup failed: {type(exc).__name__}: {exc}"
        return None


def miss_reason(request_id: Optional[str]) -> str:
    """Why :func:`lookup` found nothing for ``request_id`` — from what it saw."""
    if not request_id:
        return "no request_id given"
    with _LOCK:
        n = len(_RECENT)
    store = _MISS.pop(request_id, None) or "store not consulted"
    return (f"not in this process's refusal ring (pid {os.getpid()}, {n}/{_RECENT_MAX} "
            f"records held); {store}")


__all__ = ["build", "record", "render", "lookup", "miss_reason", "worker_row", "cap_and_source",
           "last_load_failure", "default_skip"]
