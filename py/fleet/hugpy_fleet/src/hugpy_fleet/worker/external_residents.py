"""Registry of EXTERNAL leased residents — foreign GPU batch jobs (an OCR run,
a render sweep, any long checkpoint-and-resume workload) that hold the card
LIKE a model and yield it like one.

This is the missing half of the pid_registry's ``"external"`` host mode: the
pid registry records that a foreign PID holds VRAM (measured, heartbeat-fed),
while THIS module records how to make it let go — the control URL of the
job's supervisor (``worker.gpu_lease``), which pauses/kills its child on
request and re-claims the card later. The eviction verb (``_evict_model``)
resolves a key here to find the pause endpoint, exactly as the comfy branch
resolves the adopted ComfyUI's ``/free`` URL.

Deliberately in-memory and per-process, like the pid registry: a worker
restart forgets every lease, and the supervisor's periodic re-register (its
heartbeat) heals that within a minute — the same repopulate-on-beat contract
the rest of the residency state lives by. No persistence, no reconcile pass,
no thread of its own.

The record shape:
    {model_key, pid, control_url, vram_gib, note,
     registered_at, last_seen, state,       # state: "running" | "yielded"
     evictable, resume}                     # the operator-adjustable policy

Policy semantics (2026-08-12, "wildcard process" ruling):
  * ``evictable`` (bool, default True) is the external twin of managed-model
    residency: True behaves like on-demand (any demand path may pause the
    job), False behaves like static (only an operator ``force`` evicts).
  * ``resume`` ("enabled" | "disabled", default "enabled") is CLIENT-side
    policy — stored here so consoles can show it and so /ops/external/set can
    forward a change to the supervisor's control URL, but enforced by
    gpu_lease itself (a paused resume-disabled lease exits instead of
    re-claiming).
Both survive heartbeat re-registers: a register() that omits them keeps the
stored values, so an operator adjustment is not undone by the next beat.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

_LOCK = threading.RLock()
_RECORDS: Dict[str, dict] = {}


def register(model_key: str, pid: Optional[int],
             control_url: Optional[str] = None,
             vram_gib: Optional[float] = None,
             note: Optional[str] = None,
             evictable: Optional[bool] = None,
             resume: Optional[str] = None) -> dict:
    """Record (or refresh) an external lease. Idempotent: the supervisor
    re-posts this on its heartbeat (and after every relaunch, with the new
    child pid) — a refresh updates pid/last_seen and flips state back to
    "running" without perturbing registered_at. ``evictable``/``resume``
    update only when explicitly given (None keeps the stored policy — an old
    client's heartbeat must not undo an operator's /ops/external/set)."""
    now = time.time()
    with _LOCK:
        rec = _RECORDS.get(model_key)
        if rec is None:
            rec = {"model_key": model_key, "registered_at": now,
                   "evictable": True, "resume": "enabled"}
            _RECORDS[model_key] = rec
        rec.update({
            "pid": pid,
            "control_url": (control_url or rec.get("control_url") or None),
            "vram_gib": (vram_gib if vram_gib is not None
                         else rec.get("vram_gib")),
            "note": (note if note is not None else rec.get("note")),
            "last_seen": now,
            "state": "running" if pid else (rec.get("state") or "yielded"),
        })
        if evictable is not None:
            rec["evictable"] = bool(evictable)
        if resume in ("enabled", "disabled"):
            rec["resume"] = resume
        return dict(rec)


def set_policy(model_key: str,
               evictable: Optional[bool] = None,
               resume: Optional[str] = None) -> Optional[dict]:
    """Operator adjustment of a live lease's policy. Returns the updated
    record, or None when no such lease is registered."""
    with _LOCK:
        rec = _RECORDS.get(model_key)
        if rec is None:
            return None
        if evictable is not None:
            rec["evictable"] = bool(evictable)
        if resume in ("enabled", "disabled"):
            rec["resume"] = resume
        return dict(rec)


def unregister(model_key: str) -> bool:
    """Drop the lease record. Returns True if one was present."""
    with _LOCK:
        return _RECORDS.pop(model_key, None) is not None


def get(model_key: str) -> Optional[dict]:
    with _LOCK:
        rec = _RECORDS.get(model_key)
        return dict(rec) if rec is not None else None


def keys() -> List[str]:
    with _LOCK:
        return list(_RECORDS.keys())


def mark_yielded(model_key: str) -> None:
    """Note that the lease's child was paused (evicted). The record is KEPT —
    the supervisor is still alive and will re-register with a fresh pid when
    it resumes; keeping the record lets /ops/residents and the claim path see
    a yielded lease for what it is instead of forgetting the job exists."""
    with _LOCK:
        rec = _RECORDS.get(model_key)
        if rec is not None:
            rec["state"] = "yielded"
            rec["pid"] = None


def snapshot() -> List[dict]:
    with _LOCK:
        return [dict(r) for r in _RECORDS.values()]


def clear() -> None:
    """Drop every lease record. Test seam only (the live worker relies on the
    per-process in-memory lifetime, never on an explicit clear)."""
    with _LOCK:
        _RECORDS.clear()
