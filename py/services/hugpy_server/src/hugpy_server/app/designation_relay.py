"""Designation relay: central converges to the DB (operator ruling 2026-10-02).

The console writes a pair's ``assigned`` flag to model_workers through
POST /models/database/<model>/workers/<worker>/assigned and never waits on
central's WorkerStore lock (which made the × hang 40 s+ while
_materialize_storage re-read GGUF headers). This thread, every
HUGPY_DESIGNATION_RELAY_S seconds (default 10), diffs each worker's central
``models`` list against the DB rows and applies the difference through
central's own assign_model / unassign_model — in the background, so the lock
wait is nobody's problem but this thread's.

Rules:
  * a worker with NO DB rows is unknown -> untouched (never "unassign all")
  * a pinned designation (effective_pin) is never removed by the relay
  * assign_model is called with source="operator" (the DB row IS operator intent)
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)
_started = threading.Event()
# Every central write re-materializes the worker view under the store lock
# (minutes on a busy HDD), so a pass applies at most this many changes and
# lets readers through before the next pass picks up the rest.
MAX_CHANGES_PER_PASS = 3
# Central drops every UNPINNED designation on an API restart / worker
# re-register (pins are its persistence boundary). A DB designation is
# therefore recorded as a PIN marked as the relay's own; the relay may clear
# ONLY its own pins when the DB says unassigned. Every other pin is respected:
# an OPERATOR pin (set_pin by the console) and a legacy agent-side pin (the
# worker's drive-inventory automation) both block the relay from unassigning
# (operator 2026-10-02: "respect the pins ... the pins will be crucial", mainly
# for video-generation worker delegation; drive-inventory adoption is to become
# a per-worker UI button, automatic only for a new worker with no history).
RELAY_PIN_BY = "db-designation"


def _interval() -> float:
    try:
        return max(2.0, float(os.environ.get("HUGPY_DESIGNATION_RELAY_S", "10")))
    except ValueError:
        return 10.0


def reconcile_once() -> dict:
    """One pass. Returns {"assigned": [...], "unassigned": [...], "skipped": n}."""
    from hugpy_engine.model_index import enabled, fetch_assigned_by_worker
    out = {"assigned": [], "unassigned": [], "skipped": 0}
    if not enabled():
        return out
    by_worker = fetch_assigned_by_worker()
    if not by_worker:
        return out
    from hugpy_fleet.central.workers import (effective_pin, list_workers, set_pin,
                                             unassign_model)
    applied = 0
    for w in list_workers() or []:
        wid = str(w.get("id") or "")
        rows = by_worker.get(wid)
        if rows is None:
            out["skipped"] += 1
            continue                                   # no DB knowledge of this worker
        want = {name for name, flag in rows.items() if flag}
        have = set(w.get("models") or [])
        for key in sorted(have - want):
            if applied >= MAX_CHANGES_PER_PASS:
                return out
            try:
                pin = effective_pin(w, key)
            except Exception:  # noqa: BLE001
                pin = {"pinned": False}
            if pin.get("pinned"):
                if pin.get("by") == RELAY_PIN_BY:
                    set_pin(wid, key, False, by=RELAY_PIN_BY)   # our own pin: clear it
                else:
                    continue                           # operator / agent pin: respected, never unassigned
            if unassign_model(wid, key) is not None:
                out["unassigned"].append((wid[:8], key)); applied += 1
        for key in sorted(want - have):
            if applied >= MAX_CHANGES_PER_PASS:
                return out
            # set_pin designates (source "operator") AND pins, so the
            # designation survives central restarts / worker re-registers
            if set_pin(wid, key, True, by=RELAY_PIN_BY) is not None:
                out["assigned"].append((wid[:8], key)); applied += 1
        # DB-assigned pairs central already lists but holds UNPINNED would be
        # dropped at the next restart — pin them as ours (one write each)
        for key in sorted(want & have):
            if applied >= MAX_CHANGES_PER_PASS:
                return out
            try:
                if not effective_pin(w, key)["pinned"]:
                    set_pin(wid, key, True, by=RELAY_PIN_BY); applied += 1
                    out["assigned"].append((wid[:8], key + " (pinned)"))
            except Exception:  # noqa: BLE001
                pass
    return out


def _loop() -> None:
    time.sleep(_interval())                            # let the app finish booting
    while True:
        try:
            r = reconcile_once()
            if r["assigned"] or r["unassigned"]:
                logger.info("designation relay: +%s -%s", r["assigned"], r["unassigned"])
        except Exception as exc:  # noqa: BLE001 — the relay must never die
            logger.warning("designation relay pass failed: %s", exc)
        time.sleep(_interval())


def start() -> bool:
    """Start the relay thread once per process (HUGPY_DESIGNATION_RELAY=0 disables)."""
    if os.environ.get("HUGPY_DESIGNATION_RELAY", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if _started.is_set():
        return False
    _started.set()
    # Warm the model index SYNCHRONOUSLY first: its schema init (CREATE OR
    # REPLACE trigger on hugpy_worker_registry = AccessExclusive lock) must
    # never run lazily from inside a WorkerStore transaction that already
    # holds that table's row lock — that self-deadlock froze central on
    # 2026-10-02 (heartbeats blocked -> "WorkerUnreachable").
    try:
        from hugpy_engine.model_index import fetch_assigned_by_worker
        fetch_assigned_by_worker()
    except Exception as exc:  # noqa: BLE001
        logger.warning("designation relay: model index warm-up failed: %s", exc)
    threading.Thread(target=_loop, name="designation-relay", daemon=True).start()
    return True
