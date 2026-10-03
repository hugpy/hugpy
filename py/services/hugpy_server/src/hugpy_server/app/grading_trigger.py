"""GRADING TRIGGER (operator 2026-10-02): "a model's first successful load /
any load in which the grading has not been previously attempted for that model".

Every successful call reaches central's call-success hook. The first one for a
model with NO grading attempt on record queues that model for a benchmark run on
the worker that just served it. A drain loop (one process, flock-elected) starts
the oldest queued model whenever no benchmark is running — so it never collides
with an operator's own run, and runs one model at a time.

"Attempted" = a row in ``grading_attempts`` (queued / running / done / failed —
a failed attempt counts, so a broken model is not re-queued every call) or a
grade already in ``model_metrics`` (the history before this trigger existed).
The grading run's own calls find their model already attempted, so the trigger
never re-fires on itself. Fail-open (small_pg): no DB = no automatic grading.
"""
from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS grading_attempts (
    model_key    TEXT PRIMARY KEY,
    worker_id    TEXT,
    status       TEXT NOT NULL DEFAULT 'queued',
    run_id       TEXT,
    queued_at    DOUBLE PRECISION NOT NULL,
    started_at   DOUBLE PRECISION,
    finished_at  DOUBLE PRECISION,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS grading_attempts_status ON grading_attempts (status, queued_at);
"""
DRAIN_S = float(os.environ.get("HUGPY_GRADING_DRAIN_S", "60"))
DEFAULT_TOKENS = 128

_RUNNER = None
_known: Optional[set] = None
_known_lock = threading.Lock()


def enabled() -> bool:
    return (os.environ.get("HUGPY_AUTO_GRADING") or "1").strip().lower() not in ("0", "false", "no", "off")


def _run(fn):
    global _RUNNER
    if _RUNNER is None:
        from hugpy_server.app.small_pg import Runner
        _RUNNER = Runner("grading trigger", DDL)
    return _RUNNER.run(fn)


def _load_known() -> set:
    def op(cur):
        cur.execute("SELECT model_key FROM grading_attempts")
        keys = {r[0] for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT model_name FROM model_metrics WHERE grade IS NOT NULL")
        keys |= {r[0] for r in cur.fetchall()}
        return keys
    return _run(op)


def attempted(model_key: str) -> Optional[bool]:
    """True/False, or None when the DB is unreachable (then nothing is queued)."""
    global _known
    with _known_lock:
        if _known is None:
            k = _load_known()
            if k is None:
                return None
            _known = k
        return model_key in _known


def on_call_success(worker_id: str, model_key: str, meta: Optional[dict] = None) -> bool:
    """central's call-success hook: queue a never-attempted model. True if queued."""
    if not enabled() or not model_key or not worker_id:
        return False
    if attempted(model_key) is not False:
        return False

    def op(cur):
        cur.execute("INSERT INTO grading_attempts (model_key, worker_id, status, queued_at)"
                    " VALUES (%s, %s, 'queued', %s) ON CONFLICT (model_key) DO NOTHING",
                    (model_key, str(worker_id), time.time()))
        return cur.rowcount
    n = _run(op)
    with _known_lock:
        if _known is not None:
            _known.add(model_key)
    if n:
        log.info("grading trigger: %s queued for grading on %s (first successful load, never graded)",
                 model_key, worker_id)
    return bool(n)


_ACTIVE = ("running", "resuming", "judging")
_FAILED = ("error", "cancelled", "interrupted")


def _settle_running(state: dict) -> None:
    """A 'running' attempt whose run is no longer active has finished: 'failed'
    when that run ended in error/cancel/interrupt, else 'done'."""
    def op(cur):
        cur.execute("SELECT model_key, run_id FROM grading_attempts WHERE status='running'")
        return cur.fetchall()
    for mk, rid in (_run(op) or []):
        same = rid == state.get("run_id")
        if same and state.get("status") in _ACTIVE:
            continue
        final = "failed" if (same and state.get("status") in _FAILED) else "done"
        _run(lambda cur, mk=mk, final=final: cur.execute(
            "UPDATE grading_attempts SET status=%s, finished_at=%s WHERE model_key=%s",
            (final, time.time(), mk)))


def drain_once() -> Optional[str]:
    """Start the oldest queued model's grading if no benchmark is running.
    Returns the model started, else None."""
    from hugpy_server.app.routes import review_routes as rr
    state = rr.benchmark_state()
    _settle_running(state)
    if rr.benchmark_active():
        return None

    def op(cur):
        cur.execute("SELECT model_key, worker_id FROM grading_attempts WHERE status='queued'"
                    " ORDER BY queued_at LIMIT 1")
        return cur.fetchone()
    row = _run(op)
    if not row:
        return None
    mk, wid = row
    params = {"tokens": DEFAULT_TOKENS, "models": [mk], "workers": [wid] if wid else [],
              "suite": None, "with_judge": False, "budgets": None, "resume": False,
              "force": False, "force_cold": False, "measure_cold_load": False}
    started, public = rr.start_benchmark_run(params)
    if not started:
        return None
    _run(lambda cur: cur.execute(
        "UPDATE grading_attempts SET status='running', run_id=%s, started_at=%s WHERE model_key=%s",
        (public.get("run_id"), time.time(), mk)))
    log.info("grading trigger: grading %s on %s (benchmark run %s)", mk, wid, public.get("run_id"))
    return mk


def _drain_loop(lock_fh) -> None:
    while True:
        try:
            drain_once()
        except Exception:  # noqa: BLE001 — the loop must survive
            log.warning("grading trigger drain failed", exc_info=True)
        time.sleep(DRAIN_S)


def start() -> bool:
    """Register the hook (every process) and start the drain (one process)."""
    if not enabled():
        return False
    from hugpy_fleet.central.workers import add_call_success_hook
    add_call_success_hook(on_call_success)
    try:
        from hugpy_platform.constants import PROJECTS_HOME
        lock_path = os.path.join(str(PROJECTS_HOME), "grading_trigger.lock")
    except Exception:  # noqa: BLE001
        lock_path = "/tmp/hugpy-grading-trigger.lock"
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return True                      # another process drains
    threading.Thread(target=_drain_loop, args=(fh,), daemon=True, name="grading-trigger").start()
    log.info("grading trigger: drain loop started (every %ss)", int(DRAIN_S))
    return True
