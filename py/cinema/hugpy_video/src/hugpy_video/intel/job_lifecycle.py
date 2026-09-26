"""k117 — every job finishes or fails; the clock never counts forever.

Operator (2026-08-21): *"it finishes or fails. but never stops counting the
time"* — the Active-Processes feed showed jobs at 875+ minutes in ``loading``,
25h in ``awaiting_capacity``, and DONE rows whose timers kept ticking under a
stale stage label ("archiving").

WHAT WAS ACTUALLY WRONG (keeper triage of the pasted feed, CODE_GAPS.md
"Active-Processes feed + scene-dispatch findings"):

  1. PRESENTATION. The media-bus rows were already correct — a "done" row
     reading 875m in the UI had really run 9.4s. The feed rendered
     ``now - created`` (a ticking AGE) as if it were runtime, and rendered
     ``current_stage`` (the last IN-FLIGHT stage, e.g. "archiving") on a
     terminal row. There is no frozen duration anywhere in the wire shape to
     render instead — this module adds one.
  2. NO STALL DEADLINE. Nothing in the plane bounds how long a job may sit in
     one stage without progressing. A job held for capacity, or wedged mid-load,
     stays in-flight until the 6h ORPHAN sweep (media_bus._reap_orphans) — and
     that sweep only fires on a dead PID or total movement silence, so a job
     that heartbeats while making no progress is immortal.
  3. STALE ADMISSION. A job admitted into a VRAM world it then waited 25h to
     run in blind-ran into that world and OOM'd (CODE_GAPS: *"queued 25h, then
     OOM'd into a different VRAM world than they were admitted in"*).

THE THREE PIECES HERE, in the order the lifecycle meets them:

  * ``on_run_start``  — RE-QUOTE on stale admission (piece 3) + stamp the run
    start, so the queue/run split is a recorded fact rather than a guess.
  * ``sweep``         — the STALL WATCHDOG (piece 2): a per-stage progress
    deadline, run as a periodic reaper from the same throttled hook the orphan
    sweep uses (media_bus._maybe_reap_orphans, called by every runner thread).
  * ``project`` / ``stamp_terminal`` — TERMINAL SEMANTICS (piece 1): the
    frozen ``run_s``/``queue_wait_s``/``terminal_at`` the feed renders.

STORAGE: a SIDECAR table (``media_job_lifecycle``) in the same sqlite file, not
new columns on ``media_jobs``. Two reasons, both deliberate:
  * media_bus.py is a hot, widely-shared file under concurrent edit; a sidecar
    keeps this feature's schema entirely inside its own module.
  * every field here is DERIVABLE (honestly, if more coarsely) from the columns
    media_jobs has always had — ``created``, ``updated`` (stable after the
    terminal write; ``archive()`` deliberately does not bump it) and the
    retained ``stage_log``. So the ~1950 rows that predate this module get the
    same frozen durations without a backfill, and ``project`` never needs the
    sidecar to exist. The sidecar makes the numbers EXACT going forward; the
    derivation makes them HONEST going backward.

NEVER KILLS A WORKER PROCESS. The watchdog marks the JOB failed and releases the
reservation through the existing release path. A worker whose GPU process later
finishes anyway has its result recorded as a ``late_result`` (kept, with the
state conflict noted) rather than dropped — see ``record_late_result``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# sidecar schema
# --------------------------------------------------------------------------- #
_TABLE = "media_job_lifecycle"

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    job_id        TEXT PRIMARY KEY,
    started_at    REAL,      -- when the run actually began (claim -> running)
    quote_json    TEXT,      -- the admission quote this run was started under
    terminal_at   REAL,      -- computed ONCE at the terminal transition
    queue_wait_s  REAL,      -- started_at - created   (frozen)
    run_s         REAL,      -- terminal_at - started_at (frozen)
    terminal_status TEXT,    -- done | failed | cancelled
    at_stage      TEXT,      -- the live stage the job was in when it terminated
    watchdog_json TEXT,      -- the no-progress signature + since-ts (stall basis)
    late_json     TEXT       -- a worker result that landed AFTER a terminal
)
"""

_init_lock = threading.Lock()
_initialized = False


def _bus():
    """The media_bus module, imported LAZILY. media_bus calls into this module
    from four tiny seams; importing it back at module scope would make that a
    circular import. Every function here that touches the DB goes through this."""
    from hugpy_video.intel import media_bus
    return media_bus


def _connect():
    """A read/write handle on the bus DB (same file, same WAL pragmas)."""
    return _bus()._connect()


def _ensure() -> None:
    """Create the sidecar table once per process. Idempotent + best-effort: a
    lifecycle table that cannot be created degrades this module to the pure
    DERIVATION path (``project`` still returns honest frozen durations)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        try:
            conn = _connect()
            try:
                conn.execute(_CREATE_SQL)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 — derivation-only mode; never fatal
            logger.debug("job_lifecycle: sidecar create failed (degrading to "
                         "derivation-only)", exc_info=True)
            return
        _initialized = True


# --------------------------------------------------------------------------- #
# FORK SAFETY (incident 2026-09-24). This sidecar shares media_bus's DB and, like
# it, holds no connection at module scope — the only fork-inherited state is the
# `_initialized` flag, the stall-watchdog throttle clock and the one-shot armed
# log. Drop them in a forked child so it re-creates the sidecar table on a fresh
# handle rather than trusting a pre-fork parent's belief. Belt-and-braces behind
# the real fix (app built in the gunicorn worker). Registered once; never raises.
# --------------------------------------------------------------------------- #
def _reset_after_fork_in_child() -> None:
    global _initialized, _last_sweep_ts, _armed_logged
    _initialized = False
    _last_sweep_ts = 0.0
    _armed_logged = False


try:
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)
except (AttributeError, ValueError):  # non-POSIX / interpreter without the hook
    pass


# --------------------------------------------------------------------------- #
# env knobs — every deadline is tunable; the defaults are the operator's numbers
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# Stage deadlines, in the operator's words:
#   awaiting_capacity 30 min (with a re-quote check)
#   loading           the model's historical load time x3, else 15 min
#   rendering/generating  NO-PROGRESS (no frame/step advance) 20 min
#   archiving         10 min
# A stage absent from this table has NO deadline — the watchdog never invents one
# for a stage it does not understand (the 6h orphan sweep stays the backstop).
_DEADLINE_DEFAULTS: Dict[str, float] = {
    "awaiting_capacity": 1800.0,
    "loading": 900.0,
    "rendering": 1200.0,
    "generating": 1200.0,
    "archiving": 600.0,
    # A plain queued row (no capacity hold recorded) is not "stalled" in any
    # stage — but 25h in the queue is not a queue, it is a leak. Deliberately
    # generous so a legitimately deep queue is never touched.
    "queued": 21600.0,
}

# Which stages are judged by NO-PROGRESS (the frame/step counter has not advanced)
# rather than by raw time-in-stage. A long render legitimately spends hours in
# `generating` — what condemns it is not moving, not being slow.
_NO_PROGRESS_STAGES = frozenset({"rendering", "generating", "assembling"})


def deadline_for_stage(stage: Optional[str], job_name: Optional[str] = None,
                       conn=None) -> Optional[float]:
    """The progress deadline (seconds) for ``stage``, or None when that stage has
    no deadline. Per-stage env override: ``HUGPY_JOB_DEADLINE_<STAGE>_S``.

    ``loading`` is special, as the operator specified: the deadline is this job
    kind's OWN historical load time x ``HUGPY_JOB_LOADING_HISTORY_MULT`` (3),
    measured from the stage timelines this bus already retains — falling back to
    the flat 15 min only when there is not enough history to be honest about."""
    if not stage:
        return None
    key = f"HUGPY_JOB_DEADLINE_{stage.upper()}_S"
    base = _DEADLINE_DEFAULTS.get(stage)
    raw = (os.environ.get(key) or "").strip()
    if raw:
        try:
            v = float(raw)
            # An explicit 0/negative means "no deadline for this stage" — the
            # operator's off switch for one stage without disabling the watchdog.
            return v if v > 0 else None
        except ValueError:
            pass
    if base is None:
        return None
    if stage == "loading":
        hist = historical_stage_seconds(job_name, "loading", conn=conn)
        if hist is not None:
            mult = _env_float("HUGPY_JOB_LOADING_HISTORY_MULT", 3.0)
            return max(base, hist * mult)
    return base


def watchdog_enabled() -> bool:
    """Master switch (``HUGPY_JOB_WATCHDOG``, default ON). Off = this module is
    pure observability: the frozen durations still project, nothing is failed."""
    return _env_flag("HUGPY_JOB_WATCHDOG", True)


def _sweep_interval_s() -> float:
    return _env_float("HUGPY_JOB_WATCHDOG_INTERVAL_S", 60.0)


def _requote_after_s() -> float:
    """How stale an admission quote may be before a run RE-CHECKS VRAM
    feasibility (``HUGPY_JOB_REQUOTE_AFTER_S``, default 900 = 15 min)."""
    return _env_float("HUGPY_JOB_REQUOTE_AFTER_S", 900.0)


def _requote_max() -> int:
    """How many times one job may be returned to the queue by the re-quote gate
    before it is failed typed instead (``HUGPY_JOB_REQUOTE_MAX``, default 2).
    Without this bound a job the fleet can never fit could bounce for ever; with
    it, an infeasible job reaches a terminal like everything else."""
    return max(0, _env_int("HUGPY_JOB_REQUOTE_MAX", 2))


# --------------------------------------------------------------------------- #
# historical stage timings — measured from the bus's own retained timelines
# --------------------------------------------------------------------------- #
def _stage_span(entry: dict, nxt: Optional[dict]) -> Optional[float]:
    """How long a job spent in ONE timeline entry: from its first-seen ``ts`` to
    the NEXT entry's ``ts`` (the moment it left the stage). The last entry has no
    successor and so no measurable span — a stage the job never left is exactly
    the pathology we are trying to size, not evidence of a normal duration."""
    if not isinstance(entry, dict) or not isinstance(nxt, dict):
        return None
    a, b = entry.get("ts"), nxt.get("ts")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return float(b - a) if b > a else None


def historical_stage_seconds(job_name: Optional[str], stage: str,
                             conn=None) -> Optional[float]:
    """The observed p90 seconds this job kind spends in ``stage``, from the last
    ``HUGPY_JOB_HISTORY_ROWS`` (200) terminal rows — or None when fewer than
    ``HUGPY_JOB_LOADING_HISTORY_MIN_SAMPLES`` (3) rows measured it.

    p90 rather than the mean so one fast cache-warm load does not become the
    standard the next COLD load is judged against; x3 on top of that (see
    ``deadline_for_stage``) makes the loading deadline forgiving by construction.
    Only DONE rows are sampled: a failed run's stage spans are the timings of a
    broken run, and we are trying to learn what a healthy one costs."""
    if not job_name:
        return None
    need = max(1, _env_int("HUGPY_JOB_LOADING_HISTORY_MIN_SAMPLES", 3))
    rows_limit = _env_int("HUGPY_JOB_HISTORY_ROWS", 200)
    own = conn is None
    try:
        if own:
            conn = _bus()._connect_ro()
        rows = conn.execute(
            "SELECT stage_log_json FROM media_jobs WHERE name=? AND status='done' "
            "ORDER BY updated DESC LIMIT ?", (job_name, rows_limit),
        ).fetchall()
    except Exception:  # noqa: BLE001 — no history is a fine answer
        return None
    finally:
        if own and conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    spans: List[float] = []
    for (raw,) in rows:
        log = _bus()._load_stage_log(raw)
        for i, e in enumerate(log):
            if e.get("stage") != stage:
                continue
            span = _stage_span(e, log[i + 1] if i + 1 < len(log) else None)
            if span is not None:
                spans.append(span)
    if len(spans) < need:
        return None
    spans.sort()
    idx = min(len(spans) - 1, int(round(0.9 * (len(spans) - 1))))
    return spans[idx]


# --------------------------------------------------------------------------- #
# sidecar read/write
# --------------------------------------------------------------------------- #
_SIDECAR_COLS = ("started_at", "quote_json", "terminal_at", "queue_wait_s",
                 "run_s", "terminal_status", "at_stage", "watchdog_json",
                 "late_json")


def _row_to_side(row) -> Dict[str, Any]:
    return {k: row[i] for i, k in enumerate(_SIDECAR_COLS)} if row else {}


def read_side(job_id: str, conn=None) -> Dict[str, Any]:
    """The sidecar record for one job ({} when absent — the normal case for every
    row that predates this module)."""
    _ensure()
    own = conn is None
    try:
        if own:
            conn = _bus()._connect_ro()
        row = conn.execute(
            f"SELECT {', '.join(_SIDECAR_COLS)} FROM {_TABLE} WHERE job_id=?",
            (job_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return {}
    finally:
        if own and conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return _row_to_side(row)


def read_side_many(job_ids: List[str], conn=None) -> Dict[str, Dict[str, Any]]:
    """Sidecar records for a whole listing PAGE in one query — the listing is a
    hot 2s poll and k57's whole lesson was that per-row work on it is what hung
    the feed. Missing ids simply do not appear (they derive)."""
    if not job_ids:
        return {}
    _ensure()
    own = conn is None
    marks = ",".join("?" * len(job_ids))
    try:
        if own:
            conn = _bus()._connect_ro()
        rows = conn.execute(
            f"SELECT job_id, {', '.join(_SIDECAR_COLS)} FROM {_TABLE} "
            f"WHERE job_id IN ({marks})", tuple(job_ids),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    finally:
        if own and conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return {r[0]: _row_to_side(r[1:]) for r in rows}


def _upsert(job_id: str, **fields) -> None:
    """Write the named sidecar fields, creating the row if needed. Fully
    swallowed: the sidecar is an accelerator over a derivation that always
    works, so a write failure degrades precision, never correctness."""
    if not fields:
        return
    _ensure()
    cols = list(fields)
    try:
        conn = _connect()
        try:
            conn.execute(f"INSERT OR IGNORE INTO {_TABLE} (job_id) VALUES (?)",
                         (job_id,))
            conn.execute(
                f"UPDATE {_TABLE} SET {', '.join(c + '=?' for c in cols)} "
                f"WHERE job_id=?",
                tuple(fields[c] for c in cols) + (job_id,),
            )
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        logger.debug("job_lifecycle: sidecar write failed for %s", job_id,
                     exc_info=True)


# --------------------------------------------------------------------------- #
# (1) TERMINAL SEMANTICS — computed ONCE at the terminal transition
# --------------------------------------------------------------------------- #
def stamp_start(job_id: str, created: Optional[float] = None,
                quote: Optional[dict] = None) -> float:
    """Record that the run BEGAN now (idempotent — a re-entered run_claimed, e.g.
    after a re-quote, keeps the ORIGINAL start so ``run_s`` never shrinks).
    Returns the effective started_at."""
    side = read_side(job_id)
    existing = side.get("started_at")
    if isinstance(existing, (int, float)):
        if quote is not None:
            _upsert(job_id, quote_json=json.dumps(quote))
        return float(existing)
    now = time.time()
    fields: Dict[str, Any] = {"started_at": now}
    if quote is not None:
        fields["quote_json"] = json.dumps(quote)
    if isinstance(created, (int, float)):
        fields["queue_wait_s"] = max(0.0, now - float(created))
    _upsert(job_id, **fields)
    return now


def stamp_terminal(job_id: str, status: str, at_stage: Optional[str] = None,
                   created: Optional[float] = None) -> None:
    """Freeze this job's clock. Computed ONCE: a second call (a late worker write,
    a re-run of the terminal path) is a NO-OP and can never re-time a finished
    job. ``run_s`` = terminal_at - started_at; ``queue_wait_s`` = started_at -
    created. A job that terminated before it ever started (a queued cancel, an
    admission refusal) honestly gets run_s = 0 and the whole span as queue wait."""
    _ensure()
    side = read_side(job_id)
    if isinstance(side.get("terminal_at"), (int, float)):
        return                                  # already frozen — never re-time
    now = time.time()
    started = side.get("started_at")
    if created is None:
        created = _row_created(job_id)
    fields: Dict[str, Any] = {"terminal_at": now, "terminal_status": status}
    if at_stage:
        fields["at_stage"] = at_stage
    if isinstance(started, (int, float)):
        fields["run_s"] = max(0.0, now - float(started))
        if isinstance(created, (int, float)):
            fields["queue_wait_s"] = max(0.0, float(started) - float(created))
    else:
        fields["run_s"] = 0.0
        if isinstance(created, (int, float)):
            fields["queue_wait_s"] = max(0.0, now - float(created))
    _upsert(job_id, **fields)


def _row_created(job_id: str) -> Optional[float]:
    try:
        conn = _bus()._connect_ro()
        try:
            row = conn.execute("SELECT created FROM media_jobs WHERE job_id=?",
                               (job_id,)).fetchone()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None
    return row[0] if row else None


def record_late_result(job_id: str, status: str, result_json: Optional[str],
                       worker_token: Optional[str] = None) -> None:
    """A worker finished AFTER the job was already terminalized (by this
    watchdog, by the orphan sweep, or by a cancel). Its terminal write missed —
    run_claimed's UPDATE is gated ``AND claim_token=?`` against a token the reaper
    NULLed, which is the invariant that stops a late finisher from overwriting an
    honest terminal.

    That invariant is right, but silently DROPPING the worker's answer is not:
    the artifact usually exists on disk. So we keep the result verbatim here and
    note the state conflict. Nothing about the job's frozen terminal changes —
    this is an addendum, not a correction."""
    payload = {
        "at": time.time(),
        "reported_status": status,
        "worker": worker_token,
        "note": ("worker reported a terminal AFTER the job had already been "
                 "terminalized centrally; the central terminal stands and this "
                 "result is retained as an addendum (artifact, if any, is kept)"),
    }
    if result_json:
        try:
            payload["result"] = json.loads(result_json)
        except Exception:  # noqa: BLE001 — keep it raw rather than lose it
            payload["result_raw"] = str(result_json)[:20000]
    _upsert(job_id, late_json=json.dumps(payload))
    logger.warning("job_lifecycle: LATE result for %s (worker said %s after the "
                   "central terminal) — recorded as late_result, not dropped",
                   job_id, status)
    try:
        _bus()._append_stage_log(
            job_id, "late_result",
            f"worker reported {status} after the central terminal — result kept")
    except Exception:  # noqa: BLE001 — timeline is observability
        pass


# --------------------------------------------------------------------------- #
# projection — what the feed/API actually renders
# --------------------------------------------------------------------------- #
def _first_live_stage_ts(stage_log) -> Optional[float]:
    """The ts of the first NON-terminal timeline entry — the derived run start for
    a row with no sidecar (every pre-k117 row). A job's first set_progress lands
    within a fraction of a second of dispatch, so this is a tight lower bound."""
    for e in (stage_log or []):
        s = e.get("stage")
        if (s and s not in _bus()._TERMINAL_STATES and s not in _ADDENDUM_STAGES
                and s not in ("awaiting_capacity", "queued")):
            ts = e.get("ts")
            if isinstance(ts, (int, float)):
                return float(ts)
    return None


# Timeline entries that are ADDENDA to a finished job, not stages it is "in".
# Without this a job whose worker reported late would read as currently being in
# stage "late_result" forever — the exact class of lie k117 exists to remove.
_ADDENDUM_STAGES = frozenset({"late_result"})


def _terminal_entry(stage_log) -> Optional[dict]:
    for e in reversed(stage_log or []):
        if e.get("stage") in _bus()._TERMINAL_STATES:
            return e
    return None


def _current_stage_entry(stage_log) -> Optional[dict]:
    """The timeline entry for the stage the job is CURRENTLY in (the last
    non-terminal row) — carries both ``ts`` (stage START, the in-stage elapsed
    basis) and ``ts_last`` (last movement)."""
    for e in reversed(stage_log or []):
        s = e.get("stage")
        if s and s not in _bus()._TERMINAL_STATES and s not in _ADDENDUM_STAGES:
            return e
    return None


def project(row: dict, *, side: Optional[dict] = None,
            now: Optional[float] = None) -> dict:
    """Enrich ONE projected bus row with k117's lifecycle block, in place.

    TERMINAL rows get FROZEN numbers — ``terminal_at``, ``run_s``,
    ``queue_wait_s``, ``total_s`` and ``terminal_stage`` (the true outcome:
    done/failed/cancelled), plus ``at_stage`` (the live stage it was in when it
    ended, e.g. "archiving" — kept, but never again mistaken for a CURRENT
    stage). Nothing on a terminal row is a function of ``now``, so the clock
    stops the instant the job does.

    NON-TERMINAL rows get ``elapsed_in_stage_s`` (time in the CURRENT stage) and
    ``last_progress_at``, with ``queue_wait_s`` and ``run_s`` reported
    SEPARATELY — the feed's old single ``now - created`` conflated a 25h queue
    wait with a 9s render, which is how a 9s job read as 875 minutes.

    Pure + read-only. Works with or without the sidecar: when there is no
    sidecar record (every row created before this module) the same fields are
    DERIVED from ``created``/``updated``/``stage_log``."""
    now = time.time() if now is None else now
    side = side if side is not None else {}
    stage_log = row.get("stage_log") or []
    created = row.get("created")
    updated = row.get("updated")
    status = row.get("status")
    terminal = status in _bus()._TERMINAL_STATES

    started_at = side.get("started_at")
    if not isinstance(started_at, (int, float)):
        started_at = _first_live_stage_ts(stage_log)

    row["started_at"] = started_at
    row["terminal"] = bool(terminal)

    if terminal:
        term_at = side.get("terminal_at")
        if not isinstance(term_at, (int, float)):
            te = _terminal_entry(stage_log)
            ts = te.get("ts") if te else None
            # `updated` is written by the terminal UPDATE and nothing bumps it
            # afterwards (archive() deliberately touches only archived_at), so it
            # is a stable terminal clock for every pre-k117 row.
            term_at = float(ts) if isinstance(ts, (int, float)) else updated
        run_s = side.get("run_s")
        queue_wait = side.get("queue_wait_s")
        if not isinstance(run_s, (int, float)):
            if (isinstance(term_at, (int, float))
                    and isinstance(started_at, (int, float))):
                run_s = max(0.0, float(term_at) - float(started_at))
            elif status == "cancelled":
                # Cancelled with no live stage ever recorded = cancelled BEFORE it
                # started. The run really did take zero seconds and the whole span
                # is queue wait; that is a fact, not a guess.
                run_s = 0.0
            else:
                # A pre-k117 row whose runner never wrote a single stage entry
                # (studio_tester is the live example). We know the TOTAL span and
                # nothing about the split — so we say exactly that. Asserting
                # run_s = 0 for a job that plainly ran would be a new lie in place
                # of the old one; `total_s` carries the honest number and the feed
                # renders it as "took …" rather than "ran …".
                run_s = None
        if not isinstance(queue_wait, (int, float)):
            if isinstance(started_at, (int, float)) and isinstance(created, (int, float)):
                queue_wait = max(0.0, float(started_at) - float(created))
            elif (run_s == 0.0 and isinstance(term_at, (int, float))
                    and isinstance(created, (int, float))):
                queue_wait = max(0.0, float(term_at) - float(created))
            else:
                queue_wait = None
        row["terminal_at"] = term_at
        row["run_s"] = round(float(run_s), 3) if run_s is not None else None
        row["queue_wait_s"] = (round(float(queue_wait), 3)
                               if queue_wait is not None else None)
        row["total_s"] = (round(max(0.0, float(term_at) - float(created)), 3)
                          if isinstance(term_at, (int, float))
                          and isinstance(created, (int, float)) else None)
        # The TRUE terminal stage: what the job ended as. `current_stage` stays
        # exactly what it always was (the last live stage) so nothing that reads
        # it breaks — but a terminal row now also says, unambiguously, that it is
        # over and what it ended as.
        row["terminal_stage"] = side.get("terminal_status") or status
        at_stage = side.get("at_stage") or row.get("current_stage")
        row["at_stage"] = at_stage
        row["elapsed_in_stage_s"] = None
        row["last_progress_at"] = row.get("progressed_at")
        late = side.get("late_json")
        if late:
            try:
                row["late_result"] = json.loads(late)
            except Exception:  # noqa: BLE001
                pass
        return row

    # ---- non-terminal: the two clocks, kept apart ----
    row["terminal_at"] = None
    row["terminal_stage"] = None
    cur = _current_stage_entry(stage_log)
    row["at_stage"] = cur.get("stage") if cur else None
    stage_ts = cur.get("ts") if cur else None
    if isinstance(stage_ts, (int, float)):
        row["elapsed_in_stage_s"] = round(max(0.0, now - float(stage_ts)), 3)
    elif isinstance(started_at, (int, float)):
        row["elapsed_in_stage_s"] = round(max(0.0, now - float(started_at)), 3)
    else:
        row["elapsed_in_stage_s"] = (round(max(0.0, now - float(created)), 3)
                                     if isinstance(created, (int, float)) else None)
    row["last_progress_at"] = row.get("progressed_at")
    if isinstance(started_at, (int, float)):
        row["queue_wait_s"] = (round(max(0.0, float(started_at) - float(created)), 3)
                               if isinstance(created, (int, float)) else None)
        row["run_s"] = round(max(0.0, now - float(started_at)), 3)
    else:
        # Still queued: it has waited, but it has not RUN for a single second.
        row["queue_wait_s"] = (round(max(0.0, now - float(created)), 3)
                               if isinstance(created, (int, float)) else None)
        row["run_s"] = 0.0
    return row


def project_all(rows: List[dict], now: Optional[float] = None) -> List[dict]:
    """``project`` over a whole listing page, with ONE sidecar query for the page."""
    try:
        sides = read_side_many([r.get("job_id") for r in rows if r.get("job_id")])
    except Exception:  # noqa: BLE001 — derive-only is a complete fallback
        sides = {}
    now = time.time() if now is None else now
    for r in rows:
        try:
            project(r, side=sides.get(r.get("job_id")) or {}, now=now)
        except Exception:  # noqa: BLE001 — one bad row never breaks the feed
            logger.debug("job_lifecycle: projection failed for %s",
                         r.get("job_id"), exc_info=True)
    return rows


# --------------------------------------------------------------------------- #
# (3) RE-QUOTE ON STALE ADMISSION
# --------------------------------------------------------------------------- #
def _requote(job_name: str, job_id: str) -> Tuple[bool, Optional[dict]]:
    """Re-run the reservation engine's non-destructive fit probe. Fail-OPEN
    (admit) exactly like media_bus._probe_admission — a probe hiccup must never
    be the thing that stops a render."""
    try:
        from hugpy_video.intel.reservation import can_admit
        return can_admit(job_name, None, run_id=job_id)
    except Exception:  # noqa: BLE001
        return True, None


def _requeue(job_id: str, worker_token: str, reason: Optional[dict],
             attempts: int) -> bool:
    """Put a claimed job BACK in the queue with a fresh quote, CAS-gated on our
    own claim token so we can only ever unwind OUR claim. Returns True iff the
    job was requeued (the caller must then not run it)."""
    try:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE media_jobs SET status='queued', claim_token=NULL, "
                "updated=? WHERE job_id=? AND claim_token=? AND "
                "status IN ('claimed','running')",
                (time.time(), job_id, worker_token),
            )
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — could not unwind: let the run proceed
        logger.debug("job_lifecycle: requeue failed for %s", job_id, exc_info=True)
        return False
    if cur.rowcount != 1:
        return False
    _upsert(job_id, quote_json=json.dumps(
        {"at": time.time(), "admit": False, "reason": reason,
         "requeues": attempts + 1}))
    try:
        _bus()._mark_awaiting_capacity(job_id, reason, None, 0)
    except Exception:  # noqa: BLE001
        pass
    logger.info("job_lifecycle: %s re-quoted INFEASIBLE at run start — returned to "
                "the queue with a fresh quote (requeue %d/%d)",
                job_id, attempts + 1, _requote_max())
    return True


def on_run_start(job_id: str, job_name: str, worker_token: str,
                 created: Optional[float] = None) -> bool:
    """The seam media_bus.run_claimed calls the instant a job goes 'running'.

    Does two things and returns whether the run may PROCEED:

      1. RE-QUOTE (CODE_GAPS: *"long-queued jobs should re-quote before running
         (queued 25h, then OOM'd into a different VRAM world than they were
         admitted in)"*). A job whose admission quote is older than
         ``HUGPY_JOB_REQUOTE_AFTER_S`` (15 min) re-checks VRAM feasibility
         BEFORE its runner touches the card. Infeasible ⇒ back to the queue with
         a fresh quote — and after ``HUGPY_JOB_REQUOTE_MAX`` bounces, a typed
         ``gpu_unavailable`` terminal. It NEVER blind-runs.
      2. STAMP the run start so queue-wait and run-time are recorded facts.

    Fail-OPEN in every direction: any error here proceeds with the run, because
    an admission-check bug must not be able to wedge dispatch (media_bus's own
    admission seam is built on the same rule)."""
    try:
        if created is None:
            created = _row_created(job_id)
        side = read_side(job_id)
        quote = {}
        if side.get("quote_json"):
            try:
                quote = json.loads(side["quote_json"]) or {}
            except Exception:  # noqa: BLE001
                quote = {}
        attempts = int(quote.get("requeues") or 0)
        # The quote's age. With no recorded quote the job's own ENQUEUE is the
        # last moment anybody sized it — which is precisely the 25h case.
        quoted_at = quote.get("at")
        if not isinstance(quoted_at, (int, float)):
            quoted_at = created
        age = (time.time() - float(quoted_at)) if isinstance(quoted_at, (int, float)) else 0.0
        # Re-probe when the quote is STALE — or when the last quote we have said
        # this job does NOT fit. The second clause is load-bearing: a requeue
        # writes a fresh-timestamped INFEASIBLE quote, so without it the very
        # next claim would see a "recent" quote and blind-run the job we just
        # refused. The last thing we knew was that it does not fit; that has to
        # be re-established, not aged out.
        if age > _requote_after_s() or quote.get("admit") is False:
            admit, reason = _requote(job_name, job_id)
            if not admit:
                if attempts < _requote_max():
                    if _requeue(job_id, worker_token, reason, attempts):
                        return False
                else:
                    _fail_typed(
                        job_id, job_name, "running",
                        code="gpu_unavailable",
                        message=(
                            f"admission re-quote failed {attempts + 1} time(s) before "
                            f"this run started: the quote was {int(age)}s old and the "
                            f"card no longer fits this job. "
                            f"{(reason or {}).get('reason') or ''}".strip()),
                        extra={"requote": reason, "quote_age_s": round(age, 1),
                               "requeues": attempts},
                        worker_token=worker_token)
                    return False
            _upsert(job_id, quote_json=json.dumps(
                {"at": time.time(), "admit": bool(admit), "reason": reason,
                 "requeues": attempts, "revalidated": True,
                 "prior_quote_age_s": round(age, 1)}))
        stamp_start(job_id, created)
    except Exception:  # noqa: BLE001 — never let this gate break a run
        logger.debug("job_lifecycle: on_run_start raised (proceeding)",
                     exc_info=True)
    return True


# --------------------------------------------------------------------------- #
# (2) STALL WATCHDOG
# --------------------------------------------------------------------------- #
def _progress_signature(row_progress) -> Optional[str]:
    """A stable string over the ADVANCE markers a running job reports — segment,
    step, frame, fraction. When it CHANGES the job has genuinely progressed and
    its deadline resets. None when the runner reports nothing measurable at all,
    in which case the caller falls back to the timeline's ``ts_last``.

    This is the difference between "slow" and "stalled": a 40-minute segment that
    keeps advancing its step counter is healthy; the same 40 minutes with the
    counter frozen is the job the operator watched count to 875 minutes."""
    bus = _bus()
    detail = bus._progress_detail(row_progress)
    ratio = bus._progress_ratio(row_progress)
    bits: List[str] = []
    if isinstance(detail, dict):
        for k in ("segment_done", "segment_total", "step", "steps", "fraction"):
            if k in detail:
                bits.append(f"{k}={detail[k]}")
    if isinstance(ratio, (int, float)):
        bits.append(f"r={round(float(ratio), 6)}")
    if not bits:
        return None
    return "|".join(bits)


def _worker_last_error(progress) -> Optional[str]:
    """The worker's OWN last reported error/log line, read verbatim out of the
    live progress blob — never invented. The feed already renders these
    (``log_tail`` from the relay runners, a nested ``current.worker`` blob), so a
    watchdog failure can carry the real last thing the worker said instead of a
    generic "it stalled"."""
    if not isinstance(progress, dict):
        return None
    scopes = [progress]
    cur = progress.get("current")
    if isinstance(cur, dict):
        scopes.append(cur)
        w = cur.get("worker")
        if isinstance(w, dict):
            scopes.append(w)
    for sc in scopes:
        for key in ("error", "last_error", "err", "message"):
            v = sc.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:500]
            if isinstance(v, dict):
                m = v.get("message") or v.get("error")
                if isinstance(m, str) and m.strip():
                    return m.strip()[:500]
    for sc in scopes:
        tail = sc.get("log_tail")
        if isinstance(tail, list) and tail:
            last = tail[-1]
            if isinstance(last, str) and last.strip():
                return last.strip()[:500]
        if isinstance(tail, str) and tail.strip():
            return tail.strip().splitlines()[-1][:500]
    return None


def _fail_typed(job_id: str, name: Optional[str], observed_status: str, *,
                code: str, message: str, extra: Optional[dict] = None,
                worker_token: Optional[str] = None) -> bool:
    """Terminalize ONE job as failed, with the same write discipline as the
    orphan sweep (media_bus._reap_one), which this deliberately mirrors:

      * COMPARE-AND-SWAP on the OBSERVED status — a row that moved under us
        (a real runner got there first) is left completely alone;
      * NULL the claim_token, so a late finisher's ``AND claim_token=?`` terminal
        write MISSES rather than overwriting this honest terminal (it is then
        recorded as a ``late_result`` — see ``record_late_result``);
      * then three independently-guarded after-effects: the retained stage-log
        entry, the RESERVATION RELEASE through the existing release path, and the
        JobStore bridge.

    It does NOT touch the worker. No process is signalled, no GPU is cleared —
    the job is marked failed and its reservation released; whatever the worker is
    doing, it keeps doing until it stops on its own."""
    bus = _bus()
    from hugpy_video.intel.result_schema import JobError, JobResult
    result = JobResult(job_id=job_id, ok=False,
                       error=JobError(code=code, message=message, retryable=True))
    try:
        conn = bus._connect()
        try:
            cur = conn.execute(
                "UPDATE media_jobs SET status='failed', result_json=?, "
                "claim_token=NULL, progress_json=NULL, updated=? "
                "WHERE job_id=? AND status=?",
                (bus.serialize_result(result), time.time(), job_id,
                 observed_status),
            )
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        logger.debug("job_lifecycle: watchdog terminal write failed for %s",
                     job_id, exc_info=True)
        return False
    if cur.rowcount != 1:
        logger.debug("job_lifecycle: %s moved from %s under the watchdog — skipped",
                     job_id, observed_status)
        return False
    logger.warning("job_lifecycle watchdog: %s (%s) %s -> failed [%s] %s",
                   job_id, name, observed_status, code, message)
    stage_extra = {"code": code, "message": message, "retryable": True,
                   "failed_by": "stall_watchdog", "prior_status": observed_status}
    if extra:
        stage_extra.update(extra)
    at_stage = (extra or {}).get("stage")
    if at_stage:
        stage_extra["failed_at_stage"] = at_stage
    try:
        bus._append_stage_log(job_id, "failed", message,
                              terminal_extra=stage_extra)
    except Exception:  # noqa: BLE001
        pass
    try:
        stamp_terminal(job_id, "failed", at_stage=at_stage)
    except Exception:  # noqa: BLE001
        pass
    try:
        # The job is over centrally, so its GPU claim must not outlive it. Same
        # release path every terminal uses; idempotent by contract.
        bus._release_reservation(job_id)
    except Exception:  # noqa: BLE001
        logger.debug("job_lifecycle: reservation release failed for %s", job_id,
                     exc_info=True)
    try:
        bus._bridge("on_terminal", job_id, name or "media", "failed", result=result)
    except Exception:  # noqa: BLE001
        pass
    return True


def _stall_verdict(status: str, name: Optional[str], created: Optional[float],
                   progress, stage_log, side: Dict[str, Any], now: float,
                   conn=None) -> Tuple[bool, Optional[str], Optional[float],
                                       Optional[float], Optional[str]]:
    """``(stalled, stage, elapsed_s, deadline_s, basis)`` for ONE in-flight row.

    Two kinds of deadline, deliberately different (the operator specified both):
      * TIME-IN-STAGE for ``awaiting_capacity`` / ``loading`` / ``archiving`` /
        ``queued`` — being in that stage at all is what is bounded. A model that
        has been "loading" for 875 minutes is not slow, it is gone.
      * NO-PROGRESS for ``rendering`` / ``generating`` / ``assembling`` — bounded
        by time since the frame/step counter last ADVANCED, so a long render that
        is genuinely working is never touched no matter how long it takes.

    When a no-progress stage reports nothing measurable, the basis falls back to
    the timeline's ``ts_last`` (the last set_progress). That is the safe
    direction and it is the repo's existing bias: a job that is still reporting
    is still alive as far as this plane can honestly tell, and the 6h orphan
    sweep remains the backstop."""
    bus = _bus()
    cur_entry = _current_stage_entry(stage_log)
    stage = cur_entry.get("stage") if cur_entry else None
    if stage is None and status == "queued":
        stage = "queued"                    # queued, never held, never started
    if stage is None:
        return (False, None, None, None, None)   # no stage recorded — never guess
    deadline = deadline_for_stage(stage, name, conn=conn)
    if deadline is None:
        return (False, stage, None, None, None)

    if stage in _NO_PROGRESS_STAGES:
        sig = _progress_signature(progress)
        wd = {}
        raw = side.get("watchdog_json")
        if raw:
            try:
                wd = json.loads(raw) or {}
            except Exception:  # noqa: BLE001
                wd = {}
        if sig is None:
            basis_ts = (cur_entry or {}).get("ts_last") or (cur_entry or {}).get("ts")
            basis = "ts_last (runner reports nothing measurable)"
        else:
            if wd.get("stage") == stage and wd.get("sig") == sig:
                basis_ts = wd.get("since")
            else:
                basis_ts = None             # signature moved (or is new) -> reset
            basis = "no frame/step advance"
    else:
        basis_ts = (cur_entry.get("ts") if cur_entry
                    else (created if stage == "queued" else None))
        if stage == "queued":
            basis_ts = created
        basis = "time in stage"
    if not isinstance(basis_ts, (int, float)):
        return (False, stage, None, deadline, basis)
    elapsed = max(0.0, now - float(basis_ts))
    return (elapsed > deadline, stage, elapsed, deadline, basis)


_last_sweep_ts = 0.0
_sweep_lock = threading.Lock()
_armed_logged = False

_WATCHABLE_STATES = ("queued", "claimed", "running")


def sweep() -> int:
    """One watchdog pass over the IN-FLIGHT rows. Returns how many jobs it failed.

    'cancelling' rows are deliberately NOT watched here — the orphan sweep owns
    them, with its own (short, 30 min) unhonored-cancel window.

    Read-then-write, one row at a time, each write a CAS on the observed status.
    Fully swallowed by the caller: a watchdog is a janitor, and a janitor must
    never be able to kill a runner thread."""
    if not watchdog_enabled():
        return 0
    _ensure()
    bus = _bus()
    bus._ensure_db()
    now = time.time()
    _log_armed()
    try:
        conn = bus._connect_ro()
        try:
            rows = conn.execute(
                "SELECT job_id, name, status, created, progress_json, "
                "stage_log_json FROM media_jobs WHERE status IN (?,?,?)",
                _WATCHABLE_STATES,
            ).fetchall()
            sides = read_side_many([r[0] for r in rows], conn=conn)
            verdicts = []
            for job_id, name, status, created, pj, slj in rows:
                progress = bus._load_progress(pj)
                stage_log = bus._load_stage_log(slj)
                stalled, stage, elapsed, deadline, basis = _stall_verdict(
                    status, name, created, progress, stage_log,
                    sides.get(job_id) or {}, now, conn=conn)
                verdicts.append((job_id, name, status, stage, elapsed, deadline,
                                 basis, stalled, progress))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        logger.debug("job_lifecycle: watchdog scan failed", exc_info=True)
        return 0

    failed = 0
    for (job_id, name, status, stage, elapsed, deadline, basis, stalled,
         progress) in verdicts:
        if not stalled:
            # Healthy: refresh the no-progress signature so the NEXT sweep can
            # tell "advanced since last time" from "frozen since last time".
            if stage in _NO_PROGRESS_STAGES:
                sig = _progress_signature(progress)
                if sig is not None:
                    _touch_signature(job_id, stage, sig, now, sides.get(job_id) or {})
            continue
        werr = _worker_last_error(progress)
        msg = (f"stage '{stage}' exceeded its progress deadline: "
               f"{int(elapsed)}s with no progress (limit {int(deadline)}s, "
               f"basis: {basis}). The job was marked failed centrally and its GPU "
               f"reservation released; no worker process was touched.")
        if werr:
            msg += f" Worker's last reported error: {werr}"
        extra = {"stage": stage, "elapsed_s": round(float(elapsed), 1),
                 "deadline_s": round(float(deadline), 1), "basis": basis}
        if werr:
            extra["worker_last_error"] = werr
        if stage == "awaiting_capacity":
            # The operator asked for a re-quote check on this stage specifically:
            # say, in the failure, whether the card CAN fit this job now — the
            # difference between "the fleet is busy" and "this will never fit".
            admit, reason = _requote(name, job_id)
            extra["requote"] = {"admit": bool(admit), "reason": reason}
            msg += (" Re-quote at deadline: "
                    + ("the card fits this job now, so the hold was not capacity "
                       "— it never got claimed." if admit
                       else f"still infeasible ({(reason or {}).get('reason') or 'no reason reported'})."))
        if _fail_typed(job_id, name, status, code="stage_deadline_exceeded",
                       message=msg, extra=extra):
            failed += 1
    if failed:
        logger.warning("job_lifecycle watchdog: failed %d stalled job(s) out of "
                       "%d in-flight scanned", failed, len(verdicts))
    _write_heartbeat(now, len(verdicts), failed)
    return failed


# The watchdog's DURABLE proof-of-life. `_log_armed` writes an INFO line, but this
# deployment runs the video_intel loggers above INFO, so "is it actually armed?"
# had no answer an operator could check. This row does: a reserved sidecar key
# (it can never collide with a job id) carrying the last sweep's timestamp, what
# it scanned, what it failed, and the deadlines it is enforcing. Any process —
# a console probe, a route, a psql-style poke at the DB — can read it.
_HEARTBEAT_KEY = "__watchdog__"


def _write_heartbeat(now: float, scanned: int, failed: int) -> None:
    try:
        _upsert(_HEARTBEAT_KEY, watchdog_json=json.dumps({
            "last_sweep": now, "scanned": scanned, "failed": failed,
            "pid": os.getpid(), "interval_s": _sweep_interval_s(),
            "requote_after_s": _requote_after_s(),
            "deadlines": {st: deadline_for_stage(st) for st in _DEADLINE_DEFAULTS},
        }))
    except Exception:  # noqa: BLE001 — proof-of-life is never load-bearing
        pass


def heartbeat() -> Optional[dict]:
    """The last sweep's durable record, or None if this bus has never swept."""
    raw = read_side(_HEARTBEAT_KEY).get("watchdog_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def _touch_signature(job_id: str, stage: str, sig: str, now: float,
                     side: Dict[str, Any]) -> None:
    """Persist the current advance signature for a healthy no-progress-watched
    job. Only writes when the signature actually CHANGED, so a saturated pool
    does not turn the watchdog into a per-sweep writer on every running row."""
    raw = side.get("watchdog_json")
    if raw:
        try:
            wd = json.loads(raw) or {}
            if wd.get("stage") == stage and wd.get("sig") == sig:
                return                       # unchanged — nothing to write
        except Exception:  # noqa: BLE001
            pass
    _upsert(job_id, watchdog_json=json.dumps(
        {"stage": stage, "sig": sig, "since": now}))


def _log_armed() -> None:
    """Say ONCE per process that the watchdog is armed and with what deadlines —
    the operator's "is it actually on?" answer, greppable in the api log."""
    global _armed_logged
    if _armed_logged:
        return
    _armed_logged = True
    try:
        bits = []
        for st in sorted(_DEADLINE_DEFAULTS):
            d = deadline_for_stage(st)
            kind = "no-progress" if st in _NO_PROGRESS_STAGES else "in-stage"
            bits.append(f"{st}={'off' if d is None else str(int(d)) + 's'}({kind})")
        logger.info("job_lifecycle watchdog ARMED (interval %ds, re-quote after "
                    "%ds, max %d requeues): %s", int(_sweep_interval_s()),
                    int(_requote_after_s()), _requote_max(), " ".join(bits))
    except Exception:  # noqa: BLE001
        pass


def maybe_sweep() -> None:
    """The THROTTLED hook media_bus._maybe_reap_orphans calls every runner pass.
    Same shape as the orphan sweep's throttle: the lock guards only the timestamp
    claim, so a non-winning thread returns instantly and the winner scans outside
    the lock. Fully swallowed."""
    global _last_sweep_ts
    try:
        now = time.time()
        with _sweep_lock:
            if now - _last_sweep_ts < _sweep_interval_s():
                return
            _last_sweep_ts = now
        sweep()
    except Exception:  # noqa: BLE001
        logger.debug("job_lifecycle: watchdog sweep raised (non-fatal)",
                     exc_info=True)


def status() -> dict:
    """A machine-readable "is the watchdog armed, and with what?" — for a status
    route or a console probe."""
    hb = heartbeat() or {}
    return {
        "armed": watchdog_enabled(),
        "interval_s": _sweep_interval_s(),
        # In-process clock first; the DURABLE cross-process heartbeat as the
        # fallback, so a fresh process can still answer "when did this bus last
        # sweep?" without having swept itself yet.
        "last_sweep_ts": _last_sweep_ts or hb.get("last_sweep"),
        "last_sweep": hb or None,
        "requote_after_s": _requote_after_s(),
        "requote_max": _requote_max(),
        "deadlines": {st: deadline_for_stage(st) for st in _DEADLINE_DEFAULTS},
        "no_progress_stages": sorted(_NO_PROGRESS_STAGES),
    }
