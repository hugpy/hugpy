"""Studio-assist attempt log — what each prompt-generate attempt actually returned.

WHY THIS EXISTS (operator directive, 2026-07-31): studio prompt generation keeps
failing with things like "the assistant returned only reasoning and no output"
and "the generator did not return the JSON object the spread contract requires",
and the only trace of WHY was scattered ``logger`` lines the operator had to ask
the keeper to read. This module turns each generate attempt into a stored record
— the UNTRUNCATED model reply, what was stripped from it, and the outcome — and
streams it to the studio UI so the operator can self-diagnose in real time.

    the raw reply the model returned  (untruncated — this is the whole point)
    the think-stripped ``text`` that would have been used as the prompt
    the ``reasoning`` that was stripped out, and whether the answer came FROM it
    the ``outcome`` — served / empty / parse_error / worker_error / resolve_error
    the model asked-for vs. the one that actually answered, and how long it took

This is a direct sibling of ``comms/evictions.py`` (the eviction-telemetry
store): same bounded in-process ring, same sqlite mirror keyed by rowid so the
SSE stream is correct across gunicorn workers, same ``get_store()`` /
``.append()`` / ``.recent()`` / ``.max_id()`` surface, same "off"-sentinel and
EMFILE hardening. It differs in one way that matters: studio assist runs on
CENTRAL (the route captures ``execute_prompt``'s result in-process), so there is
no worker relay — central emits straight into its own store.

OBSERVATION ONLY. Nothing here may change what a generate attempt returns. Every
entry point is wrapped so that ANY failure — a full ring, a dead sqlite, a bad
record — is swallowed and the attempt proceeds exactly as it would have without
the log. Telemetry never raises into the serve path.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome vocabulary. One string per way a generate attempt can end. The keeper
# owns nomenclature (doctrine): classify at the call site with these constants
# rather than inventing a synonym.
#
#   served        a usable prompt/spread came back and parsed
#   empty         the reply stripped to neither prose NOR reasoning (genuine void)
#   parse_error   text came back but did not satisfy the contract (spread JSON)
#   worker_error  the fleet/worker could not produce a reply (502-class)
#   resolve_error the request was caller-fixable (unknown model / bad task; 400)
# ---------------------------------------------------------------------------
OUTCOME_SERVED = "served"
OUTCOME_EMPTY = "empty"
OUTCOME_PARSE_ERROR = "parse_error"
OUTCOME_WORKER_ERROR = "worker_error"
OUTCOME_RESOLVE_ERROR = "resolve_error"

OUTCOMES = (OUTCOME_SERVED, OUTCOME_EMPTY, OUTCOME_PARSE_ERROR,
            OUTCOME_WORKER_ERROR, OUTCOME_RESOLVE_ERROR)


def classify_execute_error(status: int, error_text: Optional[str] = None) -> str:
    """Map an ``_assist_execute`` failure ``(body, status)`` to an outcome.

    The route's own error mapping already made the hard call: 400 is the
    caller-fixable resolve/validation class, 502 is a fleet/worker failure. The
    one refinement here is that a 502 whose message is "returned nothing" /
    "produced no text" is an EMPTY reply, not a worker fault — the worker
    answered, it just answered with a void, and the operator's fix for that
    ("retry, or pick another model") is different from a dead-worker fix.
    """
    try:
        if int(status) == 400:
            return OUTCOME_RESOLVE_ERROR
    except (TypeError, ValueError):
        pass
    low = (error_text or "").lower()
    if ("returned nothing" in low or "produced no text" in low
            or "neither" in low):
        return OUTCOME_EMPTY
    return OUTCOME_WORKER_ERROR


def new_run_id() -> str:
    """A correlation id for ONE generate attempt. The generation record and any
    terminal parse-outcome record share it, so the UI collapses them into one
    row (mirrors evictions' ``run_id``)."""
    return uuid.uuid4().hex


# Ring size. A display buffer, not the durable record (the journal line and the
# sqlite mirror are durable). A few hundred attempts is plenty of studio history.
RING_MAX = 500

_RING: Deque[dict] = deque(maxlen=RING_MAX)
_RING_LOCK = threading.Lock()
_SEQ = 0
_SEQ_LOCK = threading.Lock()


def _next_seq() -> int:
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ


def _worker_id() -> str:
    """The box that ran the attempt — central, here. Never empty."""
    env = (os.environ.get("HUGPY_WORKER_NAME") or "").strip()
    if env:
        return env
    try:
        return socket.gethostname() or "central"
    except Exception:  # noqa: BLE001 — telemetry never raises
        return "central"


# The fields a record carries, in the order the UI reads them. ``raw`` is kept
# UNTRUNCATED — it is the whole reason this log exists.
_RECORD_FIELDS = ("run_id", "mode", "kind", "model_requested", "model_resolved",
                  "raw", "text", "reasoning", "from_reasoning", "outcome",
                  "error", "elapsed_ms")


def build_record(**fields: Any) -> dict:
    """The record dict, fully stamped. Split out from ``append`` so tests can
    construct one without publishing it."""
    rec = {
        "ts": float(fields.pop("ts", None) or time.time()),
        "seq": _next_seq(),
        "worker_id": _worker_id(),
    }
    for k in _RECORD_FIELDS:
        if k in fields:
            v = fields.pop(k)
            if v is not None:
                rec[k] = v
    # Anything else the caller passed rides along verbatim (forward-compatible,
    # same as evictions: a new field needs no schema change, it lives in body).
    for k, v in fields.items():
        if v is not None:
            rec[k] = v
    if not rec.get("run_id"):
        rec["run_id"] = new_run_id()
    return rec


def append(**fields: Any) -> Optional[dict]:
    """Record ONE studio-assist attempt. BEST EFFORT, ALWAYS.

    Stamps the record, appends it to the process ring, writes a journal line, and
    persists it to the durable cross-process store. Every step is independently
    guarded: a store fault costs the operator history and NOTHING else. Returns
    the record (handy in tests) or None if even the build failed.

    Callers pass ``run_id`` to correlate a generation record with its terminal
    parse-outcome record; a bare record without one still logs, it just cards up
    on its own.
    """
    try:
        rec = build_record(**fields)
    except Exception:  # noqa: BLE001 — a bad field must never break an attempt
        logger.debug("studio-assist log: build failed", exc_info=True)
        return None
    try:
        with _RING_LOCK:
            _RING.append(rec)
    except Exception:  # noqa: BLE001
        pass
    try:
        logger.info("studio-assist %s", _kv_line(rec))
    except Exception:  # noqa: BLE001
        pass
    try:
        get_store().append([rec])
    except Exception:  # noqa: BLE001 — a store fault never surfaces
        logger.debug("studio-assist log: store append failed", exc_info=True)
    return rec


def _kv_line(rec: dict) -> str:
    """One greppable journal line. ``raw`` is deliberately NOT expanded here (it
    can be kilobytes); the store holds the full reply."""
    lead = ("run_id", "worker_id", "mode", "kind", "model_requested",
            "model_resolved", "outcome", "from_reasoning", "elapsed_ms")
    parts = []
    for k in lead:
        v = rec.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    err = rec.get("error")
    if err:
        parts.append(f"error={str(err)[:200]!r}")
    for k in ("raw", "text", "reasoning"):
        v = rec.get(k)
        if v:
            parts.append(f"{k}_len={len(str(v))}")
    return " ".join(parts)


from hugpy_control.shared import recent_ring

def recent(limit: int = 200) -> list[dict]:
    return recent_ring(_RING, _RING_LOCK, limit)


def reset_for_tests() -> None:
    """Drop the ring and the seq counter. Tests only."""
    global _SEQ
    with _RING_LOCK:
        _RING.clear()
    with _SEQ_LOCK:
        _SEQ = 0


# ---------------------------------------------------------------------------
# Central store — the durable, cross-process view.
#
# Dev central runs gunicorn (3 workers x 8 threads). An attempt is logged by
# whichever process served the POST, while the SSE stream is held open on
# another. An in-process queue would show a client only a third of the attempts.
# The shared comms sqlite file is the rendezvous, and its autoincrement rowid is
# the stream cursor: a reader polls "rowid > cursor" and gets exactly what it has
# not seen. Same file, same pragmas, same EMFILE hardening as the eviction and
# comms mirrors (see comms/evictions.py, comms/shared.py) — this is that pattern.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studio_assist_log (
    rowid_alias     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    seq             INTEGER,
    worker_id       TEXT,
    run_id          TEXT,
    mode            TEXT,
    kind            TEXT,
    outcome         TEXT,
    model_requested TEXT,
    model_resolved  TEXT,
    body            TEXT    NOT NULL
)
"""
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_sal_ts ON studio_assist_log(ts)",
    "CREATE INDEX IF NOT EXISTS ix_sal_run ON studio_assist_log(run_id)",
)

# Bounded history — hours of studio work in a couple of MB. Pruning is amortized.
MAX_ROWS = 5000
PRUNE_EVERY = 200
MAX_FAILURES = 5


from hugpy_control.shared import default_db_path


def _retry_on_emfile(fn):
    """Reuse the comms mirror's EMFILE burst hardening when importable."""
    try:
        from hugpy_control.shared import retry_on_emfile
        return retry_on_emfile(fn)
    except ImportError:
        return fn()


class StudioAssistStore:
    """Bounded sqlite history of studio-assist attempts, shared across processes.

    Every method is best-effort: a store fault degrades the panel to "no history"
    and is never allowed to surface into a route response or a generate attempt.
    After MAX_FAILURES consecutive faults the store disables itself rather than
    taxing every append with a doomed write — the same self-disable the comms and
    eviction mirrors use."""

    def __init__(self, path: Optional[str] = None,
                 max_rows: int = MAX_ROWS) -> None:
        from hugpy_control.shared import init_bounded_event_store
        init_bounded_event_store(self, path, max_rows, default_db_path)

    # -- plumbing ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        from hugpy_control.shared import connect_wal
        return connect_wal(self.path, retry=_retry_on_emfile)

    def _ensure(self) -> bool:
        from hugpy_control.shared import ensure_store_schema
        return ensure_store_schema(self, _SCHEMA, indexes=_INDEXES)

    def _note_failure(self, op: str, exc: BaseException) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("studio-assist log store DISABLED after %d failures "
                         "(last: %s during %s) — the panel loses history until "
                         "restart; generation is unaffected",
                         self._failures, exc, op)
        else:
            logger.warning("studio-assist log store %s failed: %s", op, exc)

    # -- write -------------------------------------------------------------
    def append(self, records: Iterable[dict]) -> int:
        """Persist a batch. Returns how many rows landed (0 on any fault).

        The full record is stored as JSON in ``body`` (so ``raw`` is kept
        UNTRUNCATED); the columns are only the ones we filter/order by. A new
        field needs no migration — it rides in the body."""
        rows = []
        for rec in records or ():
            if not isinstance(rec, dict):
                continue
            try:
                rows.append((
                    float(rec.get("ts") or time.time()),
                    int(rec.get("seq") or 0),
                    str(rec.get("worker_id") or ""),
                    str(rec.get("run_id") or ""),
                    str(rec.get("mode") or ""),
                    str(rec.get("kind") or ""),
                    str(rec.get("outcome") or ""),
                    str(rec.get("model_requested") or ""),
                    str(rec.get("model_resolved") or ""),
                    json.dumps(rec, default=str),
                ))
            except Exception:  # noqa: BLE001 — skip the bad row, keep the batch
                continue
        if not rows or not self._ensure():
            return 0
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO studio_assist_log "
                    "(ts, seq, worker_id, run_id, mode, kind, outcome, "
                    " model_requested, model_resolved, body) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            self._failures = 0
        except Exception as exc:  # noqa: BLE001
            self._note_failure("append", exc)
            return 0
        self._appends += len(rows)
        if self._appends >= PRUNE_EVERY:
            self._appends = 0
            self.prune()
        return len(rows)

    def prune(self) -> None:
        """Keep the newest ``max_rows``. Amortized, best-effort, never fatal."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM studio_assist_log WHERE rowid_alias <= ("
                    "  SELECT MAX(rowid_alias) - ? FROM studio_assist_log)",
                    (self.max_rows,))
        except Exception as exc:  # noqa: BLE001
            self._note_failure("prune", exc)

    # -- read --------------------------------------------------------------
    def recent(self, limit: int = 200, since_ts: Optional[float] = None,
               after_id: Optional[int] = None,
               raw_cap: Optional[int] = None) -> list[dict]:
        """Newest-last list of records, each carrying its store id as ``_id``.

        ``after_id`` is the STREAM cursor (rowid); ``since_ts`` is the "last N
        seconds" filter. ``raw_cap`` optionally truncates ``raw`` in the RETURNED
        copy (the backfill route bounds a huge reply for the page load) — the
        STORE always keeps the full reply, and the SSE path passes no cap."""
        if not self._ensure():
            return []
        clauses, args = [], []
        if after_id is not None:
            clauses.append("rowid_alias > ?")
            args.append(int(after_id))
        if since_ts is not None:
            clauses.append("ts >= ?")
            args.append(float(since_ts))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = max(1, min(int(limit or 200), 5000))
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT rowid_alias, body FROM studio_assist_log" + where +
                    " ORDER BY rowid_alias DESC LIMIT ?", (*args, lim))
                got = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("recent", exc)
            return []
        out = []
        for rid, body in reversed(got):
            try:
                rec = json.loads(body)
            except Exception:  # noqa: BLE001
                continue
            rec["_id"] = int(rid)
            if raw_cap and isinstance(rec.get("raw"), str) and len(rec["raw"]) > raw_cap:
                rec["raw"] = rec["raw"][:raw_cap]
                rec["raw_truncated"] = True
            out.append(rec)
        return out

    def max_id(self) -> int:
        """The current head cursor — where a live-only stream starts."""
        if not self._ensure():
            return 0
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT MAX(rowid_alias) FROM studio_assist_log").fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception as exc:  # noqa: BLE001
            self._note_failure("max_id", exc)
            return 0


_STORE: Optional[StudioAssistStore] = None
_STORE_LOCK = threading.Lock()


from hugpy_control.shared import get_or_create_singleton

def get_store() -> StudioAssistStore:
    return get_or_create_singleton(globals(), "_STORE", _STORE_LOCK, StudioAssistStore)


def set_store(store: Optional[StudioAssistStore]) -> None:
    """Swap the singleton (tests point it at a tmp file)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store
