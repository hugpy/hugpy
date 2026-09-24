"""Post-download ADMISSION — the storage half (record + queue + install hook).

A model that finishes downloading on central is not yet trusted to serve. The
admission gate (``hugpy_ops.admission``) runs, in order, the static integrity
audit, then the HugPy-native benchmark, and writes the verdict onto the
model's own record:

    hugpy.json["admission"] = {
        "status":    "pending" | "admitted" | "held",
        "reason":    str | None,        # why held / the note when admitted
        "integrity": str | None,        # the static audit verdict
        "grade":     float | None,      # aptitude grade the benchmark produced
        "at":        iso-8601 UTC,
        "job":       str | None,        # admission job id (this module's queue)
        # optional (present only when recorded):
        "failure_class": str,           # the benchmark row's class (no_lane, timeout, ...)
        "evidence":  dict,              # that row's evidence, verbatim
        "blocked_on": str,              # no_lane: placement | eligible_worker | lane
        "elapsed_s": float,             # wall seconds the admission attempt took
        "failures":  list,              # every failed row {failure_class, worker, quant, reason}
        "run_id":    str,               # benchmark run, when no row named the model
    }

hugpy.json is the model's record, so the admission lives there — one source.
Central's resolver reads it (through the persisted marker aspect every catalog
surface already reads) and refuses a ``held`` model; ``pending`` routes, so the
benchmark that decides admission can reach the model.

This module owns the parts that have to live BELOW the engine:

* :func:`write_admission` / :func:`read_admission` — atomic read-modify-write of
  the block on an existing marker (the marker writer's own ``_save_marker``).
* :class:`AdmissionQueue` — a small persistent queue (sqlite in PROJECTS_HOME)
  shared by every process on central. The download daemon's transfer child
  enqueues; the server's admission runner (one per host, flock-elected)
  claims. Idempotent per ``(model_key, captured_at)`` — a re-stamp that
  carries the install manifest over never re-admits.
* :func:`on_install_complete` — THE hook, called by every download path that
  writes a ``source: download`` marker, after the dir is promoted. Central
  only (a worker never downloads from the Hub and never admits). Never raises.

No network, ever: stdlib + the marker module.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

ADMISSION_KEY = "admission"
PENDING, ADMITTED, HELD = "pending", "admitted", "held"
STATUSES = (PENDING, ADMITTED, HELD)

# Queue row states (the JOB, not the model's admission).
Q_QUEUED, Q_RUNNING, Q_DONE, Q_FAILED = "queued", "running", "done", "failed"

DB_ENV = "HUGPY_ADMISSION_DB"
OFF_ENV = "HUGPY_ADMISSION"          # "off" disables the install hook
DB_BASENAME = "admission_queue.sqlite"
MAX_ATTEMPTS = 3                     # restarts a job may survive before it is failed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def admission_record(status: str, *, reason: Optional[str] = None,
                     integrity: Optional[str] = None, grade: Optional[float] = None,
                     job: Optional[str] = None, at: Optional[str] = None,
                     **extra: Any) -> dict:
    """The admission block, schema-complete (every key present)."""
    if status not in STATUSES:
        raise ValueError(f"admission status must be one of {STATUSES}, not {status!r}")
    rec = {"status": status, "reason": reason, "integrity": integrity,
           "grade": None if grade is None else float(grade),
           "at": at or utc_now_iso(), "job": job}
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec


def read_admission(directory: Optional[str]) -> Optional[dict]:
    """The admission block on ``directory``'s hugpy.json, or None."""
    if not directory:
        return None
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    marker = read_hugpy_marker(directory)
    block = (marker or {}).get(ADMISSION_KEY)
    return block if isinstance(block, dict) else None


def write_admission(directory: Optional[str], record: dict, *,
                    model_key: Optional[str] = None) -> Optional[str]:
    """Set the admission block on an EXISTING hugpy.json (atomic temp+replace).

    Returns the marker path, or None when there is no marker to extend (the
    record then lives only on the queue row — a dir without a marker is not
    an installed model). ``model_key`` drops that model's persisted physical
    record so every catalog surface re-reads the marker on its next look."""
    if not directory:
        return None
    from hugpy_storage.hugpy_marker import _save_marker, read_hugpy_marker
    marker = read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None
    marker[ADMISSION_KEY] = dict(record)
    path = _save_marker(directory, marker)
    if model_key:
        try:
            from hugpy_storage.model_physical import forget_physical
            forget_physical(model_key, f"admission {record.get('status')}")
        except Exception:  # noqa: BLE001 — the record landed; the cache re-derives
            logger.debug("admission: physical-record invalidation failed for %s",
                         model_key, exc_info=True)
    return path


def held_reason(block: Optional[dict]) -> Optional[str]:
    """The refusal text for a ``held`` block, else None."""
    if not isinstance(block, dict) or block.get("status") != HELD:
        return None
    return str(block.get("reason") or "held by admission (no reason recorded)")


# ── the queue ────────────────────────────────────────────────────────────────

def default_db_path() -> str:
    env = (os.environ.get(DB_ENV) or "").strip()
    if env:
        return env
    from hugpy_platform.constants import PROJECTS_HOME
    return os.path.join(str(PROJECTS_HOME), DB_BASENAME)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS admission_jobs (
    id           TEXT PRIMARY KEY,
    model_key    TEXT NOT NULL,
    directory    TEXT,
    captured_at  TEXT NOT NULL,
    source       TEXT,
    status       TEXT NOT NULL,
    owner        TEXT,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    result       TEXT,
    log          TEXT,
    attempt      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (model_key, captured_at)
);
CREATE INDEX IF NOT EXISTS admission_jobs_status ON admission_jobs (status, created_at);
"""


class AdmissionQueue:
    """Persistent, cross-process admission job queue (sqlite, WAL)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._ready = False

    @property
    def path(self) -> str:
        return self._path or default_db_path()

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # busy_timeout makes EVERY statement (pragmas included) wait for a lock
        # instead of failing at once; the connect() timeout alone does not cover
        # them. Live incident 2026-09-23: the install hook lost dreamshaper-8's
        # job with "database is locked" because the console's list() and the
        # hook opened the file together and each ran the WAL switch + schema.
        conn.execute("PRAGMA busy_timeout=30000")
        if not self._ready:
            # Switching the journal mode needs an exclusive lock and the schema
            # script is a write transaction; do both ONCE per process, and never
            # let a lost race here break a caller that only wants to read/enqueue.
            try:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                if str(mode).lower() != "wal":
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                cols = {r[1] for r in conn.execute("PRAGMA table_info(admission_jobs)")}
                if "attempt" not in cols:     # queues created before 2026-09-23 restart-survival
                    conn.execute("ALTER TABLE admission_jobs ADD COLUMN attempt "
                                 "INTEGER NOT NULL DEFAULT 0")
                self._ready = True
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(r: Optional[sqlite3.Row]) -> Optional[dict]:
        if r is None:
            return None
        d = dict(r)
        for k in ("result",):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        return d

    def enqueue(self, model_key: str, captured_at: str, *,
                directory: Optional[str] = None, source: str = "download") -> tuple:
        """``(job_row, created)``. A second enqueue of the same
        ``(model_key, captured_at)`` returns the existing row, created=False."""
        if not model_key or not captured_at:
            raise ValueError("model_key and captured_at are required")
        job_id = uuid.uuid4().hex[:16]
        with self._db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO admission_jobs (id, model_key, directory, "
                "captured_at, source, status, created_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, model_key, directory, str(captured_at), source,
                 Q_QUEUED, time.time()))
            created = cur.rowcount == 1
            row = conn.execute(
                "SELECT * FROM admission_jobs WHERE model_key=? AND captured_at=?",
                (model_key, str(captured_at))).fetchone()
        return self._row(row), created

    def claim_next(self, owner: str) -> Optional[dict]:
        """Atomically take the oldest queued job (compare-and-set)."""
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM admission_jobs WHERE status=? "
                    "ORDER BY created_at LIMIT 1", (Q_QUEUED,)).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE admission_jobs SET status=?, owner=?, started_at=? "
                    "WHERE id=? AND status=?",
                    (Q_RUNNING, owner, time.time(), row["id"], Q_QUEUED))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return self.get(row["id"])

    def requeue_stale(self, older_than_s: float = 6 * 3600) -> int:
        """Put jobs a dead runner left ``running`` back on the queue."""
        cutoff = time.time() - older_than_s
        with self._db() as conn:
            cur = conn.execute(
                "UPDATE admission_jobs SET status=?, owner=NULL WHERE status=? "
                "AND started_at < ?", (Q_QUEUED, Q_RUNNING, cutoff))
            return cur.rowcount

    def requeue_orphaned(self, reason: str = "central restarted",
                         max_attempts: int = MAX_ATTEMPTS) -> dict:
        """Called by a NEWLY ELECTED runner: the runner flock is held for the
        life of the process that runs jobs, so every row still ``running`` at
        election belongs to a runner that died (a central restart). Each goes
        back to ``queued`` with ``attempt + 1`` and a log line; a job already
        interrupted ``max_attempts`` times is ``failed`` with the reason instead
        of looping forever (the model's admission stays pending; rerun it)."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        out = {"requeued": [], "failed": []}
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute("SELECT id, attempt, log FROM admission_jobs WHERE status=?",
                                    (Q_RUNNING,)).fetchall()
                for r in rows:
                    attempt = int(r["attempt"] or 0) + 1
                    prior = (r["log"] + "\n") if r["log"] else ""
                    if attempt > max_attempts:
                        line = (f"{stamp} interrupted ({reason}) {attempt - 1}x — not re-queued "
                                f"again (max {max_attempts}); POST /llm/admission/<model>/rerun")
                        conn.execute(
                            "UPDATE admission_jobs SET status=?, owner=NULL, attempt=?, finished_at=?, "
                            "result=?, log=? WHERE id=? AND status=?",
                            (Q_FAILED, attempt, time.time(),
                             json.dumps({"status": "failed", "reason": line[len(stamp) + 1:]}),
                             prior + line, r["id"], Q_RUNNING))
                        out["failed"].append(r["id"])
                    else:
                        line = f"{stamp} re-queued: {reason} while running (attempt {attempt})"
                        conn.execute(
                            "UPDATE admission_jobs SET status=?, owner=NULL, started_at=NULL, "
                            "attempt=?, log=? WHERE id=? AND status=?",
                            (Q_QUEUED, attempt, prior + line, r["id"], Q_RUNNING))
                        out["requeued"].append(r["id"])
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return out

    def requeue_owner(self, owner: str) -> int:
        """A restarted runner re-queues what its previous incarnation held."""
        with self._db() as conn:
            cur = conn.execute(
                "UPDATE admission_jobs SET status=?, owner=NULL WHERE status=? AND owner=?",
                (Q_QUEUED, Q_RUNNING, owner))
            return cur.rowcount

    def finish(self, job_id: str, status: str, *, result: Optional[dict] = None,
               log: Optional[list] = None) -> None:
        with self._db() as conn:
            conn.execute(
                "UPDATE admission_jobs SET status=?, finished_at=?, result=?, "
                "log=COALESCE(?, log) WHERE id=?",
                (status, time.time(), json.dumps(result, default=str) if result is not None else None,
                 "\n".join(log) if log else None, job_id))

    def append_log(self, job_id: str, line: str) -> None:
        with self._db() as conn:
            row = conn.execute("SELECT log FROM admission_jobs WHERE id=?", (job_id,)).fetchone()
            prior = (row["log"] + "\n") if row and row["log"] else ""
            conn.execute("UPDATE admission_jobs SET log=? WHERE id=?", (prior + line, job_id))

    def get(self, job_id: str) -> Optional[dict]:
        with self._db() as conn:
            return self._row(conn.execute(
                "SELECT * FROM admission_jobs WHERE id=?", (job_id,)).fetchone())

    def latest_for(self, model_key: str) -> Optional[dict]:
        with self._db() as conn:
            return self._row(conn.execute(
                "SELECT * FROM admission_jobs WHERE model_key=? ORDER BY created_at DESC LIMIT 1",
                (model_key,)).fetchone())

    def list(self, *, status: Optional[str] = None, limit: int = 500) -> list:
        q, args = "SELECT * FROM admission_jobs", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with self._db() as conn:
            return [self._row(r) for r in conn.execute(q, args).fetchall()]


admission_queue = AdmissionQueue()


def hook_enabled() -> bool:
    """The install hook runs on CENTRAL only, and can be switched off."""
    if (os.environ.get(OFF_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    try:
        from hugpy_storage.provision import worker_central_url
        return worker_central_url() is None
    except Exception:  # noqa: BLE001
        return True


def on_install_complete(directory: str, model_key: Optional[str] = None, *,
                        source: str = "download", captured_at: Optional[str] = None,
                        queue: Optional[AdmissionQueue] = None) -> Optional[dict]:
    """THE install hook: enqueue one admission job for the model now at
    ``directory`` and mark it ``pending`` on its marker.

    Keys: the marker's declared ``name`` (what discovery keys the catalog on)
    wins over the caller's ``model_key`` (a repo-download job id such as
    ``owner_repo``); the runner re-resolves against the catalog by directory.
    ``captured_at`` defaults to the marker's install-manifest capture time —
    that is what makes the enqueue idempotent per install. Returns the job row,
    or None when nothing was enqueued. NEVER raises: a completed download must
    not fail over its admission bookkeeping."""
    try:
        if not hook_enabled():
            return None
        from hugpy_storage.hugpy_marker import MANIFEST_KEY, read_hugpy_marker
        marker = read_hugpy_marker(directory) if directory else None
        key = ((marker or {}).get("name") or model_key or "").strip()
        if not key:
            return None
        stamp = captured_at or ((marker or {}).get(MANIFEST_KEY) or {}).get("captured_at") \
            or (marker or {}).get("stamped_at")
        if not stamp:
            return None
        q = queue or admission_queue
        job, created = q.enqueue(key, str(stamp), directory=directory, source=source)
        if created and marker is not None:
            write_admission(directory, admission_record(
                PENDING, reason="admission queued after download", job=job["id"]),
                model_key=key)
        logger.info("admission: %s job %s for %s (%s)",
                    "queued" if created else "already queued", job["id"], key, directory)
        return job
    except Exception as exc:  # noqa: BLE001
        logger.warning("admission hook failed for %s (%s): %s", model_key, directory, exc)
        return None


def request_admission(model_key: str, directory: Optional[str], *,
                      source: str = "rerun", reason: str = "admission re-run requested",
                      queue: Optional[AdmissionQueue] = None) -> dict:
    """Queue a fresh admission job for an installed model (operator re-run,
    resweep). Unlike the install hook this always creates a job — its
    ``captured_at`` is ``<source>:<now>`` — and it raises on failure (the
    caller is an operator surface that must report it)."""
    q = queue or admission_queue
    job, _created = q.enqueue(model_key, f"{source}:{utc_now_iso()}",
                              directory=directory, source=source)
    write_admission(directory, admission_record(PENDING, reason=reason, job=job["id"]),
                    model_key=model_key)
    return job


__all__ = [
    "ADMISSION_KEY", "PENDING", "ADMITTED", "HELD", "STATUSES",
    "Q_QUEUED", "Q_RUNNING", "Q_DONE", "Q_FAILED",
    "admission_record", "read_admission", "write_admission", "held_reason",
    "AdmissionQueue", "admission_queue", "default_db_path",
    "hook_enabled", "on_install_complete", "request_admission", "utc_now_iso",
]
