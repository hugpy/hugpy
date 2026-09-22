"""Cross-process mirror for the comms control plane (SQLite, stdlib-only).

The JobStore and Bus are per-process — correct for one gunicorn worker, but
the dev service runs ``--workers 3``: a cancel POST lands on process B while
process A serves the stream, and B's store has never heard of the job. This
mirror is the smallest primitive that fixes that without a broker or an
external dependency:

    - every job transition is upserted here (JSON snapshot + cancel flag),
    - a cancel for a job we don't hold locally sets the flag on the row,
    - the process that owns the stream notices the flag on its next token
      (throttled read on the hot path) and fires its local cancel handle,
    - queue views merge live rows from sibling processes into their snapshot.

SQLite in WAL mode handles concurrent single-row writes from a handful of
processes easily. Every operation opens a short-lived connection — no shared
handles across threads, no pooling to get wrong. All methods are best-effort:
a mirror failure degrades to per-process behavior (exactly today's world),
never breaks a chat. After MAX_FAILURES consecutive errors the mirror
disables itself loudly rather than taxing every token with a doomed write.

The db path must be shared by all processes of one service: HUGPY_COMMS_DB
if set, else a per-user file under XDG_RUNTIME_DIR (or /tmp). Central and a
worker agent on the same box sharing one file is harmless — job ids are
uuids and each process still only cancels what it holds a handle for.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import random
import sqlite3
import threading
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# EMFILE burst hardening (incident 2026-07-23)
# ---------------------------------------------------------------------------
# On a restart, all gunicorn workers spawn at once and simultaneously open
# store-backed files on the virtiofs mount (/mnt/llm_storage). Under that
# concurrent-open burst the mount MOMENTARILY returns EMFILE ("Too many open
# files") — this is the mount degrading under load, NOT a real per-process
# fd-limit breach (each proc held ~50 of 65536). SQLite surfaces the very same
# transient as OperationalError("unable to open database file"). Both settle in
# well under a second. This helper retries the store-open with exponential
# backoff + small jitter so a burst settles instead of surfacing a 500.
#
# It is DELIBERATELY narrow: only EMFILE OSErrors and the one sqlite "unable to
# open database file" message are retryable. Every other error (a real
# permission fault, a corrupt db, a genuinely different sqlite error) propagates
# immediately — a burst-settle retry must never paper over a real fault.


def _is_emfile(exc: BaseException) -> bool:
    """True for the transient store-open faults the mount throws under an
    open burst: an OSError with errno EMFILE, or the sqlite OperationalError
    whose message is 'unable to open database file' (same root cause, surfaced
    through the sqlite open path). Nothing else is retryable."""
    if isinstance(exc, sqlite3.OperationalError):
        return "unable to open database file" in str(exc).lower()
    if isinstance(exc, OSError):
        return exc.errno == errno.EMFILE
    return False


def retry_on_emfile(
    fn: Callable[[], _T],
    *,
    attempts: int = 5,
    base_delay: float = 0.05,
    max_delay: float = 0.8,
    sleep: Callable[[float], None] = time.sleep,
    rng: Optional[Callable[[], float]] = random.random,
) -> _T:
    """Call ``fn()`` and, on a transient EMFILE / 'unable to open database file'
    store-open fault, retry with exponential backoff + jitter; re-raise the LAST
    error once ``attempts`` are exhausted.

    Only the transient store-open faults (see ``_is_emfile``) are retried — any
    other exception propagates immediately, unretried. Backoff is
    ``min(max_delay, base_delay * 2**i)`` plus up to one base_delay of jitter
    (``rng()`` in [0,1)); pass ``rng=None`` for deterministic (jitter-free)
    backoff in tests, and inject ``sleep`` to make tests fast. ``attempts`` is
    the total number of tries (>=1), so there are at most ``attempts - 1``
    sleeps.
    """
    if attempts < 1:
        attempts = 1
    last: BaseException
    for i in range(attempts):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised unless retryable
            if not _is_emfile(exc):
                raise
            last = exc
            if i >= attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            if rng is not None:
                delay += base_delay * rng()
            logger.warning(
                "store-open EMFILE burst (attempt %d/%d): %s — backing off %.3fs",
                i + 1, attempts, exc, delay,
            )
            sleep(delay)
    raise last

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    data             TEXT NOT NULL,
    status           TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'chat',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    updated          REAL NOT NULL,
    -- Last REAL forward-progress epoch (Job.progressed_at), distinct from
    -- `updated` which any write bumps. The orphan-sweep in prune() ages on
    -- THIS clock so a wedged render (updated by mere views/recomputes) still
    -- goes reapable, while a truly-progressing stream never does.
    progressed_at    REAL,
    -- Cross-process WORK CLAIM (download daemon, 2026-07-28). A queued job with
    -- claimed_by IS NULL is up for grabs; claim_next() takes it under a write
    -- lock so exactly one daemon ever runs a given transfer. NULL for every job
    -- that is not dispatched through a claim queue (chat, media, ...).
    claimed_by       TEXT,
    claimed_at       REAL
);
"""

# Sidecar annotations pinned to a job id (k121: keeper diagnosis of a failed
# download, discard audit). A separate table, NOT the data blob, so a note
# survives the row's owner rewriting `data` and needs no dataclass field.
# conn.execute() runs ONE statement, hence a second constant, not _SCHEMA text.
_NOTES_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_notes (
    job_id  TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    updated REAL NOT NULL,
    PRIMARY KEY (job_id, key)
);
"""

# Statuses considered "active" for the wedged-orphan sweep. Mirrors the
# _STALL_ACTIVE spirit in jobs.py: only a job that claims to be doing work can
# be "wedged". A pending/queued job is starved (waiting its turn), not wedged,
# so it is NEVER reaped by the progress-sweep.
_ORPHAN_ACTIVE = ("processing", "streaming", "running")

# Pending-orphan expiry (slice 9, defect 2): a `pending` job with no worker and
# no forward progress this long was NEVER dispatched (model unresolvable, or no
# capable worker) — it will never run, so it transitions to a terminal `expired`
# state. Aged on `progressed_at` (the movement-only clock — NEVER `updated`,
# which view/recompute writes bump; that resurrection bug was fixed once and must
# not return). Env-overridable; default 30 min.
def _pending_expiry_seconds() -> float:
    raw = (os.environ.get("HUGPY_JOB_PENDING_EXPIRY_SECONDS") or "").strip()
    if not raw:
        return 1800.0
    try:
        v = float(raw)
        return v if v > 0 else 1800.0
    except ValueError:
        return 1800.0

# ---------------------------------------------------------------------------
# CLAIM-QUEUE kinds (k119)
# ---------------------------------------------------------------------------
# Kinds that are dispatched by a DAEMON taking a claim (claim_next), not by a
# worker being assigned. They never carry a `worker`, so the worker-dispatch
# orphan sweep below would retire every one of them after 30 minutes with the
# message "no capable worker" — which is a LIE for a download (there is no
# worker in that path at all) and, worse, silently deletes the operator's
# request. Every one of the 65 download rows on the dev box read exactly that.
# These kinds get their own, far longer expiry and an honest message that names
# the service that is supposed to claim them.
CLAIM_QUEUE_KINDS = ("download",)


def _claim_queue_expiry_seconds() -> float:
    """How long a CLAIMABLE job may sit unclaimed before it is retired. Far
    longer than the worker-dispatch window: a queue of large models legitimately
    waits hours behind the two concurrent transfer slots, and a downloader that
    is merely stopped for a maintenance window must not lose the queue. Default
    24h; env-overridable."""
    raw = (os.environ.get("HUGPY_JOB_CLAIM_EXPIRY_SECONDS") or "").strip()
    if not raw:
        return 86400.0
    try:
        v = float(raw)
        return v if v > 0 else 86400.0
    except ValueError:
        return 86400.0


MAX_FAILURES = 5

# ---------------------------------------------------------------------------
# Mirror quarantine, NOT a kill switch (k119 — the download-queue outage)
# ---------------------------------------------------------------------------
# MAX_FAILURES consecutive faults used to set `_disabled = True` FOREVER: "…
# degrade to per-process until restart". For the API that is survivable (a
# gunicorn worker recycles). For the DOWNLOADER DAEMON it is fatal and silent:
# the daemon's whole job is claim_next() against this mirror, it opens ~40
# connections a minute against a virtiofs mount that demonstrably throws
# transient `disk I/O error` / `database is locked` / `unable to open database
# file`, and Downloader.run() only checks the mirror ONCE at startup. On
# 2026-08-10 08:54:51-57 five consecutive blips latched it off; for the next 11
# days the daemon logged a reassuring `polling (active=0)` every 5 minutes while
# claiming nothing, and every "Add model" the operator clicked sat pending until
# the orphan sweep expired it.
#
# So the breaker now OPENS (stop taxing every call with a doomed write) and then
# HALF-OPENS: after a cooldown one probe is allowed through, and a success
# restores the mirror and says so in the log. The cooldown backs off
# exponentially to a cap so a genuinely dead DB is still not hammered.
MIRROR_COOLDOWN_SECONDS = float(
    os.environ.get("HUGPY_COMMS_MIRROR_COOLDOWN_SECONDS", "30") or 30)
MIRROR_COOLDOWN_MAX_SECONDS = float(
    os.environ.get("HUGPY_COMMS_MIRROR_COOLDOWN_MAX_SECONDS", "300") or 300)


def default_db_path() -> str:
    env = (os.environ.get("HUGPY_COMMS_DB") or "").strip()
    if env:
        return env
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return os.path.join(base, f"hugpy-comms-{os.getuid()}.db")


class SqliteMirror:
    def __init__(self, path: Optional[str] = None,
                 retain_secs: float = 600.0) -> None:
        self.path = path or default_db_path()
        self.retain_secs = retain_secs
        self._failures = 0
        self._disabled = False
        # Quarantine bookkeeping (see MIRROR_COOLDOWN_SECONDS): when the breaker
        # opens we stamp the epoch it may next be probed and how long the next
        # cooldown should be. `_outages` is a counter for the log/health view so
        # an operator can see a mount that keeps flapping.
        self._retry_at = 0.0
        self._cooldown = MIRROR_COOLDOWN_SECONDS
        self._outages = 0
        self._recovered = 0
        self._probing = False
        self._last_error = ""
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # sqlite3.connect is the store-open point that throws the transient
        # EMFILE / 'unable to open database file' under a restart burst — retry
        # just the open, then run the (handle-local) PRAGMAs normally.
        conn = retry_on_emfile(lambda: sqlite3.connect(self.path, timeout=2.0))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=2000")
        return conn

    def _ensure(self) -> bool:
        if self._disabled:
            # HALF-OPEN: the breaker is a cooldown, never a permanent latch.
            # Before the cooldown elapses every call short-circuits (that is the
            # point — no doomed write per token). After it elapses ONE call gets
            # through to probe the DB; if that probe succeeds, `_ok()` closes the
            # breaker and logs the recovery, and if it fails `_note_failure`
            # re-opens it with a longer cooldown.
            if time.time() < self._retry_at:
                return False
            self._disabled = False
            self._initialized = False   # re-run CREATE/migrate on the probe
            self._failures = 0
            self._probing = True
            logger.warning(
                "comms mirror probing after %.0fs quarantine (%s) — one attempt",
                self._cooldown, self.path)
        if self._initialized:
            return True
        with self._init_lock:
            if self._initialized:
                return True
            try:
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
                    conn.execute(_NOTES_SCHEMA)
                    self._migrate(conn)
                self._initialized = True
                return True
            except Exception as exc:
                self._note_failure("init", exc)
                return False

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent, safe on an existing populated DB. Adds the
        `progressed_at` column to pre-existing `jobs` tables that predate it
        (a plain CREATE ... IF NOT EXISTS won't alter an already-present
        table). Guarded by a table_info probe so a second run is a no-op and a
        DB already carrying the column never raises."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "progressed_at" not in cols:
            # NULL default: old rows have no known progress clock, so they are
            # fail-open (never reaped by the progress-sweep) until a fresh
            # upsert stamps a real progressed_at.
            conn.execute("ALTER TABLE jobs ADD COLUMN progressed_at REAL")
        # The work-claim columns (download daemon). NULL on every pre-existing
        # row = unclaimed, which is exactly right: nothing before this migration
        # was dispatched through a claim queue.
        if "claimed_by" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN claimed_by TEXT")
        if "claimed_at" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN claimed_at REAL")

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._failures += 1
        self._last_error = f"{exc} during {op}"
        # A FAILED half-open probe re-opens the breaker immediately — it must not
        # spend another MAX_FAILURES calls proving what the probe just showed.
        if self._probing and self._failures < MAX_FAILURES:
            self._failures = MAX_FAILURES
        if self._failures >= MAX_FAILURES:
            # OPEN the breaker for a cooldown that grows on repeat outages, so a
            # flapping mount backs off but a one-off blip costs ~30s of queue
            # latency instead of the whole download plane until someone notices.
            if self._probing or self._disabled:
                # Still broken after a probe: back off, don't re-count the outage.
                self._cooldown = min(self._cooldown * 2,
                                     MIRROR_COOLDOWN_MAX_SECONDS)
            else:
                self._outages += 1
            self._disabled = True
            self._initialized = False
            self._retry_at = time.time() + self._cooldown
            logger.error("comms mirror QUARANTINED for %.0fs after %d failures "
                         "(last: %s during %s) — cross-process cancel/queue "
                         "degrade to per-process; it retries itself, no restart "
                         "needed",
                         self._cooldown, self._failures, exc, op)
        else:
            logger.warning("comms mirror %s failed: %s", op, exc)

    def _ok(self) -> None:
        if self._probing:
            # The half-open probe succeeded — close the breaker and SAY SO.
            # "DISABLED" was logged once and then nothing for 11 days, which is
            # precisely why the outage stayed invisible.
            self._probing = False
            self._recovered += 1
            logger.warning("comms mirror RECOVERED (%s) — cross-process "
                           "queue/cancel is live again", self.path)
        self._disabled = False
        self._retry_at = 0.0
        self._cooldown = MIRROR_COOLDOWN_SECONDS
        self._failures = 0
        self._last_error = ""

    # -- health --------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """What state the breaker is in, for a daemon's log line and the
        console's queue view. Cheap and side-effect-free — never opens the DB."""
        return {
            "path": self.path,
            "ok": not self._disabled,
            "failures": self._failures,
            "outages": self._outages,
            "recoveries": self._recovered,
            "retry_in": (max(0.0, self._retry_at - time.time())
                         if self._disabled else 0.0),
            "last_error": self._last_error,
        }

    # -- writes --------------------------------------------------------------
    def upsert(self, job_dict: dict[str, Any]) -> None:
        """Mirror a job snapshot. cancel_requested is only ever raised here,
        never lowered (a resurrection lowers it explicitly via clear_cancel) —
        so a racing sibling's cancel can't be lost under a stale local write."""
        if not self._ensure():
            return
        try:
            # progressed_at is the movement-only clock (jobs.py bumps it on real
            # forward progress, NOT on log-tail/view/recompute writes). Persist
            # it as a real column so prune()'s orphan-sweep can age on it. A
            # missing/garbage value degrades to NULL -> fail-open (never reaped).
            prog = job_dict.get("progressed_at")
            try:
                prog = float(prog) if prog is not None else None
            except (TypeError, ValueError):
                prog = None
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, data, status, kind, cancel_requested, updated, progressed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "  data=excluded.data, status=excluded.status, "
                    "  kind=excluded.kind, "
                    "  cancel_requested=MAX(jobs.cancel_requested, excluded.cancel_requested), "
                    "  updated=excluded.updated, "
                    "  progressed_at=excluded.progressed_at",
                    (str(job_dict.get("id")), json.dumps(job_dict),
                     str(job_dict.get("status") or "pending"),
                     str(job_dict.get("kind") or "chat"),
                     1 if job_dict.get("cancel_requested") else 0,
                     time.time(), prog))
            self._ok()
        except Exception as exc:
            self._note_failure("upsert", exc)

    def request_cancel(self, job_id: str) -> bool:
        """Raise the cancel flag on a row we may not hold locally. Returns
        True if a live (non-terminal) row existed to flag."""
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE jobs SET cancel_requested=1, updated=? "
                    "WHERE id=? AND status NOT IN ('done','cancelled','failed')",
                    (time.time(), job_id))
            self._ok()
            return cur.rowcount > 0
        except Exception as exc:
            self._note_failure("request_cancel", exc)
            return False

    def force_terminal(self, job_id: str, status: str, message: str = "",
                       override_terminal: bool = False) -> bool:
        """Force a LIVE mirror row terminal (slice 9) — the authoritative cancel /
        orphan-expiry path for a row NO process holds in memory (the immortal
        pending job that survived restarts). Updates the persisted `status`
        column AND rewrites the row's JSON `data` blob so /llm/jobs reads the
        terminal state consistently. Only acts on a row that is NOT already
        terminal (first-terminal-wins); returns True if a live row was retired.

        Goes through the store's own connection (never a caller-side raw sqlite
        channel) — the DB write discipline stays in one place."""
        if not self._ensure():
            return False
        term = str(status or "cancelled")
        # k121: override_terminal lets the DISCARD verb flip an
        # already-terminal record (failed/expired -> discarded); nothing else
        # may overwrite a terminal status (first-terminal-wins stands).
        guard = ("AND status <> 'discarded'" if override_terminal else
                 "AND status NOT IN ('done','cancelled','failed',"
                 "'expired','discarded')")
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT data FROM jobs WHERE id=? {guard}",
                    (job_id,)).fetchone()
                if not row:
                    return False           # unknown or already terminal
                try:
                    data = json.loads(row[0]) if row[0] else {}
                except Exception:
                    data = {}
                data["id"] = job_id
                data["status"] = term
                if message:
                    data["message"] = message
                data["ended_ts"] = data.get("ended_ts") or time.time()
                if term == "cancelled":
                    data["cancel_requested"] = True
                conn.execute(
                    "UPDATE jobs SET status=?, data=?, updated=? WHERE id=?",
                    (term, json.dumps(data), time.time(), job_id))
            self._ok()
            return True
        except Exception as exc:
            self._note_failure("force_terminal", exc)
            return False

    def set_job_note(self, job_id: str, key: str, value: str) -> bool:
        """Pin an annotation to a job id (k121: keeper `diagnosis` on a failed
        download, discard audit). Upsert; survives the owner rewriting the row's
        data blob. Swept only when the job row itself is pruned."""
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO job_notes (job_id, key, value, updated) "
                    "VALUES (?,?,?,?) ON CONFLICT(job_id, key) "
                    "DO UPDATE SET value=excluded.value, updated=excluded.updated",
                    (job_id, key, str(value), time.time()))
            self._ok()
            return True
        except Exception as exc:
            self._note_failure("set_job_note", exc)
            return False

    def job_notes_for(self, job_ids: list[str]) -> dict[str, dict[str, str]]:
        """{job_id: {key: value}} for the given ids (empty on any trouble —
        notes are garnish on the /jobs view, never a reason it fails)."""
        ids = [str(j) for j in (job_ids or []) if j]
        if not ids or not self._ensure():
            return {}
        try:
            placeholders = ",".join("?" for _ in ids)
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT job_id, key, value FROM job_notes "
                    f"WHERE job_id IN ({placeholders})", ids).fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("job_notes_for", exc)
            return {}
        out: dict[str, dict[str, str]] = {}
        for job_id, key, value in rows:
            out.setdefault(job_id, {})[key] = value
        return out

    def expire_pending_orphans(self) -> list[str]:
        """Transition never-dispatched pending jobs to terminal `expired` (slice 9,
        defect 2). A `pending` row with no worker whose `progressed_at` is older
        than the pending-expiry threshold was never picked up (unresolvable model
        or no capable worker); it will never run. Aged on `progressed_at` — the
        movement-only clock, NEVER `updated` (view/recompute writes bump `updated`,
        and aging on it is exactly the resurrection bug already fixed once). A NULL
        progressed_at is fail-open (never expired here — we only act when we
        positively know it is old). Rewrites the JSON `data` blob so /llm/jobs
        reads the honest terminal state + message. Returns the expired ids."""
        if not self._ensure():
            return []
        now = time.time()
        cutoff = now - _pending_expiry_seconds()
        # k119: a CLAIM-QUEUE row (download) has no worker BY DESIGN, so the
        # worker-dispatch cutoff above would retire every queued download after
        # 30 minutes and label it "no capable worker" — the exact lie found on
        # all 65 download rows. Claimable kinds age on their own long clock and
        # name the service that should have claimed them.
        claim_cutoff = now - _claim_queue_expiry_seconds()
        msg = ("never dispatched — model unresolvable or no capable worker "
               "(auto-expired)")
        claim_msg = ("never claimed by the downloader service — it was not "
                     "running, or its queue was unreachable (auto-expired). "
                     "Re-add the model once hugpy-downloader is up.")
        expired: list[str] = []
        try:
            with self._connect() as conn:
                # The widest (latest) cutoff of the two policies, so both kinds of
                # candidate come back; the per-row test below applies the right
                # clock. max(), not min(): these are epochs, and the 30-minute
                # cutoff is the LATER one.
                rows = conn.execute(
                    "SELECT id, data, kind, progressed_at, claimed_by FROM jobs "
                    "WHERE status='pending' "
                    "AND progressed_at IS NOT NULL AND progressed_at < ?",
                    (max(cutoff, claim_cutoff),)).fetchall()
                for job_id, data_json, kind, prog, claimed_by in rows:
                    try:
                        data = json.loads(data_json) if data_json else {}
                    except Exception:
                        data = {}
                    # A pending job that somehow has a worker assigned is being
                    # dispatched — leave it (belt-and-braces; the query already
                    # only takes pending, but a worker means an owner exists).
                    if data.get("worker"):
                        continue
                    claimable = str(kind or "") in CLAIM_QUEUE_KINDS
                    if claimable:
                        # A claimed row has an owner about to start it, and an
                        # unclaimed one still inside the long window is simply
                        # QUEUED — that is the row the operator is waiting on and
                        # it must survive a downloader restart/maintenance window.
                        if claimed_by:
                            continue
                        if float(prog) >= claim_cutoff:
                            continue
                    elif float(prog) >= cutoff:
                        continue
                    data["id"] = job_id
                    data["status"] = "expired"
                    data["message"] = claim_msg if claimable else msg
                    data["ended_ts"] = data.get("ended_ts") or time.time()
                    conn.execute(
                        "UPDATE jobs SET status='expired', data=?, updated=? "
                        "WHERE id=?",
                        (json.dumps(data), time.time(), job_id))
                    expired.append(job_id)
            self._ok()
        except Exception as exc:
            self._note_failure("expire_pending_orphans", exc)
        return expired

    def clear_cancel(self, job_id: str) -> None:
        """Deliberate resurrection (download retry) starts a fresh run."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute("UPDATE jobs SET cancel_requested=0, updated=? "
                             "WHERE id=?", (time.time(), job_id))
            self._ok()
        except Exception as exc:
            self._note_failure("clear_cancel", exc)

    def prune(self) -> None:
        if not self._ensure():
            return
        try:
            now = time.time()
            with self._connect() as conn:
                # (1) Terminal rows past the retain window — correctly keyed on
                # `updated` (any-write clock): once terminal, no more real
                # progress happens, so time-since-any-write is the right age.
                # k121 (operator ruling 2026-08-20): a FAILURE RECORD is never
                # auto-deleted — `failed` and `expired` rows persist, reason and
                # all, until an admin explicitly discards them (POST
                # /jobs/<id>/discard flips them to `discarded`, which IS swept
                # here). Before this, every failure evaporated after 10 minutes.
                conn.execute(
                    "DELETE FROM jobs WHERE "
                    "status IN ('done','cancelled','discarded') AND updated < ?",
                    (now - self.retain_secs,))
                # Notes of swept rows go with them (tiny table, cheap subquery).
                conn.execute(
                    "DELETE FROM job_notes WHERE job_id NOT IN "
                    "(SELECT id FROM jobs)")
                # (2) Wedged-orphan sweep: a job whose owner died mid-stream
                # never goes terminal. It MUST age on real forward-progress
                # silence (`progressed_at`), NOT on `updated` — `updated` is
                # bumped by mere views/sibling-recomputes/API-restart re-reads,
                # which kept these rows immortal. Only reap an ACTIVE row whose
                # progressed_at is genuinely ancient. NULL progressed_at is
                # fail-open (never reaped here) — we only reap when we
                # positively know it's old. A healthy long render bumps
                # progressed_at on every stage, so this window of TOTAL
                # progress silence = genuinely wedged.
                orphan_secs = max(self.retain_secs * 6, 3600.0)
                placeholders = ",".join("?" for _ in _ORPHAN_ACTIVE)
                conn.execute(
                    "DELETE FROM jobs WHERE "
                    f"status IN ({placeholders}) "
                    "AND progressed_at IS NOT NULL AND progressed_at < ?",
                    (*_ORPHAN_ACTIVE, now - orphan_secs))
            self._ok()
        except Exception as exc:
            self._note_failure("prune", exc)

    # -- work claims (the download daemon's dispatch queue) -------------------
    #
    # A job created by the API in `pending` with no claim is WORK WAITING. A
    # separate process (the downloader daemon) takes it with claim_next(), which
    # is a compare-and-set under an IMMEDIATE (write) transaction — so two
    # daemons, or a daemon racing its own restart, can never run one transfer
    # twice. The claim is a column, NOT part of the JSON blob, so the owner's
    # ordinary upsert()s (which rewrite `data`) can never clobber it.

    def claim_next(self, kinds: tuple, owner: str) -> Optional[dict[str, Any]]:
        """Atomically claim the oldest unclaimed `pending` job of *kinds* for
        *owner*. Returns its JSON snapshot (with the claim stamped on), or None
        when there is nothing to claim. Best-effort: a mirror fault returns None
        (the daemon simply idles) rather than raising into its poll loop."""
        kinds = tuple(kinds or ())
        if not kinds or not self._ensure():
            return None
        placeholders = ",".join("?" for _ in kinds)
        try:
            conn = self._connect()
            try:
                # IMMEDIATE takes the write lock up front, so the SELECT and the
                # UPDATE cannot interleave with another daemon's claim.
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT id, data FROM jobs WHERE status='pending' "
                    f"AND kind IN ({placeholders}) "
                    "AND (claimed_by IS NULL OR claimed_by='') "
                    "AND cancel_requested=0 "
                    "ORDER BY updated ASC LIMIT 1", kinds).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    self._ok()
                    return None
                job_id, data_json = row
                now = time.time()
                conn.execute(
                    "UPDATE jobs SET claimed_by=?, claimed_at=?, updated=? "
                    "WHERE id=?", (owner, now, now, job_id))
                conn.execute("COMMIT")
            finally:
                conn.close()
            self._ok()
        except Exception as exc:
            self._note_failure("claim_next", exc)
            return None
        try:
            data = json.loads(data_json) if data_json else {}
        except Exception:
            data = {}
        data["id"] = job_id
        data["claimed_by"] = owner
        return data

    def claimable_count(self, kinds: tuple) -> int:
        """How many jobs of *kinds* are queued and up for grabs RIGHT NOW.

        The observability counterpart of claim_next: that method answers None for
        both "nothing to do" and "I cannot see the queue", which is exactly the
        ambiguity that let a quarantined mirror look like an idle one for eleven
        days. Returns -1 when the mirror itself is unavailable, so a caller can
        tell "0 queued" from "I don't know"."""
        kinds = tuple(kinds or ())
        if not kinds or not self._ensure():
            return -1
        placeholders = ",".join("?" for _ in kinds)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status='pending' "
                    f"AND kind IN ({placeholders}) "
                    "AND (claimed_by IS NULL OR claimed_by='') "
                    "AND cancel_requested=0", kinds).fetchone()
            self._ok()
            return int(row[0]) if row else 0
        except Exception as exc:
            self._note_failure("claimable_count", exc)
            return -1

    def release_claim(self, job_id: str) -> None:
        """Drop the claim without touching status — the job becomes claimable
        again on its next pass through the queue (used by requeue/retry)."""
        if not self._ensure():
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET claimed_by=NULL, claimed_at=NULL, updated=? "
                    "WHERE id=?", (time.time(), job_id))
            self._ok()
        except Exception as exc:
            self._note_failure("release_claim", exc)

    def claim_of(self, job_id: str) -> Optional[str]:
        """Who holds the claim on *job_id*, or None (unclaimed / unknown)."""
        if not self._ensure():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT claimed_by FROM jobs WHERE id=?",
                                   (job_id,)).fetchone()
            self._ok()
            return (row[0] or None) if row else None
        except Exception as exc:
            self._note_failure("claim_of", exc)
            return None

    def requeue(self, job_id: str, message: str = "",
                kinds: tuple = ()) -> bool:
        """Put a job BACK on the claim queue: status -> `pending`, claim dropped,
        cancel flag lowered, run telemetry reset, JSON blob rewritten so every
        reader sees the queued state. This is the cross-process retry primitive
        (the requesting process does not own the row and has no local record of
        it). Optionally constrained to *kinds* so a caller cannot re-queue
        something that is not its kind of work. Returns True if a row moved."""
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT data, kind FROM jobs WHERE id=?",
                                   (job_id,)).fetchone()
                if not row:
                    return False
                if kinds and row[1] not in tuple(kinds):
                    return False
                try:
                    data = json.loads(row[0]) if row[0] else {}
                except Exception:
                    data = {}
                data["id"] = job_id
                data["status"] = "pending"
                data["cancel_requested"] = False
                data["ended_ts"] = None
                data["error"] = None
                data["stalled"] = False
                data["bytes_per_second"] = None
                if message:
                    data["message"] = message
                now = time.time()
                data["progressed_at"] = now
                conn.execute(
                    "UPDATE jobs SET status='pending', data=?, "
                    "cancel_requested=0, claimed_by=NULL, claimed_at=NULL, "
                    "updated=?, progressed_at=? WHERE id=?",
                    (json.dumps(data), now, now, job_id))
            self._ok()
            return True
        except Exception as exc:
            self._note_failure("requeue", exc)
            return False

    def adopt_stale(self, kinds: tuple, owner: str,
                    message: str = "") -> list[str]:
        """Fail-over on daemon startup: every job of *kinds* that some PREVIOUS
        owner left mid-flight (an active status, or a `pending` row still holding
        a stale claim) goes back on the queue so this daemon can resume it.

        Only rows claimed by someone OTHER than *owner* are touched — a daemon
        restart adopts a dead predecessor's work, and can never yank a row out
        from under a live sibling that claimed it under its own id. Partial files
        stay on disk; the resumed transfer adopts the staging dir (see
        imports/apis/download_models._adopt_or_reap_staging), so nothing re-fetches
        from zero. Returns the ids re-queued."""
        kinds = tuple(kinds or ())
        if not kinds or not self._ensure():
            return []
        placeholders = ",".join("?" for _ in kinds)
        active = ("processing", "streaming", "running")
        act_ph = ",".join("?" for _ in active)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id FROM jobs WHERE kind IN ({placeholders}) "
                    "AND (claimed_by IS NOT NULL AND claimed_by <> ?) "
                    f"AND status IN ({act_ph},'pending')",
                    (*kinds, owner, *active)).fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("adopt_stale", exc)
            return []
        out = []
        for (job_id,) in rows:
            if self.requeue(job_id, message=message, kinds=kinds):
                out.append(job_id)
        return out

    # -- reads ---------------------------------------------------------------
    def row(self, job_id: str) -> Optional[dict[str, Any]]:
        """The JSON snapshot of ONE mirrored job (live or terminal), or None.
        The single-job read complement to live_rows()/terminal_rows() — a route
        answering GET /jobs/<id> for a job another process owns has no local
        record to serialize."""
        if not self._ensure():
            return None
        try:
            with self._connect() as conn:
                r = conn.execute("SELECT data FROM jobs WHERE id=?",
                                 (job_id,)).fetchone()
            self._ok()
        except Exception as exc:
            self._note_failure("row", exc)
            return None
        if not r or not r[0]:
            return None
        try:
            return json.loads(r[0])
        except Exception:
            return None

    def cancel_requested(self, job_id: str) -> bool:
        if not self._ensure():
            return False
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT cancel_requested FROM jobs WHERE id=?",
                    (job_id,)).fetchone()
            self._ok()
            return bool(row and row[0])
        except Exception as exc:
            self._note_failure("cancel_requested", exc)
            return False

    def flagged_ids(self) -> set:
        """Ids with the cancel flag raised — the owner-side watcher intersects
        these with its local live jobs."""
        if not self._ensure():
            return set()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE cancel_requested=1").fetchall()
            self._ok()
            return {r[0] for r in rows}
        except Exception as exc:
            self._note_failure("flagged_ids", exc)
            return set()

    def live_rows(self) -> list[dict[str, Any]]:
        """Snapshots of every live job any process mirrored here."""
        if not self._ensure():
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data FROM jobs WHERE "
                    "status NOT IN ('done','cancelled','failed',"
                    "'expired','discarded')").fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("live_rows", exc)
            return []
        out = []
        for (data,) in rows:
            try:
                out.append(json.loads(data))
            except Exception:
                continue
        return out

    def terminal_rows(self, kinds: tuple) -> list[dict[str, Any]]:
        """Snapshots of TERMINAL jobs of the given kinds — the additive
        complement to live_rows() (which excludes terminal by design). Callers
        pass a strict kind allowlist (MEDIA_KINDS) so this surfaces a sibling's
        finished media job cross-process WITHOUT touching chat/download terminal
        behavior. Empty kinds -> nothing (never a full-table terminal scan)."""
        kinds = tuple(kinds or ())
        if not kinds:
            return []
        if not self._ensure():
            return []
        try:
            # Placeholders are '?' derived from the count only — the kind values
            # themselves are always bound parameters (no SQL injection surface).
            placeholders = ",".join("?" for _ in kinds)
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data FROM jobs WHERE "
                    "status IN ('done','cancelled','failed','expired') "
                    f"AND kind IN ({placeholders})", kinds).fetchall()
            self._ok()
        except Exception as exc:
            self._note_failure("terminal_rows", exc)
            return []
        out = []
        for (data,) in rows:
            try:
                out.append(json.loads(data))
            except Exception:
                continue
        return out
