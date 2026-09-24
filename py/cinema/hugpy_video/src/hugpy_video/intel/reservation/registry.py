"""The central reservation registry — claims keyed by (worker/gpu, run_id).

A claim records that a heavy video run has PRE-CLAIMED a card for its template's
peak GPU need. It carries a LEASE (TTL): the owning dispatch path refreshes it
on a heartbeat while the run is live, and an orphaned claim (the P-central
process crashed mid-run, or a runner died without releasing) SELF-EXPIRES when
its lease lapses — so a dead run never permanently starves the LLM fleet.

Persistence reuses the proven comms-SQLite idiom (``comms/calibration.py``'s
CalibrationStore / ``comms/shared.py``'s SqliteMirror): one short-lived
connection per op, WAL, stdlib-only, best-effort (a store failure degrades to
"no reservation" rather than breaking a render), self-disabling after
MAX_FAILURES. DURABLE by default (survives a restart), but leases make a stale
row harmless — a claim from a prior process is expired-on-read.

Claims are created/released ONLY by the video dispatch path (``engine`` called
from ``media_bus``). There is NO public create route this slice; the operator's
console only READS the listing (``GET /llm/reservations``).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from hugpy_control.shared import retry_on_emfile

logger = logging.getLogger(__name__)

MAX_FAILURES = 5
DEFAULT_LEASE_TTL_S = 180.0          # a claim lapses this long after its last refresh

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    run_id        TEXT PRIMARY KEY,     -- the media-bus job_id (one claim per run)
    worker_id     TEXT,                 -- registry worker id holding the card
    gpu           TEXT,                 -- gpu_affinity label (today "ae")
    task          TEXT,                 -- bus job name
    peak_bytes    INTEGER,              -- the GPU bytes this run reserves
    state         TEXT NOT NULL,        -- active | released | expired
    reason        TEXT,                 -- release/refusal note (honest provenance)
    made_room     INTEGER NOT NULL DEFAULT 0,   -- did acquisition evict anything?
    evicted_json  TEXT,                 -- what make-room yielded (for the listing)
    created_at    REAL NOT NULL,
    heartbeat_at  REAL NOT NULL,        -- last refresh (lease anchor)
    lease_ttl_s   REAL NOT NULL,
    released_at   REAL                  -- terminal timestamp (released|expired)
);
CREATE INDEX IF NOT EXISTS idx_resv_state ON reservations(state, worker_id);
"""


def default_db_path() -> str:
    """The reservations ledger path: ``HUGPY_RESERVATIONS_DB`` if set, else
    ``<PROJECTS_HOME>/reservations.db`` — resolved by ``hugpy_video.state``."""
    from hugpy_video.state import reservations_db_path
    return reservations_db_path()


class ReservationRegistry:
    def __init__(self, path: Optional[str] = None,
                 lease_ttl_s: float = DEFAULT_LEASE_TTL_S) -> None:
        self.path = path or default_db_path()
        self.lease_ttl_s = float(lease_ttl_s)
        self._failures = 0
        self._disabled = False
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # The reservation-registry init was one of the restart-burst EMFILE
        # casualties (2026-07-23). Retry the store-open (see
        # comms.shared.retry_on_emfile) before the handle-local PRAGMAs.
        from hugpy_control.shared import connect_wal
        return connect_wal(self.path, retry=retry_on_emfile)

    def _ensure(self) -> bool:
        from hugpy_control.shared import ensure_store_schema
        return ensure_store_schema(self, _SCHEMA, script=True)

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._failures += 1
        if self._failures >= MAX_FAILURES and not self._disabled:
            self._disabled = True
            logger.error("reservation registry DISABLED after %d failures "
                         "(last: %s during %s) — video runs proceed WITHOUT a "
                         "reservation until restart", self._failures, exc, op)
        else:
            logger.warning("reservation registry %s failed: %s", op, exc)

    def _ok(self) -> None:
        self._failures = 0

    # -- writes (dispatch path only) -----------------------------------------
    def claim(self, run_id: str, worker_id: Optional[str], gpu: Optional[str],
              task: str, peak_bytes: Optional[int]) -> bool:
        """Record an ACTIVE claim for ``run_id`` (idempotent upsert — a re-claim of
        the same run refreshes its lease). Returns True on persist. Best-effort:
        a store failure returns False and the caller proceeds WITHOUT a claim
        (never blocks a render on a store hiccup)."""
        if not run_id or not self._ensure():
            return False
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO reservations "
                    "(run_id, worker_id, gpu, task, peak_bytes, state, reason, "
                    " made_room, evicted_json, created_at, heartbeat_at, "
                    " lease_ttl_s, released_at) "
                    "VALUES (?,?,?,?,?, 'active', NULL, 0, NULL, ?, ?, ?, NULL) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "  worker_id=excluded.worker_id, gpu=excluded.gpu, "
                    "  task=excluded.task, peak_bytes=excluded.peak_bytes, "
                    "  state='active', reason=NULL, heartbeat_at=excluded.heartbeat_at, "
                    "  lease_ttl_s=excluded.lease_ttl_s, released_at=NULL",
                    (run_id, worker_id, gpu, task,
                     (int(peak_bytes) if peak_bytes else None),
                     now, now, self.lease_ttl_s))
            self._ok()
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_failure("claim", exc)
            return False

    def refresh(self, run_id: str) -> bool:
        """Bump the lease anchor for an ACTIVE claim (the heartbeat). No-op (False)
        for an unknown/terminal claim. Best-effort."""
        if not run_id or not self._ensure():
            return False
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE reservations SET heartbeat_at=? "
                    "WHERE run_id=? AND state='active'", (time.time(), run_id))
            self._ok()
            return cur.rowcount == 1
        except Exception as exc:  # noqa: BLE001
            self._note_failure("refresh", exc)
            return False

    def note_make_room(self, run_id: str, evicted: List[str]) -> None:
        """Record what acquisition's make-room yielded (for the listing). Best-effort."""
        if not run_id or not self._ensure():
            return
        import json as _json
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE reservations SET made_room=?, evicted_json=? "
                    "WHERE run_id=?",
                    (1 if evicted else 0, _json.dumps(list(evicted or [])), run_id))
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("note_make_room", exc)

    def release(self, run_id: str, reason: Optional[str] = None,
                state: str = "released") -> bool:
        """Terminal a claim (release on any terminal run path, or expired). Idempotent:
        releasing an unknown/already-terminal claim is a clean no-op. Best-effort."""
        if not run_id or not self._ensure():
            return False
        if state not in ("released", "expired"):
            state = "released"
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE reservations SET state=?, reason=?, released_at=? "
                    "WHERE run_id=? AND state='active'",
                    (state, reason, time.time(), run_id))
            self._ok()
            return cur.rowcount == 1
        except Exception as exc:  # noqa: BLE001
            self._note_failure("release", exc)
            return False

    # -- sweeps / reads ------------------------------------------------------
    def _sweep_expired(self, conn) -> None:
        """Flip lapsed-lease active claims to 'expired' (self-healing). An orphaned
        claim (crashed run, no refresher) frees the card here without operator
        action. Called on every read so a listing/accounting is always honest."""
        conn.execute(
            "UPDATE reservations SET state='expired', "
            "  reason=COALESCE(reason, 'lease expired (no heartbeat)'), "
            "  released_at=? "
            "WHERE state='active' AND (heartbeat_at + lease_ttl_s) < ?",
            (time.time(), time.time()))

    def active(self, worker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Live (unexpired) ACTIVE claims — optionally scoped to one worker. Runs
        the expiry sweep first so a lapsed claim never counts as active. [] on a
        store error (fail-open: no claim beats a broken-store deadlock)."""
        if not self._ensure():
            return []
        try:
            with self._connect() as conn:
                self._sweep_expired(conn)
                if worker_id is not None:
                    cur = conn.execute(
                        "SELECT * FROM reservations WHERE state='active' "
                        "AND worker_id=?", (worker_id,))
                else:
                    cur = conn.execute(
                        "SELECT * FROM reservations WHERE state='active'")
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            self._ok()
            return rows
        except Exception as exc:  # noqa: BLE001
            self._note_failure("active", exc)
            return []

    def reserved_bytes(self, worker_id: str) -> int:
        """Total peak bytes CURRENTLY reserved on a worker's card (admission-respect:
        these bytes are not free for other placements). 0 on unknown/store error —
        so a store hiccup never wrongly starves placement."""
        if not worker_id:
            return 0
        return sum(int(r.get("peak_bytes") or 0) for r in self.active(worker_id))

    def active_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """READ-ONLY {run_id: row} of the claims that are active RIGHT NOW — the
        listing/observability read (k57).

        Unlike ``active()``/``get()`` this NEVER writes: the lapsed-lease sweep is
        expressed as a WHERE predicate (``heartbeat_at + lease_ttl_s >= now``)
        instead of the UPDATE those paths run, and the connection is opened
        ``mode=ro``. That matters because the placement projection behind
        GET /video/jobs consulted the registry ONCE PER ROW: each call took a write
        lock on a store that live renderers heartbeat into, so a 40-row listing
        serialized behind ~40 write transactions (busy_timeout 2s each) and the feed
        hung for minutes. A read path must never contend with a running render — an
        unexpired-only SELECT is exactly as honest, since the sweep's only effect on
        a READER was hiding the very rows this predicate hides.

        {} on any store error (fail-open: no claim beats a broken-store stall)."""
        if not self._ensure():
            return {}
        try:
            conn = retry_on_emfile(lambda: sqlite3.connect(
                "file:" + self.path + "?mode=ro", uri=True, timeout=2.0))
            try:
                conn.execute("PRAGMA busy_timeout=2000")
                cur = conn.execute(
                    "SELECT * FROM reservations WHERE state='active' "
                    "AND (heartbeat_at + lease_ttl_s) >= ?", (time.time(),))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            finally:
                conn.close()
            self._ok()
            return {str(r.get("run_id")): r for r in rows if r.get("run_id")}
        except Exception as exc:  # noqa: BLE001
            self._note_failure("active_snapshot", exc)
            return {}

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not run_id or not self._ensure():
            return None
        try:
            with self._connect() as conn:
                self._sweep_expired(conn)
                cur = conn.execute(
                    "SELECT * FROM reservations WHERE run_id=?", (run_id,))
                cols = [c[0] for c in cur.description]
                row = cur.fetchone()
            self._ok()
            return dict(zip(cols, row)) if row else None
        except Exception as exc:  # noqa: BLE001
            self._note_failure("get", exc)
            return None

    def listing(self, include_terminal: bool = False,
                limit: int = 100) -> List[Dict[str, Any]]:
        """Read-only view for GET /llm/reservations. Active claims by default;
        include_terminal adds recent released/expired rows (newest first) so the
        console can show what just finished / self-expired."""
        if not self._ensure():
            return []
        import json as _json
        try:
            with self._connect() as conn:
                self._sweep_expired(conn)
                if include_terminal:
                    cur = conn.execute(
                        "SELECT * FROM reservations "
                        "ORDER BY (state='active') DESC, "
                        "  COALESCE(released_at, heartbeat_at) DESC LIMIT ?",
                        (int(limit),))
                else:
                    cur = conn.execute(
                        "SELECT * FROM reservations WHERE state='active' "
                        "ORDER BY created_at DESC LIMIT ?", (int(limit),))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            self._ok()
        except Exception as exc:  # noqa: BLE001
            self._note_failure("listing", exc)
            return []
        now = time.time()
        for r in rows:
            ev = r.pop("evicted_json", None)
            try:
                r["evicted"] = _json.loads(ev) if ev else []
            except Exception:  # noqa: BLE001
                r["evicted"] = []
            r["made_room"] = bool(r.get("made_room"))
            lease_at = (r.get("heartbeat_at") or 0) + (r.get("lease_ttl_s") or 0)
            r["lease_expires_in_s"] = (round(lease_at - now, 1)
                                       if r.get("state") == "active" else None)
        return rows


# One process-wide registry (central). Best-effort throughout; stdlib-only.
reservation_registry = ReservationRegistry()


def reserved_bytes(worker_id: str) -> int:
    """Module helper for the central admission-respect chokepoint (fleet_snapshot)."""
    try:
        return reservation_registry.reserved_bytes(worker_id)
    except Exception:  # noqa: BLE001 — admission-respect must never break placement
        return 0
