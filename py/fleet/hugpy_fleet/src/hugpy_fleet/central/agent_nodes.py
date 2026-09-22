"""P3.1 — the agent-node registry + dispatch store (SQLite, stdlib-only).

Phase 3 gives central a fleet of *agent nodes*: remote P2.7 daemons that
enroll and heartbeat like GPU workers, and that the operator dispatches tasks
to. This module is their cross-process store — the same discipline as
``comms.shared.SqliteMirror`` (short-lived connections, WAL, one shared db
file), so it is correct under gunicorn ``--workers 3``: a ``register`` can land
on process B while a ``heartbeat`` for that node lands on process A, and both
see the same rows.

It differs from the job *mirror* in one deliberate way: this table is the
SOURCE OF TRUTH, not a best-effort reflection of an in-memory store. There is
no per-process fallback to degrade to, so a failed write must surface (the
route returns 500) rather than silently swallow — losing a node registration
or a dispatched task is never acceptable. (The job mirror can swallow because
each process still holds the authoritative in-memory Job.)

Three concerns, two tables, one db file (shared with the jobs mirror):

    agent_nodes  — one row per enrolled node. Carries the sha256 of the
                   node's enroll token (NEVER the plaintext — that is returned
                   exactly once, from register), plus the last heartbeat's
                   status/current_task/version and a last_seen clock.
    agent_tasks  — the dispatch queue. Each dispatch appends a row; a node
                   pulls with a monotonic ``since`` cursor (the autoincrement
                   seq), so pulls are idempotent and need no delivery bit.
                   P3.1b adds a completion channel: ``complete_task`` transitions
                   a row queued → done/error with a size-capped ``result`` +
                   ``finished_at``. LIFECYCLE CHOICE: the status transition lives
                   ONLY on the result route — the pull (``tasks_since``) stays a
                   pure, side-effect-free, re-pullable GET (its whole point is
                   "no delivery bit"), so there is no ``running`` state written on
                   pull; a task is queued until its node reports it done/error,
                   and the node's heartbeat (busy + current_task=<seq>) is the
                   in-flight signal meanwhile.

Tokens: ``agt_`` + hex, sha256-hashed at rest (same shape as the enrollment /
principal token stores). Authentication is (node_id, token) -> the token must
hash to THIS node's stored hash and the node must be un-revoked. Fail-closed:
a missing/mismatched/revoked token authenticates nothing.

The db path is ``comms.shared.default_db_path()`` (HUGPY_COMMS_DB, else a
per-user file) — the same file the job mirror uses. A bare ``AgentNodeStore()``
in a test can be pointed at a scratch path.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from hugpy_control.shared import default_db_path, retry_on_emfile

_TOKEN_PREFIX = "agt_"
_NODE_PREFIX = "agn_"
_TASK_PREFIX = "atsk_"

# P3.1b — a completed task's stored result is size-capped (a run's answer is
# operator-defined text; without a cap a node could push an unbounded blob into
# the shared comms db). Over the cap the result is TRUNCATED (never rejected —
# refusing would leave the task un-finalizable), with a byte-count marker.
_MAX_RESULT_BYTES = 65536
_TASK_TERMINAL = ("done", "error")

_SCHEMA_NODES = """
CREATE TABLE IF NOT EXISTS agent_nodes (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    host          TEXT,
    capabilities  TEXT NOT NULL DEFAULT '[]',
    token_hash    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'enrolled',
    current_task  TEXT,
    version       TEXT,
    revoked       INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    last_seen     REAL
);
"""

_SCHEMA_TASKS = """
CREATE TABLE IF NOT EXISTS agent_tasks (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    task        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    created_at  REAL NOT NULL,
    result      TEXT,
    finished_at REAL
);
"""

# P3.1b additive columns, backfilled onto a db created before the result route
# existed (SQLite ADD COLUMN is online + cheap). Applied idempotently in
# _ensure, guarded by PRAGMA table_info — a fresh db already has them from
# _SCHEMA_TASKS, so the guard makes this a no-op there.
_TASK_MIGRATIONS = (
    ("result", "ALTER TABLE agent_tasks ADD COLUMN result TEXT"),
    ("finished_at", "ALTER TABLE agent_tasks ADD COLUMN finished_at REAL"),
)

_SCHEMA_TASK_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_agent_tasks_node_seq "
    "ON agent_tasks (node_id, seq)"
)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _loads_list(raw: Any) -> list:
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return list(v) if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def _cap_result(result: Any) -> Optional[str]:
    """Coerce a reported result to a size-capped string. None stays None; a
    non-str is JSON-encoded (so a node can report structured output); anything
    over ``_MAX_RESULT_BYTES`` is truncated on a UTF-8 boundary with a
    byte-count marker (the task still finalizes — see _MAX_RESULT_BYTES)."""
    if result is None:
        return None
    s = result if isinstance(result, str) else json.dumps(result)
    raw = s.encode("utf-8")
    if len(raw) <= _MAX_RESULT_BYTES:
        return s
    marker = "\n…[truncated %d bytes]" % (len(raw) - _MAX_RESULT_BYTES)
    keep = max(_MAX_RESULT_BYTES - len(marker.encode("utf-8")), 0)
    return raw[:keep].decode("utf-8", "ignore") + marker


class AgentNodeStore:
    """Authoritative, cross-process registry + dispatch queue for agent nodes.

    Every method opens a short-lived WAL connection — no shared handles across
    threads, no pooling to get wrong (exactly the SqliteMirror discipline).
    Writes/reads are NOT swallowed: this is the source of truth, so a db fault
    propagates to the caller (fail-closed) instead of pretending success.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_db_path()
        self._init_lock = threading.Lock()
        self._initialized = False

    # -- plumbing ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # This store is the SOURCE OF TRUTH for agent-node registration — a
        # failed open here surfaces as a 500 (by design; no per-process
        # fallback). The restart-burst EMFILE is transient, so retry the open
        # (see comms.shared.retry_on_emfile) before letting it propagate.
        conn = retry_on_emfile(lambda: sqlite3.connect(self.path, timeout=5.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with self._connect() as conn:
                conn.execute(_SCHEMA_NODES)
                conn.execute(_SCHEMA_TASKS)
                conn.execute(_SCHEMA_TASK_INDEX)
                cols = {r["name"] for r in
                        conn.execute("PRAGMA table_info(agent_tasks)")}
                for col, ddl in _TASK_MIGRATIONS:
                    if col not in cols:
                        conn.execute(ddl)
            self._initialized = True

    # -- serialization -------------------------------------------------------
    @staticmethod
    def _public_node(row: sqlite3.Row) -> dict[str, Any]:
        """Safe shape for API callers — NEVER includes token_hash/plaintext."""
        return {
            "id": row["id"],
            "name": row["name"],
            "host": row["host"],
            "capabilities": _loads_list(row["capabilities"]),
            "status": row["status"],
            "current_task": row["current_task"],
            "version": row["version"],
            "revoked": bool(row["revoked"]),
            "created_at": row["created_at"],
            "last_seen": row["last_seen"],
        }

    @staticmethod
    def _task_view(row: sqlite3.Row) -> dict[str, Any]:
        try:
            task = json.loads(row["task"])
        except Exception:
            task = row["task"]
        return {
            "seq": row["seq"],
            "id": row["id"],
            "node_id": row["node_id"],
            "task": task,
            "status": row["status"],
            "created_at": row["created_at"],
            "result": row["result"],
            "finished_at": row["finished_at"],
        }

    # -- node lifecycle ------------------------------------------------------
    def register(self, *, name: str, host: str = "",
                 capabilities: Optional[list] = None) -> dict[str, Any]:
        """Enroll a node. Returns the public node view PLUS a one-time
        ``token`` (the node's enroll credential — shown once, only the hash is
        stored)."""
        self._ensure()
        token = _TOKEN_PREFIX + secrets.token_hex(24)
        node_id = _NODE_PREFIX + uuid.uuid4().hex[:12]
        now = time.time()
        caps = list(capabilities or [])
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_nodes "
                "(id, name, host, capabilities, token_hash, status, "
                " current_task, version, revoked, created_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, 'enrolled', NULL, NULL, 0, ?, ?)",
                (node_id, str(name), str(host or ""), json.dumps(caps),
                 _hash(token), now, now))
            row = conn.execute("SELECT * FROM agent_nodes WHERE id=?",
                               (node_id,)).fetchone()
        view = self._public_node(row)
        view["token"] = token  # shown ONCE; never stored or returned again
        return view

    def get(self, node_id: str) -> Optional[dict[str, Any]]:
        self._ensure()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_nodes WHERE id=?",
                               (node_id,)).fetchone()
        return self._public_node(row) if row else None

    def all(self) -> list[dict[str, Any]]:
        self._ensure()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_nodes ORDER BY created_at").fetchall()
        return [self._public_node(r) for r in rows]

    def authenticate(self, node_id: str, token: Optional[str]) -> bool:
        """True iff ``token`` is THIS node's enroll token and the node is
        un-revoked. Fail-closed on every other path (missing token, wrong
        prefix, unknown node, hash mismatch, revoked)."""
        if not token or not token.startswith(_TOKEN_PREFIX):
            return False
        self._ensure()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token_hash, revoked FROM agent_nodes WHERE id=?",
                (node_id,)).fetchone()
        if row is None or row["revoked"]:
            return False
        return secrets.compare_digest(str(row["token_hash"]), _hash(token))

    def heartbeat(self, node_id: str, *, status: Optional[str] = None,
                  current_task: Optional[str] = None,
                  version: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Record a beat. Returns the updated public view, or None if central
        has no such node (the caller answers 410 -> re-register). Only the
        provided fields are written; last_seen always bumps."""
        self._ensure()
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_nodes WHERE id=?",
                               (node_id,)).fetchone()
            if row is None:
                return None
            new_status = status if status is not None else row["status"]
            new_task = current_task if current_task is not None \
                else row["current_task"]
            new_version = version if version is not None else row["version"]
            conn.execute(
                "UPDATE agent_nodes SET status=?, current_task=?, version=?, "
                "last_seen=? WHERE id=?",
                (new_status, new_task, new_version, now, node_id))
            row = conn.execute("SELECT * FROM agent_nodes WHERE id=?",
                               (node_id,)).fetchone()
        return self._public_node(row)

    def revoke(self, node_id: str) -> bool:
        self._ensure()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE agent_nodes SET revoked=1 WHERE id=?", (node_id,))
        return cur.rowcount > 0

    # -- dispatch queue ------------------------------------------------------
    def dispatch(self, node_id: str, task: Any) -> Optional[dict[str, Any]]:
        """Queue a task for a node. Returns the queued task view, or None if
        the node is unknown/revoked (the caller answers 404)."""
        self._ensure()
        now = time.time()
        task_id = _TASK_PREFIX + uuid.uuid4().hex[:12]
        with self._connect() as conn:
            node = conn.execute(
                "SELECT revoked FROM agent_nodes WHERE id=?",
                (node_id,)).fetchone()
            if node is None or node["revoked"]:
                return None
            conn.execute(
                "INSERT INTO agent_tasks (id, node_id, task, status, created_at) "
                "VALUES (?, ?, ?, 'queued', ?)",
                (task_id, node_id, json.dumps(task), now))
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE id=? ORDER BY seq DESC LIMIT 1",
                (task_id,)).fetchone()
        return self._task_view(row)

    def tasks_since(self, node_id: str,
                    since: int = 0) -> list[dict[str, Any]]:
        """Tasks queued for ``node_id`` with seq > ``since``, oldest first.
        The monotonic seq makes the pull idempotent — the node advances its
        cursor to the max seq it has seen and re-requests from there."""
        self._ensure()
        try:
            since_i = int(since or 0)
        except (TypeError, ValueError):
            since_i = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_tasks WHERE node_id=? AND seq>? "
                "ORDER BY seq ASC", (node_id, since_i)).fetchall()
        return [self._task_view(r) for r in rows]

    def get_task(self, node_id: str,
                 seq: Any) -> Optional[dict[str, Any]]:
        """One task's full view (incl. result/finished_at), scoped to its owning
        node — the operator drill-in for P3.3. None if there is no such seq for
        THIS node (a wrong-node or unknown seq → the route answers 404)."""
        self._ensure()
        try:
            seq_i = int(seq)
        except (TypeError, ValueError):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE seq=? AND node_id=?",
                (seq_i, node_id)).fetchone()
        return self._task_view(row) if row else None

    def complete_task(self, node_id: str, seq: Any, *, status: str,
                      result: Any = None) -> dict[str, Any]:
        """P3.1b — record a node's completion of task ``seq``. Transitions the
        row queued → ``status`` (``done``/``error``), storing the size-capped
        result + ``finished_at``. Scoped to the owning node (fail-closed).

        Returns a small outcome dict the route maps to a status code:
          * ``{"ok": True,  "task": <view>}``                 finalized → 200
          * ``{"ok": False, "reason": "not_found"}``          no such seq for
            this node (unknown/other node's task) → 404
          * ``{"ok": False, "reason": "conflict", "task":…}`` already finalized
            → 409; the FIRST report wins and is NOT overwritten (idempotent-safe
            re-post; a node treats 200 AND 409 as 'recorded, advance cursor')."""
        self._ensure()
        try:
            seq_i = int(seq)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "not_found"}
        status = status if status in _TASK_TERMINAL else "error"
        capped = _cap_result(result)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_tasks WHERE seq=?",
                               (seq_i,)).fetchone()
            if row is None or row["node_id"] != node_id:
                return {"ok": False, "reason": "not_found"}
            if row["status"] in _TASK_TERMINAL:
                return {"ok": False, "reason": "conflict",
                        "task": self._task_view(row)}
            conn.execute(
                "UPDATE agent_tasks SET status=?, result=?, finished_at=? "
                "WHERE seq=?", (status, capped, now, seq_i))
            row = conn.execute("SELECT * FROM agent_tasks WHERE seq=?",
                               (seq_i,)).fetchone()
        return {"ok": True, "task": self._task_view(row)}


# Module singleton (mirrors job_store / principal_store / token_store). Points
# at the shared comms db by default; tests construct their own with a path.
agent_node_store = AgentNodeStore()
