"""Durable DAG runtime (k111): execute a :class:`PlanGraph` node-by-node under a
SQLite (WAL) journal so a killed process resumes without repeating finished work.

Directive §5 invariants this module owns:

* **record-before-run** — every node transition is written and committed BEFORE
  the executor is called; the executor never sees a node the journal does not
  already show as RUNNING.
* **leases + heartbeat** — a RUNNING node carries ``lease_owner`` and
  ``lease_expires_at``; :meth:`DagRuntime.resume` returns expired leases to
  PENDING (attempt count preserved) instead of trusting a dead worker.
* **idempotency** — each attempt has a key ``(run, revision, node, attempt)``
  or the node's declared ``idempotency_key``; a SUCCEEDED key is never re-run.
* **content-addressed cache** — ``cache_key`` = node.cache_key or
  sha256(capability, params, input digests). Cache rows are global (across
  runs), so an identical expensive node completed in any earlier run is served
  from the journal with ``cached=True`` on its receipt.
* **controls** — pause / cancel / approve / reject / retry_node / replan are
  journaled control events, not in-memory flags.
* **graph revisions** — replan stores a new ``PlanGraph`` revision; nodes whose
  structural digest is unchanged keep their state, replaced nodes and their
  descendants reset. Revisions are bounded by ``repair_budget``.
* **reservations** — resource grants and releases are journaled rows with
  receipts, against an injectable :class:`ResourceBroker`.

What this module deliberately is NOT: a scheduler for a fleet. One process,
one journal, cooperative stepping (``run()`` loops ``step()`` until nothing is
READY). Fleet fan-out is Phase 10 and keeps this journal as the source of truth.

Executor protocol::

    def executor(node: PlanNode, inputs: dict[str, Any], ctx: NodeContext) -> Mapping[str, Any] | NodeResult

``inputs`` is keyed by the node's input-port names (``many`` ports receive
lists). The return maps output-port names to JSON-safe payloads. Raising
marks the attempt FAILED and is classified via ``runtime.classify_failure``
when importable. FANOUT nodes are invoked ``candidates`` times (``ctx.candidate``
0..n-1) and their outputs are collected into lists per port. JOIN and GATE
nodes without a capability pass their inputs through; every other kind needs
the executor.

Acceptance: if ``evaluator`` is supplied it is called with
``(node, outputs, ctx)`` and must return a :class:`Scorecard` or ``None``. A
card with ``hard_pass=False`` fails the attempt with the card's ``repair_code``.
The evaluator is a seam so k112 (repair controller) and k115 (judge policy)
plug in without touching this file.

Conventions: stdlib-only, frozen slotted dataclasses, str-Enums, lossless
``to_dict``/``from_dict``, ``os.path`` only (no pathlib).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from hugpy_oracle.contracts import RepairCode, Scorecard
from hugpy_oracle.plan import PlanGraph, PlanNode, NodeKind, content_digest

__all__ = [
    "DagRuntime",
    "JournalError",
    "NodeContext",
    "NodeRecord",
    "NodeResult",
    "NodeState",
    "RepairBudgetExceeded",
    "ResourceBroker",
    "ResourceDenied",
    "RunJournal",
    "RunRecord",
    "RunState",
    "SelectionGap",
    "StepReport",
    "TERMINAL_NODE_STATES",
    "derive_cache_key",
]

SCHEMA_VERSION = 1
DEFAULT_LEASE_S = 300.0
DEFAULT_REPAIR_BUDGET = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _loads(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #


class NodeState(str, Enum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RunState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_NODE_STATES: frozenset[NodeState] = frozenset(
    {NodeState.SUCCEEDED, NodeState.FAILED, NodeState.REJECTED, NodeState.CANCELLED}
)
TERMINAL_RUN_STATES: frozenset[RunState] = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
)


class JournalError(RuntimeError):
    """The journal refused an operation (unknown run, bad transition, closed)."""


class RepairBudgetExceeded(JournalError):
    """A replan would exceed the run's bounded repair budget (invariant 6)."""


class ResourceDenied(RuntimeError):
    """The broker could not grant the node's :class:`ResourceRequest`."""


class SelectionGap(RuntimeError):
    """The selector found no model it could defend for this call. Carries the
    full decision so the receipt explains every rejection."""

    def __init__(self, capability: str, decision: Mapping[str, Any]) -> None:
        self.capability = capability
        self.decision = dict(decision)
        rej = [f"{r.get('model_id')}@{r.get('rejected_at')}" for r in decision.get("rejected", [])]
        super().__init__(f"{capability}: no selectable model ({decision.get('rationale')}); "
                         f"rejected={rej}")


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NodeResult:
    outputs: Mapping[str, Any]
    receipt: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NodeContext:
    run_id: str
    revision: int
    node_id: str
    attempt: int
    idempotency_key: str
    cache_key: str
    candidate: int = 0
    candidates: int = 1
    lease_owner: str = ""
    model_id: str | None = None                  # chosen by the selector for THIS call
    selection: Mapping[str, Any] | None = None   # SelectionDecision.to_dict()
    exclude_models: tuple[str, ...] = ()         # models that already failed this node

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "revision": self.revision, "node_id": self.node_id,
            "attempt": self.attempt, "idempotency_key": self.idempotency_key,
            "cache_key": self.cache_key, "candidate": self.candidate,
            "candidates": self.candidates, "lease_owner": self.lease_owner,
            "model_id": self.model_id,
            "selection": dict(self.selection) if self.selection is not None else None,
            "exclude_models": tuple(self.exclude_models),
        }


@dataclass(frozen=True, slots=True)
class NodeRecord:
    run_id: str
    node_id: str
    revision: int
    state: NodeState
    attempt: int = 0
    idempotency_key: str | None = None
    cache_key: str | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    outputs: Mapping[str, Any] | None = None
    receipt: Mapping[str, Any] | None = None
    failure: str | None = None
    failure_class: str | None = None
    repair_code: RepairCode | None = None
    approved: bool = False
    cached: bool = False
    force_rerun: bool = False
    updated_at: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_NODE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "node_id": self.node_id, "revision": self.revision,
            "state": self.state.value, "attempt": self.attempt,
            "idempotency_key": self.idempotency_key, "cache_key": self.cache_key,
            "lease_owner": self.lease_owner, "lease_expires_at": self.lease_expires_at,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "outputs": dict(self.outputs) if self.outputs is not None else None,
            "receipt": dict(self.receipt) if self.receipt is not None else None,
            "failure": self.failure, "failure_class": self.failure_class,
            "repair_code": self.repair_code.value if self.repair_code else None,
            "approved": self.approved, "cached": self.cached,
            "force_rerun": self.force_rerun, "updated_at": self.updated_at,
        }

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "NodeRecord":
        rc = row["repair_code"]
        return cls(
            run_id=row["run_id"], node_id=row["node_id"], revision=row["revision"],
            state=NodeState(row["state"]), attempt=row["attempt"],
            idempotency_key=row["idempotency_key"], cache_key=row["cache_key"],
            lease_owner=row["lease_owner"], lease_expires_at=row["lease_expires_at"],
            started_at=row["started_at"], ended_at=row["ended_at"],
            outputs=_loads(row["outputs_json"]), receipt=_loads(row["receipt_json"]),
            failure=row["failure"], failure_class=row["failure_class"],
            repair_code=RepairCode(rc) if rc else None,
            approved=bool(row["approved"]), cached=bool(row["cached"]),
            force_rerun=bool(row["force_rerun"]), updated_at=row["updated_at"],
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    graph_id: str
    goal_digest: str
    revision: int
    state: RunState
    repair_budget: int
    revisions_used: int
    created_at: str
    updated_at: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "graph_id": self.graph_id, "goal_digest": self.goal_digest,
            "revision": self.revision, "state": self.state.value,
            "repair_budget": self.repair_budget, "revisions_used": self.revisions_used,
            "created_at": self.created_at, "updated_at": self.updated_at, "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class StepReport:
    """What one :meth:`DagRuntime.step` did — the unit a UI/agent polls."""
    run_id: str
    run_state: RunState
    executed: tuple[str, ...] = ()
    cached: tuple[str, ...] = ()
    succeeded: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    awaiting_approval: tuple[str, ...] = ()
    blocked_on_resources: tuple[str, ...] = ()
    ready_remaining: int = 0

    @property
    def idle(self) -> bool:
        return not (self.executed or self.cached or self.awaiting_approval)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "run_state": self.run_state.value,
            "executed": list(self.executed), "cached": list(self.cached),
            "succeeded": list(self.succeeded), "failed": list(self.failed),
            "awaiting_approval": list(self.awaiting_approval),
            "blocked_on_resources": list(self.blocked_on_resources),
            "ready_remaining": self.ready_remaining,
        }


# --------------------------------------------------------------------------- #
# resource broker
# --------------------------------------------------------------------------- #


class ResourceBroker:
    """Capacity accounting for :class:`ResourceRequest`. In-memory by design —
    the *grants* are journaled (table ``reservations``) so a restart can
    reconcile; live capacity comes from Hugpy central in production and from
    this class in tests. ``None`` capacity means unlimited."""

    def __init__(self, vram_gib: float | None = None, ram_gib: float | None = None,
                 gpus: int | None = None) -> None:
        self.cap_vram = vram_gib
        self.cap_ram = ram_gib
        self.cap_gpus = gpus
        self._held: dict[str, tuple[float, float, int]] = {}
        self._lock = threading.Lock()

    def _used(self) -> tuple[float, float, int]:
        v = sum(h[0] for h in self._held.values())
        r = sum(h[1] for h in self._held.values())
        g = sum(h[2] for h in self._held.values())
        return v, r, g

    def can_grant(self, node: PlanNode) -> bool:
        req = node.resources
        v, r, g = self._used()
        want_v, want_r, want_g = req.vram_gib or 0.0, req.ram_gib or 0.0, 1 if req.gpu else 0
        if self.cap_vram is not None and v + want_v > self.cap_vram:
            return False
        if self.cap_ram is not None and r + want_r > self.cap_ram:
            return False
        if self.cap_gpus is not None and g + want_g > self.cap_gpus:
            return False
        return True

    def grant(self, reservation_id: str, node: PlanNode) -> dict[str, Any]:
        with self._lock:
            if not self.can_grant(node):
                raise ResourceDenied(f"{node.node_id}: insufficient capacity")
            req = node.resources
            self._held[reservation_id] = (req.vram_gib or 0.0, req.ram_gib or 0.0, 1 if req.gpu else 0)
            return {"vram_gib": req.vram_gib, "ram_gib": req.ram_gib, "gpu": req.gpu}

    def release(self, reservation_id: str) -> None:
        with self._lock:
            self._held.pop(reservation_id, None)

    @property
    def held(self) -> dict[str, tuple[float, float, int]]:
        return dict(self._held)


# --------------------------------------------------------------------------- #
# journal
# --------------------------------------------------------------------------- #

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, graph_id TEXT NOT NULL, goal_digest TEXT NOT NULL,
    revision INTEGER NOT NULL, state TEXT NOT NULL,
    repair_budget INTEGER NOT NULL, revisions_used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS graph_revisions (
    run_id TEXT NOT NULL, revision INTEGER NOT NULL, parent_revision INTEGER,
    reason TEXT NOT NULL DEFAULT '', graph_json TEXT NOT NULL,
    structure_digest TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, revision)
);
CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL, node_id TEXT NOT NULL, revision INTEGER NOT NULL,
    state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT, cache_key TEXT, lease_owner TEXT, lease_expires_at REAL,
    started_at TEXT, ended_at TEXT, outputs_json TEXT, receipt_json TEXT,
    failure TEXT, failure_class TEXT, repair_code TEXT,
    approved INTEGER NOT NULL DEFAULT 0, cached INTEGER NOT NULL DEFAULT 0,
    force_rerun INTEGER NOT NULL DEFAULT 0,
    node_digest TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, node_id)
);
CREATE TABLE IF NOT EXISTS transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, node_id TEXT,
    from_state TEXT, to_state TEXT NOT NULL, attempt INTEGER, revision INTEGER,
    reason TEXT NOT NULL DEFAULT '', at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS transitions_run ON transitions(run_id, seq);
CREATE TABLE IF NOT EXISTS controls (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    node_id TEXT, payload_json TEXT, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY, capability TEXT, outputs_json TEXT NOT NULL,
    receipt_json TEXT, produced_by_run TEXT NOT NULL, produced_by_node TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node_id TEXT NOT NULL,
    attempt INTEGER NOT NULL, request_json TEXT NOT NULL, grant_json TEXT,
    granted_at TEXT NOT NULL, released_at TEXT, release_reason TEXT
);
"""


class RunJournal:
    """The durable record. One SQLite file, WAL mode, ``busy_timeout`` set.

    Every public mutator commits before returning — that is what makes
    "record-before-run" a property of the journal rather than a promise of
    the caller."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_DDL)  # executescript commits on its own
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- plumbing ---------------------------------------------------------- #

    class _Tx:
        def __init__(self, journal: "RunJournal") -> None:
            self.j = journal

        def __enter__(self) -> sqlite3.Connection:
            self.j._lock.acquire()
            self.j._conn.execute("BEGIN IMMEDIATE")
            return self.j._conn

        def __exit__(self, exc_type, exc, tb) -> None:
            try:
                if exc_type is None:
                    self.j._conn.execute("COMMIT")
                else:
                    self.j._conn.execute("ROLLBACK")
            finally:
                self.j._lock.release()

    def _tx(self) -> "RunJournal._Tx":
        return RunJournal._Tx(self)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def schema_version(self) -> int:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    # -- runs / revisions --------------------------------------------------- #

    def create_run(self, graph: PlanGraph, run_id: str | None = None, *,
                   repair_budget: int = DEFAULT_REPAIR_BUDGET) -> RunRecord:
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        with self._tx() as c:
            if c.execute("SELECT 1 FROM runs WHERE run_id=?", (rid,)).fetchone():
                raise JournalError(f"run exists: {rid}")
            c.execute(
                "INSERT INTO runs(run_id, graph_id, goal_digest, revision, state, repair_budget, "
                "revisions_used, created_at, updated_at) VALUES(?,?,?,?,?,?,0,?,?)",
                (rid, graph.graph_id, graph.goal_digest, graph.revision, RunState.RUNNING.value,
                 int(repair_budget), now, now),
            )
            self._insert_revision(c, rid, graph, reason="initial", now=now)
            for node in graph.nodes:
                c.execute(
                    "INSERT INTO nodes(run_id, node_id, revision, state, node_digest, updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (rid, node.node_id, graph.revision, NodeState.PENDING.value,
                     content_digest(node.to_dict()), now),
                )
                self._transition(c, rid, node.node_id, None, NodeState.PENDING, 0,
                                 graph.revision, "created", now)
            self._transition(c, rid, None, None, RunState.RUNNING, None, graph.revision, "run created", now)
        return self.run(rid)

    def _insert_revision(self, c: sqlite3.Connection, run_id: str, graph: PlanGraph,
                         *, reason: str, now: str) -> None:
        c.execute(
            "INSERT INTO graph_revisions(run_id, revision, parent_revision, reason, graph_json, "
            "structure_digest, created_at) VALUES(?,?,?,?,?,?,?)",
            (run_id, graph.revision, graph.parent_revision, reason, _dumps(graph.to_dict()),
             graph.structure_digest(), now),
        )

    def run(self, run_id: str) -> RunRecord:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise JournalError(f"unknown run: {run_id}")
        return RunRecord(
            run_id=row["run_id"], graph_id=row["graph_id"], goal_digest=row["goal_digest"],
            revision=row["revision"], state=RunState(row["state"]),
            repair_budget=row["repair_budget"], revisions_used=row["revisions_used"],
            created_at=row["created_at"], updated_at=row["updated_at"], note=row["note"],
        )

    def runs(self) -> tuple[RunRecord, ...]:
        ids = [r["run_id"] for r in self._conn.execute("SELECT run_id FROM runs ORDER BY created_at")]
        return tuple(self.run(i) for i in ids)

    def graph(self, run_id: str, revision: int | None = None) -> PlanGraph:
        if revision is None:
            revision = self.run(run_id).revision
        row = self._conn.execute(
            "SELECT graph_json FROM graph_revisions WHERE run_id=? AND revision=?",
            (run_id, revision),
        ).fetchone()
        if row is None:
            raise JournalError(f"no revision {revision} for run {run_id}")
        return PlanGraph.from_dict(_loads(row["graph_json"]))

    def revisions(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            "SELECT revision, parent_revision, reason, structure_digest, created_at "
            "FROM graph_revisions WHERE run_id=? ORDER BY revision", (run_id,)
        ).fetchall()
        return tuple(dict(r) for r in rows)

    def set_run_state(self, run_id: str, state: RunState, reason: str = "") -> None:
        now = _utc_now()
        with self._tx() as c:
            cur = self.run(run_id)
            c.execute("UPDATE runs SET state=?, updated_at=?, note=? WHERE run_id=?",
                      (state.value, now, reason, run_id))
            self._transition(c, run_id, None, cur.state, state, None, cur.revision, reason, now)

    # -- nodes ------------------------------------------------------------- #

    def node(self, run_id: str, node_id: str) -> NodeRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=?", (run_id, node_id)
            ).fetchone()
        if row is None:
            raise JournalError(f"unknown node {node_id} in run {run_id}")
        return NodeRecord._from_row(row)

    def nodes(self, run_id: str) -> dict[str, NodeRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM nodes WHERE run_id=?", (run_id,)).fetchall()
        return {r["node_id"]: NodeRecord._from_row(r) for r in rows}

    def _transition(self, c: sqlite3.Connection, run_id: str, node_id: str | None,
                    from_state: Enum | None, to_state: Enum, attempt: int | None,
                    revision: int | None, reason: str, now: str) -> None:
        c.execute(
            "INSERT INTO transitions(run_id, node_id, from_state, to_state, attempt, revision, reason, at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (run_id, node_id, from_state.value if from_state else None, to_state.value,
             attempt, revision, reason, now),
        )

    def _update_node(self, c: sqlite3.Connection, run_id: str, node_id: str,
                     to_state: NodeState, reason: str, **cols: Any) -> NodeRecord:
        cur = self.node(run_id, node_id)
        now = _utc_now()
        cols["state"] = to_state.value
        cols["updated_at"] = now
        sets = ", ".join(f"{k}=?" for k in cols)
        c.execute(f"UPDATE nodes SET {sets} WHERE run_id=? AND node_id=?",
                  (*cols.values(), run_id, node_id))
        attempt = cols.get("attempt", cur.attempt)
        self._transition(c, run_id, node_id, cur.state, to_state, attempt, cur.revision, reason, now)
        return self.node(run_id, node_id)

    def transitions(self, run_id: str, node_id: str | None = None) -> tuple[dict[str, Any], ...]:
        if node_id is None:
            rows = self._conn.execute(
                "SELECT * FROM transitions WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM transitions WHERE run_id=? AND node_id=? ORDER BY seq",
                (run_id, node_id)).fetchall()
        return tuple(dict(r) for r in rows)

    # -- record-before-run primitives -------------------------------------- #

    def lease(self, run_id: str, node_id: str, *, owner: str, lease_s: float,
              idempotency_key: str, cache_key: str) -> NodeRecord:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state not in (NodeState.PENDING, NodeState.AWAITING_APPROVAL):
                raise JournalError(f"{node_id}: cannot lease from {cur.state.value}")
            return self._update_node(
                c, run_id, node_id, NodeState.LEASED, "leased",
                attempt=cur.attempt + 1, lease_owner=owner,
                lease_expires_at=time.time() + float(lease_s),
                idempotency_key=idempotency_key, cache_key=cache_key,
                started_at=None, ended_at=None, failure=None, failure_class=None,
                repair_code=None, outputs_json=None, receipt_json=None, cached=0,
                force_rerun=0,
            )

    def mark_running(self, run_id: str, node_id: str) -> NodeRecord:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state is not NodeState.LEASED:
                raise JournalError(f"{node_id}: cannot run from {cur.state.value}")
            return self._update_node(c, run_id, node_id, NodeState.RUNNING, "executor invoked",
                                     started_at=_utc_now())

    def heartbeat(self, run_id: str, node_id: str, *, owner: str, lease_s: float) -> None:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state is not NodeState.RUNNING or cur.lease_owner != owner:
                raise JournalError(f"{node_id}: heartbeat from non-owner or non-running node")
            c.execute("UPDATE nodes SET lease_expires_at=?, updated_at=? WHERE run_id=? AND node_id=?",
                      (time.time() + float(lease_s), _utc_now(), run_id, node_id))

    def succeed(self, run_id: str, node_id: str, outputs: Mapping[str, Any],
                receipt: Mapping[str, Any] | None, *, cached: bool = False,
                capability: str | None = None) -> NodeRecord:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state not in (NodeState.RUNNING, NodeState.LEASED):
                raise JournalError(f"{node_id}: cannot succeed from {cur.state.value}")
            rec = self._update_node(
                c, run_id, node_id, NodeState.SUCCEEDED, "cache hit" if cached else "succeeded",
                ended_at=_utc_now(), outputs_json=_dumps(dict(outputs)),
                receipt_json=_dumps(dict(receipt)) if receipt is not None else None,
                lease_owner=None, lease_expires_at=None, cached=1 if cached else 0,
            )
            if not cached and cur.cache_key:
                c.execute(
                    "INSERT OR IGNORE INTO cache(cache_key, capability, outputs_json, receipt_json, "
                    "produced_by_run, produced_by_node, created_at) VALUES(?,?,?,?,?,?,?)",
                    (cur.cache_key, capability, _dumps(dict(outputs)),
                     _dumps(dict(receipt)) if receipt is not None else None,
                     run_id, node_id, _utc_now()),
                )
            return rec

    def fail(self, run_id: str, node_id: str, failure: str, *, failure_class: str | None = None,
             repair_code: RepairCode | None = None, receipt: Mapping[str, Any] | None = None,
             retry: bool = False) -> NodeRecord:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state not in (NodeState.RUNNING, NodeState.LEASED):
                raise JournalError(f"{node_id}: cannot fail from {cur.state.value}")
            to = NodeState.PENDING if retry else NodeState.FAILED
            return self._update_node(
                c, run_id, node_id, to, "failed; retry scheduled" if retry else "failed",
                ended_at=_utc_now(), failure=failure, failure_class=failure_class,
                repair_code=repair_code.value if repair_code else None,
                receipt_json=_dumps(dict(receipt)) if receipt is not None else None,
                lease_owner=None, lease_expires_at=None,
            )

    def await_approval(self, run_id: str, node_id: str) -> NodeRecord:
        with self._tx() as c:
            cur = self.node(run_id, node_id)
            if cur.state is not NodeState.PENDING:
                raise JournalError(f"{node_id}: cannot gate from {cur.state.value}")
            return self._update_node(c, run_id, node_id, NodeState.AWAITING_APPROVAL, "approval gate")

    def set_state(self, run_id: str, node_id: str, state: NodeState, reason: str,
                  **cols: Any) -> NodeRecord:
        """Control-plane transition (reset/cancel/reject). Not for executors."""
        with self._tx() as c:
            return self._update_node(c, run_id, node_id, state, reason, **cols)

    def expire_leases(self, run_id: str, *, now: float | None = None) -> tuple[str, ...]:
        """Return LEASED/RUNNING nodes whose lease has lapsed to PENDING."""
        t = time.time() if now is None else now
        expired: list[str] = []
        with self._tx() as c:
            for nid, rec in self.nodes(run_id).items():
                if rec.state in (NodeState.LEASED, NodeState.RUNNING) and (
                    rec.lease_expires_at is None or rec.lease_expires_at <= t
                ):
                    self._update_node(c, run_id, nid, NodeState.PENDING,
                                      f"lease expired (owner={rec.lease_owner})",
                                      lease_owner=None, lease_expires_at=None)
                    expired.append(nid)
        return tuple(expired)

    # -- cache ------------------------------------------------------------- #

    def cache_lookup(self, cache_key: str) -> tuple[Mapping[str, Any], Mapping[str, Any] | None] | None:
        row = self._conn.execute("SELECT * FROM cache WHERE cache_key=?", (cache_key,)).fetchone()
        if row is None:
            return None
        return _loads(row["outputs_json"]), _loads(row["receipt_json"])

    def cache_size(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])

    # -- controls ---------------------------------------------------------- #

    def record_control(self, run_id: str, kind: str, node_id: str | None = None,
                       payload: Mapping[str, Any] | None = None) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO controls(run_id, kind, node_id, payload_json, at) VALUES(?,?,?,?,?)",
                (run_id, kind, node_id, _dumps(dict(payload)) if payload else None, _utc_now()),
            )

    def controls(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            "SELECT * FROM controls WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return tuple(dict(r) for r in rows)

    # -- reservations ------------------------------------------------------ #

    def record_reservation(self, reservation_id: str, run_id: str, node_id: str, attempt: int,
                           request: Mapping[str, Any], grant: Mapping[str, Any]) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO reservations(reservation_id, run_id, node_id, attempt, request_json, "
                "grant_json, granted_at) VALUES(?,?,?,?,?,?,?)",
                (reservation_id, run_id, node_id, attempt, _dumps(dict(request)),
                 _dumps(dict(grant)), _utc_now()),
            )

    def release_reservation(self, reservation_id: str, reason: str) -> dict[str, Any]:
        with self._tx() as c:
            c.execute("UPDATE reservations SET released_at=?, release_reason=? WHERE reservation_id=?",
                      (_utc_now(), reason, reservation_id))
            row = c.execute("SELECT * FROM reservations WHERE reservation_id=?",
                            (reservation_id,)).fetchone()
        return dict(row) if row else {}

    def reservations(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            "SELECT * FROM reservations WHERE run_id=? ORDER BY granted_at", (run_id,)).fetchall()
        return tuple(dict(r) for r in rows)

    def open_reservations(self, run_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(r for r in self.reservations(run_id) if r["released_at"] is None)

    # -- replan ------------------------------------------------------------ #

    def replan(self, run_id: str, new_graph: PlanGraph, reason: str) -> tuple[str, ...]:
        """Store ``new_graph`` as the next revision. Nodes whose digest is
        unchanged AND whose ancestors are all unchanged keep their record;
        everything else is reset to PENDING. Returns the reset node ids."""
        now = _utc_now()
        with self._tx() as c:
            cur = self.run(run_id)
            if cur.revisions_used + 1 > cur.repair_budget:
                raise RepairBudgetExceeded(
                    f"run {run_id}: repair budget {cur.repair_budget} exhausted")
            if new_graph.revision <= cur.revision:
                raise JournalError(
                    f"new revision {new_graph.revision} must exceed current {cur.revision}")
            old_nodes = self.nodes(run_id)
            old_digest = {
                r["node_id"]: r["node_digest"]
                for r in c.execute("SELECT node_id, node_digest FROM nodes WHERE run_id=?", (run_id,))
            }
            new_digest = {n.node_id: content_digest(n.to_dict()) for n in new_graph.nodes}
            changed: set[str] = {
                nid for nid, d in new_digest.items() if old_digest.get(nid) != d
            }
            # a node downstream of any changed node is invalid too
            for nid in list(new_digest):
                if new_graph.ancestors(nid) & changed:
                    changed.add(nid)
            self._insert_revision(c, run_id, new_graph, reason=reason, now=now)
            c.execute("UPDATE runs SET revision=?, revisions_used=revisions_used+1, updated_at=?, "
                      "state=? WHERE run_id=?",
                      (new_graph.revision, now, RunState.RUNNING.value, run_id))
            reset: list[str] = []
            kept_ids = set(new_digest)
            for nid in list(old_nodes):
                if nid not in kept_ids:
                    c.execute("DELETE FROM nodes WHERE run_id=? AND node_id=?", (run_id, nid))
                    self._transition(c, run_id, nid, old_nodes[nid].state, NodeState.CANCELLED,
                                     old_nodes[nid].attempt, new_graph.revision,
                                     "removed by replan", now)
            for node in new_graph.nodes:
                nid = node.node_id
                if nid in old_nodes and nid not in changed:
                    c.execute("UPDATE nodes SET revision=? WHERE run_id=? AND node_id=?",
                              (new_graph.revision, run_id, nid))
                    continue
                if nid in old_nodes:
                    c.execute(
                        "UPDATE nodes SET revision=?, state=?, node_digest=?, lease_owner=NULL, "
                        "lease_expires_at=NULL, started_at=NULL, ended_at=NULL, outputs_json=NULL, "
                        "receipt_json=NULL, failure=NULL, failure_class=NULL, repair_code=NULL, "
                        "cached=0, updated_at=? WHERE run_id=? AND node_id=?",
                        (new_graph.revision, NodeState.PENDING.value, new_digest[nid], now, run_id, nid),
                    )
                    self._transition(c, run_id, nid, old_nodes[nid].state, NodeState.PENDING,
                                     old_nodes[nid].attempt, new_graph.revision,
                                     f"reset by replan: {reason}", now)
                else:
                    c.execute(
                        "INSERT INTO nodes(run_id, node_id, revision, state, node_digest, updated_at) "
                        "VALUES(?,?,?,?,?,?)",
                        (run_id, nid, new_graph.revision, NodeState.PENDING.value, new_digest[nid], now),
                    )
                    self._transition(c, run_id, nid, None, NodeState.PENDING, 0,
                                     new_graph.revision, f"added by replan: {reason}", now)
                reset.append(nid)
            self._transition(c, run_id, None, cur.state, RunState.RUNNING, None,
                             new_graph.revision, f"replan: {reason}", now)
        return tuple(reset)


# --------------------------------------------------------------------------- #
# cache key
# --------------------------------------------------------------------------- #


def derive_cache_key(node: PlanNode, inputs: Mapping[str, Any]) -> str:
    """Content address of *what this node will do*: capability + params +
    the digests of its resolved inputs. Two nodes with the same key produce
    the same artifact, so the second one is a journal read."""
    if node.cache_key:
        return node.cache_key
    payload = {
        "capability": node.capability,
        "kind": node.kind.value,
        "params": (node.params.to_dict() if hasattr(node.params, "to_dict") else dict(node.params)),
        "candidates": node.candidates,
        "inputs": {k: content_digest(v) for k, v in sorted(inputs.items())},
    }
    return content_digest(payload)


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #

Executor = Callable[[PlanNode, dict[str, Any], NodeContext], Any]
Evaluator = Callable[[PlanNode, Mapping[str, Any], NodeContext], "Scorecard | None"]


class DagRuntime:
    """Cooperative stepper over a :class:`RunJournal`.

    ``executor`` does the work; ``evaluator`` (optional) judges it;
    ``broker`` (optional) accounts resources. All three are seams."""

    def __init__(self, journal: RunJournal, executor: Executor, *,
                 evaluator: Evaluator | None = None,
                 broker: ResourceBroker | None = None,
                 owner: str | None = None,
                 lease_s: float = DEFAULT_LEASE_S,
                 max_parallel: int = 1,
                 selector: Any = None) -> None:
        """``selector`` (optional) must offer ``for_node(node, ctx, inputs) ->
        SelectionDecision | None`` and ``record_outcome(node, ctx, *, ok,
        hard_pass, repair_code, latency_s)`` — see ``selection.Selector``.
        With it, every capability call is routed per candidate and every
        outcome is written back as reliability evidence."""
        self.journal = journal
        self.executor = executor
        self.evaluator = evaluator
        self.selector = selector
        self.broker = broker or ResourceBroker()
        self.owner = owner or f"{os.getpid()}@{os.uname().nodename if hasattr(os, 'uname') else 'host'}"
        self.lease_s = float(lease_s)
        self.max_parallel = max(1, int(max_parallel))

    # -- lifecycle --------------------------------------------------------- #

    def start(self, graph: PlanGraph, run_id: str | None = None, *,
              repair_budget: int = DEFAULT_REPAIR_BUDGET) -> RunRecord:
        graph.topological_order()  # raises CycleError before anything is journaled
        return self.journal.create_run(graph, run_id, repair_budget=repair_budget)

    def resume(self, run_id: str) -> tuple[str, ...]:
        """Reconcile after a crash: expire dead leases, release orphaned
        reservations, keep every SUCCEEDED node. Returns expired node ids."""
        rec = self.journal.run(run_id)
        for res in self.journal.open_reservations(run_id):
            self.broker.release(res["reservation_id"])
            self.journal.release_reservation(res["reservation_id"], "orphaned at resume")
        expired = self.journal.expire_leases(run_id)
        self.journal.record_control(run_id, "resume", payload={"expired": list(expired),
                                                                "owner": self.owner})
        if rec.state in (RunState.FAILED,) and expired:
            self.journal.set_run_state(run_id, RunState.RUNNING, "resumed after lease expiry")
        return expired

    # -- controls ---------------------------------------------------------- #

    def pause(self, run_id: str, reason: str = "operator pause") -> None:
        rec = self.journal.run(run_id)
        if rec.state in TERMINAL_RUN_STATES:
            raise JournalError(f"cannot pause a {rec.state.value} run")
        self.journal.record_control(run_id, "pause", payload={"reason": reason})
        self.journal.set_run_state(run_id, RunState.PAUSED, reason)

    def unpause(self, run_id: str, reason: str = "operator resume") -> None:
        rec = self.journal.run(run_id)
        if rec.state is not RunState.PAUSED:
            raise JournalError(f"run is {rec.state.value}, not paused")
        self.journal.record_control(run_id, "unpause", payload={"reason": reason})
        self.journal.set_run_state(run_id, RunState.RUNNING, reason)

    def cancel(self, run_id: str, reason: str = "operator cancel") -> tuple[str, ...]:
        self.journal.record_control(run_id, "cancel", payload={"reason": reason})
        cancelled: list[str] = []
        for nid, rec in self.journal.nodes(run_id).items():
            if not rec.terminal:
                self.journal.set_state(run_id, nid, NodeState.CANCELLED, reason,
                                       lease_owner=None, lease_expires_at=None)
                cancelled.append(nid)
        for res in self.journal.open_reservations(run_id):
            self.broker.release(res["reservation_id"])
            self.journal.release_reservation(res["reservation_id"], "cancelled")
        self.journal.set_run_state(run_id, RunState.CANCELLED, reason)
        return tuple(cancelled)

    def approve(self, run_id: str, node_id: str, *, by: str = "operator", note: str = "") -> NodeRecord:
        rec = self.journal.node(run_id, node_id)
        if rec.state is not NodeState.AWAITING_APPROVAL:
            raise JournalError(f"{node_id} is {rec.state.value}, not awaiting approval")
        self.journal.record_control(run_id, "approve", node_id, {"by": by, "note": note})
        out = self.journal.set_state(run_id, node_id, NodeState.PENDING, f"approved by {by}", approved=1)
        run = self.journal.run(run_id)
        if run.state is RunState.AWAITING_APPROVAL and not self._awaiting(run_id):
            self.journal.set_run_state(run_id, RunState.RUNNING, "approvals satisfied")
        return out

    def reject(self, run_id: str, node_id: str, *, by: str = "operator", note: str = "") -> NodeRecord:
        rec = self.journal.node(run_id, node_id)
        if rec.state is not NodeState.AWAITING_APPROVAL:
            raise JournalError(f"{node_id} is {rec.state.value}, not awaiting approval")
        self.journal.record_control(run_id, "reject", node_id, {"by": by, "note": note})
        out = self.journal.set_state(run_id, node_id, NodeState.REJECTED, f"rejected by {by}: {note}")
        self.journal.set_run_state(run_id, RunState.FAILED, f"{node_id} rejected")
        return out

    def retry_node(self, run_id: str, node_id: str, *, reason: str = "operator retry",
                   cascade: bool = True) -> tuple[str, ...]:
        """Reset ``node_id`` (and, by default, every descendant) to PENDING.
        Accepted ancestors are untouched — this is the local-repair primitive."""
        graph = self.journal.graph(run_id)
        if graph.node(node_id) is None:
            raise JournalError(f"unknown node {node_id}")
        targets = [node_id] + (sorted(graph.descendants(node_id), key=graph.node_ids.index)
                               if cascade else [])
        return self.retry_nodes(run_id, targets, reason=reason, force=(node_id,),
                                control="retry_node", anchor=node_id)

    def retry_nodes(self, run_id: str, node_ids: Iterable[str], *, reason: str,
                    force: Iterable[str] = (), control: str = "retry_nodes",
                    anchor: str | None = None) -> tuple[str, ...]:
        """Reset exactly ``node_ids`` to PENDING. Nodes in ``force`` bypass
        the content cache on their next attempt (they must really re-run);
        the others may legitimately be served from cache if their inputs turn
        out identical. This is the smallest-subgraph primitive k112 drives."""
        graph = self.journal.graph(run_id)
        targets = [n for n in graph.topological_order() if n in set(node_ids)]
        missing = set(node_ids) - set(targets)
        if missing:
            raise JournalError(f"unknown node(s) {sorted(missing)}")
        forced = set(force)
        self.journal.record_control(run_id, control, anchor,
                                    {"reason": reason, "targets": targets, "force": sorted(forced)})
        reset: list[str] = []
        for nid in targets:
            rec = self.journal.node(run_id, nid)
            if rec.state in (NodeState.LEASED, NodeState.RUNNING):
                raise JournalError(f"{nid} is {rec.state.value}; cannot reset a live node")
            self.journal.set_state(
                run_id, nid, NodeState.PENDING, reason, lease_owner=None, lease_expires_at=None,
                started_at=None, ended_at=None, outputs_json=None, receipt_json=None,
                failure=None, failure_class=None, repair_code=None, cached=0,
                cache_key=None if nid in forced else rec.cache_key,
                force_rerun=1 if nid in forced else 0,
            )
            reset.append(nid)
        run = self.journal.run(run_id)
        if run.state in (RunState.FAILED, RunState.COMPLETED):
            self.journal.set_run_state(run_id, RunState.RUNNING, f"retry {', '.join(targets)}")
        return tuple(reset)

    def replan(self, run_id: str, new_graph: PlanGraph, reason: str) -> tuple[str, ...]:
        new_graph.topological_order()
        self.journal.record_control(run_id, "replan", payload={
            "reason": reason, "revision": new_graph.revision,
            "structure_digest": new_graph.structure_digest()})
        return self.journal.replan(run_id, new_graph, reason)

    # -- stepping ---------------------------------------------------------- #

    def _awaiting(self, run_id: str) -> tuple[str, ...]:
        return tuple(n for n, r in self.journal.nodes(run_id).items()
                     if r.state is NodeState.AWAITING_APPROVAL)

    def ready_nodes(self, run_id: str) -> tuple[str, ...]:
        graph = self.journal.graph(run_id)
        recs = self.journal.nodes(run_id)
        preds = graph.predecessors()
        out: list[str] = []
        for nid in graph.topological_order():
            rec = recs.get(nid)
            if rec is None or rec.state is not NodeState.PENDING:
                continue
            if all(recs[p].state is NodeState.SUCCEEDED for p in preds.get(nid, ())):
                out.append(nid)
        return tuple(out)

    def gather_inputs(self, run_id: str, node_id: str, graph: PlanGraph | None = None,
                      recs: Mapping[str, NodeRecord] | None = None) -> dict[str, Any]:
        graph = graph or self.journal.graph(run_id)
        recs = recs or self.journal.nodes(run_id)
        node = graph.node(node_id)
        if node is None:
            raise JournalError(f"unknown node {node_id}")
        inputs: dict[str, Any] = {}
        many = {p.name for p in node.inputs if p.many}
        for p in node.inputs:
            if p.many:
                inputs[p.name] = []
        for edge in graph.incoming(node_id):
            src = recs[edge.src_node]
            if src.state is not NodeState.SUCCEEDED or src.outputs is None:
                raise JournalError(f"{node_id}: input {edge.src_node} not succeeded")
            value = src.outputs.get(edge.src_port)
            if edge.dst_port in many:
                inputs[edge.dst_port].append(value)
            else:
                inputs[edge.dst_port] = value
        for p in node.inputs:
            if p.required and p.name not in inputs and p.name not in many:
                raise JournalError(f"{node_id}: required input port '{p.name}' unbound")
        return inputs

    def step(self, run_id: str) -> StepReport:
        run = self.journal.run(run_id)
        if run.state is not RunState.RUNNING:
            return StepReport(run_id, run.state, ready_remaining=len(self.ready_nodes(run_id)))
        graph = self.journal.graph(run_id)
        ready = self.ready_nodes(run_id)
        executed: list[str] = []
        cached: list[str] = []
        succeeded: list[str] = []
        failed: list[str] = []
        awaiting: list[str] = []
        blocked: list[str] = []

        # Phase 1 (calling thread, in ready order): gate / gather / cache /
        # grant / lease / reserve / mark_running — every journal write of the
        # record-before-run prefix happens HERE, before any executor runs, in
        # exactly the order the sequential stepper wrote it.
        launches: list[tuple[PlanNode, dict[str, Any], NodeContext, str]] = []
        for nid in ready[: self.max_parallel]:
            node = graph.node(nid)
            assert node is not None
            rec = self.journal.node(run_id, nid)
            if node.approval_gate and not rec.approved:
                self.journal.await_approval(run_id, nid)
                awaiting.append(nid)
                continue
            recs = self.journal.nodes(run_id)
            try:
                inputs = self.gather_inputs(run_id, nid, graph, recs)
            except JournalError as exc:
                self.journal.lease(run_id, nid, owner=self.owner, lease_s=self.lease_s,
                                   idempotency_key=self._idem(run, node, rec), cache_key="")
                self.journal.fail(run_id, nid, str(exc), failure_class="plan_error")
                failed.append(nid)
                continue
            cache_key = derive_cache_key(node, inputs)
            idem = self._idem(run, node, rec)

            hit = None if rec.force_rerun else (self.journal.cache_lookup(cache_key) if cache_key else None)
            if hit is not None:
                self.journal.lease(run_id, nid, owner=self.owner, lease_s=self.lease_s,
                                   idempotency_key=idem, cache_key=cache_key)
                outputs, receipt = hit
                receipt = dict(receipt or {})
                receipt["cached"] = True
                self.journal.succeed(run_id, nid, outputs, receipt, cached=True)
                cached.append(nid)
                succeeded.append(nid)
                continue

            if not self.broker.can_grant(node):
                blocked.append(nid)
                continue

            # ---- record-before-run ----
            leased = self.journal.lease(run_id, nid, owner=self.owner, lease_s=self.lease_s,
                                        idempotency_key=idem, cache_key=cache_key)
            res_id = f"res_{uuid.uuid4().hex[:12]}"
            grant = self.broker.grant(res_id, node)
            self.journal.record_reservation(res_id, run_id, nid, leased.attempt,
                                            _resources_dict(node), grant)
            self.journal.mark_running(run_id, nid)
            ctx = NodeContext(run_id=run_id, revision=run.revision, node_id=nid,
                              attempt=leased.attempt, idempotency_key=idem, cache_key=cache_key,
                              candidates=node.candidates, lease_owner=self.owner,
                              exclude_models=_failed_models(rec))
            executed.append(nid)
            launches.append((node, inputs, ctx, res_id))

        # Phase 2: run. max_parallel=1 executes inline on the calling thread
        # (the original behavior, unchanged); otherwise the sliced ready set
        # runs on a bounded pool. A node failure fails only its node: the
        # journal write is per node and ``_run_one`` never raises.
        if self.max_parallel == 1:
            outcomes = []
            for node, inputs, ctx, res_id in launches:
                ok = self._execute(run_id, graph, node, inputs, ctx)
                self.broker.release(res_id)
                self.journal.release_reservation(res_id, "succeeded" if ok else "failed")
                outcomes.append(ok)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(launches)),
                                    thread_name_prefix="dag-node") as pool:
                futs = [pool.submit(self._run_one, run_id, graph, *l) for l in launches]
                outcomes = [f.result() for f in futs]
        for (node, _inputs, _ctx, _res), ok in zip(launches, outcomes):
            (succeeded if ok else failed).append(node.node_id)

        self._settle(run_id, graph)
        run = self.journal.run(run_id)
        return StepReport(
            run_id=run_id, run_state=run.state, executed=tuple(executed), cached=tuple(cached),
            succeeded=tuple(succeeded), failed=tuple(failed), awaiting_approval=tuple(awaiting),
            blocked_on_resources=tuple(blocked), ready_remaining=len(self.ready_nodes(run_id)),
        )

    def run(self, run_id: str, *, max_steps: int = 10_000) -> RunRecord:
        """Step until the run leaves RUNNING or nothing is ready."""
        for _ in range(max_steps):
            report = self.step(run_id)
            if report.run_state is not RunState.RUNNING:
                break
            if report.idle and report.ready_remaining == 0:
                break
            if report.idle and report.blocked_on_resources and not report.executed:
                break  # nothing can proceed; caller waits on capacity
        return self.journal.run(run_id)

    # -- internals --------------------------------------------------------- #

    def _run_one(self, run_id: str, graph: PlanGraph, node: PlanNode, inputs: dict[str, Any],
                 ctx: NodeContext, res_id: str) -> bool:
        """Execute ONE already-leased, already-running node and release its
        reservation. Never raises: an unexpected error (including a journal
        error from the write-back) fails this node only."""
        try:
            ok = self._execute(run_id, graph, node, inputs, ctx)
        except Exception as exc:  # noqa: BLE001 — one node's failure is one node's failure
            ok = False
            try:
                self.journal.fail(run_id, node.node_id, f"{type(exc).__name__}: {exc}",
                                  failure_class="runtime_error")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.broker.release(res_id)
        try:
            self.journal.release_reservation(res_id, "succeeded" if ok else "failed")
        except Exception:  # noqa: BLE001
            pass
        return ok

    @staticmethod
    def _idem(run: RunRecord, node: PlanNode, rec: NodeRecord) -> str:
        if node.idempotency_key:
            return node.idempotency_key
        return content_digest({"run": run.run_id, "revision": run.revision,
                               "node": node.node_id, "attempt": rec.attempt + 1})

    def _select(self, node: PlanNode, ctx: NodeContext, inputs: dict[str, Any]) -> NodeContext:
        """Ask the selector which model THIS call should use; a gap is a
        typed failure here, not a silent default."""
        if self.selector is None or not node.capability:
            return ctx
        if _param(node, "model_free"):
            # deterministic capability (ffmpeg concat, manifest validation, ...):
            # nothing to select; the receipt says so
            return NodeContext(**{**ctx.to_dict(), "selection": {"rationale": "model_free capability"}})
        decision = self.selector.for_node(node, ctx, inputs)
        if decision is None:
            return ctx
        d = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        if getattr(decision, "gap", False) or not getattr(decision, "model_id", None):
            if node.kind is NodeKind.JUDGE:
                # a judge seam resolves its own model (and returns an honest
                # "unscored" when it cannot); selection is advisory here and
                # the gap rides on the receipt instead of failing the node
                d["advisory"] = "judge node: selection gap recorded, seam resolves its own judge"
                return NodeContext(**{**ctx.to_dict(), "model_id": None, "selection": d})
            raise SelectionGap(node.capability, d)
        return NodeContext(**{**ctx.to_dict(), "model_id": decision.model_id, "selection": d})

    def _call(self, node: PlanNode, inputs: dict[str, Any], ctx: NodeContext) -> NodeResult:
        ctx = self._select(node, ctx, inputs)
        t0 = time.monotonic()
        try:
            r = self._coerce(self.executor(node, inputs, ctx))
        except Exception as exc:
            self._outcome(node, ctx, ok=False, hard_pass=None, repair_code=None,
                          latency_s=time.monotonic() - t0)
            try:
                exc.oracle_ctx = ctx  # the selected model rides with the failure
            except Exception:  # noqa: BLE001
                pass
            raise
        latency = time.monotonic() - t0
        receipt = dict(r.receipt or {})
        receipt.setdefault("model_id", ctx.model_id)
        receipt.setdefault("selection", ctx.selection)
        receipt.setdefault("latency_s", round(latency, 4))
        receipt.setdefault("candidate", ctx.candidate)
        return NodeResult(outputs=r.outputs, receipt=receipt, warnings=r.warnings)

    def _outcome(self, node: PlanNode, ctx: NodeContext, *, ok: bool, hard_pass: bool | None,
                 repair_code: Any, latency_s: float | None) -> None:
        if self.selector is None or not getattr(ctx, "model_id", None):
            return
        try:
            self.selector.record_outcome(node, ctx, ok=ok, hard_pass=hard_pass,
                                         repair_code=repair_code, latency_s=latency_s)
        except Exception:  # noqa: BLE001 — evidence recording must never break execution
            pass

    def _invoke(self, node: PlanNode, inputs: dict[str, Any], ctx: NodeContext) -> NodeResult:
        if node.capability is None and node.kind in (NodeKind.JOIN, NodeKind.GATE):
            return NodeResult(outputs=dict(inputs))
        if node.kind is NodeKind.FANOUT and node.candidates > 1:
            per_port: dict[str, list[Any]] = {}
            receipts: list[Any] = []
            warnings: list[str] = []
            for i in range(node.candidates):
                cctx = NodeContext(**{**ctx.to_dict(), "candidate": i})
                r = self._call(node, inputs, cctx)
                for k, v in r.outputs.items():
                    per_port.setdefault(k, []).append(v)
                if r.receipt is not None:
                    receipts.append(dict(r.receipt))
                warnings.extend(r.warnings)
            models = [rc.get("model_id") for rc in receipts]
            return NodeResult(outputs=per_port,
                              receipt={"candidates": receipts, "models": models,
                                       "model_id": models[0] if models else None},
                              warnings=tuple(warnings))
        return self._call(node, inputs, ctx)

    @staticmethod
    def _coerce(value: Any) -> NodeResult:
        if isinstance(value, NodeResult):
            return value
        if isinstance(value, Mapping):
            return NodeResult(outputs=dict(value))
        raise TypeError(f"executor must return a Mapping or NodeResult, got {type(value).__name__}")

    def _execute(self, run_id: str, graph: PlanGraph, node: PlanNode,
                 inputs: dict[str, Any], ctx: NodeContext) -> bool:
        try:
            result = self._invoke(node, inputs, ctx)
        except SelectionGap as exc:
            # no defensible model: typed gap, never a silent default, never a blind retry
            self.journal.fail(run_id, node.node_id, str(exc), failure_class="capability_gap",
                              retry=False, repair_code=RepairCode.CAPABILITY_GAP,
                              receipt={"selection": exc.decision, "capability": exc.capability})
            return False
        except Exception as exc:  # noqa: BLE001 - classified, journaled, never swallowed silently
            if _is_gap(exc):
                # the seam/adapter is not there: typed gap, no retry (retrying
                # an absent backend is the prayer this runtime refuses)
                self.journal.fail(run_id, node.node_id, f"{type(exc).__name__}: {exc}",
                                  failure_class="capability_gap", retry=False,
                                  repair_code=RepairCode.CAPABILITY_GAP)
                return False
            fclass = _classify(exc)
            retry = ctx.attempt < node.retry.max_attempts
            fctx = getattr(exc, "oracle_ctx", None)
            receipt = None
            if fctx is not None and getattr(fctx, "model_id", None):
                receipt = {"model_id": fctx.model_id, "selection": fctx.selection,
                           "candidate": fctx.candidate, "attempt": ctx.attempt}
            self.journal.fail(run_id, node.node_id, f"{type(exc).__name__}: {exc}",
                              failure_class=fclass, retry=retry,
                              repair_code=_repair_code_for(fclass), receipt=receipt)
            return False
        receipt = dict(result.receipt or {})
        receipt.setdefault("capability", node.capability)
        receipt.setdefault("attempt", ctx.attempt)
        receipt.setdefault("idempotency_key", ctx.idempotency_key)
        receipt.setdefault("cache_key", ctx.cache_key)
        receipt.setdefault("lease_owner", ctx.lease_owner)
        if result.warnings:
            receipt["warnings"] = list(result.warnings)
        card: Scorecard | None = None
        if self.evaluator is not None and (node.acceptance or node.kind is NodeKind.JUDGE
                                           or node.kind in (NodeKind.TASK, NodeKind.MAP, NodeKind.FANOUT)):
            try:
                card = self.evaluator(node, result.outputs, ctx)
            except Exception as exc:  # noqa: BLE001
                self.journal.fail(run_id, node.node_id, f"evaluator raised: {exc}",
                                  failure_class="evaluator_error", receipt=receipt)
                return False
        if card is not None:
            receipt["scorecard"] = card.to_dict() if hasattr(card, "to_dict") else {"hard_pass": card.hard_pass}
            self._record_judged(node, ctx, receipt, card)
            if not card.hard_pass:
                retry = ctx.attempt < node.retry.max_attempts
                self.journal.fail(run_id, node.node_id,
                                  card.diagnosis or "acceptance failed",
                                  failure_class="acceptance", repair_code=card.repair_code,
                                  receipt=receipt, retry=retry)
                return False
        if card is None:
            self._record_judged(node, ctx, receipt, None)
        self.journal.succeed(run_id, node.node_id, result.outputs, receipt,
                             capability=node.capability)
        return True

    def _record_judged(self, node: PlanNode, ctx: NodeContext, receipt: Mapping[str, Any],
                       card: "Scorecard | None") -> None:
        """One ledger row per candidate call: ok=True (it executed), plus the
        judge's verdict when there is one."""
        hard_pass = None if card is None else bool(card.hard_pass)
        code = None if card is None else card.repair_code
        if node.kind is NodeKind.JUDGE:
            # a judge's verdict is evidence about the PRODUCER (written back via
            # producer attribution), not about the judge model's own reliability
            hard_pass, code = None, None
        cands = receipt.get("candidates")
        if isinstance(cands, list) and cands:
            for rc in cands:
                cctx = NodeContext(**{**ctx.to_dict(), "candidate": rc.get("candidate", 0),
                                      "model_id": rc.get("model_id")})
                self._outcome(node, cctx, ok=True, hard_pass=hard_pass, repair_code=code,
                              latency_s=rc.get("latency_s"))
            return
        cctx = NodeContext(**{**ctx.to_dict(), "model_id": receipt.get("model_id", ctx.model_id)})
        self._outcome(node, cctx, ok=True, hard_pass=hard_pass, repair_code=code,
                      latency_s=receipt.get("latency_s"))

    def _settle(self, run_id: str, graph: PlanGraph) -> None:
        recs = self.journal.nodes(run_id)
        run = self.journal.run(run_id)
        if run.state is not RunState.RUNNING:
            return
        states = {r.state for r in recs.values()}
        if all(r.state is NodeState.SUCCEEDED for r in recs.values()):
            self.journal.set_run_state(run_id, RunState.COMPLETED, "all nodes succeeded")
            return
        # A terminal failure does not stop independent siblings: the run only
        # becomes FAILED once nothing else can make progress, so every sibling
        # take that CAN be produced and judged is, and the repair controller
        # sees the whole picture (directive §7 stage 14: sequential or parallel
        # execution must not change the dependency structure or its outcome).
        if self.ready_nodes(run_id):
            return
        if NodeState.FAILED in states or NodeState.REJECTED in states:
            self.journal.set_run_state(run_id, RunState.FAILED, "a node failed terminally")
        elif NodeState.AWAITING_APPROVAL in states:
            self.journal.set_run_state(run_id, RunState.AWAITING_APPROVAL, "operator gate")


def _param(node: PlanNode, key: str, default: Any = None) -> Any:
    try:
        return node.params.get(key, default)
    except Exception:  # noqa: BLE001
        return default


def _failed_models(rec: NodeRecord) -> tuple[str, ...]:
    """Models named on the previous (failed) attempt's receipt, so the next
    attempt is asked of a different eligible model when one exists."""
    if rec.failure is None or not rec.receipt:
        return ()
    out: list[str] = []
    m = rec.receipt.get("model_id")
    if m:
        out.append(str(m))
    for rc in rec.receipt.get("candidates", []) or []:
        if isinstance(rc, Mapping) and rc.get("model_id"):
            out.append(str(rc["model_id"]))
    return tuple(dict.fromkeys(out))


def _resources_dict(node: PlanNode) -> dict[str, Any]:
    r = node.resources
    if hasattr(r, "to_dict"):
        return dict(r.to_dict())
    return {"vram_gib": r.vram_gib, "ram_gib": r.ram_gib, "gpu": r.gpu, "est_seconds": r.est_seconds}


def _is_gap(exc: BaseException) -> bool:
    """Seam-unavailable / capability-gap style exceptions, recognised by
    name so this module stays import-light."""
    name = type(exc).__name__
    return name in ("SeamUnavailable", "CapabilityGap", "SelectionGap") or bool(getattr(exc, "oracle_gap", False))


def _classify(exc: BaseException) -> str:
    try:
        from hugpy_oracle.runtime import classify_failure
        fc = classify_failure(exc)
        return getattr(fc, "value", str(fc))
    except Exception:  # noqa: BLE001
        return type(exc).__name__


def _repair_code_for(failure_class: str) -> RepairCode | None:
    fc = failure_class.lower()
    if "selectiongap" in fc or "capability_gap" in fc:
        return RepairCode.CAPABILITY_GAP
    if "timeout" in fc:
        return RepairCode.TIMEOUT
    if "worker" in fc or "unavailable" in fc:
        return RepairCode.WORKER_UNAVAILABLE
    if "decode" in fc:
        return RepairCode.DECODE_FAILED
    return None
