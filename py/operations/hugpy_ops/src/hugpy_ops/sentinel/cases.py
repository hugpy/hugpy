"""Case store: one OPEN case per fingerprint, SQLite-backed.

Dedupe is structural, not conventional: a partial UNIQUE index on
fingerprint over non-closed rows makes "re-detection touches last_seen,
never re-spawns" a database guarantee — a second opener loses the INSERT
race and lands in the touch path.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

# Lifecycle: open -> agent_running -> documented | escalated
#            documented -> remedied (only ever by an operator/enabled remedy)
#            anything -> closed (operator).
STATES = ("open", "agent_running", "documented", "remedied", "escalated",
          "closed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    evidence    TEXT NOT NULL,           -- JSON dict, the anomaly evidence
    state       TEXT NOT NULL DEFAULT 'open',
    opened_at   REAL NOT NULL,
    last_seen   REAL NOT NULL,
    agent_run_id TEXT,
    report_path  TEXT,
    note         TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_cases_open_fingerprint
    ON cases(fingerprint) WHERE state != 'closed';
"""


@dataclass
class Anomaly:
    """One bound-exceeded observation from checks.py.

    fingerprint identifies the CONDITION (stable across re-detections);
    evidence carries the observation that proved it this pass.
    """
    fingerprint: str
    kind: str
    severity: str                    # "warn" | "critical"
    evidence: dict = field(default_factory=dict)


@dataclass
class Case:
    id: int
    fingerprint: str
    kind: str
    severity: str
    evidence: dict
    state: str
    opened_at: float
    last_seen: float
    agent_run_id: str | None = None
    report_path: str | None = None
    note: str | None = None


def _row_to_case(row) -> Case:
    return Case(id=row["id"], fingerprint=row["fingerprint"],
                kind=row["kind"], severity=row["severity"],
                evidence=json.loads(row["evidence"] or "{}"),
                state=row["state"], opened_at=row["opened_at"],
                last_seen=row["last_seen"],
                agent_run_id=row["agent_run_id"],
                report_path=row["report_path"], note=row["note"])


class CaseStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.realpath(db_path)) or ".",
                    exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def open_or_touch(self, anomaly: Anomaly,
                      now: float | None = None) -> tuple[Case, bool]:
        """Open a case for the anomaly, or touch the existing open one.

        Returns (case, created). created=False means an open case for this
        fingerprint already exists — the caller must NOT spawn an agent.
        """
        now = time.time() if now is None else now
        try:
            cur = self._conn.execute(
                "INSERT INTO cases (fingerprint, kind, severity, evidence,"
                " state, opened_at, last_seen) VALUES (?,?,?,?,?,?,?)",
                (anomaly.fingerprint, anomaly.kind, anomaly.severity,
                 json.dumps(anomaly.evidence, sort_keys=True),
                 "open", now, now))
            self._conn.commit()
            return self.get(cur.lastrowid), True
        except sqlite3.IntegrityError:
            # The partial unique index fired: a non-closed case exists.
            self._conn.execute(
                "UPDATE cases SET last_seen = ? WHERE fingerprint = ?"
                " AND state != 'closed'", (now, anomaly.fingerprint))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM cases WHERE fingerprint = ?"
                " AND state != 'closed'",
                (anomaly.fingerprint,)).fetchone()
            return _row_to_case(row), False

    def get(self, case_id: int) -> Case:
        row = self._conn.execute("SELECT * FROM cases WHERE id = ?",
                                 (case_id,)).fetchone()
        if row is None:
            raise KeyError("no case %r" % case_id)
        return _row_to_case(row)

    def set_state(self, case_id: int, state: str,
                  note: str | None = None) -> None:
        if state not in STATES:
            raise ValueError("unknown case state %r" % state)
        if note is None:
            self._conn.execute("UPDATE cases SET state = ? WHERE id = ?",
                               (state, case_id))
        else:
            self._conn.execute(
                "UPDATE cases SET state = ?, note = ? WHERE id = ?",
                (state, note, case_id))
        self._conn.commit()

    def attach_agent(self, case_id: int, run_id: str | None,
                     report_path: str | None) -> None:
        self._conn.execute(
            "UPDATE cases SET agent_run_id = ?, report_path = ?"
            " WHERE id = ?", (run_id, report_path, case_id))
        self._conn.commit()

    def list(self, state: str | None = None) -> list[Case]:
        if state is None:
            rows = self._conn.execute(
                "SELECT * FROM cases ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM cases WHERE state = ? ORDER BY id",
                (state,)).fetchall()
        return [_row_to_case(r) for r in rows]
