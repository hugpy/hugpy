"""HELP TICKETS (operator 2026-10-02): things the system found that the
operator may want acted on — today, calibration runs whose gate prediction
disagreed with the measurement or whose unload leaked. They wait in the help
panel as PRE-EXISTING APPROVALS: nothing happens until the operator picks an
action (run calibrate again, send to the keeper — when one is reachable —,
discuss in a help session, or dismiss). The Help button shows the pending count.

One pending ticket per (kind, model, worker): a repeat finding updates it and
bumps ``occurrences`` instead of stacking duplicates. Table ``help_tickets`` in
the hugpy DB, fail-open (small_pg)."""
from __future__ import annotations

import json
import time
from typing import Optional

DDL = """
CREATE TABLE IF NOT EXISTS help_tickets (
    id           BIGSERIAL PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    created      DOUBLE PRECISION NOT NULL,
    updated      DOUBLE PRECISION NOT NULL,
    model_key    TEXT,
    worker_id    TEXT,
    worker_name  TEXT,
    title        TEXT NOT NULL,
    detail       JSONB NOT NULL DEFAULT '{}'::jsonb,
    result       JSONB,
    occurrences  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS help_tickets_status ON help_tickets (status, updated DESC);
"""
STATUSES = ("pending", "acted", "sent", "dismissed")
COLS = ("id, kind, status, created, updated, model_key, worker_id, worker_name, title,"
        " detail, result, occurrences")

_RUNNER = None


def _run(fn):
    global _RUNNER
    if _RUNNER is None:
        from hugpy_server.app.small_pg import Runner
        _RUNNER = Runner("help tickets", DDL)
    return _RUNNER.run(fn)


def _rows(cur) -> list:
    from hugpy_server.app.small_pg import rows
    return rows(cur)


def file_ticket(kind: str, title: str, *, model_key: Optional[str] = None,
                worker_id: Optional[str] = None, worker_name: Optional[str] = None,
                detail: Optional[dict] = None) -> Optional[dict]:
    """Create — or refresh the pending one for the same (kind, model, worker)."""
    now = time.time()
    det = json.dumps(detail or {}, default=str)

    def op(cur):
        cur.execute(f"SELECT {COLS} FROM help_tickets WHERE kind=%s AND status='pending'"
                    " AND model_key IS NOT DISTINCT FROM %s AND worker_id IS NOT DISTINCT FROM %s"
                    " ORDER BY id DESC LIMIT 1", (kind, model_key, worker_id))
        have = _rows(cur)
        if have:
            cur.execute(f"UPDATE help_tickets SET updated=%s, title=%s, detail=%s::jsonb,"
                        f" occurrences=occurrences+1 WHERE id=%s RETURNING {COLS}",
                        (now, title, det, have[0]["id"]))
        else:
            cur.execute(f"INSERT INTO help_tickets (kind, created, updated, model_key, worker_id,"
                        f" worker_name, title, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)"
                        f" RETURNING {COLS}", (kind, now, now, model_key, worker_id, worker_name, title, det))
        return _rows(cur)[0]
    return _run(op)


def list_tickets(status: Optional[str] = None, limit: int = 100) -> list:
    def op(cur):
        if status:
            cur.execute(f"SELECT {COLS} FROM help_tickets WHERE status=%s ORDER BY updated DESC LIMIT %s",
                        (status, int(limit)))
        else:
            cur.execute(f"SELECT {COLS} FROM help_tickets ORDER BY updated DESC LIMIT %s", (int(limit),))
        return _rows(cur)
    return _run(op) or []


def get_ticket(tid: int) -> Optional[dict]:
    def op(cur):
        cur.execute(f"SELECT {COLS} FROM help_tickets WHERE id=%s", (int(tid),))
        r = _rows(cur)
        return r[0] if r else None
    return _run(op)


def pending_count() -> int:
    def op(cur):
        cur.execute("SELECT count(*) FROM help_tickets WHERE status='pending'")
        return int(cur.fetchone()[0])
    return _run(op) or 0


def set_status(tid: int, status: str, result: Optional[dict] = None) -> Optional[dict]:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")

    def op(cur):
        cur.execute(f"UPDATE help_tickets SET status=%s, result=%s::jsonb, updated=%s WHERE id=%s"
                    f" RETURNING {COLS}", (status, json.dumps(result or {}, default=str), time.time(), int(tid)))
        r = _rows(cur)
        return r[0] if r else None
    return _run(op)


def calibration_ticket(row: dict) -> Optional[dict]:
    """File the ticket for a finished calibration whose verdict is not 'agree'."""
    verdict = row.get("verdict")
    if verdict in (None, "agree"):
        return None
    det = row.get("detail") or {}
    err = det.get("gate_error_pct")
    title = (f"calibration {verdict}: {row.get('model_key')} on {row.get('worker_name') or row.get('worker_id')}"
             + (f" (gate {err:+.1f}% vs measured)" if isinstance(err, (int, float)) else ""))
    return file_ticket("calibration", title, model_key=row.get("model_key"),
                       worker_id=row.get("worker_id"), worker_name=row.get("worker_name"),
                       detail={"verdict": verdict, "bnb": bool(row.get("bnb")), "file": row.get("file"),
                               "gpu": row.get("gpu"), "ctx": row.get("ctx"),
                               "predicted": row.get("predicted"), "measured": row.get("measured"),
                               "calibration": det})
