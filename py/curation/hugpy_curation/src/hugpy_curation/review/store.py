"""review/store.py — persisted reviews.

The timer must not re-screen the same repo every night, and the operator wants
to look up what a model scored last time without re-running anything. SQLite,
one row per (criteria, hub_id, stage), newest wins.

WHERE THE DB LIVES, AND WHY THERE ARE TWO
    The pipeline runs on a WORKER box (ae holds the 3090), so its rows land in
    that box's local file — ``REVIEW_DB``, e.g. /var/lib/hugpy-worker/review/
    reviews.db. Central's /llm/review/* routes read CENTRAL's file under
    DEFAULT_ROOT. Without a push those two never meet and the console shows an
    empty leaderboard while the worker quietly reviews all night.

    So: the worker's file is the ON-BOX RECORD (it is what survives a network
    outage and what a `--force` re-screen consults), and CENTRAL'S FILE IS THE
    SOURCE OF TRUTH the operator reads. ``review/push.py`` moves rows one way,
    worker -> central, and ``ingest_run``/``ingest_results`` below are the
    central-side landing zone. Nothing ever flows back down.

INGEST COLUMNS
    ``source_host`` — which box produced the row. NULL means "this box", i.e. a
    locally-produced row; every pushed row carries a host. That NULL is load
    bearing: SQLite treats NULLs as distinct in a UNIQUE index, so local rows
    are never constrained by the ingest key, while pushed rows are.
    ``runs.remote_run_id`` / ``reviews.run_id`` — the run's id ON ITS SOURCE
    HOST. Central keeps its own autoincrement ``runs.id``; the natural key that
    makes a retried push idempotent is (source_host, remote_run_id) for a run
    and (source_host, run_id, hub_id, stage) for a result.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    criteria      TEXT NOT NULL,
    hub_id        TEXT NOT NULL,
    stage         TEXT NOT NULL,          -- screened | downloaded | smoked
    passed        INTEGER,
    score         REAL,
    verdict       TEXT,                   -- agent verdict when judged
    payload       TEXT NOT NULL,          -- full review JSON
    reviewed_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reviews_lookup ON reviews(criteria, hub_id);
CREATE INDEX IF NOT EXISTS ix_reviews_time   ON reviews(reviewed_at DESC);
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    criteria      TEXT NOT NULL,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    screened      INTEGER DEFAULT 0,
    passed        INTEGER DEFAULT 0,
    downloaded    INTEGER DEFAULT 0,
    smoked        INTEGER DEFAULT 0,
    error         TEXT
);
"""

# Columns added after the first schema shipped. ALTER TABLE ADD COLUMN, not a
# bumped CREATE TABLE: existing boxes already have rows in these tables and a
# nightly reviewer must never need a hand migration to keep running.
_ADDED_COLUMNS = {
    "reviews": {"source_host": "TEXT", "run_id": "INTEGER"},
    "runs": {"source_host": "TEXT", "remote_run_id": "INTEGER",
             "pushed_at": "REAL"},
}

# Idempotency backstops for ingest. Both are partial-by-NULL: a locally produced
# row has source_host NULL, and SQLite treats NULLs as distinct, so these
# constrain pushed rows only.
_INGEST_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_source
    ON runs(source_host, remote_run_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_reviews_source
    ON reviews(source_host, run_id, hub_id, stage);
"""


def db_path() -> str:
    """``REVIEW_DB``, else ``<review_root>/reviews.db`` — see
    :mod:`hugpy_curation.config` for the whole resolution order."""
    from hugpy_curation.config import review_db_path
    p = review_db_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _migrate(c: sqlite3.Connection) -> None:
    """Add the ingest columns to a DB created by an older release.

    Cheap enough to run on every connect (two PRAGMAs) and it means the FIRST
    push against a long-lived central DB just works instead of erroring on an
    unknown column."""
    for table, columns in _ADDED_COLUMNS.items():
        have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    c.executescript(_INGEST_INDEXES)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(db_path(), timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    _migrate(c)
    return c


def record(criteria: str, hub_id: str, stage: str, payload: dict,
           passed: bool | None = None, score: float | None = None,
           verdict: str | None = None, run_id: int | None = None) -> int:
    """Persist one review row.

    ``run_id`` ties the row to the run that produced it so ``push`` can ship a
    run and exactly its results as one batch. Optional and NULL-tolerant: an
    ad-hoc `review` from the CLI belongs to no run, and rows written by older
    releases have none."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO reviews (criteria, hub_id, stage, passed, score, "
            "verdict, payload, reviewed_at, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (criteria, hub_id, stage,
             None if passed is None else int(passed), score, verdict,
             json.dumps(payload, default=str), time.time(), run_id))
        return cur.lastrowid


def latest(criteria: str, hub_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM reviews WHERE criteria=? AND hub_id=? "
            "ORDER BY reviewed_at DESC LIMIT 1", (criteria, hub_id)).fetchone()
    return _row(row)


def seen_since(criteria: str, hub_id: str, max_age_seconds: float) -> bool:
    """True when this repo was already reviewed recently enough to skip."""
    row = latest(criteria, hub_id)
    return bool(row and (time.time() - row["reviewed_at"]) < max_age_seconds)


def recent(criteria: str | None = None, limit: int = 50,
           stage: str | None = None) -> list[dict]:
    sql = "SELECT * FROM reviews WHERE 1=1"
    args: list = []
    if criteria:
        sql += " AND criteria=?"
        args.append(criteria)
    if stage:
        sql += " AND stage=?"
        args.append(stage)
    sql += " ORDER BY reviewed_at DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def leaderboard(criteria: str, limit: int = 20) -> list[dict]:
    """Best-scoring distinct repos for a criteria — the actual answer to
    "what should I run?"."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM reviews r WHERE criteria=? AND passed=1 AND "
            "reviewed_at = (SELECT MAX(reviewed_at) FROM reviews r2 "
            "               WHERE r2.criteria=r.criteria AND r2.hub_id=r.hub_id) "
            "ORDER BY score DESC LIMIT ?", (criteria, int(limit))).fetchall()
    return [_row(r) for r in rows]


def start_run(criteria: str) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO runs (criteria, started_at) VALUES (?,?)",
                         (criteria, time.time())).lastrowid


def finish_run(run_id: int, **counts) -> None:
    fields = {k: v for k, v in counts.items()
              if k in ("screened", "passed", "downloaded", "smoked", "error")}
    sets = ", ".join(f"{k}=?" for k in fields)
    args = list(fields.values()) + [time.time(), run_id]
    with _conn() as c:
        c.execute(f"UPDATE runs SET {sets + ', ' if sets else ''}"
                  f"finished_at=? WHERE id=?", args)


def runs(criteria: str | None = None, limit: int = 20) -> list[dict]:
    sql = "SELECT * FROM runs"
    args: list = []
    if criteria:
        sql += " WHERE criteria=?"
        args.append(criteria)
    sql += " ORDER BY started_at DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def get_run(run_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id=?", (int(run_id),)).fetchone()
    return dict(row) if row is not None else None


def results_for_run(run_id: int) -> list[dict]:
    """Every review row this run produced, oldest first — the push batch."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM reviews WHERE run_id=? "
                         "ORDER BY id ASC", (int(run_id),)).fetchall()
    return [_row(r) for r in rows]


def unpushed_runs(limit: int = 50) -> list[dict]:
    """Finished runs this box has not successfully pushed yet — what
    ``push --all`` replays after an outage."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM runs WHERE source_host IS NULL AND pushed_at IS NULL "
            "ORDER BY started_at ASC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def mark_pushed(run_id: int, when: float | None = None) -> None:
    with _conn() as c:
        c.execute("UPDATE runs SET pushed_at=? WHERE id=?",
                  (when if when is not None else time.time(), int(run_id)))


# ── central-side ingest ────────────────────────────────────────────────────
# SELECT-then-UPDATE/INSERT rather than a bare INSERT OR REPLACE: the natural
# keys contain nullable columns, and `col IS ?` is the only comparison that
# matches a NULL. It also lets ingest report accepted-vs-rejected honestly.

def ingest_run(source_host: str, run: dict) -> int | None:
    """Upsert one pushed run. Returns central's local row id, or None when the
    row is unusable (no remote id / no criteria) — the caller counts it
    rejected. Raises nothing the caller has to translate: bad TYPES are coerced
    here, because a worker on an older release must not be able to 500 central.
    """
    try:
        remote_id = int(run.get("run_id") if run.get("run_id") is not None
                        else run.get("id"))
    except (TypeError, ValueError):
        return None
    criteria = run.get("criteria")
    if not source_host or not isinstance(criteria, str) or not criteria:
        return None
    fields = {
        "criteria": criteria,
        "started_at": _num(run.get("started_at")) or time.time(),
        "finished_at": _num(run.get("finished_at")),
        "screened": _int(run.get("screened")),
        "passed": _int(run.get("passed")),
        "downloaded": _int(run.get("downloaded")),
        "smoked": _int(run.get("smoked")),
        "error": None if run.get("error") is None else str(run.get("error")),
    }
    with _conn() as c:
        row = c.execute("SELECT id FROM runs WHERE source_host IS ? AND "
                        "remote_run_id IS ?", (source_host, remote_id)).fetchone()
        if row is not None:
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE runs SET {sets} WHERE id=?",
                      list(fields.values()) + [row["id"]])
            return int(row["id"])
        cols = list(fields) + ["source_host", "remote_run_id"]
        cur = c.execute(
            f"INSERT INTO runs ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            list(fields.values()) + [source_host, remote_id])
        return int(cur.lastrowid)


def ingest_results(source_host: str, results: list[dict]) -> tuple[int, int]:
    """Upsert pushed review rows. Returns (accepted, rejected).

    Keyed on (source_host, run_id, hub_id, stage) so the same batch delivered
    twice updates in place. A row missing hub_id/stage/criteria is rejected —
    counted, never raised."""
    accepted = rejected = 0
    with _conn() as c:
        for r in results:
            if not isinstance(r, dict):
                rejected += 1
                continue
            hub_id, stage = r.get("hub_id"), r.get("stage")
            criteria = r.get("criteria")
            if not (isinstance(hub_id, str) and hub_id
                    and isinstance(stage, str) and stage
                    and isinstance(criteria, str) and criteria):
                rejected += 1
                continue
            payload = r.get("payload")
            if not isinstance(payload, str):
                payload = json.dumps(payload if payload is not None else {},
                                     default=str)
            fields = {
                "criteria": criteria,
                "passed": _int(r.get("passed"), default=None),
                "score": _num(r.get("score")),
                "verdict": None if r.get("verdict") is None else str(r.get("verdict")),
                "payload": payload,
                "reviewed_at": _num(r.get("reviewed_at")) or time.time(),
            }
            run_id = _int(r.get("run_id"), default=None)
            try:
                row = c.execute(
                    "SELECT id FROM reviews WHERE source_host IS ? AND run_id IS ? "
                    "AND hub_id IS ? AND stage IS ?",
                    (source_host, run_id, hub_id, stage)).fetchone()
                if row is not None:
                    sets = ", ".join(f"{k}=?" for k in fields)
                    c.execute(f"UPDATE reviews SET {sets} WHERE id=?",
                              list(fields.values()) + [row["id"]])
                else:
                    cols = list(fields) + ["hub_id", "stage", "source_host", "run_id"]
                    c.execute(
                        f"INSERT INTO reviews ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})",
                        list(fields.values()) + [hub_id, stage, source_host, run_id])
                accepted += 1
            except sqlite3.Error:
                rejected += 1
    return accepted, rejected


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _int(v, default: int | None = 0) -> int | None:
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _row(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    try:
        d["payload"] = json.loads(d["payload"])
    except (ValueError, TypeError):
        pass
    return d
