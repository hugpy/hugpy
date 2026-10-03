"""MODEL NOTES + FLAGS (operator 2026-10-02): "a notes option for these models
that both user and hugpy-brain can attribute to them … with a few flags such as
broken, trash, archive, unknown". One current annotation per model
(``model_annotations``) plus an append-only history of every change, who made
it and whether an operator or an agent (``model_annotation_log``). Fail-open
(small_pg)."""
from __future__ import annotations

import json
import time
from typing import Optional

FLAGS = ("broken", "trash", "archive", "unknown", "needs-env", "experimental", "keep")
MAX_NOTE = 4000

DDL = """
CREATE TABLE IF NOT EXISTS model_annotations (
    model_id    BIGINT PRIMARY KEY,
    model_key   TEXT,
    flags       TEXT[] NOT NULL DEFAULT '{}',
    note        TEXT NOT NULL DEFAULT '',
    updated_by  TEXT,
    by_kind     TEXT,
    updated_at  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS model_annotation_log (
    id          BIGSERIAL PRIMARY KEY,
    model_id    BIGINT NOT NULL,
    flags       TEXT[] NOT NULL DEFAULT '{}',
    note        TEXT NOT NULL DEFAULT '',
    by          TEXT,
    by_kind     TEXT,
    at          DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS model_annotation_log_model ON model_annotation_log (model_id, at DESC);
"""

_RUNNER = None


def _run(fn):
    global _RUNNER
    if _RUNNER is None:
        from hugpy_server.app.small_pg import Runner
        _RUNNER = Runner("model notes", DDL)
    return _RUNNER.run(fn)


def validate(flags, note) -> "tuple[list, str]":
    """(flags, note) cleaned; raises ValueError on an unknown flag."""
    flags = [str(f).strip().lower() for f in (flags or []) if str(f).strip()]
    bad = [f for f in flags if f not in FLAGS]
    if bad:
        raise ValueError(f"unknown flag(s) {bad}; allowed: {list(FLAGS)}")
    return sorted(set(flags), key=FLAGS.index), str(note or "")[:MAX_NOTE]


def get(model_id: int) -> Optional[dict]:
    def op(cur):
        cur.execute("SELECT model_id, model_key, flags, note, updated_by, by_kind, updated_at"
                    " FROM model_annotations WHERE model_id = %s", (int(model_id),))
        r = cur.fetchone()
        if r is None:
            return {"model_id": int(model_id), "flags": [], "note": "", "history": []}
        out = dict(zip(("model_id", "model_key", "flags", "note", "updated_by", "by_kind", "updated_at"), r))
        cur.execute("SELECT flags, note, by, by_kind, at FROM model_annotation_log WHERE model_id = %s"
                    " ORDER BY at DESC LIMIT 20", (int(model_id),))
        out["history"] = [dict(zip(("flags", "note", "by", "by_kind", "at"), h)) for h in cur.fetchall()]
        return out
    return _run(op)


def put(model_id: int, model_key: str, flags, note, *, by: str, by_kind: str = "operator") -> Optional[dict]:
    flags, note = validate(flags, note)
    now = time.time()

    def op(cur):
        cur.execute("INSERT INTO model_annotations (model_id, model_key, flags, note, updated_by, by_kind, updated_at)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (model_id) DO UPDATE SET"
                    " model_key=EXCLUDED.model_key, flags=EXCLUDED.flags, note=EXCLUDED.note,"
                    " updated_by=EXCLUDED.updated_by, by_kind=EXCLUDED.by_kind, updated_at=EXCLUDED.updated_at",
                    (int(model_id), model_key, flags, note, by, by_kind, now))
        cur.execute("INSERT INTO model_annotation_log (model_id, flags, note, by, by_kind, at)"
                    " VALUES (%s,%s,%s,%s,%s,%s)", (int(model_id), flags, note, by, by_kind, now))
        return True
    if not _run(op):
        return None
    return get(model_id)


def all_flags() -> dict:
    """{model_key: {"flags": [...], "note": str}} for every annotated model (UI)."""
    def op(cur):
        cur.execute("SELECT model_key, flags, note, updated_by, by_kind, updated_at FROM model_annotations")
        return {r[0]: {"flags": list(r[1] or []), "note": r[2], "updated_by": r[3], "by_kind": r[4],
                       "updated_at": r[5]} for r in cur.fetchall() if r[0]}
    return _run(op) or {}
