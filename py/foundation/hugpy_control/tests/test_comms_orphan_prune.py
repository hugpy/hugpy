"""Orphan-sweep in SqliteMirror.prune() must age on real forward-progress
silence (progressed_at), NOT on `updated`.

Regression: identity_mesh_build render jobs wedge mid-render (ae drops during
Hunyuan3D texture load). They read stalled=true but never go terminal, and they
were IMMORTAL because upsert() bumps `updated`=now on EVERY write — including
snapshot re-upserts triggered merely by VIEWING /llm/jobs and the API-restart
re-read. The orphan-sweep keyed on `updated`, so its deadline never elapsed.

Fix: the orphan-sweep keys on progressed_at (the movement-only clock).
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from hugpy_control.shared import SqliteMirror

# retain_secs=600 -> orphan window = max(600*6, 3600) = 3600s (1h).
RETAIN = 600.0


def _row_count(mirror, job_id):
    with sqlite3.connect(mirror.path) as conn:
        return conn.execute("SELECT COUNT(*) FROM jobs WHERE id=?", (job_id,)).fetchone()[0]


def _set_updated(mirror, job_id, when):
    with sqlite3.connect(mirror.path) as conn:
        conn.execute("UPDATE jobs SET updated=? WHERE id=?", (when, job_id))


@pytest.fixture
def mirror(tmp_path):
    return SqliteMirror(str(tmp_path / "comms.db"), retain_secs=RETAIN)


def test_wedged_render_with_fresh_updated_is_reaped(mirror):
    # THE BUG: ancient progressed_at, FRESH updated (a view re-upserted it).
    now = time.time()
    mirror.upsert({"id": "wedged-1", "status": "processing", "kind": "video",
                   "progressed_at": now - 2 * 3600})
    assert _row_count(mirror, "wedged-1") == 1
    with sqlite3.connect(mirror.path) as conn:
        upd = conn.execute("SELECT updated FROM jobs WHERE id=?", ("wedged-1",)).fetchone()[0]
    assert (now - upd) < 5, "upsert stamped a fresh `updated` (the immortality mechanism)"
    mirror.prune()
    assert _row_count(mirror, "wedged-1") == 0


def test_healthy_progressing_render_survives_an_ancient_updated(mirror):
    now = time.time()
    mirror.upsert({"id": "healthy-1", "status": "processing", "kind": "video",
                   "progressed_at": now - 10})
    _set_updated(mirror, "healthy-1", now - 10 * 3600)
    mirror.prune()
    assert _row_count(mirror, "healthy-1") == 1


def test_pending_job_is_starved_not_wedged(mirror):
    now = time.time()
    mirror.upsert({"id": "queued-1", "status": "pending", "kind": "video",
                   "progressed_at": now - 5 * 3600})
    mirror.prune()
    assert _row_count(mirror, "queued-1") == 1


def test_null_progressed_at_is_fail_open(mirror):
    now = time.time()
    mirror.upsert({"id": "noprog-1", "status": "processing", "kind": "video"})
    _set_updated(mirror, "noprog-1", now - 10 * 3600)
    mirror.prune()
    assert _row_count(mirror, "noprog-1") == 1


def test_terminal_cleanup_is_still_keyed_on_updated(mirror):
    now = time.time()
    mirror.upsert({"id": "done-1", "status": "done", "kind": "chat", "progressed_at": now})
    _set_updated(mirror, "done-1", now - 2 * RETAIN)
    mirror.prune()
    assert _row_count(mirror, "done-1") == 0


def test_legacy_db_without_the_column_migrates_idempotently(tmp_path):
    legacy_db = str(tmp_path / "legacy.db")
    now = time.time()
    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL, "
            "status TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'chat', "
            "cancel_requested INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)")
        conn.execute(
            "INSERT INTO jobs (id, data, status, kind, cancel_requested, updated) "
            "VALUES (?,?,?,?,?,?)", ("legacy-1", "{}", "processing", "video", 0, now))

    legacy = SqliteMirror(legacy_db, retain_secs=RETAIN)
    legacy.prune()                      # first touch -> _ensure() -> _migrate()
    with sqlite3.connect(legacy_db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert {"progressed_at", "claimed_by", "claimed_at"} <= cols
    assert _row_count(legacy, "legacy-1") == 1, "NULL progressed_at -> fail-open"
    SqliteMirror(legacy_db, retain_secs=RETAIN).prune()   # re-migration is a no-op
