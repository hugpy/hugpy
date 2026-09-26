"""media_bus's at-fork guard: a forked child drops the parent's cached DB init
state and reconnects on its OWN fresh handle, and the parent closing its handle
does not break the child.

Incident 2026-09-24: the app was built (opening media_jobs.db) in the process
that becomes the gunicorn arbiter, then a forked worker inherited those sqlite
handles; when the arbiter's connection went away the worker's inherited WAL/SHM
fds went "(deleted)". The primary fix builds the app in the worker; this test
pins the belt-and-braces backstop — os.register_at_fork(after_in_child=...) that
resets media_bus._initialized so a child can never lean on a pre-fork handle.

POSIX-only (needs os.fork); skipped elsewhere.
"""
from __future__ import annotations

import os

import pytest

from hugpy_video.intel import media_bus

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"),
                                reason="fork-safety test needs os.fork (POSIX)")


def test_child_reopens_after_fork_and_survives_parent_close(tmp_path, monkeypatch):
    # Private WAL DB under the pytest temp tree (a system temp dir the storage
    # audit guard permits). conftest snapshots/restores media_bus globals per
    # test, so repointing DB_PATH/_initialized here is safe.
    db = tmp_path / "media_jobs.db"
    monkeypatch.setattr(media_bus, "DB_PATH", str(db))
    media_bus._initialized = False

    # Parent opens the store, writes a committed row (autocommit + WAL), and
    # KEEPS its handle open across the fork — the inherited-handle scenario.
    media_bus._ensure_db()
    assert media_bus._initialized is True
    pconn = media_bus._connect()
    pconn.execute(
        "INSERT INTO media_jobs (job_id, name, status, created, updated) "
        "VALUES (?,?,?,?,?)", ("j-fork", "crop", "queued", 1.0, 1.0))

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # -- child ------------------------------------------------
        code = 0
        try:
            os.close(w)
            # The at-fork guard must have dropped the parent's cached init flag.
            if media_bus._initialized is not False:
                os._exit(11)
            # Wait until the PARENT has closed its own handle, then read on a
            # FRESH connection: proves the child reopens (never reuses the
            # inherited handle) and the parent's close did not break it.
            os.read(r, 1)
            media_bus._ensure_db()
            conn = media_bus._connect()
            try:
                n = conn.execute("SELECT COUNT(*) FROM media_jobs").fetchone()[0]
            finally:
                conn.close()
            if int(n) < 1:
                os._exit(12)
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    # -- parent ---------------------------------------------------------------
    os.close(r)
    pconn.close()          # the parent drops its handle (the incident's trigger)
    os.write(w, b"x")      # release the child to read on a fresh connection
    os.close(w)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), "child was signalled, not a clean exit"
    assert os.WEXITSTATUS(status) == 0, f"child failed with code {os.WEXITSTATUS(status)}"


def test_at_fork_reset_is_registered():
    """The reset callback exists and clears the fork-inherited init/throttle
    state (unit-level guard so a refactor can't silently drop it)."""
    media_bus._initialized = True
    media_bus._last_reap_ts = 123.0
    media_bus._reset_after_fork_in_child()
    assert media_bus._initialized is False
    assert media_bus._last_reap_ts == 0.0
