"""media_bus stale-handle (disk-I/O) recovery — incident 2026-09-24.

A deleted -wal/-shm sidecar surfaces as ``sqlite3.OperationalError: disk I/O
error``. The read paths (get / list_jobs) must degrade to the honest empty view
AND drop the cached handle so the next call reconnects fresh — never a 500, and
never a quarantine of the (intact) main file. A lock error must still propagate.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from hugpy_video.intel import media_bus


@pytest.fixture
def tmp_db(monkeypatch):
    d = tempfile.mkdtemp(prefix="stale-handle-")
    monkeypatch.setattr(media_bus, "DB_PATH", os.path.join(d, "media_jobs.db"))
    monkeypatch.setattr(media_bus, "_initialized", False)
    media_bus._ensure_db()
    return media_bus.DB_PATH


def test_is_disk_io_matches_only_disk_io():
    assert media_bus._is_disk_io(sqlite3.OperationalError("disk I/O error"))
    assert not media_bus._is_disk_io(sqlite3.OperationalError("database is locked"))
    assert not media_bus._is_disk_io(sqlite3.DatabaseError("database disk image is malformed"))
    assert not media_bus._is_disk_io(ValueError("disk I/O error"))


def test_get_degrades_and_resets_on_disk_io(tmp_db, monkeypatch):
    media_bus._initialized = True

    def _boom():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(media_bus, "_connect_ro", _boom)
    monkeypatch.setattr(media_bus, "_ensure_db", lambda: None)

    view = media_bus.get("job-xyz")
    # Honest unknown-id view, not a raise.
    assert view["job_id"] == "job-xyz"
    assert view["status"] is None and view["result"] is None
    # Cached handle dropped so the next call reconnects fresh.
    assert media_bus._initialized is False


def test_list_jobs_degrades_on_disk_io(tmp_db, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(media_bus, "_connect_ro", _boom)
    monkeypatch.setattr(media_bus, "_ensure_db", lambda: None)

    assert media_bus.list_jobs(include_terminal=True) == []
    assert media_bus._initialized is False


def test_get_still_raises_on_lock(tmp_db, monkeypatch):
    def _locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(media_bus, "_connect_ro", _locked)
    monkeypatch.setattr(media_bus, "_ensure_db", lambda: None)

    with pytest.raises(sqlite3.OperationalError):
        media_bus.get("job-xyz")
