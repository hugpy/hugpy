"""Tests for the test-storage isolation + live-storage audit guard.

These guard the fix for incident 2026-09-24 (a test process resolving a storage
root to the operator's LIVE storage). They exercise the two entry points of
``hugpy_platform.test_isolation`` without installing a process-wide audit hook.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from hugpy_platform import test_isolation as ti


@pytest.fixture
def _saved_roots():
    keys = tuple(ti._ROOT_SUBPATHS) + ("HUGPY_TEST_STORAGE_BASE", "HUGPY_TEST_LIVE_ROOT")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_isolate_forces_roots_under_base_over_inherited_live(_saved_roots):
    # Simulate the incident: an inherited DEFAULT_ROOT pointing at "live" storage.
    os.environ["DEFAULT_ROOT"] = "/mnt/16T_toshiba/llm_storage"
    os.environ.pop("HUGPY_TEST_LIVE_ROOT", None)
    base = tempfile.mkdtemp(prefix="iso-test-")

    returned = ti.isolate_storage_to_tmp(base=base)

    assert returned == base
    # DEFAULT_ROOT is FORCED to the temp base, not left at the inherited live path.
    assert os.environ["DEFAULT_ROOT"] == base
    assert os.environ["UPLOADS_HOME"] == os.path.join(base, "uploads")
    assert os.environ["PROJECTS_HOME"] == os.path.join(base, "projects")
    # The video state DIR is pinned; the media_jobs.db / reservations.db FILES
    # derive from it and PROJECTS_HOME (not pinned directly, so a test that
    # repoints the dir var still gets its own derived DB path).
    assert os.environ["HUGPY_VIDEO_STATE_DIR"] == os.path.join(base, "video_intel")
    # Directories are created.
    assert os.path.isdir(os.path.join(base, "uploads"))
    assert os.path.isdir(os.path.join(base, "video_intel"))
    assert os.path.isdir(os.path.join(base, "projects"))
    # The would-be live root was captured for the guard's DENY list; the temp
    # base is NOT captured (nothing under it must ever be denied).
    assert os.path.realpath("/mnt/16T_toshiba/llm_storage") in ti._LIVE_ROOTS
    assert os.path.realpath(base) not in ti._LIVE_ROOTS


def test_isolate_optout_leaves_env_untouched(_saved_roots):
    os.environ["HUGPY_TEST_LIVE_ROOT"] = "1"
    os.environ["DEFAULT_ROOT"] = "/some/live/root"

    assert ti.isolate_storage_to_tmp(base=tempfile.mkdtemp()) is None
    # Opt-out: the inherited value is left exactly as-is.
    assert os.environ["DEFAULT_ROOT"] == "/some/live/root"


def test_is_db_path_predicate():
    assert ti._is_db_path("/x/media_jobs.db")
    assert ti._is_db_path("/x/media_jobs.db-wal")
    assert ti._is_db_path("/x/media_jobs.db-shm")
    assert ti._is_db_path("/x/reservations.sqlite")
    assert ti._is_db_path("/x/jobs.db-journal")
    assert not ti._is_db_path("/x/profile.json")
    assert not ti._is_db_path("/x/frame_0001.png")


def test_guard_denies_live_roots_and_allows_everything_else():
    # DENY list: the captured live roots. Everything NOT under them passes —
    # including coverage.py's own sqlite data file in the job work dir, which the
    # old allow-list wrongly blocked (verify job-46 INTERNALERROR).
    live_root = "/mnt/16T_toshiba/llm_storage"
    per_job_root = "/srv/pkgtest/work/job-46/home/llm_storage"
    hook = ti._make_guard_hook([live_root, per_job_root])
    live = live_root + "/video_intel/media_jobs.db"

    # Every dangerous shape against a live-root path is refused.
    with pytest.raises(RuntimeError):
        hook("sqlite3.connect", (live,))
    with pytest.raises(RuntimeError):
        hook("sqlite3.connect", ("file:" + live + "?mode=ro",))
    with pytest.raises(RuntimeError):
        hook("open", (live + "-wal", "w"))
    with pytest.raises(RuntimeError):
        hook("os.remove", (live + "-shm",))
    with pytest.raises(RuntimeError):
        hook("os.rename", (live, live + ".x"))
    with pytest.raises(RuntimeError):
        hook("shutil.copyfile", ("/etc/x.db", live))
    # reservations.db under the per-job storage root is also denied.
    with pytest.raises(RuntimeError):
        hook("sqlite3.connect", (per_job_root + "/projects/reservations.db",))

    # ALLOWED — none of these resolve under a live root:
    # coverage.py's data file in the job cov/ dir (the verify-pipeline regression)
    hook("sqlite3.connect", ("/srv/pkgtest/work/job-46/cov/hugpy.coverage.ae.pid1115041.X17jklXx",))
    # a system-temp DB (per-test tmp files)
    hook("sqlite3.connect", (os.path.join(tempfile.gettempdir(), "x.db"),))
    # a DB under a fresh private base
    fresh = os.path.realpath(tempfile.mkdtemp(prefix="guard-base-"))
    hook("sqlite3.connect", (os.path.join(fresh, "video_intel", "media_jobs.db"),))
    # a sibling directory that only shares a name prefix must NOT match.
    hook("sqlite3.connect", (live_root + "2/x.db",))
    # a non-DB open under a live root is not the guard's concern (DB-shaped only).
    hook("open", (live_root + "/models/config.json", "w"))


def test_guard_with_no_captured_roots_is_a_noop():
    hook = ti._make_guard_hook([])
    hook("sqlite3.connect", ("/mnt/16T_toshiba/llm_storage/video_intel/media_jobs.db",))
