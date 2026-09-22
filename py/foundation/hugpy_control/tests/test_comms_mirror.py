"""Cross-process mirror: two JobStores sharing one SQLite file simulate two
gunicorn workers — cancel lands on the wrong one, queue views merge."""
from __future__ import annotations

import threading
import time

import pytest

from hugpy_control.jobs import JobStore
from hugpy_control.shared import SqliteMirror


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "comms.db")


def test_cancel_on_sibling_process_reaches_the_owner(db):
    A = JobStore(mirror=SqliteMirror(db))   # "gunicorn worker A" — owns the stream
    B = JobStore(mirror=SqliteMirror(db))   # "gunicorn worker B" — gets the POST

    fired = threading.Event()
    A.create("qwen", id="x-1", kind="chat", transport="web")
    A.attach_cancel("x-1", fired.set)
    assert B.get("x-1") is None, "B sees no local job"

    # queue merge: B's snapshot shows A's live job via the mirror
    assert any(d["id"] == "x-1" for d in B.snapshot())
    assert B.counts()["total"] >= 1

    # cancel lands on B -> True (remote live), flag raised on the mirror only
    assert B.cancel("x-1", reason="user stop")
    assert not fired.is_set(), "A's handle not fired yet (flag only)"

    # A's watcher notices within ~2s and fires the local handle
    assert fired.wait(timeout=4), "A's watcher fires the handle"
    assert A.get("x-1").cancel_requested
    A.finish("x-1")
    assert A.get("x-1").to_dict()["status"] == "cancelled"

    # terminal state propagates: B's view no longer shows it (after A's upsert)
    time.sleep(0.1)
    assert not any(d["id"] == "x-1" for d in B.snapshot())


def test_cancel_before_attach_fires_from_the_mirror_flag(db):
    B2 = JobStore(mirror=SqliteMirror(db))
    A2 = JobStore(mirror=SqliteMirror(db))
    A2.create("m", id="x-2", kind="chat")
    assert B2.cancel("x-2"), "B2 flags the unattached job"
    late = threading.Event()
    A2.attach_cancel("x-2", late.set)   # attach checks the mirror flag directly
    assert late.wait(timeout=4), "late attach fires from mirror flag"


def test_unknown_id_cancels_false(db):
    B = JobStore(mirror=SqliteMirror(db))
    assert not B.cancel("nope")


def test_resurrection_clears_the_shared_flag(db):
    A3 = JobStore(mirror=SqliteMirror(db))
    A3.create("m", id="x-3", kind="download")
    A3.cancel("x-3")
    A3.finish("x-3")
    A3.update("x-3", status="running")          # retry
    assert not A3.mirror.cancel_requested("x-3")
