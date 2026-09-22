"""The download work queue: enqueue -> claim -> run, across processes.

WHAT THIS PROTECTS (operator, 2026-07-27/28: a download "pushes off all of the
workers"). Downloads moved OUT of the console API into the hugpy-downloader
daemon. The API and the daemon are different processes that share only the comms
SQLite mirror, so the contract between them has to hold under exactly the
conditions a unit test can reproduce:

  1. an enqueued job is visible to a DIFFERENT store instance (the daemon) and
     is claimable;
  2. a claim is a COMPARE-AND-SET — two daemons racing the same row get one
     winner and one None, or the same transfer runs twice;
  3. the enqueuer must NOT keep a local record: snapshot() prefers a local row
     over the mirror row of the same id, so a leftover `pending` row in the API
     would permanently MASK the daemon's live progress (the whole console would
     show 0% forever);
  4. terminal download rows stay VISIBLE cross-process — snapshot() hides
     sibling terminals except media kinds, and a job that vanished at the finish
     line instead of reading "completed" is a worse UI than the bug we fixed;
  5. cancel crosses the boundary: the API raises the flag, the OWNER tears down;
  6. retry re-queues the row WITH its payload, so the daemon can resume;
  7. fail-over: a dead daemon's claimed rows go back on the queue.

Runs entirely in-process against a temp DB — two JobStore instances standing in
for the two processes.
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_control.jobs import JobStore, to_legacy
from hugpy_control.shared import SqliteMirror

KIND = "download"


@pytest.fixture()
def two_processes(tmp_path):
    """(api, daemon) — two independent stores over ONE mirror file, which is
    exactly the topology in production (gunicorn + hugpy-downloader-dev)."""
    db = str(tmp_path / "comms.db")
    api = JobStore(mirror=SqliteMirror(db))
    daemon = JobStore(mirror=SqliteMirror(db))
    return api, daemon


def _payload(key="tiny"):
    return {"model": {"hub_id": f"acme/{key}", "name": key}}


# ── 1 + 3. enqueue is visible to the daemon, and NOT masked by the API ──────
def test_enqueued_job_is_claimable_and_not_masked_locally(two_processes):
    api, daemon = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())

    # The API disowned it: no local record, so nothing can shadow the mirror.
    assert api.get(job.id) is None
    # …but it is still READABLE there, through the mirror.
    assert api.get_dict(job.id)["status"] == "pending"

    claimed = daemon.claim_next((KIND,), owner="daemon-1")
    assert claimed is not None and claimed["id"] == job.id
    assert claimed["payload"]["model"]["hub_id"] == "acme/tiny"

    # The daemon becomes the owner and starts running it.
    daemon.create("tiny", id=job.id, kind=KIND, payload=claimed["payload"])
    daemon.update(job.id, status="running", progress=0.25)

    # The API now sees the DAEMON's live state — the transition the console
    # polls for. Legacy wire shape preserved: "running", not "processing".
    view = api.snapshot(kinds={KIND}, live_only=False,
                        terminal_kinds=(KIND,))
    assert len(view) == 1
    assert to_legacy(view[0])["status"] == "running"
    assert view[0]["progress"] == 0.25


# ── 2. the claim is a compare-and-set ──────────────────────────────────────
def test_only_one_daemon_can_claim_a_job(two_processes):
    api, daemon = two_processes
    api.enqueue("tiny", kind=KIND, payload=_payload())

    first = daemon.claim_next((KIND,), owner="daemon-1")
    second = daemon.claim_next((KIND,), owner="daemon-2")
    assert first is not None
    assert second is None, "a claimed job must never be handed out twice"


def test_claim_only_takes_its_own_kind(two_processes):
    api, daemon = two_processes
    api.enqueue("chatty", kind="chat")
    assert daemon.claim_next((KIND,), owner="daemon-1") is None


# ── 4. terminal download rows stay visible cross-process ───────────────────
def test_completed_download_is_still_visible_to_the_api(two_processes):
    api, daemon = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())
    claimed = daemon.claim_next((KIND,), owner="daemon-1")
    daemon.create("tiny", id=job.id, kind=KIND, payload=claimed["payload"])
    daemon.finish(job.id, "completed")

    rows = api.snapshot(kinds={KIND}, live_only=False, terminal_kinds=(KIND,))
    assert [to_legacy(r)["status"] for r in rows] == ["completed"]
    # …and the single-job read agrees.
    assert to_legacy(api.get_dict(job.id))["status"] == "completed"

    # Without the download opt-in the sibling's terminal row is hidden — this is
    # the default that would have made a finished download vanish.
    assert api.snapshot(kinds={KIND}, live_only=False) == []


# ── 5. cancel crosses the process boundary ─────────────────────────────────
def test_cancel_from_the_api_reaches_the_owning_daemon(two_processes):
    api, daemon = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())
    claimed = daemon.claim_next((KIND,), owner="daemon-1")
    daemon.create("tiny", id=job.id, kind=KIND, payload=claimed["payload"])
    daemon.update(job.id, status="running")

    torn_down = []
    daemon.attach_cancel(job.id, lambda: torn_down.append(True))

    assert api.cancel_authoritative(job.id)["cancelled"] is True
    # The flag is what the daemon's watcher thread polls; assert the mechanism
    # directly rather than sleeping on the 1s tick.
    assert job.id in daemon.mirror.flagged_ids()
    daemon.cancel(job.id, reason="cancelled by sibling process")
    assert torn_down == [True], "the OWNER must run the teardown"


# ── 6. retry re-queues WITH the payload ────────────────────────────────────
def test_retry_requeues_the_job_with_its_payload(two_processes):
    api, daemon = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())
    claimed = daemon.claim_next((KIND,), owner="daemon-1")
    daemon.create("tiny", id=job.id, kind=KIND, payload=claimed["payload"])
    daemon.finish(job.id, "failed", error="boom")

    assert api.requeue(job.id, message="retrying", kinds=(KIND,)) is True
    assert api.get_dict(job.id)["status"] == "pending"

    again = daemon.claim_next((KIND,), owner="daemon-2")
    assert again is not None and again["id"] == job.id
    assert again["payload"]["model"]["hub_id"] == "acme/tiny", (
        "a retry must resume the SAME model, not an empty job")


def test_requeue_refuses_a_foreign_kind(two_processes):
    api, _daemon = two_processes
    job = api.enqueue("chatty", kind="chat")
    assert api.requeue(job.id, kinds=(KIND,)) is False


# ── 7. fail-over: a dead daemon's work goes back on the queue ──────────────
def test_a_dead_daemons_running_job_is_adopted(two_processes):
    api, dead = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())
    claimed = dead.claim_next((KIND,), owner="host:111")
    dead.create("tiny", id=job.id, kind=KIND, payload=claimed["payload"])
    dead.update(job.id, status="running", progress=0.4)

    fresh = JobStore(mirror=SqliteMirror(api.mirror.path))
    adopted = fresh.adopt_stale((KIND,), owner="host:222",
                                message="resuming after restart")
    assert adopted == [job.id]
    assert fresh.get_dict(job.id)["status"] == "pending"

    reclaimed = fresh.claim_next((KIND,), owner="host:222")
    assert reclaimed is not None and reclaimed["id"] == job.id


def test_a_live_daemon_never_has_its_claim_yanked(two_processes):
    api, daemon = two_processes
    job = api.enqueue("tiny", kind=KIND, payload=_payload())
    daemon.claim_next((KIND,), owner="host:111")
    daemon.update(job.id, status="running") if daemon.get(job.id) else None

    # The SAME owner re-running adopt (a poll loop, not a restart) must not
    # re-queue its own in-flight work.
    assert api.adopt_stale((KIND,), owner="host:111") == []


# ── the graceful-degradation message (no daemon running) ───────────────────
def test_queued_job_says_it_is_waiting_when_no_daemon_is_running(monkeypatch,
                                                                 tmp_path):
    from hugpy_storage.downloader import presence, queue as dlq

    monkeypatch.setattr(presence, "heartbeat_path",
                        lambda: str(tmp_path / "hb"))
    monkeypatch.setattr(dlq, "downloader_alive", lambda: False)

    fresh = {"status": "pending", "progressed_at": time.time(), "message": "x"}
    assert dlq.annotate_waiting(fresh)["message"] == "x", (
        "a just-enqueued job must not be accused of waiting")

    old = {"status": "pending", "message": "x",
           "progressed_at": time.time() - dlq.WAITING_GRACE_SECONDS - 1}
    msg = dlq.annotate_waiting(old)["message"]
    assert "not running" in msg, msg

    # A RUNNING job is never annotated, whatever the heartbeat says.
    running = {"status": "processing", "message": "Downloading…",
               "progressed_at": 0}
    assert dlq.annotate_waiting(running)["message"] == "Downloading…"
