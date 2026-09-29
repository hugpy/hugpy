"""KNOWN-GOOD CONTRACT — queue RESUME-WHEN-CAUGHT + session/FIFO claim ordering.

Catalogue: notes/KNOWN-GOOD-QUEUE.md (pillars 3 and 4).
Source under test: hugpy_control/jobs.py (JobStore.reconcile_dispatch,
requeue, adopt_stale, claim_next) and hugpy_control/shared.py (SqliteMirror
claim_next / requeue / adopt_stale / release_claim).

These behaviours were verified live but not pinned by a test (the pre-existing
call-queue contracts cover marking/processing/cancel; the RECOVERY paths — a
lease-expired dispatch being requeued only when the worker is provably idle, and
the cross-process claim/requeue/fail-over queue — were not). This file pins the
CURRENT good behaviour; it adds a NEW file and touches no existing case.

Deterministic: one JobStore over a tmp SQLite mirror, an injected ``now`` and an
injected ``worker_state`` callback — no live GPU, no worker, no network.
Established: reviewer pass 2026-09-29 (operator: "allowing resume for when its
caught … session call allocations over strict fifo").
"""
from __future__ import annotations

import importlib
import time

import pytest

J = importlib.import_module("hugpy_control.jobs")
from hugpy_control.shared import SqliteMirror  # noqa: E402

RESOLVED = "Qwen~Qwen3-Coder-Next-GGUF"


@pytest.fixture
def store(tmp_path):
    return J.JobStore(mirror=SqliteMirror(path=str(tmp_path / "comms.db")))


def _state(online=True, slot_busy=False, request_active=False):
    return lambda job: {"online": online, "slot_busy": slot_busy,
                        "request_active": request_active}


# ---------------------------------------------------------------------------
# Pillar 3 — RESUME-WHEN-CAUGHT: reconcile_dispatch requeues a lease-expired
# dispatch ONLY when the worker is provably idle (never on a busy GPU alone).
# ---------------------------------------------------------------------------
def test_reconcile_requeues_only_a_lease_expired_and_provably_idle_dispatch(store):
    """INVARIANT: a job marked ``processing`` with a dispatch lease is requeued
    to ``pending`` (worker/slot/lease cleared) ONLY when its lease has expired
    AND the worker reports online + no busy slot + no active request. It is NOT
    requeued while the lease is live, nor when the worker's slot is busy or its
    request is active (that is forward progress, not a lost relay), nor when the
    worker's state is unknown/offline (fail-closed: a quiet GPU is never proof).
    Established: JobStore.reconcile_dispatch (jobs.py:651) — "We never infer
    staleness from a quiet GPU alone"."""
    store.create(RESOLVED, id="r1", kind="chat")
    store.begin_dispatch("r1", worker="ae-worker", lease_s=120)
    assert store.get("r1").status == "processing"

    now0 = time.time()
    expired = now0 + 300     # past the 120s lease

    # Lease still live -> never touched, whatever the worker says.
    assert store.reconcile_dispatch(worker_state=_state(), now=now0) == []
    # Lease expired but the worker slot is BUSY -> real work -> keep.
    assert store.reconcile_dispatch(
        worker_state=_state(slot_busy=True), now=expired) == []
    assert store.get("r1").status == "processing"
    # Lease expired but the worker reports an ACTIVE request -> keep.
    assert store.reconcile_dispatch(
        worker_state=_state(request_active=True), now=expired) == []
    assert store.get("r1").status == "processing"
    # Lease expired and the worker is OFFLINE / unknown -> fail-closed, keep.
    assert store.reconcile_dispatch(
        worker_state=_state(online=False), now=expired) == []
    assert store.get("r1").status == "processing"

    # Lease expired AND worker online + idle + no active request -> requeue.
    stale = store.reconcile_dispatch(worker_state=_state(), now=expired)
    assert stale == ["r1"]
    job = store.get("r1")
    assert job.status == "pending"
    assert job.worker is None and job.slot is None
    assert job.dispatch_lease_until is None and job.dispatch_started_at is None
    assert "dispatch lease expired" in job.message


# ---------------------------------------------------------------------------
# Pillar 3 — cross-process resume: mirror.requeue puts a claimed/failed job back
# on the queue (a fresh run), and adopt_stale fails a dead owner's work over.
# ---------------------------------------------------------------------------
def test_requeue_puts_a_claimed_job_back_on_the_queue_as_a_fresh_run(store):
    """INVARIANT: requeue() of a claimed cross-process job resets it to
    ``pending``, DROPS the claim (claimable again), lowers the cancel flag and
    clears run telemetry — the cross-process retry primitive. The requeuing
    process holds no local record afterwards (detach), so it cannot mask the
    mirror row it just reset.
    Established: JobStore.requeue (jobs.py:496) / SqliteMirror.requeue
    (shared.py:836)."""
    job = store.enqueue("x", kind="download")          # pending, disowned locally
    assert store.claim_next(("download",), "dl-1")["id"] == job.id
    assert store.mirror.claim_of(job.id) == "dl-1"

    assert store.requeue(job.id, message="retry after fail",
                         kinds=("download",)) is True
    row = store.mirror.row(job.id)
    assert row["status"] == "pending"
    assert row.get("cancel_requested") is False
    assert store.mirror.claim_of(job.id) is None       # up for grabs again
    assert store.get(job.id) is None                   # no masking local record
    # And it is genuinely re-claimable.
    assert store.claim_next(("download",), "dl-2")["id"] == job.id


def test_adopt_stale_failovers_a_dead_owners_work_but_never_a_live_siblings(store):
    """INVARIANT: adopt_stale re-queues every job a DIFFERENT (dead) owner left
    claimed, so a restarted daemon resumes it; it never yanks a row the CALLING
    owner itself claimed (a live sibling's work is safe).
    Established: JobStore.adopt_stale (jobs.py:486) / SqliteMirror.adopt_stale
    (shared.py:880)."""
    dead = store.enqueue("a", kind="download")
    store.claim_next(("download",), "dead-owner")

    adopted = store.adopt_stale(("download",), "new-owner", message="failover")
    assert adopted == [dead.id]
    assert store.mirror.claim_of(dead.id) is None
    assert store.mirror.row(dead.id)["status"] == "pending"

    # A row the caller holds under its OWN id is never re-queued out from under it.
    store.claim_next(("download",), "new-owner")
    assert store.adopt_stale(("download",), "new-owner") == []


# ---------------------------------------------------------------------------
# Pillar 4 — SESSION-CALL ALLOCATION over strict FIFO: the cross-process claim
# queue hands out the OLDEST unclaimed job (FIFO within a kind), atomically, and
# never twice.
# ---------------------------------------------------------------------------
def test_claim_next_is_atomic_oldest_first_and_never_double_claims(store):
    """INVARIANT: claim_next hands the OLDEST unclaimed pending job of the
    requested kinds (FIFO by ``updated``), under a write lock, so two daemons
    can never run one transfer twice; claimable_count reflects only the
    still-queued rows; a cancelled row is never claimed.
    Established: SqliteMirror.claim_next (shared.py:738) — BEGIN IMMEDIATE +
    ORDER BY updated ASC."""
    ids = []
    for n in range(3):
        ids.append(store.enqueue("m%d" % n, kind="download").id)
        time.sleep(0.01)                               # distinct `updated` epochs
    assert store.mirror.claimable_count(("download",)) == 3

    got = [store.claim_next(("download",), "dl")["id"] for _ in range(3)]
    assert got == ids                                  # oldest-first FIFO
    assert store.claim_next(("download",), "dl") is None
    assert store.mirror.claimable_count(("download",)) == 0

    # A cancelled queued row is skipped by the claim, never handed out.
    c = store.enqueue("c", kind="download")
    store.mirror.request_cancel(c.id)
    assert store.claim_next(("download",), "dl") is None
