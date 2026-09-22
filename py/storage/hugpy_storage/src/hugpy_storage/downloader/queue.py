"""ENQUEUE / CANCEL / RETRY — everything the API is allowed to do to a download.

The console API's entire relationship with downloading is now this module. It
creates a queued job and returns; it never starts a transfer, never spawns a
child, never touches the network on a request path, and never runs a monitor
thread. The daemon (daemon.py) is the only process that executes work.

Flask-free on purpose: the daemon imports the same helpers for its own
bookkeeping, and the route layer keeps only thin shims that translate to HTTP.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from hugpy_control.jobs import Job, job_store, normalize_status, to_legacy
from hugpy_storage.downloader.engine import DOWNLOAD_KIND, invalidate_model_status_cache
from hugpy_storage.downloader.presence import downloader_alive, last_beat

# How long a job may sit unclaimed before the view says so out loud. Short
# enough that an operator learns within one poll cycle that no daemon is
# running; long enough that a normal claim (well under a second) never trips it.
WAITING_GRACE_SECONDS = 30.0

_NO_DAEMON_MESSAGE = (
    "Queued — waiting for the downloader service (hugpy-downloader-dev is not "
    "running).")
_NO_QUEUE_MESSAGE = (
    "Queued, but the shared download queue is unreachable — the downloader "
    "cannot see this job yet. It retries itself; if this persists, check "
    "HUGPY_COMMS_DB.")
_WAITING_MESSAGE = "Queued — waiting for downloader…"


def queue_depth() -> int:
    """How many download jobs are queued and unclaimed, or -1 if the shared
    queue is unreachable. -1 is NOT 0: "nothing waiting" and "I cannot see the
    queue" were the same answer for eleven days, and that is what made the
    outage invisible."""
    mirror = job_store.mirror
    if mirror is None:
        return -1
    try:
        return mirror.claimable_count((DOWNLOAD_KIND,))
    except Exception:  # noqa: BLE001 — a depth read must never break an enqueue
        return -1


def queue_healthy() -> bool:
    """False ONLY when a shared queue exists but is currently unusable
    (quarantined mirror). ``HUGPY_COMMS_DB=off`` is not a fault — it is a
    deliberate per-process configuration in which no daemon runs at all, and
    that case is reported by ``downloader_alive()`` / ``queue_depth() == -1``.
    Keeping the two apart matters: "the daemon is stopped" and "the daemon
    cannot see the queue" need different actions from the operator."""
    mirror = job_store.mirror
    if mirror is None:
        return True
    try:
        return bool(mirror.health().get("ok"))
    except Exception:  # noqa: BLE001
        return True


def enqueue_download(model_key: str, model: dict,
                     total_bytes: Optional[int] = None,
                     transport: str = "web") -> Job:
    """Create a QUEUED download job and hand it to the daemon. Returns
    immediately — the only work done here is one mirror row.

    The model spec rides in ``payload`` because the daemon is a different
    process: ``Job._model`` is runtime-only and would arrive empty. ``payload``
    is also what makes RETRY work after an API restart — the spec is persisted,
    not held in some thread's closure.
    """
    payload: dict[str, Any] = {"model": model}
    if total_bytes:
        payload["total_bytes"] = int(total_bytes)
    job = job_store.enqueue(
        model_key, kind=DOWNLOAD_KIND, transport=transport,
        status="pending", message=_WAITING_MESSAGE,
        total_bytes=total_bytes, payload=payload,
        model_name=(model or {}).get("name") or model_key,
    )
    # SAY SO WHEN THE ENQUEUE DID NOT LAND. The enqueue is best-effort against
    # the mirror; if the mirror is quarantined the local row still exists and
    # this used to return a happy job object that no daemon would ever see. The
    # route turns this message into an operator-visible warning.
    if not queue_healthy():
        job_store.update(job.id, message=_NO_QUEUE_MESSAGE)
        job = job_store.get(job.id) or job
    return job


def annotate_waiting(d: dict) -> dict:
    """Make a queued-but-unstarted job SAY SO.

    Graceful degradation is the point (there is deliberately no in-process
    fallback — falling back would resurrect the very bug this separation fixes).
    A job that has been queued past the grace window gets an honest message, and
    when the daemon's heartbeat is stale it names the service that is missing.
    Read-time only: nothing is written, so an operator starting the daemon makes
    the message disappear on the next poll."""
    try:
        if normalize_status(d.get("status")) != "pending":
            return d
        age = time.time() - float(d.get("progressed_at") or 0)
        if age < WAITING_GRACE_SECONDS:
            return d
        d = dict(d)
        if not queue_healthy():
            # The most specific truth first: a daemon can be perfectly alive and
            # still be unable to SEE this job. That was the whole 2026-08-10
            # outage, and "waiting for downloader…" actively misdirected.
            d["message"] = _NO_QUEUE_MESSAGE
        elif downloader_alive():
            d["message"] = _WAITING_MESSAGE
        else:
            d["message"] = _NO_DAEMON_MESSAGE
    except (TypeError, ValueError):
        pass
    return d


def list_downloads() -> list[dict]:
    """The /jobs view: every download row, MIRROR-MERGED, legacy wire shape.

    Two things this must get right, both of which the old local-only read got
    for free by owning everything:
      * live rows now belong to the DAEMON, so they only exist in the mirror;
      * TERMINAL rows must stay visible — snapshot() hides cross-process
        terminals except for media kinds, and a download that vanished at 100%
        instead of showing "completed" would be a worse UI than before.
    """
    rows = job_store.snapshot(kinds={DOWNLOAD_KIND}, live_only=False,
                              terminal_kinds=(DOWNLOAD_KIND,))
    return [to_legacy(annotate_waiting(d)) for d in rows]


def get_download(job_id: str) -> Optional[dict]:
    """One download row, mirror-merged, legacy wire shape (None if unknown)."""
    d = job_store.get_dict(job_id)
    if d is None:
        return None
    return to_legacy(annotate_waiting(d))


def cancel_download(job_id: str) -> dict:
    """Cancel a download WHEREVER it runs.

    ``cancel_authoritative`` already does exactly the right thing across
    processes: with no live owner in THIS process it raises the shared cancel
    flag (which the daemon's store watcher turns into a real teardown) AND
    force-marks the row terminal, so a job nobody owns can never stay immortal.
    """
    d = job_store.get_dict(job_id)
    if d is None:
        return {"cancelled": False, "reason": "unknown job"}
    if normalize_status(d.get("status")) in ("done", "cancelled", "failed",
                                             "expired"):
        return {"cancelled": False, "reason": f"job is {d.get('status')}"}
    res = job_store.cancel_authoritative(job_id, reason="Cancelled by user.")
    if res.get("cancelled"):
        invalidate_model_status_cache(
            f"download cancelled: {d.get('model_key')}",
            model_key=d.get("model_key") or None)
    return {"cancelled": bool(res.get("cancelled")), "mode": res.get("mode")}


def discard_download(job_id: str) -> dict:
    """ADMIN DISCARD of a terminal failure record (k121). The ONLY way a
    failed/expired row leaves the queue view — they are never auto-pruned. A
    live job cannot be discarded (cancel is the verb for those); the honest
    refusal names the current status."""
    d = job_store.get_dict(job_id)
    if d is None:
        return {"discarded": False, "reason": "unknown job"}
    if normalize_status(d.get("status")) not in ("done", "cancelled", "failed",
                                                 "expired"):
        return {"discarded": False,
                "reason": f"job is {d.get('status')} — cancel it first"}
    ok = job_store.discard(job_id)
    return {"discarded": bool(ok), "id": job_id}


# A claimed row is only reclaimed when the daemon's heartbeat has been silent
# for MUCH longer than presence.STALE_SECONDS — a transiently busy daemon must
# never have an actively-downloading row yanked out from under it. adopt_stale
# on daemon startup remains the fast path; this is the backstop for the daemon
# that dies and NEVER restarts (the "claimed by a ghost, queued forever" hole).
REAP_DEAD_CLAIM_SECONDS = 180.0


def reap_dead_claims() -> list[str]:
    """Re-queue download rows claimed by a downloader that is provably gone.

    Read-path safe (called from GET /jobs): does nothing unless the heartbeat
    has been silent past REAP_DEAD_CLAIM_SECONDS. Re-queued rows carry an
    honest message; partial files on disk are resumed by the next daemon."""
    mirror = job_store.mirror
    if mirror is None:
        return []
    beat = last_beat()
    if beat is not None and (time.time() - beat) < REAP_DEAD_CLAIM_SECONDS:
        return []
    try:
        return mirror.adopt_stale(
            (DOWNLOAD_KIND,), "__jobs-view-reaper__",
            message=("Re-queued — the downloader service went away while this "
                     "job was claimed; it resumes when the service is back."))
    except Exception:  # noqa: BLE001 — a reap must never break the /jobs view
        return []


def set_diagnosis(job_id: str, text: str) -> bool:
    """Pin a keeper diagnosis to a job record (survives until the record is
    discarded)."""
    mirror = job_store.mirror
    if mirror is None:
        return False
    return mirror.set_job_note(job_id, "diagnosis", text)


def merge_notes(rows: list[dict]) -> list[dict]:
    """Attach pinned notes (diagnosis) to /jobs view rows, in place."""
    mirror = job_store.mirror
    if mirror is None or not rows:
        return rows
    notes = mirror.job_notes_for([d.get("id") for d in rows])
    for d in rows:
        for key, value in (notes.get(d.get("id")) or {}).items():
            d.setdefault(key, value)
    return rows


def retry_download(job_id: str) -> dict:
    """Re-queue a failed/cancelled download so the daemon picks it up again.

    Same job id and the SAME persisted payload, so partial files already on disk
    are resumed (HF resume + staging adoption), not re-fetched."""
    d = job_store.get_dict(job_id)
    if d is None:
        return {"retried": False, "reason": "unknown job"}
    if normalize_status(d.get("status")) not in ("done", "cancelled", "failed",
                                                 "expired"):
        return {"retried": False, "reason": f"job is already {d.get('status')}"}
    if not (d.get("payload") or {}).get("model"):
        return {"retried": False, "reason": "no model context to resume from"}
    ok = job_store.requeue(job_id, message=_WAITING_MESSAGE,
                           kinds=(DOWNLOAD_KIND,))
    if not ok:
        return {"retried": False, "reason": "job row is gone"}
    return {"retried": True, "id": job_id}
