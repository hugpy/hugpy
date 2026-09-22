"""Parked during the partition (curation, 2026-09-22).

The review pipeline's former in-process download driver: it created a job row
in the control job store itself and drove ``hugpy_storage.downloader.engine
.run_download_job`` — a second copy of the daemon's claim/run lifecycle inside
curation. Replaced by ``hugpy_curation.review.download.download_quant``, which
uses storage's public queue (``enqueue_download`` / ``get_download``) when a
daemon is alive and ``download_models.download_one`` in-process otherwise.
Kept here for reference only; nothing imports it.
"""
from __future__ import annotations


# ── download ───────────────────────────────────────────────────────────────
def _download(hub_id: str, quant: str, files: list[str]) -> str:
    """Fetch one quant through the SHARED staged downloader guard and return the
    directory it landed in.

    Routed through ``downloader.engine.run_download_job`` (NOT a bare
    ``download_one``) on purpose: that is the very same stall-killer / resume /
    backoff path the download daemon uses — no new bytes for STALL_SECONDS kills
    the transfer's process group and resumes it, up to MAX_ATTEMPTS. The nightly
    review runs UNATTENDED for up to 4h, so a wedged HF connection has to
    self-heal instead of hanging the whole run (finding C9). Atomic staging,
    orphan reaping and provenance stamping still come for free (the guard runs
    ``download_one`` inside its monitored child). Raises on any non-completed
    terminal so ``review_one`` records it as a 'download failed' row, exactly as
    a bare download_one raise did before."""
    import uuid
    from hugpy_control.jobs import job_store
    from hugpy_storage.downloader.engine import DOWNLOAD_KIND, run_download_job
    from hugpy_platform.constants import DEFAULT_ROOT
    from hugpy_storage.model_paths import resolve_model_dir, route_destination

    model = {
        "hub_id": hub_id,
        "framework": "gguf",
        "primary_task": "text-generation",
    }
    if len(files) == 1:
        model["filename"] = files[0]
    else:
        model["include"] = [f"*{quant}*.gguf"]      # sharded quant
    key = f"{hub_id.split('/')[-1]}-{quant}"

    # The guard owns a job row in the local comms store; create one, run it to a
    # terminal state synchronously in this (review) process, then turn anything
    # other than a clean completion into the exception review_one already knows
    # how to record. run_download_job downloads to route_destination(model) —
    # the same DEFAULT_ROOT target download_one used, so the resolver below is
    # unchanged.
    job_id = f"review-{uuid.uuid4().hex}"
    job_store.create(key, id=job_id, kind=DOWNLOAD_KIND, model_name=hub_id)
    status = run_download_job(job_id, key, model)
    if status != "completed":
        job = job_store.get(job_id)
        detail = None
        if job is not None:
            detail = getattr(job, "error", None) or getattr(job, "message", None)
        raise RuntimeError(f"download {status}: {detail or status}")
    return resolve_model_dir(model, DEFAULT_ROOT) or route_destination(model) or ""


