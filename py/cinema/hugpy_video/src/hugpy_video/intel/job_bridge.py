"""One-directional bridge: media_bus job lifecycle -> comms.JobStore (A/P0-2).

The roadmap ratified a BRIDGE, not a collapse:

  * comms.JobStore  = attribution / queue / cancel source of truth. It is what
    ``GET /llm/jobs`` reads. Before this bridge NO media_bus job (crop / scene /
    movie / studio_i2v) surfaced there — media jobs were queryable ONLY at
    ``GET /video/jobs/<id>``.
  * media_bus       = execution source of truth (its own durable media_jobs.db).

This module mirrors media-bus transitions INTO the JobStore, and does so
STRICTLY one-directionally (bus -> JobStore). It NEVER writes back into the bus,
so there is no distributed-write deadlock: cancel/attribution/queue stay the
JobStore's job, execution stays the bus's job.

Best-effort by construction: every function swallows and logs on failure, so a
bridge hiccup can NEVER fail a media job (the mirror plumbing in comms.jobs has
the same discipline). media_bus calls these at exactly three points:
``enqueue`` -> ``on_enqueue`` (queued), ``run_claimed`` -> ``on_running``
(running) then ``on_terminal`` (done | failed | cancelled).

Cross-process shape (the dev service runs gunicorn --workers 3 sharing ONE comms
mirror via HUGPY_COMMS_DB): the media daemon that RUNS a job is frequently a
different process than the one that enqueued it (each process runs a daemon
thread and the bus's atomic sqlite claim hands the job to whichever wins). So:

  * ``on_enqueue`` writes the QUEUED snapshot straight to the shared MIRROR and
    does NOT anchor a local JobStore record. A local "pending" record in the
    enqueue process would be a record that process never advances (a sibling
    runs the job) — it would shadow the fresher cross-process mirror view and
    stick at "pending". Mirror-only avoids that: every process still shows the
    job as queued (snapshot merges live mirror rows), and there is no stale
    local shadow.
  * ``on_running`` / ``on_terminal`` run in the process that actually owns the
    job. They anchor a LOCAL JobStore record and mark it terminal there. The
    JobStore retains terminal jobs (~600s), and ``GET /llm/jobs?live=0`` reads
    that retained representation — so a completed media job SHOWS, it does not
    vanish. Terminal is deliberately carried by local retention, NOT by the
    mirror's live_rows() (which excludes terminal rows — the exact gap a prior
    investigation flagged as the naive-bridge trap).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# media_bus status vocab -> comms canonical status vocab. media_bus speaks
# queued/claimed/running/cancelling/done/failed/cancelled; comms speaks
# pending/processing/streaming/done/cancelled/failed. Running maps to
# "processing" (a generic in-flight state) rather than "streaming" on purpose:
# "streaming" carries token-stream semantics (tokens/sec, on_output) that are
# meaningless for a frame-rendering media job — it would render oddly in the
# activity view. A media job is fully visible in /llm/jobs either way.
_STATUS_MAP = {
    "queued": "pending",
    "claimed": "processing",
    "running": "processing",
    "cancelling": "processing",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}

# Attribution transport tag for every bridged media job. Parallels the
# chat/web/discord/worker transports the JobStore already carries; lets the
# console filter /llm/jobs?transport=media.
_TRANSPORT = "media"


def _placement(job_id: str, name: str):
    """WHERE this media job executes (video_intel.placement) — a
    {source,host,worker_id,gpu,process,reserved_bytes} object, or None. Lazily
    imported + fully guarded (bridge discipline: zero import-time coupling, and a
    placement hiccup can never affect the mirror or the job). Set-only-when-present
    at the call sites so a None never blanks a placement a prior write set."""
    try:
        from hugpy_video.intel.placement import job_placement
        return job_placement(job_id, name)
    except Exception:
        return None


def _store():
    """The comms JobStore singleton. Imported lazily (never at media_bus import
    time) so media_bus stays importable even if comms is unhappy, and so there
    is zero import-time coupling / cycle risk."""
    from hugpy_control.jobs import job_store
    return job_store


def _result_uri(result) -> str | None:
    """Best-effort clip/artifact uri from a JobResult's first output — the
    back-reference to what the media job produced. Tolerates both a live
    JobResult (outputs = tuple[MediaRef]) and a plain dict (already asdict'd)."""
    try:
        outputs = getattr(result, "outputs", None)
        if outputs is None and isinstance(result, dict):
            outputs = result.get("outputs")
        if not outputs:
            return None
        first = outputs[0]
        uri = getattr(first, "uri", None)
        if uri is None and isinstance(first, dict):
            uri = first.get("uri")
        return uri or None
    except Exception:
        return None


def _error_dict(result):
    """Best-effort {code, message, retryable} from a failed JobResult's JobError.

    Since the Task 2 collapse there is ONE JobError class — ``result_schema.JobError``
    IS ``comms.jobs.JobError`` — so this is now a thin ``coerce -> to_dict`` of that
    single class rather than a hand-rolled cross-vocabulary conversion. The emitted
    shape is unchanged for the /llm/jobs contract: ``to_dict`` emits ``retryable``
    ALWAYS (None when absent) and ``detail`` only when truthy — bridged media errors
    carry ``detail=None``, so they still serialize as exactly {code, message,
    retryable}. Tolerates a live JobError, a plain dict, or an already-asdict'd
    result. The comms import stays LAZY (bridge discipline: zero import-time
    coupling)."""
    try:
        err = getattr(result, "error", None)
        if err is None and isinstance(result, dict):
            err = result.get("error")
        from hugpy_control.jobs import JobError
        job_err = JobError.coerce(err)   # JobError | dict | None -> JobError | None
        return job_err.to_dict() if job_err is not None else None
    except Exception:
        return None


def on_enqueue(job_id: str, name: str, principal: str | None = None) -> None:
    """queued -> pending. Mirror-ONLY (no local anchor): every gunicorn process
    then shows the job as queued via the shared mirror, and the enqueue process
    does not pin a record it will never advance. No-op when the mirror is off
    (single-process installs still get running/terminal from the local store).
    ``principal`` (k9) rides onto the mirrored row so a queued media job is
    attributed in /llm/jobs from the moment it lands."""
    try:
        store = _store()
        mirror = getattr(store, "mirror", None)
        if mirror is None:
            return
        # Reuse Job.to_dict() so the mirrored row is always shape-correct for
        # snapshot()/live_rows(); building the Job locally does NOT insert it
        # into the store (no stale local shadow) — we only upsert its dict.
        from hugpy_control.jobs import Job
        # Placement (WHERE it will run) rides onto the queued row so a held/queued
        # heavy render shows its locus in /llm/jobs from the moment it lands. At
        # enqueue there is no active reservation yet -> this is the TEMPLATE hint;
        # on_running upgrades it once a claim is held. Set-only-when-present.
        placement = _placement(job_id, name)
        row = Job(id=job_id, kind=name, status="pending",
                  transport=_TRANSPORT, model_name=name,
                  principal=principal,
                  **({"placement": placement} if placement else {})).to_dict()
        mirror.upsert(row)
    except Exception:
        logger.warning("job_bridge.on_enqueue failed for %s (%s)",
                       job_id, name, exc_info=True)


def on_running(job_id: str, name: str, worker: str | None = None,
               principal: str | None = None) -> None:
    """claimed/running -> processing, in the process that actually runs the job.
    Anchors a LOCAL JobStore record (create-if-missing, else advance) so this
    process owns the row for its visible life and mirrors the live status to
    siblings. ``principal`` (k9) is carried onto that local record so the
    attribution survives once this process's record wins over the mirror row
    (snapshot: local records win on id)."""
    try:
        store = _store()
        # Only carry principal when we actually have one, so a None (unattributed
        # job) never blanks an attribution a prior write already set.
        attrib = {"principal": principal} if principal else {}
        # Placement — WHERE the run executes. on_running fires BEFORE the
        # reservation acquire in run_claimed, so at first this is the TEMPLATE hint
        # (host/gpu/process); it is refreshed to the reservation truth on the next
        # progress push. Set-only-when-present (a None never blanks it).
        placement = _placement(job_id, name)
        if placement:
            attrib["placement"] = placement
        if store.get(job_id) is None:
            store.create(id=job_id, kind=name, status="processing",
                         worker=worker, transport=_TRANSPORT, model_name=name,
                         **attrib)
        else:
            store.update(job_id, status="processing", worker=worker,
                         transport=_TRANSPORT, model_name=name, **attrib)
    except Exception:
        logger.warning("job_bridge.on_running failed for %s (%s)",
                       job_id, name, exc_info=True)


def _summary(stage: str, progress) -> str:
    """A short human message for the comms Job — ``"<stage> <pct>%"`` — from a
    progress blob's stage + 0..1 float. Degrades: stage-only when there is no
    usable number, percent-only when there is no stage, "" when neither."""
    pct = None
    try:
        if progress is not None:
            p = float(progress)
            if p == p:  # not NaN
                pct = max(0, min(100, int(round(p * 100))))
    except (TypeError, ValueError):
        pct = None
    if stage and pct is not None:
        return f"{stage} {pct}%"
    if stage:
        return stage
    if pct is not None:
        return f"{pct}%"
    return ""


def _mark_progressed(store, job) -> None:
    """Stamp ``job.progressed_at`` to now and re-sync the mirror — the movement-only
    clock ``comms.jobs`` ages its stalled/wedged decisions on.

    Written HERE rather than by widening ``JobStore.update``'s advance rule, because
    that rule is deliberately narrow (a status transition, a numeric progress ADVANCE,
    a stage change — never a log line) and the narrowness is load-bearing: a wedged
    render that keeps spewing retry lines must still age out. This is the ONE extra
    movement signal, gated on a runner having authored a message that CHANGED.
    Best-effort like the rest of the bridge — a private/renamed field degrades to
    "no extra movement signal", never an exception into a render."""
    try:
        import time
        job.progressed_at = time.time()
        mirror = getattr(store, "mirror", None)
        if mirror is not None:
            mirror.upsert(job.to_dict())
    except Exception:
        logger.debug("job_bridge._mark_progressed failed", exc_info=True)


def on_progress(job_id: str, progress: dict) -> None:
    """running -> running, carrying live per-stage progress into the comms Job so
    GET /llm/jobs shows a render's stage + rolling log tail + an honest stall
    clock (not green-but-empty). Fed by media_bus.set_progress (every runner that
    reports progress) via the ``_bridge`` dispatcher — so this is BEST-EFFORT and
    swallows everything, exactly like the other bridge points.

    The blob is a runner's own free-form dict; we defensively pull only the keys
    the /llm/jobs contract carries (stage/progress/log_tail/message) and update
    just those. A store.update on an unknown/terminal id is a safe no-op (returns
    None), so a progress push that races a terminal write can never resurrect or
    corrupt a finished job. log_tail capping lives in the JobStore (LOG_TAIL_CAP).

    RUNNER-AUTHORED MESSAGE (k91). A runner MAY publish its own human ``message``
    in the blob ("segment 2/14 - step 18/32"); it wins over the ``_summary``
    derived from stage+percent, which cannot say anything a one-stage job hasn't
    already said. It is also treated as FORWARD PROGRESS when it CHANGES — the
    JobStore's own advance rule (status transition / numeric progress advance /
    stage change) is blind to a job whose every tick happens INSIDE one stage, and
    that blindness is what let ``expire_pending_orphans`` retire a live 14-segment
    movie as "wedged". This is deliberately opt-in per blob rather than a blanket
    relaxation: only a runner that authors a movement message gets it, so the
    "a wedged render still spews log lines" guard is untouched."""
    try:
        if not isinstance(progress, dict):
            return
        changes: dict = {}
        if "stage" in progress:
            changes["stage"] = str(progress.get("stage") or "")
        prog = progress.get("progress")
        if prog is not None:
            try:
                changes["progress"] = float(prog)
            except (TypeError, ValueError):
                pass
        lt = progress.get("log_tail")
        if isinstance(lt, (list, tuple)):
            changes["log_tail"] = [str(x) for x in lt]
        authored = progress.get("message")
        authored = str(authored) if isinstance(authored, str) and authored.strip() else None
        if authored is not None:
            changes["message"] = authored
        # Nothing recognizable in the blob -> nothing to mirror (a runner may
        # write a private progress shape we don't surface). No-op, no message.
        if not changes:
            return
        # On a real progress tick, refresh placement so the row upgrades from the
        # enqueue/on_running TEMPLATE hint to the live RESERVATION truth (real
        # worker_id + reserved bytes) once the run's claim is held. Set-only-when-
        # present: a None (no active claim) never blanks the template placement.
        placement = _placement(job_id, None)
        if placement:
            changes["placement"] = placement
        if authored is None:
            msg = _summary(changes.get("stage", ""), changes.get("progress"))
            if msg:
                changes["message"] = msg
        store = _store()
        # Snapshot the PRIOR message before the write, so "did the runner report
        # something new" is answerable afterwards (update overwrites it in place).
        prior_msg = None
        if authored is not None:
            existing = store.get(job_id)
            prior_msg = getattr(existing, "message", None) if existing is not None else None
        job = store.update(job_id, **changes)
        # A CHANGED runner-authored message is forward progress the JobStore's own
        # advance rule cannot see (same stage, and a progress float that may be
        # equal for two consecutive ticks). Bump progressed_at here so the honest
        # stall clock — and therefore expire_pending_orphans — sees the movement the
        # runner just reported. ``job is None`` (unknown/terminal id) is a no-op,
        # exactly like every other bridge write.
        if authored is not None and job is not None and authored != prior_msg:
            _mark_progressed(store, job)
    except Exception:
        logger.warning("job_bridge.on_progress failed for %s", job_id,
                       exc_info=True)


def on_terminal(job_id: str, name: str, status: str, result=None,
                worker: str | None = None, principal: str | None = None) -> None:
    """done|failed|cancelled -> terminal, in the process that ran the job. Marks
    the local record terminal (retained ~600s) so GET /llm/jobs?live=0 shows the
    completed job — the JobStore's own terminal representation, NOT the mirror's
    live_rows() (which excludes terminal). On success the produced clip uri rides
    along as the job message (back-reference to the artifact). ``principal`` (k9)
    is carried onto the record so a finished media job stays attributed."""
    try:
        store = _store()
        comms_status = _STATUS_MAP.get(status, "failed")
        attrib = {"principal": principal} if principal else {}
        # Ensure a local record exists to mark terminal (defensive: normally
        # on_running already anchored it in this same process).
        if store.get(job_id) is None:
            store.create(id=job_id, kind=name, status="processing",
                         worker=worker, transport=_TRANSPORT, model_name=name,
                         **attrib)
        error = _error_dict(result) if comms_status == "failed" else None
        store.finish(job_id, status=comms_status, error=error)
        if comms_status == "done":
            uri = _result_uri(result)
            if uri:
                # Non-status update on a terminal job is allowed (finish/update
                # keep the first terminal status; message is free-form).
                store.update(job_id, message=uri)
    except Exception:
        logger.warning("job_bridge.on_terminal failed for %s (%s -> %s)",
                       job_id, name, status, exc_info=True)
