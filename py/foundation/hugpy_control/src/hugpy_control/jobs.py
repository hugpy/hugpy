"""F5 — one Job schema + JobStore for every unit of work, every transport.

Before this module the tree had TWO half-job systems:

    - flask_app .../schemas/job_schemas.py — download jobs only (progress bytes,
      retry telemetry), invisible to chat.
    - managers/dispatch/activity.py — in-flight chat requests only, ephemeral
      (popped on stream end), no principal/transport/worker metadata, no cancel.

This store unifies them. Lifecycle is frozen:

    pending -> processing -> streaming -> (done | cancelled | failed)

Legacy names still arrive from old callers and old UI expectations
(queued/running/completed); normalize_status() maps them on write and read so
nothing breaks while consumers migrate.

Cancellation lives ON the job: the owning stream attaches a zero-arg cancel
handle (e.g. one that sets its asyncio.Event on the shared runtime loop), and
anyone — an HTTP route, a bus control message — cancels through the store.
The race that matters (DISC-03): cancel-vs-just-finished. Rules:

    - cancel() only *requests*: fires the handle, flags cancel_requested. It
      never force-marks the job cancelled while the stream is live.
    - the first terminal status wins; finish() on an already-terminal job is a
      no-op. A job that completed a microsecond before the cancel arrives
      stays "done".
    - the stream's own teardown converts cancel_requested into status
      "cancelled" (via finish(job_id, "cancelled") or the finish() default
      described below), so "cancelled" always means the resources were
      actually released by the code that held them.

Like its predecessors this is per-process (one gunicorn process, many
threads) and thread-safe. Terminal jobs are retained briefly so UIs can show
"just finished", then pruned — the store never grows without bound.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from hugpy_control.shared import SqliteMirror

CANONICAL_STATUSES = ("pending", "processing", "streaming",
                      "done", "cancelled", "failed", "expired", "discarded")
# `expired` (slice 9): a terminal state for a pending job that was NEVER
# dispatched (no worker, no progress past the orphan threshold) — distinct from
# `cancelled` (an operator/owner asked) and `failed` (it ran and errored). Making
# it terminal means the orphan sweep + the authoritative cancel can retire a
# stuck pending row that no owner will ever finish.
# `discarded` (k121): an admin explicitly dismissed a persistent FAILURE RECORD
# (failed/expired rows are never auto-pruned — operator ruling 2026-08-20).
# Hidden from every view and swept by the mirror prune shortly after.
TERMINAL_STATUSES = frozenset(("done", "cancelled", "failed", "expired",
                               "discarded"))

# The media_bus job kinds bridged in via video_intel.job_bridge (transport
# "media"). snapshot(live_only=False) surfaces a sibling process's *terminal*
# rows for THESE kinds only (via mirror.terminal_rows), so a finished media job
# is visible on every process — not just the one that ran it. Strictly gated to
# media so chat/download terminal cross-process behavior is unchanged (their
# terminal rows still show only via each process's local ~600s retention).
MEDIA_KINDS = ("crop", "frame_extract", "generate_scene",
               "generate_movie", "studio_i2v", "generate_studio_movie")

# Old JOBSTATUS names (and activity.py's view states) -> canonical.
_LEGACY_STATUS = {
    "queued": "pending",
    "waiting": "pending",
    "running": "processing",
    "active": "streaming",
    "completed": "done",
}
# Canonical -> the old JOBSTATUS wire names, for surfaces (the download UI's
# /jobs contract) that still speak them. streaming has no legacy analogue;
# it reads as running there.
LEGACY_FOR_CANONICAL = {
    "pending": "queued",
    "processing": "running",
    "streaming": "running",
    "done": "completed",
}


def normalize_status(status: Any) -> str:
    s = str(status or "pending").strip().lower()
    s = _LEGACY_STATUS.get(s, s)
    return s if s in CANONICAL_STATUSES else "pending"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# HONEST stalled (media/relay live-progress): a job in an active/running state
# that has made no FORWARD progress for this long reads as stuck, not green.
# "Forward progress" is tracked by Job.progressed_at (bumped on a token, a
# progress advance, a stage change, a status transition) — NOT updated_at, which
# also bumps on a log-tail-only write. Computed at serialize/snapshot time so a
# wedged render turns stalled without any writer having to notice (a stored bool
# would go stale). Only "processing"/"streaming" can be stalled — a pending job
# waiting its turn in the queue is starved, not wedged.
_STALL_ACTIVE = frozenset(("processing", "streaming"))
# The last-N log lines a bridged job carries on /llm/jobs (newest-last). Caps the
# wire size (and the mirror row's JSON) so a chatty render can't bloat the view.
LOG_TAIL_CAP = 40


def _stall_seconds() -> float:
    """The forward-progress silence (seconds) after which an active job reads as
    stalled. Env-overridable (HUGPY_JOB_STALL_SECONDS); default 90s. Read at
    compute time so an operator can retune it without a restart, and a bad value
    degrades to the default rather than raising into a serialize/snapshot path."""
    raw = (os.environ.get("HUGPY_JOB_STALL_SECONDS") or "").strip()
    if not raw:
        return 90.0
    try:
        v = float(raw)
        return v if v > 0 else 90.0
    except ValueError:
        return 90.0


def _stalled_expiry_seconds() -> float:
    """The forward-progress silence after which a WEDGED ACTIVE job is RETIRED,
    not merely labelled stalled. Env-overridable
    (``HUGPY_JOB_STALLED_EXPIRY_SECONDS``); default 900s.

    Deliberately an order of magnitude above ``_stall_seconds`` (90s): saying a
    row LOOKS stalled is cheap and reversible, ending someone's call is not. 900
    also lines up with the cold-hold hard ceiling — past that, central has
    already given up on the request, so a job row still claiming to be working
    is describing nothing that is running."""
    raw = (os.environ.get("HUGPY_JOB_STALLED_EXPIRY_SECONDS") or "").strip()
    if not raw:
        return 900.0
    try:
        v = float(raw)
        return v if v > 0 else 900.0
    except ValueError:
        return 900.0


def _compute_stalled(status: Any, progressed_at: Any, now: float) -> bool:
    """True when *status* is active/running AND *progressed_at* (epoch seconds) is
    older than the stall threshold. Fail-open: a missing/garbage progressed_at is
    treated as "just now" (never a false stall), and this never raises — it runs
    on every /llm/jobs row."""
    try:
        if normalize_status(status) not in _STALL_ACTIVE:
            return False
        if progressed_at is None:
            return False
        return (now - float(progressed_at)) > _stall_seconds()
    except (TypeError, ValueError):
        return False


@dataclass
class JobError:
    """Error-as-data. A failed job carries one of these, never a bare string
    (str(exc) leaks internals and can't be routed on)."""
    code: str = "error"
    message: str = ""
    detail: Optional[dict] = None
    # Field-aligned with video_intel.result_schema.JobError (which serializes on
    # /video/jobs and REQUIRES retryable). Here it is nullable and additive: chat
    # /download rows serialize "retryable": null; bridged media rows carry the
    # real bool. Deliberately NOT required — never break the non-media callers.
    retryable: Optional[bool] = None

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        # Emitted ALWAYS (nullable) so every /llm/jobs error object carries the
        # field — a real bool for media, null for chat/download.
        out["retryable"] = self.retryable
        return out

    @classmethod
    def coerce(cls, err: Any) -> "JobError | None":
        if err is None or isinstance(err, JobError):
            return err
        if isinstance(err, dict):
            return cls(code=str(err.get("code") or "error"),
                       message=str(err.get("message") or ""),
                       detail=err.get("detail"),
                       retryable=err.get("retryable"))
        if isinstance(err, BaseException):
            return cls(code=type(err).__name__, message=str(err))
        return cls(message=str(err))


@dataclass
class Job:
    id: str
    model_key: str = ""
    status: str = "pending"
    kind: str = "chat"                    # chat | v1 | discord | cli | download | ...
    # Addressing / attribution (F5.2). All optional until F2 lands a real
    # Principal — carry them now so nothing has to re-plumb later.
    principal: Optional[str] = None       # who asked
    transport: Optional[str] = None       # web | v1 | discord | cli | worker
    channel: Optional[str] = None         # discord channel id, web session, ...
    worker: Optional[str] = None          # worker name/id serving it
    slot: Optional[str] = None            # slot id on that worker
    model_name: Optional[str] = None      # display name (model_key is the key)
    message: str = ""
    # Human-readable input attached to an in-flight inference job. This is
    # available to the queue UI so operators can identify what is waiting.
    prompt: str = ""
    # Original incoming request body, shown by the Calls panel's row expander.
    request: Optional[dict] = None
    error: Optional[JobError] = None
    # A TYPED failure code the console can branch on — "gated_repo" | "auth" |
    # "not_found" | "no_space" (downloader/engine.classify_download_error).
    # `error` keeps the raw detail; this says what KIND of wrong it is, so the
    # UI can offer "save an HF token" instead of printing a 401 traceback.
    # Omit-when-unset: every non-download row's wire shape is unchanged.
    error_reason: Optional[str] = None
    # WHERE a media/video job physically executes — {source, host, worker_id, gpu,
    # process, reserved_bytes} (omit-when-unset). Stamped by the media-bus job
    # bridge (video_intel.placement); None for every non-media job, and omitted
    # from to_dict() when None so no chat/download row's wire shape changes.
    placement: Optional[dict] = None
    # WHAT a queued job needs in order to RUN, for a job that will be executed by
    # a DIFFERENT process than the one that created it (the download daemon).
    # `_model` below is runtime-only, so a job handed across the process boundary
    # through the mirror would arrive with no idea what to download; `payload`
    # is the serialized half — e.g. {"model": {...}, "total_bytes": N}. Omitted
    # from to_dict() when None, so no existing row's wire shape changes.
    payload: Optional[dict] = None
    # Live-stream telemetry (was activity.py).
    tokens: int = 0
    started_ts: float = field(default_factory=time.time)
    first_output_ts: Optional[float] = None
    ended_ts: Optional[float] = None
    cancel_requested: bool = False
    # Download telemetry (was the flask job_schemas Job) — unused for chat.
    progress: float = 0.0                 # 0.0–1.0
    total_bytes: Optional[int] = None
    downloaded_bytes: Optional[int] = None
    attempt: int = 0
    max_attempts: int = 0
    stalled: bool = False
    bytes_per_second: Optional[float] = None
    # Live per-stage progress carried from a bridged execution plane (media_bus
    # runners via job_bridge.on_progress) — a wedged render surfaces its stage +
    # rolling log tail in /llm/jobs instead of reading green-but-empty. Both
    # default empty (backward-compatible; chat/download rows carry "" and []).
    stage: str = ""
    log_tail: list = field(default_factory=list)
    # Last time this job made FORWARD progress (epoch seconds) — the honest
    # stalled clock (see _compute_stalled). Distinct from updated_at, which also
    # bumps on a log-tail-only write; this bumps only on real advancement.
    progressed_at: float = field(default_factory=time.time)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    # Runtime-only, never serialized: download subprocess, resolved model dict
    # (kept so a manual retry can resume without re-resolving), cancel handle,
    # and the last time this job synced to the cross-process mirror.
    _proc: Any = field(default=None, repr=False, compare=False)
    _model: Optional[dict] = field(default=None, repr=False, compare=False)
    _cancel: Optional[Callable[[], None]] = field(default=None, repr=False,
                                                  compare=False)
    _last_sync: float = field(default=0.0, repr=False, compare=False)

    @property
    def terminal(self) -> bool:
        return normalize_status(self.status) in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot — superset of both predecessors' shapes."""
        now = time.time()
        ended = self.ended_ts or now
        d = {
            "id": self.id,
            "model_key": self.model_key,
            "status": normalize_status(self.status),
            "kind": self.kind,
            "principal": self.principal,
            "transport": self.transport,
            "channel": self.channel,
            "worker": self.worker,
            "slot": self.slot,
            "model": self.model_name or self.model_key or "?",
            "message": self.message,
            "prompt": self.prompt,
            "error": self.error.to_dict() if self.error else None,
            "tokens": self.tokens,
            "elapsed": round(ended - self.started_ts, 1),
            # seconds spent waiting before the first output (queue time).
            "wait": round((self.first_output_ts or ended) - self.started_ts, 1),
            "cancel_requested": self.cancel_requested,
            "progress": round(self.progress, 4),
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            # HONEST stalled: an explicit stall (a download writer sets self.stalled)
            # OR a computed forward-progress silence for an active job. Computed
            # HERE so it is always fresh — a stored-only bool goes stale the moment
            # the render wedges. progressed_at rides along so a sibling process
            # reading this from the mirror can recompute it just as freshly.
            "stalled": bool(self.stalled)
            or _compute_stalled(self.status, self.progressed_at, now),
            "bytes_per_second": self.bytes_per_second,
            "stage": self.stage or "",
            # Cap defensively at serialize time too (a writer that stored more never
            # bloats the wire / the mirror row).
            "log_tail": list(self.log_tail or [])[-LOG_TAIL_CAP:],
            "progressed_at": self.progressed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        # Omit-when-unset: only media/video jobs carry a placement — every other
        # row's shape is byte-identical to before this field existed.
        if self.placement is not None:
            d["placement"] = self.placement
        if self.payload is not None:
            d["payload"] = self.payload
        if self.request is not None:
            d["request"] = self.request
        if self.error_reason:
            d["error_reason"] = self.error_reason
        return d

    def to_legacy_dict(self) -> dict[str, Any]:
        """The pre-comms /jobs wire shape: legacy status names, error as a
        plain string. The download UI (ModelTable) reads exactly this; new
        surfaces should read to_dict() and canonical states instead."""
        return to_legacy(self.to_dict())


def to_legacy(d: dict[str, Any]) -> dict[str, Any]:
    """Job.to_dict() shape -> the legacy /jobs wire shape (queued/running/
    completed, error as a plain string).

    Module-level, not just a Job method, because rows that arrive from the
    cross-process MIRROR are plain dicts — the process serving GET /jobs does
    not hold a Job object for a download another process owns. Same output as
    Job.to_legacy_dict(), which now routes through here so the two can never
    drift."""
    d = dict(d or {})
    d["status"] = LEGACY_FOR_CANONICAL.get(d.get("status"), d.get("status"))
    if isinstance(d.get("error"), dict):
        d["error"] = d["error"].get("message") or d["error"].get("code")
    return d


def _default_mirror() -> Optional[SqliteMirror]:
    """Cross-process mirror for the module singleton. Off-switch:
    HUGPY_COMMS_DB=off. Bare JobStore() instances (tests, embedders) stay
    purely in-process unless given a mirror explicitly."""
    env = (os.environ.get("HUGPY_COMMS_DB") or "").strip().lower()
    if env in ("off", "none", "0", "disabled"):
        return None
    try:
        return SqliteMirror()
    except Exception:
        return None


class JobStore:
    """Thread-safe, per-process, with an optional cross-process mirror.

    ``on_change`` (set once at wiring time) is called with (job, prior_status)
    after every status transition, outside no locks the caller can see — the
    bus adapter uses it to publish job.* events without this module importing
    the bus (keeps comms.jobs importable by absolutely everything).

    ``mirror`` (comms.shared.SqliteMirror) makes cancel and queue views
    correct when the service runs multiple processes (gunicorn --workers N):
    transitions are mirrored, a cancel for a job another process owns raises
    a flag on the shared row, and the owner notices it on its next token.
    Everything mirror-related is best-effort — no mirror, or a broken one,
    degrades to per-process behavior.
    """

    def __init__(self, *, retain_terminal: int = 100,
                 retain_secs: float = 600.0,
                 mirror: Optional[SqliteMirror] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._retain_terminal = retain_terminal
        self._retain_secs = retain_secs
        self.on_change: Optional[Callable[[Job, str], None]] = None
        self.mirror = mirror
        self._last_mirror_prune = 0.0
        self._watcher: Optional[threading.Thread] = None

    # -- creation ----------------------------------------------------------
    def create(self, model_key: str = "", *, id: Optional[str] = None,
               kind: str = "chat", **meta: Any) -> Job:
        """Old signature preserved: download callers do create(model_key).
        New callers pass id=request_id so the job id IS the request id and
        every layer (SSE, bus, cancel route) correlates on one key."""
        job = Job(id=id or str(uuid.uuid4()), model_key=model_key or "",
                  kind=kind)
        for k, v in meta.items():
            if hasattr(job, k):
                setattr(job, k, v)
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
        try:  # CALL-LOG-20260910
            from hugpy_control import calllog as _calllog
            _calllog.record("start", job)
        except Exception:  # noqa: BLE001
            pass
        self._emit(job, "")
        self._mirror_upsert(job)
        self._maybe_prune_mirror()
        self._ensure_watcher()
        return job

    # -- cross-process work queue -------------------------------------------
    #
    # ENQUEUE / CLAIM: a job whose work runs in ANOTHER process (the download
    # daemon). The creating process must NOT keep an authoritative local record
    # — snapshot() deliberately prefers a local row over the mirror row of the
    # same id ("the in-memory record is the truth for them"), so a leftover
    # `pending` row in the API would permanently MASK the daemon's live progress.
    # enqueue() therefore mirrors the job and then drops the local copy: from
    # that instant the mirror row is the only truth, and every reader (including
    # the enqueuer) sees the owner's state.

    def enqueue(self, model_key: str = "", *, kind: str = "download",
                **meta: Any) -> Job:
        """Create a job for ANOTHER process to run, and disown it locally.

        Returns the Job as created (so the caller can serialize the accepted
        response); the store does not retain it — read it back via snapshot()/
        get_dict(), which merge the mirror."""
        job = self.create(model_key, kind=kind, **meta)
        self.detach(job.id)
        return job

    def detach(self, job_id: str) -> None:
        """Forget a job LOCALLY without touching the mirror row. Used to disown
        an enqueued job (see enqueue) — never to retire one."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def claim_next(self, kinds: set | tuple, owner: str) -> Optional[dict]:
        """Atomically take the oldest queued job of *kinds* for *owner*, through
        the mirror's compare-and-set. Returns its dict snapshot, or None when
        there is nothing waiting (or no mirror — with no mirror there is no
        cross-process queue at all)."""
        if self.mirror is None:
            return None
        try:
            return self.mirror.claim_next(tuple(kinds), owner)
        except Exception:
            return None

    def adopt_stale(self, kinds: set | tuple, owner: str,
                    message: str = "") -> list[str]:
        """Re-queue work a DEAD owner left mid-flight (daemon fail-over)."""
        if self.mirror is None:
            return []
        try:
            return self.mirror.adopt_stale(tuple(kinds), owner, message)
        except Exception:
            return []

    def requeue(self, job_id: str, message: str = "",
                kinds: set | tuple = ()) -> bool:
        """Put a job back on the claim queue (the cross-process retry). Clears
        any local record too, so the requeuing process cannot mask the mirror
        row it just reset."""
        if self.mirror is None:
            return False
        try:
            ok = self.mirror.requeue(job_id, message=message,
                                     kinds=tuple(kinds))
        except Exception:
            return False
        if ok:
            self.detach(job_id)
        return ok

    # -- mutation ----------------------------------------------------------
    def update(self, job_id: str, **changes: Any) -> Optional[Job]:
        prior = None
        resurrected = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            prior = normalize_status(job.status)
            if "status" in changes:
                new = normalize_status(changes["status"])
                if job.terminal and new in TERMINAL_STATUSES and new != prior:
                    # First terminal state wins (cancel-vs-just-finished race):
                    # done never becomes cancelled after the fact.
                    changes = {k: v for k, v in changes.items() if k != "status"}
                else:
                    changes["status"] = new
                    if new in TERMINAL_STATUSES:
                        if job.ended_ts is None:
                            job.ended_ts = time.time()
                    elif job.terminal:
                        # Deliberate terminal->live resurrection (download
                        # retry reuses the job id): reset the run telemetry.
                        job.ended_ts = None
                        job.cancel_requested = False
                        job.first_output_ts = None
                        resurrected = True
            if "error" in changes:
                changes["error"] = JobError.coerce(changes["error"])
            # Forward-progress detection (the honest stalled clock): a status
            # transition, a numeric progress advance, or a stage change is real
            # movement and resets progressed_at. A log_tail-only write is NOT —
            # a wedged render can still spew retry lines, so logs update but the
            # stall clock keeps ticking. Bumping updated_at stays UNCHANGED (it
            # marks any field write); progressed_at is the movement-only mirror.
            advanced = ("status" in changes
                        and normalize_status(changes["status"]) != prior)
            try:
                if "progress" in changes \
                        and float(changes["progress"]) > float(job.progress):
                    advanced = True
            except (TypeError, ValueError):
                pass
            if "stage" in changes and changes.get("stage") != job.stage:
                advanced = True
            # Cap the rolling log tail on write so neither the wire nor the mirror
            # row's JSON can be bloated by a chatty producer.
            if "log_tail" in changes:
                lt = changes["log_tail"]
                changes["log_tail"] = list(lt or [])[-LOG_TAIL_CAP:]
            for k, v in changes.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = _utcnow_iso()
            if advanced:
                job.progressed_at = time.time()
        if normalize_status(job.status) != prior:
            self._emit(job, prior)
        if resurrected and self.mirror is not None:
            # upsert never lowers the shared cancel flag; a fresh run must.
            try:
                self.mirror.clear_cancel(job_id)
            except Exception:
                pass
        self._mirror_upsert(job)
        return job

    def on_output(self, job_id: str, n: int = 1) -> None:
        """First output flips the job to streaming; every output bumps the
        counter. One lock acquisition per token — same cost activity.py paid.

        The mirror sync here is write-only and throttled (~1/sec per job) so
        sibling queue views stay fresh without taxing the token hot path;
        noticing a sibling's cancel flag is the watcher thread's job."""
        emit = False
        sync = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.terminal:
                return
            if job.first_output_ts is None:
                job.first_output_ts = time.time()
            if normalize_status(job.status) != "streaming":
                prior = normalize_status(job.status)
                job.status = "streaming"
                emit = True
            job.tokens += n
            # A token is forward progress — reset the honest stall clock so a
            # long, healthy stream never reads as stalled (on_output deliberately
            # does NOT touch updated_at, so progressed_at is the only movement
            # marker for the token hot path).
            job.progressed_at = time.time()
            if self.mirror is not None and \
                    time.time() - job._last_sync >= 1.0:
                sync = True
        if emit:
            self._emit(job, prior)
        if sync:
            self._mirror_upsert(job)

    def finish(self, job_id: str, status: Optional[str] = None,
               error: Any = None) -> Optional[Job]:
        """Terminal marking, from the code that actually owned the stream.
        With no explicit status: failed if an error is given, cancelled if a
        cancel was requested, else done. No-op if already terminal."""
        job = self.get(job_id)
        if job is None:
            return None
        if status is None:
            if error is not None:
                status = "failed"
            elif job.cancel_requested:
                status = "cancelled"
            else:
                status = "done"
        changes: dict[str, Any] = {"status": status}
        if error is not None:
            changes["error"] = error
        _done = self.update(job_id, **changes)
        try:  # CALL-LOG-20260910
            from hugpy_control import calllog as _calllog
            _calllog.record("end", _done if _done is not None else job)
        except Exception:  # noqa: BLE001
            pass
        return _done

    # -- cancel plane ------------------------------------------------------
    def attach_cancel(self, job_id: str, handle: Callable[[], None]) -> None:
        fire = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job._cancel = handle
                # A cancel that raced in before the stream attached its handle
                # must still take effect — fire immediately.
                if job.cancel_requested:
                    fire, job._cancel = job._cancel, None
        # Same race, cross-process flavor: a sibling flagged the mirror row
        # before we attached. Fires the handle we just stored via cancel().
        if (fire is None and job is not None and not job.cancel_requested
                and self.mirror is not None):
            try:
                if self.mirror.cancel_requested(job_id):
                    self.cancel(job_id, reason="cancelled by sibling process")
                    return
            except Exception:
                pass
        self._fire(fire)

    def cancel(self, job_id: str, reason: str = "") -> bool:
        """Back-compat bool wrapper. Returns True when the cancel took effect
        (relayed to a live owner OR force-marked terminal in the store). See
        cancel_authoritative for the honest {cancelled, mode} result."""
        return self.cancel_authoritative(job_id, reason)["cancelled"]

    def cancel_authoritative(self, job_id: str, reason: str = "") -> dict:
        """AUTHORITATIVE, HONEST cancel (slice 9, defect 1).

        Returns ``{"cancelled": bool, "mode": "relayed"|"store"|"noop",
        "status": str|None}``.

          * A job with a LIVE OWNER (a local stream has attached a cancel handle,
            OR a sibling process holds a live mirror row) -> RELAY: request cancel,
            fire the handle, raise the shared flag. The owner's teardown marks it
            terminal. mode="relayed".
          * A job with NO live owner (pending/worker-null, or a local row that is
            terminal/handle-less AND no live sibling) -> STORE: force-mark it
            terminal `cancelled` DIRECTLY, persisted via the mirror, surviving
            restarts. This is the fix for the immortal pending row that answered
            cancelled:true forever while nothing changed. mode="store".
          * Nothing anywhere claims the id -> {cancelled:false, mode:"noop"}. A
            cancel that changes nothing must NOT lie cancelled:true (the cardinal
            sin this slice closes).

        "Live owner" = a cancel handle attached HERE (a real streaming producer in
        THIS process). When one exists we relay only and let its teardown write
        the terminal status (first-terminal-wins). Otherwise we ALWAYS raise the
        shared flag (so a live sibling's watcher can still act — the cross-process
        relay) AND force-terminal the row, so an owner-less immortal job goes
        terminal immediately instead of answering cancelled:true forever. When a
        sibling DOES own it, its watcher fires the handle and its finish() no-ops
        against the already-terminal row (first-terminal-wins) — the handle still
        runs, so cross-process teardown is preserved."""
        fire = None
        has_local_owner = False          # a real producer attached a handle HERE
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and not job.terminal:
                has_local_owner = job._cancel is not None
                job.cancel_requested = True
                if reason:
                    job.message = reason
                job.updated_at = _utcnow_iso()
                fire, job._cancel = job._cancel, None   # fire at most once
        self._fire(fire)

        if has_local_owner:
            # A live LOCAL stream is being torn down; its teardown writes the
            # terminal status (first-terminal-wins keeps a just-finished job
            # honest). Relay only — do NOT force-terminal out from under it.
            if self.mirror is not None:
                try:
                    self.mirror.request_cancel(job_id)
                except Exception:
                    pass
                self._mirror_upsert(job)
            return {"cancelled": True, "mode": "relayed",
                    "status": normalize_status(job.status) if job else None}

        # No live LOCAL owner. Raise the shared cancel flag FIRST so a live
        # sibling's watcher (if any) still fires its handle and tears its stream
        # down — the existing cross-process relay is preserved. Then force-mark
        # the row terminal so an owner-less immortal job (the operator's case) is
        # retired NOW, not left for a watcher that will never come. First-terminal-
        # wins means a genuine sibling finish() after this is a no-op — the handle
        # still ran, so nothing cross-process is lost.
        if self.mirror is not None:
            try:
                self.mirror.request_cancel(job_id)
            except Exception:
                pass
        marked = self._force_cancel_terminal(job_id, reason)
        if marked is not None:
            return {"cancelled": True, "mode": "store", "status": "cancelled"}

        # Nothing anywhere claims this id — do NOT claim success.
        return {"cancelled": False, "mode": "noop", "status": None}

    def _force_cancel_terminal(self, job_id: str, reason: str = "") -> Optional[str]:
        """Mark an owner-less job terminal `cancelled`, in the store AND the
        mirror (persisted). Handles a LOCAL row and — critically for the immortal
        pending case — a MIRROR-ONLY row this process never held in memory.
        Returns "cancelled" if a row was retired, else None (nothing to cancel).
        Idempotent: an already-terminal row is left as-is (first-terminal wins)."""
        # Local row present?
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                if job.terminal:
                    return None            # already terminal — nothing to change
                job.status = "cancelled"
                job.cancel_requested = True
                if reason:
                    job.message = reason
                if job.ended_ts is None:
                    job.ended_ts = time.time()
                job.updated_at = _utcnow_iso()
                prior = "pending"
        if job is not None:
            self._emit(job, prior)
            self._mirror_upsert(job)
            return "cancelled"
        # No local row — force the MIRROR row terminal directly through the store
        # layer (never a raw sqlite side-channel). Only when a LIVE row exists to
        # retire; a truly-unknown id returns None so the caller reports noop.
        if self.mirror is not None:
            try:
                if self.mirror.force_terminal(job_id, "cancelled",
                                               reason or "cancelled (no live owner)"):
                    return "cancelled"
            except Exception:
                pass
        return None

    def discard(self, job_id: str) -> bool:
        """ADMIN DISCARD of a terminal failure record (k121). Flips the row to
        `discarded` — the only way a persistent failed/expired record leaves the
        queue view — locally AND in the mirror (override_terminal: this verb
        alone may overwrite a terminal status). Never touches a live job; the
        caller gates on terminal status (cancel is the verb for live rows)."""
        removed = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                if not job.terminal:
                    return False
                self._jobs.pop(job_id, None)
                removed = True
        if self.mirror is not None:
            try:
                if self.mirror.force_terminal(job_id, "discarded",
                                              "discarded by operator",
                                              override_terminal=True):
                    removed = True
            except Exception:
                pass
        return removed

    @staticmethod
    def _fire(handle: Optional[Callable[[], None]]) -> None:
        # Always outside the store lock: a handle may touch the store.
        if handle is not None:
            try:
                handle()
            except Exception:
                pass

    # -- views -------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_dict(self, job_id: str) -> Optional[dict]:
        """One job's dict snapshot, MIRROR-MERGED: the local record if we hold
        it, else the sibling-owned mirror row (live OR terminal). get() can only
        ever answer for jobs this process owns — a route serving GET /jobs/<id>
        for a download the daemon owns must use this."""
        job = self.get(job_id)
        if job is not None:
            return job.to_dict()
        if self.mirror is None:
            return None
        try:
            d = self.mirror.row(job_id)
        except Exception:
            return None
        if d:
            now = time.time()
            d["stalled"] = bool(d.get("stalled")) or _compute_stalled(
                d.get("status"), d.get("progressed_at"), now)
        return d

    def snapshot(self, *, kinds: Optional[set[str]] = None,
                 live_only: bool = True,
                 terminal_kinds: Optional[tuple] = None) -> list[dict]:
        """JSON-safe view for queue UIs. Waiting first, then longest-running —
        the exact ordering the console's activity view has always shown.
        With a mirror, live jobs owned by sibling processes are merged in
        (their snapshots are at most ~1s stale); local records win on id."""
        with self._lock:
            jobs = [j for j in self._jobs.values()
                    if (kinds is None or j.kind in kinds)
                    and (not live_only or not j.terminal)]
            local_ids = set(self._jobs)
        out = [j.to_dict() for j in jobs]
        if self.mirror is not None:
            now = time.time()
            try:
                for d in self.mirror.live_rows():
                    # Skip our own rows — the in-memory record is the truth
                    # for them (including ones the mirror hasn't caught up on).
                    if d.get("id") in local_ids:
                        continue
                    if kinds is not None and d.get("kind") not in kinds:
                        continue
                    # A sibling's row carries the `stalled` value computed WHEN it
                    # was last upserted — stale by now. Recompute it FRESH from the
                    # progressed_at epoch that rides in the row (OR the sibling's
                    # own explicit stall) so a wedged cross-process render reads as
                    # stalled here too, without the owner having to re-upsert.
                    d["stalled"] = bool(d.get("stalled")) or _compute_stalled(
                        d.get("status"), d.get("progressed_at"), now)
                    out.append(d)
            except Exception:
                pass
            # Cross-process TERMINAL media rows (media-gated). live_rows() hides
            # terminal rows by design; this additive merge surfaces a sibling's
            # finished media job on the full (live_only=False) view — and ONLY
            # for MEDIA_KINDS, so chat/download terminal behavior is unchanged.
            if not live_only:
                # Which kinds' terminal rows cross the process boundary. Default
                # MEDIA_KINDS (unchanged). The download manager passes
                # ("download",) because completion now happens in the DAEMON:
                # without this the console would watch a download climb to 99%
                # and then have the row vanish at the finish line.
                tk = MEDIA_KINDS if terminal_kinds is None else tuple(terminal_kinds)
                try:
                    seen = {d.get("id") for d in out}   # local + live-merged ids
                    for d in self.mirror.terminal_rows(tk):
                        if d.get("id") in seen:
                            continue
                        if kinds is not None and d.get("kind") not in kinds:
                            continue
                        out.append(d)
                        seen.add(d.get("id"))
                except Exception:
                    pass
        out.sort(key=lambda d: (d["status"] not in ("pending", "processing"),
                                -d.get("elapsed", 0)))
        return out

    def counts(self, *, kinds: Optional[set[str]] = None) -> dict:
        snap = self.snapshot(kinds=kinds)
        waiting = sum(1 for d in snap
                      if d["status"] in ("pending", "processing"))
        active = sum(1 for d in snap if d["status"] == "streaming")
        return {"waiting": waiting, "active": active, "total": len(snap)}

    def expire_pending_orphans(self) -> list[str]:
        """Expire never-dispatched pending jobs (slice 9, defect 2). Event-driven
        (called from the /llm/jobs view + heartbeat ingest — NO timer thread).
        Expires LOCAL pending rows with no worker and stale progressed_at, and —
        via the mirror — mirror-only rows (the immortal cross-process case). Ages
        on progressed_at, never `updated` (no view-driven resurrection). Returns
        the ids retired this pass.

        ALSO expires WEDGED ACTIVE jobs — the 2026-07-28 defect. A chat job sat
        ``status=processing stage=provision`` for 45+ minutes with elapsed frozen
        at 0.1 and "downloading … (?/?)", and nothing retired it: this sweep
        only ever looked at rows that were BOTH ``pending`` AND worker-less, and
        that job was neither. It rendered as stalled (``_compute_stalled`` was
        telling the truth the whole time) and stayed forever, because being
        honestly labelled stalled and being RETIRED were two different things
        with only the first implemented. The operator ended it by hand.

        The guard against retiring healthy work is ``progressed_at`` itself: it
        advances only on real movement (a status transition, a numeric progress
        ADVANCE, a stage change, a token — never a log line, never a plain field
        write), so a long download that is actually downloading keeps resetting
        this clock and is never touched. The threshold is deliberately far above
        the display-stall clock: 90s makes a row SAY stalled, expiry ENDS a call,
        and those two decisions should not share a number.

        Best-effort: never raises into a view."""
        from hugpy_control.shared import (
            CLAIM_QUEUE_KINDS,
            _claim_queue_expiry_seconds,
            _pending_expiry_seconds,
        )
        expired: list[str] = []
        try:
            cutoff = time.time() - _pending_expiry_seconds()
            # k119: a claim-queue kind (download) is dispatched by a DAEMON
            # taking a claim, never by a worker being assigned, so the
            # worker-dispatch cutoff retired every queued download after 30
            # minutes under the message "no capable worker" — a lie, and a
            # silent loss of the operator's request. Own clock, own message.
            claim_cutoff = time.time() - _claim_queue_expiry_seconds()
            wedge_cutoff = time.time() - _stalled_expiry_seconds()
            msg = ("never dispatched — model unresolvable or no capable worker "
                   "(auto-expired)")
            claim_msg = ("never claimed by the downloader service — it was not "
                         "running, or its queue was unreachable (auto-expired). "
                         "Re-add the model once hugpy-downloader is up.")
            wedged: list[tuple] = []
            with self._lock:
                for jid, job in list(self._jobs.items()):
                    status = normalize_status(job.status)
                    claimable = str(job.kind or "") in CLAIM_QUEUE_KINDS
                    pending_cutoff = claim_cutoff if claimable else cutoff
                    if (status == "pending"
                            and not job.worker
                            and float(job.progressed_at or 0) < pending_cutoff):
                        job.status = "expired"
                        job.message = claim_msg if claimable else msg
                        if job.ended_ts is None:
                            job.ended_ts = time.time()
                        job.updated_at = _utcnow_iso()
                        expired.append(jid)
                        continue
                    # WEDGED ACTIVE: dispatched (or not) but demonstrably not
                    # moving. Named honestly — the stage it died in and how long
                    # it sat there, so the row explains itself in the queue view.
                    if (status in _STALL_ACTIVE
                            and float(job.progressed_at or 0) < wedge_cutoff):
                        idle = int(time.time() - float(job.progressed_at or 0))
                        where = f" in stage {job.stage!r}" if job.stage else ""
                        job.status = "expired"
                        job.message = (
                            f"no forward progress{where} for {idle}s — the job "
                            f"wedged and was auto-expired (last message: "
                            f"{(job.message or 'none')[:160]})")
                        if job.ended_ts is None:
                            job.ended_ts = time.time()
                        job.updated_at = _utcnow_iso()
                        wedged.append((jid, status))
            for jid in expired:
                job = self._jobs.get(jid)
                if job is not None:
                    self._emit(job, "pending")
                    self._mirror_upsert(job)
            for jid, prior in wedged:
                job = self._jobs.get(jid)
                if job is not None:
                    self._emit(job, prior)
                    self._mirror_upsert(job)
                    # Inside the guard on purpose: a jid whose job vanished
                    # between the locked pass and here got no emit and no mirror
                    # write, and reporting it as expired would claim work that
                    # never happened.
                    expired.append(jid)
        except Exception:
            pass
        # Mirror-only pending rows (a row no process holds in memory — the exact
        # shape of the operator's immortal flux2 job after a restart).
        if self.mirror is not None:
            try:
                mirror_expired = self.mirror.expire_pending_orphans()
                expired.extend(x for x in mirror_expired if x not in expired)
            except Exception:
                pass
        return expired

    # -- retention ---------------------------------------------------------
    def _prune_locked(self) -> None:
        now = time.time()
        terminal = [(j.ended_ts or now, jid) for jid, j in self._jobs.items()
                    if j.terminal]
        stale = [jid for ended, jid in terminal
                 if now - ended > self._retain_secs]
        terminal.sort()
        overflow = [jid for _, jid in
                    terminal[:max(0, len(terminal) - self._retain_terminal)]]
        for jid in {*stale, *overflow}:
            self._jobs.pop(jid, None)

    def _emit(self, job: Job, prior: str) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(job, prior)
        except Exception:
            pass

    # -- mirror plumbing (all best-effort) -----------------------------------
    def _mirror_upsert(self, job: Optional[Job]) -> None:
        if self.mirror is None or job is None:
            return
        try:
            job._last_sync = time.time()
            self.mirror.upsert(job.to_dict())
        except Exception:
            pass

    def _maybe_prune_mirror(self) -> None:
        if self.mirror is None:
            return
        now = time.time()
        if now - self._last_mirror_prune < 60.0:
            return
        self._last_mirror_prune = now
        try:
            self.mirror.prune()
        except Exception:
            pass

    def _ensure_watcher(self) -> None:
        """Owner-side cancel watcher: a sibling process can only raise the
        shared flag; the process holding the stream's cancel handle must fire
        it. A 1s tick covers the waiting/provisioning phase too — exactly
        when users cancel most, and when no tokens flow to piggyback on."""
        if self.mirror is None or self._watcher is not None:
            return
        with self._lock:
            if self._watcher is not None:
                return
            t = threading.Thread(target=self._watch_mirror,
                                 name="comms-mirror-watch", daemon=True)
            self._watcher = t
        t.start()

    def _watch_mirror(self) -> None:
        while True:
            time.sleep(1.0)
            try:
                with self._lock:
                    candidates = [j.id for j in self._jobs.values()
                                  if not j.terminal and not j.cancel_requested]
                if not candidates:
                    continue
                flagged = self.mirror.flagged_ids()
                for jid in candidates:
                    if jid in flagged:
                        self.cancel(jid, reason="cancelled by sibling process")
            except Exception:
                pass


job_store = JobStore(mirror=_default_mirror())


# FORK SAFETY (incident 2026-09-24). The store keeps no sqlite connection at
# module scope (the mirror opens per-call), but it DOES lazily start an owner-
# side cancel-watcher thread and the mirror caches an `_initialized` flag. A
# thread does not survive fork, so a child that inherited a non-None `_watcher`
# would never re-arm its own watcher; and the child must re-create the mirror
# schema on its own handle. Reset both in a forked child. Belt-and-braces behind
# the real fix (app built in the gunicorn worker). Registered once; never raises.
def _reset_after_fork_in_child() -> None:
    try:
        job_store._watcher = None
        if job_store.mirror is not None:
            job_store.mirror._initialized = False
    except Exception:  # noqa: BLE001 — an at-fork hook must never raise
        pass


try:
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)
except (AttributeError, ValueError):  # non-POSIX / interpreter without the hook
    pass
