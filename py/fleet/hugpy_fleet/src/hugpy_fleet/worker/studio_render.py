"""Worker-side studio render endpoints (GPU offload, option a).

A studio render (``produce_clip`` -> content-addressed clip) runs HERE, on a GPU
worker, while the central VM keeps the whole control plane (media_jobs.db stays
single-writer on central). Central's studio_i2v bus adapter
(``video_intel/runners/studio_i2v.py``) delegates a REAL-model render to
``POST /studio/render``, polls ``GET /studio/render/<job_id>``, and — on done —
``ingest``s the clip from the SHARED content-addressed path (central + worker share
``/mnt/llm_storage``), so there is no b64 round-trip.

This module is import-light on purpose: Flask + stdlib at module top, and the
studio spine (numpy/PIL/torch via ``produce_clip``) is imported LAZILY inside the
render thread. It does NOT import ``worker_agent.agent``, so it can be mounted on a
bare Flask app (the tree's E2E test does exactly that) without the agent's heavy
boot.

Three routes (registered by :func:`register_studio_routes`):
    POST /studio/render            {job_id, spec, central_version?} -> 202 | 409
    GET  /studio/render/<job_id>   -> {status, position?, progress?, result?, pkg_version}
    POST /studio/cancel/<job_id>   -> sets the render's local cancel flag

Concurrency: ONE render EXECUTES at a time, but a second ``/studio/render`` (a
DIFFERENT job) while one is running is no longer rejected — it is QUEUED. The
manager keeps a bounded FIFO queue (``HUGPY_STUDIO_QUEUE_DEPTH``, default 4);
renders drain serially in the existing per-job thread model (each render's finally
promotes the next queued job). The 202 body then carries ``accepted:"queued"`` +
``position`` (1-based place in line). A GET status for a queued job reports
``{status:"queued", position:N}`` and central keeps polling (forwarding the
position as progress). 409 is returned ONLY when the queue is FULL — central then
retries kick-off with backoff within its delegation window (worker_busy means WAIT,
not fail). A re-POST of the SAME job_id is idempotent (``accepted:"exists"``).

Cancel is cooperative and works in BOTH states: a RUNNING job's cancel sets the
render thread's ``should_cancel()`` local flag (the exact probe ``produce_clip``
already accepts, T1 semantics) — the runner aborts BEFORE writing a clip and
returns ``Err(CANCELLED)``; a QUEUED job's cancel removes it from the queue and
settles it as a cancelled result WITHOUT ever rendering. Both settle with an
Err(cancelled) payload so central records a 'cancelled' terminal, not 'failed'.

No pathlib anywhere. os.path only.
"""
from __future__ import annotations

import logging
import os
import threading
import time

from flask import jsonify, request

from hugpy_fleet.worker import _studio_subproc

logger = logging.getLogger(__name__)

# Bounded FIFO depth for jobs WAITING behind the one in flight (env-tunable). The
# in-flight render is NOT counted against this; the queue caps how many additional
# jobs a worker will hold before it returns 409 (queue full). Default 4.
_QUEUE_DEPTH_ENV = "HUGPY_STUDIO_QUEUE_DEPTH"
_DEFAULT_QUEUE_DEPTH = 4


def _queue_depth() -> int:
    """Resolve the bounded queue depth from env (default 4). A non-positive or
    unparseable value falls back to the default so the worker always queues at
    least a few."""
    raw = os.environ.get(_QUEUE_DEPTH_ENV)
    if raw in (None, ""):
        return _DEFAULT_QUEUE_DEPTH
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QUEUE_DEPTH
    return n if n > 0 else _DEFAULT_QUEUE_DEPTH


def _cancelled_payload(message: str) -> dict:
    """The settled-result payload for a cancelled render — the SAME shape a
    mid-render ``Err(CANCELLED)`` produces via ``artifact_result_to_payload``, so a
    queued-cancel and a running-cancel look identical to central (both map to a
    JobResult with code 'cancelled', not retryable)."""
    return {"ok": False, "error": {
        "code": "cancelled", "message": message, "retryable": False}}


def _pkg_version() -> str:
    """This worker's package version — echoed on every response so central can log
    a version skew against its own. Best-effort; 'unknown' if unresolved."""
    try:
        from hugpy_fleet import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


class _RenderJob:
    __slots__ = ("job_id", "spec", "status", "result", "progress", "cancel",
                 "created_at", "started_at", "thread")

    def __init__(self, job_id: str, spec_dict: dict):
        self.job_id = job_id
        self.spec = spec_dict      # carried so a queued job can launch later
        self.status = "queued"     # queued | running | done | error
        self.result = None         # produce_clip payload dict once settled
        self.progress = None       # coarse live progress blob (or None)
        self.cancel = threading.Event()
        self.created_at = time.time()
        self.started_at = None     # set when the render actually begins running
        self.thread = None


class StudioRenderManager:
    """Thread-safe studio render table for a worker: ONE render EXECUTES at a time,
    additional renders WAIT in a bounded FIFO queue.

    In-memory by design: the DURABLE job ledger stays single-writer on central
    (media_jobs.db). This table only tracks the in-flight + queued renders so
    central can poll them; a worker restart drops the table, which central reads as
    ``worker_lost`` (queued jobs included — central will retry).

    State machine per job: ``queued`` -> ``running`` -> ``done`` | ``error``. A
    cancel settles the job with a cancelled result payload (from a queued job:
    immediately, never rendered; from a running job: cooperatively, mid-render). The
    bounded queue depth is read once at construction from ``HUGPY_STUDIO_QUEUE_DEPTH``
    (default 4); it caps jobs WAITING behind the in-flight one, and a POST that would
    overflow it returns ``(False, "full", None)`` -> the route's only 409."""

    def __init__(self, queue_depth: "int | None" = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, _RenderJob] = {}
        self._active: str | None = None    # in-flight render's job_id, or None
        self._queue: list[str] = []        # FIFO of QUEUED job_ids (waiting)
        self._depth = queue_depth if queue_depth is not None else _queue_depth()

    # -- internal: promote / launch (both callers already hold self._lock) ------- #
    def _launch_locked(self, job: "_RenderJob") -> None:
        """Flip a job to running and start its render thread. Caller holds the lock;
        ``Thread.start`` returns immediately and the thread blocks on the lock only
        when it reaches its first ``with self._lock`` inside ``_run`` (no deadlock)."""
        job.status = "running"
        job.started_at = time.time()
        self._active = job.job_id
        t = threading.Thread(
            target=self._run, args=(job,),
            name=f"studio-render-{job.job_id[:8]}", daemon=True)
        job.thread = t
        t.start()

    def _start_next_locked(self) -> None:
        """Promote the next still-queued job (skipping any cancelled-while-queued),
        or leave the manager idle. Caller holds the lock."""
        while self._queue:
            nxt = self._queue.pop(0)
            job = self._jobs.get(nxt)
            if job is None or job.status != "queued":
                continue                    # cancelled while queued -> skip it
            self._launch_locked(job)
            return
        self._active = None                 # nothing left -> idle

    def start(self, job_id: str, spec_dict: dict) -> tuple[bool, str, "int | None"]:
        """Accept a render. Returns ``(ok, reason, position)``:
          * ``(True, "started", None)``  idle -> render kicked off immediately;
          * ``(True, "queued", N)``      busy -> enqueued at 1-based position N;
          * ``(True, "exists", None)``   idempotent re-POST of a known job;
          * ``(False, "full", None)``    queue is FULL (caller returns 409).
        ``worker_busy`` now means WAIT: a second render is queued, not rejected,
        unless the bounded queue would overflow."""
        with self._lock:
            if job_id in self._jobs:
                return True, "exists", None    # idempotent re-POST of a known job
            active = self._jobs.get(self._active) if self._active else None
            busy = (active is not None and active.status == "running") or bool(self._queue)
            if busy:
                if len(self._queue) >= self._depth:
                    return False, "full", None  # bounded queue overflow -> 409
                job = _RenderJob(job_id, spec_dict)   # status "queued"
                self._jobs[job_id] = job
                self._queue.append(job_id)
                return True, "queued", len(self._queue)  # 1-based place in line
            # idle (or a stale active pointer from a settled render) -> run now.
            self._active = None
            job = _RenderJob(job_id, spec_dict)
            self._jobs[job_id] = job
            self._launch_locked(job)
            return True, "started", None

    def _run(self, job: "_RenderJob") -> None:
        """Render thread: rebuild the spec through the SAME validating deserializer
        the bus uses, run the SHARED ``run_produce_clip`` (identical to central's
        in-process path), record the JSON-safe result payload, and — in the finally
        — promote the next queued render so the FIFO drains serially."""
        # VRAM RESERVE (2026-08-13): template evict-to-fit + a non-evictable
        # registry hold for the render's lifetime, so a demand load (LLM slot,
        # image gen) can never re-fill the card under a running render — the OOM
        # class the model battery kept hitting. Best-effort: no template / claim
        # short / seam down -> render proceeds exactly as before.
        _release_reserve = lambda: None  # noqa: E731
        try:
            from hugpy_fleet.worker import studio_reserve as _reserve
            _release_reserve = _reserve.acquire(
                job.job_id, (job.spec or {}).get("model_id"))
        except Exception:  # noqa: BLE001 — the reserve must never fail a render
            logger.warning("studio render %s: reserve acquire crashed — "
                           "rendering without a reserve", job.job_id,
                           exc_info=True)
        try:
            with self._lock:
                job.progress = {"phase": "rendering", "started_at": job.started_at}

            def _publish(blob: dict) -> None:
                """Merge a render-progress frame into the job's live blob (k57), so
                GET /studio/status/<id> — and central's poller, which forwards this
                blob verbatim to the media bus — reports STEP i/N instead of a bare
                "rendering" for the whole render."""
                with self._lock:
                    if job.status != "running":
                        return          # settled/cancelled — don't resurrect a blob
                    job.progress = {"phase": "rendering",
                                    "started_at": job.started_at, **blob}
            # k17 deadlock fix: run the GPU render in a KILLABLE, timeout-bounded
            # CHILD PROCESS (spawn), so a native torch/CUDA/PIL stall in the render
            # tail (fp32 VAE decode / postprocess under VRAM contention) can never
            # freeze THIS worker thread — the worker itself runs no native torch, and
            # a wedged render is killed and failed honestly instead of hanging forever
            # and requiring a manual restart. The legacy in-thread path stays behind
            # an escape-hatch env for a box that wants it.
            if _studio_subproc.render_inprocess_forced():
                # Lazy imports (torch/diffusers/numpy pulled only here, never at mount).
                from hugpy_video.intel.runners.studio_i2v import artifact_result_to_payload, run_produce_clip
                from hugpy_video.intel.studio.job import studio_i2v_from_dict

                spec = studio_i2v_from_dict(job.spec)
                should_cancel = lambda: job.cancel.is_set()  # noqa: E731
                result = run_produce_clip(
                    spec, should_cancel,
                    on_step=lambda step, steps: _publish(
                        {"step": step, "steps": steps}))
                payload = artifact_result_to_payload(result)
            else:
                payload = _studio_subproc.run_render_subprocess(
                    job.spec, job.cancel, on_progress=_publish)
            with self._lock:
                job.result = payload
                job.status = "done"
                job.progress = None
        except Exception as exc:  # noqa: BLE001 — a crash is errors-as-data too
            logger.exception("studio render %s crashed", job.job_id)
            with self._lock:
                job.result = {"ok": False, "error": {
                    "code": "internal",
                    "message": f"{type(exc).__name__}: {exc}",
                    "retryable": False}}
                job.status = "error"
                job.progress = None
        finally:
            try:
                _release_reserve()
            except Exception:  # noqa: BLE001 — registry clears on agent restart
                logger.warning("studio render %s: reserve release failed",
                               job.job_id, exc_info=True)
            with self._lock:
                if self._active == job.job_id:
                    self._active = None
                self._start_next_locked()       # drain the FIFO serially

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "unknown",
                        "position": None, "progress": None, "result": None}
            position = None
            if job.status == "queued":
                try:
                    position = self._queue.index(job_id) + 1   # 1-based place in line
                except ValueError:
                    position = None
            return {"job_id": job_id, "status": job.status, "position": position,
                    "progress": dict(job.progress) if job.progress else None,
                    "result": job.result}

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "cancelled": False, "status": "unknown"}
            if job.status == "queued":
                # Never-rendered: drop from the queue and settle as cancelled data.
                # (Internal status flips to "done" carrying the cancelled payload, so
                # central's terminal-status poll settles it as 'cancelled'.)
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.result = _cancelled_payload("cancelled while queued (never rendered)")
                job.status = "done"
                job.progress = None
                return {"job_id": job_id, "cancelled": True, "status": "cancelled"}
            if job.status != "running":
                return {"job_id": job_id, "cancelled": False, "status": job.status}
            job.cancel.set()
            return {"job_id": job_id, "cancelled": True, "status": "cancelling"}


def register_studio_routes(app, *, manager=None, worker_id=None, worker_name=None):
    """Mount the studio render endpoints on ``app``.

    ``manager`` lets a test inject its own :class:`StudioRenderManager`; production
    (``build_app``) passes ``None`` (a fresh one). ``worker_id`` / ``worker_name``
    are echoed for attribution. Returns the manager so a caller/test can inspect
    it."""
    mgr = manager or StudioRenderManager()
    ver = _pkg_version()

    @app.route("/studio/render", methods=["POST"])
    def studio_render():
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id")
        spec_dict = body.get("spec")
        if not job_id or not isinstance(spec_dict, dict):
            return jsonify({"ok": False, "error": {
                "code": "BadRequest",
                "message": 'body must include {"job_id": str, "spec": {...}}'},
                "pkg_version": ver}), 400
        central_version = str(body.get("central_version") or "")
        if central_version and central_version != ver:
            logger.warning("studio offload VERSION SKEW: central=%s worker=%s "
                           "(job %s) — running anyway; behavior may differ",
                           central_version, ver, job_id)
        ok, reason, position = mgr.start(job_id, spec_dict)
        if not ok:
            # The ONLY 409: the bounded FIFO queue is full. Central treats this as
            # "wait" — it retries kick-off with backoff inside its delegation window.
            return jsonify({"ok": False, "error": {
                "code": "QueueFull",
                "message": f"studio render queue is full (depth {mgr._depth}); "
                           "retry shortly"},
                "pkg_version": ver, "worker_id": worker_id}), 409
        body_out = {"ok": True, "job_id": job_id, "accepted": reason,
                    "pkg_version": ver, "worker_id": worker_id,
                    "worker_name": worker_name}
        if position is not None:
            body_out["position"] = position     # 1-based place in the FIFO queue
        return jsonify(body_out), 202

    @app.route("/studio/render/<job_id>", methods=["GET"])
    def studio_render_status(job_id):
        st = mgr.status(job_id)
        st["pkg_version"] = ver
        st["worker_id"] = worker_id
        return jsonify(st), 200

    @app.route("/studio/cancel/<job_id>", methods=["POST"])
    def studio_render_cancel(job_id):
        out = mgr.cancel(job_id)
        out["pkg_version"] = ver
        return jsonify(out), 200

    logger.info("studio render endpoints mounted (worker=%s, pkg=%s)",
                worker_name or worker_id or "?", ver)
    return mgr
