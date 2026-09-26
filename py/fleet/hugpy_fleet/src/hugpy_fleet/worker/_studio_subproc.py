"""Killable, timeout-bounded studio render in a CHILD PROCESS (k17 deadlock fix).

Why this exists
---------------
A studio render's heavy tail — the fp32 Wan VAE decode of a long (e.g. 81-frame)
latent plus the ``output_type="pil"`` post-processing — is a single SYNCHRONOUS
native torch/CUDA + PIL call inside the runner (``run_wan_i2v`` /
``run_wan_vace``). There is no Python-level concurrency primitive anywhere in the
studio runner path, so when that native call STALLS — a GIL-holding tensor->PIL
postprocess starving every other worker thread, or a CUDA alloc/sync that never
returns because the shared 3090 is VRAM-squatted by an out-of-band process
(comfy / a slot llama-server) — it cannot be interrupted from Python: you cannot
kill a thread, and a GIL-holding native call starves any in-process watchdog
thread too. The worker's render thread parks at 0% CPU with the job stuck
"running" forever, and the whole ``hugpy-worker-agent`` needs a MANUAL restart
(recorded 2026-07-18, k11-triggered).

The fix
-------
Run the render in a SEPARATE, KILLABLE child process and join it with a hard
timeout. The worker process then NEVER executes native torch/CUDA itself, so:

  * no in-worker GIL-holding stall can freeze the heartbeat/Flask threads;
  * ``spawn`` (not the default ``fork``) gives the child a CLEAN CUDA context —
    a render forked from a parent that already CUDA-initialised in-process would
    inherit a poisoned context and deadlock (the classic fork-after-CUDA trap);
  * a wedged render is time-bounded: on timeout the parent kills the child and
    settles the job as an HONEST error (worker survives, central stops polling),
    which beats a silent hang (project doctrine: honest failure > silent hang).

Cooperative cancel is preserved: the parent bridges the job's ``threading.Event``
onto a ``multiprocessing`` cancel event the child polls as ``should_cancel`` (the
same probe the runner already honours), and if the child does not settle within a
short grace window it is killed.

The result crosses the process boundary as the SAME JSON-safe payload dict
``artifact_result_to_payload`` produces, so nothing downstream changes.

stdlib only here; the studio spine (torch/diffusers) is imported ONLY in the
child. os.path only, no pathlib.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

logger = logging.getLogger(__name__)

# Hard wall-clock budget for a single render child before the parent kills it and
# fails the job honestly. Generous by default — an 81-frame Wan render at 32 steps
# plus the fp32 VAE decode is many minutes on a shared 3090 — so a legitimate
# render never trips it; it exists to bound a WEDGE, not a slow render. Env-tunable.
_TIMEOUT_ENV = "HUGPY_STUDIO_RENDER_TIMEOUT_S"
_DEFAULT_TIMEOUT_S = 2400.0        # 40 minutes

# Escape hatch: force the legacy IN-PROCESS render (render on the worker's own
# thread). Off by default — the whole point is to NOT run native torch in-worker.
_INPROCESS_ENV = "HUGPY_STUDIO_RENDER_INPROCESS"

# Seconds to let a cancelled child settle cooperatively (abort BEFORE writing a
# clip, returning Err(CANCELLED)) before we hard-kill it.
_CANCEL_GRACE_S = 20.0

# How long to wait for a killed child to actually die before escalating to SIGKILL.
_KILL_WAIT_S = 5.0


def render_inprocess_forced() -> bool:
    """True iff the operator pinned the legacy in-process render path."""
    return os.environ.get(_INPROCESS_ENV, "").strip() in ("1", "true", "TRUE", "yes")


def render_timeout_s() -> float:
    """Resolve the per-render wall-clock timeout (seconds) from env; a non-positive
    or unparseable value falls back to the default so a render is always bounded."""
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw in (None, ""):
        return _DEFAULT_TIMEOUT_S
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return v if v > 0 else _DEFAULT_TIMEOUT_S


def _internal_payload(message: str, *, retryable: bool = True) -> dict:
    """A settled-result payload for a worker-side failure (child crash / timeout /
    lost result), shaped identically to the manager's own error payloads so central
    rebuilds the JobError verbatim."""
    return {"ok": False, "error": {
        "code": "internal", "message": message, "retryable": retryable}}


def _cancelled_payload(message: str) -> dict:
    """Cancelled-result payload (matches studio_render._cancelled_payload)."""
    return {"ok": False, "error": {
        "code": "cancelled", "message": message, "retryable": False}}


# Progress frames (k57) share the result pipe with the settled payload; this key
# is what tells them apart. A frame is {_PROGRESS_KEY: {"phase": "rendering",
# "step": i, "steps": n}} — never a render result, so an old parent that somehow
# received one would simply see a dict without "ok" rather than a corrupt result.
_PROGRESS_KEY = "__progress__"


def _progress_frame(frame) -> "dict | None":
    """The progress blob inside a pipe frame, or None when the frame is a settled
    render payload."""
    if isinstance(frame, dict) and isinstance(frame.get(_PROGRESS_KEY), dict):
        return frame[_PROGRESS_KEY]
    return None


def _child_main(spec_dict: dict, conn, cancel_event) -> None:
    """CHILD entrypoint (module-level so it is spawn-picklable). Rebuild the spec
    through the SAME validating deserializer the bus uses, run the SHARED
    ``run_produce_clip`` with the cancel event bridged in as ``should_cancel``, and
    send the JSON-safe payload back over ``conn``. Every failure — including a bad
    spec — rides back as an error payload (errors-as-data); the child never lets an
    exception escape unsent, so the parent never waits on a silent crash.

    It ALSO streams denoise progress up the same pipe (k57): the render runs in
    THIS process, so its step counter is invisible to the worker (and therefore to
    central, and therefore to the operator's bar) unless it is sent."""
    payload: dict
    try:
        from hugpy_video.intel.runners.studio_i2v import artifact_result_to_payload, run_produce_clip
        from hugpy_video.intel.studio.job import studio_i2v_from_dict

        spec = studio_i2v_from_dict(spec_dict)
        should_cancel = cancel_event.is_set  # zero-arg probe the runner already accepts

        def _on_step(step: int, steps: int) -> None:
            try:
                conn.send({_PROGRESS_KEY: {"phase": "rendering",
                                           "step": int(step), "steps": int(steps)}})
            except Exception:  # noqa: BLE001 — a broken pipe must not stop the render
                pass

        result = run_produce_clip(spec, should_cancel, on_step=_on_step)
        payload = artifact_result_to_payload(result)
    except Exception as exc:  # noqa: BLE001 — a child crash is errors-as-data too
        payload = _internal_payload(
            f"studio render child crashed: {type(exc).__name__}: {exc}")
    try:
        conn.send(payload)
    except Exception:  # noqa: BLE001 — parent gone / pipe broken; nothing we can do
        pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _terminate(proc) -> None:
    """Best-effort kill: SIGTERM, wait briefly, then SIGKILL. A CUDA-wedged child in
    user space dies on SIGTERM; SIGKILL is the backstop."""
    try:
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join(_KILL_WAIT_S)
        if proc.is_alive():
            proc.kill()
            proc.join(_KILL_WAIT_S)
    except Exception:  # noqa: BLE001
        pass


def run_render_subprocess(
    spec_dict: dict,
    cancel_flag,
    timeout_s: "float | None" = None,
    *,
    on_progress=None,
    on_child_pid=None,
    _target=_child_main,
    _ctx=None,
    _poll_s: float = 0.5,
) -> dict:
    """Render ``spec_dict`` in a spawned child process, bounded by ``timeout_s``.

    Returns the JSON-safe render payload dict (``artifact_result_to_payload``
    shape). ``cancel_flag`` is the job's ``threading.Event`` (or any object with
    ``is_set()``); when it fires the child is asked to cancel cooperatively and, if
    it does not settle within a short grace window, killed. On timeout the child is
    killed and an honest ``internal`` (retryable) error payload is returned — the
    worker itself never blocks in native code, so it stays responsive throughout.

    ``on_progress`` (k57) receives each progress blob the child streams up the
    result pipe ({"phase": "rendering", "step": i, "steps": n}) — the worker points
    it at the job's ``progress`` field so /studio/status, and through it central's
    poller and the console's bar, sees movement WITHIN a render. Best-effort: a
    throwing sink is logged and ignored. None (default) drops the frames.

    ``on_child_pid`` (2026-09-24) is called ONCE with the spawned render child's
    pid right after it starts — the seam the studio VRAM reserve uses to
    re-attribute its non-evictable hold from the worker pid (which holds no torch
    VRAM) to the process that actually holds the card, so the render is a
    first-class, MEASURED external resident in the same eviction ledger as an LLM
    slot rather than an unattributed own-venv squatter. Best-effort: a throwing
    sink is logged and ignored; None (default) skips it.

    ``_target`` / ``_ctx`` / ``_poll_s`` are test seams (inject a fake child target,
    context, or a faster poll); production uses the real spawn context + child.
    """
    if timeout_s is None:
        timeout_s = render_timeout_s()
    ctx = _ctx or mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    cancel_event = ctx.Event()
    proc = ctx.Process(
        target=_target, args=(spec_dict, child_conn, cancel_event), daemon=False)
    proc.start()
    child_conn.close()  # only the child writes; parent keeps the read end
    if on_child_pid is not None:
        try:
            on_child_pid(proc.pid)
        except Exception:  # noqa: BLE001 — pid re-attribution never fails a render
            logger.debug("studio render on_child_pid sink failed", exc_info=True)

    deadline = time.monotonic() + float(timeout_s)
    cancel_deadline: "float | None" = None
    outcome_kind = "ok"   # ok | timeout | cancelled
    payload: "dict | None" = None
    try:
        while True:
            if parent_conn.poll(_poll_s):
                # A frame is waiting. It is either a PROGRESS frame (k57 — the
                # child's denoise step counter, which must cross the process
                # boundary or the worker can only ever report "rendering") or the
                # terminal payload. Progress frames are consumed and relayed; only
                # a terminal payload ends the loop.
                try:
                    frame = parent_conn.recv()
                except EOFError:
                    break  # child closed the pipe without settling
                prog = _progress_frame(frame)
                if prog is not None:
                    if on_progress is not None:
                        try:
                            on_progress(prog)
                        except Exception:  # noqa: BLE001 — telemetry is never fatal
                            logger.debug("studio render progress sink failed",
                                         exc_info=True)
                    continue
                payload = frame if isinstance(frame, dict) else None
                break
            if not proc.is_alive():
                break  # child exited (crashed / finished without a readable payload)
            now = time.monotonic()
            # Cooperative cancel: signal the child, then hard-kill after a grace window.
            if cancel_flag is not None and cancel_flag.is_set():
                if not cancel_event.is_set():
                    cancel_event.set()
                    cancel_deadline = now + _CANCEL_GRACE_S
                elif cancel_deadline is not None and now >= cancel_deadline:
                    outcome_kind = "cancelled"
                    logger.warning("studio render child ignored cancel; killing")
                    _terminate(proc)
                    break
            # Watchdog: a wedged render is killed and the job fails honestly.
            if now >= deadline:
                outcome_kind = "timeout"
                logger.error(
                    "studio render child exceeded %.0fs; killing (deadlock watchdog)",
                    timeout_s)
                _terminate(proc)
                break

        # A child that raced the kill may still have a settled payload queued.
        while payload is None and parent_conn.poll(0):
            try:
                frame = parent_conn.recv()
            except EOFError:
                break
            if _progress_frame(frame) is None:
                payload = frame if isinstance(frame, dict) else None
    finally:
        try:
            parent_conn.close()
        except Exception:  # noqa: BLE001
            pass
        _terminate(proc)  # ensure no orphan child survives this call

    if payload is not None and isinstance(payload, dict):
        return payload
    if outcome_kind == "timeout":
        return _internal_payload(
            f"studio render timed out after {timeout_s:.0f}s (deadlock watchdog "
            "killed the render child; the worker stayed healthy)")
    if outcome_kind == "cancelled":
        return _cancelled_payload("cancelled (render child killed after grace)")
    # Child died without sending a payload (segfault / OOM-kill / SIGKILL).
    return _internal_payload(
        "studio render child exited without a result (crash / OOM-kill)")
