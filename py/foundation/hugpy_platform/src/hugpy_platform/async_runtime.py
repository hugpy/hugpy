"""Process-wide async runtime — ONE long-lived event loop in a daemon thread.

Replaces the per-request ``asyncio.new_event_loop()`` pattern every SSE / one-shot
endpoint used. That pattern caused two problems:

  * Loop-binding crashes — an asyncio sync primitive (Semaphore/Lock/Event)
    cached on a process singleton binds to the FIRST request's loop and then
    raises "bound to a different event loop" on the next request. With one
    persistent loop, cached primitives stay valid for the life of the process.
  * Per-request loop churn — creating/closing a loop per request and pinning a
    thread in run_until_complete for the whole stream. The shared loop interleaves
    many streams cooperatively; blocking model work runs in the default executor
    (asyncio.to_thread), so the loop stays responsive.

Sync callers submit coroutines via ``run()`` / ``iter_sync()``; the loop runs them
and the caller blocks on a ``concurrent.futures.Future``. All entry points are
thread-safe (``run_coroutine_threadsafe``). Usable from both central (gunicorn
threads) and the worker agent (its request threads).

ABANDON-ON-DISCONNECT (2026-07-27). These two functions are the ONE place a WSGI
thread parks while the loop works, so they are also the only place that can hand
the thread back when the caller gives up. Instead of blocking forever on
``fut.result()`` they wake every ``client_liveness.poll_s()`` and ask the
request's socket probe whether the client is still there; on a definite
disconnect they cancel the in-flight coroutine (CancelledError unwinds its
``finally`` blocks, releasing relay slots / cold-hold permits / httpx streams)
and give the request slot back. With no probe bound — a background thread, an
internal drain, a WSGI server that publishes no socket — ``alive`` is None and
the behaviour is byte-identical to before.
"""
from __future__ import annotations

import asyncio
import threading
import logging
import concurrent.futures as _cf

from hugpy_platform import client_liveness

logger = logging.getLogger(__name__)

_loop: "asyncio.AbstractEventLoop | None" = None
_thread: "threading.Thread | None" = None
_start_lock = threading.Lock()


def loop() -> "asyncio.AbstractEventLoop":
    """The shared event loop, starting its daemon thread on first use."""
    global _loop, _thread
    lp = _loop
    if lp is not None and lp.is_running():
        return lp
    with _start_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        lp = asyncio.new_event_loop()
        ready = threading.Event()

        def _run():
            asyncio.set_event_loop(lp)
            ready.set()
            lp.run_forever()

        t = threading.Thread(target=_run, name="hugpy-async-runtime", daemon=True)
        t.start()
        ready.wait(5)
        _loop, _thread = lp, t
        logger.info("async runtime started (thread=%s)", t.name)
        return _loop


def submit(coro) -> "_cf.Future":
    """Schedule a coroutine on the shared loop; return its concurrent Future."""
    return asyncio.run_coroutine_threadsafe(coro, loop())


def _client_gone(alive) -> bool:
    """Ask a liveness checker, treating ANY failure as "still connected"."""
    if alive is None:
        return False
    try:
        return bool(alive())
    except Exception:       # noqa: BLE001 — never abandon a caller on doubt
        return False


def _abandon(fut) -> None:
    """Cancel an in-flight step and let its ``finally`` blocks unwind."""
    if fut is None or fut.done():
        return
    fut.cancel()
    try:
        fut.result(5)
    except BaseException:   # noqa: BLE001 — cancellation is the expected outcome
        pass


def run(coro, *, alive=None):
    """Run a coroutine on the shared loop from a sync thread; block for its result.

    ``alive`` is a zero-arg "is my caller still connected?"; it defaults to the
    probe this thread's Flask request bound (see _platform.client_liveness). When
    one is available and reports a disconnect, the coroutine is CANCELLED and
    ``ClientGone`` is raised — the WSGI thread is returned to the pool instead of
    sitting out (up to) a 25-minute cold hold for a caller who left. No probe ⇒
    plain blocking ``.result()``, exactly as before.
    """
    if not asyncio.iscoroutine(coro):
        # Tolerate already-resolved values (callers that may pass a plain result).
        return coro
    if alive is None:
        alive = client_liveness.current_checker()
    fut = submit(coro)
    if alive is None:
        return fut.result()
    poll = client_liveness.poll_s()
    while True:
        try:
            return fut.result(poll)
        except _cf.TimeoutError:
            if not _client_gone(alive):
                continue
            _abandon(fut)
            logger.info("client disconnected — abandoned in-flight work "
                        "and released the request slot")
            raise client_liveness.ClientGone(
                "the client disconnected before the reply was ready; the "
                "in-flight work was cancelled and its slot released")


def call_soon_threadsafe(callback, *args) -> None:
    """Schedule a plain callback on the shared loop (e.g. ``Event.set`` from
    another thread, which is otherwise unsafe to call cross-loop)."""
    loop().call_soon_threadsafe(callback, *args)


def _step_wait(heartbeat, heartbeat_secs: float, poll, waited: float):
    """How long to block on the current step: whichever of the heartbeat tick and
    the disconnect poll comes first. None (block forever) only when neither
    applies — the pre-existing internal-drain behaviour."""
    hb = (heartbeat_secs - waited) if heartbeat is not None else None
    if hb is not None and hb <= 0:
        hb = heartbeat_secs
    if poll is None:
        return hb
    if hb is None:
        return poll
    return max(0.01, min(hb, poll))


def iter_sync(agen, heartbeat: "bytes | None" = None, heartbeat_secs: float = 15.0,
              alive=None):
    """Drive an async generator from a sync (WSGI) thread on the SHARED loop.

    Mirrors the old per-request driver semantics:
      * With ``heartbeat`` bytes, each step waits at most ``heartbeat_secs`` and
        yields the keepalive on timeout while the SAME step keeps running — so a
        slow first token can't trip an upstream proxy, and every keepalive write
        lets the WSGI server notice a dead client and trigger teardown.
      * ``heartbeat=None`` blocks for each real event (internal/worker drains).
      * On teardown the in-flight step is cancelled, then ``aclose()`` cascades
        GeneratorExit through every ``async for`` / ``async with`` so a relayed
        worker's httpx stream is released rather than leaked.

    ABANDON-ON-DISCONNECT: when a client probe is available (``alive``, defaulting
    to this request thread's), the step wait is additionally bounded by the poll
    interval and a definite disconnect ends the drain — teardown below then
    cancels and acloses exactly as it does for any other early exit. This is what
    covers ``heartbeat=None`` drains such as the /v1 non-streaming completion,
    which never writes to the client and so can never learn of a disconnect from
    a failed write.
    """
    lp = loop()
    if alive is None:
        alive = client_liveness.current_checker()
    poll = client_liveness.poll_s() if alive is not None else None
    fut = None
    waited = 0.0
    try:
        while True:
            if fut is None:
                fut = asyncio.run_coroutine_threadsafe(agen.__anext__(), lp)
                waited = 0.0
            step = _step_wait(heartbeat, heartbeat_secs, poll, waited)
            try:
                item = fut.result(step)
                fut = None
            except _cf.TimeoutError:
                waited += step or 0.0
                if _client_gone(alive):
                    logger.info("client disconnected mid-drain — abandoning the "
                                "stream and releasing the request slot")
                    break
                # Next event still cooking — keep the connection warm, keep
                # awaiting the SAME step (it's still running on the loop).
                if heartbeat is not None and waited >= heartbeat_secs - 1e-9:
                    waited = 0.0
                    yield heartbeat
                continue
            except StopAsyncIteration:
                fut = None
                break
            if isinstance(item, str):
                item = item.encode("utf-8")
            yield item
    finally:
        try:
            # Cancel the in-flight step first: CancelledError unwinds the
            # chain's `async with` blocks (closing the worker httpx stream),
            # after which aclose() can finalize without "already running".
            _abandon(fut)
            closer = asyncio.run_coroutine_threadsafe(agen.aclose(), lp)
            try:
                closer.result(10)
            except BaseException:
                pass
        except Exception:
            pass
