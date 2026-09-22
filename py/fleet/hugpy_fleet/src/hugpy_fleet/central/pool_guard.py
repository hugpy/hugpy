"""k59 — keep a reserve of the gunicorn thread pool for FAST READS.

Central runs `gunicorn --workers 1 --threads N`: one process, N threads, and
every request holds one of them for its whole lifetime. Long-lived SSE streams
hold a thread for minutes by design. So with N=8, five console tabs on the
eviction feed leave three threads for the entire rest of the API — that is the
"blips over just a few calls" the operator has seen since day one, and no
timeout tuning fixes it, because those threads are not stuck, they are working
exactly as designed.

Gunicorn cannot give streams their own pool inside one worker process. What it
CAN do is refuse to let them take the last threads. This module is that reserve:
a counting semaphore sized as a fraction of the pool, taken for the lifetime of
a stream. Past the cap, a NEW stream is refused immediately with an honest 503 +
Retry-After — the browser's EventSource reconnects on its own a moment later,
and meanwhile /llm/workers, /llm/jobs and /models still answer.

Refusing a stream is strictly better than the alternative it replaces: without
the reserve the stream is accepted and something else — a heartbeat, a roster
poll — times out instead, with no message explaining why.

Per-process state, deliberately: the pool it guards is per-process too, so a
second gunicorn worker has its own threads and its own reserve. That is the one
piece of in-process state that is *correct* precisely because it is not shared.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# What the unit's --threads says. Set it alongside the flag; the default matches
# the shipped unit so a mismatch degrades to the conservative old number rather
# than to an unguarded pool.
DEFAULT_THREADS = 8

# Fraction of the pool long-lived streams may hold at once. 0.6 keeps ~40% of
# the threads for fast reads no matter how many tabs are open.
DEFAULT_STREAM_FRACTION = 0.6

# How long to tell a refused client to wait. Short: capacity frees the moment
# any stream ends or hits its own lifetime bound.
RETRY_AFTER_S = 5


def pool_threads() -> int:
    try:
        v = int((os.environ.get("HUGPY_GUNICORN_THREADS") or "").strip()
                or DEFAULT_THREADS)
        return v if v > 0 else DEFAULT_THREADS
    except (TypeError, ValueError):
        return DEFAULT_THREADS


def stream_slots() -> int:
    """How many threads long-lived streams may hold at once.

    ``HUGPY_STREAM_SLOTS`` sets it outright; otherwise it is a fraction of the
    pool. Never less than 2 — a reserve that refuses the FIRST stream would be
    a broken feature, not a protected one.
    """
    env = (os.environ.get("HUGPY_STREAM_SLOTS") or "").strip()
    if env:
        try:
            v = int(env)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return max(2, int(pool_threads() * DEFAULT_STREAM_FRACTION))


class _Reserve:
    """A counting gate that never blocks — it admits or refuses at once.

    Blocking would defeat the purpose: a stream waiting for a slot is still
    holding the thread it was trying not to monopolize.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = 0
        self._limit: Optional[int] = None
        self.refused = 0

    def _cap(self) -> int:
        # Read the limit once and remember it: the pool size cannot change
        # under a running process, and re-reading env per request is waste.
        if self._limit is None:
            self._limit = stream_slots()
        return self._limit

    def try_acquire(self) -> bool:
        with self._lock:
            if self._held >= self._cap():
                self.refused += 1
                if self.refused in (1, 10) or self.refused % 100 == 0:
                    logger.warning(
                        "stream reserve full: %d/%d slots held, refused %d "
                        "stream(s) so far — fast reads are being protected",
                        self._held, self._cap(), self.refused)
                return False
            self._held += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._held > 0:
                self._held -= 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"held": self._held, "limit": self._cap(),
                    "refused": self.refused}

    def reset(self) -> None:
        """Test seam."""
        with self._lock:
            self._held, self._limit, self.refused = 0, None, 0


_RESERVE = _Reserve()


class StreamCapacityExceeded(RuntimeError):
    """No stream slot free. Carries the honest wire body."""

    retry_after = RETRY_AFTER_S

    def as_error(self) -> dict:
        snap = _RESERVE.snapshot()
        return {"ok": False, "error": {
            "code": "StreamCapacity",
            "message": (f"all {snap['limit']} live-stream slots are in use — "
                        f"reconnect in a few seconds (the reserve keeps the "
                        f"remaining threads answering ordinary requests)"),
            "retry_after_s": RETRY_AFTER_S}}


@contextmanager
def stream_slot() -> Iterator[None]:
    """Hold a stream slot for the lifetime of the block, or refuse now.

    Wrap the *whole* SSE response, generator included — the thread is held
    until the generator is exhausted or the client disconnects, so releasing
    any earlier would count the slot back while it is still occupied.
    """
    if not _RESERVE.try_acquire():
        raise StreamCapacityExceeded()
    try:
        yield
    finally:
        _RESERVE.release()


def snapshot() -> dict:
    """Introspection: how much of the reserve is in use right now."""
    return _RESERVE.snapshot()


def reset() -> None:
    """Test seam — drop every held slot and re-read the limit."""
    _RESERVE.reset()
