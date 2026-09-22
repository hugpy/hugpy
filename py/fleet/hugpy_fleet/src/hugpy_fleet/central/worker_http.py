"""k59 — the ONE sanctioned central→worker HTTP client.

Why this module exists (operator, 2026-07-31): "the API really needs to be more
robust — it constantly blips over just a few calls. The worker calls shouldn't
fry the endpoints."

The dev central serves EVERYTHING on ``gunicorn --workers 1 --threads 8``. Eight
threads, one process. Every central→worker call therefore spends a thread of a
*very* small pool for as long as it takes, and the historic call sites asked
httpx for a single scalar timeout::

    httpx.post(url, timeout=900.0)      # <- 900s CONNECT timeout too

httpx applies a bare float to connect AND read AND write AND pool. So a worker
whose box is powered off (SYN black-holed, no RST) held a gunicorn thread for
fifteen minutes on the *connect*, and three such clicks left five threads for
the whole console. That is the blip.

Two disciplines fix it, and this module is where both live:

1. **Split timeouts.** Connect is always short (~3 s): reaching a live box on the
   LAN is a millisecond affair, so a slow connect means "not there", and no
   amount of waiting improves the answer. Read is per CALL CLASS — a /health
   probe that hasn't answered in 4 s has told us what we needed; a model load
   legitimately takes minutes. See ``READ_TIMEOUTS``.

2. **A per-worker circuit breaker.** Timeouts cluster: the box that just ate a
   900 s connect will eat the next one too. After ``HUGPY_WORKER_BREAKER_FAILURES``
   consecutive TRANSPORT failures the worker's calls fail fast for a cooldown,
   so a dead box costs the pool one thread-second per attempt instead of one
   thread-minute. The refusal is honest and typed — ``WorkerUnreachable``,
   surfaced as ``{"ok": false, "error": {"code": "WorkerUnreachable", ...}}``
   with the retry-after — never a silent success and never a bare 502.

Only TRANSPORT failures (connect refused/timed out, read timed out) trip the
breaker. An HTTP 500 from the worker means the worker ANSWERED: it is reachable,
the call is data, and the breaker stays closed.

This module is the only sanctioned way for central to call a worker;
``tests/test_worker_http_discipline.py`` enforces that with an AST scan.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

# Reaching a worker on the LAN is a sub-millisecond TCP handshake. Three seconds
# is already 1000x headroom; anything longer is a box that is not there, and the
# only thing waiting buys is a held thread.
CONNECT_TIMEOUT_S = 3.0

# Read timeout per CALL CLASS. The class names are the vocabulary every call site
# uses — pick the one that describes what the worker is being asked to DO, not
# how long you hope it takes.
READ_TIMEOUTS: Dict[str, float] = {
    # Liveness. A /health that needs more than a few seconds is not healthy; the
    # cold-hold poll re-asks in ~3 s anyway, so waiting longer tells nobody more.
    "probe": 4.0,
    # Small read-only documents the worker has already computed (/ops/aggregate).
    "status": 10.0,
    # Ordinary control verbs: cancel, unload, config. The worker does real work
    # (dropping weights) but bounded work.
    "control": 30.0,
    # Heavy operator ops the worker executes synchronously: pip installs,
    # restarts, redownload kickoffs.
    "op": 600.0,
    # A model LOAD. Weights come off disk (or the network) and onto the card;
    # minutes is normal and there is no cheaper way to ask.
    "load": 900.0,
    # A full model transfer central→worker.
    "transfer": 3600.0,
    # Token relays. Excepted from the "bounded read" rule by design (task k59):
    # a generation legitimately runs long. The read timeout is the SILENCE
    # budget between SSE chunks, not the call's total length.
    "relay": 600.0,
    # A one-shot (non-streaming) relay: the whole generation arrives as one
    # response body, so the silence budget IS the generation length.
    "relay_long": 3600.0,
}

_DEFAULT_CALL = "control"


def connect_timeout_s() -> float:
    """Connect budget, shared by every call class. ``HUGPY_WORKER_CONNECT_TIMEOUT_S``
    overrides (an operator on a pathological link may need more)."""
    try:
        v = float((os.environ.get("HUGPY_WORKER_CONNECT_TIMEOUT_S") or "").strip()
                  or CONNECT_TIMEOUT_S)
        return v if v > 0 else CONNECT_TIMEOUT_S
    except (TypeError, ValueError):
        return CONNECT_TIMEOUT_S


def read_timeout_s(call: str = _DEFAULT_CALL) -> float:
    """Read budget for a call class. ``HUGPY_WORKER_READ_TIMEOUT_<CLASS>_S``
    overrides one class. An unknown class is a bug at the call site, so it gets
    the conservative ``control`` budget rather than an unbounded wait."""
    if call not in READ_TIMEOUTS:
        logger.warning("worker_http: unknown call class %r — using %r",
                       call, _DEFAULT_CALL)
        call = _DEFAULT_CALL
    env = (os.environ.get(f"HUGPY_WORKER_READ_TIMEOUT_{call.upper()}_S") or "").strip()
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return READ_TIMEOUTS[call]


def timeout_for(call: str = _DEFAULT_CALL,
                read_timeout: Optional[float] = None) -> httpx.Timeout:
    """The explicit four-way httpx timeout for a call class.

    Every field is named on purpose — a bare ``httpx.Timeout(900.0)`` is exactly
    the defect this module exists to remove. ``write`` tracks read (a redownload
    POST body is small, but a slow socket must not hang forever); ``pool`` is
    short because waiting on a connection slot is queueing, not progress.

    ``read_timeout`` overrides the class budget for the ONE caller that already
    carries a per-verb table (``_relay_worker_op``'s /ops/* budgets: a restart
    ACKs in seconds, a pip install does not). That is a per-verb fact, not an
    invented number — and note what it CANNOT override: the connect budget.
    Nothing may lengthen connect, because a slow connect is never a slow
    operation, it is an absent box.
    """
    read = float(read_timeout) if read_timeout else read_timeout_s(call)
    return httpx.Timeout(connect=connect_timeout_s(), read=read,
                         write=max(30.0, min(read, 60.0)), pool=10.0)


# ── the per-worker circuit breaker ─────────────────────────────────────────
#
# State is per PROCESS, deliberately (see the k59 state inventory): it is a
# latency optimization derived entirely from observation, not a fact anyone
# needs to agree on. A second gunicorn process simply re-learns that a box is
# down after its own few failures, and no correctness property depends on the
# two agreeing.
_BREAKER_LOCK = threading.Lock()
_BREAKERS: Dict[str, Dict[str, Any]] = {}

_FAILURES_DEFAULT = 3
_COOLDOWN_DEFAULT_S = 30.0

# The transport failures that mean "this box did not answer". Everything else
# (a 500, a malformed body) means it DID answer and is not a breaker event.
TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.ConnectError,
                    httpx.NetworkError, httpx.TransportError)

# The subset that means "the socket never came up" — distinct from a read
# timeout, where the worker accepted the call and then went quiet. The /ops/config
# retry rides this distinction: a re-exec'ing agent refuses the connection, and
# THAT is worth one re-dial; a call the agent accepted and never answered is not.
CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError)


class WorkerUnreachable(RuntimeError):
    """An honest, typed refusal: central did not reach the worker.

    Raised both when a call actually fails at the transport layer and when the
    breaker refuses to make the call at all. Callers turn it into the same
    ``{"ok": false, "error": {...}}`` envelope the rest of the relay surface
    uses — a refusal must always be visible as "the worker is unreachable",
    never as an empty result.
    """

    def __init__(self, key: str, reason: str, *, url: str = "",
                 retry_after: float = 0.0, tripped: bool = False) -> None:
        super().__init__(reason)
        self.key = key
        self.reason = reason
        self.url = url
        self.retry_after = max(0.0, float(retry_after))
        # True when the breaker refused without dialing (as opposed to a real
        # attempt that failed) — the console renders the two differently.
        self.tripped = tripped

    def as_error(self) -> Dict[str, Any]:
        """The wire envelope. Same shape as _relay_worker_op's failures."""
        return {"ok": False, "error": {
            "code": "WorkerUnreachable",
            "message": self.reason,
            "retry_after_s": round(self.retry_after, 1),
            "breaker_open": self.tripped,
            "url": self.url}}


def is_connect_error(exc: BaseException) -> bool:
    """Did this failure mean the socket never came up? Unwraps a
    ``WorkerUnreachable`` to look at the httpx error underneath."""
    cause = exc.__cause__ if isinstance(exc, WorkerUnreachable) else exc
    return isinstance(cause, CONNECT_ERRORS)


def _breaker_enabled() -> bool:
    return (os.environ.get("HUGPY_WORKER_BREAKER", "").strip().lower()
            not in ("off", "0", "false", "no"))


def _breaker_failures() -> int:
    try:
        v = int((os.environ.get("HUGPY_WORKER_BREAKER_FAILURES") or "").strip()
                or _FAILURES_DEFAULT)
        return v if v > 0 else _FAILURES_DEFAULT
    except (TypeError, ValueError):
        return _FAILURES_DEFAULT


def _breaker_cooldown_s() -> float:
    try:
        v = float((os.environ.get("HUGPY_WORKER_BREAKER_COOLDOWN_S") or "").strip()
                  or _COOLDOWN_DEFAULT_S)
        return v if v > 0 else _COOLDOWN_DEFAULT_S
    except (TypeError, ValueError):
        return _COOLDOWN_DEFAULT_S


def guard(key: str, *, url: str = "", force: bool = False) -> None:
    """Raise ``WorkerUnreachable`` if ``key``'s breaker is open.

    Half-open by single trial: once the cooldown elapses ONE caller is let
    through to find out whether the box came back; everyone else keeps failing
    fast until that trial resolves. Without the single-trial rule a burst of
    held calls would all pile onto a still-dead worker the instant the cooldown
    expired — re-creating the thread storm the breaker exists to prevent.

    ``force=True`` bypasses (the explicit "is it back yet" health probe).
    """
    if force or not _breaker_enabled():
        return
    now = time.monotonic()
    with _BREAKER_LOCK:
        st = _BREAKERS.get(key)
        if not st or not st.get("opened_at"):
            return
        elapsed = now - float(st["opened_at"])
        cooldown = _breaker_cooldown_s()
        if elapsed >= cooldown and not st.get("trial"):
            st["trial"] = True
            return
        retry_after = max(0.0, cooldown - elapsed)
        reason = st.get("reason") or "no answer"
    raise WorkerUnreachable(
        key,
        (f"worker is unreachable — {_breaker_failures()} consecutive failed "
         f"calls ({reason}); not retrying for {retry_after:.0f}s"),
        url=url, retry_after=retry_after, tripped=True)


def note_ok(key: str) -> None:
    """A call reached the worker. Closes the breaker and clears the streak."""
    with _BREAKER_LOCK:
        st = _BREAKERS.pop(key, None)
    if st and st.get("opened_at"):
        logger.info("worker_http: breaker CLOSED for %s (worker answered)", key)


def note_failure(key: str, exc: BaseException) -> None:
    """A call failed at the transport layer. Opens the breaker on the Nth in a row."""
    reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    threshold = _breaker_failures()
    with _BREAKER_LOCK:
        st = _BREAKERS.setdefault(key, {"fails": 0, "opened_at": 0.0, "reason": ""})
        st["fails"] = int(st.get("fails") or 0) + 1
        st["reason"] = reason
        st["trial"] = False
        newly_open = st["fails"] >= threshold and not st.get("opened_at")
        # A failed half-open trial re-arms the full cooldown from now.
        st["opened_at"] = time.monotonic() if st["fails"] >= threshold else 0.0
        fails = st["fails"]
    if newly_open:
        logger.warning("worker_http: breaker OPEN for %s after %d consecutive "
                       "transport failures (%s) — calls fail fast for %.0fs",
                       key, fails, reason, _breaker_cooldown_s())


def breaker_snapshot() -> Dict[str, Dict[str, Any]]:
    """Read-only view of every tracked worker's breaker — for introspection
    and tests. Never mutates."""
    now = time.monotonic()
    with _BREAKER_LOCK:
        out = {}
        for key, st in _BREAKERS.items():
            opened = float(st.get("opened_at") or 0.0)
            out[key] = {
                "fails": int(st.get("fails") or 0),
                "open": bool(opened) and (now - opened) < _breaker_cooldown_s(),
                "reason": st.get("reason") or "",
                "retry_after_s": (max(0.0, _breaker_cooldown_s() - (now - opened))
                                  if opened else 0.0),
            }
        return out


def reset_breakers() -> None:
    """Test seam / operator reset — forget every observed failure."""
    with _BREAKER_LOCK:
        _BREAKERS.clear()


# ── the calls ──────────────────────────────────────────────────────────────

def base_url(worker: Any) -> str:
    """The worker's base URL, from a registry row or a bare string."""
    if isinstance(worker, dict):
        return (worker.get("url") or "").rstrip("/")
    return (worker or "").rstrip("/")


def breaker_key(worker: Any) -> str:
    """What the breaker counts against. The registry id when we have one (a
    worker that changes address is still the same box), else the URL."""
    if isinstance(worker, dict):
        return str(worker.get("id") or base_url(worker) or "?")
    return base_url(worker) or "?"


def _url(worker: Any, path: str) -> str:
    base = base_url(worker)
    if not base:
        raise WorkerUnreachable(breaker_key(worker),
                                "worker has no callback url on record")
    return base + ("" if path.startswith("/") else "/") + path


def request(method: str, worker: Any, path: str, *,
            call: str = _DEFAULT_CALL, force: bool = False,
            read_timeout: Optional[float] = None,
            **kwargs: Any) -> httpx.Response:
    """One central→worker call, with split timeouts and the breaker applied.

    Returns the ``httpx.Response`` verbatim — an HTTP error status is DATA the
    caller decides about (the relay surface forwards the worker's own typed
    error body). Raises ``WorkerUnreachable`` only when the worker did not
    answer at all.

    ``timeout`` is derived from ``call`` (see ``timeout_for``); a caller may
    narrow the READ budget via ``read_timeout`` but nothing may pass a raw
    ``timeout=`` through — that scalar is precisely the defect being removed,
    so it is dropped if present.
    """
    key = breaker_key(worker)
    url = _url(worker, path)
    guard(key, url=url, force=force)
    kwargs.pop("timeout", None)
    try:
        resp = httpx.request(method.upper(), url,
                             timeout=timeout_for(call, read_timeout),
                             **kwargs)
    except TRANSPORT_ERRORS as exc:
        note_failure(key, exc)
        raise WorkerUnreachable(key, f"{type(exc).__name__}: {exc}",
                                url=url) from exc
    note_ok(key)
    return resp


def get(worker: Any, path: str, **kwargs: Any) -> httpx.Response:
    return request("GET", worker, path, **kwargs)


def post(worker: Any, path: str, **kwargs: Any) -> httpx.Response:
    return request("POST", worker, path, **kwargs)


@contextmanager
def stream(method: str, worker: Any, path: str, *,
           call: str = "relay", force: bool = False,
           **kwargs: Any) -> Iterator[httpx.Response]:
    """Streaming twin of ``request`` — same timeouts, same breaker bookkeeping.

    The breaker is settled at CONNECT time (an established stream that later
    dies mid-body is a generation failure, not an unreachable box), so the
    success is noted as soon as the response head arrives.
    """
    key = breaker_key(worker)
    url = _url(worker, path)
    guard(key, url=url, force=force)
    kwargs.pop("timeout", None)
    client = httpx.Client(timeout=timeout_for(call))
    try:
        with client.stream(method.upper(), url, **kwargs) as resp:
            note_ok(key)
            yield resp
    except TRANSPORT_ERRORS as exc:
        note_failure(key, exc)
        raise WorkerUnreachable(key, f"{type(exc).__name__}: {exc}",
                                url=url) from exc
    finally:
        client.close()


def async_client(call: str = "relay",
                 read_timeout: Optional[float] = None) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` carrying this module's timeouts.

    For the async relay path (managers.resolvers.remote), which owns its own
    stream/response handling and cannot use ``request`` directly. Callers there
    still ``guard()`` before dialing and ``note_ok``/``note_failure`` after, so
    the breaker sees the relay traffic too — that traffic is exactly what holds
    a thread for minutes.
    """
    return httpx.AsyncClient(timeout=timeout_for(call, read_timeout))


@contextmanager
def breaker_scope(worker: Any, *, force: bool = False) -> Iterator[str]:
    """Apply the breaker around a block that does its own HTTP (the async relay).

    Enters by ``guard()``; on a clean exit records success, on a transport
    error records the failure and re-raises the ORIGINAL exception — the async
    relay's callers classify httpx errors themselves (cold-hold vs honest
    refusal), so this must not swap the exception type out from under them.
    """
    key = breaker_key(worker)
    guard(key, url=base_url(worker), force=force)
    try:
        yield key
    except TRANSPORT_ERRORS as exc:
        note_failure(key, exc)
        raise
    else:
        note_ok(key)
