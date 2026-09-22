"""Is the HTTP client that started this request still connected?

WSGI has no disconnect API. Flask/Werkzeug cannot tell a view whether the caller
is still on the other end of the socket, and for a NON-STREAMING request there is
not even a write to fail on — the response is assembled and sent once, at the
end. So a request whose caller gave up at 60s keeps its gunicorn thread for as
long as the work runs (up to HUGPY_COLD_HOLD_MAX_S on a held cold load: 25
minutes on the live unit), serving nobody. With `--workers 3 --threads 8` the
site only has 24 threads; a couple of dozen abandoned holds is the whole site.

What IS available is the server's own socket. Gunicorn puts it in the WSGI
environ as ``gunicorn.socket`` (gunicorn/http/wsgi.py: default_environ), and a
zero-timeout ``select`` + a one-byte ``MSG_PEEK`` answers the only question that
matters:

  * not readable                 -> connected (nothing to say)
  * readable and peek returns b"" -> the peer sent FIN: GONE
  * readable and peek returns data -> a pipelined next request: connected

That is deliberately CONSERVATIVE in one direction only. Every uncertainty —
another server (waitress / the Flask dev server) that publishes no socket, a TLS
socket where MSG_PEEK is not permitted, a select/recv that errors for any reason
we did not explicitly classify — reads as CONNECTED. Abandoning a live caller's
work would be a far worse bug than holding a dead one's slot, so the probe never
guesses in that direction.

Behind nginx (the dev/prod topology) this is exact: with the default
``proxy_ignore_client_abort off`` nginx closes the upstream connection when the
browser aborts, and gunicorn's socket sees the FIN.

The probe is bound to a thread-local by a Flask ``before_request`` hook
(``install(app)``) so any code running ON the request thread can find it —
notably ``_platform.async_runtime``, which is where a WSGI thread blocks waiting
for loop work and is therefore the one place that can give the slot back.

Stdlib-only at import time (flask is imported lazily, inside ``install``), so
core modules may import this freely.
"""
from __future__ import annotations

import os
import errno
import select
import socket
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_LOCAL = threading.local()

# errnos that mean the peer is definitively gone (anything else = keep serving).
_DEAD_ERRNOS = frozenset((
    errno.ECONNRESET, errno.EPIPE, errno.ENOTCONN, errno.ESHUTDOWN, errno.EBADF,
))


class ClientGone(Exception):
    """The caller disconnected and the work it started was cancelled.

    Not an error in the system: nothing failed, there is simply nobody left to
    answer. Routes turn this into a quiet 499-style reply (which no one reads)
    rather than logging a 500.
    """


def enabled() -> bool:
    """Abandon-on-disconnect is ON by default; ``HUGPY_CLIENT_DISCONNECT_ABANDON=off``
    restores the old behaviour (hold the thread until the work finishes)."""
    return (os.environ.get("HUGPY_CLIENT_DISCONNECT_ABANDON", "").strip().lower()
            not in ("off", "0", "false", "no"))


def poll_s() -> float:
    """How often a blocked WSGI thread re-checks its client. Default 2s — the
    cost is one syscall per in-flight request per 2s (nothing at 24 threads),
    and it bounds how long a dead client's slot stays occupied."""
    raw = (os.environ.get("HUGPY_CLIENT_DISCONNECT_POLL_S") or "").strip()
    if not raw:
        return 2.0
    try:
        v = float(raw)
        return v if v > 0 else 2.0
    except (TypeError, ValueError):
        return 2.0


class SocketProbe:
    """A one-question probe over a server socket: has the peer hung up?

    ``gone()`` is safe to call from a thread other than the one serving the
    request (the WSGI thread is blocked, nobody else reads this socket), and it
    latches: once GONE, always GONE.
    """

    __slots__ = ("_sock", "_gone")

    def __init__(self, sock: Any):
        self._sock = sock
        self._gone = False

    def gone(self) -> bool:
        if self._gone:
            return True
        sock = self._sock
        try:
            readable, _w, _x = select.select([sock], [], [], 0)
        except Exception:       # noqa: BLE001 — can't tell ⇒ still connected
            return False
        if not readable:
            return False        # nothing pending: the peer is just quiet
        try:
            data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError as exc:
            if exc.errno in _DEAD_ERRNOS:
                self._gone = True
                return True
            return False        # unclassified ⇒ still connected
        except Exception:       # noqa: BLE001 — never abandon on doubt
            return False
        if data == b"":
            self._gone = True   # FIN from the peer — the only positive signal
            return True
        return False            # pipelined bytes: very much still connected


def probe_for_environ(environ: Optional[dict]) -> Optional[SocketProbe]:
    """A probe for this request's client, or None when we cannot honestly tell.

    None (⇒ "assume connected forever", today's behaviour) for: the feature
    switched off, a WSGI server that publishes no socket (waitress, the Flask
    dev server, the worker agent), a TLS socket (``MSG_PEEK`` raises on
    ``ssl.SSLSocket``, and a raising probe must never be read as a disconnect),
    or an already-closed fd.
    """
    if not enabled():
        return None
    sock = (environ or {}).get("gunicorn.socket")
    if sock is None:
        return None
    try:
        import ssl
        if isinstance(sock, ssl.SSLSocket):
            return None
    except Exception:           # noqa: BLE001 — no ssl module ⇒ not a TLS socket
        pass
    if not hasattr(sock, "recv") or not hasattr(sock, "fileno"):
        return None
    try:
        if sock.fileno() < 0:
            return None
    except Exception:           # noqa: BLE001
        return None
    return SocketProbe(sock)


# ---------------------------------------------------------------------------
# Thread-local binding — set on the request thread, read on the request thread.
# ---------------------------------------------------------------------------

def bind(probe: Optional[SocketProbe]) -> None:
    _LOCAL.probe = probe


def clear() -> None:
    _LOCAL.probe = None


def current() -> Optional[SocketProbe]:
    return getattr(_LOCAL, "probe", None)


def current_checker() -> Optional[Callable[[], bool]]:
    """A zero-arg "has my caller hung up?" for the CURRENT thread, or None when
    this thread is not serving a probeable HTTP request (background work,
    internal drains, the worker agent) — in which case nothing changes."""
    probe = current()
    if probe is None:
        return None
    return probe.gone


def install(app) -> None:
    """Bind a probe for the life of each request on a Flask app (idempotent)."""
    if getattr(app, "_client_liveness_installed", False):
        return
    app._client_liveness_installed = True

    from flask import request as _request     # lazy: keeps this module stdlib-only

    @app.before_request
    def _bind_client_probe():                 # noqa: ANN202 — flask hook
        try:
            bind(probe_for_environ(_request.environ))
        except Exception:                     # noqa: BLE001 — never break a request
            bind(None)
        return None

    @app.teardown_request
    def _clear_client_probe(_exc=None):       # noqa: ANN202 — flask hook
        clear()

    logger.info("client-liveness probe installed (abandon=%s, poll=%.1fs)",
                enabled(), poll_s())
