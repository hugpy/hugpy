"""Is a downloader daemon actually running?

The API no longer downloads anything, so a queued job with no daemon behind it
would sit at 0% forever with a cheerful "queued" and no explanation. That is the
one failure mode this separation must not introduce: never silently dead.

The daemon touches a heartbeat file every few seconds; anyone can ask
``downloader_alive()``. A file, not a mirror row, deliberately:

  * it costs one ``os.stat`` on a read path that runs per /jobs poll,
  * it cannot be confused with a JOB and so can never show up in a queue view,
  * a hard-killed daemon stops touching it with no cleanup step to get wrong.

It lives next to the comms DB (same directory the API and the daemon already
must share), so wiring HUGPY_COMMS_DB wires this too.
"""
from __future__ import annotations

import os
import time

# How stale the heartbeat may get before we call the daemon dead. The daemon
# beats every BEAT_INTERVAL; the window is generous enough that a busy poll loop
# or a slow shared mount never produces a false "no downloader".
STALE_SECONDS = 30.0
BEAT_INTERVAL = 5.0


def heartbeat_path() -> str:
    """``<dir of the comms DB>/hugpy-downloader.heartbeat``."""
    from hugpy_control.shared import default_db_path
    db = default_db_path()
    parent = os.path.dirname(db) or "/tmp"
    return os.path.join(parent, "hugpy-downloader.heartbeat")


def beat(owner: str = "") -> None:
    """Stamp the heartbeat. Best-effort — a daemon must never die because it
    could not write a status file."""
    path = heartbeat_path()
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{owner}\n{time.time()}\n")
    except OSError:
        pass


def clear() -> None:
    """Remove the heartbeat on a clean shutdown, so "no downloader" is immediate
    rather than waiting out STALE_SECONDS."""
    try:
        os.remove(heartbeat_path())
    except OSError:
        pass


def last_beat() -> float | None:
    """Epoch seconds of the last beat, or None if there has never been one."""
    try:
        return os.path.getmtime(heartbeat_path())
    except OSError:
        return None


def downloader_alive() -> bool:
    """True when a daemon has beaten within STALE_SECONDS. Fail-safe direction:
    an unreadable heartbeat reads as NOT alive, so the honest "waiting for
    downloader" message is what an operator sees when anything is wrong."""
    ts = last_beat()
    return ts is not None and (time.time() - ts) <= STALE_SECONDS
