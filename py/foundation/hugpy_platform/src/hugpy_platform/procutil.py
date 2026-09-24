"""Portable process spawn / terminate / re-exec.

The Linux build leaned on POSIX-only primitives:

  * ``subprocess.Popen(..., start_new_session=True)`` to detach a child so it
    survives a gunicorn worker reload — Windows rejects that kwarg.
  * ``os.killpg(os.getpgid(pid), SIGTERM)`` to take down a child and anything it
    spawned — ``os.getpgid``/``killpg`` don't exist on Windows.
  * ``os.execv(sys.executable, ...)`` to restart in place — no Windows analogue.

This module wraps each so call sites stay one-liners and the OS branch lives in
exactly one place.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Optional, Sequence

from hugpy_platform.platform_facade import IS_WINDOWS


def pid_alive(pid: int) -> bool:
    """Treat only a definite missing process as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 — uncertainty must not reap live work
        return True
    return True


def popen_detached(argv: Sequence[str], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` that starts in its own session/process group.

    On POSIX the child gets a new session (``start_new_session=True``) so it
    outlives a parent reload and can later be torn down as a group. On Windows we
    use ``CREATE_NEW_PROCESS_GROUP`` (the closest analogue) instead of the
    POSIX-only kwarg, which would otherwise raise ``ValueError``.
    """
    if IS_WINDOWS:
        flags = kwargs.pop("creationflags", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(list(argv), **kwargs)


def terminate_tree(proc, sig: int = signal.SIGTERM) -> None:
    """Terminate ``proc`` and the children it spawned, best-effort.

    Accepts a :class:`subprocess.Popen` or a :class:`multiprocessing.Process`
    (anything exposing ``pid`` and, ideally, ``terminate``). Never raises if the
    process is already gone or we lack permission.

    POSIX: signal the whole process group (matches the old
    ``killpg(getpgid(pid), SIGTERM)`` behaviour). Windows: ``taskkill /T`` to walk
    the child tree, falling back to ``proc.terminate()``.
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            return
        except Exception:
            pass
        _safe_terminate(proc)
        return
    # POSIX: try the process group first, then the bare pid.
    #
    # NEVER signal our OWN process group (k119). A spawned child calls
    # os.setpgrp() as its first statement, but there is a real window between
    # Process.start() and that call — under `spawn` the child re-imports the
    # whole package first, which on a busy virtiofs box takes tens of seconds.
    # Signal the group in that window and getpgid(child) is still the PARENT's
    # group: killpg then SIGTERMs the caller too. Observed 2026-08-21 02:47:15 —
    # the download stall killer fired during a slow child import and took
    # hugpy-downloader-dev down with the transfer it meant to kill. Fall back to
    # the bare pid, which is always correct and merely less thorough.
    try:
        pgid = os.getpgid(pid)
        if pgid != os.getpgrp():
            os.killpg(pgid, sig)
            return
    except (ProcessLookupError, PermissionError):
        return
    except (AttributeError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _safe_terminate(proc) -> None:
    term = getattr(proc, "terminate", None)
    if callable(term):
        try:
            term()
        except Exception:
            pass


def _module_argv() -> "list[str] | None":
    """argv that relaunches a ``python -m pkg`` invocation correctly.

    Under ``-m``, ``sys.argv[0]`` is the path to the package's ``__main__.py``;
    exec'ing that runs it as a plain script with no package context, so every
    relative import inside dies (the post-self-update crash workers hit). The
    main module's ``__spec__`` remembers the real module name — rebuild the
    ``-m`` form from it. None when the process wasn't started with ``-m``.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    name = getattr(spec, "name", None)
    if not name:
        return None
    if name.endswith(".__main__"):
        name = name[: -len(".__main__")]
    return [sys.executable, "-m", name] + sys.argv[1:]


def reexec(argv: Optional[Sequence[str]] = None) -> "int | None":
    """Restart the current interpreter with ``argv`` (default: this process).

    POSIX replaces the image via ``os.execv`` and never returns. Windows has no
    exec; we spawn a fresh process and exit, which is what callers (the worker
    self-update) want anyway. Returns nothing on POSIX; calls ``sys.exit`` on
    Windows.
    """
    if argv is not None:
        args = list(argv)
    else:
        args = _module_argv() or [sys.executable] + sys.argv
    if IS_WINDOWS:
        proc = subprocess.Popen(args)
        sys.exit(proc.wait())
    os.execv(args[0], args)
