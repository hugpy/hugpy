"""Can THIS box speak? — one answer, read by the runner and the heartbeat.

The chatterbox backend is seated one of two ways, and the two callers that need
to know must never disagree:

  * IN-PROCESS — ``chatterbox`` is importable in this interpreter. Cheap,
    and the only mode a box whose torch already matches the backend's pins can
    use.
  * PROFILE VENV — the fleet's per-model env-profile
    (``managers/serve/profiles.py``: ``<worker_root>/envs/<name>``) holds the
    backend under its own pinned torch, and synthesis runs as a CHILD of that
    venv's interpreter. This is how a worker seats chatterbox WITHOUT replacing
    the torch every other model on the box is served with.

``worker_agent.agent._task_capabilities`` overlays ``text-to-speech`` from this
module for the same reason it overlays whisper from a real import: the base
``task_deps`` probe is ``find_spec`` in the AGENT's interpreter, which answers
"no" for a profile-venv seat that works perfectly. Advertising is STRICT and
AFFIRMATIVE either way (``oracle.catalog._worker_seats_task``) — this returns
True only when a concrete interpreter has been shown to hold the package.

Deliberately stdlib-only and cached: it is consulted on EVERY heartbeat.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import time

#: The pip distribution and the import name (they differ — the classic probe
#: bug). Mirrors the k98 adapter's own constants; asserted equal by a test
#: rather than imported, because this module must stay import-cheap.
BACKEND_PIP = "chatterbox-tts"
BACKEND_PACKAGE = "chatterbox"
TASK = "text-to-speech"

#: Operator overrides. ``HUGPY_TTS_PYTHON`` names an interpreter outright;
#: ``HUGPY_TTS_PROFILE`` names the env-profile to look for (default below).
PYTHON_ENV = "HUGPY_TTS_PYTHON"
PROFILE_ENV = "HUGPY_TTS_PROFILE"
DEFAULT_PROFILE = "chatterbox-tts"

#: How long a resolved seat is trusted before it is re-checked. The check that
#: costs anything (spawning the profile python) must not run per heartbeat.
_TTL_S = 300.0
_CACHE: dict = {"at": 0.0, "seat": None}


def profile_name() -> str:
    return (os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE


def _importable_here() -> bool:
    try:
        return importlib.util.find_spec(BACKEND_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def _profile_python() -> str | None:
    """The env-profile interpreter for the TTS backend, or None when the fleet's
    profile machinery cannot name one on this box."""
    override = (os.environ.get(PYTHON_ENV) or "").strip()
    if override:
        return override if os.path.isfile(override) else None
    try:
        from hugpy_engine.serve import profiles
        candidate = profiles.profile_python(profile_name())
    except Exception:  # noqa: BLE001 — no profile machinery -> no profile seat
        return None
    return candidate if os.path.isfile(candidate) else None


def _child_can_import(python: str) -> tuple[bool, str]:
    """Ask the interpreter itself. ``find_spec`` in a CHILD, not an import: the
    same cheap question this module asks of its own process, asked of the one
    that would actually do the work."""
    try:
        proc = subprocess.run(
            [python, "-c",
             "import importlib.util,sys;"
             "sys.exit(0 if importlib.util.find_spec('chatterbox') else 3)"],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 — a probe never becomes the failure
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, ""
    if proc.returncode == 3:
        return False, (f"{python} does not have {BACKEND_PACKAGE!r} "
                       f"(pip install {BACKEND_PIP} into that venv)")
    return False, ((proc.stderr or proc.stdout or "").strip()[-300:]
                   or f"probe exited {proc.returncode}")


def resolve(*, refresh: bool = False) -> dict:
    """``{available, mode, python, reason, profile}`` for the TTS backend.

    ``mode`` is ``"in-process"`` | ``"profile-venv"`` | ``""``. ``reason`` is
    empty exactly when ``available`` is True.
    """
    now = time.monotonic()
    cached = _CACHE.get("seat")
    if cached is not None and not refresh and (now - _CACHE["at"]) < _TTL_S:
        return dict(cached)

    name = profile_name()
    if _importable_here():
        seat = {"available": True, "mode": "in-process", "python": "",
                "reason": "", "profile": name}
    else:
        python = _profile_python()
        if not python:
            seat = {
                "available": False, "mode": "", "python": "", "profile": name,
                "reason": (
                    f"python package {BACKEND_PACKAGE!r} is not importable in "
                    f"this interpreter and no env-profile {name!r} venv exists "
                    f"on this box (materialize one — "
                    f"managers.serve.profiles.materialize({name!r}, "
                    f"[{BACKEND_PIP!r}]) — or set {PYTHON_ENV})"),
            }
        else:
            ok, why = _child_can_import(python)
            seat = {
                "available": ok, "mode": "profile-venv" if ok else "",
                "python": python if ok else "", "profile": name,
                "reason": "" if ok else why,
            }
    _CACHE["seat"], _CACHE["at"] = seat, now
    return dict(seat)


def available(*, refresh: bool = False) -> bool:
    """Whether this box can actually synthesize speech. What the heartbeat
    advertises as ``task_capabilities['text-to-speech']``."""
    return bool(resolve(refresh=refresh).get("available"))


__all__ = ["BACKEND_PACKAGE", "BACKEND_PIP", "DEFAULT_PROFILE", "PROFILE_ENV",
           "PYTHON_ENV", "TASK", "available", "profile_name", "resolve"]
