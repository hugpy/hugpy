"""Process/telemetry hooks: how the fleet worker observes media-spawned work.

Media runners sometimes drive an EXTERNAL GPU process (ComfyUI) or spawn a
child interpreter (the TTS profile venv). The fleet worker wants to know about
those so its PID registry can attribute VRAM to the model that was called for.
``hugpy_media`` must not import ``hugpy_fleet`` (lower layer reaching up), so
this module is the seam: a tiny callback registry with no-op defaults.

Protocol (all callbacks are best-effort; exceptions are swallowed so telemetry
can never perturb a generation):

    on_process_spawned(pid: int, label: str) -> None
        A child/external process ``pid`` now serves ``label`` (e.g. "comfy",
        "tts").

    on_foreign_call_started(service: str, model_key: str | None, job_id: str | None) -> None
        A call to the external ``service`` began on behalf of ``model_key``.

    on_foreign_call_ended(service: str, job_id: str | None) -> None
        That call finished (success or failure).

The fleet worker installs its implementations at boot:

    from hugpy_media import hooks
    hooks.set_process_hooks(
        on_process_spawned=pid_registry.record_child,
        on_foreign_call_started=pid_registry.record_foreign_call,
        on_foreign_call_ended=pid_registry.end_foreign_call,
    )

Anything not installed stays a no-op, so central / a no-GPU box behaves
exactly as before.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Protocol

__all__ = [
    "ProcessHooks",
    "get_process_hooks",
    "set_process_hooks",
    "reset_process_hooks",
    "on_process_spawned",
    "on_foreign_call_started",
    "on_foreign_call_ended",
]

logger = logging.getLogger(__name__)

SpawnedHook = Callable[[int, str], None]
CallStartedHook = Callable[[str, Optional[str], Optional[str]], None]
CallEndedHook = Callable[[str, Optional[str]], None]


class ProcessHooks(Protocol):
    def on_process_spawned(self, pid: int, label: str) -> None: ...
    def on_foreign_call_started(self, service: str, model_key: Optional[str],
                                job_id: Optional[str]) -> None: ...
    def on_foreign_call_ended(self, service: str, job_id: Optional[str]) -> None: ...


class _NoopHooks:
    """Default: nothing is observing. Every method is a no-op."""

    def on_process_spawned(self, pid: int, label: str) -> None:
        return None

    def on_foreign_call_started(self, service: str, model_key: Optional[str],
                                job_id: Optional[str]) -> None:
        return None

    def on_foreign_call_ended(self, service: str, job_id: Optional[str]) -> None:
        return None


class _CallbackHooks:
    """Adapter from three optional plain callables to the ProcessHooks shape."""

    def __init__(self, spawned: Optional[SpawnedHook], started: Optional[CallStartedHook],
                 ended: Optional[CallEndedHook]) -> None:
        self._spawned = spawned
        self._started = started
        self._ended = ended

    def on_process_spawned(self, pid: int, label: str) -> None:
        if self._spawned is not None:
            self._spawned(pid, label)

    def on_foreign_call_started(self, service: str, model_key: Optional[str],
                                job_id: Optional[str]) -> None:
        if self._started is not None:
            self._started(service, model_key, job_id)

    def on_foreign_call_ended(self, service: str, job_id: Optional[str]) -> None:
        if self._ended is not None:
            self._ended(service, job_id)


_lock = threading.RLock()
_hooks: ProcessHooks = _NoopHooks()


def get_process_hooks() -> ProcessHooks:
    with _lock:
        return _hooks


def set_process_hooks(
    hooks: Optional[ProcessHooks] = None,
    *,
    on_process_spawned: Optional[SpawnedHook] = None,
    on_foreign_call_started: Optional[CallStartedHook] = None,
    on_foreign_call_ended: Optional[CallEndedHook] = None,
) -> ProcessHooks:
    """Install the observer. Pass a full ``ProcessHooks`` object, or any subset
    of the three callbacks by keyword (the rest stay no-ops). Returns the
    previously installed hooks so a caller can restore them."""
    global _hooks
    if hooks is None:
        hooks = _CallbackHooks(on_process_spawned, on_foreign_call_started,
                               on_foreign_call_ended)
    with _lock:
        previous = _hooks
        _hooks = hooks
    return previous


def reset_process_hooks() -> None:
    """Back to the no-op default (tests, worker shutdown)."""
    global _hooks
    with _lock:
        _hooks = _NoopHooks()


def _safe(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - telemetry must never break a generation
        logger.debug("media process hook %s failed", getattr(fn, "__name__", fn), exc_info=True)


def on_process_spawned(pid: int, label: str) -> None:
    """Report a spawned/attached process. Never raises."""
    _safe(get_process_hooks().on_process_spawned, pid, label)


def on_foreign_call_started(service: str, model_key: Optional[str] = None,
                            job_id: Optional[str] = None) -> None:
    """Report the start of a call into an external service. Never raises."""
    _safe(get_process_hooks().on_foreign_call_started, service, model_key, job_id)


def on_foreign_call_ended(service: str, job_id: Optional[str] = None) -> None:
    """Report the end of a call into an external service. Never raises."""
    _safe(get_process_hooks().on_foreign_call_ended, service, job_id)
