"""Optional worker capabilities: media/video runners and their process hooks.

The full worker must start on a box with only ``hugpy-fleet`` (+ engine)
installed and advertise only what it can actually run. Media and video
runners plug into the engine through ``hugpy_engine.tasks`` entry points
(group ``hugpy_engine.tasks``); this module loads them lazily, and — when
``hugpy_media`` is importable — installs the worker's PID-registry callbacks
on ``hugpy_media.hooks`` so Comfy/child processes media spawns are attributed
in the heartbeat instead of landing in ``pid_registry.unattributed``.

Nothing here imports ``hugpy_media`` or ``hugpy_video`` at module level.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from typing import Any, Dict, Optional

__all__ = [
    "MEDIA_TASKS",
    "load_worker_capabilities",
    "overlay_task_capabilities",
    "media_present",
    "video_present",
    "reset_for_tests",
]

log = logging.getLogger("hugpy_fleet.worker.plugins")

# Tasks whose runner lives in hugpy_media (everything in the engine's task map
# except the engine-native vision-chat task). Without hugpy_media on the box a
# worker must not advertise them, whatever ``find_spec`` says about torch.
MEDIA_TASKS = frozenset({
    "automatic-speech-recognition",
    "text-to-speech",
    "text-summarization",
    "keyword-extraction",
    "feature-extraction",
    "sentence-similarity",
    "text-to-image",
    "document-extraction",
    "url-extraction",
    "depth-estimation",
    "object-detection",
    "image-classification",
    "image-segmentation",
})

_lock = threading.Lock()
_state: Dict[str, Any] = {"loaded": False, "summary": None}


def _present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def media_present() -> bool:
    return _present("hugpy_media")


def video_present() -> bool:
    return _present("hugpy_video")


def _attach(hooks: Any, name: str, callback) -> bool:
    """Install ``callback`` on ``hooks.<name>`` whatever shape media chose:
    a ``register_<name>``/``set_<name>`` function, a list of callbacks, a
    callable registrar, or a plain settable slot."""
    for setter in (f"register_{name}", f"set_{name}", f"add_{name}"):
        fn = getattr(hooks, setter, None)
        if callable(fn):
            fn(callback)
            return True
    slot = getattr(hooks, name, None) if hasattr(hooks, name) else "__missing__"
    if isinstance(slot, list):
        if callback not in slot:
            slot.append(callback)
        return True
    if slot is None or slot == "__missing__":
        setattr(hooks, name, callback)
        return True
    if callable(slot):
        try:
            slot(callback)   # a registrar function named like the event
            return True
        except TypeError:
            setattr(hooks, name, callback)
            return True
    return False


def _install_media_hooks() -> Dict[str, Any]:
    """PID attribution for processes hugpy_media drives (Comfy, env-profile
    children) through ``hugpy_media.hooks.set_process_hooks``. Best-effort:
    a missing/older media package is not an error."""
    out: Dict[str, Any] = {"available": False, "installed": False}
    if not media_present():
        return out
    try:
        hooks = importlib.import_module("hugpy_media.hooks")
    except Exception as exc:  # noqa: BLE001 - older media without hooks
        log.info("hugpy_media.hooks not available: %s", exc)
        return out
    out["available"] = True
    from hugpy_fleet.worker import pid_registry

    def on_process_spawned(pid: int, label: str) -> None:
        # An external/child process now serves ``label`` (comfy, tts): an
        # active foreign call for that service, so reconcile() attributes the
        # PID's VRAM to it instead of leaving it unattributed.
        pid_registry.record_foreign_call(str(label or "media"), None, job_id=f"pid:{pid}")

    def on_foreign_call_started(service: str, model_key: Optional[str], job_id: Optional[str]) -> None:
        pid_registry.record_foreign_call(service, model_key, job_id=job_id)

    def on_foreign_call_ended(service: str, job_id: Optional[str]) -> None:
        pid_registry.end_foreign_call(service, job_id=job_id)

    setter = getattr(hooks, "set_process_hooks", None)
    try:
        if callable(setter):
            setter(on_process_spawned=on_process_spawned,
                   on_foreign_call_started=on_foreign_call_started,
                   on_foreign_call_ended=on_foreign_call_ended)
            out["installed"] = True
        else:  # pre-set_process_hooks media: plain module attributes
            for name, cb in (("on_process_spawned", on_process_spawned),
                             ("on_foreign_call_started", on_foreign_call_started),
                             ("on_foreign_call_ended", on_foreign_call_ended)):
                out[name] = _attach(hooks, name, cb)
            out["installed"] = all(out.get(n) for n in ("on_process_spawned",
                                                         "on_foreign_call_started",
                                                         "on_foreign_call_ended"))
    except Exception as exc:  # noqa: BLE001
        log.info("could not install media process hooks: %s", exc)
    return out


def load_worker_capabilities(force: bool = False) -> Dict[str, Any]:
    """Load every ``hugpy_engine.tasks`` entry point (through the engine's
    ``ensure_plugins_loaded``) and install media hooks.

    Idempotent (memoised) — call it once at worker start; safe to call again.
    Returns ``{"entry_points": <distinct plugin sources>, "tasks": [...],
    "media": bool, "video": bool, "media_hooks": {...}}``.
    """
    with _lock:
        if _state["loaded"] and not force:
            return dict(_state["summary"])
        from hugpy_engine import tasks as T
        try:
            # Idempotent, never raises: loads every ``hugpy_engine.tasks``
            # entry point (media/video runners) exactly once per process.
            T.ensure_plugins_loaded()
        except Exception as exc:  # noqa: BLE001 — plugin discovery never blocks boot
            log.warning("task entry points failed to load: %s", exc)
        registered = T.registered_tasks()
        summary = {
            "entry_points": len({spec.source for spec in registered.values() if spec.source}),
            "tasks": sorted(registered),
            "media": media_present(),
            "video": video_present(),
            "media_hooks": _install_media_hooks(),
        }
        _state["loaded"] = True
        _state["summary"] = summary
        log.info("worker capabilities: %d task plugin(s), media=%s video=%s, tasks=%s",
                 summary["entry_points"], summary["media"], summary["video"],
                 ",".join(summary["tasks"]) or "-")
        return dict(summary)


def overlay_task_capabilities(caps: Dict[str, bool]) -> Dict[str, bool]:
    """Gate media-backed tasks on the media package actually being present
    (or a runner having registered for the task). ``caps`` is the engine's
    dependency probe; the result is what the heartbeat advertises."""
    from hugpy_engine import tasks as T
    registered = set(T.registered_tasks())
    media_ok = media_present()
    out = dict(caps)
    for task in MEDIA_TASKS:
        if task in out:
            out[task] = bool(out[task]) and (task in registered or media_ok)
    return out


def reset_for_tests() -> None:
    with _lock:
        _state["loaded"] = False
        _state["summary"] = None
