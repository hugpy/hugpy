"""Per-OS application directories — one source of truth.

Replaces the scattered hardcoded ``/srv/abstractendeavors/...``,
``~/.local/share/hugpy``, ``/etc/llama-swap``, and ``/mnt/llm_storage`` literals.
Every path is overridable by the same env vars the code already honoured, so
existing Linux deployments are unaffected; only the *defaults* become per-OS:

    data_dir()    Linux ~/.local/share/hugpy   macOS ~/Library/Application Support/hugpy   Windows %LOCALAPPDATA%\\hugpy
    config_dir()  Linux ~/.config/hugpy        macOS ~/Library/Application Support/hugpy   Windows %LOCALAPPDATA%\\hugpy
    cache_dir()   Linux ~/.cache/hugpy         macOS ~/Library/Caches/hugpy                Windows %LOCALAPPDATA%\\hugpy\\Cache
    engine_dir()  data_dir()/engine            — where the fetched llama.cpp binary lands
    models_root() DEFAULT_ROOT or data_dir()/llm_storage

We use ``platformdirs`` when available (added to base deps) and fall back to a
hand-rolled per-OS layout so this module never hard-fails on import.
"""
from __future__ import annotations

import os

from hugpy_platform.platform_facade import IS_MACOS, IS_WINDOWS, env_value

_APP = "hugpy"


def _home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _fallback_data() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP)
    if IS_MACOS:
        return _home("Library", "Application Support", _APP)
    return os.path.join(os.environ.get("XDG_DATA_HOME") or _home(".local", "share"), _APP)


def _fallback_config() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP)
    if IS_MACOS:
        return _home("Library", "Application Support", _APP)
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or _home(".config"), _APP)


def _fallback_cache() -> str:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, _APP, "Cache")
    if IS_MACOS:
        return _home("Library", "Caches", _APP)
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or _home(".cache"), _APP)


def _dirs():
    try:
        import platformdirs

        return platformdirs.PlatformDirs(_APP, appauthor=False)
    except Exception:
        return None


def data_dir() -> str:
    override = env_value("HUGPY_DATA_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_data_dir if d else _fallback_data())


def config_dir() -> str:
    override = env_value("HUGPY_CONFIG_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_config_dir if d else _fallback_config())


def cache_dir() -> str:
    override = env_value("HUGPY_CACHE_DIR")
    if override:
        return _ensure(override)
    d = _dirs()
    return _ensure(d.user_cache_dir if d else _fallback_cache())


def engine_dir() -> str:
    """Where ``hugpy install-engine`` unpacks the native llama.cpp binaries."""
    override = env_value("HUGPY_ENGINE_DIR") or env_value("LLAMA_CPP_DIR")
    if override:
        return _ensure(override)
    return _ensure(os.path.join(data_dir(), "engine"))


def models_root() -> str:
    """Model/upload/dataset storage root.

    Honours the legacy ``DEFAULT_ROOT``/``MODELS_HOME`` env vars first — but only
    if that path can actually be created and written. A stale/un-writable override
    (e.g. ``DEFAULT_ROOT=/mnt/llm_storage`` carried in a server ``.env`` onto a
    worker or a phone where ``/mnt`` is read-only) is ignored in favour of a
    per-user dir under ``data_dir()``, so storage never lands on a dead path.
    """
    override = env_value("DEFAULT_ROOT")
    if override and _usable(override):
        return override
    # Preserve the historical Linux mount when it exists and is writable.
    legacy = "/mnt/llm_storage"
    try:
        if os.path.isdir(legacy) and os.access(legacy, os.W_OK):
            return legacy
    except OSError:
        pass
    return _ensure(os.path.join(data_dir(), "llm_storage"))


def demo_media_base() -> str:
    """Base URL the video arm's canned demo loads its sample media from."""
    return env_value("HUGPY_DEMO_MEDIA_BASE") or "https://hugpy.ai/demo-media"


def demo_media_dir() -> str:
    """Local demo-media tree to serve at ``/demo-media/`` (self-hosters).

    Empty string means "not configured" — deliberately NO default and NO
    directory creation; the ``/demo-media/`` route only exists when this is set.
    """
    return env_value("HUGPY_DEMO_MEDIA_DIR") or ""


def _usable(path: str) -> bool:
    """True only if *path* exists (or can be created) AND is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.isdir(path) and os.access(path, os.W_OK)
    except OSError:
        return False


def _ensure(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------------
# hugpy runtime "mechanic" files  (HUGPY_HOME, default ~/.hugpy)
# ---------------------------------------------------------------------------
# Historically these state/config/log files were written straight into $HOME
# (``~/todo.json``, ``~/steward-state.json``, ``~/.abstract_hugpy_worker.json``,
# ``~/model_metadata.db`` …), strewing the home directory. They now live under a
# single base dir — ``HUGPY_HOME`` (default ``~/.hugpy``) — split into::
#
#     state/   todo.json, steward-state.json, model_metadata.db, flow.json, …
#     config/  abstract_hugpy_worker.json (+ .settings.json/.update.json)
#     logs/    bridge-mail.jsonl, bugreport.json
#     run/     *.lock
#
# Each named accessor MIGRATES a legacy ``~/<name>`` file into its new home the
# first time it is resolved (atomic same-fs rename, cross-fs copy fallback), so
# upgrades are seamless and idempotent — no flag-day, no data loss. Every path
# stays overridable so existing deployments and tests can pin locations.
#
# NOTE: standalone station-stack scripts (keeper_relay.py, bugreport/scanner.py)
# run under the system interpreter and cannot import this package; they carry a
# byte-identical inline resolver. Keep the two in sync — this module is the spec.

def hugpy_home() -> str:
    """Single base directory for hugpy's runtime files. Default ``~/.hugpy``."""
    return _ensure(env_value("HUGPY_HOME") or _home(".hugpy"))


def hugpy_state_dir() -> str:
    return _ensure(os.path.join(hugpy_home(), "state"))


def hugpy_config_dir() -> str:
    return _ensure(os.path.join(hugpy_home(), "config"))


def hugpy_logs_dir() -> str:
    return _ensure(os.path.join(hugpy_home(), "logs"))


def hugpy_run_dir() -> str:
    return _ensure(os.path.join(hugpy_home(), "run"))


def _relocate(new_path: str, *legacy_names: str) -> str:
    """Return *new_path*, first migrating a legacy ``~/<name>`` into it once.

    Idempotent and safe: if *new_path* already exists nothing moves. Uses an
    atomic rename on the same filesystem and falls back to a copy+unlink across
    filesystems. Any failure leaves the legacy file untouched and returns the
    new path anyway (a fresh file is then created there).
    """
    try:
        if os.path.exists(new_path):
            return new_path
        for name in legacy_names:
            old = _home(name)
            if not os.path.exists(old):
                continue
            if os.path.abspath(old) == os.path.abspath(new_path):
                continue
            try:
                os.replace(old, new_path)          # atomic, same filesystem
            except OSError:
                import shutil
                shutil.move(old, new_path)          # cross-filesystem fallback
            break
    except OSError:
        pass
    return new_path


# --- named accessors --------------------------------------------------------

def worker_id_file() -> str:
    """``config/abstract_hugpy_worker.json`` — the worker identity file.

    Honors an explicit ``WORKER_ID_FILE`` override (used by tests / multi-worker
    hosts); otherwise migrates the legacy ``~/.abstract_hugpy_worker.json`` plus
    its ``.settings.json`` / ``.update.json`` sidecars into ``config/``.
    """
    override = env_value("WORKER_ID_FILE")
    if override:
        return override
    new = os.path.join(hugpy_config_dir(), "abstract_hugpy_worker.json")
    _relocate(new, ".abstract_hugpy_worker.json")
    for suffix in (".settings.json", ".update.json"):
        _relocate(new + suffix, ".abstract_hugpy_worker.json" + suffix)
    return new


def gguf_worker_id_file() -> str:
    """``config/gguf_worker.json`` — sibling worker identity (same pattern)."""
    override = env_value("WORKER_ID_FILE")
    if override:
        return override
    new = os.path.join(hugpy_config_dir(), "gguf_worker.json")
    return _relocate(new, ".gguf_worker.json")


def todo_file() -> str:
    return _relocate(os.path.join(hugpy_state_dir(), "todo.json"), "todo.json")


def todo_history_file() -> str:
    return _relocate(
        os.path.join(hugpy_state_dir(), "todo-history.jsonl"), "todo-history.jsonl"
    )


def todo_lock() -> str:
    return _relocate(os.path.join(hugpy_run_dir(), "todo.lock"), ".todo.lock")


def steward_config() -> str:
    return _relocate(os.path.join(hugpy_config_dir(), "steward.json"), "steward.json")


def steward_state() -> str:
    return _relocate(
        os.path.join(hugpy_state_dir(), "steward-state.json"), "steward-state.json"
    )


def model_metadata_db() -> str:
    return _relocate(
        os.path.join(hugpy_state_dir(), "model_metadata.db"), "model_metadata.db"
    )


def model_physical_json() -> str:
    # The ``.lock`` sidecar is derived by callers as ``path + ".lock"`` and rides
    # along in state/ next to the file — intentionally not split into run/.
    return _relocate(
        os.path.join(hugpy_state_dir(), "model_physical.json"), "model_physical.json"
    )


def flow_json() -> str:
    return _relocate(os.path.join(hugpy_state_dir(), "flow.json"), "flow.json")


def bridge_mail() -> str:
    return _relocate(
        os.path.join(hugpy_logs_dir(), "bridge-mail.jsonl"), ".bridge-mail.jsonl"
    )


def bugreport_json() -> str:
    return _relocate(os.path.join(hugpy_logs_dir(), "bugreport.json"), "bugreport.json")


def ensure_hugpy_home() -> str:
    """Create the full ``HUGPY_HOME`` skeleton. Called by installers/postinst."""
    hugpy_state_dir(); hugpy_config_dir(); hugpy_logs_dir(); hugpy_run_dir()
    return hugpy_home()
