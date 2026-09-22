"""Where fleet-central state lives (workers registry, tokens, phone bricks, ...).

The central stores used to read ``settings.manifest_path`` from the server's
schema aggregator. That made fleet depend on ``hugpy_server``. The roots now
come from ``hugpy_platform`` (env-backed constants) and are injectable:

* ``HUGPY_FLEET_STATE_DIR`` — directory for every JSON/SQLite state file this
  package owns. Default: the directory of the model manifest (historically
  ``<DEFAULT_ROOT>/projects``), so an existing deployment keeps its files.
* ``settings`` — a process-wide, mutable object (``manifest_path``,
  ``storage_root``). Tests point it at a temporary directory
  (``settings.manifest_path = tmp/model_manifest.json``) or call
  :func:`configure`; :func:`reset` restores the environment-derived values.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "FleetSettings",
    "settings",
    "configure",
    "reset",
    "state_dir",
    "state_path",
    "manifest_path",
    "storage_root",
]


def _platform_defaults() -> tuple[str, str]:
    """(manifest_path, storage_root) from hugpy_platform, env-first."""
    try:
        from hugpy_platform.constants import DEFAULT_ROOT, MODELS_DICT_PATH
        return str(MODELS_DICT_PATH), str(DEFAULT_ROOT)
    except Exception:  # noqa: BLE001 — platform constants unavailable: home dir
        root = os.environ.get("DEFAULT_ROOT") or os.path.expanduser("~/.hugpy")
        return os.path.join(root, "projects", "model_manifest.json"), root


class FleetSettings:
    """Mutable, process-wide roots. Attribute assignment is the injection
    point (``settings.manifest_path = ...``), mirroring the old server
    ``settings`` object so existing tests and helpers keep working."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        manifest, root = _platform_defaults()
        self.manifest_path: str = manifest
        self.storage_root: str = root
        self.state_dir_override: Optional[str] = None

    @property
    def state_dir(self) -> str:
        if self.state_dir_override:
            return self.state_dir_override
        env = (os.environ.get("HUGPY_FLEET_STATE_DIR") or "").strip()
        if env:
            return os.path.expanduser(env)
        return os.path.dirname(self.manifest_path) or "."


settings = FleetSettings()


def configure(*, state_dir: Optional[str] = None,
              manifest_path: Optional[str] = None,
              storage_root: Optional[str] = None) -> FleetSettings:
    """Inject roots (tests, embedding hosts). ``None`` leaves a value alone."""
    if state_dir is not None:
        settings.state_dir_override = state_dir or None
    if manifest_path is not None:
        settings.manifest_path = manifest_path
    if storage_root is not None:
        settings.storage_root = storage_root
    return settings


def reset() -> FleetSettings:
    settings.reset()
    return settings


def state_dir() -> str:
    return settings.state_dir


def state_path(*parts: str) -> str:
    return os.path.join(settings.state_dir, *parts)


def manifest_path() -> str:
    return settings.manifest_path


def storage_root() -> str:
    return settings.storage_root
