"""Version snapshot for worker environment reports."""

from __future__ import annotations

import os
import platform
from importlib.metadata import version


def environment_status(packages: tuple[str, ...]) -> dict:
    """Report the tier, Python version, and installed requested packages."""
    tier = (os.environ.get("WORKER_ENV_TIER") or "stable").strip().lower()
    info = {"tier": tier or "stable", "python": platform.python_version()}
    for package in packages:
        try:
            info[package] = version(package)
        except Exception:  # noqa: BLE001 — absent package is unreported
            pass
    return info
