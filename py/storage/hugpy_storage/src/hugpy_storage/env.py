"""Environment parsing shared by storage status readers."""

from __future__ import annotations

import os


def env_float(name: str, default: float, *, logger) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.debug("ignoring non-numeric %s=%r", name, raw)
        return default
