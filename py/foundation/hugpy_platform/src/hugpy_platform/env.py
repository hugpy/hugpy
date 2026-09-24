"""Small environment-value parsers shared by package adapters."""

from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    value = value.strip() if value else ""
    return value or None
