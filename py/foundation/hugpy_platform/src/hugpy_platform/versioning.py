"""Best-effort version lookup for a running package."""

from __future__ import annotations

from importlib import import_module


def module_version(module_name: str) -> str:
    """Return a package's declared version or ``unknown`` when unavailable."""
    try:
        return str(import_module(module_name).__version__)
    except Exception:  # noqa: BLE001 — version logging is best-effort
        return "unknown"
