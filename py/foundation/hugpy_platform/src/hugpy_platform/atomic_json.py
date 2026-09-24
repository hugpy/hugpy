"""Atomic JSON writes for the small shared state files."""

from __future__ import annotations

import json
import os
import secrets


def read_json_dict(path: str) -> dict | None:
    """Read a JSON object, returning None for absent or malformed files."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 — unreadable and malformed are both absent
        return None
    return data if isinstance(data, dict) else None


def save_json(path: str, data: dict) -> None:
    """Write JSON with a unique temporary file and an atomic replacement."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
