"""Small filesystem probes shared across packages."""

from __future__ import annotations

import os
import shutil


def free_bytes(path: str) -> int | None:
    """Free bytes on the target's filesystem, falling back to root."""
    try:
        probe = path if os.path.exists(path) else "/"
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def ensure_parent_best_effort(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass


def write_text(path: str, content: str) -> str:
    """Write UTF-8 text, creating its parent, and return the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def is_within(path: str, root: str) -> bool:
    """Whether a real path stays under a real root."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    try:
        return os.path.commonpath([resolved_path, resolved_root]) == resolved_root
    except ValueError:
        return False


def directory_bytes(path: str) -> int:
    """Count readable file bytes under a directory, tolerating vanished files."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
