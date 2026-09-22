"""Resolve executables portably.

On Windows binaries carry an ``.exe`` suffix and aren't found by a bare name in
the same way; on every OS we want PATH lookup plus a chance to check explicit
directories (e.g. the fetched engine dir). ``resolve_bin`` centralises that so
call sites stop hardcoding ``"ffmpeg"`` / ``"nvidia-smi"`` / ``"llama-server"``.
"""
from __future__ import annotations

import os
import shutil
from typing import Iterable, Optional

from hugpy_platform.platform_facade import EXE_SUFFIX, IS_WINDOWS


def candidate_names(name: str) -> list[str]:
    """The filenames to look for, accounting for the Windows ``.exe`` suffix."""
    if IS_WINDOWS and not name.lower().endswith(".exe"):
        return [name + ".exe", name]
    return [name]


def resolve_bin(name: str, extra_dirs: Optional[Iterable[str]] = None) -> Optional[str]:
    """Absolute path to ``name`` or ``None``.

    Search order: each explicit dir in ``extra_dirs`` (checking for a ``.exe``
    variant on Windows), then ``PATH`` via :func:`shutil.which`.
    """
    names = candidate_names(name)
    for d in extra_dirs or ():
        if not d:
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.X_OK if not IS_WINDOWS else os.F_OK):
                return p
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return None


def with_exe(name: str) -> str:
    """``name`` with the platform executable suffix appended (idempotent)."""
    if EXE_SUFFIX and not name.lower().endswith(EXE_SUFFIX):
        return name + EXE_SUFFIX
    return name
