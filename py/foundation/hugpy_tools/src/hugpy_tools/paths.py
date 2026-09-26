"""Path confinement and path-part helpers (stdlib only).

`confine` is the jail primitive ported from the agent's tools/fs.py: a path
(absolute or root-relative) is resolved through ``os.path.realpath`` and must
land under the realpath of ``root``. This rejects ``..`` escapes AND symlink
escapes in one check. For writes the *parent* directory is confined (the
target file may not exist yet, but its directory does), then the basename is
re-attached — so a dangling path with a symlinked ancestor cannot slip through.

Fail-closed: any ambiguity about where a path lands is a ``PathEscape``, never
a guess. Everything here is pure stdlib so the module ports cleanly into a
standalone package.
"""
from __future__ import annotations

import os
import re


class PathEscape(ValueError):
    """A path resolved outside the confinement root."""


def confine(root: str, path: str, for_write: bool = False) -> str:
    """Resolve ``path`` to a real absolute path confined under ``root``.

    Raises :class:`PathEscape` on any escape (``..``, absolute-outside, or a
    symlink pointing out of the jail).
    """
    real_root = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(real_root, path)
    if for_write:
        parent = os.path.realpath(os.path.dirname(candidate) or real_root)
        resolved = os.path.join(parent, os.path.basename(candidate))
    else:
        resolved = os.path.realpath(candidate)
    if resolved != real_root and not resolved.startswith(real_root + os.sep):
        raise PathEscape("path %r escapes the root (%s)" % (path, real_root))
    return resolved


def file_parts(path: str) -> dict:
    """Decompose ``path`` into its named components (dir/base/name/ext plus the
    two enclosing directory levels), never raising on a bare or odd path."""
    path = str(path)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, ext = os.path.splitext(basename)
    parent_dirname = os.path.dirname(dirname)
    super_dirname = os.path.dirname(parent_dirname)
    return {
        "file_path": path,
        "dirname": dirname,
        "basename": basename,
        "filename": filename,
        "ext": ext,
        "dirbase": os.path.basename(dirname),
        "parent_dirname": parent_dirname,
        "parent_dirbase": os.path.basename(parent_dirname),
        "super_dirname": super_dirname,
        "super_dirbase": os.path.basename(super_dirname),
    }


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str, replacement: str = "_", max_len: int = 255) -> str:
    """Reduce ``name`` to a safe single path component: strip directory
    separators, collapse unsafe characters, and bound the length. Never
    returns an empty string (falls back to ``"file"``)."""
    name = os.path.basename(str(name)).strip()
    name = _UNSAFE_NAME.sub(replacement, name).strip("._-" + replacement)
    if not name:
        name = "file"
    if len(name) > max_len:
        stem, ext = os.path.splitext(name)
        keep = max_len - len(ext)
        name = (stem[:keep] if keep > 0 else stem[:max_len]) + (ext if keep > 0 else "")
    return name
