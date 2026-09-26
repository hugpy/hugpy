"""Safe file/directory operations (stdlib only).

Every function here takes an ALREADY-CONFINED absolute path — confinement is
the caller's job (see :func:`paths.confine`). That split keeps this module a
pure file-ops library with no policy of its own, which is exactly what a
standalone package wants.

Highlights ported from abstract_utilities:
  * encoding detection on read (BOM sniff -> utf-8 -> cp1252 fallback)
  * atomic write (temp file in the same dir + ``os.replace``)
  * exact string edit with an occurrence-count contract
  * directory listing + bounded recursive tree + rich file info
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

from . import hashkit

# BOM -> declared encoding
_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


def detect_encoding(data: bytes) -> str:
    """Best-effort encoding guess for ``data`` using only stdlib: honour a BOM,
    then prefer strict utf-8, then fall back to cp1252 (a superset of latin-1
    that decodes any byte). Never raises."""
    for bom, enc in _BOMS:
        if data.startswith(bom):
            return enc
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def read_text(path: str, max_bytes: int | None = None) -> dict:
    """Read a text file with encoding detection. Returns
    ``{path, encoding, bytes, truncated, text}``."""
    with open(path, "rb") as fh:
        if max_bytes is None:
            raw = fh.read()
            truncated = False
        else:
            raw = fh.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            raw = raw[:max_bytes]
    enc = detect_encoding(raw)
    return {
        "path": path,
        "encoding": enc,
        "bytes": len(raw),
        "truncated": truncated,
        "text": raw.decode(enc, errors="replace"),
    }


def read_lines(path: str, start: int = 1, end: int | None = None,
               number: bool = False) -> dict:
    """Read a 1-based inclusive line range. ``end=None`` reads to EOF.
    Returns ``{path, start, end, total_lines, returned, text}``."""
    start = max(1, int(start))
    with open(path, "rb") as fh:
        raw = fh.read()
    enc = detect_encoding(raw)
    lines = raw.decode(enc, errors="replace").splitlines()
    total = len(lines)
    last = total if end is None else min(int(end), total)
    chosen = lines[start - 1:last] if start <= total else []
    if number:
        width = len(str(last))
        body = "\n".join("%*d\t%s" % (width, start + i, ln)
                         for i, ln in enumerate(chosen))
    else:
        body = "\n".join(chosen)
    return {
        "path": path,
        "start": start,
        "end": last,
        "total_lines": total,
        "returned": len(chosen),
        "text": body,
    }


def atomic_write(path: str, content: str, append: bool = False,
                 encoding: str = "utf-8") -> dict:
    """Write ``content`` durably. Overwrite is atomic (temp file in the same
    directory + ``os.replace``, so a reader never sees a half-written file).
    Append opens the real file directly (append has no atomic swap). Returns
    ``{path, bytes, mode}``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = content.encode(encoding, errors="replace")
    if append:
        with open(path, "ab") as fh:
            fh.write(data)
    else:
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return {"path": path, "bytes": len(data),
            "mode": "append" if append else "overwrite"}


def edit_replace(path: str, old: str, new: str, count: int | None = None) -> dict:
    """Exact string replacement in a text file. By default every occurrence is
    replaced; pass ``count`` to bound it. Raises ``ValueError`` if ``old`` is
    empty or not found. Returns ``{path, replaced, bytes}``; the write is
    atomic."""
    if old == "":
        raise ValueError("old must be a non-empty string")
    info = read_text(path)
    text = info["text"]
    occurrences = text.count(old)
    if occurrences == 0:
        raise ValueError("old string not found in %s" % path)
    replaced = occurrences if count is None else min(occurrences, int(count))
    text = text.replace(old, new, -1 if count is None else int(count))
    out = atomic_write(path, text, encoding="utf-8"
                       if info["encoding"] in ("utf-8", "utf-8-sig", "cp1252")
                       else info["encoding"])
    return {"path": path, "replaced": replaced, "bytes": out["bytes"]}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def file_info(path: str, hash_files: bool = False) -> dict:
    """Rich stat for one path. For a regular file with ``hash_files`` the
    sha256 is included."""
    st = os.lstat(path)
    is_link = os.path.islink(path)
    real = os.path.realpath(path)
    is_dir = os.path.isdir(real)
    info = {
        "path": path,
        "name": os.path.basename(path.rstrip(os.sep)) or path,
        "type": "dir" if is_dir else ("symlink" if is_link and not os.path.exists(real) else "file"),
        "is_dir": is_dir,
        "is_symlink": is_link,
        "size": st.st_size,
        "mtime": _iso(st.st_mtime),
    }
    if hash_files and not is_dir and os.path.isfile(real):
        info["sha256"] = hashkit.sha256_file(real)
    return info


def list_dir(path: str, glob: str | None = None, files_only: bool = False,
             dirs_only: bool = False, limit: int = 500) -> dict:
    """List the immediate entries of a directory (sorted: dirs first, then
    files, each alphabetical). Optional fnmatch ``glob`` on the entry name.
    Returns ``{path, count, truncated, entries:[{name,type,size,mtime}]}``."""
    import fnmatch
    names = sorted(os.listdir(path))
    entries = []
    for name in names:
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        if files_only and is_dir:
            continue
        if dirs_only and not is_dir:
            continue
        if glob and not fnmatch.fnmatch(name, glob):
            continue
        try:
            st = os.lstat(full)
            size, mtime = st.st_size, _iso(st.st_mtime)
        except OSError:
            size, mtime = None, None
        entries.append({"name": name,
                        "type": "dir" if is_dir else "file",
                        "size": size, "mtime": mtime})
    entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))
    truncated = len(entries) > limit
    return {"path": path, "count": len(entries), "truncated": truncated,
            "entries": entries[:limit]}


def tree(path: str, max_depth: int = 3, max_entries: int = 500,
         show_files: bool = True) -> dict:
    """Bounded recursive directory tree rendered as indented text. Skips
    symlinked directories (never recurses out through a link). Returns
    ``{path, entries, truncated, text}`` where ``entries`` is the count
    emitted."""
    lines: list[str] = []
    state = {"n": 0, "truncated": False}

    def walk(cur: str, depth: int, prefix: str) -> None:
        if depth > max_depth or state["truncated"]:
            return
        try:
            names = sorted(os.listdir(cur))
        except OSError:
            return
        dirs = [n for n in names if os.path.isdir(os.path.join(cur, n))
                and not os.path.islink(os.path.join(cur, n))]
        files = [n for n in names if not os.path.isdir(os.path.join(cur, n))]
        ordered = [(n, True) for n in dirs] + \
                  ([(n, False) for n in files] if show_files else [])
        for name, is_dir in ordered:
            if state["n"] >= max_entries:
                state["truncated"] = True
                return
            state["n"] += 1
            lines.append("%s%s%s" % (prefix, name, "/" if is_dir else ""))
            if is_dir:
                walk(os.path.join(cur, name), depth + 1, prefix + "  ")

    walk(path, 1, "")
    return {"path": path, "entries": state["n"],
            "truncated": state["truncated"], "text": "\n".join(lines)}
