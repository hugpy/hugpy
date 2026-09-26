"""hugpy-tools: a slim, stdlib-only capability suite for autonomous agents.

Extracted from the sprawl of ``abstract_utilities`` / ``abstract_webtools`` and
rebuilt with no third-party runtime dependency, so a machine with only Python
can use it. It imports NO other ``hugpy_*`` package (foundation layer).

Modules:

    paths     path confinement (jail) + path-part helpers + filename sanitising
    fs        safe file/dir ops: encoding-detecting read, line-range read,
              atomic write, exact-string edit, list/tree, rich file info
    hashkit   sha256 (text/file, streamed) + a cheap size+head quick hash
    text      approximate token counting, token/line chunking, unified diffs
    data      JSON/TOML/YAML read + atomic JSON write (safe dumping)
    timekit   UTC ISO / epoch conversions
    web       assess_webpage / prescreen_webpage — assessManager-parity webpage
              assessment over urllib + html.parser (opt-in JS render)
    search    reserved namespace (owned elsewhere; not implemented here)

Every ``fs`` function takes an already-confined absolute path — call
``paths.confine(root, path)`` first to enforce a jail. The web fetch never
follows a redirect off http(s) and disables ``file://`` / ``ftp://``.
"""
from __future__ import annotations

from hugpy_tools.data import read_data, safe_json_dumps, write_json
from hugpy_tools.fs import (
    atomic_write,
    detect_encoding,
    edit_replace,
    file_info,
    list_dir,
    read_lines,
    read_text,
    tree,
)
from hugpy_tools.hashkit import quick_hash, sha256_file, sha256_text
from hugpy_tools.paths import PathEscape, confine, file_parts, sanitize_filename
from hugpy_tools.text import (
    chunk_by_lines,
    chunk_by_tokens,
    count_tokens,
    unified_diff,
)
from hugpy_tools.timekit import epoch_to_iso, iso_to_epoch, now_epoch, now_iso
from hugpy_tools.web import assess_webpage, prescreen_webpage

__all__ = [
    # paths
    "PathEscape",
    "confine",
    "file_parts",
    "sanitize_filename",
    # fs
    "atomic_write",
    "detect_encoding",
    "edit_replace",
    "file_info",
    "list_dir",
    "read_lines",
    "read_text",
    "tree",
    # hashkit
    "quick_hash",
    "sha256_file",
    "sha256_text",
    # text
    "chunk_by_lines",
    "chunk_by_tokens",
    "count_tokens",
    "unified_diff",
    # data
    "read_data",
    "safe_json_dumps",
    "write_json",
    # timekit
    "epoch_to_iso",
    "iso_to_epoch",
    "now_epoch",
    "now_iso",
    # web
    "assess_webpage",
    "prescreen_webpage",
]
