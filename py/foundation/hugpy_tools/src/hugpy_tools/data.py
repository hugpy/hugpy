"""Structured-data read/write: JSON always, TOML via stdlib ``tomllib``, YAML
only when PyYAML happens to be installed. Writing is JSON-only on purpose
(stdlib has no TOML/YAML serializer); the write path is atomic.

Every format degrades honestly: an unavailable parser raises a clear error
naming exactly what is missing, never a silent wrong-format guess. Ported in
spirit from abstract_utilities.json_utils (safe_dump / safe_read) but slimmed
to the two calls an agent needs.
"""
from __future__ import annotations

import json
import os

from . import fs

# Map a lowercase extension to a logical format.
_EXT_FORMAT = {
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _format_for(path: str, explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    return _EXT_FORMAT.get(os.path.splitext(path)[1].lower(), "json")


def read_data(path: str, fmt: str | None = None):
    """Parse a JSON/TOML/YAML file to a Python object. Format is inferred from
    the extension unless ``fmt`` is given. Raises ``RuntimeError`` when a
    parser is unavailable, ``ValueError`` on an unknown format."""
    fmt = _format_for(path, fmt)
    if fmt == "json":
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8", errors="replace"))
    if fmt == "toml":
        try:
            import tomllib  # stdlib >= 3.11
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML reading needs Python 3.11+ (stdlib 'tomllib'); this "
                "interpreter is older. Install the 'tomli' backport and read "
                "it yourself, or upgrade Python.") from exc
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    if fmt == "yaml":
        try:
            import yaml  # optional third-party
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML reading needs PyYAML, which is not installed "
                "(hugpy_tools is stdlib-only). Install 'PyYAML' to enable "
                "YAML.") from exc
        with open(path, "rb") as fh:
            return yaml.safe_load(fh)
    raise ValueError("unknown data format %r (use json/toml/yaml)" % fmt)


def safe_json_dumps(obj, indent: int = 2, sort_keys: bool = False) -> str:
    """``json.dumps`` that never explodes on a non-serialisable value: anything
    the encoder cannot handle is stringified via ``default=str`` and non-ASCII
    is preserved (``ensure_ascii=False``)."""
    return json.dumps(obj, indent=indent, sort_keys=sort_keys,
                      ensure_ascii=False, default=str)


def write_json(path: str, obj, indent: int = 2, sort_keys: bool = False) -> dict:
    """Serialise ``obj`` to JSON and write it atomically. Returns
    ``{path, bytes, mode}`` (from :func:`fs.atomic_write`)."""
    return fs.atomic_write(path, safe_json_dumps(obj, indent, sort_keys) + "\n")
