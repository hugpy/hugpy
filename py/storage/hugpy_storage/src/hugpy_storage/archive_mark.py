"""The ARCHIVE MARK — an operator's "retire this model" flag on its own record.

Two states only: LIVE or ARCHIVE (operator ruling 2026-09-23). The mark is the
operator's recorded intent to move a live model into the archive; the sweep
(``hugpy_ops.model_archive`` / ``hugpy-model-archive --apply``) is what turns
the intent into the archive state (the directory moved under
``ARCHIVE/MODELS_ARCHIVED-<date>/``, a MANIFEST row with the restore command,
the catalog row removed). Marking deletes, moves and evicts NOTHING.

The mark lives on the model's hugpy.json, beside ``admission``:

    hugpy.json["archive"] = {
        "marked": true,
        "at":     iso-8601 UTC,     # when it was marked
        "by":     str,              # who marked it (operator username / key name / 'console')
        "reason": str | None,       # the operator's words, verbatim; None when none given
    }

Unmarking removes the block (the marker is the record of CURRENT intent; the
server audit log keeps the history of mark/unmark events).

While marked, central refuses the model everywhere a placement or routing
choice could land on it: new designations (assign / load / bulk alloc / group
allocate / template activate / placement preference), warm + provisioning
sweeps, and the resolver (``hugpy_fleet.central.archive_gate``). Every refusal
carries :func:`archive_text` — the recorded facts, never a canned line.

stdlib + the marker module; no network.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

ARCHIVE_KEY = "archive"
# The phrase every refusal carries (the engine's cold-hold classifier keys on
# it as PERMANENT: fail fast, never held/retried, never cached as a load verdict).
ARCHIVE_MARKER = "marked for archive"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def archive_record(*, by: str, reason: Optional[str] = None,
                   at: Optional[str] = None) -> dict:
    """The archive block, schema-complete (every key present)."""
    r = (reason or "").strip() if isinstance(reason, str) else None
    return {"marked": True, "at": at or utc_now_iso(),
            "by": str(by or "").strip() or "console", "reason": r or None}


def is_marked(block: Optional[dict]) -> bool:
    return isinstance(block, dict) and bool(block.get("marked"))


def archive_view(block: Optional[dict]) -> dict:
    """The ``archived`` projection every surface exposes:
    ``{marked, at, by, reason}`` (``marked: false`` + nulls when unmarked)."""
    if not is_marked(block):
        return {"marked": False, "at": None, "by": None, "reason": None}
    return {"marked": True, "at": block.get("at"), "by": block.get("by"),
            "reason": block.get("reason")}


def archive_text(block: Optional[dict]) -> Optional[str]:
    """``marked for archive by <by> at <at>: <reason>`` from the recorded block,
    or None when not marked. A mark recorded without a reason says so."""
    if not is_marked(block):
        return None
    reason = block.get("reason")
    tail = f": {reason}" if reason else " (no reason was given when it was marked)"
    return f"{ARCHIVE_MARKER} by {block.get('by')} at {block.get('at')}{tail}"


def read_archive_mark(directory: Optional[str]) -> Optional[dict]:
    """The archive block on ``directory``'s hugpy.json, or None."""
    if not directory:
        return None
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    block = (read_hugpy_marker(directory) or {}).get(ARCHIVE_KEY)
    return block if isinstance(block, dict) else None


def _forget(model_key: Optional[str], why: str) -> None:
    if not model_key:
        return
    try:
        from hugpy_storage.model_physical import forget_physical
        forget_physical(model_key, why)
    except Exception:  # noqa: BLE001 — the record landed; the cache re-derives
        logger.debug("archive mark: physical-record invalidation failed for %s",
                     model_key, exc_info=True)


def write_archive_mark(directory: Optional[str], record: dict, *,
                       model_key: Optional[str] = None) -> Optional[str]:
    """Set the archive block on an EXISTING hugpy.json (atomic temp+replace).

    Returns the marker path, or None when the dir carries no marker (a dir
    without a marker is not an installed model — nothing to mark). Drops the
    model's persisted physical record so every catalog surface re-reads it."""
    if not directory:
        return None
    from hugpy_storage.hugpy_marker import _save_marker, read_hugpy_marker
    marker = read_hugpy_marker(directory)
    if not isinstance(marker, dict):
        return None
    marker[ARCHIVE_KEY] = dict(record)
    path = _save_marker(directory, marker)
    _forget(model_key, f"archive marked by {record.get('by')}")
    return path


def clear_archive_mark(directory: Optional[str], *,
                       model_key: Optional[str] = None) -> Optional[dict]:
    """Remove the archive block. Returns the block that was removed, or None
    when there was none (or no marker)."""
    if not directory:
        return None
    from hugpy_storage.hugpy_marker import _save_marker, read_hugpy_marker
    marker = read_hugpy_marker(directory)
    if not isinstance(marker, dict) or ARCHIVE_KEY not in marker:
        return None
    was = marker.pop(ARCHIVE_KEY)
    _save_marker(directory, marker)
    _forget(model_key, "archive mark cleared")
    return was if isinstance(was, dict) else None


__all__ = ["ARCHIVE_KEY", "ARCHIVE_MARKER", "archive_record", "archive_text",
           "archive_view", "clear_archive_mark", "is_marked", "read_archive_mark",
           "write_archive_mark"]
