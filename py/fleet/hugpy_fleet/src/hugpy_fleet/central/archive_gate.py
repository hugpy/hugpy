"""Central's ARCHIVE gate — refuse every placement/routing choice that would land
on a model the operator marked for archive (hugpy_storage.archive_mark).

The mark lives on the model's own ``hugpy.json["archive"]``; this module reads
it through the PERSISTED marker aspect every catalog surface already reads
(``lookup_physical(.., ASPECT_MARKER)``, derived live on a miss) — the same
path ``admission_gate`` uses — so a gate check is a dict lookup, not a file
read per request.

Distinct from the operator BLOCKLIST (``blocklist.py``, settings-store backed,
reversible routing override) and from ADMISSION (a machine verdict): the mark
is the operator's recorded intent to move the model out of the live set. It is
enforced at least as strictly as a block — resolver, designation, warm,
provisioning — and, unlike the admission hold, ``alloc.force`` never bypasses
it. The refusal text is :func:`hugpy_storage.archive_mark.archive_text` (who
marked it, when, why) and carries ``ARCHIVE_MARKER`` so the engine's cold-hold
classifier treats it as PERMANENT.

Fail-open like the blocklist: a gate that cannot read the marker answers
"not marked".
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from hugpy_storage.archive_mark import (ARCHIVE_KEY, ARCHIVE_MARKER, archive_text,
                                        archive_view, is_marked)

logger = logging.getLogger(__name__)

_KEYS_TTL_S = 5.0
_keys_lock = threading.Lock()
_keys_cache: dict = {"at": 0.0, "keys": frozenset()}


def _row_and_key(model_key: str):
    from hugpy_fleet.central import workers as _w
    key = _w._canonical_registry_key(model_key)
    return key, _w._registry_row(key)


def _marker_of(key: str, row: dict) -> dict:
    from hugpy_storage.model_physical import ASPECT_MARKER, lookup_physical
    fields, state = lookup_physical(key, row, ASPECT_MARKER)
    if state != "fresh":
        from hugpy_storage.console.model_physical import marker_fields
        fields = marker_fields(row, key)
    marker = (fields or {}).get("hugpy_marker") or {}
    return marker if isinstance(marker, dict) else {}


def archive_block(model_key: Optional[str]) -> Optional[dict]:
    """The model's archive block (``{marked, at, by, reason}``) when marked, else None."""
    if not model_key:
        return None
    try:
        key, row = _row_and_key(str(model_key))
        if row is None:
            return None
        block = _marker_of(key, row).get(ARCHIVE_KEY)
        return dict(block) if is_marked(block) else None
    except Exception:  # noqa: BLE001 — fail open
        logger.debug("archive gate: unreadable for %s", model_key, exc_info=True)
        return None


def archive_reason(model_key: Optional[str]) -> Optional[str]:
    """``'<key>' is marked for archive by <by> at <at>: <reason>`` or None."""
    block = archive_block(model_key)
    if not block:
        return None
    return f"'{model_key}' is {archive_text(block)}"


def is_archived(model_key: Optional[str]) -> bool:
    return archive_block(model_key) is not None


def refusal(model_key: str) -> Optional[dict]:
    """The structured refusal body for a route (HTTP 409), or None when not
    marked: ``{"error": <archive_reason>, "archive": {marked, at, by, reason}}``."""
    block = archive_block(model_key)
    if not block:
        return None
    return {"error": f"'{model_key}' is {archive_text(block)}",
            "archive": archive_view(block)}


def archived_keys(*, fresh: bool = False) -> frozenset:
    """Every catalog key currently marked for archive (5 s process cache —
    warm sweeps and heartbeat replies call this; a mark/unmark in this process
    drops the cache at once via :func:`invalidate`)."""
    now = time.monotonic()
    with _keys_lock:
        if not fresh and now - _keys_cache["at"] < _KEYS_TTL_S:
            return _keys_cache["keys"]
    out = set()
    try:
        from hugpy_engine.config.models.models_config import get_models_dict
        for key, row in (get_models_dict(dict_return=True) or {}).items():
            mk = (row or {}).get("model_key") or key
            try:
                if is_marked(_marker_of(mk, row).get(ARCHIVE_KEY)):
                    out.add(mk)
                    out.add(key)
            except Exception:  # noqa: BLE001 — one unreadable row is "not marked"
                continue
    except Exception:  # noqa: BLE001 — fail open
        logger.debug("archive gate: catalog enumerate failed", exc_info=True)
    keys = frozenset(out)
    with _keys_lock:
        _keys_cache.update(at=time.monotonic(), keys=keys)
    return keys


def invalidate() -> None:
    with _keys_lock:
        _keys_cache.update(at=0.0, keys=frozenset())


__all__ = ["ARCHIVE_MARKER", "archive_block", "archive_reason", "archived_keys",
           "invalidate", "is_archived", "refusal"]
