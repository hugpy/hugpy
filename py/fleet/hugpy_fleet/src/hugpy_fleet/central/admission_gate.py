"""Central's ADMISSION gate — refuse to route to a model the post-download
admission held (hugpy_storage.admission / hugpy_ops.admission).

The verdict lives on the model's own ``hugpy.json["admission"]``; this module
reads it through the PERSISTED marker aspect every catalog surface already
reads (``lookup_physical(.., ASPECT_MARKER)``, derived live on a miss), so a
routing decision costs a dict lookup, not a file read per request.

``pending`` routes (the benchmark that decides admission has to reach the
model); only ``held`` refuses. The refusal text carries :data:`HELD_MARKER`,
which the engine's cold-hold classifier treats as PERMANENT (fail fast, never
retried). An explicit ``alloc.force=true`` on a request bypasses the gate for
diagnosis — that decision is the engine resolver's (resolvers/remote.py), which
is the only place that sees the request.

Fail-open like the blocklist: a gate that cannot read the marker answers
"not held" — a routing gate that raises is worse than a momentarily-unheld
model.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

HELD_MARKER = "held from the serving pool by admission"


def _row_and_key(model_key: str):
    from hugpy_fleet.central import workers as _w
    key = _w._canonical_registry_key(model_key)
    return key, _w._registry_row(key)


def admission_block(model_key: Optional[str]) -> Optional[dict]:
    """The model's admission block (``{"status", "reason", ...}``) or None."""
    if not model_key:
        return None
    try:
        key, row = _row_and_key(str(model_key))
        if row is None:
            return None
        from hugpy_storage.model_physical import ASPECT_MARKER, lookup_physical
        fields, state = lookup_physical(key, row, ASPECT_MARKER)
        if state != "fresh":
            from hugpy_storage.console.model_physical import marker_fields
            fields = marker_fields(row, key)
        marker = (fields or {}).get("hugpy_marker") or {}
        block = marker.get("admission") if isinstance(marker, dict) else None
        return block if isinstance(block, dict) else None
    except Exception:  # noqa: BLE001 — fail open
        logger.debug("admission gate: unreadable for %s", model_key, exc_info=True)
        return None


def refusal_text(model_key: str, block: dict) -> str:
    """The factual refusal: which gate, the recorded verdict, when, which job."""
    return (f"'{model_key}' is {HELD_MARKER}: {block.get('reason') or 'no reason recorded'} "
            f"[integrity={block.get('integrity')}, grade={block.get('grade')}, "
            f"at={block.get('at')}, job={block.get('job')}; "
            f"POST /llm/admission/{model_key}/rerun re-admits, alloc.force=true bypasses]")


def admission_reason(model_key: Optional[str]) -> Optional[str]:
    """The refusal text when ``model_key`` is held, else None."""
    block = admission_block(model_key)
    if not block or block.get("status") != "held":
        return None
    return refusal_text(str(model_key), block)


def is_held(model_key: Optional[str]) -> bool:
    return admission_reason(model_key) is not None


__all__ = ["HELD_MARKER", "admission_block", "admission_reason", "is_held",
           "refusal_text"]
