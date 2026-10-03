"""FAILED CALLS INTO model_calls (operator 2026-10-02: "is the call log being
stored successfully and attributed to each model's table?" — 1,312 of 4,626
calls in a week failed and none reached the DB). The relay records a call it
completed; this records every call that ENDED otherwise (failed / cancelled /
any non-done status) from the call log's end row, so a model's record shows
its failures too. Deduped by request_id; off the finish path (a thread)."""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_DONE = ("done", "completed", "complete", "ok", "succeeded", None, "")


def _state(row: dict) -> dict:
    return {"outcome": {"ok": False, "status": row.get("status"),
                        "error": (str(row.get("error") or "")[:1000] or None)},
            "stage": row.get("stage"), "kind": row.get("kind"),
            "caller": row.get("principal") or "relay", "transport": row.get("transport"),
            "source": "call-log-end", "node": row.get("node")}


def write(row: dict) -> bool:
    """Record one non-done end row (synchronous; the hook runs it in a thread)."""
    model = row.get("model_key") or row.get("model")
    if not model or not row.get("id"):
        return False
    from hugpy_engine.model_index import record_call_if_absent
    dur = row.get("duration_ms")
    return record_call_if_absent(
        str(row["id"]), str(model), str(row.get("worker") or "unassigned"),
        elapsed_s=(round(dur / 1000.0, 3) if isinstance(dur, (int, float)) else None),
        prompt_tokens=row.get("input_tokens"), completion_tokens=row.get("output_tokens"),
        task=row.get("kind"), state=_state(row))


def on_call_end(row: dict) -> None:
    if (row or {}).get("status") in _DONE:
        return
    threading.Thread(target=_safe_write, args=(dict(row),), daemon=True,
                     name="failed-call-record").start()


def _safe_write(row: dict) -> None:
    try:
        write(row)
    except Exception:  # noqa: BLE001 — recording never breaks anything
        log.debug("failed-call record failed", exc_info=True)


def start() -> None:
    from hugpy_control.calllog import add_end_hook
    add_end_hook(on_call_end)
