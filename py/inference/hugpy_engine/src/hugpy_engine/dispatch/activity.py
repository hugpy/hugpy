"""Process-wide registry of in-flight chat requests, for the console's live
queue view.

Now a thin shim over the shared comms JobStore (F5): begin/on_token/end
create and drive real Jobs in the one store every transport uses, so the
queue view, the /jobs surface, and the cancel plane all see the same records.
The public API and the ``GET /api/llm/queue`` snapshot shape are unchanged:

States (view-level): ``waiting`` (accepted, no token yet — canonical
pending/processing) → ``active`` (producing tokens — canonical streaming).
Entries leave the live view when the stream ends (the stream's ``finally``
still calls ``end``, which marks the job terminal instead of popping it).

Download jobs live in the same store but are excluded here — this is the
chat/inference queue, and downloads have their own /jobs surface.
"""
from __future__ import annotations

import json

from hugpy_control.jobs import job_store, normalize_status


def format_prompt(messages=None, prompt=None) -> str:
    """Return the exact prompt input in a readable form for the queue UI."""
    if prompt is not None:
        return str(prompt)
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    return json.dumps(messages, ensure_ascii=False, indent=2, default=str)

# view state <- canonical status
_VIEW_STATE = {"pending": "waiting", "processing": "waiting",
               "streaming": "active"}
_QUEUE_KINDS_EXCLUDED = {"download"}


def begin(request_id, model_key=None, model_name=None, kind="chat", prompt=None,
          request=None) -> None:
    if not request_id:
        return
    # Preserve an existing live entry if the same id re-registers.
    existing = job_store.get(request_id)
    if existing is not None and not existing.terminal:
        job_store.update(request_id, model_key=model_key or existing.model_key,
                         model_name=model_name or existing.model_name,
                         prompt=prompt or existing.prompt,
                         request=request or existing.request)
        return
    job_store.create(model_key or "", id=request_id, kind=kind,
                     model_name=model_name, transport=kind,
                     prompt=prompt or "", request=request)


def mark_active(request_id) -> None:
    job_store.on_output(request_id, n=0)


def on_token(request_id) -> None:
    """First token flips the entry active; every token bumps the counter."""
    job_store.on_output(request_id)


def end(request_id) -> None:
    if not request_id:
        return
    # finish() resolves the terminal state: cancelled if a cancel was
    # requested, else done. No-op if the stream already marked it.
    job_store.finish(request_id)


def fail(request_id, error=None) -> None:
    """Terminal-with-error, for callers that know the stream broke."""
    if request_id:
        job_store.finish(request_id, error=error)


def snapshot() -> list:
    """Public, JSON-safe view. Waiting first, then longest-running —
    identical shape to the pre-comms implementation."""
    out = []
    for d in job_store.snapshot():
        if d["kind"] in _QUEUE_KINDS_EXCLUDED:
            continue
        state = _VIEW_STATE.get(d["status"], d["status"])
        out.append({
            "request_id": d["id"],
            "model_key": d["model_key"] or None,
            "model": d["model"],
            "kind": d["kind"],
            "prompt": d.get("prompt") or "",
            "state": state,
            "elapsed": d["elapsed"],
            "wait": d["wait"],
            "tokens": d["tokens"],
        })
    out.sort(key=lambda x: (x["state"] != "waiting", -x["elapsed"]))
    return out


def counts() -> dict:
    live = [d for d in job_store.snapshot()
            if d["kind"] not in _QUEUE_KINDS_EXCLUDED]
    waiting = sum(1 for d in live
                  if _VIEW_STATE.get(d["status"]) == "waiting")
    active = sum(1 for d in live
                 if _VIEW_STATE.get(d["status"]) == "active")
    return {"waiting": waiting, "active": active, "total": waiting + active}
