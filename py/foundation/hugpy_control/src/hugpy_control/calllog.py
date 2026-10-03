"""CALL-LOG-20260910: append-only call log (one JSON line per event) + tail reader.

Events: phase "start" (job created; carries client/ua/route when a request
context exists), the STAGE STAMPS "processing" (a worker accepted the call /
prefill started) and "first_token" (first streamed token) — each written AT the
transition by the JobStore (operator 2026-09-29: a call written only on
completion is invisible while it is stuck) — and phase "end" (job finished;
carries the terminal status, worker, tokens, duration_ms, error). The reader
merges them by job id, newest first, into one row carrying queued_ts /
processing_ts / first_token_ts / processed_ts (+ processed_status).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_HOST = socket.gethostname()
_PROCESS_STARTED_AT = time.time()


def path() -> str:
    env = (os.environ.get("HUGPY_CALL_LOG") or "").strip()
    if env:
        return env
    db = (os.environ.get("HUGPY_COMMS_DB") or "").strip()
    base = os.path.dirname(db) if db else ""
    if not base:
        try:
            from hugpy_platform.constants import DEFAULT_ROOT
            base = os.path.join(str(DEFAULT_ROOT), "comms")
        except Exception:  # noqa: BLE001
            base = os.path.expanduser("~/.hugpy")
    return os.path.join(base, "calls.jsonl")


def _request_context() -> dict:
    try:
        from flask import has_request_context, request
        if not has_request_context():
            return {}
        forwarded_for = (request.headers.get("X-Forwarded-For") or "")[:512]
        xff = forwarded_for.split(",")[0].strip()
        return {
            "client": xff or request.remote_addr,
            "peer": request.remote_addr,
            "forwarded_for": forwarded_for or None,
            "ua": (request.headers.get("User-Agent") or "")[:160],
            # HTTP does not expose the remote OS process or account. Clients
            # that want this attribution can declare it with these headers;
            # authenticated HugPy identity remains in `principal` separately.
            "client_process": (request.headers.get("X-Hugpy-Client-Process") or "")[:160] or None,
            "client_pid": (request.headers.get("X-Hugpy-Client-Pid") or "")[:24] or None,
            "client_user": (request.headers.get("X-Hugpy-Client-User") or "")[:160] or None,
            "client_session": (request.headers.get("X-Hugpy-Client-Session") or "")[:200] or None,
            "client_turn": (request.headers.get("X-Hugpy-Client-Turn") or "")[:200] or None,
            "client_request": (request.headers.get("X-Hugpy-Client-Request") or "")[:200] or None,
            "client_task": (request.headers.get("X-Hugpy-Client-Task") or "")[:200] or None,
            "client_platform": (request.headers.get("X-Hugpy-Client-Platform") or "")[:100] or None,
            "route": request.path,
            "method": request.method,
            "host": request.host,
            "scheme": request.scheme,
        }
    except Exception:  # noqa: BLE001
        return {}


def prompt_hash(request_body, prompt=None) -> "tuple[str | None, int | None]":
    """A STABLE short digest of what the caller asked (sha256 of the ordered
    role/content pairs, 16 hex) plus its size in chars — so an identical-prompt
    loop (the same reducer slice re-submitted every turn, the same probe fired
    in a tight loop) is detectable from the call log WITHOUT storing prompts.
    Falls back to the formatted prompt text when no messages ride the request.
    (None, None) when neither exists."""
    try:
        msgs = request_body.get("messages") if isinstance(request_body, dict) else None
        if isinstance(msgs, list) and msgs:
            pairs = []
            for m in msgs:
                if isinstance(m, dict):
                    c = m.get("content")
                    if not isinstance(c, str):
                        c = json.dumps(c, sort_keys=True, default=str, ensure_ascii=True)
                    pairs.append([str(m.get("role") or ""), c])
                else:
                    pairs.append(["", str(m)])
            blob = json.dumps(pairs, ensure_ascii=True, separators=(",", ":"))
        elif isinstance(request_body, dict) and isinstance(request_body.get("prompt"), str):
            blob = request_body["prompt"]
        elif isinstance(prompt, str) and prompt:
            blob = prompt
        else:
            return None, None
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], len(blob)
    except Exception:  # noqa: BLE001
        return None, None


def _err_text(err) -> str | None:
    if err is None:
        return None
    for attr in ("message", "msg"):
        v = getattr(err, attr, None)
        if v:
            return str(v)[:400]
    if isinstance(err, dict):
        return str(err.get("message") or err)[:400]
    return str(err)[:400]


def _session_hook(phase: str, row: dict) -> None:
    """SESSION-LEASE-20260929: bind a job to the client session it declared
    (X-Hugpy-Client-Session/Turn/Request) at start; mark it ended at end."""
    try:
        from hugpy_control import sessions
        if phase == "start" and row.get("client_session"):
            sessions.note_job(row.get("id"), row)
        elif phase == "end":
            sessions.note_job_end(row.get("id"))
    except Exception:  # noqa: BLE001
        log.debug("session hook failed", exc_info=True)


# Called with the "end" row after it is written (2026-10-02: central records
# FAILED / cancelled calls into model_calls through this; the call log itself
# stays the complete record). A hook never breaks logging.
_END_HOOKS: list = []


def add_end_hook(fn) -> None:
    if fn not in _END_HOOKS:
        _END_HOOKS.append(fn)


def record(phase: str, job, **extra) -> None:
    """Append one event for `job`. Never raises — logging must not break serving."""
    try:
        row = {
            "ts": time.time(), "phase": phase, "node": _HOST,
            "id": getattr(job, "id", None), "kind": getattr(job, "kind", None),
            "model_key": getattr(job, "model_key", None), "model": getattr(job, "model_name", None),
            "principal": getattr(job, "principal", None), "transport": getattr(job, "transport", None),
            "channel": getattr(job, "channel", None), "worker": getattr(job, "worker", None),
            "status": getattr(job, "status", None),
            "tokens": (getattr(job, "total_tokens", None)
                       if getattr(job, "total_tokens", None) is not None
                       else getattr(job, "tokens", 0)),
            "input_tokens": getattr(job, "input_tokens", None),
            "output_tokens": getattr(job, "output_tokens", None),
            "total_tokens": getattr(job, "total_tokens", None),
            "started_ts": getattr(job, "started_ts", None),
        }
        # The request body rides ONLY on the start row: a 50 KB reducer prompt
        # must not be re-appended on every stage stamp.
        request_body = getattr(job, "request", None)
        if request_body is not None and phase == "start":
            row["request"] = request_body
        if phase == "start":
            row["prompt_hash"], row["prompt_chars"] = prompt_hash(
                request_body, getattr(job, "prompt", None))
        row["stage"] = getattr(job, "stage", None) or None
        if phase == "start":
            row.update(_request_context())
        elif phase in ("processing", "first_token"):
            # Stage stamp: what is known at the transition, nothing invented.
            row["slot"] = getattr(job, "slot", None)
        else:
            st = getattr(job, "started_ts", None)
            row["duration_ms"] = int((time.time() - st) * 1000) if st else None
            row["error"] = _err_text(getattr(job, "error", None))
        row.update(extra)
        _session_hook(phase, row)
        line = json.dumps(row, default=str)
        p = path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with _LOCK, open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if phase == "end":
            for hook in list(_END_HOOKS):
                try:
                    hook(row)
                except Exception:  # noqa: BLE001
                    log.debug("call log end hook failed", exc_info=True)
    except Exception:  # noqa: BLE001
        log.debug("call log write failed", exc_info=True)


def read(limit: int = 300, since: float | None = None, tail_bytes: int = 4 * 1024 * 1024) -> list[dict]:
    """Newest-first merged calls from the tail of the log."""
    p = path()
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = data.split("\n")
    if size > tail_bytes and lines:
        lines = lines[1:]                       # drop the partial first line
    calls: dict = {}
    order: list = []
    for line in lines:
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        jid = ev.get("id") or f"anon-{ev.get('ts')}"
        if jid not in calls:
            calls[jid] = {}
            order.append(jid)
        cur = calls[jid]
        phase = ev.get("phase")
        if phase == "start":
            cur.update({k: v for k, v in ev.items() if k != "phase"})
            cur.setdefault("started_ts", ev.get("ts"))
            cur.setdefault("queued_ts", cur.get("started_ts"))
        elif phase in ("processing", "first_token"):
            # A stage stamp only ADVANCES the row: the phase's own timestamp
            # plus the live facts known at that instant (status, worker, stage).
            cur.update({k: v for k, v in ev.items()
                        if k not in ("phase", "ts") and v is not None})
            cur.setdefault(f"{phase}_ts", ev.get("ts"))
        else:
            cur.update({k: v for k, v in ev.items() if k not in ("phase", "ts") and v is not None})
            cur["ended_ts"] = ev.get("ts")
            cur["processed_ts"] = ev.get("ts")
            cur["processed_status"] = ev.get("status")
    rows = [calls[j] for j in order]
    for row in rows:
        if (row.get("status") in ("pending", "processing", "streaming")
                and (row.get("started_ts") or 0) < _PROCESS_STARTED_AT
                and not row.get("ended_ts")):
            # Honest terminal for a call the restart orphaned: it still gets a
            # processed stamp (the process start) and a terminal status.
            row["status"] = "interrupted"
            row["processed_status"] = "interrupted"
            row.setdefault("processed_ts", _PROCESS_STARTED_AT)
            row["error"] = "central API restarted before this call completed"
    if since:
        rows = [r for r in rows if (r.get("started_ts") or r.get("ts") or 0) >= since]
    rows.sort(key=lambda r: r.get("started_ts") or r.get("ts") or 0, reverse=True)
    return rows[:limit]
