"""CALL-LOG-20260910: append-only call log (one JSON line per event) + tail reader.

Events: phase "start" (job created; carries client/ua/route when a request
context exists) and phase "end" (job finished; carries status, worker, tokens,
duration_ms, error). The reader merges the two by job id, newest first.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_HOST = socket.gethostname()


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
        xff = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return {
            "client": xff or request.remote_addr,
            "ua": (request.headers.get("User-Agent") or "")[:160],
            "route": request.path,
            "method": request.method,
            "host": request.host,
        }
    except Exception:  # noqa: BLE001
        return {}


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


def record(phase: str, job, **extra) -> None:
    """Append one event for `job`. Never raises — logging must not break serving."""
    try:
        row = {
            "ts": time.time(), "phase": phase, "node": _HOST,
            "id": getattr(job, "id", None), "kind": getattr(job, "kind", None),
            "model_key": getattr(job, "model_key", None), "model": getattr(job, "model_name", None),
            "principal": getattr(job, "principal", None), "transport": getattr(job, "transport", None),
            "channel": getattr(job, "channel", None), "worker": getattr(job, "worker", None),
            "status": getattr(job, "status", None), "tokens": getattr(job, "tokens", 0),
            "started_ts": getattr(job, "started_ts", None),
        }
        if phase == "start":
            row.update(_request_context())
        else:
            st = getattr(job, "started_ts", None)
            row["duration_ms"] = int((time.time() - st) * 1000) if st else None
            row["error"] = _err_text(getattr(job, "error", None))
        row.update(extra)
        line = json.dumps(row, default=str)
        p = path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with _LOCK, open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
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
        if ev.get("phase") == "start":
            cur.update({k: v for k, v in ev.items() if k != "phase"})
            cur.setdefault("started_ts", ev.get("ts"))
        else:
            cur.update({k: v for k, v in ev.items() if k not in ("phase", "ts") and v is not None})
            cur["ended_ts"] = ev.get("ts")
    rows = [calls[j] for j in order]
    if since:
        rows = [r for r in rows if (r.get("started_ts") or r.get("ts") or 0) >= since]
    rows.sort(key=lambda r: r.get("started_ts") or r.get("ts") or 0, reverse=True)
    return rows[:limit]
