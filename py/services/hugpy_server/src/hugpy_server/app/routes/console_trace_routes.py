"""Permanent browser-to-Flask execution tracing for the operator console.

The trace is deliberately opt-in (``X-Hugpy-Trace: 1``), bounded, and stored
in its OWN SQLite file (``console_trace.sqlite3`` beside the comms database, or
``HUGPY_CONSOLE_TRACE_DB``) so gunicorn workers and the console agree on one
stream.  Diagnostics never contend with the fleet's database: the trace file is
WAL with a short busy timeout, and every write is best-effort (a locked or
failing write drops the row, logs at debug, and never raises into or blocks the
request).  2026-10-01: writing to the main comms DB locked out agent heartbeats.  A traced request records the HTTP envelope plus nested Python
``call``/``return``/``exception`` events from ``sys.setprofile``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from flask import Response, jsonify, request, stream_with_context
from abstract_flask import get_bp
from hugpy_control.shared import default_db_path

trace_bp, logger = get_bp("console_trace_bp", __name__)
_tls = threading.local()
_MAX_SPANS = 12000
_MAX_EVENTS = 1000
_MAX_TEXT = 4096


_TRACE_DB_NAME = "console_trace.sqlite3"
_TIMEOUT_S = 0.25
_schema_ready = set()
_schema_lock = threading.Lock()
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS console_trace_requests (
      id TEXT PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL,
      method TEXT NOT NULL, path TEXT NOT NULL, status INTEGER,
      duration_ms REAL, browser TEXT, error TEXT
    );
    CREATE TABLE IF NOT EXISTS console_trace_spans (
      id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL,
      seq INTEGER NOT NULL, event TEXT NOT NULL, depth INTEGER NOT NULL,
      function TEXT NOT NULL, file TEXT, line INTEGER, at REAL NOT NULL,
      duration_ms REAL, error TEXT
    );
    CREATE TABLE IF NOT EXISTS console_trace_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, at REAL NOT NULL,
      kind TEXT NOT NULL, payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_console_trace_spans_trace
      ON console_trace_spans(trace_id, seq);
    CREATE INDEX IF NOT EXISTS ix_console_trace_events_id
      ON console_trace_events(id);
    """


def trace_db_path():
    """The trace store: never the fleet/comms database itself."""
    env = (os.environ.get("HUGPY_CONSOLE_TRACE_DB") or "").strip()
    if env:
        return env
    return os.path.join(os.path.dirname(default_db_path()) or ".", _TRACE_DB_NAME)


def _db(timeout=_TIMEOUT_S):
    path = trace_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    con = sqlite3.connect(path, timeout=timeout)
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    if path not in _schema_ready:
        with _schema_lock:
            if path not in _schema_ready:
                con.execute("PRAGMA journal_mode=WAL")
                con.executescript(_SCHEMA)
                _schema_ready.add(path)
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _write(fn):
    """Best-effort trace write: True when stored, False when dropped."""
    con = None
    try:
        con = _db()
        fn(con)
        con.commit()
        return True
    except Exception as exc:  # noqa: BLE001 — diagnostics never break a request
        logger.debug("console trace write dropped: %s", exc)
        return False
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass


def _clean(value, limit=_MAX_TEXT):
    text = str(value or "")
    return text[:limit]


def _profile(frame, event, arg):
    state = getattr(_tls, "trace", None)
    if not state or event not in ("call", "return", "exception"):
        return _profile if state else None
    if len(state["spans"]) >= _MAX_SPANS:
        return _profile
    now = time.time()
    key = (id(frame), frame.f_code.co_firstlineno, frame.f_code.co_name)
    if event == "call":
        state["stack"].append((key, now, len(state["stack"])))
        state["spans"].append({"event": "call", "depth": len(state["stack"])-1,
                                "function": frame.f_code.co_name,
                                "file": frame.f_code.co_filename,
                                "line": frame.f_lineno, "at": now})
    elif event == "return":
        started = None
        if state["stack"]:
            _, started, depth = state["stack"].pop()
        state["spans"].append({"event": "return", "depth": max(0, len(state["stack"])),
                                "function": frame.f_code.co_name,
                                "file": frame.f_code.co_filename,
                                "line": frame.f_lineno, "at": now,
                                "duration_ms": (now-started)*1000 if started else None})
    else:
        exc = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
        state["spans"].append({"event": "exception", "depth": len(state["stack"]),
                                "function": frame.f_code.co_name,
                                "file": frame.f_code.co_filename,
                                "line": frame.f_lineno, "at": now,
                                "error": _clean(exc)})
    return _profile


def install_console_trace(app):
    """Install request hooks once on the Flask app."""
    if app.extensions.get("console_trace"):
        return

    @app.before_request
    def _trace_before():
        if request.path.startswith("/console/trace") or request.path.startswith("/api/console/trace"):
            return
        if request.headers.get("X-Hugpy-Trace") != "1":
            return
        trace_id = _clean(request.headers.get("X-Hugpy-Trace-Id"), 96) or uuid.uuid4().hex
        state = {"id": trace_id, "started": time.time(), "stack": [], "spans": []}
        _tls.trace = state
        sys.setprofile(_profile)
        row = (trace_id, state["started"], request.method, request.path,
               _clean(request.headers.get("User-Agent"), 512))
        _write(lambda con: con.execute(
            "INSERT OR REPLACE INTO console_trace_requests "
            "(id,started_at,method,path,browser) VALUES (?,?,?,?,?)", row))

    @app.after_request
    def _trace_after(response):
        state = getattr(_tls, "trace", None)
        if not state:
            return response
        finished = time.time()
        sys.setprofile(None)
        _tls.trace = None
        meta = {"method": request.method, "path": request.path,
                "status": response.status_code,
                "duration_ms": (finished-state["started"])*1000,
                "spans": len(state["spans"])}
        spans = [(state["id"], n, s["event"], s["depth"], s["function"],
                  s["file"], s["line"], s["at"], s.get("duration_ms"), s.get("error"))
                 for n, s in enumerate(state["spans"][:_MAX_SPANS])]

        def _finish(con):
            con.execute("UPDATE console_trace_requests SET finished_at=?,status=?,duration_ms=? WHERE id=?",
                        (finished, response.status_code, meta["duration_ms"], state["id"]))
            con.execute("INSERT INTO console_trace_events(trace_id,at,kind,payload) VALUES (?,?,?,?)",
                        (state["id"], finished, "flask.response", json.dumps(meta)))
            con.executemany("INSERT INTO console_trace_spans "
                            "(trace_id,seq,event,depth,function,file,line,at,duration_ms,error) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)", spans)
        _write(_finish)
        return response

    app.extensions["console_trace"] = True


def _event_rows(con, since=0, limit=100):
    rows = con.execute("SELECT id,trace_id,at,kind,payload FROM console_trace_events "
                       "WHERE id>? ORDER BY id LIMIT ?", (since, limit)).fetchall()
    return [{"id": r[0], "trace_id": r[1], "at": r[2], "kind": r[3],
             "payload": json.loads(r[4])} for r in rows]


@trace_bp.route("/console/trace/events", methods=["POST"])
def trace_event():
    body = request.get_json(silent=True) or {}
    kind = _clean(body.get("kind"), 64) or "browser"
    payload = body.get("payload") if isinstance(body, dict) else {}
    raw = json.dumps(payload if isinstance(payload, dict) else {"value": str(payload)}, default=str)
    row = (_clean(body.get("trace_id"), 96) or None, time.time(), kind, raw[:_MAX_TEXT])
    stored = _write(lambda con: con.execute(
        "INSERT INTO console_trace_events(trace_id,at,kind,payload) VALUES (?,?,?,?)", row))
    return jsonify({"ok": True, "stored": stored})


@trace_bp.route("/console/trace", methods=["GET"])
def trace_list():
    trace_id = request.args.get("trace_id")
    limit = max(1, min(int(request.args.get("limit", 100)), 500))
    con = _db()
    if trace_id:
        req = con.execute("SELECT * FROM console_trace_requests WHERE id=?", (trace_id,)).fetchone()
        spans = con.execute("SELECT * FROM console_trace_spans WHERE trace_id=? ORDER BY seq LIMIT ?",
                            (trace_id, _MAX_SPANS)).fetchall()
        con.close()
        return jsonify({"request": dict(req) if req else None, "spans": [dict(x) for x in spans]})
    rows = con.execute("SELECT * FROM console_trace_requests ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return jsonify({"requests": [dict(x) for x in rows]})


@trace_bp.route("/console/trace/stream", methods=["GET"])
def trace_stream():
    since = int(request.args.get("since", 0) or 0)
    def generate():
        cursor = since; deadline = time.time() + 3600
        while time.time() < deadline:
            con = _db(); rows = _event_rows(con, cursor, 100); con.close()
            if rows:
                for row in rows:
                    cursor = row["id"]
                    yield "data: " + json.dumps(row, default=str) + "\n\n"
            else:
                yield ": keepalive\n\n"
            time.sleep(0.5)
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
