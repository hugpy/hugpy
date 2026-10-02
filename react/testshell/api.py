#!/usr/bin/env python3
"""hugpy testshell API: the UI reflects the DB, the knobs turn the DB.

Reads one row per model from the `model_full` view (DDL: model_full.sql).
Writes exactly two things:
  * operator knobs      model_workers.user_settings   (POST .../knobs)
  * computed-once facts models.weights, model_workers.plan  (compute.py)
The computed facts are filled in lazily: a background pass runs at startup
and every ENSURE_S seconds, and a knob write recomputes that pair at once.
Serves ./dist as the SPA when built.

  HUGPY_TESTSHELL_DSN   libpq DSN (default: dbname=hugpy, local socket)
  HUGPY_TESTSHELL_PORT  listen port (default 7013)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute  # noqa: E402  (shim: path rule for the editable checkouts, then the engine module)

# The planner's DB side moved into the engine (2026-10-02); this file is the
# testshell HTTP host over it: same routes, same background cadence.
from hugpy_engine.model_index.planner import store as _store  # noqa: E402
from hugpy_engine.model_index.planner.store import (  # noqa: E402
    DSN, J, _as_json, _connect, _ensure_state, _json_default, _registry, _rows, _selected_verdict,
    TABLES_SQL, ensure_all, ensure_model, load_models, load_workers, refresh_worker_budgets)

PORT = int(os.environ.get("HUGPY_TESTSHELL_PORT", "7013"))
ENSURE_S = int(os.environ.get("HUGPY_TESTSHELL_ENSURE_S", "60"))
DIST = Path(__file__).resolve().parent / "dist"


def _ensure_loop():
    while True:
        try:
            ensure_all()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        time.sleep(ENSURE_S)


# ── http ─────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "hugpy-testshell/0.2"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode() or "{}")

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path.startswith("/api/"):
                return self._api_get(path[len("/api"):])
            return self._static(path)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path.startswith("/api/"):
                return self._api_post(path[len("/api"):], self._body())
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _api_get(self, path):
        t0 = time.time()
        with _connect(DSN) as conn, conn.cursor() as cur:
            if path == "/health":
                cur.execute("SELECT current_database(), now()")
                db, now = cur.fetchone()
                return self._send(200, {"ok": True, "db": db, "now": now, "tables": _rows(cur, TABLES_SQL),
                                        "ensure": _ensure_state, "knob_keys": compute.KNOB_KEYS,
                                        "alloc_modes": compute.ALLOC_MODES})
            if path == "/workers":
                return self._send(200, load_workers(cur))
            if path == "/models":
                models = load_models(cur)
                return self._send(200, {"models": models, "workers": load_workers(cur), "count": len(models),
                                        "ensure": _ensure_state, "alloc_modes": compute.ALLOC_MODES,
                                        "knob_keys": compute.KNOB_KEYS,
                                        "took_ms": round((time.time() - t0) * 1000)})
            if path.startswith("/models/"):
                mid = _int(path.rsplit("/", 1)[1])
                if mid is None:
                    return self._send(400, {"error": "model id must be an integer"})
                rows = load_models(cur, mid)
                return self._send(200, rows[0]) if rows else self._send(404, {"error": "no such model"})
        self._send(404, {"error": f"unknown endpoint {path}"})

    def _api_post(self, path, body):
        parts = path.strip("/").split("/")
        with _connect(DSN) as conn, conn.cursor() as cur:
            # POST /api/models/<id>/recompute
            if len(parts) == 3 and parts[0] == "models" and parts[2] == "recompute":
                mid = _int(parts[1])
                specs, plans = ensure_model(cur, mid, force=True)
                rows = load_models(cur, mid)
                return self._send(200, {"model": rows[0] if rows else None, "specs": specs, "plans": plans})
            # POST /api/models/<id>/workers/<worker_id>/knobs  {"set": {...}, "unset": [...]}
            if len(parts) == 5 and parts[0] == "models" and parts[2] == "workers" and parts[4] == "knobs":
                mid, wid = _int(parts[1]), parts[3]
                reg = _registry(cur)
                if mid is None or wid not in reg:
                    return self._send(400, {"error": "unknown model id or worker id"})
                to_set = body.get("set") or {}
                to_unset = [str(k) for k in (body.get("unset") or [])]
                bad = [k for k in list(to_set) + to_unset if k not in compute.KNOB_KEYS]
                if bad:
                    return self._send(400, {"error": f"unknown knob(s) {bad}; allowed: {list(compute.KNOB_KEYS)}"})
                if to_set.get("moe") is True:
                    cur.execute("SELECT user_settings FROM model_workers WHERE model_id = %s AND worker_id = %s", (mid, wid))
                    r = cur.fetchone()
                    knobs_now = dict(_as_json(r[0]) or {}) if r else {}
                    knobs_now.update(to_set)
                    for k in to_unset: knobs_now.pop(k, None)
                    sel = _selected_verdict(cur, mid, wid, knobs_now) or {}
                    if not sel.get("moe_offered"):
                        return self._send(409, {"error": "MoE is not offered for this quant on this worker — "
                                                         + ((sel.get("moe") or {}).get("why") or "no verdict computed yet")})
                # Knobs ride an EXISTING pair only. A pair row exists when the
                # model is assigned to (or live on) the worker; a knob written
                # for any other worker would be residue (operator, 2026-10-01).
                cur.execute(
                    "UPDATE model_workers SET"
                    "   user_settings = (COALESCE(user_settings, '{}'::jsonb) || %s::jsonb) - %s::text[],"
                    "   updated_at = now()"
                    " WHERE model_id = %s AND worker_id = %s",
                    (J(to_set), to_unset, mid, wid))
                if cur.rowcount == 0:
                    return self._send(409, {"error": "model is not assigned to this worker — assign it first; "
                                                     "knobs on an unassigned pair would be residue"})
                specs, plans = ensure_model(cur, mid, reg)
                rows = load_models(cur, mid)
                return self._send(200, {"model": rows[0] if rows else None, "plans": plans})
            if path == "/ensure":
                threading.Thread(target=ensure_all, daemon=True).start()
                return self._send(202, {"started": not _ensure_state["running"]})
        self._send(404, {"error": f"unknown endpoint {path}"})

    def _static(self, path):
        if not DIST.is_dir():
            return self._send(503, {"error": "dist/ not built; run `npm run build` or use the vite dev server"})
        target = (DIST / path.lstrip("/")).resolve()
        if not str(target).startswith(str(DIST)) or not target.is_file():
            target = DIST / "index.html"
        ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                 ".svg": "image/svg+xml", ".json": "application/json"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    with _connect(DSN) as c, c.cursor() as cur:
        cur.execute("SELECT count(*), count(weights) FROM models")
        n, done = cur.fetchone()
        print(f"hugpy testshell api :{PORT}  dsn={DSN!r}  models={n}  weights filled={done}", flush=True)
    threading.Thread(target=_ensure_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
