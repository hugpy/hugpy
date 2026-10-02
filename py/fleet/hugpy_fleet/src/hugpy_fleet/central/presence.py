"""hugpy-link presence (2026-10-02): the CONNECTION, apart from the STATE.

A worker's online/offline used to be "last heavy heartbeat < 45 s", and that
beat gathers slots / pid registry / nvidia-smi / allocations / margins before
it posts — so a slow beat, or central mid-restart, flipped a live worker to
"disconnected". The hugpy-link daemon on each worker host pings
``POST /api/llm/workers/<id>/presence`` every few seconds with the local
worker's process state (up / busy / restarting / down); this module stores it
in its OWN table over its OWN connection (never the heartbeat client's lock, so
it cannot queue behind a heavy beat transaction) and folds it into the liveness
rows. A late heartbeat then means "state N s old", not "disconnected".

Fail-open everywhere, like heartbeat_db: no DSN / no driver / no table just
means no presence data, and liveness falls back to the heartbeat alone."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

TABLE = "hugpy_worker_presence"
_DDL = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} ("
    " worker_id TEXT PRIMARY KEY, ts DOUBLE PRECISION NOT NULL,"
    " worker_state TEXT, version TEXT, payload JSONB NOT NULL DEFAULT '{}'::jsonb)"
)
WORKER_STATES = ("up", "busy", "restarting", "down")

_lock = threading.Lock()
_conn = None
_conn_pid = None
_ddl_done = False
_warned = False


def link_stale_s() -> float:
    """Seconds without a ping before the LINK counts as down (HUGPY_LINK_STALE_S)."""
    try:
        return max(5.0, float(os.environ.get("HUGPY_LINK_STALE_S") or 20.0))
    except ValueError:
        return 20.0


def _dsn():
    from hugpy_fleet.central.heartbeat_db import dsn
    return dsn()


def _run(fn):
    """One short autocommit statement on presence's own connection, with a
    5 s lock_timeout; None on any failure (fail-open)."""
    global _conn, _conn_pid, _ddl_done, _warned
    if not _dsn():
        return None
    with _lock:
        try:
            import psycopg
            if _conn is None or _conn.closed or _conn_pid != os.getpid():
                _conn = psycopg.connect(_dsn(), autocommit=True, connect_timeout=5)
                _conn_pid = os.getpid()
                with _conn.cursor() as cur:
                    cur.execute("SET lock_timeout = '5s'")
            with _conn.cursor() as cur:
                if not _ddl_done:
                    cur.execute(_DDL)
                    _ddl_done = True
                return fn(cur)
        except Exception as exc:  # noqa: BLE001
            if not _warned:
                log.warning("presence: unavailable (%s: %s) — liveness uses the heartbeat alone",
                            type(exc).__name__, exc)
                _warned = True
            try:
                if _conn is not None:
                    _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None
            return None


def record(worker_id: str, body: dict) -> bool:
    """Upsert one ping. ``worker_state`` outside WORKER_STATES is stored as None."""
    state = str((body or {}).get("worker_state") or "").strip().lower() or None
    if state not in WORKER_STATES:
        state = None
    payload = {k: body.get(k) for k in ("uptime_s", "worker_pid", "health_ms", "link_version",
                                        "unit_state", "host") if k in (body or {})}

    def op(cur):
        cur.execute(f"INSERT INTO {TABLE} (worker_id, ts, worker_state, version, payload) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb) ON CONFLICT (worker_id) DO UPDATE SET "
                    "ts=EXCLUDED.ts, worker_state=EXCLUDED.worker_state, version=EXCLUDED.version, "
                    "payload=EXCLUDED.payload",
                    (str(worker_id), time.time(), state, (body or {}).get("version"),
                     json.dumps(payload, default=str)))
        return True
    return bool(_run(op))


def rows() -> "dict[str, dict]":
    def op(cur):
        cur.execute(f"SELECT worker_id, ts, worker_state, version, payload FROM {TABLE}")
        return cur.fetchall()
    out = {}
    for wid, ts, state, version, payload in (_run(op) or []):
        out[wid] = {"ts": ts, "worker_state": state, "version": version, "payload": payload or {}}
    return out


def merge(live: "list[dict] | None", now: "float | None" = None) -> "list[dict] | None":
    """Fold presence into liveness rows (in place + returned).

    Adds ``link`` (up/down/None = no daemon), ``link_ts``, ``worker_state`` and
    ``state_stale`` (the heavy heartbeat is past its window). The state's age is
    computed by the console from ``last_seen`` — a per-build age would change
    the feed digest every rebuild and re-push it every 2 s.
    ``status`` becomes online when the heartbeat is fresh OR the link is up and
    the worker process is up/busy — a late state report alone never reads as
    disconnected. Workers without a daemon keep the heartbeat-only status."""
    if not live:
        return live
    pres = rows()
    now = time.time() if now is None else now
    stale = link_stale_s()
    for row in live:
        p = pres.get(row.get("id"))
        if not p:
            row["link"] = None
            continue
        link_up = (now - float(p["ts"])) < stale
        row["link"] = "up" if link_up else "down"
        row["link_ts"] = p["ts"]
        row["worker_state"] = p["worker_state"] if link_up else None
        heartbeat_online = row.get("status") == "online"
        row["state_stale"] = not heartbeat_online
        if link_up and p["worker_state"] in ("up", "busy"):
            row["status"] = "online"
    return live
