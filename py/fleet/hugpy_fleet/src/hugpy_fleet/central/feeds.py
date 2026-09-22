"""FEEDS-DB-V2.2-20260910: console feeds materialized in Postgres + a push channel.

Design: deploy/patches/feeds-db-0.1.249.py (v1) and feeds-v2-liveness-0.1.249.py."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

# volatile roster fields: their change alone must not re-push the 260 KB `workers` feed
_WORKER_VOLATILE = {"last_seen", "aggregate", "model_call_stats", "model_tok_stats", "last_picked",
                    "model_last_picked", "free_ram", "free_ram_raw", "ram_worker_bytes", "ram_external_bytes",
                    "vram_attributed_bytes", "vram_unattributed_bytes", "vram_holders", "calibration_samples"}
_WORKER_VOLATILE_PREFIX = ("bar_", "ram_bar_")


def _strip_volatile(rows):
    out = []
    for w in rows or []:
        if not isinstance(w, dict):
            out.append(w)
            continue
        c = {k: v for k, v in w.items() if k not in _WORKER_VOLATILE and not k.startswith(_WORKER_VOLATILE_PREFIX)}
        c["gpus"] = [{k: v for k, v in (g or {}).items() if k not in ("memory_free", "memory_used", "utilization")}
                     for g in (w.get("gpus") or [])]
        c["allocations"] = [{k: v for k, v in (a or {}).items() if k not in ("busy", "last_used", "vram_bytes", "rss_bytes",
                                                                             "rss_anon_bytes", "ram_resident_bytes")}
                            for a in (w.get("allocations") or [])]
        out.append(c)
    return out


def _liveness_builder(app):
    from hugpy_fleet.central import heartbeat_db
    live = heartbeat_db.liveness()
    have = {x["id"] for x in (live or [])}
    # workers that do not write to the DB yet: derive from the roster's own record
    roster = (_memory.get("workers") or {}).get("payload") or []
    for w in roster:
        if isinstance(w, dict) and w.get("id") not in have:
            (live := live if live is not None else []).append(
                heartbeat_db.liveness_from_record(w, w.get("last_seen"), 45.0))
    return sorted(live or [], key=lambda x: str(x.get("name")))


# `slots.resources` is central's own host-RAM snapshot (free/used/cache/available
# bytes) and drifts every second; the console never renders it, so keep it in the
# payload but leave it out of the digest (only the slot rows decide a re-push).
def _bucket_slot_resources(payload):
    if not isinstance(payload, dict) or "resources" not in payload:
        return payload
    return {k: v for k, v in payload.items() if k != "resources"}


# feed -> (builder: central path str | callable(app), min rebuild interval s, digest transform)
FEEDS = {
    "liveness":  (_liveness_builder, 2.0, None),
    "workers":   ("/llm/workers", 30.0, _strip_volatile),
    "slots":     ("/llm/slots", 3.0, _bucket_slot_resources),
    "queue":     ("/llm/queue", 2.0, None),
    "jobs":      ("/llm/jobs?live=0", 3.0, None),
    "downloads": ("/jobs", 3.0, None),
    "phones":    ("/phone-brick/phones", 10.0, None),
    "serving":   ("/llm/serving", 15.0, None),
    "peers":     ("/llm/peers", 15.0, None),
}
_ADVISORY_KEY = 7477_2026_0910
_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS hugpy_feed ("
    " feed TEXT PRIMARY KEY, version BIGINT NOT NULL DEFAULT 0,"
    " updated DOUBLE PRECISION NOT NULL, digest TEXT, payload JSONB NOT NULL)"
)

_state_lock = threading.Lock()
_memory: dict = {}
_started = False
_leader = False
_db = None
_db_ok = None


def _client():
    global _db
    if _db is None:
        from hugpy_engine.model_index.client import DatabaseClient
        _db = DatabaseClient()
    return _db


def _pg(fn):
    global _db_ok
    try:
        db = _client()
        with db.lock:
            with db.cursor() as cur:
                out = fn(cur)
        if _db_ok is not True:
            _db_ok = True
        return out
    except Exception as exc:  # noqa: BLE001
        if _db_ok is not False:
            log.warning("feeds: postgres unavailable (%s: %s) — serving from memory", type(exc).__name__, exc)
        _db_ok = False
        try:
            _client().mark_unavailable(exc, "feeds")
        except Exception:  # noqa: BLE001
            pass
        return None


def _ensure_table():
    _pg(lambda cur: cur.execute(_TABLE_SQL))


def _try_lead() -> bool:
    row = _pg(lambda cur: (cur.execute("SELECT pg_try_advisory_lock(%s)", (_ADVISORY_KEY,)), cur.fetchone())[1])
    return bool(row and row[0])


def _digest(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _write(feed: str, payload, digest: str, now: float) -> None:
    def op(cur):
        from psycopg.types.json import Json
        cur.execute(
            "INSERT INTO hugpy_feed (feed, version, updated, digest, payload) VALUES (%s, 1, %s, %s, %s)"
            " ON CONFLICT (feed) DO UPDATE SET version = hugpy_feed.version + 1, updated = EXCLUDED.updated,"
            " digest = EXCLUDED.digest, payload = EXCLUDED.payload RETURNING version",
            (feed, now, digest, Json(payload, dumps=lambda o: json.dumps(o, default=str))))
        return cur.fetchone()[0]
    version = _pg(op)
    with _state_lock:
        prev = _memory.get(feed) or {}
        _memory[feed] = {"version": version if version is not None else (prev.get("version") or 0) + 1,
                         "updated": now, "digest": digest, "payload": payload}


def _build(app, builder):
    if callable(builder):
        return builder(app)
    with app.test_client() as c:
        r = c.get(builder, environ_base={"REMOTE_ADDR": "127.0.0.1"}, headers={"User-Agent": "hugpy-feeds/1"})
        if r.status_code != 200:
            raise RuntimeError(f"{builder} -> HTTP {r.status_code}")
        return r.get_json()


def _refresh_loop(app) -> None:
    global _leader
    last_try: dict = {}
    pending: dict = {}
    # One bounded task per feed: slow inventory/provisioning reads cannot
    # delay the liveness feed, and slow feeds never accumulate duplicate work.
    pool = ThreadPoolExecutor(max_workers=len(FEEDS), thread_name_prefix="hugpy-feed")
    _ensure_table()
    while True:
        try:
            if not _try_lead():
                _leader = False
                time.sleep(5.0)
                continue
            _leader = True
            now = time.time()
            _schedule_due(app, pool, pending, last_try, now)
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            log.exception("feeds: refresher iteration failed")
            time.sleep(3.0)


def _schedule_due(app, pool, pending, last_try, now):
    for feed, (builder, interval, transform) in FEEDS.items():
        if feed in pending and not pending[feed].done():
            continue
        if now - last_try.get(feed, 0.0) < interval:
            continue
        last_try[feed] = now
        pending[feed] = pool.submit(_refresh_one, app, feed, builder, transform)


def _refresh_one(app, feed, builder, transform):
    try:
        payload = _build(app, builder)
        d = _digest(transform(payload) if transform else payload)
        with _state_lock:
            prev = (_memory.get(feed) or {}).get("digest")
        if d != prev:
            _write(feed, payload, d, time.time())
    except Exception as exc:
        log.warning("feeds: %s failed: %s", feed, exc)


def ensure_refresher(app) -> None:
    global _started
    with _state_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_refresh_loop, args=(app,), name="hugpy-feeds", daemon=True).start()


def read_versions() -> dict:
    rows = _pg(lambda cur: (cur.execute("SELECT feed, version FROM hugpy_feed"), cur.fetchall())[1])
    if rows is not None:
        return {f: int(v) for f, v in rows}
    with _state_lock:
        return {f: e["version"] for f, e in _memory.items()}


def read_all(feeds=None) -> dict:
    want = set(feeds) if feeds else set(FEEDS)
    rows = _pg(lambda cur: (cur.execute("SELECT feed, version, updated, payload FROM hugpy_feed"), cur.fetchall())[1])
    out = {}
    if rows is not None:
        for f, v, u, p in rows:
            if f in want:
                out[f] = {"version": int(v), "updated": u, "payload": p}
        return out
    with _state_lock:
        for f, e in _memory.items():
            if f in want:
                out[f] = {"version": e["version"], "updated": e["updated"], "payload": e["payload"]}
    return out


def is_leader() -> bool:
    return _leader


def db_ok():
    return _db_ok
