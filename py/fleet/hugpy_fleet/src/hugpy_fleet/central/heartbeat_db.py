"""FEEDS-DB-V2.2-20260910: worker heartbeats in Postgres.

Workers upsert one row per beat (dual-write beside the HTTP beat to central);
central derives the small `liveness` feed from these rows. Fail-open everywhere:
a missing DSN, driver or database just means "no DB heartbeat", never an error
on the request path."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

TABLE = "hugpy_worker_heartbeat"
_DDL = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} ("
    " worker_id TEXT PRIMARY KEY, name TEXT, ts DOUBLE PRECISION NOT NULL,"
    " payload JSONB NOT NULL)"
)
# keep the stored beat useful but bounded: these keys are large and never needed for liveness
_DROP = ("models_local", "load_reports", "model_call_stats", "model_tok_stats", "model_last_picked",
         "pid_registry", "storage", "calibration_samples", "vram_holders", "loaded_detail", "planned_split",
         "moe_by_model", "bnb_by_model", "env", "install", "task_capabilities", "vram_evictions")

_lock = threading.Lock()
_db = None
_warned = False
_ddl_done = False
_activity_ddl_done = False


def dsn() -> str | None:
    v = (os.environ.get("HUGPY_REGISTRY_PG_DSN") or "").strip()
    return v or None


class _DirectClient:
    """Fallback when imports.src.model_index is not shipped (stock wheels <= 0.1.249):
    same surface as DatabaseClient (lock / cursor() / mark_unavailable), psycopg3 autocommit."""

    def __init__(self):
        self.lock = threading.Lock()
        self._conn = None
        self._pid = None

    def cursor(self):
        import psycopg
        if self._conn is None or self._conn.closed or self._pid != os.getpid():
            self._conn = psycopg.connect(dsn(), autocommit=True, connect_timeout=5)
            self._pid = os.getpid()
        return self._conn.cursor()

    def mark_unavailable(self, exc, doing):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._conn = None


def _client():
    global _db
    if _db is None:
        try:
            from hugpy_engine.model_index.client import DatabaseClient
            _db = DatabaseClient(dsn_resolver=dsn)
        except ImportError:
            _db = _DirectClient()
    return _db


def _run(fn):
    global _warned, _ddl_done
    if not dsn():
        return None
    try:
        db = _client()
        with db.lock:
            with db.cursor() as cur:
                if not _ddl_done:
                    cur.execute(_DDL)
                    _ddl_done = True
                return fn(cur)
    except Exception as exc:  # noqa: BLE001
        if not _warned:
            log.warning("heartbeat_db: unavailable (%s: %s) — continuing without DB heartbeats", type(exc).__name__, exc)
            _warned = True
        try:
            _client().mark_unavailable(exc, "heartbeat_db")
        except Exception:  # noqa: BLE001
            pass
        return None


def _activity_table(cur):
    global _activity_ddl_done
    if not _activity_ddl_done:
        cur.execute("CREATE TABLE IF NOT EXISTS hugpy_slot_activity ("
                    "worker_id TEXT NOT NULL, slot_id TEXT NOT NULL, model_key TEXT,"
                    "busy BOOLEAN NOT NULL, observed_at DOUBLE PRECISION NOT NULL,"
                    "PRIMARY KEY(worker_id, slot_id))")
        _activity_ddl_done = True


def record_activity(worker_id, slot_id, model_key, busy, observed_at):
    """Small edge update; older asynchronous writes cannot undo newer ones."""
    def op(cur):
        _activity_table(cur)
        cur.execute("INSERT INTO hugpy_slot_activity VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (worker_id,slot_id) DO UPDATE SET "
                    "model_key=EXCLUDED.model_key,busy=EXCLUDED.busy,observed_at=EXCLUDED.observed_at "
                    "WHERE hugpy_slot_activity.observed_at < EXCLUDED.observed_at",
                    (worker_id, str(slot_id), model_key, bool(busy), float(observed_at)))
        return True
    return bool(_run(op))


def _merge_activity(live, records, events):
    by_id = {r['worker_id']: r for r in records}
    by_live = {r['id']: r for r in live}
    for wid, slot_id, key, busy, observed in events:
        row = by_live.get(wid)
        rec = by_id.get(wid)
        if not row or not rec or row['status'] != 'online':
            continue
        payload = rec.get('payload') or {}
        sampled = payload.get('activity_sampled_at', rec.get('ts') or 0)
        if observed <= sampled:
            continue
        answering = set(row['answering'])
        # A slot may have changed models since the full heartbeat.
        for a in payload.get('allocations') or []:
            if str(a.get('slot_id')) == str(slot_id):
                answering.discard(a.get('model_key'))
        if busy and key:
            answering.add(key)
        else:
            answering.discard(key)
        row['answering'] = sorted(answering)
    return live


def upsert(worker_id: str, name: str, payload: dict) -> bool:
    """Worker side: store this beat. Returns True when written."""
    if not worker_id or not dsn():
        return False
    slim = {k: v for k, v in (payload or {}).items() if k not in _DROP}
    slim["name"] = name

    def op(cur):
        from psycopg.types.json import Json
        cur.execute(
            f"INSERT INTO {TABLE} (worker_id, name, ts, payload) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (worker_id) DO UPDATE SET name = EXCLUDED.name, ts = EXCLUDED.ts, payload = EXCLUDED.payload",
            (worker_id, name, time.time(), Json(slim, dumps=lambda o: json.dumps(o, default=str))))
        return True
    return bool(_run(op))


def rows() -> list[dict] | None:
    """Central side: every stored beat. None when the DB is unavailable."""
    out = _run(lambda cur: (cur.execute(f"SELECT worker_id, name, ts, payload FROM {TABLE}"), cur.fetchall())[1])
    if out is None:
        return None
    return [{"worker_id": w, "name": n, "ts": t, "payload": p} for w, n, t, p in out]


def liveness_from_record(rec: dict, ts: float | None, stale_s: float) -> dict:
    """The small per-worker liveness row (shared shape for DB rows and roster rows)."""
    now = time.time()
    allocs = rec.get("allocations") or []
    gpus = [{"index": g.get("index"), "memory_total": g.get("memory_total"), "memory_free": g.get("memory_free")}
            for g in (rec.get("gpus") or []) if isinstance(g, dict)]
    return {
        "id": rec.get("id") or rec.get("worker_id"),
        "name": rec.get("name"),
        "last_seen": ts,
        "status": ("online" if (ts is not None and now - ts < stale_s) else "offline"),
        "answering": sorted({a.get("model_key") for a in allocs if isinstance(a, dict) and a.get("model_key") and a.get("busy")}),
        "loaded_models": list(rec.get("loaded_models") or []),
        "loading": list(rec.get("loading") or []),
        "gpus": gpus,
        "free_ram": rec.get("free_ram"),
        "pkg_version": rec.get("pkg_version"),
    }


def liveness(stale_s: float = 45.0) -> list[dict] | None:
    r = rows()
    if r is None:
        return None
    live = [liveness_from_record({**(x["payload"] or {}), "id": x["worker_id"], "name": x["name"]}, x["ts"], stale_s) for x in r]
    def activity(cur):
        _activity_table(cur)
        cur.execute("SELECT worker_id,slot_id,model_key,busy,observed_at FROM hugpy_slot_activity")
        return cur.fetchall()
    return _merge_activity(live, r, _run(activity) or [])
