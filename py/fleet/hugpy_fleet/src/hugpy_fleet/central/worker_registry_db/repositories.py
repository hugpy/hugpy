"""Database operations for the central worker registry."""
from __future__ import annotations

import time

from . import query_registry as sql
from hugpy_engine.model_index.query_registry import ModelQueries


class WorkerRegistryRepository:
    def ensure(self, cur) -> None:
        # bounded lock waits: a contended schema statement fails THIS call (the
        # service resets _schema_ready and retries later) instead of parking the
        # shared connection behind another session's lock (2026-10-02 outage).
        cur.execute(ModelQueries.DDL_LOCK_TIMEOUT_ON)
        try:
            cur.execute(sql.CREATE_META)
            cur.execute(sql.CREATE_WORKERS)
            cur.execute(sql.ENSURE_META)
            # Model allocation/activity is a database-level projection of every
            # worker-registry write, including writes from older central processes.
            cur.execute(ModelQueries.SYNC_WORKER_REGISTRY_FUNCTION)
            cur.execute(ModelQueries.INSTALL_WORKER_REGISTRY_TRIGGER)
        finally:
            try:
                cur.execute(ModelQueries.DDL_LOCK_TIMEOUT_OFF)
            except Exception:  # noqa: BLE001
                pass

    def lock_meta(self, cur) -> bool:
        cur.execute(sql.LOCK_META)
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("worker registry metadata row is missing")
        return bool(row[0])

    def set_initialized(self, cur) -> None:
        cur.execute(sql.SET_INITIALIZED)

    def read_all(self, cur) -> dict:
        cur.execute(sql.READ_ALL)
        rows = cur.fetchall()
        workers = {}
        for worker_id, payload in rows:
            if not isinstance(payload, dict) or payload.get("id") != worker_id:
                raise RuntimeError(f"invalid worker registry row {worker_id!r}")
            workers[worker_id] = payload
        return workers

    def upsert(self, cur, worker_id: str, payload: dict) -> None:
        from psycopg.types.json import Jsonb

        cur.execute(sql.UPSERT, (worker_id, time.time(), Jsonb(payload)))

    def delete(self, cur, worker_id: str) -> None:
        cur.execute(sql.DELETE, (worker_id,))
