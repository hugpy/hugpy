"""Database operations for the central worker registry."""
from __future__ import annotations

import time

from . import query_registry as sql


class WorkerRegistryRepository:
    def ensure(self, cur) -> None:
        cur.execute(sql.CREATE_META)
        cur.execute(sql.CREATE_WORKERS)
        cur.execute(sql.ENSURE_META)

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
