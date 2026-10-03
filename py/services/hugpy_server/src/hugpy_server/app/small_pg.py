"""One small fail-open PostgreSQL runner for central's side stores (test-fire
history, help tickets): its OWN autocommit connection with a 5 s lock_timeout
(never the registry's or heartbeat's lock), DDL once per process, and None on
any failure so a DB outage costs the feature, never the caller."""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)


def registry_dsn() -> Optional[str]:
    """HUGPY_REGISTRY_PG_DSN, else the registry's own DSN when it runs on PG."""
    v = (os.environ.get("HUGPY_REGISTRY_PG_DSN") or "").strip()
    if v:
        return v
    try:
        from hugpy_engine.model_index.client import enabled, resolve_dsn
    except ImportError:
        return None
    return resolve_dsn() if enabled() else None


class Runner:
    def __init__(self, name: str, ddl: str, dsn: Callable[[], Optional[str]] = registry_dsn):
        self.name, self.ddl, self._dsn_fn = name, ddl, dsn
        self._lock = threading.Lock()
        self._conn = None
        self._pid = None
        self._ddl_done = False
        self._warned = False

    def run(self, fn):
        dsn = self._dsn_fn()
        if not dsn:
            return None
        with self._lock:
            try:
                import psycopg
                if self._conn is None or self._conn.closed or self._pid != os.getpid():
                    self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
                    self._pid = os.getpid()
                    with self._conn.cursor() as cur:
                        cur.execute("SET lock_timeout = '5s'")
                with self._conn.cursor() as cur:
                    if not self._ddl_done:
                        cur.execute(self.ddl)
                        self._ddl_done = True
                    return fn(cur)
            except Exception as exc:  # noqa: BLE001
                if not self._warned:
                    log.warning("%s unavailable (%s: %s)", self.name, type(exc).__name__, exc)
                    self._warned = True
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
                return None


def rows(cur) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
