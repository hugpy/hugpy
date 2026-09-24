"""DatabaseClient — the ONE place the model-index touches a connection
(explicit-wiring ruling 2026-09-10, modeled on solcatcher's DatabaseClient).

Owns: DSN resolution from env, the PID fork-guard, psycopg3 autocommit +
explicit transactions, and the fail-open "unavailable" bookkeeping. Repos and
the service never import psycopg or hold a connection themselves.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger("abstract_hugpy_dev.model_index")


def enabled() -> bool:
    """HUGPY_REGISTRY_DB=pg turns the store on; anything else = inert (worker
    boxes have neither the flag nor psycopg and must stay on the JSON path)."""
    return (os.environ.get("HUGPY_REGISTRY_DB") or "").strip().lower() == "pg"


def resolve_dsn() -> str:
    """HUGPY_REGISTRY_PG_DSN verbatim (URI form — a space-separated DSN breaks
    shell sourcing of the env file), else composed from the toolserver's
    SOLCATCHER_POSTGRESQL_* host/port with hugpy's own user/database."""
    dsn = (os.environ.get("HUGPY_REGISTRY_PG_DSN") or "").strip()
    if dsn:
        return dsn
    host = os.environ.get("SOLCATCHER_POSTGRESQL_HOST", "127.0.0.1")
    port = os.environ.get("SOLCATCHER_POSTGRESQL_PORT", "5432")
    user = (os.environ.get("HUGPY_REGISTRY_PG_USER")
            or os.environ.get("SOLCATCHER_POSTGRESQL_USER") or "hugpy")
    pw = (os.environ.get("HUGPY_REGISTRY_PG_PASS")
          or os.environ.get("SOLCATCHER_POSTGRESQL_PASS") or "")
    db = os.environ.get("HUGPY_REGISTRY_PG_DB", "hugpy")
    return f"host={host} port={port} dbname={db} user={user} password={pw}"


class DatabaseClient:
    """Process-local psycopg3 connection with fork-guard and a lock.

    NEVER ``with conn:`` — psycopg3's connection context manager CLOSES the
    connection on exit (unlike psycopg2's transaction-only semantics). Use
    ``client.transaction()`` / ``client.cursor()``.
    """

    def __init__(self, dsn_resolver=resolve_dsn):
        self._dsn_resolver = dsn_resolver
        self._lock = threading.Lock()
        self._conn = None
        self._pid = None
        self._warned = False
        # The last recorded fault (what was being done, the real exception
        # text, when) — surfaced by read/write routes so an empty result or a
        # failed write says WHY instead of reading as "nothing recorded".
        self.last_error = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def connect(self):
        """The live connection for THIS process, (re)opening as needed.

        FORK GUARD: gunicorn forks after import-time discovery may have opened
        a connection; two processes sharing one socket corrupt the stream
        ("SSL error: decryption failed or bad record mac", 2026-09-10). A
        connection is only ever reused by the PID that opened it."""
        import psycopg
        if (self._conn is not None and not self._conn.closed
                and self._pid == os.getpid()):
            return self._conn
        self._pid = os.getpid()
        self._conn = psycopg.connect(self._dsn_resolver(), connect_timeout=5,
                                     autocommit=True)
        return self._conn

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def transaction(self):
        return self.connect().transaction()

    def cursor(self):
        return self.connect().cursor()

    def mark_unavailable(self, exc: BaseException, doing: str) -> None:
        """Drop the connection and log ONCE per process — every caller then
        degrades to its JSON fallback rather than raising."""
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        self._conn = None
        import time
        self.last_error = {"doing": doing, "error": f"{type(exc).__name__}: {exc}",
                           "at": time.time()}
        if not self._warned:
            self._warned = True
            logger.warning("registry DB unavailable while %s (%s) — falling "
                           "back to the JSON report (logged once)", doing, exc)
