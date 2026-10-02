"""Transactional PostgreSQL storage for central worker records."""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Callable

from hugpy_engine.model_index.client import DatabaseClient
from .repositories import WorkerRegistryRepository


class WorkerRegistryDB:
    def __init__(self) -> None:
        self.db = DatabaseClient()
        self.repo = WorkerRegistryRepository()
        self._schema_ready = False
        self._initialized = False

    def _ensure(self, cur) -> None:
        if not self._schema_ready:
            self.repo.ensure(cur)
            self._schema_ready = True

    def _bootstrap(self, seed: Callable[[], dict]) -> None:
        # The metadata row is the migration lock. Even an empty imported fleet
        # is initialized, so a later read cannot resurrect stale JSON workers.
        try:
            with self.db.transaction():
                with self.db.cursor() as cur:
                    self._ensure(cur)
                    if not self.repo.lock_meta(cur):
                        for worker_id, row in seed().items():
                            self.repo.upsert(cur, worker_id, row)
                        self.repo.set_initialized(cur)
        except Exception:
            # DDL participates in the transaction and may have rolled back.
            self._schema_ready = False
            raise
        self._initialized = True

    def read(self, seed: Callable[[], dict]) -> dict:
        with self.db.lock:
            if not self._initialized:
                self._bootstrap(seed)
            with self.db.cursor() as cur:
                return self.repo.read_all(cur)

    @contextmanager
    def transaction(self, seed: Callable[[], dict]):
        with self.db.lock:
            if not self._initialized:
                self._bootstrap(seed)
            with self.db.transaction():
                with self.db.cursor() as cur:
                    # Serialize cross-process read/modify/write transactions.
                    # Individual workers remain separate rows and only changed
                    # rows are written, even when a heartbeat changes one box.
                    self.repo.lock_meta(cur)
                    workers = self.repo.read_all(cur)
                    before = {worker_id: json.dumps(row, sort_keys=True)
                              for worker_id, row in workers.items()}
                    yield workers
                    for worker_id in before.keys() - workers.keys():
                        self.repo.delete(cur, worker_id)
                    for worker_id, row in workers.items():
                        if json.dumps(row, sort_keys=True) != before.get(worker_id):
                            self.repo.upsert(cur, worker_id, row)
