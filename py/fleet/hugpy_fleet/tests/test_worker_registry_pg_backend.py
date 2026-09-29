"""The worker registry's PostgreSQL cutover must not write its JSON seed."""
from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager

import pytest

from hugpy_fleet.central.workers import WorkerStore
from hugpy_fleet.central.worker_registry_db.service import WorkerRegistryDB


def test_backend_requires_explicit_cutover(monkeypatch, tmp_path):
    import hugpy_fleet.central.worker_registry_db as package

    marker = object()
    monkeypatch.setattr(package, "WorkerRegistryDB", lambda: marker)
    monkeypatch.setenv("HUGPY_WORKER_REGISTRY_BACKEND", "pg")
    monkeypatch.setenv("HUGPY_REGISTRY_DB", "pg")
    assert WorkerStore()._pg is marker
    assert WorkerStore(path=str(tmp_path / "isolated.json"))._pg is None
    monkeypatch.setenv("HUGPY_WORKER_REGISTRY_BACKEND", "psql")
    with pytest.raises(ValueError):
        WorkerStore()


def test_shared_storage_is_not_reported_as_reclaimable_collapse(caplog):
    from hugpy_fleet.central.workers import storage_proposal

    worker = {"id": "w1", "name": "a-brain", "storage": {"models": [
        {"model_key": "m", "bytes": 1024, "store": "shared",
         "counts_toward_budget": False, "why": "shared/central storage — never reaped"}]}}
    with caplog.at_level("DEBUG"):
        storage_proposal(worker)
        storage_proposal(worker)
    assert not [r for r in caplog.records if "reclaimable collapse" in r.message]


class _FakeRegistryDB:
    def __init__(self):
        self.rows = None

    def read(self, seed):
        if self.rows is None:
            self.rows = copy.deepcopy(seed())
        return copy.deepcopy(self.rows)

    @contextmanager
    def transaction(self, seed):
        current = self.read(seed)
        yield current
        self.rows = copy.deepcopy(current)


def test_postgres_registry_imports_once_and_owns_subsequent_mutations(tmp_path):
    path = tmp_path / "workers.json"
    path.write_text(json.dumps({"workers": [{"id": "w1", "name": "ae"}]}))
    db = _FakeRegistryDB()
    store = WorkerStore(path=str(path))
    store._pg = db

    assert store._load()["w1"]["name"] == "ae"
    with store._transaction() as workers:
        workers["w1"]["name"] = "a-brain"
    assert store._load()["w1"]["name"] == "a-brain"
    assert json.loads(path.read_text())["workers"][0]["name"] == "ae"

    other_process = WorkerStore(path=str(path))
    other_process._pg = db
    assert other_process._load()["w1"]["name"] == "a-brain"

    with pytest.raises(RuntimeError):
        with store._transaction() as workers:
            workers["w1"]["name"] = "rolled back"
            raise RuntimeError("abort")
    assert other_process._pg.read(lambda: {})["w1"]["name"] == "a-brain"


def test_storage_view_is_materialized_on_change_or_expiry(monkeypatch, tmp_path):
    import hugpy_fleet.central.workers as module

    path = tmp_path / "workers.json"
    path.write_text(json.dumps({"workers": [{"id": "w1", "name": "ae",
        "storage": {"models": [{"model_key": "m", "store": "shared",
                                 "counts_toward_budget": False}]}}]}))
    calls = []
    monkeypatch.setattr(module, "storage_proposal", lambda worker: calls.append(1) or {"n": len(calls)})
    db = _FakeRegistryDB()
    store = WorkerStore(path=str(path))
    store._pg = db

    assert store._load()["w1"]["_storage_view_cache"]["view"]["n"] == 1
    with monkeypatch.context() as patch:
        patch.setattr(module, "storage_proposal", lambda worker: (_ for _ in ()).throw(AssertionError("recomputed on read")))
        assert store.get("w1")["storage"]["n"] == 1
    store._cache_at = 0
    assert store._load()["w1"]["_storage_view_cache"]["view"]["n"] == 1
    with store._transaction() as workers:
        workers["w1"]["storage"]["models"][0]["store"] = "reapable"
    assert db.rows["w1"]["_storage_view_cache"]["view"]["n"] == 2

    db.rows["w1"]["_storage_view_cache"]["expires_at"] = 0
    store._cache_at = 0
    assert store._load()["w1"]["_storage_view_cache"]["view"]["n"] == 3


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, repo):
        self.lock = threading.Lock()
        self.repo = repo

    def cursor(self):
        return _Cursor()

    @contextmanager
    def transaction(self):
        saved = copy.deepcopy(self.repo.rows)
        initialized = self.repo.initialized
        try:
            yield
        except Exception:
            self.repo.rows = saved
            self.repo.initialized = initialized
            raise


class _FakeRepository:
    def __init__(self):
        self.rows = {"w1": {"id": "w1", "name": "ae"},
                     "w2": {"id": "w2", "name": "op"}}
        self.initialized = False
        self.upserts = []

    def ensure(self, cur):
        pass

    def lock_meta(self, cur):
        return self.initialized

    def set_initialized(self, cur):
        self.initialized = True

    def read_all(self, cur):
        return copy.deepcopy(self.rows)

    def upsert(self, cur, worker_id, row):
        self.upserts.append(worker_id)
        self.rows[worker_id] = copy.deepcopy(row)

    def delete(self, cur, worker_id):
        del self.rows[worker_id]


def test_postgres_transaction_writes_only_changed_worker_rows():
    repo = _FakeRepository()
    service = WorkerRegistryDB.__new__(WorkerRegistryDB)
    service.repo = repo
    service.db = _FakeClient(repo)
    service._schema_ready = True
    service._initialized = True

    with service.transaction(lambda: {}) as workers:
        workers["w1"]["name"] = "a-brain"
    assert repo.upserts == ["w1"]
    assert repo.rows["w2"]["name"] == "op"

    with pytest.raises(RuntimeError):
        with service.transaction(lambda: {}) as workers:
            workers["w1"]["name"] = "rolled back"
            raise RuntimeError("abort")
    assert repo.rows["w1"]["name"] == "a-brain"


def test_postgres_bootstrap_does_not_resurrect_deleted_workers():
    repo = _FakeRepository()
    repo.rows = {}
    client = _FakeClient(repo)
    service = WorkerRegistryDB.__new__(WorkerRegistryDB)
    service.repo = repo
    service.db = client
    service._schema_ready = False
    service._initialized = False

    seed = lambda: {"w1": {"id": "w1", "name": "ae"}}
    assert service.read(seed)["w1"]["name"] == "ae"
    with service.transaction(seed) as workers:
        workers.clear()
    assert service.read(seed) == {}

    another_process = WorkerRegistryDB.__new__(WorkerRegistryDB)
    another_process.repo = repo
    another_process.db = client
    another_process._schema_ready = False
    another_process._initialized = False
    assert another_process.read(seed) == {}
