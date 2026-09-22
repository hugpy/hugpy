"""Worker side of the boot-load star: boot-once adoption (load on the first
reply carrying a star, never re-warm after eviction, latch cleared only by a
restart) and disk-vs-fetch (fetch only when the star is absent on disk).
Converted from the monolith's script-style ``test_worker_boot_prewarm.py``
sections [3]-[4]; the flag-store CRUD and routes stay with engine/server.
"""

from __future__ import annotations

import importlib
import time
import types

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
agent_imports = importlib.import_module("hugpy_fleet.worker.imports")
provision = importlib.import_module("hugpy_storage.provision")
slots = importlib.import_module("hugpy_engine.serve.slots")
dispatch = importlib.import_module("hugpy_engine.dispatch.dispatch")


def _reset_latch():
    with A._BOOT_PREWARM_LOCK:
        A._BOOT_PREWARM_DONE.clear()


def _wait(pred, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture(autouse=True)
def _latch():
    _reset_latch()
    yield
    _reset_latch()


@pytest.fixture
def stubbed(monkeypatch):
    loaded, resident = [], set()

    def _runner_for(model_key=None, **kw):
        loaded.append(model_key)
        resident.add(model_key)
        return types.SimpleNamespace(ensure_loaded=lambda: None)

    monkeypatch.setattr(provision, "ensure_model_registered", lambda mk, url=None, **kw: mk)
    monkeypatch.setattr(provision, "model_is_local", lambda mk: True)
    monkeypatch.setattr(provision, "ensure_model_present", lambda *a, **kw: loaded.append(("pull", a[0])))
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: set())
    monkeypatch.setattr(A, "loaded_model_keys", lambda: sorted(resident))
    monkeypatch.setattr(dispatch, "runner_for", _runner_for)
    monkeypatch.setattr(agent_imports, "get_model_config",
                        lambda mk, **kw: types.SimpleNamespace(framework="gguf"))
    monkeypatch.setattr(slots, "slots_enabled", lambda: False)
    return loaded, resident


def test_boot_once_adoption(stubbed, monkeypatch):
    loaded, resident = stubbed
    st = A.WorkerState(name="t", url=None, worker_id="w-prewarm")

    A._adopt_boot_prewarm(st, {"limits": {}, "required_pkg_version": "0.1.200"})
    time.sleep(0.15)
    assert loaded == []
    A._adopt_boot_prewarm(st, None)
    assert loaded == []

    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    assert _wait(lambda: loaded == ["M~star"])
    assert "M~star" in resident and "M~star" in A._BOOT_PREWARM_DONE

    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    time.sleep(0.2)
    assert loaded == []

    resident.discard("M~star")            # eviction: boot-once means no reload
    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    time.sleep(0.2)
    assert loaded == [] and "M~star" not in resident

    _reset_latch()                        # a restart re-arms the star
    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    assert _wait(lambda: loaded == ["M~star"])

    _reset_latch()
    resident.clear()
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: {"M~slotstar"})
    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~slotstar"})
    time.sleep(0.15)
    assert loaded == [] and "M~slotstar" in A._BOOT_PREWARM_DONE
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: set())

    _reset_latch()
    loaded.clear()
    resident.clear()

    def _boom(*a, **kw):
        raise RuntimeError("model absent")
    monkeypatch.setattr(provision, "model_is_local", lambda mk: False)
    monkeypatch.setattr(provision, "ensure_model_present", _boom)
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~missing"})
    assert _wait(lambda: "M~missing" in A._BOOT_PREWARM_DONE)
    assert "M~missing" not in resident
    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~missing"})
    time.sleep(0.15)
    assert loaded == []

    monkeypatch.setattr(provision, "model_is_local", lambda mk: True)
    monkeypatch.setattr(provision, "ensure_model_present", lambda *a, **kw: loaded.append(("pull", a[0])))
    _reset_latch()
    loaded.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": 123})
    time.sleep(0.1)
    assert loaded == []
    A._adopt_boot_prewarm(st, {"boot_prewarm": ""})
    time.sleep(0.1)
    assert loaded == []


def test_boot_load_fetches_only_on_disk_absence(monkeypatch):
    disk, resident, fetches, warms = set(), set(), [], []

    def _fetch(mk, *a, **kw):
        fetches.append(mk)
        disk.add(mk)

    def _warm(model_key=None, **kw):
        warms.append(model_key)
        resident.add(model_key)
        return types.SimpleNamespace(ensure_loaded=lambda: None)

    monkeypatch.setattr(provision, "ensure_model_registered", lambda mk, url=None, **kw: mk)
    monkeypatch.setattr(provision, "model_is_local", lambda mk: mk in disk)
    monkeypatch.setattr(provision, "ensure_model_present", _fetch)
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: set())
    monkeypatch.setattr(A, "loaded_model_keys", lambda: sorted(resident))
    monkeypatch.setattr(dispatch, "runner_for", _warm)
    monkeypatch.setattr(agent_imports, "get_model_config",
                        lambda mk, **kw: types.SimpleNamespace(framework="gguf"))
    monkeypatch.setattr(slots, "slots_enabled", lambda: False)
    st = A.WorkerState(name="t", url=None, worker_id="w-reentry")

    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    assert _wait(lambda: fetches == ["M~star"])
    assert _wait(lambda: warms == ["M~star"])
    assert "M~star" in disk and "M~star" in resident

    resident.discard("M~star")
    _reset_latch()
    fetches.clear(); warms.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    assert _wait(lambda: warms == ["M~star"])
    assert fetches == []
    assert "M~star" in resident

    resident.discard("M~star")
    disk.discard("M~star")
    _reset_latch()
    fetches.clear(); warms.clear()
    A._adopt_boot_prewarm(st, {"boot_prewarm": "M~star"})
    assert _wait(lambda: fetches == ["M~star"])
