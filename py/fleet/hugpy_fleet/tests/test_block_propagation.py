"""Worker side of block propagation: adoption of central's ``blocked_models``
reply, the slot-fill reconciler skip (log-once), the provisioning choke and
the reconcile loop. Converted from the monolith's script-style
``test_block_propagation.py`` sections [2]-[5]; section [1] (the heartbeat
route reply) stays with the server.
"""

from __future__ import annotations

import importlib
import time
import types

import pytest

A = importlib.import_module("hugpy_fleet.worker.agent")
agent_imports = importlib.import_module("hugpy_fleet.worker.imports")


@pytest.fixture(autouse=True)
def _reset_block_state():
    with A._BLOCKED_LOCK:
        A._BLOCKED_MODELS.clear()
        A._BLOCKED_LOGGED.clear()
    yield
    with A._BLOCKED_LOCK:
        A._BLOCKED_MODELS.clear()
        A._BLOCKED_LOGGED.clear()


def test_adopt_blocked_models_shapes():
    A._adopt_blocked_models({"blocked_models": ["m1", "m2"]})
    assert A._BLOCKED_MODELS == {"m1", "m2"}
    assert A._is_blocked_locally("m1") is True
    assert A._is_blocked_locally("m3") is False
    assert A._is_blocked_locally(None) is False and A._is_blocked_locally("") is False
    A._adopt_blocked_models({"blocked_models": []})
    assert A._BLOCKED_MODELS == set()
    A._adopt_blocked_models({"blocked_models": ["m1"]})
    A._adopt_blocked_models({})
    assert A._BLOCKED_MODELS == set()
    A._adopt_blocked_models({"blocked_models": ["m1"]})
    A._adopt_blocked_models({"limits": {}, "required_pkg_version": "0.1.191"})
    assert A._BLOCKED_MODELS == set()
    assert A._adopt_blocked_models(None) is None and A._BLOCKED_MODELS == set()
    A._adopt_blocked_models({"blocked_models": ["m1"]})
    A._adopt_blocked_models({"blocked_models": "not-a-list"})
    assert A._BLOCKED_MODELS == set()
    A._adopt_blocked_models({"blocked_models": ["m1", None, "", 123]})
    assert A._BLOCKED_MODELS == {"m1", "123"}


class _FillPoolBlock:
    def __init__(self, urls=None):
        pass

    def statuses(self):
        return [{"_control": "u1", "model_key": None}, {"_control": "u2", "model_key": None}]


def _skip_count(logged, mk):
    return sum(1 for a in logged if "skipping" in str(a) and mk in str(a))


def test_retired_slot_fill_is_a_strict_noop(monkeypatch):
    slots = importlib.import_module("hugpy_engine.serve.slots")
    dispatch = importlib.import_module("hugpy_engine.dispatch.dispatch")
    seated, logged = [], []
    monkeypatch.setattr(slots, "slots_enabled", lambda: True)
    monkeypatch.setattr(slots, "SlotPool", _FillPoolBlock)
    monkeypatch.setattr(A, "_models_local", lambda st: ["m-ok", "m-blocked"])
    monkeypatch.setattr(agent_imports, "get_model_config",
                        lambda mk, **kw: types.SimpleNamespace(framework="gguf"))
    monkeypatch.setattr(dispatch, "last_used_snapshot", lambda: {})
    monkeypatch.setattr(dispatch, "runner_for", lambda model_key=None, **kw: seated.append(model_key))
    monkeypatch.setattr(A.logger, "info", lambda *a, **kw: logged.append(a))
    monkeypatch.setattr(A, "_worker_fit_check", lambda mk: True)   # background seats must fit
    monkeypatch.setenv("HUGPY_SLOT_FILL", "1")   # background seating is operator opt-in (2026-09-10)

    A._adopt_blocked_models({"blocked_models": ["m-blocked"]})
    st = A.WorkerState(name="t", url=None, worker_id="w-fill-block")
    st.assigned_models = ["m-ok", "m-blocked"]
    A._fill_empty_slots(st)
    assert seated == []
    assert logged == []


def _wait_done(st, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with st._provision_lock:
            if not st._provisioning:
                return True
        time.sleep(0.02)
    return False


def test_kick_provision_is_the_single_choke(monkeypatch):
    provision = importlib.import_module("hugpy_storage.provision")
    pulled, logged = [], []
    monkeypatch.setattr(provision, "ensure_model_present",
                        lambda mk, url, progress=None, **kw: pulled.append(mk))
    monkeypatch.setattr(provision, "model_is_local", lambda mk: False)
    monkeypatch.setattr(provision, "ensure_model_registered", lambda mk, url: mk)
    monkeypatch.setattr(A, "restart_requested", lambda: False)
    monkeypatch.setattr(A.logger, "info", lambda *a, **kw: logged.append(a))

    A._adopt_blocked_models({"blocked_models": ["m-blocked-static"]})
    st = A.WorkerState(name="t", url=None, worker_id="w-kick-block")
    A._kick_provision(st, "m-blocked-static")
    assert "m-blocked-static" not in st._provisioning
    time.sleep(0.1)
    assert pulled == []
    assert sum(1 for a in logged if "m-blocked-static" in str(a)) == 1

    A._kick_provision(st, "m-ok-static")
    assert _wait_done(st)
    assert pulled == ["m-ok-static"]
    A._kick_provision(st, "m-blocked-static")
    assert sum(1 for a in logged if "m-blocked-static" in str(a)) == 1


def test_reconcile_loop_routes_through_kick_provision(monkeypatch):
    kicked = []
    monkeypatch.setattr(A, "_kick_provision",
                        lambda state, mk, purpose="reconcile": kicked.append(mk))
    monkeypatch.setattr(A.time, "sleep", lambda _s: None)
    flags = iter([False, True])
    monkeypatch.setattr(A, "restart_requested", lambda: next(flags, True))
    monkeypatch.setattr(A, "_models_local", lambda state: [])
    A._RUNTIME_SETTINGS.clear()
    A._RUNTIME_SETTINGS.update({"residency": {"m-blocked-static": "static", "m-ok-static": "static"}})
    try:
        A._adopt_blocked_models({"blocked_models": ["m-blocked-static"]})
        st = A.WorkerState(name="t", url=None, worker_id="w-reconcile-block")
        st.assigned_models = ["m-blocked-static", "m-ok-static"]
        A._reconcile_loop(st)
        assert "m-blocked-static" in kicked and "m-ok-static" in kicked
    finally:
        A._RUNTIME_SETTINGS.clear()
