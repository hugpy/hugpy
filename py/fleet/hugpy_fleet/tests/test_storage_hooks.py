"""The worker installs the fleet's callbacks into hugpy_storage.providers at
boot (and slot children do the same through the fleet-owned child wrapper)."""

from __future__ import annotations

import pytest

from hugpy_storage import providers
from hugpy_fleet.central import evictions
from hugpy_fleet.worker import storage_hooks


@pytest.fixture(autouse=True)
def _reset():
    providers.reset_providers()
    yield
    providers.reset_providers()


def test_install_wires_gate_telemetry_and_registrar():
    out = storage_hooks.install_storage_providers()
    assert providers.get_budget_gate() is storage_hooks._budget_gate
    assert providers.get_transfer_telemetry() is evictions
    assert providers.get_executor_registrar() is storage_hooks._executor_registrar
    assert out["transfer_telemetry"] is evictions
    tele = providers.get_transfer_telemetry()
    for name in ("emit_provision_start", "emit_provision_fail", "emit_provision_done", "serve_scope"):
        assert callable(getattr(tele, name))


def test_budget_gate_delegates_to_evict_to_fit(monkeypatch):
    from hugpy_fleet.worker import budget
    seen = []
    monkeypatch.setattr(budget, "evict_to_fit", lambda state, mk, need: seen.append((state, mk, need)))
    storage_hooks.install_storage_providers()
    providers.get_budget_gate()("STATE", "Org~M", 123)
    assert seen == [("STATE", "Org~M", 123)]


def test_registrar_reaches_agent(monkeypatch):
    from hugpy_fleet.worker import agent as A
    seen = []
    monkeypatch.setattr(A, "register_executor", lambda ex: seen.append(ex))
    storage_hooks.install_storage_providers()
    providers.get_executor_registrar()("POOL")
    assert seen == ["POOL"]


def test_slot_child_installs_without_registrar(monkeypatch):
    import hugpy_engine.serve.slot_agent as SA
    from hugpy_fleet.worker import slot_child
    monkeypatch.setattr(SA, "main", lambda: 0)
    assert slot_child.main() == 0
    assert providers.get_budget_gate() is storage_hooks._budget_gate
    assert providers.get_transfer_telemetry() is evictions
    # the child has no agent restart path: registrar left at the default no-op
    assert providers.get_executor_registrar() is not storage_hooks._executor_registrar


def test_worker_main_installs_at_boot(monkeypatch):
    """main() installs the providers before argument parsing (so --help proves it)."""
    from hugpy_fleet.worker import agent as A
    seen = []
    monkeypatch.setattr(storage_hooks, "install_storage_providers",
                        lambda **kw: seen.append("installed"))
    with pytest.raises(SystemExit) as exc:
        A.main(["--help"])
    assert exc.value.code == 0
    assert seen == ["installed"]


def test_supervisor_spawns_fleet_slot_child():
    import inspect
    from hugpy_fleet.worker import agent as A
    src = inspect.getsource(A._supervise_slots)
    assert 'module = "hugpy_fleet.worker.slot_child"' in src
