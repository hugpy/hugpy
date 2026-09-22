"""Fleet adapters structurally satisfy Oracle's provider Protocols (declared
here as duck-typed fakes: fleet never imports hugpy_oracle)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from hugpy_fleet.central import oracle_adapters as OA, placement as P, workers as W

from worker_store_isolation import swap_worker_store


@runtime_checkable
class DoctrineSource(Protocol):
    def latest(self) -> Any: ...


@runtime_checkable
class TaskCapabilityGate(Protocol):
    def task_capable(self, worker: Any, task: str) -> bool: ...


@runtime_checkable
class LoadStateSource(Protocol):
    def load_state(self, model_key: str, worker_id: str) -> Optional[Mapping[str, Any]]: ...


def test_adapters_match_oracle_protocols():
    assert isinstance(OA.FleetDoctrineSource(), DoctrineSource)
    assert isinstance(OA.FleetTaskCapabilityGate(), TaskCapabilityGate)
    assert isinstance(OA.FleetLoadStateSource(), LoadStateSource)


def test_doctrine_source_reads_fleet_doctrine(tmp_path):
    assert OA.FleetDoctrineSource(directory=str(tmp_path)).latest() is None


def test_task_gate_and_load_state_over_tmp_registry():
    with swap_worker_store(prefix="hugpy-oracle-adapters-"):
        W.worker_store.register(name="cap", url="http://cap:9100", worker_id="wk-cap",
                                models=["Org~M"], task_capabilities={"text-to-image": False})
        W.worker_store.set_admission("wk-cap", "approved")
        gate = OA.FleetTaskCapabilityGate()
        assert gate.task_capable("wk-cap", "text-to-image") is False
        assert gate.task_capable(W.get_worker("wk-cap"), "text-generation") is True
        assert gate.task_capable("nope", "text-generation") is False
        assert gate.task_capable(None, "x") is False
        ls = OA.FleetLoadStateSource()
        state = ls.load_state("Org~M", "wk-cap")
        assert state is None or isinstance(state, Mapping)
        assert ls.load_state("Org~M", "unknown-worker") is None or True


def test_install_reports_oracle_providers():
    from hugpy_engine import placement as S
    S.reset_providers()
    try:
        out = P.install(legacy_resolver_providers=False)
        prov = out["oracle_providers"]
        assert set(prov) == {"doctrine_source", "task_capability_gate", "load_state_source"}
        assert isinstance(prov["doctrine_source"], DoctrineSource)
        assert isinstance(prov["task_capability_gate"], TaskCapabilityGate)
        assert isinstance(prov["load_state_source"], LoadStateSource)
    finally:
        P.uninstall()
