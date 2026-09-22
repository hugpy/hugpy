"""Fleet data in the shapes ``hugpy_oracle.providers`` expects.

Oracle sits above the fleet and declares Protocols for the fleet facts it
needs (``DoctrineSource``, ``TaskCapabilityGate``, ``LoadStateSource``).
Fleet may not import Oracle, so these adapters match those Protocols
*structurally*; the server passes them to ``hugpy_oracle.providers.set_*``.
``hugpy_fleet.central.placement.install()`` returns them under
``"oracle_providers"``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "FleetDoctrineSource",
    "FleetTaskCapabilityGate",
    "FleetLoadStateSource",
    "oracle_providers",
]


class FleetDoctrineSource:
    """``latest() -> Doctrine | None`` over ``hugpy_fleet.doctrine.latest``."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = directory

    def latest(self) -> Any:
        from hugpy_fleet.doctrine import doctrine
        try:
            return doctrine.latest(self.directory)
        except Exception:  # noqa: BLE001 - an unreadable doctrine is "no doctrine"
            return None


class FleetTaskCapabilityGate:
    """``task_capable(worker, task) -> bool`` over ``central.workers._task_capable``.
    ``worker`` may be a registry row or a worker id."""

    def task_capable(self, worker: Any, task: Optional[str]) -> bool:
        from hugpy_fleet.central import workers as W
        row = worker
        if isinstance(worker, str):
            row = W.get_worker(worker) or W.lookup_worker(worker)
        if not isinstance(row, Mapping):
            return False
        try:
            return bool(W._task_capable(dict(row), task))
        except Exception:  # noqa: BLE001 - never raise into Oracle
            return False


class FleetLoadStateSource:
    """``load_state(model_key, worker_id) -> Mapping | None`` over
    ``central.workers.load_state_for_model``."""

    def load_state(self, model_key: str, worker_id: str, since_ts: float = 0.0) -> Optional[Mapping[str, Any]]:
        from hugpy_fleet.central import workers as W
        try:
            return W.load_state_for_model(model_key, worker_id, since_ts)
        except Exception:  # noqa: BLE001
            return None


def oracle_providers() -> Dict[str, Any]:
    """The three adapters keyed by the ``hugpy_oracle.providers.set_<key>`` name."""
    return {
        "doctrine_source": FleetDoctrineSource(),
        "task_capability_gate": FleetTaskCapabilityGate(),
        "load_state_source": FleetLoadStateSource(),
    }
