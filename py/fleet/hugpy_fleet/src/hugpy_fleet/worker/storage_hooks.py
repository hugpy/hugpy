"""Install the fleet's callbacks into ``hugpy_storage.providers``.

Storage sits below the fleet, yet a transfer (``ensure_model_present``)
needs three things only the worker holds: the storage-budget gate
(``worker.budget.evict_to_fit``), provisioning telemetry
(``central.evictions``) and the executor registrar
(``worker.agent.register_executor``, so a restart can drain transfer
pools). The worker agent calls :func:`install_storage_providers` at boot and
every slot child does so through :mod:`hugpy_fleet.worker.slot_child`,
because both processes call ``ensure_model_present``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

__all__ = ["install_storage_providers", "uninstall_storage_providers"]

log = logging.getLogger("hugpy_fleet.worker.storage_hooks")


def _budget_gate(state: Any, model_key: str, need_bytes: int) -> None:
    """Evict cold models to seat the pull, or raise ``BudgetRefusal``. Without
    a worker state (a slot child, a bare CLI) the plan has no live inputs and
    the pull is admitted — ``evict_to_fit`` is best-effort by construction."""
    from hugpy_fleet.worker.budget import evict_to_fit
    return evict_to_fit(state, model_key, need_bytes)


def _executor_registrar(pool: Any) -> None:
    from hugpy_fleet.worker.agent import register_executor
    register_executor(pool)


def install_storage_providers(*, with_registrar: bool = True) -> Dict[str, Any]:
    """Wire budget gate, transfer telemetry and (optionally) the executor
    registrar into storage. Idempotent. Returns what was installed."""
    from hugpy_storage import providers
    from hugpy_fleet.central import evictions

    providers.set_budget_gate(_budget_gate)
    providers.set_transfer_telemetry(evictions)
    if with_registrar:
        providers.set_executor_registrar(_executor_registrar)
    # Storage's catalog seam defaults to NullCatalogSource (knows no models):
    # without the engine bridge every scan row is not_local/no_config, every
    # catalog_register "does not stick", and calls die "unknown model_key".
    try:
        from hugpy_engine import catalog_bridge
        catalog_bridge.install()
    except Exception:  # noqa: BLE001
        log.warning("catalog bridge not installed; storage will see no models", exc_info=True)
    out = {
        "budget_gate": providers.get_budget_gate(),
        "transfer_telemetry": providers.get_transfer_telemetry(),
        "executor_registrar": providers.get_executor_registrar() if with_registrar else None,
    }
    log.info("storage providers installed (budget gate, telemetry%s)",
             ", executor registrar" if with_registrar else "")
    return out


def uninstall_storage_providers() -> None:
    from hugpy_storage import providers
    providers.set_budget_gate(None)
    providers.set_transfer_telemetry(None)
    providers.set_executor_registrar(None)
