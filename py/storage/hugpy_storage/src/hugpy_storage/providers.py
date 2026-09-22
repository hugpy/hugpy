"""Injected callbacks from the layers ABOVE storage (fleet, engine, server).

Storage sits under the engine and the fleet, yet a few transfer decisions
need something only they hold. Each is a small callable with a SAFE default,
registered by the upper package at composition time; storage never imports
the package that provides it. Naming follows the guide: ``set_<thing>`` /
``get_<thing>``.

    budget gate           fleet.worker.budget.evict_to_fit
        ``gate(state, model_key, need_bytes)`` — evict cold models to seat a
        pull of ``need_bytes``, or raise the fleet's BudgetRefusal. Default:
        admit (log only). The worker agent installs the real gate at boot and
        in every slot child, or its transfers run ungated.

    transfer telemetry    fleet.central.evictions (emit_provision_*, serve_scope)
        An object with ``emit_provision_start / fail / done`` and
        ``serve_scope``; every method optional. Default: None (no events).

    executor registrar    fleet.worker.agent.register_executor
        ``fn(pool)`` — the agent tracks transfer thread pools so a restart can
        shut them down first. Default: no-op.

    serve path            engine.serve.hot_cache.use
        ``fn(path) -> path`` — after a model resolves on shared storage the
        engine may redirect a load to its box-local hot-cache copy. Default:
        identity.

    footprint selector    server format_select.effective_bytes
        ``fn(listing, framework) -> int`` — the single-format effective size of
        a model dir from its ``[(relpath, size)]`` listing. Default: the sum of
        the listing (the whole-snapshot size).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("hugpy_storage.providers")

BudgetGate = Callable[[Any, str, int], Any]
Registrar = Callable[[Any], Any]
PathHook = Callable[[str], str]
Footprint = Callable[[Iterable[tuple], Optional[str]], Optional[int]]


# ── budget gate ──────────────────────────────────────────────────────────────
def _admit_all(state, model_key: str, need_bytes: int) -> None:
    logger.info("budget: no gate installed — admitting %s (%s bytes) ungated",
                model_key, need_bytes)


_BUDGET_GATE: BudgetGate = _admit_all


def set_budget_gate(gate: Optional[BudgetGate]) -> None:
    global _BUDGET_GATE
    _BUDGET_GATE = gate if gate is not None else _admit_all


def get_budget_gate() -> BudgetGate:
    return _BUDGET_GATE


# ── transfer telemetry ───────────────────────────────────────────────────────
_TELEMETRY: Any = None


def set_transfer_telemetry(sink: Any) -> None:
    global _TELEMETRY
    _TELEMETRY = sink


def get_transfer_telemetry() -> Any:
    return _TELEMETRY


# ── executor registrar ───────────────────────────────────────────────────────
_REGISTRAR: Registrar = lambda pool: None  # noqa: E731


def set_executor_registrar(fn: Optional[Registrar]) -> None:
    global _REGISTRAR
    _REGISTRAR = fn if fn is not None else (lambda pool: None)


def get_executor_registrar() -> Registrar:
    return _REGISTRAR


# ── serve path (hot cache) ───────────────────────────────────────────────────
_SERVE_PATH: PathHook = lambda path: path  # noqa: E731


def set_serve_path_hook(fn: Optional[PathHook]) -> None:
    global _SERVE_PATH
    _SERVE_PATH = fn if fn is not None else (lambda path: path)


def get_serve_path_hook() -> PathHook:
    return _SERVE_PATH


def serve_path(path: str) -> str:
    """Apply the hook; a failing hook returns the path unchanged."""
    try:
        return _SERVE_PATH(path) or path
    except Exception:  # noqa: BLE001 — never turn a resolved model into an error
        logger.debug("serve path hook failed for %s", path, exc_info=True)
        return path


# ── footprint selector ───────────────────────────────────────────────────────
def _sum_listing(listing: Iterable[tuple], framework: Optional[str] = None) -> Optional[int]:
    total = 0
    for _rel, size in listing:
        try:
            total += int(size or 0)
        except (TypeError, ValueError):
            continue
    return total


_FOOTPRINT: Footprint = _sum_listing


def set_footprint_selector(fn: Optional[Footprint]) -> None:
    global _FOOTPRINT
    _FOOTPRINT = fn if fn is not None else _sum_listing


def get_footprint_selector() -> Footprint:
    return _FOOTPRINT


def reset_providers() -> None:
    """Every hook back to its default (tests)."""
    set_budget_gate(None)
    set_transfer_telemetry(None)
    set_executor_registrar(None)
    set_serve_path_hook(None)
    set_footprint_selector(None)


__all__ = [
    "set_budget_gate", "get_budget_gate",
    "set_transfer_telemetry", "get_transfer_telemetry",
    "set_executor_registrar", "get_executor_registrar",
    "set_serve_path_hook", "get_serve_path_hook", "serve_path",
    "set_footprint_selector", "get_footprint_selector",
    "reset_providers",
]
