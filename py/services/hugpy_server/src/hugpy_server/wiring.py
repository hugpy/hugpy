"""Composition wiring for the hugpy server process.

``install_all()`` performs every row of ``py/WIRING.md`` marked "server": it
hands the fleet's implementations to the engine/oracle/curation seams, loads
the media/video task plugins, bridges the storage catalog into the engine,
installs the oracle's video hooks, the storage footprint selector and the
HF-token listener, and wires the control bus. Each step is isolated — a
failure is logged and recorded, and the next step still runs — so the app
always boots; the returned report says exactly what was and was not wired.

Call it once per process from the app factory (``wsgi_app.get_hugpy_flask``)
or from ``main()``. Every underlying installer is idempotent, so re-entrant
app creation (tests, embedding hosts) is harmless.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# The order matters only for the oracle providers (they come out of the
# placement install) and for curation (it reuses the fleet doctrine source).
STEPS = (
    "placement",
    "oracle_providers",
    "task_plugins",
    "catalog_bridge",
    "oracle_hooks",
    "curation_providers",
    "footprint_selector",
    "hf_token_listener",
    "comms_bus",
    "eviction_store_sink",
)

_last_report: dict[str, Any] = {}


# ── individual installers (each returns a JSON-friendly detail) ───────────────

def install_placement() -> dict[str, Any]:
    """WorkerRegistry/Transport/EvictionLedger/Blocklist/ModelMetrics/PriorityGroups
    (``hugpy_engine.placement``) implemented by ``hugpy_fleet.central``."""
    from hugpy_fleet.central import placement
    installed = placement.install()
    return {k: type(v).__name__ for k, v in installed.items() if k != "oracle_providers"}


def install_oracle_providers(oracle_providers: Optional[dict] = None) -> dict[str, Any]:
    """Fleet facts the oracle consumes: doctrine, task capability gate, load state."""
    from hugpy_oracle import providers as op
    if oracle_providers is None:
        from hugpy_fleet.central.oracle_adapters import oracle_providers as _factory
        oracle_providers = _factory()
    setters: dict[str, Callable] = {
        "doctrine_source": op.set_doctrine_source,
        "task_capability_gate": op.set_task_capability_gate,
        "load_state_source": op.set_load_state_source,
    }
    out: dict[str, Any] = {}
    for key, setter in setters.items():
        impl = oracle_providers.get(key)
        if impl is None:
            out[key] = None
            continue
        setter(impl)
        out[key] = type(impl).__name__
    return out


def install_task_plugins() -> dict[str, Any]:
    """Media/video runners into the engine task table: the entry-point group
    first; explicit ``register()`` calls as a belt-and-braces fallback for
    checkouts where entry points are not installed."""
    from hugpy_engine import tasks
    tasks.ensure_plugins_loaded()
    out: dict[str, Any] = {"entry_points": True}
    for name in ("hugpy_media.plugin", "hugpy_video.plugin"):
        try:
            import importlib
            mod = importlib.import_module(name)
            reg = getattr(mod, "register", None)
            regs = reg() if callable(reg) else None
            out[name] = ([getattr(r, "task", None) or str(r) for r in regs]
                         if regs is not None else None)
        except Exception as exc:  # noqa: BLE001 — one optional stack must not block the other
            out[name] = f"error: {type(exc).__name__}: {exc}"
    out["registered_tasks"] = sorted(tasks.registered_tasks())
    return out


def install_catalog_bridge() -> dict[str, Any]:
    from hugpy_engine import catalog_bridge
    ok = catalog_bridge.install()
    return {"installed": bool(catalog_bridge.installed()), "result": bool(ok)}


def install_oracle_hooks() -> dict[str, Any]:
    from hugpy_oracle import install_hooks
    return dict(install_hooks())


def install_curation_providers(doctrine_source: Any = None) -> dict[str, Any]:
    from hugpy_curation import install_providers
    if doctrine_source is None:
        try:
            from hugpy_oracle import providers as op
            doctrine_source = op.get_doctrine_source()
        except Exception:  # noqa: BLE001
            doctrine_source = None
    return dict(install_providers(doctrine_source=doctrine_source))


def install_footprint_selector() -> dict[str, Any]:
    from hugpy_storage import providers as sp
    from hugpy_storage.format_select import effective_bytes
    sp.set_footprint_selector(effective_bytes)
    return {"selector": f"{effective_bytes.__module__}.{effective_bytes.__name__}"}


def _on_hf_token_change(token: Optional[str]) -> None:
    from hugpy_server.app.functions.imports.utils import constants as _c
    _c.rebuild_hf_api(token)


def install_hf_token_listener() -> dict[str, Any]:
    from hugpy_storage import hf_token
    hf_token.add_token_listener(_on_hf_token_change)
    return {"listener": f"{__name__}._on_hf_token_change"}


def install_comms_bus(source: str = "central") -> dict[str, Any]:
    """control.cancel reaches the shared job store; job and settings lifecycle
    transitions publish back onto the bus. Idempotent per (bus, store)."""
    from hugpy_control.bus import wire_cancel, wire_job_events
    from hugpy_control.settings import wire_settings_events
    wire_cancel()
    wire_job_events(source=source)
    wire_settings_events(source=source)
    return {"cancel": True, "job_events": source, "settings_events": source}


def install_eviction_store_sink() -> dict[str, Any]:
    """Central's own eviction events land in the shared store (same console
    stream as the fleet's relayed events)."""
    from hugpy_fleet.central import evictions
    evictions.install_store_sink()
    return {"installed": True}


# ── orchestration ─────────────────────────────────────────────────────────────

def _run(report: dict[str, Any], name: str, fn: Callable[[], Any]) -> Any:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 — isolate every step
        logger.warning("wiring: %s failed: %s: %s", name, type(exc).__name__, exc,
                       exc_info=True)
        report[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["errors"].append(name)
        return None
    report[name] = {"ok": True, "detail": detail}
    return detail


def install_all(app: Any = None, *, source: str = "central") -> dict[str, Any]:
    """Wire every server-side seam. Returns a report keyed by step name
    (``STEPS``) plus ``ok`` (all steps succeeded) and ``errors`` (failed step
    names). When ``app`` is a Flask app the report is also stored at
    ``app.extensions["hugpy_wiring"]``."""
    report: dict[str, Any] = {"errors": []}

    placement_installed: dict = {}

    def _placement():
        from hugpy_fleet.central import placement
        nonlocal placement_installed
        placement_installed = placement.install()
        return {k: type(v).__name__ for k, v in placement_installed.items()
                if k != "oracle_providers"}

    _run(report, "placement", _placement)
    _run(report, "oracle_providers",
         lambda: install_oracle_providers(placement_installed.get("oracle_providers")))
    _run(report, "task_plugins", install_task_plugins)
    _run(report, "catalog_bridge", install_catalog_bridge)
    _run(report, "oracle_hooks", install_oracle_hooks)
    _run(report, "curation_providers", install_curation_providers)
    _run(report, "footprint_selector", install_footprint_selector)
    _run(report, "hf_token_listener", install_hf_token_listener)
    _run(report, "comms_bus", lambda: install_comms_bus(source=source))
    _run(report, "eviction_store_sink", install_eviction_store_sink)

    report["ok"] = not report["errors"]
    if report["errors"]:
        logger.error("wiring: %d step(s) failed: %s", len(report["errors"]),
                     ", ".join(report["errors"]))
    else:
        logger.info("wiring: all %d steps installed", len(STEPS))
    _last_report.clear()
    _last_report.update(report)
    if app is not None:
        try:
            app.extensions["hugpy_wiring"] = report
        except Exception:  # noqa: BLE001
            pass
    return report


def last_report() -> dict[str, Any]:
    """The report from the most recent ``install_all()`` in this process."""
    return dict(_last_report)


__all__ = [
    "STEPS", "install_all", "last_report",
    "install_placement", "install_oracle_providers", "install_task_plugins",
    "install_catalog_bridge", "install_oracle_hooks", "install_curation_providers",
    "install_footprint_selector", "install_hf_token_listener", "install_comms_bus",
    "install_eviction_store_sink",
]
