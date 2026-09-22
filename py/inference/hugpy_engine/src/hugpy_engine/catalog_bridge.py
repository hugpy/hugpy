"""Catalog bridge: install the engine's registry as the storage catalog source.

``hugpy_storage`` moves bytes for registry rows but must never import the
engine, so it reads the registry through ``hugpy_storage.catalog_source``
(a Protocol with a null default) and redirects loads to the box-local hot
cache through ``hugpy_storage.providers.set_serve_path_hook``. Storage, in
turn, publishes ``hugpy_control.bus.TOPIC_CATALOG_CHANGED`` after a download /
wipe / promote. :func:`install` wires all three:

    * :class:`EngineCatalogSource` -> ``set_catalog_source`` (rows, get,
      canonical_key, resolve_dir, register, refresh over
      ``config.models.models_config`` / ``config.main`` / ``assure_model_key``);
    * ``serve.hot_cache.use`` -> ``set_serve_path_hook``;
    * a ``catalog.changed`` subscriber that refreshes discovery (coalesced,
      on a daemon thread) so the registry follows the physical inventory.

``install`` is idempotent and best-effort: a missing storage/control seam is
tolerated (logged at debug) so single-box use never depends on it. The facade
(``backends.local.LocalBackend``) calls it lazily on first use.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["EngineCatalogSource", "install", "uninstall", "installed", "refresh_from_event"]


class EngineCatalogSource:
    """``hugpy_storage.catalog_source.CatalogSource`` over the live registry."""

    name = "hugpy_engine"

    def rows(self) -> dict[str, dict]:
        from hugpy_engine.config.models.models_config import MODEL_REGISTRY_DICT

        return {k: dict(v) for k, v in MODEL_REGISTRY_DICT.items()}

    def get(self, model_key: str) -> Optional[Any]:
        from hugpy_engine.config.main import get_model_config

        key = self.canonical_key(model_key) or model_key
        try:
            return get_model_config(key)
        except KeyError:
            return None

    def canonical_key(self, model_key: str) -> Optional[str]:
        from hugpy_engine.resolvers.assure_model_key import assure_model_key

        return assure_model_key(model_key)

    def resolve_dir(self, model_key: str) -> Optional[str]:
        from hugpy_engine.config.main import get_model_path

        key = self.canonical_key(model_key)
        if key is None:
            return None
        try:
            return get_model_path(key)
        except KeyError:
            return None

    def register(self, model_key: str, row: dict) -> bool:
        from hugpy_engine.config.models.models_config import (
            MODEL_REGISTRY,
            MODEL_REGISTRY_DICT,
            derive_model_config_row,
            update_model_config_dict,
        )

        if not model_key or not isinstance(row, dict):
            return False
        derived, why = derive_model_config_row(model_key, dict(row))
        if not derived:
            logger.warning("catalog register(%s) refused: %s", model_key, why)
            return False
        cfgs = update_model_config_dict(model_key, derived, {})
        if model_key not in cfgs:
            return False
        MODEL_REGISTRY[model_key] = cfgs[model_key]
        MODEL_REGISTRY_DICT[model_key] = dict(derived)
        try:
            from hugpy_engine.config.models.models_default import refresh_task_registries

            refresh_task_registries()
        except Exception:  # noqa: BLE001 - task defaults are advisory
            logger.debug("refresh_task_registries after register failed", exc_info=True)
        return True

    def refresh(self) -> None:
        from hugpy_engine.config.models.models_config import refresh_registry

        refresh_registry(run_discovery=True)


_lock = threading.Lock()
_installed = False
_subscription: Any = None
_listener: Optional[threading.Thread] = None
_pending = threading.Event()
_source = EngineCatalogSource()


def installed() -> bool:
    return _installed


def refresh_from_event(message: Any = None) -> None:
    """Refresh discovery after a ``catalog.changed`` event (never raises)."""
    try:
        payload = getattr(message, "payload", None) or {}
        logger.info("catalog changed (%s %s): refreshing discovery",
                    payload.get("change") or "changed", payload.get("model_key") or "")
        _source.refresh()
    except Exception:  # noqa: BLE001 - a refresh must never kill the listener
        logger.warning("catalog refresh after change event failed", exc_info=True)


def _listen(sub: Any) -> None:
    # Coalesce bursts: drain everything queued, refresh once.
    while True:
        try:
            msg = sub.get(timeout=1.0)
        except Exception:  # noqa: BLE001
            if getattr(sub, "closed", False):
                return
            continue
        if msg is None:
            if getattr(sub, "closed", False):
                return
            continue
        latest = msg
        while True:
            try:
                nxt = sub.get(timeout=0.0)
            except Exception:  # noqa: BLE001
                nxt = None
            if nxt is None:
                break
            latest = nxt
        refresh_from_event(latest)


def install(*, subscribe: bool = True) -> bool:
    """Wire the engine into storage/control. Idempotent; returns True when the
    catalog source was installed."""
    global _installed, _subscription, _listener
    with _lock:
        if _installed:
            return True
        ok = False
        try:
            from hugpy_storage.catalog_source import set_catalog_source

            set_catalog_source(_source)
            ok = True
        except Exception:  # noqa: BLE001 - storage seam absent/old
            logger.debug("hugpy_storage.catalog_source unavailable; registry not shared", exc_info=True)
        try:
            from hugpy_storage.providers import set_serve_path_hook

            def _serve_path(path: str) -> str:
                from hugpy_engine.serve.hot_cache import use

                return use(path)

            set_serve_path_hook(_serve_path)
        except Exception:  # noqa: BLE001
            logger.debug("hugpy_storage.providers unavailable; no hot-cache path hook", exc_info=True)
        if subscribe:
            try:
                from hugpy_control.bus import TOPIC_CATALOG_CHANGED, bus

                _subscription = bus.subscribe(TOPIC_CATALOG_CHANGED)
                _listener = threading.Thread(target=_listen, args=(_subscription,),
                                             name="hugpy-engine-catalog-refresh", daemon=True)
                _listener.start()
            except Exception:  # noqa: BLE001
                logger.debug("hugpy_control.bus unavailable; no catalog.changed subscription", exc_info=True)
        _installed = True
        return ok


def uninstall() -> None:
    """Undo :func:`install` (tests)."""
    global _installed, _subscription, _listener
    with _lock:
        if not _installed:
            return
        try:
            from hugpy_storage.catalog_source import reset_catalog_source

            reset_catalog_source()
        except Exception:  # noqa: BLE001
            pass
        try:
            from hugpy_storage.providers import set_serve_path_hook

            set_serve_path_hook(None)
        except Exception:  # noqa: BLE001
            pass
        if _subscription is not None:
            try:
                _subscription.close()
            except Exception:  # noqa: BLE001
                pass
        _subscription = None
        _listener = None
        _installed = False
