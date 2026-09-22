"""The catalog seam — how storage asks "which models exist" without the engine.

The model REGISTRY (keys, hub ids, frameworks, tasks, pinned filenames, the
folder each row was discovered under) is engine-owned state. Storage moves
bytes for those rows but must never import ``hugpy_engine`` (PARTITION.md:
engine depends on storage, not the reverse). So storage talks to the registry
through this Protocol, and the composition root (``hugpy_server.wsgi_app``,
the worker agent, or ``hugpy-engine``'s own bootstrap) installs the real
implementation with :func:`set_catalog_source` at start-up.

The default is :class:`NullCatalogSource`: it knows NO models. Every storage
code path degrades honestly against it — a transfer still lands under the
flat layout (``model_paths.route_destination`` is pure path arithmetic), a
presence probe answers from the on-disk truth via ``model_presence``, and a
registry mutation reports ``False`` instead of raising. That is what lets
``hugpy-storage daemon`` and ``model_sync`` run on a box with no engine.

Protocol (shaped from what provision / download_models / the console helpers
actually call):

    rows()                      every registry row as a plain dict, keyed by
                                model_key  ({key: {hub_id, framework, ...}})
    get(model_key)              one row as an attribute-style config (the
                                engine's ModelConfig) or None
    canonical_key(model_key)    the registry key a key / hub_id / suffix
                                resolves to, or None (engine: assure_model_key)
    resolve_dir(model_key)      the on-disk dir the engine would load from
                                (env override, folder, read-through), or None
    register(model_key, row)    insert a central-provided row into the live
                                registry; True when it stuck
    refresh()                   re-derive the registry from the discovery
                                report after the physical inventory changed

Every method is optional in spirit: an implementation may raise
``NotImplementedError`` for what it does not support and storage treats that
exactly like the null answer.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("hugpy_storage.catalog_source")


@runtime_checkable
class CatalogSource(Protocol):
    """What storage needs to know about the model registry. See module doc."""

    def rows(self) -> dict[str, dict]:
        ...

    def get(self, model_key: str) -> Optional[Any]:
        ...

    def canonical_key(self, model_key: str) -> Optional[str]:
        ...

    def resolve_dir(self, model_key: str) -> Optional[str]:
        ...

    def register(self, model_key: str, row: dict) -> bool:
        ...

    def refresh(self) -> None:
        ...


class NullCatalogSource:
    """The safe default: an empty registry. Never raises."""

    name = "null"

    def rows(self) -> dict[str, dict]:
        return {}

    def get(self, model_key: str) -> Optional[Any]:
        return None

    def canonical_key(self, model_key: str) -> Optional[str]:
        return None

    def resolve_dir(self, model_key: str) -> Optional[str]:
        return None

    def register(self, model_key: str, row: dict) -> bool:
        logger.debug("no catalog source installed; cannot register %s", model_key)
        return False

    def refresh(self) -> None:
        return None


class DictCatalogSource:
    """An in-memory catalog over a ``{model_key: row_dict}`` mapping.

    Useful for tests, for ``model_sync`` runs seeded from central's model list,
    and as a reference implementation of the Protocol. Rows are exposed both
    as dicts (:meth:`rows`) and as attribute-style objects (:meth:`get`), the
    two shapes storage code reads. ``resolve_dir`` resolves through the
    on-disk layout (flat + legacy) with storage's own completeness rule.
    """

    name = "dict"

    def __init__(self, rows: Optional[dict[str, dict]] = None,
                 root: Optional[str] = None) -> None:
        self._rows: dict[str, dict] = {k: dict(v) for k, v in (rows or {}).items()}
        self._root = root
        self.refreshed = 0

    def rows(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self._rows.items()}

    def get(self, model_key: str) -> Optional[Any]:
        key = self.canonical_key(model_key)
        if key is None:
            return None
        from types import SimpleNamespace
        row = dict(self._rows[key])
        row.setdefault("model_key", key)
        return SimpleNamespace(**row)

    def canonical_key(self, model_key: str) -> Optional[str]:
        if not model_key:
            return None
        if model_key in self._rows:
            return model_key
        want = str(model_key).strip("/").lower()
        for key, row in self._rows.items():
            hub = str(row.get("hub_id") or "").strip("/").lower()
            if want == hub or want == hub.rsplit("/", 1)[-1] or want == key.lower():
                return key
        return None

    def resolve_dir(self, model_key: str) -> Optional[str]:
        key = self.canonical_key(model_key)
        if key is None:
            return None
        from hugpy_storage.model_paths import resolve_model_dir
        row = self._rows[key]
        kwargs = {"root": self._root} if self._root else {}
        return resolve_model_dir(dict(row), require_complete=False, **kwargs)

    def register(self, model_key: str, row: dict) -> bool:
        if not model_key or not isinstance(row, dict):
            return False
        self._rows[model_key] = dict(row)
        return True

    def refresh(self) -> None:
        self.refreshed += 1


_DEFAULT = NullCatalogSource()
_SOURCE: Any = _DEFAULT
_LOCK = threading.Lock()


def set_catalog_source(source: Optional[Any]) -> None:
    """Install the registry implementation (``None`` restores the null default).

    Called once by the composition root. Idempotent; the last writer wins."""
    global _SOURCE
    with _LOCK:
        _SOURCE = source if source is not None else _DEFAULT
    logger.info("catalog source installed: %s",
                getattr(_SOURCE, "name", type(_SOURCE).__name__))


def get_catalog_source() -> Any:
    """The installed :class:`CatalogSource` (the null default when none)."""
    return _SOURCE


def reset_catalog_source() -> None:
    """Back to the null default (tests)."""
    set_catalog_source(None)


# ---------------------------------------------------------------------------
# Guarded accessors — the shapes storage code actually calls. Each swallows an
# implementation error into the null answer so a registry hiccup can never
# turn a transfer into a traceback.
# ---------------------------------------------------------------------------
def catalog_rows() -> dict[str, dict]:
    try:
        rows = _SOURCE.rows()
    except Exception:  # noqa: BLE001
        logger.debug("catalog rows() failed", exc_info=True)
        return {}
    return rows if isinstance(rows, dict) else {}


def catalog_get(model_key: str) -> Optional[Any]:
    if not model_key:
        return None
    try:
        return _SOURCE.get(model_key)
    except Exception:  # noqa: BLE001
        logger.debug("catalog get(%s) failed", model_key, exc_info=True)
        return None


def catalog_canonical_key(model_key: str) -> Optional[str]:
    if not model_key:
        return None
    try:
        return _SOURCE.canonical_key(model_key)
    except Exception:  # noqa: BLE001
        logger.debug("catalog canonical_key(%s) failed", model_key, exc_info=True)
        return None


def catalog_resolve_dir(model_key: str) -> Optional[str]:
    if not model_key:
        return None
    try:
        return _SOURCE.resolve_dir(model_key)
    except Exception:  # noqa: BLE001
        logger.debug("catalog resolve_dir(%s) failed", model_key, exc_info=True)
        return None


def catalog_register(model_key: str, row: dict) -> bool:
    try:
        return bool(_SOURCE.register(model_key, row))
    except Exception:  # noqa: BLE001
        logger.warning("catalog register(%s) failed", model_key, exc_info=True)
        return False


def catalog_refresh() -> None:
    try:
        _SOURCE.refresh()
    except Exception:  # noqa: BLE001
        logger.warning("catalog refresh failed", exc_info=True)


def routing_of(cfg: Any, **overrides: Any) -> dict:
    """A plain routing dict (what ``model_paths`` reads) from a config object
    OR a dict row — the one adapter so call sites never re-spell the fields."""
    def _f(name):
        if isinstance(cfg, dict):
            return cfg.get(name)
        return getattr(cfg, name, None)
    out = {
        "hub_id": _f("hub_id"),
        "name": _f("name"),
        "framework": _f("framework"),
        "primary_task": _f("primary_task") or _f("task"),
        "tasks": _f("tasks"),
        "filename": _f("filename"),
        "include": _f("include"),
        "folder": _f("folder"),
        "dir": _f("dir"),
    }
    out.update(overrides)
    return out


__all__ = [
    "CatalogSource", "NullCatalogSource", "DictCatalogSource",
    "set_catalog_source", "get_catalog_source", "reset_catalog_source",
    "catalog_rows", "catalog_get", "catalog_canonical_key",
    "catalog_resolve_dir", "catalog_register", "catalog_refresh",
    "routing_of",
]
