"""Relocation finder for the transitional ``abstract_hugpy_dev`` namespace.

Copied from the monolith's generated ``_relocations.py`` (py/tooling/
extract_package.py) and extended for the compatibility distribution:

* ``_relocations.json`` maps every module that lived in the monolith to the
  module that now owns it in a ``hugpy_*`` package. The finder aliases the old
  dotted path to the *same* module object, so nothing is imported twice and
  ``unittest.mock.patch("abstract_hugpy_dev.comms.jobs.job_store")`` still
  patches the real thing.
* ``_namespace.json`` describes the retired aggregator modules (``imports/``,
  ``comms/__init__``, ``managers/__init__``, every ``imports.py`` and so on).
  Those files no longer exist anywhere; the finder synthesises a package-like
  module for each (``__path__ == []``) whose attribute lookups resolve lazily:
  submodules through the map, names through the import statements the old
  aggregator used to execute (``from .jobs import Job``, ``from .embed import *``).
  Nothing heavy is imported until a name is actually asked for.
* Every dotted prefix of a known module (``abstract_hugpy_dev.flask_app``,
  ``...flask_app.app.functions.downloads``) is importable as a plain namespace
  even when the old ``__init__`` had no alias target, so deep paths such as
  ``abstract_hugpy_dev.flask_app.app.routes.chat_routes`` keep resolving.

This package carries no implementation; it is deleted once every caller uses
the ``hugpy_*`` imports directly (see RELOCATIONS.md for the table).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MAP: dict[str, str] = json.loads((_HERE / "_relocations.json").read_text(encoding="utf-8"))
_NAMESPACE: dict = json.loads((_HERE / "_namespace.json").read_text(encoding="utf-8"))
#: old aggregator module -> ordered import statements it used to execute
_AGGREGATORS: dict[str, list[dict]] = _NAMESPACE.get("aggregators", {})
#: old module -> reason it has no home any more (dead code, unwired copy, ...)
_RETIRED: dict[str, str] = _NAMESPACE.get("retired", {})

_ROOT = __name__.rsplit(".", 1)[0]
_PREFIX = _ROOT + "."


def _package_like() -> set[str]:
    out: set[str] = set()
    for name in list(_MAP) + list(_AGGREGATORS):
        parts = name.split(".")
        for cut in range(2, len(parts)):
            out.add(".".join(parts[:cut]))
    return out


#: every strict dotted prefix of a known old module (package-like paths)
_PACKAGE_LIKE: set[str] = _package_like()


def relocated(fullname: str) -> str | None:
    """New dotted name for an old one, or ``None`` when the map has no entry."""
    if fullname in _MAP:
        return _MAP[fullname]
    parts = fullname.split(".")
    for cut in range(len(parts) - 1, 1, -1):
        head = ".".join(parts[:cut])
        if head in _MAP:
            return _MAP[head] + "." + ".".join(parts[cut:])
    return None


def _target_exists(new: str) -> bool:
    try:
        return importlib.util.find_spec(new) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def is_namespace(fullname: str) -> bool:
    """True when the finder synthesises ``fullname`` instead of aliasing it."""
    return fullname in _AGGREGATORS or fullname in _PACKAGE_LIKE


def resolvable(fullname: str) -> bool:
    """True when ``import fullname`` can be satisfied by this finder."""
    if not fullname.startswith(_PREFIX):
        return False
    new = relocated(fullname)
    if new is not None and new != fullname and _target_exists(new):
        return True
    return is_namespace(fullname)


def children(fullname: str) -> list[str]:
    """Direct submodule names known under an old package path."""
    prefix = fullname + "."
    out: set[str] = set()
    for name in list(_MAP) + list(_AGGREGATORS) + list(_PACKAGE_LIKE):
        if name.startswith(prefix):
            out.add(name[len(prefix):].split(".", 1)[0])
    return sorted(out)


# ---------------------------------------------------------------------------
# alias loader: old path -> the same module object from the new package


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        # An old *package* aliased to a plain module still needs submodule
        # lookups (``from old.pkg import sub``) to reach this finder.
        if not hasattr(module, "__path__") and (
            any(k.startswith(spec.name + ".") for k in _MAP) or spec.name in _PACKAGE_LIKE
        ):
            module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        return None


# ---------------------------------------------------------------------------
# synthesised namespace / aggregator modules

_guard = threading.local()


def _entered(key: tuple[str, str]) -> bool:
    active = getattr(_guard, "active", None)
    if active is None:
        active = _guard.active = set()
    if key in active:
        return True
    active.add(key)
    return False


def _leave(key: tuple[str, str]) -> None:
    _guard.active.discard(key)


def _bound_name(stmt: dict) -> str:
    kind = stmt["kind"]
    if kind == "import":
        return stmt.get("as") or stmt["module"].split(".")[0]
    if kind in ("alias", "call"):
        return stmt["name"]
    return ""


def _resolve_statement(stmt: dict, name: str, owner: str):
    """Value ``name`` would have after executing ``stmt``; raises AttributeError."""
    kind = stmt["kind"]
    if kind == "import":
        if _bound_name(stmt) != name:
            raise AttributeError(name)
        module = importlib.import_module(stmt["module"])
        if stmt.get("as"):
            return module
        return sys.modules[stmt["module"].split(".")[0]]
    if kind == "alias":
        if stmt["name"] != name:
            raise AttributeError(name)
        return aggregator_getattr(owner, stmt["target"])
    if kind == "call":  # ``logger = get_logFile(__name__)``
        if stmt["name"] != name:
            raise AttributeError(name)
        func = aggregator_getattr(owner, stmt["func"])
        return func(*([owner] * stmt.get("argc", 1)))
    if kind == "from":
        for src, bound in stmt["names"]:
            if bound == name:
                module = importlib.import_module(stmt["module"])
                try:
                    return getattr(module, src)
                except AttributeError:
                    return importlib.import_module(stmt["module"] + "." + src)
        raise AttributeError(name)
    if kind == "star":
        if name.startswith("_"):
            raise AttributeError(name)
        module = importlib.import_module(stmt["module"])
        exported = module.__dict__.get("__all__")
        if exported is not None and name not in exported:
            raise AttributeError(name)
        return getattr(module, name)
    raise AttributeError(name)


def aggregator_getattr(owner: str, name: str):
    """Attribute resolution for a retired aggregator module named ``owner``.

    Order: a relocated submodule ``owner.name`` first, then the aggregator's
    import statements in reverse (last binding wins, as it did at import time).
    """
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    full = owner + "." + name
    if resolvable(full):
        try:
            return importlib.import_module(full)
        except ImportError:
            pass
    if name == "__all__":
        return aggregator_public_names(owner)
    key = (owner, name)
    if _entered(key):
        raise AttributeError(name)
    try:
        for stmt in reversed(_AGGREGATORS.get(owner, [])):
            try:
                return _resolve_statement(stmt, name, owner)
            except (ImportError, AttributeError):
                continue
    finally:
        _leave(key)
    raise AttributeError(
        f"module {owner!r} has no attribute {name!r} "
        f"(retired aggregator of abstract_hugpy_dev; see RELOCATIONS.md)"
    )


def aggregator_public_names(owner: str) -> list[str]:
    """Names ``from owner import *`` used to bind (computed lazily, imports sources)."""
    key = (owner, "__all__")
    if _entered(key):
        return []
    names: set[str] = set(children(owner))
    try:
        for stmt in _AGGREGATORS.get(owner, []):
            kind = stmt["kind"]
            if kind in ("import", "alias", "call"):
                names.add(_bound_name(stmt))
            elif kind == "from":
                names.update(bound for _, bound in stmt["names"])
            elif kind == "star":
                try:
                    module = importlib.import_module(stmt["module"])
                except ImportError:
                    continue
                exported = module.__dict__.get("__all__")
                if exported is None and isinstance(module, _NamespaceModule):
                    exported = aggregator_public_names(module.__name__)
                if exported is None:
                    exported = [n for n in module.__dict__ if not n.startswith("_")]
                names.update(exported)
    finally:
        _leave(key)
    return sorted(n for n in names if not n.startswith("_"))


class _NamespaceModule(types.ModuleType):
    """Package-like stand-in for a retired aggregator or a bare old package path."""

    def __getattr__(self, name: str):
        return aggregator_getattr(self.__name__, name)

    def __dir__(self):
        base = set(super().__dir__())
        base.update(children(self.__name__))
        for stmt in _AGGREGATORS.get(self.__name__, []):
            if stmt["kind"] in ("import", "alias", "call"):
                base.add(_bound_name(stmt))
            elif stmt["kind"] == "from":
                base.update(bound for _, bound in stmt["names"])
        return sorted(base)

    def __repr__(self):
        return f"<module {self.__name__!r} (abstract_hugpy_dev compat namespace)>"


class _NamespaceLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _NamespaceModule(spec.name, _NAMESPACE_DOC)
        module.__path__ = []  # type: ignore[attr-defined]
        module.__package__ = spec.name
        module.__file__ = None
        return module

    def exec_module(self, module):
        return None


_NAMESPACE_DOC = (
    "Retired abstract_hugpy_dev aggregator kept alive by the compat finder. "
    "Attributes resolve lazily to the hugpy_* packages; see RELOCATIONS.md."
)


# ---------------------------------------------------------------------------


class RelocationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        new = relocated(fullname)
        if new is not None and new != fullname and _target_exists(new):
            return importlib.util.spec_from_loader(fullname, _AliasLoader(new))
        if is_namespace(fullname):
            return importlib.util.spec_from_loader(fullname, _NamespaceLoader(), is_package=True)
        return None


def install() -> None:
    if not any(isinstance(f, RelocationFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, RelocationFinder())


def iter_old_modules():
    """Every old dotted path this finder answers for: (old, new-or-None, kind)."""
    seen: set[str] = set()
    for old, new in sorted(_MAP.items()):
        seen.add(old)
        yield old, new, "relocated"
    for old in sorted(_AGGREGATORS):
        if old not in seen and old != _ROOT:
            seen.add(old)
            yield old, None, "aggregator"
    for old in sorted(_PACKAGE_LIKE):
        if old not in seen:
            seen.add(old)
            yield old, None, "namespace"
