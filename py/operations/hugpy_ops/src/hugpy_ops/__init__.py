"""hugpy_ops — operational consumers of the Hugpy public APIs.

Leaf package: nothing here is imported during normal server startup. Each
tool is a separate console script and is loaded lazily from this namespace:

* :mod:`hugpy_ops.sentinel` — bound-exceeded detection over central's HTTP
  read surfaces -> deduplicated cases -> one bounded diagnosis agent per new
  case (``hugpy-sentinel``).
* :mod:`hugpy_ops.chaos` — the chaos-and-learn exerciser and the k7 offload
  sweep, an HTTP client of central (``hugpy-chaos``).
* :mod:`hugpy_ops.keeper` — the stationary terminal REPL in which a served
  model keeps a machine (``hugpy-keeper``).
* :mod:`hugpy_ops.todo_keeper` / :mod:`hugpy_ops.todo_keeper_daemon` — the
  pure todo-keeper contract and the enrolling node daemon
  (``hugpy-todo-keeper``).
* :mod:`hugpy_ops.provisioner` — declared-but-missing weights across the
  engine, studio and comfy registries, enqueued on ``hugpy_storage``'s
  download queue (``hugpy-provisioner``).
* :mod:`hugpy_ops.versions` — the ecosystem distribution versions the
  sentinel reports next to a worker version skew.

This ``__init__`` is deliberately light: submodules load on first attribute
access, so ``import hugpy_ops`` pulls in nothing beyond the stdlib.
"""

from __future__ import annotations

import importlib

try:  # the installed distribution's version: the workspace tag/commit, never a literal
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("hugpy-ops")
except Exception:  # noqa: BLE001 — source tree without metadata
    __version__ = "0.0.0+unknown"

_SUBMODULES = ("chaos", "drift", "keeper", "provisioner", "sentinel", "todo_keeper",
               "todo_keeper_daemon", "versions")

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
