"""hugpy-control: the job/settings/principal control plane of the Hugpy ecosystem.

Submodules (all stdlib + ``hugpy_platform``; none import an engine, fleet or
server package):

    bus         in-process typed message bus + topic constants
    jobs        JobStore — the one all-transport job store (chat, media, downloads)
    shared      SqliteMirror — the cross-process mirror the JobStore rides on
    settings    SettingsStore — namespaced runtime settings (F4)
    principals  PrincipalStore — identities, tokens, capability checks (F2)
    calllog     append-only call log + tail reader
    job_schemas compat re-export of the download-job names

Submodules are resolved lazily (PEP 562) so ``import hugpy_control`` stays
side-effect free: the store singletons are only created when the module that
owns them is first imported.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "bus",
    "calllog",
    "job_schemas",
    "jobs",
    "principals",
    "settings",
    "shared",
]

_SUBMODULES = frozenset(n for n in __all__ if not n.startswith("__"))


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
