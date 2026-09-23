"""hugpy: the friendly front door of the Hugpy ecosystem.

This meta distribution is deliberately thin. It ships two commands:

* ``hugpy`` (:mod:`hugpy.cli`) — dispatches ``serve``, ``worker``, ``bot``,
  ``download`` and friends to the package that owns each feature, resolving the
  owner lazily so a bare install stays importable.
* ``hpy`` (:mod:`hugpy.hpy`) — a stdlib-only fleet browser and emergency
  inference client that talks HTTP to a running central.

Importing ``hugpy`` loads nothing but the standard library and
``hugpy_platform``; every other ``hugpy_*`` package is an optional extra.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("hugpy")
except PackageNotFoundError:  # running from a source tree without metadata
    __version__ = "0.0.0+unknown"  # source tree without metadata

__all__ = ["__version__", "cli", "hpy"]


def __getattr__(name: str):
    # ``hugpy.cli`` / ``hugpy.hpy`` resolve on demand so ``import hugpy``
    # never parses the argparse trees or touches the network.
    if name in ("cli", "hpy"):
        import importlib

        return importlib.import_module(f"hugpy.{name}")
    raise AttributeError(f"module 'hugpy' has no attribute {name!r}")
