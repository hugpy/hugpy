"""hugpy-server: the Flask composition root of the Hugpy ecosystem.

Public surface (all lazy — importing this package imports no Flask app):

* ``create_app`` / ``get_hugpy_flask`` — the app factory (``hugpy_server.wsgi_app``)
* ``install_all`` — the composition wiring (``hugpy_server.wiring``)
* ``main`` — the ``hugpy-serve`` entry point
"""
from __future__ import annotations

try:  # the installed distribution's version: the workspace tag/commit, never a literal
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("hugpy-server")
except Exception:  # noqa: BLE001 — source tree without metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "create_app", "get_hugpy_flask", "install_all", "main"]


def __getattr__(name: str):
    if name in ("create_app", "get_hugpy_flask", "main"):
        from hugpy_server import wsgi_app
        return getattr(wsgi_app, name)
    if name == "install_all":
        from hugpy_server.wiring import install_all
        return install_all
    raise AttributeError(f"module 'hugpy_server' has no attribute {name!r}")
