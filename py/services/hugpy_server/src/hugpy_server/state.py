"""Server-owned state roots (API keys, video share keys, Discord bindings,
install links).

The server owns exactly these small JSON stores (PARTITION.md "State
ownership"). Their directory is resolved through ``hugpy_platform`` only:

* ``HUGPY_SERVER_STATE_DIR`` (env or ``.env``) wins when set;
* otherwise ``hugpy_platform.constants.PROJECTS_HOME`` — the same directory
  the monolith used (``dirname(settings.manifest_path)``), so an existing
  deployment keeps finding its ``api_keys.json`` where it always was.

No other package's state files are opened here.
"""
from __future__ import annotations

import os

from hugpy_platform.constants import PROJECTS_HOME
from hugpy_platform.platform_facade import env_value


def server_state_dir() -> str:
    """Directory holding the server's own JSON stores (created lazily by the
    writers, never here — a read must not create directories)."""
    override = os.environ.get("HUGPY_SERVER_STATE_DIR") or env_value("HUGPY_SERVER_STATE_DIR")
    return str(override or PROJECTS_HOME)


def server_state_path(name: str) -> str:
    return os.path.join(server_state_dir(), name)


__all__ = ["server_state_dir", "server_state_path"]
