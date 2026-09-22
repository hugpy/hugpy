from ._relocations import install as _install_relocations
_install_relocations()
# Must run before anything imports pydantic: on platforms without pydantic_core
# (e.g. Termux/Android, where it has no wheel and needs Rust to build), install a
# pure-Python pydantic shim so the package still imports. No-op where the real
# pydantic is available. See _compat_pydantic.py.
from hugpy_platform.compat_pydantic import ensure_pydantic as _ensure_pydantic
_ensure_pydantic()

# THE version number for the package. Single-sourced: pyproject.toml reads this
# attribute (`[tool.setuptools.dynamic] version = {attr = ...}`), so the wheel's
# metadata and the running module can never disagree. That desync is exactly what
# shipped in 0.1.224 — dist metadata 0.1.224, this literal 0.1.223 — which made
# every worker on the required version report version_ok:false forever.
# Version follows the INSTALLED package metadata, which is single-sourced from
# pyproject's static `version`. Reading it back here means the running module can
# never drift from what was packaged (the 2026-07-20 skew-honesty goal), while
# the version still lives in the .toml where abstract-pypit and every other tool
# expect it. Falls back to a literal only for an uninstalled source run.
# Exposed over HTTP at GET /version.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("abstract_hugpy_dev")
except Exception:
    __version__ = "0.1.252"

from .imports import *
from .managers import *
from .utils import *
