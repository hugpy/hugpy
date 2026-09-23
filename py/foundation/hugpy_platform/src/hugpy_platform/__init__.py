"""hugpy-platform: the stdlib-first foundation every Hugpy package stands on.

What lives here (and nothing above it — no engine, storage, fleet or server):

    central          canonical central base URL (``HUGPY_BASE_URL`` + aliases)
    platform_facade  OS facts (``IS_WINDOWS``/``IS_MACOS``/``IS_LINUX``) and the
                     sanitising ``env_value`` seam over ``abstract_essentials``
    app_dirs         per-OS data/config/cache/engine dirs and the ``~/.hugpy``
                     runtime-file accessors
    constants        storage roots, HF cache layout, fleet-wide defaults
                     (import has side effects: creates dirs — import it explicitly)
    utils            small filesystem / message / naming helpers
    module_imports   ``get_<package>()`` lazy accessors for heavy ML libraries
    async_runtime    one process-wide asyncio loop + sync bridges
    client_liveness  "is the HTTP client still there" probe (Flask optional)
    binaries / procutil / hardware   executables, process trees, RAM/GPU probes
    compat_pydantic  pure-Python pydantic stand-in for platforms without pydantic_core
    except_utils     ``caught`` / ``attempt`` / ``catching`` logging helpers
    trust            Hugging Face publisher trust tiers
    buildinfo        which build is this process: workspace version, sha/dirty,
                     editable source, lockstep check (``/health``, ``/build``, heartbeat)

This ``__init__`` re-exports only the light, side-effect-free names below; the
heavier modules are imported by dotted path. ``buildinfo`` is exported lazily
(``hugpy_platform.buildinfo`` resolves on first attribute access, PEP 562) so
importing the package never pays for it.
"""
from __future__ import annotations

from hugpy_platform.app_dirs import (
    cache_dir,
    config_dir,
    data_dir,
    engine_dir,
    ensure_hugpy_home,
    hugpy_config_dir,
    hugpy_home,
    hugpy_logs_dir,
    hugpy_run_dir,
    hugpy_state_dir,
    models_root,
)
from hugpy_platform.central import CENTRAL_ENV_VARS, DEFAULT_CENTRAL, central_base_url
from hugpy_platform.compat_pydantic import ensure_pydantic
from hugpy_platform.except_utils import FAILED, attempt, catching, caught, caught_block
from hugpy_platform.platform_facade import EXE_SUFFIX, IS_LINUX, IS_MACOS, IS_WINDOWS, env_value
from hugpy_platform.trust import trust_label, trust_tier

try:  # the installed distribution's version: the workspace tag/commit, never a literal
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("hugpy-platform")
except Exception:  # noqa: BLE001 — source tree without metadata
    __version__ = "0.0.0+unknown"

_LAZY_SUBMODULES = ("buildinfo",)


def __getattr__(name: str):
    """Lazy submodule export: ``hugpy_platform.buildinfo`` without an import-time cost."""
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # central
    "CENTRAL_ENV_VARS",
    "DEFAULT_CENTRAL",
    "central_base_url",
    # platform facts + env seam
    "EXE_SUFFIX",
    "IS_LINUX",
    "IS_MACOS",
    "IS_WINDOWS",
    "env_value",
    # app dirs
    "cache_dir",
    "config_dir",
    "data_dir",
    "engine_dir",
    "ensure_hugpy_home",
    "hugpy_config_dir",
    "hugpy_home",
    "hugpy_logs_dir",
    "hugpy_run_dir",
    "hugpy_state_dir",
    "models_root",
    # pydantic shim
    "ensure_pydantic",
    # exception helpers
    "FAILED",
    "attempt",
    "catching",
    "caught",
    "caught_block",
    # trust tiers
    "trust_label",
    "trust_tier",
    # build identity (lazy submodule)
    "buildinfo",
]
