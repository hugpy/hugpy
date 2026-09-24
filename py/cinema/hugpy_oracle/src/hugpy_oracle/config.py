"""Where the oracle keeps its own state — one place, every root injectable.

The oracle is the sole owner of its plans, run journals, reliability ledger,
routing matrices, benchmark runs, scorecards and repair state
(``PARTITION.md`` "State ownership"). Every module that used to derive its
root on its own — from ``hugpy_platform.constants.DEFAULT_ROOT``, a
box-specific literal, or ``~`` — now asks here, so an operator (or a test
conftest) can move the whole set with one variable and still override any
single root with the env var that module always honoured.

Resolution order for every root:

1. the module's own env var (``ORACLE_LEDGER_PATH``, ``ORACLE_BENCHMARK_ROOT``,
   ``HUGPY_PERFORMANCE_RUN_ROOT``, ...) — kept verbatim, so existing
   deployments and the test conftest keep working;
2. the package-wide ``HUGPY_ORACLE_HOME`` / ``HUGPY_ORACLE_RUNS_ROOT``;
3. the platform's per-OS application directories (``hugpy_platform.app_dirs``):
   ``hugpy_home()/oracle`` for the small stores, ``models_root()/video_intel``
   for the run journals (the layout the movie runners already share on disk,
   so an operator finds every run under one tree).

Env reads are plain ``os.environ`` on purpose: a state root must not depend
on a ``.env`` loader having run, and a test that sets ``monkeypatch.setenv``
must see the change immediately.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ORACLE_HOME_ENV",
    "ORACLE_RUNS_ROOT_ENV",
    "LEDGER_PATH_ENV",
    "BENCHMARK_ROOT_ENV",
    "MATRIX_PATH_ENV",
    "PERFORMANCE_RUN_ROOT_ENV",
    "SCRIPT_FIRST_RUN_ROOT_ENV",
    "LEDGER_CACHE_ROOT_ENV",
    "oracle_home",
    "runs_root",
    "ledger_path",
    "benchmark_root",
    "matrix_path",
    "performance_run_root",
    "script_first_run_root",
    "ledger_cache_root",
    "tts_output_dir",
]

ORACLE_HOME_ENV = "HUGPY_ORACLE_HOME"
ORACLE_RUNS_ROOT_ENV = "HUGPY_ORACLE_RUNS_ROOT"

# Per-store env vars, unchanged from the modules that introduced them.
LEDGER_PATH_ENV = "ORACLE_LEDGER_PATH"
BENCHMARK_ROOT_ENV = "ORACLE_BENCHMARK_ROOT"
MATRIX_PATH_ENV = "ORACLE_ROUTING_MATRIX"
PERFORMANCE_RUN_ROOT_ENV = "HUGPY_PERFORMANCE_RUN_ROOT"
SCRIPT_FIRST_RUN_ROOT_ENV = "HUGPY_SCRIPT_FIRST_RUN_ROOT"
LEDGER_CACHE_ROOT_ENV = "HUGPY_INTERIM_LEDGER_ROOT"

#: The benchmark battery's historical home on the dev unit. Honoured only when
#: it already exists, so a fresh box never grows a ``/home/ubuntu`` tree.
LEGACY_BENCHMARK_ROOT = "/home/ubuntu/station/model-battery"


from hugpy_platform.env import env_value as _env


def _platform_home() -> str:
    try:
        from hugpy_platform.app_dirs import hugpy_home
        return hugpy_home()
    except Exception as exc:  # noqa: BLE001 — a root must always resolve
        logger.warning("oracle config: hugpy_home unreadable (%s: %s); using ~/.hugpy",
                       type(exc).__name__, exc)
        return os.path.join(os.path.expanduser("~"), ".hugpy")


def _platform_models_root() -> str:
    try:
        from hugpy_platform.app_dirs import models_root
        return models_root()
    except Exception as exc:  # noqa: BLE001
        logger.warning("oracle config: models_root unreadable (%s: %s); using ~/.hugpy",
                       type(exc).__name__, exc)
        return os.path.join(os.path.expanduser("~"), ".hugpy")


def oracle_home() -> str:
    """The small stores: reliability ledger, routing matrices, steward state.
    ``HUGPY_ORACLE_HOME`` else ``<hugpy_home>/oracle`` (``~/.hugpy/oracle``)."""
    return _env(ORACLE_HOME_ENV) or os.path.join(_platform_home(), "oracle")


def runs_root() -> str:
    """The run journals (performance / script-first / interim-ledger cache).
    ``HUGPY_ORACLE_RUNS_ROOT`` else ``<models_root>/video_intel`` — the tree
    the movie runners already write under, so every run of a job is found
    beside the job's artifacts."""
    return _env(ORACLE_RUNS_ROOT_ENV) or os.path.join(_platform_models_root(), "video_intel")


def ledger_path() -> str:
    """The reliability ledger (sqlite). ``ORACLE_LEDGER_PATH`` wins."""
    return _env(LEDGER_PATH_ENV) or os.path.join(oracle_home(), "reliability.sqlite")


def benchmark_root() -> str:
    """Where benchmark sweeps (``oracle-<stamp>/``) are written and read."""
    override = _env(BENCHMARK_ROOT_ENV)
    if override:
        return override
    if os.path.isdir(LEGACY_BENCHMARK_ROOT):
        return LEGACY_BENCHMARK_ROOT
    return os.path.join(oracle_home(), "model-battery")


def matrix_path() -> Optional[str]:
    """An explicitly pinned routing matrix, or ``None`` (then the newest
    registry-verified matrix under :func:`benchmark_root` is used)."""
    return _env(MATRIX_PATH_ENV)


def performance_run_root() -> str:
    return _env(PERFORMANCE_RUN_ROOT_ENV) or runs_root()


def script_first_run_root() -> str:
    return _env(SCRIPT_FIRST_RUN_ROOT_ENV) or runs_root()


def ledger_cache_root() -> str:
    return _env(LEDGER_CACHE_ROOT_ENV) or runs_root()


def tts_output_dir() -> str:
    """Where relayed TTS bytes are materialised for hashing (runtime)."""
    return os.path.join(runs_root(), "tts")
