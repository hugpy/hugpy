"""Where curation keeps its own state — one place, every root injectable.

Curation is the sole owner of the review record (sqlite), the discovery
dossiers (one JSON file per criteria/repo, verdict included), the trial
artifacts a dossier's sample battery writes, the saved review criteria and the
dossier fetch cache (``PARTITION.md`` "State ownership"). Every module that
used to derive its root on its own — from ``hugpy_platform.constants
.DEFAULT_ROOT``, ``~/.config``, ``~/.cache`` or, for trials, the ORACLE's
benchmark run root — now asks here, so an operator (or a test conftest) can
move the whole set with one variable and still override any single root with
the env var that module always honoured.

Resolution order for every root:

1. the module's own env var (``REVIEW_DB``, ``DOSSIER_DIR``,
   ``DOSSIER_TRIAL_ROOT``, ``REVIEW_CRITERIA_DIR``, ``DOSSIER_CACHE_DIR``) —
   kept verbatim, so existing deployments (the ``hugpy-review@`` units set
   ``REVIEW_DB``) and the tests keep working;
2. the package-wide ``HUGPY_CURATION_ROOT`` (state) / ``HUGPY_CURATION_CACHE``;
3. the platform's per-OS application directories (``hugpy_platform.app_dirs``):
   ``models_root()/review`` for the record, dossiers and trials — the tree the
   review has always written under (``<DEFAULT_ROOT>/review``), so a box that
   upgrades finds its history where it left it — ``config_dir()/review`` for
   the saved criteria and ``cache_dir()/discovery-dossier`` for the fetch
   cache (both byte-identical to the historical ``~/.config/hugpy/review`` and
   ``~/.cache/hugpy/discovery-dossier`` on Linux).

Trials used to fall back to ``hugpy_oracle.benchmark.default_run_root()``;
that is the oracle's state and curation must not write into it, so the
default is now ``<review_root>/trials``. ``DOSSIER_TRIAL_ROOT`` still wins.

Env reads are plain ``os.environ`` on purpose: a state root must not depend
on a ``.env`` loader having run, and a test that sets ``monkeypatch.setenv``
must see the change immediately. Nothing here creates directories except
:func:`ensure_dir`; callers that write call it, callers that read do not.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "CURATION_ROOT_ENV",
    "CURATION_CACHE_ENV",
    "REVIEW_DB_ENV",
    "DOSSIER_DIR_ENV",
    "TRIAL_ROOT_ENV",
    "CRITERIA_DIR_ENV",
    "FETCH_CACHE_ENV",
    "review_root",
    "review_db_path",
    "dossiers_root",
    "trials_root",
    "criteria_dir",
    "fetch_cache_dir",
    "ensure_dir",
]

CURATION_ROOT_ENV = "HUGPY_CURATION_ROOT"
CURATION_CACHE_ENV = "HUGPY_CURATION_CACHE"

# Per-store env vars, unchanged from the modules that introduced them.
REVIEW_DB_ENV = "REVIEW_DB"
DOSSIER_DIR_ENV = "DOSSIER_DIR"
TRIAL_ROOT_ENV = "DOSSIER_TRIAL_ROOT"
CRITERIA_DIR_ENV = "REVIEW_CRITERIA_DIR"
FETCH_CACHE_ENV = "DOSSIER_CACHE_DIR"


from hugpy_platform.env import env_value as _env


def _home_fallback(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), ".local", "share", "hugpy", *parts)


def _platform_models_root() -> str:
    try:
        from hugpy_platform.app_dirs import models_root
        return models_root()
    except Exception as exc:  # noqa: BLE001 — a root must always resolve
        logger.warning("curation config: models_root unreadable (%s: %s); "
                       "using ~/.local/share/hugpy", type(exc).__name__, exc)
        return _home_fallback()


def _platform_config_dir() -> str:
    try:
        from hugpy_platform.app_dirs import config_dir
        return config_dir()
    except Exception as exc:  # noqa: BLE001
        logger.warning("curation config: config_dir unreadable (%s: %s); "
                       "using ~/.config/hugpy", type(exc).__name__, exc)
        return os.path.join(os.path.expanduser("~"), ".config", "hugpy")


def _platform_cache_dir() -> str:
    try:
        from hugpy_platform.app_dirs import cache_dir
        return cache_dir()
    except Exception as exc:  # noqa: BLE001
        logger.warning("curation config: cache_dir unreadable (%s: %s); "
                       "using ~/.cache/hugpy", type(exc).__name__, exc)
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.join(base, "hugpy")


def ensure_dir(path: str) -> str:
    """``makedirs`` that never raises — a state root that cannot be created is
    reported by the write that follows, not by the path lookup."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logger.warning("curation config: cannot create %s (%s)", path, exc)
    return path


def review_root() -> str:
    """The review record, dossiers and trials live under this tree.
    ``HUGPY_CURATION_ROOT`` else ``<models_root>/review`` (historically
    ``<DEFAULT_ROOT>/review``)."""
    return _env(CURATION_ROOT_ENV) or os.path.join(_platform_models_root(), "review")


def review_db_path() -> str:
    """The on-box review record (sqlite). ``REVIEW_DB`` wins."""
    return _env(REVIEW_DB_ENV) or os.path.join(review_root(), "reviews.db")


def dossiers_root() -> str:
    """``<dossiers_root>/<criteria>/<org__repo>.json``. ``DOSSIER_DIR`` wins."""
    return _env(DOSSIER_DIR_ENV) or os.path.join(review_root(), "dossiers")


def trials_root() -> str:
    """Sample artifacts of a dossier's trial battery. ``DOSSIER_TRIAL_ROOT``
    wins; otherwise ``<review_root>/trials`` (curation's own tree, never the
    oracle's benchmark root)."""
    return _env(TRIAL_ROOT_ENV) or os.path.join(review_root(), "trials")


def criteria_dir() -> str:
    """Saved review criteria (``<name>.json``). ``REVIEW_CRITERIA_DIR`` wins;
    otherwise ``<config_dir>/review``."""
    return _env(CRITERIA_DIR_ENV) or os.path.join(_platform_config_dir(), "review")


def fetch_cache_dir() -> str:
    """The dossier fetch cache. ``DOSSIER_CACHE_DIR`` wins, then
    ``HUGPY_CURATION_CACHE``, then ``<cache_dir>/discovery-dossier``."""
    return (_env(FETCH_CACHE_ENV) or _env(CURATION_CACHE_ENV)
            or os.path.join(_platform_cache_dir(), "discovery-dossier"))
