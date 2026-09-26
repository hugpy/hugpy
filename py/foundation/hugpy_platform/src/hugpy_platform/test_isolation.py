"""Test-storage isolation + a live-storage audit guard (single source of truth).

Why this exists (incident 2026-09-24): the media job bus, the reservation
ledger, identity profiles and the studio scratch tree all resolve under
``hugpy_platform.constants.DEFAULT_ROOT``. A test process whose ``DEFAULT_ROOT``
resolved to the LIVE storage root (``/mnt/.../llm_storage`` — inherited in the
process environment, or from a ``.env`` in the CWD) therefore opened WAL sqlite
connections against, and wrote temp trees INTO, the operator's live storage:
``reservations.db``, ``admission_queue.sqlite`` and ``video_intel/_scratch``
(the same ``video_intel`` directory that holds ``media_jobs.db``). The old
per-package ``os.environ.setdefault("DEFAULT_ROOT", <tmp>)`` did NOT override an
inherited live value, so the isolation was silently bypassed.

Two entry points, both idempotent and stdlib-only:

* :func:`isolate_storage_to_tmp` — FORCE every storage-root env var to a private
  temp dir BEFORE ``hugpy_platform.constants`` is imported (and rebind the
  constants module + video state if they were already imported). Opt out with
  ``HUGPY_TEST_LIVE_ROOT=1`` for a deliberate run against a real catalog.
* :func:`install_live_storage_guard` — an ``sys.addaudithook`` that RAISES on any
  ``sqlite3.connect`` / write-``open`` / ``os.remove`` / ``os.rename`` /
  ``shutil.copyfile`` of a DB-shaped path that does not resolve under the private
  temp base (or a system temp dir). The safety net: a test can never touch live
  storage even if a new code path forgets to honour the env roots.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Storage-root env vars, mapped to their DIRECTORY under the private temp base.
# Only DIRECTORY roots are pinned: the DB *files* (media_jobs.db under
# HUGPY_VIDEO_STATE_DIR, reservations.db under PROJECTS_HOME, ...) then DERIVE
# from these, so a test that repoints one directory var (e.g. HUGPY_VIDEO_STATE_
# DIR) still gets its own derived DB path instead of a pinned file var winning.
# ORACLE_LEDGER_PATH is a leaf file (no derivation) and is safe to pin directly.
_ROOT_SUBPATHS = {
    "DEFAULT_ROOT": "",
    "MODELS_HOME": "models",
    "UPLOADS_HOME": "uploads",
    "PROJECTS_HOME": "projects",
    "IDENTITIES_HOME": "identities",
    "DATASETS_HOME": "datasets",
    "HF_CACHE": "cache",
    "HUGPY_HOME": "hugpy_home",
    "HUGPY_VIDEO_STATE_DIR": "video_intel",
    "ORACLE_LEDGER_PATH": "oracle-reliability.sqlite",
}

_GUARD_INSTALLED = False

# The LIVE storage roots captured BEFORE isolation overrode the environment — the
# guard's DENY list. Anything NOT under one of these (coverage data files, the
# pytest cache, the job work dir, system temp, site-packages) is fine; only a
# DB-shaped access that resolves under a captured live root is refused.
_LIVE_ROOTS: list[str] = []

# The storage-root env vars whose PRE-override values name a would-be live root.
_LIVE_ROOT_VARS = (
    "DEFAULT_ROOT", "MODELS_HOME", "UPLOADS_HOME", "PROJECTS_HOME",
    "IDENTITIES_HOME", "DATASETS_HOME", "HF_CACHE", "HUGPY_HOME",
    "HUGPY_VIDEO_STATE_DIR",
)


def _truthy(val) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _capture_live_roots(base: str) -> list[str]:
    """The storage roots this process WOULD have used without isolation — the
    inherited/``.env``-resolved DEFAULT_ROOT and siblings, plus the historical
    ``/mnt/llm_storage`` mount ``hugpy_platform.app_dirs.models_root`` falls back
    to. Captured (realpath'd) BEFORE the env is overridden. Anything overlapping
    the private temp *base* is dropped so the guard never denies the base."""
    try:
        from hugpy_platform.platform_facade import env_value  # light; no constants
    except Exception:  # noqa: BLE001
        env_value = None  # type: ignore
    real_base = os.path.realpath(base)
    roots: set[str] = set()
    for var in _LIVE_ROOT_VARS:
        vals = [os.environ.get(var)]
        if env_value is not None:
            try:
                vals.append(env_value(var))   # picks up a .env-resolved value
            except Exception:  # noqa: BLE001
                pass
        for val in vals:
            if isinstance(val, str) and val.strip():
                roots.add(os.path.realpath(val.strip()))
    # The legacy mount constants/app_dirs use when DEFAULT_ROOT is unset.
    try:
        if os.path.isdir("/mnt/llm_storage"):
            roots.add(os.path.realpath("/mnt/llm_storage"))
    except OSError:
        pass
    # Never deny the private temp base (or an ancestor/descendant of it): after
    # isolation ALL of a test's storage lives there, and a captured value may
    # coincide with it (e.g. a HUGPY_TEST_STORAGE_BASE carried in from base_env).
    out = []
    for r in sorted(roots):
        if r == real_base or r.startswith(real_base + os.sep) or real_base.startswith(r + os.sep):
            continue
        out.append(r)
    return out


def isolate_storage_to_tmp(base: str | None = None) -> str | None:
    """Point every hugpy storage root at a private temp dir and return it.

    No-op returning ``None`` when ``HUGPY_TEST_LIVE_ROOT`` is truthy. Call this
    at the TOP of a package's ``conftest.py``, before importing
    ``hugpy_platform.constants`` (or any module that imports it), so the
    import-time resolution lands in the temp tree.
    """
    if _truthy(os.environ.get("HUGPY_TEST_LIVE_ROOT")):
        return None
    if base is None:
        base = os.environ.get("HUGPY_TEST_STORAGE_BASE")
    if not base:
        base = tempfile.mkdtemp(prefix="hugpy-test-storage-")

    # Capture the would-be LIVE roots BEFORE we overwrite the environment.
    global _LIVE_ROOTS
    _LIVE_ROOTS = _capture_live_roots(base)
    os.environ["HUGPY_TEST_STORAGE_BASE"] = base

    resolved: dict[str, str] = {}
    for var, sub in _ROOT_SUBPATHS.items():
        path = base if sub == "" else os.path.join(base, sub)
        os.environ[var] = path
        resolved[var] = path
        # create the directory (or the parent of a DB file)
        d = path if not sub.endswith(".db") and not sub.endswith(".sqlite") else os.path.dirname(path)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    _rebind_already_imported(resolved)
    return base


def _rebind_already_imported(resolved: dict[str, str]) -> None:
    """If storage modules were imported before isolation, repoint their frozen
    module-level globals so nothing keeps a live path captured at import."""
    consts = sys.modules.get("hugpy_platform.constants")
    if consts is not None:
        root = resolved["DEFAULT_ROOT"]
        for name, path in (
            ("DEFAULT_ROOT", root),
            ("MODELS_HOME", resolved["MODELS_HOME"]), ("MODELS_DIR", resolved["MODELS_HOME"]),
            ("UPLOADS_HOME", resolved["UPLOADS_HOME"]), ("CHAT_UPLOAD_DIR", resolved["UPLOADS_HOME"]),
            ("PROJECTS_HOME", resolved["PROJECTS_HOME"]), ("PROJECTS_DIR", resolved["PROJECTS_HOME"]),
            ("IDENTITIES_HOME", resolved["IDENTITIES_HOME"]), ("IDENTITIES_DIR", resolved["IDENTITIES_HOME"]),
            ("DATASETS_HOME", resolved["DATASETS_HOME"]), ("DATASETS_DIR", resolved["DATASETS_HOME"]),
        ):
            if hasattr(consts, name):
                setattr(consts, name, path)

    state = sys.modules.get("hugpy_video.state")
    if state is not None and hasattr(state, "reset_state"):
        try:
            state.reset_state()   # drop any override; re-resolve from the new env
        except Exception:
            pass

    bus = sys.modules.get("hugpy_video.intel.media_bus")
    if bus is not None and hasattr(bus, "DB_PATH"):
        # media_jobs.db derives from the (pinned) video state dir.
        bus.DB_PATH = os.path.join(resolved["HUGPY_VIDEO_STATE_DIR"], "media_jobs.db")
        if hasattr(bus, "_initialized"):
            bus._initialized = False


def _is_db_path(s: str) -> bool:
    low = s.lower()
    return low.endswith((".db", ".sqlite", ".sqlite3")) or ".db-" in low or ".sqlite-" in low \
        or low.endswith("-wal") or low.endswith("-shm") or low.endswith("-journal")


_GUARD_EVENTS = {
    "sqlite3.connect": (0,),
    "open": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "shutil.copyfile": (0, 1),
    "shutil.move": (0, 1),
}


def _norm_audit_arg(a):
    if isinstance(a, bytes):
        try:
            a = a.decode("utf-8", "replace")
        except Exception:
            return None
    if not isinstance(a, str):
        return None
    s = a
    if s.startswith("file:"):
        s = s[5:].split("?", 1)[0]   # sqlite URI -> plain path
    return s


def _under(rp: str, s: str, root: str) -> bool:
    """True iff path *rp* (its realpath) or the raw *s* is *root* or under it.
    Uses a separator-terminated prefix so ``/mnt/llm_storage`` never matches a
    sibling like ``/mnt/llm_storage2``."""
    r = root.rstrip(os.sep)
    return (rp == r or rp.startswith(r + os.sep)
            or s == r or s.startswith(r + os.sep))


def _make_guard_hook(live_roots):
    """Build (without installing) the audit hook that RAISES on a DB-shaped
    access resolving under one of the captured LIVE storage roots — a DENY list,
    NOT an allow list. Everything else (coverage data files, the pytest cache,
    the job work dir, system temp, site-packages) passes untouched. Separated so
    it is unit-testable without mutating the process's global audit-hook list."""
    roots = [os.path.realpath(r).rstrip(os.sep) for r in live_roots if r]

    def _hook(event, args):
        if not roots:
            return
        idxs = _GUARD_EVENTS.get(event)
        if not idxs:
            return
        for i in idxs:
            if i >= len(args):
                continue
            s = _norm_audit_arg(args[i])
            if not s:
                continue
            if not (event == "sqlite3.connect" or _is_db_path(s)):
                continue
            rp = os.path.realpath(s)
            for r in roots:
                if _under(rp, s, r):
                    raise RuntimeError(
                        "live-storage guard: refused DB access under live storage "
                        f"root {r!r} — event={event} path={s}. A test resolved a "
                        "storage root to live storage; call "
                        "hugpy_platform.test_isolation.isolate_storage_to_tmp() "
                        "first, or set HUGPY_TEST_LIVE_ROOT=1 for a deliberate "
                        "live run.")

    return _hook


def install_live_storage_guard(live_roots=None) -> None:
    """Install a process-wide audit hook that RAISES on any DB-shaped filesystem
    access resolving under a captured LIVE storage root (a DENY list). Call
    :func:`isolate_storage_to_tmp` first — it captures the roots. Idempotent; a
    no-op when no live roots were captured (e.g. the HUGPY_TEST_LIVE_ROOT opt-out)."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    roots = list(_LIVE_ROOTS if live_roots is None else live_roots)
    if not roots:
        return
    _GUARD_INSTALLED = True
    sys.addaudithook(_make_guard_hook(roots))
