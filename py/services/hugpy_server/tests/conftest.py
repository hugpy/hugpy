"""Test session guard and fixtures for hugpy_server.

* The retired monolith must never satisfy an import from these tests
  (``HUGPY_ALLOW_MONOLITH=1`` lifts the block for transitional runs).
* Server-owned state (API keys, share keys, Discord bindings, install links)
  and the hugpy home go to a per-session temp dir — a test must never write
  into an operator's live state.
* Per-process daemons (the video job worker) are never started by tests.
* Auth mode defaults to ``open`` (the ``hugpy-serve`` default).
* Every provider registry (``hugpy_engine.placement``, ``hugpy_oracle.providers``,
  ``hugpy_storage.providers``, ``hugpy_video.hooks``, ``hugpy_curation.providers``)
  is snapshotted around each test so a fake installed by one test can never
  leak into the next.
* ``tests/integration`` is importable (``worker_store_isolation``).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if not os.environ.get("HUGPY_ALLOW_MONOLITH"):
    sys.modules.setdefault("abstract_hugpy_dev", None)

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "integration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SESSION_TMP = tempfile.mkdtemp(prefix="hugpy-server-tests-")
os.environ.setdefault("HUGPY_START_DAEMONS", "0")
# The distribution default (what `hugpy-serve` sets): single-operator, no login
# wall. Tests of the external gates set HUGPY_AUTH_MODE explicitly.
os.environ.setdefault("HUGPY_AUTH_MODE", "open")
os.environ.setdefault("HUGPY_SERVER_STATE_DIR", os.path.join(_SESSION_TMP, "server_state"))
os.environ.setdefault("HUGPY_HOME", os.path.join(_SESSION_TMP, "home"))
# Storage roots: hugpy_platform.constants derives MODELS/UPLOADS/PROJECTS/... from
# DEFAULT_ROOT at import time, and the control settings store (blocklist, model
# groups, ...) from HUGPY_SETTINGS_PATH. Both go to the session temp dir so no
# test can read or write an operator's live store. HUGPY_TEST_LIVE_ROOT=1 opts
# out (manual runs against a real catalog).
if not os.environ.get("HUGPY_TEST_LIVE_ROOT"):
    _root = os.path.join(_SESSION_TMP, "llm_storage")
    os.environ.setdefault("DEFAULT_ROOT", _root)
    os.environ.setdefault("HUGPY_SETTINGS_PATH", os.path.join(_root, "projects", "settings.json"))
    for _sub in ("models", "uploads", "projects", "identities", "datasets",
                 os.path.join("video_intel", "_scratch"),
                 os.path.join("video_intel", "studio")):
        os.makedirs(os.path.join(_root, _sub), exist_ok=True)
os.environ.setdefault("ORACLE_LEDGER_PATH", os.path.join(_SESSION_TMP, "test-reliability.sqlite"))
os.makedirs(os.environ["HUGPY_SERVER_STATE_DIR"], exist_ok=True)

import pytest  # noqa: E402

_PROVIDER_MODULES = (
    "hugpy_engine.placement",
    "hugpy_oracle.providers",
    "hugpy_storage.providers",
    "hugpy_video.hooks",
    "hugpy_curation.providers",
)


def _snapshot(mod):
    out = {}
    for k, v in vars(mod).items():
        if not k.startswith("_") or k.startswith("__"):
            continue
        if callable(v) and not isinstance(v, (dict, list, set)):
            continue
        if isinstance(v, dict):
            out[k] = ("dict", dict(v))
        elif isinstance(v, list):
            out[k] = ("list", list(v))
        elif isinstance(v, set):
            out[k] = ("set", set(v))
        else:
            out[k] = ("val", v)
    return out


def _restore(mod, snap):
    for k, (kind, v) in snap.items():
        cur = getattr(mod, k, None)
        if kind == "dict" and isinstance(cur, dict):
            cur.clear(); cur.update(v)
        elif kind == "list" and isinstance(cur, list):
            cur[:] = v
        elif kind == "set" and isinstance(cur, set):
            cur.clear(); cur.update(v)
        else:
            setattr(mod, k, v)


# ── collection-time isolation ────────────────────────────────────────────────
# Many modules moved in from the monolith rebind shared globals at IMPORT time
# (``media_bus.DB_PATH = <tmp>``, ``identity_profiles.IDENTITIES_HOME = <tmp>``).
# pytest imports every module during collection, before any test runs, so the
# LAST module collected used to win and every other module ran against its temp
# DB / identity store (or a fake ``media_bus.enqueue``). Record what each
# module's import rebound on the tracked modules, undo it right after collecting
# that module, and re-apply it only while that module's tests run
# (``_module_import_state``). The same is done for the few environment
# variables that point a module at its OWN fake identity service
# (``_COLLECTION_TRACKED_ENV``). Other import-time environment is NOT isolated:
# several studio modules rely on another module's ``STUDIO_ALLOW_UNPINNED`` /
# ``PROJECTS_HOME`` setdefault, as they did in the monolith.
_COLLECTION_TRACKED = (
    "hugpy_video.intel.media_bus",
    "hugpy_video.intel.identity_profiles",
)
_COLLECTION_TRACKED_ENV = (
    "IDENTITY_RENDER_URL",
    "IDENTITY_RENDER_TOKEN",
    "HUGPY_CENTRAL_URL",
)
_MODULE_IMPORT_STATE: dict = {}


def _tracked_modules():
    import importlib
    mods = []
    for name in _COLLECTION_TRACKED:
        try:
            mods.append(importlib.import_module(name))
        except Exception:  # noqa: BLE001 - optional package absent
            pass
    return mods


def _globals_snapshot(mods):
    return {m.__name__: dict(vars(m)) for m in mods}


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if not isinstance(collector, pytest.Module):
        yield
        return
    mods = _tracked_modules()
    glob_before = _globals_snapshot(mods)
    env_before = {k: os.environ.get(k) for k in _COLLECTION_TRACKED_ENV}
    yield
    env_delta = {}
    for k, old in env_before.items():
        new = os.environ.get(k)
        if new != old:
            env_delta[k] = new
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    glob_delta = []
    for m in mods:
        before = glob_before[m.__name__]
        for k, v in list(vars(m).items()):
            if k.startswith("__"):
                continue
            if k not in before or before[k] is not v:
                glob_delta.append((m, k, v))
                if k in before:
                    setattr(m, k, before[k])
                else:
                    delattr(m, k)
    if glob_delta or env_delta:
        _MODULE_IMPORT_STATE[os.path.realpath(str(collector.path))] = (glob_delta, env_delta)


@pytest.fixture(autouse=True, scope="module")
def _module_import_state(request):
    """Re-apply, for this module only, what importing it changed."""
    state = _MODULE_IMPORT_STATE.get(
        os.path.realpath(str(getattr(request.module, "__file__", "") or "")))
    if not state:
        yield
        return
    glob_delta, env_delta = state
    glob_saved = [(m, k, getattr(m, k, _MISSING)) for m, k, _v in glob_delta]
    env_saved = {k: os.environ.get(k) for k in env_delta}
    for m, k, v in glob_delta:
        setattr(m, k, v)
    _reinit_media_bus(glob_delta)
    _set_env(env_delta)
    yield
    _set_env(env_saved)
    for m, k, v in glob_saved:
        if v is _MISSING:
            if hasattr(m, k):
                delattr(m, k)
        else:
            setattr(m, k, v)
    _reinit_media_bus(glob_delta)


def _set_env(values):
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _reinit_media_bus(glob_delta):
    """A repointed ``media_bus.DB_PATH`` must be (re)initialised — schema and
    migrations — on first use. ``_initialized`` itself may not show up in the
    delta (False before and after the import), so reset it explicitly."""
    for m, _k, _v in glob_delta:
        if m.__name__ == "hugpy_video.intel.media_bus":
            m._initialized = False
            return


_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_environ():
    """Tests that mutate ``os.environ`` directly (auth mode, operator token,
    feature flags) must not leak into the next test."""
    before = dict(os.environ)
    yield
    for k in list(os.environ):
        if k not in before:
            del os.environ[k]
    for k, v in before.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
    try:
        from hugpy_server.app import operator_auth as _oa
        cache = getattr(_oa, "_SESSION_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(autouse=True)
def _isolate_providers():
    """Snapshot/restore every provider registry around each test."""
    snaps = {}
    for name in _PROVIDER_MODULES:
        mod = sys.modules.get(name)
        if mod is not None:
            snaps[name] = _snapshot(mod)
    yield
    for name, snap in snaps.items():
        mod = sys.modules.get(name)
        if mod is not None:
            _restore(mod, snap)


@pytest.fixture
def server_app():
    """A fresh hugpy Flask app (no daemons, fully wired) for route contract tests."""
    from hugpy_server.wsgi_app import create_app
    app = create_app(name=f"hugpy-test-{os.getpid()}", start_daemons=False)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(server_app):
    return server_app.test_client()


@pytest.fixture(autouse=True)
def _cwd_outside_package(tmp_path, monkeypatch):
    """abstract_flask attaches ``logging.FileHandler("<app name>.log")`` in the
    current directory when an app is built. Run every test from a scratch
    directory so those files never land in the package tree."""
    monkeypatch.chdir(tmp_path)
