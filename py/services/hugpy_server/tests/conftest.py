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
                 os.path.join("video_intel", "_scratch")):
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
