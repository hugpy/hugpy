"""The gunicorn glue must build the Flask app INSIDE the worker (load()), never
in the arbiter/master.

Incident 2026-09-24: ``hugpy serve`` built the app in the process that becomes
the gunicorn arbiter, then forked one worker that inherited the pre-opened
media_jobs.db WAL/SHM handles (fds marked "(deleted)") -> "database is locked" /
"disk I/O error". The fix makes ``cli._serve``/``_run_wsgi`` pass a zero-arg
factory THUNK that gunicorn calls from ``load()`` (post-fork). This test pins
that contract: the factory is not called until load(), and then exactly once.
"""
from __future__ import annotations

import sys
import types

from hugpy.cli import _run_wsgi


class _Factory:
    """A counting build-app thunk; returns a sentinel app object."""

    def __init__(self):
        self.calls = 0
        self.app = object()
        self.count_before_load = None
        self.count_after_load = None
        self.loaded_app = None

    def __call__(self):
        self.calls += 1
        return self.app


def _install_fake_gunicorn(monkeypatch, factory):
    """Inject a fake ``gunicorn.app.base.BaseApplication`` that mimics gunicorn's
    own lifecycle: ``__init__`` runs ``load_config()`` (like the real arbiter)
    but NEVER ``load()``; ``run()`` is where the worker would live, so THAT is
    the only place ``load()`` — and thus the factory — may be called."""

    class _FakeCfg:
        def set(self, key, value):  # gunicorn cfg.set — a no-op for the test
            pass

    class _FakeBaseApplication:
        def __init__(self):
            self.cfg = _FakeCfg()
            # The real Application.__init__ loads CONFIG (load_config), not the
            # app. If the factory were called here, that would be pre-fork.
            self.load_config()

        def run(self):
            # gunicorn calls load() only in the worker, after the fork. Record on
            # the factory (the subclass _App is what runs, so class attrs on the
            # base would be shadowed).
            factory.count_before_load = factory.calls
            factory.loaded_app = self.load()
            factory.count_after_load = factory.calls

    base_mod = types.ModuleType("gunicorn.app.base")
    base_mod.BaseApplication = _FakeBaseApplication
    app_mod = types.ModuleType("gunicorn.app")
    app_mod.base = base_mod
    root_mod = types.ModuleType("gunicorn")
    root_mod.app = app_mod
    monkeypatch.setitem(sys.modules, "gunicorn", root_mod)
    monkeypatch.setitem(sys.modules, "gunicorn.app", app_mod)
    monkeypatch.setitem(sys.modules, "gunicorn.app.base", base_mod)
    return _FakeBaseApplication


def test_gunicorn_app_defers_factory_until_load(monkeypatch):
    factory = _Factory()
    fake = _install_fake_gunicorn(monkeypatch, factory)

    rc = _run_wsgi(factory, host="127.0.0.1", port=0, threads=1, debug=False)

    assert rc == 0
    # Nothing built the app before load() ran (i.e. not in the arbiter):
    assert factory.count_before_load == 0
    # load() built it exactly once, and that is the app gunicorn will serve:
    assert factory.count_after_load == 1
    assert factory.calls == 1
    assert factory.loaded_app is factory.app
