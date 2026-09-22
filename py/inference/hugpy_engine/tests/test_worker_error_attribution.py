"""Slice 7: relayed worker failures must carry the worker's name+id.

The 2026-07-05 operator report: a chat/scene failure surfaced as raw cause
frames ("frame 0: ModuleNotFoundError: No module named 'torch'") with no hint
of WHICH worker it happened on. Root mechanism: the worker's dispatch plane
ships failures AS DATA ({ok: false, error: …}, HTTP 200), and the central
relay (_worker_run_once) validated + returned them anonymously. The fix
stamps "on worker <name> (<id>): " onto typed error results at the relay.

Runs like the other tests here:
    venv/bin/python tests/test_worker_error_attribution.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel

remote = importlib.import_module("hugpy_engine.resolvers.remote")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_worker_error_attribution():
    """Converted from the script-style suite: every `check` is an assert."""


    class _Result(BaseModel):
        ok: bool = True
        error: str | None = None
        text: str | None = None


    W = {"id": "a1f3c9d2", "name": "ae"}

    r = remote._stamp_worker_error(_Result(ok=False, error="ModuleNotFoundError: No module named 'torch'"), W)
    check("typed error result gets the worker stamp",
          r.error == "on worker ae (a1f3c9d2): ModuleNotFoundError: No module named 'torch'")

    r = remote._stamp_worker_error(_Result(ok=True, text="hi"), W)
    check("success results untouched", r.ok is True and r.error is None)

    r = remote._stamp_worker_error(
        _Result(ok=False, error="on worker ae (a1f3c9d2): already stamped"), W)
    check("no double-stamping", r.error.count("on worker") == 1)

    r = remote._stamp_worker_error(_Result(ok=False, error=None), W)
    check("errorless failure passes through unbroken", r.error is None)

    r = remote._stamp_worker_error(
        _Result(ok=False, error="boom"), {"id": "w9", "name": ""})
    check("nameless worker falls back to its id", r.error == "on worker w9: boom")

    # ok=True but error set (partial-failure shape some runners use) still stamps
    r = remote._stamp_worker_error(_Result(ok=True, error="soft fail"), W)
    check("ok=true + error text still attributed",
          r.error == "on worker ae (a1f3c9d2): soft fail")

    print(f"\nall {ok} checks passed")
