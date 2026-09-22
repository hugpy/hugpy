"""Worker version-reporting honesty + re-exec failure survival (k31).

The 2026-07-20 ae incident: after the 0.1.196 self-update, ae's heartbeat
reported pkg_version 0.1.196 / version_ok:true while the RUNNING agent still
served the OLD route set (404 on the new /slots/<id>/relaunch). Root cause: the
heartbeat sourced its version from live DISK metadata (importlib.metadata, which
pip had already flipped) instead of the code actually running — cosmetic
convergence. The fix reports the RUNNING image version (a source-file literal
snapshotted at import), so a not-yet-effective pip upgrade shows as a version
SKEW, never a green lie.

Covers:
  * _running_pkg_version(): reports the import-time running image, and does NOT
    move when disk metadata is bumped under it (simulated `pip install` of a new
    version while this process still runs the old code).
  * _installed_pkg_version() (disk): DOES move with the simulated upgrade — the
    exact divergence that used to leak into the heartbeat.
  * copied-file fallback: with no package __version__ to trust, _running falls
    back to disk metadata.
  * re-exec hardening: a standalone _restart whose reexec_fn RAISES does not
    propagate (no bubbling into the heartbeat's silent swallow); it logs loudly
    (logger.error) and returns, leaving the old image running — now honest.

Runs like the neighbours: venv/bin/python tests/test_worker_version_honesty.py
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent
from hugpy_fleet.worker import gen_gate

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


_ENV_BACKUP = {k: os.environ.get(k) for k in ("HUGPY_WORKER_SYSTEMD",)}
_RUNNING_BACKUP = agent._RUNNING_IMAGE_VERSION
_INSTALLED_ORIG = agent._installed_pkg_version
_TIF_ORIG = gen_gate.total_in_flight
_LOGGER_ERROR_ORIG = agent.logger.error


def _restore():
    for k, v in _ENV_BACKUP.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    agent._RUNNING_IMAGE_VERSION = _RUNNING_BACKUP
    agent._installed_pkg_version = _INSTALLED_ORIG
    gen_gate.total_in_flight = _TIF_ORIG
    agent.logger.error = _LOGGER_ERROR_ORIG
    agent._RESTART_EVENT.clear()


try:
    # ─────────────────────────────────────────────────────────────────────────
    # 1. HONESTY: _running_pkg_version reports the import-time running image and
    #    is IMMUNE to a disk-metadata bump (simulated pip upgrade under us).
    # ─────────────────────────────────────────────────────────────────────────
    running_at_import = agent._RUNNING_IMAGE_VERSION
    check("running image version was snapshotted at import (a real x.y.z, not None)",
          isinstance(running_at_import, str) and running_at_import.count(".") >= 2)

    # Baseline: with no upgrade, both sources agree.
    agent._installed_pkg_version = lambda name: running_at_import
    check("no upgrade -> running == disk (both truthful)",
          agent._running_pkg_version("hugpy-fleet") == running_at_import)

    # Simulate `pip install abstract_hugpy_dev==9.9.9` landing on disk while THIS
    # process keeps running the old code (exactly the ae cosmetic path).
    NEW = "9.9.999"
    agent._installed_pkg_version = lambda name: NEW
    check("simulated upgrade: DISK metadata moved to the new version",
          agent._installed_pkg_version("hugpy-fleet") == NEW)
    check("HONEST: running image version did NOT move (still the old code)",
          agent._running_pkg_version("hugpy-fleet") == running_at_import
          and agent._running_pkg_version("hugpy-fleet") != NEW)
    # -> central compares this OLD value against required_pkg_version=NEW and keeps
    #    version_ok:false (a visible skew) instead of cosmetically converging.

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FALLBACK: no package __version__ to trust (a standalone copied agent.py)
    #    -> _running falls back to disk metadata (best available signal).
    # ─────────────────────────────────────────────────────────────────────────
    agent._RUNNING_IMAGE_VERSION = None
    agent._installed_pkg_version = lambda name: NEW
    check("copied-file fallback: no __version__ -> report disk metadata",
          agent._running_pkg_version("hugpy-fleet") == NEW)
    agent._RUNNING_IMAGE_VERSION = running_at_import
    agent._installed_pkg_version = _INSTALLED_ORIG

    # ─────────────────────────────────────────────────────────────────────────
    # 3. RE-EXEC HARDENING: a standalone _restart whose reexec_fn RAISES must NOT
    #    propagate (no bubbling into the heartbeat's silent swallow); it logs
    #    loudly and returns, leaving the (now honest) old image running.
    # ─────────────────────────────────────────────────────────────────────────
    os.environ["HUGPY_WORKER_SYSTEMD"] = "0"      # force the standalone/execv branch
    gen_gate.total_in_flight = lambda: 0
    agent._RESTART_EVENT.clear()

    errors = []
    agent.logger.error = lambda *a, **k: errors.append(a)

    def _boom_reexec():
        raise OSError(8, "Exec format error")     # a real os.execv-style failure

    st = agent.WorkerState(name="t", url=None, worker_id="w-reexec-fail")
    returned = True
    try:
        agent._restart(st, reason="self-update", reexec_fn=_boom_reexec)
    except Exception as exc:  # noqa: BLE001
        returned = False
        raise AssertionError(f"re-exec failure must not propagate, got {exc!r}")
    check("raising re-exec does NOT propagate (never reaches the silent swallow)",
          returned is True)
    check("re-exec failure was logged LOUDLY (logger.error fired)", len(errors) == 1)
    check("the loud log names the failure as a re-exec that did not replace the image",
          any("RE-EXEC FAILED" in str(a[0]) for a in errors))
    agent.logger.error = _LOGGER_ERROR_ORIG

    # Control: a well-behaved (no-op) reexec_fn returns cleanly, no error logged.
    errors2 = []
    agent.logger.error = lambda *a, **k: errors2.append(a)
    agent._RESTART_EVENT.clear()
    called = {"n": 0}
    st2 = agent.WorkerState(name="t", url=None, worker_id="w-reexec-ok")
    agent._restart(st2, reason="self-update", reexec_fn=lambda: called.__setitem__("n", 1))
    check("no-op reexec_fn: _restart calls it and returns without an error log",
          called["n"] == 1 and len(errors2) == 0)

finally:
    _restore()

print(f"\nall {ok} checks passed")
