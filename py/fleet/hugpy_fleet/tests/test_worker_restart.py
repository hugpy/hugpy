"""Worker restart mechanism — execv-under-systemd fix (CODE_GAPS 2026-07-12 #3).

Two real incidents (computron restart-loop 160→219; op's 403 dueling-worker
saga) traced to os.execv under systemd: the exec'd image kept squatting :9100
while systemd respawned a colliding fresh process ("Address already in use").
The fix: UNDER SYSTEMD, never execv — drain, release resources, and EXIT with a
distinct non-zero code so systemd's Restart= respawns exactly one tracked
process. STANDALONE keeps execv.

Covers:
  * _under_systemd(): both branches + the descendant-leak immunity (INVOCATION_ID
    inherited from an ancestor service must NOT read as "this is a service").
  * _prepare_restart() exit-path: the clean-shutdown steps (flag set, drained,
    executors shut down, slot children killed, listening socket closed) run and a
    distinct exit code is planned — WITHOUT terminating this process (the seam is
    _restart, tested by exit code in a subprocess).
  * Exit-code convention: a real forced-systemd _restart exits _RESTART_EXIT_CODE.
  * Standalone dispatch: _restart execs in place (calls the resolved reexec_fn).
  * Bounded drain: honors in-flight generations up to the timeout, never hangs.
  * Executor shutdown: registered executors are shut down (cancel_futures), and a
    shutdown that raises is swallowed — no 'cannot schedule new futures' spam.
  * Settings survive a simulated fresh boot: _apply_settings_env projects over a
    clean env in both lifecycles.

Runs like the other tests here: venv/bin/python tests/test_worker_restart.py
"""
import logging
logging.disable(logging.CRITICAL)

import argparse
import os
import subprocess
import sys
import tempfile
import time
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


# Snapshot every global/env we mutate so the file leaves no residue.
_ENV_KEYS = ["HUGPY_WORKER_SYSTEMD", "COMFY_URL", agent._ENV_HOT_CACHE_ROOT,
             agent._COMFY_URL_BASE_ENV, agent._HOT_CACHE_ROOT_BASE_ENV]
_ENV_BACKUP = {k: os.environ.get(k) for k in _ENV_KEYS}
_SETTINGS_BACKUP = dict(agent._RUNTIME_SETTINGS)
_SOURCE_BACKUP = dict(agent._SETTINGS_SOURCE)
_SLOT_BACKUP = dict(agent._SLOT_PROCS)
_TIF_ORIG = gen_gate.total_in_flight


def _restore():
    for k, v in _ENV_BACKUP.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    agent._RUNTIME_SETTINGS.clear(); agent._RUNTIME_SETTINGS.update(_SETTINGS_BACKUP)
    agent._SETTINGS_SOURCE.clear(); agent._SETTINGS_SOURCE.update(_SOURCE_BACKUP)
    agent._SLOT_PROCS.clear(); agent._SLOT_PROCS.update(_SLOT_BACKUP)
    agent._RESTART_EVENT.clear()
    gen_gate.total_in_flight = _TIF_ORIG


try:
    # ─────────────────────────────────────────────────────────────────────────
    # 1. _under_systemd(): both branches + the descendant-leak immunity.
    # ─────────────────────────────────────────────────────────────────────────
    os.environ["HUGPY_WORKER_SYSTEMD"] = "1"
    check("explicit override =1 -> under systemd", agent._under_systemd() is True)
    os.environ["HUGPY_WORKER_SYSTEMD"] = "0"
    check("explicit override =0 -> standalone", agent._under_systemd() is False)
    os.environ.pop("HUGPY_WORKER_SYSTEMD", None)

    # Leak immunity: this test process is a DESCENDANT of a systemd service, so
    # INVOCATION_ID is inherited — but our parent is a shell/test-runner, NOT the
    # systemd manager, so we must read FALSE (never os._exit out from under a run).
    check("parent is not the systemd manager here", agent._parent_is_systemd() is False)
    check("inherited INVOCATION_ID does NOT falsely read as a service unit "
          "(descendant-leak immunity)", agent._under_systemd() is False)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Bounded drain: honors in-flight gens up to the timeout, never hangs.
    # ─────────────────────────────────────────────────────────────────────────
    gen_gate.total_in_flight = lambda: 0
    t0 = time.monotonic()
    waited = agent._drain_generations(5.0)
    check("no in-flight gens -> drain returns ~immediately",
          (time.monotonic() - t0) < 1.0 and waited < 1.0)

    gen_gate.total_in_flight = lambda: 2          # never drains
    t0 = time.monotonic()
    waited = agent._drain_generations(0.5)        # bounded
    dt = time.monotonic() - t0
    check("stuck in-flight gens -> drain returns at the bound (never hangs)",
          0.4 <= dt < 3.0 and 0.4 <= waited < 3.0)
    gen_gate.total_in_flight = _TIF_ORIG

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Executor shutdown: registered executors are shut down; a raising
    #    shutdown is swallowed (no 'cannot schedule new futures' escaping).
    # ─────────────────────────────────────────────────────────────────────────
    agent._RESTART_EVENT.clear()

    class _FakeExec:
        def __init__(self, raise_on_shutdown=False):
            self.shutdown_calls = []
            self._raise = raise_on_shutdown
        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))
            if self._raise:
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    good_ex = _FakeExec()
    bad_ex = _FakeExec(raise_on_shutdown=True)
    agent.register_executor(good_ex)
    agent.register_executor(bad_ex)
    agent._shutdown_executors()                   # must not raise
    check("registered executor was shut down", len(good_ex.shutdown_calls) == 1)
    check("executor shutdown requested no-wait + cancel_futures",
          good_ex.shutdown_calls[0] == (False, True))
    check("a raising executor shutdown is swallowed (no spam escapes)",
          len(bad_ex.shutdown_calls) == 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. _prepare_restart() EXIT path: performs the clean-shutdown steps and
    #    PLANS a distinct exit code, WITHOUT exiting (the seam is _restart).
    # ─────────────────────────────────────────────────────────────────────────
    agent._RESTART_EVENT.clear()
    agent._ACTIVE_EXECUTORS.clear()
    agent._SLOT_PROCS.clear()
    gen_gate.total_in_flight = lambda: 0

    class _FakeServer:
        def __init__(self): self.closed = False
        def server_close(self): self.closed = True

    class _FakeProc:
        def __init__(self): self.terminated = False
        def poll(self): return None               # alive
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 0

    fake_srv = _FakeServer()
    fake_proc = _FakeProc()
    fake_exec2 = _FakeExec()
    agent._SLOT_PROCS[1] = fake_proc
    agent.register_executor(fake_exec2)
    st = agent.WorkerState(name="t", url=None, worker_id="w-exit")
    st.http_server = fake_srv

    plan = agent._prepare_restart(st, reason="unit-test", mode="exit",
                                  kill_slots=False)
    check("exit plan carries the distinct restart exit code",
          plan["exit_code"] == agent._RESTART_EXIT_CODE and plan["exit_code"] != 0)
    check("exit plan mode is 'exit'", plan["mode"] == "exit")
    check("restart flag was set (loops stop scheduling new work)",
          agent.restart_requested() is True)
    check("in-flight was drained (step recorded)", "drained" in plan["steps"])
    check("executors were shut down", len(fake_exec2.shutdown_calls) == 1)
    check("slot children were killed under exit (systemd tears the cgroup anyway)",
          fake_proc.terminated is True and 1 not in agent._SLOT_PROCS)
    check("listening socket was closed (releases :9100)",
          fake_srv.closed is True and plan.get("socket_closed") is True)
    check("all clean-shutdown steps present in order",
          plan["steps"] == ["shutdown_flag", "drained", "executors", "slots", "socket"])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. _prepare_restart() EXECV path: NO socket close, slots only when asked
    #    (standalone re-exec adopts live slots to avoid a blip, as today).
    # ─────────────────────────────────────────────────────────────────────────
    agent._RESTART_EVENT.clear()
    agent._SLOT_PROCS.clear()
    fake_proc2 = _FakeProc()
    fake_srv2 = _FakeServer()
    agent._SLOT_PROCS[1] = fake_proc2
    st2 = agent.WorkerState(name="t", url=None, worker_id="w-execv")
    st2.http_server = fake_srv2
    plan = agent._prepare_restart(st2, reason="unit-test", mode="execv",
                                  kill_slots=False)
    check("execv plan has no exit code", plan["exit_code"] is None)
    check("execv path does NOT close the socket (CLOEXEC drops it on exec)",
          fake_srv2.closed is False and "socket" not in plan["steps"])
    check("execv path adopts slots when not asked to kill (no blip)",
          fake_proc2.terminated is False and "slots" not in plan["steps"])

    # ... but self-update (kill_slots=True) DOES tear slots down even on execv.
    agent._RESTART_EVENT.clear()
    agent._SLOT_PROCS.clear()
    fake_proc3 = _FakeProc()
    agent._SLOT_PROCS[1] = fake_proc3
    st3 = agent.WorkerState(name="t", url=None, worker_id="w-execv2")
    plan = agent._prepare_restart(st3, reason="self-update", mode="execv",
                                  kill_slots=True)
    check("execv + kill_slots (self-update) tears slots down (no stale-code slot)",
          fake_proc3.terminated is True and "slots" in plan["steps"])

    # ─────────────────────────────────────────────────────────────────────────
    # 6. _restart() STANDALONE dispatch: execs in place (calls reexec_fn), does
    #    NOT exit. Forced via the override so it's deterministic here.
    # ─────────────────────────────────────────────────────────────────────────
    agent._RESTART_EVENT.clear()
    agent._SLOT_PROCS.clear()
    gen_gate.total_in_flight = lambda: 0
    os.environ["HUGPY_WORKER_SYSTEMD"] = "0"
    called = {"n": 0}
    def _fake_reexec():
        called["n"] += 1
    st4 = agent.WorkerState(name="t", url=None, worker_id="w-r")
    agent._restart(st4, reason="unit-test", reexec_fn=_fake_reexec)
    check("standalone _restart calls the resolved reexec_fn (execv path)",
          called["n"] == 1)
    os.environ.pop("HUGPY_WORKER_SYSTEMD", None)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Exit-code convention END-TO-END: a forced-systemd _restart os._exit's
    #    with the distinct code. Run in a subprocess so it can actually exit.
    # ─────────────────────────────────────────────────────────────────────────
    driver = (
        "import sys; sys.path.insert(0, %r)\n"
        "import logging; logging.disable(logging.CRITICAL)\n"
        "import os\n"
        "os.environ['HUGPY_WORKER_SYSTEMD'] = '1'\n"
        "from hugpy_fleet.worker import agent, gen_gate\n"
        "gen_gate.total_in_flight = lambda: 0\n"
        "st = agent.WorkerState(name='t', url=None, worker_id='w-sub')\n"
        "agent._restart(st, reason='sub', reexec_fn=lambda: None)\n"
        "print('UNREACHABLE'); sys.exit(7)\n"
    ) % str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run([sys.executable, "-c", driver],
                          capture_output=True, text=True, timeout=60)
    check("forced-systemd _restart exits the distinct restart code (not execv, "
          "not 0, not 7)", proc.returncode == agent._RESTART_EXIT_CODE)
    check("systemd restart path never reached the standalone/execv fallthrough",
          "UNREACHABLE" not in proc.stdout)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Settings survive a simulated fresh boot (both lifecycles): the
    #    _apply_settings_env projection still lands over a clean env.
    # ─────────────────────────────────────────────────────────────────────────
    tmp = tempfile.mkdtemp(prefix="hugpy-restart-")
    args = argparse.Namespace(id_file=os.path.join(tmp, "agent.id"))
    # SYSTEMD lifecycle: fresh process, clean env (no sentinel, no projected env).
    for k in (agent._COMFY_URL_BASE_ENV, agent._HOT_CACHE_ROOT_BASE_ENV,
              "COMFY_URL", agent._ENV_HOT_CACHE_ROOT):
        os.environ.pop(k, None)
    hot = os.path.join(tmp, "hot")
    agent._save_settings(args, {"comfy_url": "http://comfy.example.ai",
                                "hot_cache_root": hot})
    agent._apply_settings_env(args)
    eff = agent._effective_config()
    check("fresh-boot projection lands comfy_url over the clean unit env",
          os.environ.get("COMFY_URL") == "http://comfy.example.ai")
    check("fresh-boot projection lands hot_cache_root over the clean unit env",
          os.environ.get(agent._ENV_HOT_CACHE_ROOT) == hot)
    check("effective config reports both as source=settings",
          eff.get("comfy_url_source") == "settings"
          and eff.get("hot_cache_root_source") == "settings")

    # STANDALONE lifecycle: the sentinel (captured on the previous boot) SURVIVES
    # the execv, so a later CLEAR reverts to the true base, never the projection.
    # Here the base was empty (clean), so clearing reverts to unset/default.
    agent._save_settings(args, {})
    agent._apply_settings_env(args)               # sentinel still in os.environ
    eff = agent._effective_config()
    check("clear across the (surviving) sentinel reverts comfy_url to base/default",
          os.environ.get("COMFY_URL") is None and eff.get("comfy_url_source") == "default")
    check("clear reverts hot_cache_root to base/default too",
          os.environ.get(agent._ENV_HOT_CACHE_ROOT) is None
          and eff.get("hot_cache_root_source") == "default")

finally:
    _restore()

print(f"\nall {ok} checks passed")

# Re-enable logging for the rest of the pytest session (the disable above is import-time only).
logging.disable(logging.NOTSET)
