"""The idle-STOP path of the comfy watchdog (2026-09-24) — the worker owns
comfy's lifecycle, so a fully idle MANAGED comfy is STOPPED after a longer
window than the /free debounce, releasing even the bare CUDA context.

Guards:
  * only a worker-MANAGED launcher stops (external is never stopped);
  * only when comfy is RUNNING (nothing to stop otherwise);
  * NEVER mid-render — an in-flight registered call or a non-empty /queue (or an
    unreadable one) resets the clock and skips;
  * the stop persists past HUGPY_COMFY_IDLE_STOP_S (a persistence test, not a
    stopwatch), with the clock separate from the VRAM-gated /free clock;
  * a stop that fails is surfaced, never escalated.
"""
import pytest

from hugpy_fleet.worker import comfy_watchdog as cw
from hugpy_fleet.worker.comfy_watchdog import ComfyIdleWatchdog, UNKNOWN

IDLE_QUEUE = {"running": 0, "pending": 0}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def build(queue=IDLE_QUEUE, call=None, stop=(True, "stopped"), clock=None):
    state = {"stops": 0}

    def stop_call():
        state["stops"] += 1
        return stop

    wd = ComfyIdleWatchdog(
        vram_probe=lambda fresh=False: 0,          # unused by stop_tick
        url_probe=lambda: "http://x:8188",
        free_call=lambda: (True, "ok"),            # unused by stop_tick
        queue_probe=lambda url: queue,
        call_probe=lambda: call,
        clock=clock or Clock(),
        sleep=lambda s: None,
        stop_call=stop_call,
    )
    return wd, state


def test_external_launcher_never_stops():
    wd, state = build()
    res = wd.stop_tick(managed=False, running=True)
    assert res["action"] == "skip"
    assert "not worker-managed" in res["reason"]
    assert state["stops"] == 0


def test_not_running_is_a_silent_skip():
    wd, state = build()
    res = wd.stop_tick(managed=True, running=False)
    assert res["action"] == "skip"
    assert "not running" in res["reason"]
    assert state["stops"] == 0


def test_in_flight_call_is_never_stopped():
    wd, state = build(call={"model_key": "sdxl"})
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "skip"
    assert "in flight" in res["reason"]
    assert state["stops"] == 0


def test_non_empty_queue_is_never_stopped():
    wd, state = build(queue={"running": 1, "pending": 0})
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "skip"
    assert "queue busy" in res["reason"]
    assert state["stops"] == 0


def test_unreadable_queue_reads_as_busy():
    wd, state = build(queue=None)
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "skip"
    assert "unreadable" in res["reason"]
    assert state["stops"] == 0


def test_unreadable_call_table_reads_as_busy():
    wd, state = build(call=UNKNOWN)
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "skip"
    assert state["stops"] == 0


def test_idle_within_window_waits_then_stops_after_window(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_STOP_S", "900")
    clock = Clock()
    wd, state = build(clock=clock)
    # first idle beat: clock starts, well within the window
    r1 = wd.stop_tick(managed=True, running=True)
    assert r1["action"] == "wait"
    assert state["stops"] == 0
    # still within the window
    clock.advance(300)
    assert wd.stop_tick(managed=True, running=True)["action"] == "wait"
    # past the window -> STOP
    clock.advance(601)
    r3 = wd.stop_tick(managed=True, running=True)
    assert r3["action"] == "stopped"
    assert state["stops"] == 1


def test_activity_resets_the_idle_stop_clock(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_STOP_S", "900")
    clock = Clock()
    wd, state = build(clock=clock)
    wd.stop_tick(managed=True, running=True)         # clock started
    clock.advance(800)
    # a render appears: clock must reset, not fire
    wd._queue_probe = lambda url: {"running": 1, "pending": 0}
    assert wd.stop_tick(managed=True, running=True)["action"] == "skip"
    # queue empties again; the window restarts from here
    wd._queue_probe = lambda url: IDLE_QUEUE
    clock.advance(901)
    # only 901s since the RESET, but the clock was cleared so this beat re-arms
    r = wd.stop_tick(managed=True, running=True)
    assert r["action"] == "wait"          # re-armed, not immediately fired
    assert state["stops"] == 0


def test_stop_failure_is_surfaced_not_escalated(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_STOP_S", "0")   # fire immediately
    wd, state = build(stop=(False, "systemctl stop returned 5"))
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "failed"
    assert "returned 5" in res["reason"]
    assert state["stops"] == 1


def test_disabled_switch_is_a_noop(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_STOP", "0")
    wd, state = build()
    res = wd.stop_tick(managed=True, running=True)
    assert res["action"] == "skip"
    assert "disabled" in res["reason"]
    assert state["stops"] == 0
