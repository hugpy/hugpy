"""ComfyUI stops on NEED, not on a timer (operator 2026-10-02): only when a load
needs the room AND no image work is registered/queued."""
from hugpy_fleet.worker import comfy_watchdog as cw
from hugpy_fleet.worker.comfy_watchdog import ComfyIdleWatchdog, UNKNOWN

MIB = 1 << 20


def build(queue=None, call=None, stop=(True, "stopped"), held=378 * MIB):
    state = {"stops": 0}

    def stop_call():
        state["stops"] += 1
        return stop
    wd = ComfyIdleWatchdog(vram_probe=lambda fresh=False: held, url_probe=lambda: "http://x",
                           free_call=lambda: (True, "ok"),
                           queue_probe=lambda url: ({"running": 0, "pending": 0} if queue is None else queue),
                           call_probe=lambda: call, sleep=lambda s: None, stop_call=stop_call)
    return wd, state


def test_timer_stop_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HUGPY_COMFY_IDLE_STOP", raising=False)
    assert cw.stop_enabled() is False
    wd, state = build()
    assert wd.stop_tick(managed=True, running=True)["action"] == "skip" and state["stops"] == 0


def test_idle_managed_comfy_stops_for_a_load_and_reports_its_bytes():
    wd, state = build()
    res = wd.stop_for_need(managed=True, running=True, incoming_model="M", need_bytes=1)
    assert res["action"] == "stopped" and res["freed_bytes"] == 378 * MIB and state["stops"] == 1


def test_image_work_in_flight_or_queued_blocks_the_stop():
    for kw in ({"call": {"model_key": "sdxl"}}, {"queue": {"running": 1, "pending": 0}},
               {"queue": {"running": 0, "pending": 2}}, {"call": UNKNOWN}):
        wd, state = build(**kw)
        assert wd.stop_for_need(managed=True, running=True)["action"] == "skip"
        assert state["stops"] == 0


def test_external_or_not_running_never_stops():
    wd, state = build()
    assert wd.stop_for_need(managed=False, running=True)["action"] == "skip"
    assert wd.stop_for_need(managed=True, running=False)["action"] == "skip"
    assert state["stops"] == 0


def test_failed_stop_frees_nothing():
    wd, _ = build(stop=(False, "unit refused"))
    res = wd.stop_for_need(managed=True, running=True)
    assert res["action"] == "failed" and res["freed_bytes"] == 0
