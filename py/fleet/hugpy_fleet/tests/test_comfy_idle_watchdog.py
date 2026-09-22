"""k54 — the ComfyUI idle-VRAM watchdog: predicate, TTL, reclaim, telemetry.

The operator's directive: a dead/idle ComfyUI process must never permanently
squat the GPU (the live case held 2874 MiB for 61 h with an empty queue). The
watchdog frees it through comfy's OWN ``POST /free`` — never a kill.

What these guard, in the order the module applies them:
  * the four idle clauses (VRAM above the context floor / no registered comfy
    call / comfy's own /queue empty / persisted past the TTL) and the ANDing —
    any clause we cannot PROVE must read as "not idle" and leave comfy alone;
  * the TTL is a DEBOUNCE, not a protection class: ``reclaim()`` (contention)
    waives it while clauses 1-3 still bind absolutely;
  * verification by RE-MEASURE, fresh: "/free returned 200" is not evidence
    that bytes came back;
  * a /free that fails or doesn't take is SURFACED (telemetry + log), never
    escalated to a SIGKILL of an adopted external process;
  * the telemetry stream carries the reclaim (tier="comfy") so the console
    shows WHY VRAM moved.

No GPU, no ComfyUI, no HTTP: every probe is injected.
"""
import pytest

from hugpy_fleet.worker import comfy_watchdog as cw
from hugpy_fleet.worker.pid_registry import PidRegistry

MIB = 1024 * 1024
FLOOR = cw._DEFAULT_FLOOR_MIB * MIB
BIG = 2874 * MIB                      # the live incident's figure
IDLE_QUEUE = {"running": 0, "pending": 0}


class Clock:
    """A hand-cranked clock — the TTL is time logic and must be tested without
    sleeping."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def build(vram, queue=IDLE_QUEUE, call=None, free=(True, "ok"), clock=None,
          after=None):
    """A watchdog over scripted probes. ``vram`` is the pre-/free figure and
    ``after`` (default: the context floor) what a fresh re-measure reports once
    /free has been accepted."""
    state = {"freed": 0, "emitted": [], "slept": []}
    after = FLOOR - MIB if after is None else after

    def vram_probe(fresh=False):
        if state["freed"] and fresh:
            return after
        return vram

    def free_call():
        state["freed"] += 1
        return free

    wd = cw.ComfyIdleWatchdog(
        vram_probe=vram_probe,
        url_probe=lambda: "http://comfy.test:8188",
        free_call=free_call,
        queue_probe=lambda url: queue,
        call_probe=lambda: call,
        emit=lambda stage, **f: state["emitted"].append((stage, f)),
        clock=clock or Clock(),
        sleep=state["slept"].append)
    return wd, state


def stages(state):
    return [s for s, _ in state["emitted"]]


def field(state, stage, key):
    for s, f in state["emitted"]:
        if s == stage:
            return f.get(key)
    return None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("HUGPY_COMFY_IDLE_FREE", "HUGPY_COMFY_IDLE_FREE_S",
              "HUGPY_COMFY_VRAM_FLOOR_MIB", "HUGPY_COMFY_FREE_SETTLE_S"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- #
# knobs
# --------------------------------------------------------------------------- #

def test_defaults_are_the_briefed_ones():
    assert cw.idle_ttl_s() == 600.0
    assert cw.context_floor_bytes() == 400 * MIB
    assert cw.enabled() is True


def test_knobs_are_env_overridable(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_FREE_S", "45")
    monkeypatch.setenv("HUGPY_COMFY_VRAM_FLOOR_MIB", "128")
    assert cw.idle_ttl_s() == 45.0
    assert cw.context_floor_bytes() == 128 * MIB


def test_a_garbage_env_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_FREE_S", "soon")
    assert cw.idle_ttl_s() == 600.0


def test_the_kill_switch_disables_both_entry_points(monkeypatch):
    monkeypatch.setenv("HUGPY_COMFY_IDLE_FREE", "0")
    wd, state = build(BIG, clock=Clock())
    assert wd.tick()["action"] == "skip"
    assert wd.reclaim()["action"] == "skip"
    assert state["freed"] == 0


# --------------------------------------------------------------------------- #
# the idle predicate — every clause, and every unprovable clause
# --------------------------------------------------------------------------- #

def test_vram_above_the_floor_with_an_empty_queue_and_no_call_is_idle():
    wd, _ = build(BIG)
    obs = wd.observe()
    assert obs["idle"] is True
    assert obs["vram_bytes"] == BIG


def test_at_the_context_floor_there_is_nothing_to_reclaim():
    wd, state = build(FLOOR - MIB)
    obs = wd.observe()
    assert obs["idle"] is False
    assert "floor" in obs["why"]
    assert wd.tick()["action"] == "skip"
    assert state["freed"] == 0


def test_a_registered_comfy_call_vetoes_the_free():
    wd, state = build(BIG, call={"model_key": "sdxl", "job_id": "j1"})
    assert wd.observe()["idle"] is False
    assert "in flight" in wd.observe()["why"]
    assert wd.tick()["action"] == "skip"
    # …and contention does NOT override it: freeing mid-render kills the render.
    assert wd.reclaim(incoming_model="coder")["action"] == "skip"
    assert state["freed"] == 0


def test_a_busy_comfy_queue_vetoes_the_free():
    wd, state = build(BIG, queue={"running": 1, "pending": 3})
    assert wd.observe()["idle"] is False
    assert wd.reclaim(incoming_model="coder")["action"] == "skip"
    assert state["freed"] == 0


def test_an_unreadable_queue_is_not_idle():
    """Degrade-to-no-op: we free only what we can PROVE is idle."""
    wd, state = build(BIG, queue=None)
    assert wd.observe()["idle"] is False
    assert "cannot prove idle" in wd.observe()["why"]
    assert state["freed"] == 0


def test_an_unreadable_call_table_is_not_idle():
    wd, state = build(BIG, call=cw.UNKNOWN)
    assert wd.observe()["idle"] is False
    assert state["freed"] == 0


def test_unmeasurable_vram_is_not_idle():
    wd, state = build(None)
    assert wd.observe()["idle"] is False
    assert wd.tick()["action"] == "skip"
    assert state["freed"] == 0


def test_a_raising_vram_probe_degrades_rather_than_propagates():
    wd, _ = build(BIG)
    wd._vram_probe = lambda fresh=False: (_ for _ in ()).throw(RuntimeError("smi"))
    assert wd.observe()["idle"] is False


# --------------------------------------------------------------------------- #
# the TTL is a debounce
# --------------------------------------------------------------------------- #

def test_the_ttl_must_elapse_before_the_watchdog_frees():
    clock = Clock()
    wd, state = build(BIG, clock=clock)
    assert wd.tick()["action"] == "wait"          # clock starts now
    clock.advance(599)
    assert wd.tick()["action"] == "wait"
    assert state["freed"] == 0
    clock.advance(2)
    assert wd.tick()["action"] == "freed"
    assert state["freed"] == 1


def test_activity_resets_the_idle_clock():
    """Persistence, not a stopwatch: a render in the middle of the window means
    the TTL starts again from zero."""
    clock = Clock()
    wd, state = build(BIG, clock=clock)
    wd.tick()
    clock.advance(590)
    wd._queue_probe = lambda url: {"running": 1, "pending": 0}
    assert wd.tick()["action"] == "skip"          # busy -> clock cleared
    wd._queue_probe = lambda url: IDLE_QUEUE
    clock.advance(20)
    assert wd.tick()["action"] == "wait"          # only 20s into the NEW streak
    clock.advance(600)
    assert wd.tick()["action"] == "freed"
    assert state["freed"] == 1


def test_contention_waives_the_ttl():
    """no-timeblock-on-eviction: freshness is RANK, never a veto."""
    clock = Clock()
    wd, state = build(BIG, clock=clock)
    wd.observe()                                  # idle streak just started
    res = wd.reclaim(incoming_model="coder-next", need_bytes=8 * 2**30)
    assert res["action"] == "freed"
    assert res["freed_bytes"] == BIG - (FLOOR - MIB)
    assert state["freed"] == 1


# --------------------------------------------------------------------------- #
# the free itself — verify by RE-MEASURE, never escalate
# --------------------------------------------------------------------------- #

def test_a_successful_free_reports_the_measured_delta():
    clock = Clock()
    wd, state = build(BIG, clock=clock, after=378 * MIB)
    wd.observe()                                  # the idle streak starts here
    clock.advance(601)
    res = wd.tick()
    assert res["action"] == "freed"
    assert res["before_bytes"] == BIG
    assert res["after_bytes"] == 378 * MIB
    assert res["freed_bytes"] == BIG - 378 * MIB
    assert state["slept"] == [cw.settle_s()]      # settled before re-measuring


def test_the_verification_read_is_fresh():
    """The nvidia-smi read is cached ~8s; a stale post-/free re-measure would
    report every successful reclaim as a failure."""
    seen = []
    wd, _ = build(BIG)
    inner = wd._vram_probe
    wd._vram_probe = lambda fresh=False: (seen.append(fresh), inner(fresh=fresh))[1]
    wd.reclaim()
    assert seen == [False, True]


def test_a_refused_free_is_surfaced_not_escalated():
    clock = Clock()
    wd, state = build(BIG, clock=clock, free=(False, "comfy /free returned HTTP 404"))
    wd.observe()
    clock.advance(601)
    res = wd.tick()
    assert res["action"] == "failed"
    assert res["freed_bytes"] == 0
    assert "evict.fail" in stages(state)
    assert field(state, "headroom.done", "outcome") == "proceeded-unfit"


def test_a_raising_free_call_is_reported_as_a_failure_not_a_crash():
    clock = Clock()
    wd, state = build(BIG, clock=clock)
    wd._free_call = lambda: (_ for _ in ()).throw(OSError("connection refused"))
    wd.observe()
    clock.advance(601)
    res = wd.tick()
    assert res["action"] == "failed"
    assert "OSError" in res["reason"]


def test_vram_that_stays_high_after_an_accepted_free_is_partial_and_surfaced():
    """The operator's question is about BYTES, not HTTP codes — and the answer
    is still never a kill."""
    clock = Clock()
    wd, state = build(BIG, clock=clock, after=2000 * MIB)
    wd.observe()
    clock.advance(601)
    res = wd.tick()
    assert res["action"] == "partial"
    assert res["after_bytes"] == 2000 * MIB
    assert res["freed_bytes"] == BIG - 2000 * MIB
    assert "evict.fail" in stages(state)
    assert "evict.done" not in stages(state)


def test_the_watchdog_never_kills_a_process():
    """There is no kill path at all: the module owns no signalling primitive."""
    src = open(cw.__file__).read()
    assert "os.kill(" not in src
    assert "import signal" not in src
    assert "subprocess" not in src


# --------------------------------------------------------------------------- #
# telemetry
# --------------------------------------------------------------------------- #

def test_a_reclaim_streams_the_eviction_telemetry_stages():
    clock = Clock()
    wd, state = build(BIG, clock=clock)
    wd.observe()
    clock.advance(601)
    wd.tick()
    got = stages(state)
    for stage in ("headroom.start", "evict.start", "evict.done", "reclaim.done",
                  "headroom.done"):
        assert stage in got, stage
    assert field(state, "evict.done", "freed_bytes") == BIG - (FLOOR - MIB)
    assert field(state, "evict.start", "tier") == "comfy"
    assert field(state, "headroom.start", "trigger") == "comfy-idle"
    assert field(state, "headroom.done", "evicted") == ["comfy"]


def test_the_contention_path_names_the_incoming_model():
    wd, state = build(BIG)
    wd.reclaim(incoming_model="coder-next", need_bytes=123)
    assert field(state, "headroom.start", "trigger") == "contention"
    assert field(state, "headroom.start", "incoming_model") == "coder-next"
    assert field(state, "evict.done", "model_key") == "comfy"


def test_an_idle_but_pre_ttl_box_emits_nothing():
    """A quiet box must not spam the stream on its 60s beat."""
    wd, state = build(BIG, clock=Clock())
    wd.tick()
    assert state["emitted"] == []


def test_a_box_with_no_comfy_emits_nothing_and_calls_nothing():
    wd, state = build(None, queue=None)
    wd.tick()
    assert state["emitted"] == []
    assert state["freed"] == 0


def test_a_broken_emitter_never_breaks_the_reclaim():
    wd, state = build(BIG)
    wd._emit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sink down"))
    with pytest.raises(RuntimeError):
        wd._emit("x")                              # the sink really is broken…
    # …and the module's own default emitter swallows everything, which is the
    # contract the worker binds. Prove the real emit path is total:
    cw._default_emit("evict.done", model_key="comfy", tier="comfy")


# --------------------------------------------------------------------------- #
# the pid-registry call-table read the predicate depends on
# --------------------------------------------------------------------------- #

def test_active_foreign_call_sees_a_registered_call_and_its_end():
    reg = PidRegistry(proc_info=lambda pid: None)
    assert reg.active_foreign_call("comfy") is None
    reg.record_foreign_call("comfy", "sdxl-base", job_id="j1")
    got = reg.active_foreign_call("comfy")
    assert got and got["model_key"] == "sdxl-base" and got["job_id"] == "j1"
    reg.end_foreign_call("comfy", job_id="j1")
    assert reg.active_foreign_call("comfy") is None


def test_a_leaked_call_record_ages_out_by_the_ttl():
    """The safety net that stops a missed ``end_foreign_call`` from pinning
    comfy's VRAM forever — the exact 61-hour failure mode, one layer up."""
    reg = PidRegistry(proc_info=lambda pid: None)
    reg.record_foreign_call("comfy", "sdxl-base", job_id="lost")
    assert reg.active_foreign_call("comfy", ttl=0.0) is None
    assert reg.active_foreign_call("comfy", ttl=3600.0) is not None


def test_the_queue_probe_reads_comfys_own_account():
    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"queue_running": [["x"]], "queue_pending": [], "extra": 1}

    class C:
        def get(self, url):
            assert url.endswith("/queue")
            return R()

    assert cw.queue_state("http://c:8188", client=C()) == {"running": 1, "pending": 0}


def test_the_queue_probe_returns_none_on_a_non_200_or_an_error():
    class Bad:
        status_code = 503

        @staticmethod
        def json():
            return {}

    class C:
        def get(self, url):
            return Bad()

    class Boom:
        def get(self, url):
            raise OSError("refused")

    assert cw.queue_state("http://c:8188", client=C()) is None
    assert cw.queue_state("http://c:8188", client=Boom()) is None
    assert cw.queue_state("", client=C()) is None


# --------------------------------------------------------------------------- #
# worker wiring — the two places the reclaim is bolted onto the agent
# --------------------------------------------------------------------------- #
# These import the agent (torch + the runner stack) and stub the box seams, the
# same shape test_vram_evict_to_fit.py uses. No GPU, no comfy, no HTTP.
import importlib                                        # noqa: E402
import sys                                              # noqa: E402
from pathlib import Path                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as A
from hugpy_fleet.worker import gen_gate
# managers/__init__ star-imports shadow the subpackage attrs — bind the REAL
# module the agent uses via import_module (the module-shadowing landmine).
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


class _State:
    pass


class _FakeWatchdog:
    """A watchdog whose reclaim hands back a fixed number of bytes (and puts
    them into the rig's free-VRAM cell, as a real one would)."""

    def __init__(self, card, freed):
        self.card = card
        self.freed = freed
        self.calls = []

    def reclaim(self, incoming_model=None, need_bytes=None, trigger="contention"):
        self.calls.append((incoming_model, need_bytes))
        self.card["free"] += self.freed
        return {"action": "freed", "freed_bytes": self.freed}


@pytest.fixture
def rig(monkeypatch):
    card = {"total": 24 * GIB, "free": 0, "need": 0}
    residents = {}
    evicted = []
    for leak in ("HUGPY_GPU_MEM_GIB", "HUGPY_CPU_MEM_GIB", "HUGPY_ALLOC_MODE",
                 "HUGPY_VRAM_CEILING_FRAC", "HUGPY_VRAM_RESERVE_GIB",
                 "HUGPY_VRAM_CEILING_CUSHION_GIB", "HUGPY_BNB_4BIT",
                 "HUGPY_N_GPU_LAYERS", "HUGPY_EVICT_MIN_RESIDENCY_S"):
        monkeypatch.delenv(leak, raising=False)
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: card["total"])
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: card["free"])
    monkeypatch.setattr(A, "_raw_free_vram_bytes", lambda: card["free"], raising=False)
    monkeypatch.setattr(A, "_incoming_need_bytes", lambda mk: card["need"])
    monkeypatch.setattr(A, "_vram_residents",
                        lambda s: [{"model_key": k, "vram_bytes": v["vram_bytes"],
                                    "host_mode": v["host_mode"], "alive": True}
                                   for k, v in residents.items()])
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_busy_slot_models", set)
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    monkeypatch.setattr(A, "_trim_host_ram", lambda: None)
    monkeypatch.setattr(D, "last_used_snapshot", dict)
    monkeypatch.setattr(A, "_evict_model",
                        lambda s, mk, force=False: (evicted.append(mk) or
                                                    {"model_key": mk, "evicted": False,
                                                     "reason": "test rig"}))
    A._VRAM_EVICTIONS.update(count=0, last=None, last_at=0.0)
    return type("Rig", (), {"card": card, "residents": residents,
                            "evicted": evicted})()


def test_a_fresh_read_invalidates_the_nvidia_smi_cache(monkeypatch):
    """Without this the post-/free verification would re-read the PRE-free
    figure (the cache holds ~8s) and call every success a failure."""
    A._GPU_PROC_CACHE.update(at=12345.0, value={})
    monkeypatch.setattr(A, "_comfy_process_vram", lambda: 7)
    assert A._comfy_vram_now() == 7
    assert A._GPU_PROC_CACHE["at"] == 12345.0        # cached read left alone
    assert A._comfy_vram_now(fresh=True) == 7
    assert A._GPU_PROC_CACHE["at"] == 0.0            # forced re-poll


def test_the_contention_reclaim_reports_bytes_and_never_raises(monkeypatch, rig):
    monkeypatch.setattr(A, "_comfy_watchdog",
                        lambda s: _FakeWatchdog(rig.card, 3 * GIB))
    assert A._comfy_reclaim_idle_vram(_State(), "coder", need_bytes=1) == 3 * GIB

    def _boom(s):
        raise RuntimeError("comfy probe exploded")
    monkeypatch.setattr(A, "_comfy_watchdog", _boom)
    assert A._comfy_reclaim_idle_vram(_State(), "coder") == 0


def test_the_headroom_sweep_takes_idle_comfy_before_any_model(monkeypatch, rig):
    """Reclaiming an idle comfy costs nobody a reload; evicting a managed
    resident does. So the squatter pays first — and if that clears the pressure
    reserve, no model is touched at all."""
    rig.card["free"] = 100 * MIB                     # deep under the ceiling
    rig.residents["idle_coder"] = {"vram_bytes": 12 * GIB, "host_mode": "subprocess"}
    monkeypatch.setattr(A, "_comfy_watchdog",
                        lambda s: _FakeWatchdog(rig.card, 6 * GIB))
    A._vram_headroom_sweep(_State())
    assert rig.evicted == []                         # the model kept its seat


def test_the_sweep_still_evicts_when_the_comfy_reclaim_is_not_enough(monkeypatch, rig):
    rig.card["free"] = 100 * MIB
    rig.residents["idle_coder"] = {"vram_bytes": 12 * GIB, "host_mode": "subprocess"}
    monkeypatch.setattr(A, "_comfy_watchdog",
                        lambda s: _FakeWatchdog(rig.card, 100 * MIB))
    A._vram_headroom_sweep(_State())
    assert rig.evicted == ["idle_coder"]


def test_admission_claims_idle_comfy_vram_instead_of_refusing(monkeypatch, rig):
    """The k54 gap: every managed candidate is walked, it still doesn't fit, and
    the only thing left on the card is a comfy holding bytes nobody is using."""
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    wd = _FakeWatchdog(rig.card, 14 * GIB)
    monkeypatch.setattr(A, "_comfy_watchdog", lambda s: wd)
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "evicted"
    assert plan["evicted"] == []                     # no model paid for it
    assert plan["comfy_freed_bytes"] == 14 * GIB
    assert "ComfyUI" in plan["note"]
    assert wd.calls == [("subject", 10 * GIB)]       # priced against the real need


def test_admission_still_refuses_honestly_when_comfy_is_busy(monkeypatch, rig):
    """A reclaim that frees nothing must change nothing: the refusal is the
    same one the operator sees today."""
    rig.card["free"] = 1 * GIB
    rig.card["need"] = 10 * GIB
    monkeypatch.setattr(A, "_comfy_reclaim_idle_vram",
                        lambda s, mk, need_bytes=None: 0)
    plan = A._vram_evict_to_fit(_State(), "subject")
    assert plan["action"] == "refuse"
    assert "comfy_freed_bytes" not in plan
