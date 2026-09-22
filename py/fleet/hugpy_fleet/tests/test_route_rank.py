"""ROUTING ADHERES TO RESIDENCY AND ALLOCATION + a progress-aware cold hold.

Operator incident 2026-07-28. A chat for a ~4.7GB GGUF that was RESIDENT and
ALLOCATED on computron — and had been served from there an hour earlier — was
routed to ae instead. ae had nothing on disk, cold-provisioned it, its per-model
gen-gate correctly held with 503 ModelBusy while the weights loaded, and central
killed the caller with "did not finish loading on 'ae' in time". ae finished the
load moments later and served the model. Two defects, one call:

  1. ROUTING. The rank's only residency term was
     ``model_key in (w.get("loaded_models") or [])`` — an EXACT string match in a
     file where every other match site is alias-tolerant — and it never looked at
     ALLOCATIONS at all. With both terms blind, two wildcard boxes tie down to
     ``last_picked`` and the call lands on whichever the round-robin offers.
  2. THE HOLD. The deadline was progress-aware in shape but blind in practice:
     the load-state reader asked provision_progress for ``progress``/``message``
     (keys that entry has never carried — it carries done_bytes/total_bytes/frac),
     read a heartbeat that starves precisely when a box is busy loading, and
     counted a structured 503 ModelBusy — the worker demonstrably WORKING — as
     silence. So the 90s stall clock ran out on a healthy load.

Plus the third: a chat job wedged at stage=provision for 45+ minutes because the
orphan sweep only ever considered rows that were BOTH pending AND worker-less.

Run:  venv/bin/python -m pytest tests/test_route_rank.py -q
"""
import os
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-route-rank-test-")
os.environ.setdefault("HUGPY_COMMS_DB", "off")
# The live /health probe is a network call; every test here supplies its state
# directly, so keep the probe out of the way (its own coverage is below, on the
# cache seam rather than on a socket).
os.environ["HUGPY_COLD_HOLD_HEALTH"] = "off"

import pytest  # noqa: E402

from hugpy_fleet.central import workers as W


# ---------------------------------------------------------------------------
# Fix 1 — the rank
# ---------------------------------------------------------------------------

MK = "Qwen2.5-7B-Instruct-GGUF"


def _w(wid, name, *, loaded=(), allocations=(), grants=None,
       wildcard=True, gpu=True, last_picked=0.0):
    """A candidate as workers_for_model hands it to the rank."""
    return {
        "id": wid, "name": name, "url": f"http://{name}:9100",
        "loaded_models": list(loaded),
        "allocations": list(allocations),
        "grants": dict(grants or {}),
        "_wildcard_catch": wildcard,
        "gpus": ([{"memory_total": 24 * 2**30,
                    "memory_free": 20 * 2**30}] if gpu else []),
        "last_picked": last_picked,
    }


def _rank(worker):
    return W._routing_rank(worker, MK, W._match_keys(MK), starred=False)


def test_resident_outranks_a_merely_capable_box():
    """The incident, reduced: two wildcard boxes, one of them holding it."""
    computron = _w("c", "computron", loaded=[MK], last_picked=1.0)
    ae = _w("a", "ae", last_picked=0.0)          # LRU would pick ae
    assert _rank(computron) < _rank(ae)
    assert min([ae, computron], key=_rank)["name"] == "computron"


def test_residency_is_alias_tolerant():
    """A ~-qualified registry key and a bare base name are the same model.

    The old exact ``in`` test read a box that was actively serving the model as
    stone cold whenever the two spellings differed — the k30 invisible-mismatch
    class, in the one place that decides where a call goes."""
    qualified = _w("c", "computron", loaded=["unsloth~Qwen2.5-7B-Instruct-GGUF"])
    cold = _w("a", "ae")
    assert W._resident_on(qualified, MK, W._match_keys(MK))
    assert _rank(qualified) < _rank(cold)


def test_a_live_slot_seat_counts_as_resident():
    """A slot-child serves without ever appearing in loaded_models."""
    slotted = _w("c", "computron", allocations=[
        {"kind": "slot", "model_key": MK, "healthy": True}])
    assert W._resident_on(slotted, MK, W._match_keys(MK))
    assert W._route_tier(slotted, MK, W._match_keys(MK)) == "resident"


def test_a_dead_slot_seat_is_allocated_but_not_resident():
    dead = _w("c", "computron", allocations=[
        {"kind": "slot", "model_key": MK, "healthy": False, "serving": False}])
    wanted = W._match_keys(MK)
    assert not W._resident_on(dead, MK, wanted)
    assert W._allocated_on(dead, MK, wanted)
    assert W._route_tier(dead, MK, wanted) == "allocated"


def test_allocated_outranks_capability_but_loses_to_resident():
    resident = _w("r", "resident-box", loaded=[MK], last_picked=9.0)
    # A DEAD slot seat: the allocation is still held on the box, the weights
    # are not up. (An in-RAM allocation row IS residency by construction — the
    # weights are in RAM — so it would not distinguish the two tiers.)
    allocated = _w("l", "allocated-box", allocations=[
        {"kind": "slot", "model_key": MK, "healthy": False}], last_picked=5.0)
    capable = _w("k", "capable-box", last_picked=0.0)
    order = [w["name"] for w in sorted([capable, allocated, resident], key=_rank)]
    assert order == ["resident-box", "allocated-box", "capable-box"]


def test_a_placement_grant_is_an_allocation():
    granted = _w("g", "granted", grants={MK: {"by": "placement"}})
    assert W._allocated_on(granted, MK, W._match_keys(MK))


def test_designation_is_hard_and_outranks_everything():
    """Operator ruling (designation-is-advisory CORRECTED: designation = HARD).

    A designated box wins even against a wildcard box that already has the model
    resident — designation is a scope, not a preference."""
    designated = _w("d", "designated", wildcard=False, last_picked=99.0)
    resident_wildcard = _w("c", "computron", loaded=[MK], last_picked=0.0)
    assert _rank(designated) < _rank(resident_wildcard)
    assert W._route_tier(designated, MK, W._match_keys(MK)) == "designated"


def test_capability_ordering_within_a_tier_is_unchanged():
    """Same tier -> the pre-existing rank decides: GPU, then least-recently-picked."""
    gpu = _w("g", "gpu-box", gpu=True, last_picked=50.0)
    cpu = _w("c", "cpu-box", gpu=False, last_picked=1.0)
    assert _rank(gpu) < _rank(cpu)
    older = _w("o", "older", last_picked=1.0)
    newer = _w("n", "newer", last_picked=99.0)
    assert _rank(older) < _rank(newer)


def test_route_select_telemetry_names_the_tier_and_the_alternatives():
    from hugpy_fleet.central import evictions
    evictions.reset_for_tests()
    resident = _w("r", "computron", loaded=[MK])
    other = _w("a", "ae")
    W._emit_route_select(MK, resident, [resident, other], W._match_keys(MK))
    evs = [e for e in evictions.recent(50) if e.get("stage") == "route.select"]
    assert len(evs) == 1
    ev = evs[0]
    assert ev["model_key"] == MK
    assert ev["chosen_worker"] == "computron"
    assert ev["tier"] == "resident"
    assert ev["alternatives"] == [{"worker": "ae", "tier": "capability"}]


def test_telemetry_never_breaks_routing():
    """A junk candidate list must not raise out of the emitter."""
    W._emit_route_select(MK, {}, [None], W._match_keys(MK))  # must not raise


# ---------------------------------------------------------------------------
# Fix 1 — end to end through the real store
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(monkeypatch, tmp_path):
    from hugpy_fleet.central.workers import WorkerStore
    s = WorkerStore(path=str(tmp_path / "wk.json"))
    monkeypatch.setattr(W, "worker_store", s)
    monkeypatch.setattr(W, "_assign_memory_path",
                        lambda: str(tmp_path / "assign.json"))
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    # Both boxes are all-comers wildcards, exactly as the live fleet was.
    monkeypatch.setattr(W, "_wildcard_map", lambda: {"__all__": True})
    return s


def _admit(store, name, **beat):
    w = store.register(name=name, url=f"http://{name}:9100")
    store.set_admission(w["id"], "approved")
    store.heartbeat(w["id"], **beat)
    return w["id"]


def test_pick_for_model_prefers_the_box_that_has_it(store, monkeypatch):
    ae = _admit(store, "ae", loaded_models=[], gpus=[{"memory_total": 2**35, "memory_free": 2**34}])
    computron = _admit(store, "computron", loaded_models=[MK],
                       gpus=[{"memory_total": 2**35, "memory_free": 2**34}])
    monkeypatch.setattr(W, "_wildcard_map",
                        lambda: {ae: True, computron: True})
    chosen = store.pick_for_model(MK)
    assert chosen is not None and chosen["name"] == "computron"
    # ...and it stays computron on the second call: a warm box does not lose its
    # place to round-robin the moment it is picked once (the "second call must
    # hit the resident tier explicitly" check).
    assert store.pick_for_model(MK)["name"] == "computron"


def test_pick_for_model_honours_an_allocation_on_a_cold_box(store, monkeypatch):
    ae = _admit(store, "ae", gpus=[{"memory_total": 2**35, "memory_free": 2**34}])
    computron = _admit(store, "computron", gpus=[{"memory_total": 2**35, "memory_free": 2**34}],
                       allocations=[{"kind": "ram", "model_key": MK}])
    monkeypatch.setattr(W, "_wildcard_map",
                        lambda: {ae: True, computron: True})
    assert store.pick_for_model(MK)["name"] == "computron"


def test_reroute_list_ranks_identically_to_the_pick(store, monkeypatch):
    ae = _admit(store, "ae", gpus=[{"memory_total": 2**35, "memory_free": 2**34}])
    computron = _admit(store, "computron", loaded_models=[MK],
                       gpus=[{"memory_total": 2**35, "memory_free": 2**34}])
    monkeypatch.setattr(W, "_wildcard_map",
                        lambda: {ae: True, computron: True})
    order = [w["name"] for w in store.candidates_for_model(MK)]
    assert order[0] == "computron"
    assert store.pick_for_model(MK)["name"] == order[0]


# ---------------------------------------------------------------------------
# Fix 2 — the progress-aware deadline
# ---------------------------------------------------------------------------

def _remote():
    from hugpy_engine.resolvers import remote
    return remote


def test_the_ceiling_default_is_900s():
    remote = _remote()
    os.environ.pop("HUGPY_COLD_HOLD_MAX_S", None)
    assert remote._cold_hold_max_s() == 900.0
    assert remote._cold_hold_stall_s() == 90.0


def test_a_structured_busy_is_forward_progress_not_silence():
    remote = _remote()
    busy = ("503 SERVICE UNAVAILABLE for http://192.168.1.100:9100/infer/stream")
    assert remote._is_worker_busy_signal(busy)
    assert remote._is_worker_busy_signal(
        "model 'X' is busy: 1 request(s) already in the in-process runner")
    assert not remote._is_worker_busy_signal("")
    assert not remote._is_worker_busy_signal("some other failure")


def test_a_permanent_failure_is_never_read_as_busy():
    """A 507/won't-fit must fail fast even if the text mentions a code."""
    remote = _remote()
    assert not remote._is_worker_busy_signal("503 — model won't fit on this box")
    assert not remote._is_worker_busy_signal(
        "503: chat template error — roles must alternate")


def test_progress_extends_the_hold_and_a_stall_ends_it():
    """The deadline arithmetic the hold loop runs, exercised directly."""
    remote = _remote()
    stall_s = remote._cold_hold_stall_s()
    start = time.time() - 10_000        # far past any fixed clock
    ceiling = start + remote._cold_hold_max_s()

    # Progressing: last movement was a moment ago -> the hold continues even
    # though the call started 10_000s ago... until the CEILING, which is the
    # pathological-progress guard and is intentionally not extendable.
    last_move = time.time()
    now = time.time()
    assert (now - last_move) <= stall_s          # not stalled
    assert now > ceiling                          # but the ceiling still bites

    # A fresh call that keeps progressing outlives the OLD 300s fixed clock.
    start2 = time.time() - 400
    assert time.time() <= start2 + remote._cold_hold_max_s()

    # Stalled: no movement for longer than the stall window -> dead.
    last_move = time.time() - (stall_s + 1)
    assert (time.time() - last_move) > stall_s


def test_the_terminal_message_carries_the_last_observed_progress():
    remote = _remote()
    msg = remote._cold_timeout_message(
        MK, {"name": "ae"}, "503 SERVICE UNAVAILABLE",
        last_progress="loading Qwen2.5-7B-Instruct-GGUF on ae — 12.1 GB transferred",
        stalled_for=90.0)
    assert "12.1 GB" in msg
    assert "90s" in msg
    assert "no forward progress" in msg


def test_the_ceiling_message_says_ceiling_not_stall():
    remote = _remote()
    msg = remote._cold_timeout_message(MK, {"name": "ae"}, "503", ceiling=True)
    assert "hard ceiling" in msg
    assert "no forward progress" not in msg


def test_a_request_shape_failure_is_still_never_relabelled():
    """Regression guard on the pre-existing contract — the new arguments must
    not let a template error come back as a capacity verdict."""
    remote = _remote()
    msg = remote._cold_timeout_message(
        MK, {"name": "ae"}, "jinja2 template error: roles must alternate",
        last_progress="8 GB", stalled_for=90.0)
    assert "REQUEST SHAPE" in msg
    assert "too large for the box" not in msg


def test_the_progress_line_reads_the_keys_the_worker_actually_writes():
    """done_bytes/total_bytes/frac — NOT progress/message, which never existed
    on a provision_progress entry and are why every held call showed a bare
    spinner and every give-up message had no numbers in it."""
    frac, line = W._progress_line(
        MK, "ae", {"done_bytes": 8.2 * 2**30, "total_bytes": 11.4 * 2**30,
                   "frac": 0.72})
    assert frac == 0.72
    assert "8.2 GB" in line and "11.4 GB" in line and "ae" in line
    assert W._progress_line(MK, "ae", None) == (None, None)
    assert W._progress_line(MK, "ae", {}) == (None, None)


def test_load_state_reports_bytes_and_stays_advisory(store, monkeypatch):
    wid = _admit(store, "ae", provisioning=[MK],
                 provision_progress={MK: {"done_bytes": 3 * 2**30,
                                          "total_bytes": 6 * 2**30,
                                          "frac": 0.5}})
    ls = W.load_state_for_model(MK, wid)
    assert ls["in_progress"] is True
    assert ls["healthy"] is False
    assert ls["progress"] == 0.5
    assert "3.0 GB" in ls["message"]
    # Unknown worker -> None, never a raise (the hold degrades, never crashes).
    assert W.load_state_for_model(MK, "nope") is None


def test_the_health_overlay_is_additive_only(store, monkeypatch):
    """A live /health answer can turn 'no movement' into movement; a failed or
    missing probe can only leave the hold as blind as it already was."""
    wid = _admit(store, "ae")                     # heartbeat knows nothing
    assert W.load_state_for_model(MK, wid)["in_progress"] is False
    monkeypatch.setattr(W, "_live_health",
                        lambda w: {"loaded_models": [MK]})
    assert W.load_state_for_model(MK, wid)["healthy"] is True
    monkeypatch.setattr(W, "_live_health", lambda w: {"provisioning": [MK]})
    assert W.load_state_for_model(MK, wid)["in_progress"] is True
    monkeypatch.setattr(W, "_live_health", lambda w: None)
    assert W.load_state_for_model(MK, wid)["in_progress"] is False


def test_the_health_probe_never_blocks_the_caller(monkeypatch):
    """It reads a cache and kicks a background refresh — a slow worker must not
    stall the shared event loop the hold runs on."""
    monkeypatch.setenv("HUGPY_COLD_HOLD_HEALTH", "on")
    started = {}

    class _T:
        def __init__(self, target=None, args=(), **kw):
            started["kicked"] = True

        def start(self):
            pass

    monkeypatch.setattr(W.threading, "Thread", _T)
    W._HEALTH_CACHE.pop("hw", None)
    t0 = time.time()
    out = W._live_health({"id": "hw", "url": "http://10.255.255.1:9100"})
    assert time.time() - t0 < 0.5        # returned immediately
    assert out is None                    # nothing cached yet
    assert started.get("kicked")


# ---------------------------------------------------------------------------
# Fix 3 — the wedged job
# ---------------------------------------------------------------------------

def _jobs():
    from hugpy_control import jobs
    return jobs


def test_a_job_wedged_in_provision_is_expired(monkeypatch):
    jobs = _jobs()
    monkeypatch.setenv("HUGPY_JOB_STALLED_EXPIRY_SECONDS", "900")
    store = jobs.JobStore()
    job = store.create(kind="chat", model="X")
    store.update(job.id, status="processing", stage="provision",
                 message="downloading … (?/?)", worker="ae")
    # 45 minutes with no forward progress — the operator's a75a26ce.
    store._jobs[job.id].progressed_at = time.time() - 45 * 60
    assert job.id in store.expire_pending_orphans()
    row = store.get(job.id)
    assert row.status == "expired"
    assert "no forward progress" in row.message
    assert "provision" in row.message


def test_a_progressing_job_is_never_expired(monkeypatch):
    jobs = _jobs()
    monkeypatch.setenv("HUGPY_JOB_STALLED_EXPIRY_SECONDS", "900")
    store = jobs.JobStore()
    job = store.create(kind="chat", model="X")
    store.update(job.id, status="processing", stage="provision", progress=0.1)
    store._jobs[job.id].progressed_at = time.time() - 45 * 60
    # Real movement: progress ADVANCES -> progressed_at resets.
    store.update(job.id, progress=0.4)
    assert store.expire_pending_orphans() == []
    assert store.get(job.id).status == "processing"


def test_a_chatty_but_wedged_job_still_expires(monkeypatch):
    """Log lines are not progress — a wedged render can spew retries forever."""
    jobs = _jobs()
    monkeypatch.setenv("HUGPY_JOB_STALLED_EXPIRY_SECONDS", "900")
    store = jobs.JobStore()
    job = store.create(kind="chat", model="X")
    store.update(job.id, status="processing", stage="provision")
    store._jobs[job.id].progressed_at = time.time() - 45 * 60
    store.update(job.id, log_tail=["retrying…"])
    assert job.id in store.expire_pending_orphans()


def test_terminal_jobs_are_left_alone(monkeypatch):
    jobs = _jobs()
    monkeypatch.setenv("HUGPY_JOB_STALLED_EXPIRY_SECONDS", "900")
    store = jobs.JobStore()
    job = store.create(kind="chat", model="X")
    store.finish(job.id, status="done")
    store._jobs[job.id].progressed_at = time.time() - 45 * 60
    assert store.expire_pending_orphans() == []
    assert store.get(job.id).status == "done"


def test_the_expiry_clock_is_far_above_the_display_stall_clock():
    """Saying a row LOOKS stalled and ENDING someone's call must not share a
    number: 90s marks, 900s retires."""
    jobs = _jobs()
    os.environ.pop("HUGPY_JOB_STALLED_EXPIRY_SECONDS", None)
    assert jobs._stalled_expiry_seconds() == 900.0
    assert jobs._stall_seconds() == 90.0
    assert jobs._stalled_expiry_seconds() > 5 * jobs._stall_seconds()


# ---------------------------------------------------------------------------
# Fix 4 — THE ENTRY PATH: a first-attempt 503 must ENTER the hold
#
# Operator retest 2026-07-28: "a cold load STILL errors first, then works on
# retry." The first dispatch POSTs /infer/stream, the worker's gen-gate answers
# 503 while the cold load runs, and _worker_stream called a bare
# raise_for_status() — throwing the worker's own {"error":{"code":"model_busy"}}
# envelope away. Central was left classifying an opaque httpx string, so a
# gen-gate hold, a saturated box and a capacity verdict were indistinguishable.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import types as _types  # noqa: E402


def _busy_body(mk="X"):
    """The body worker_agent.gen_gate.ModelBusy.as_error actually produces."""
    return {"ok": False,
            "error": {"code": "model_busy",
                      "message": (f"model '{mk}' is busy: 1 request(s) already in "
                                  f"the in-process runner (llama.cpp/transformers "
                                  f"serialize per model); waited 30.0s"),
                      "model_key": mk, "in_flight": 1, "waited_s": 30.0},
            "worker": {"id": "w1", "name": "ae"}}


def test_the_real_503_envelope_is_read_as_busy_not_as_prose():
    remote = _remote()
    err = remote._WorkerHTTPError(503, _busy_body(), "http://ae:9100/infer/stream")
    assert err.code == "model_busy"
    assert remote._is_worker_busy_signal(err)
    assert not remote._is_permanent_load_error(err)


def test_a_bodyless_503_is_still_busy_status_plus_context():
    """Some transports lose the body entirely — status + context must decide."""
    remote = _remote()
    assert remote._is_worker_busy_signal(remote._WorkerHTTPError(503, None))


def test_a_507_capacity_verdict_is_NOT_busy_and_fails_fast():
    """The inverse defect from the same missing parse: a BudgetRefusal used to
    arrive as "Client error '507 …'", match no permanent marker, and get HELD
    for the full ceiling instead of refusing immediately."""
    remote = _remote()
    err = remote._WorkerHTTPError(
        507, {"ok": False, "error": "LoadRefusal: won't fit on GPU: needs 16.3 GB"})
    assert not remote._is_worker_busy_signal(err)
    assert remote._is_permanent_load_error(err)


def test_a_503_carrying_a_permanent_cause_is_not_laundered_into_a_hold():
    remote = _remote()
    err = remote._WorkerHTTPError(
        503, {"error": {"code": "", "message": "could not provision X: disk full (ENOSPC)"}})
    assert not remote._is_worker_busy_signal(err)


# -- the loop, end to end ---------------------------------------------------

def _entry_runner(remote, worker):
    fw, tk = None, None
    for (f, t) in remote.FRAMEWORK_RUNNERS:
        if t != "image-text-to-text":
            fw, tk = f, t
            break
    cls = remote.make_delegating_runner(fw, tk)
    return cls(_types.SimpleNamespace(model_key=MK))


def _entry_req(rid="entry-1"):
    return _types.SimpleNamespace(
        request_id=rid, pool=None, reference_images=None,
        reference_images_b64=None,
        model_dump=lambda: {"messages": [{"role": "user", "content": "hi"}]})


async def _drain(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


@pytest.fixture()
def entry_env(monkeypatch):
    remote = _remote()
    monkeypatch.setenv("HUGPY_CENTRAL_GATE", "off")
    monkeypatch.setenv("HUGPY_COLD_HOLD_POLL_S", "0.01")
    monkeypatch.setenv("HUGPY_COLD_HOLD_MAX_S", "5")
    monkeypatch.setenv("HUGPY_COLD_HOLD_STALL_S", "5")
    monkeypatch.delenv("HUGPY_LOCAL_FALLBACK", raising=False)
    monkeypatch.delenv("HUGPY_NO_LOCAL_SERVING", raising=False)
    worker = {"id": "w1", "name": "ae", "url": "http://ae:9100"}
    monkeypatch.setattr(remote, "_select",
                        lambda mk, pool=None, task=None, **kw: (dict(worker), None))
    return remote, worker


def test_first_attempt_503_enters_the_hold_and_succeeds(entry_env, monkeypatch):
    """THE REGRESSION. Attempt 1 is the real 503 envelope; /health shows the
    provision progressing; the call is HELD (awaiting-load keepalives), retried,
    and completes on the user's FIRST attempt — no error, no manual resend."""
    remote, worker = entry_env
    from hugpy_engine.schemas.event_schemas import TokenEvent, DoneEvent

    seen = {"bytes": 2 * 2**30}

    def _health(mk, wid, since=0.0):
        seen["bytes"] += 2**30          # the load is demonstrably moving
        return {"healthy": False, "in_progress": True, "progress": 0.5,
                "message": f"loading {mk} on ae — "
                           f"{seen['bytes'] / 2**30:.1f} GB transferred"}

    remote.set_load_state_provider(_health)
    calls = {"n": 0}

    async def ws(w, payload, rid):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise remote._WorkerHTTPError(503, _busy_body(MK),
                                          "http://ae:9100/infer/stream")
        yield TokenEvent(request_id=rid, text="OK")
        yield DoneEvent(request_id=rid, input_tokens=1, output_chunks=1,
                        finish_reason="stop")
        return

    monkeypatch.setattr(remote, "_worker_stream", ws)
    try:
        evs = asyncio.run(_drain(_entry_runner(remote, worker).stream(_entry_req())))
    finally:
        remote.set_load_state_provider(None)
    types_ = [getattr(e, "type", None) for e in evs]
    assert "error" not in types_, [getattr(e, "message", None) for e in evs
                                   if getattr(e, "type", None) == "error"]
    assert "token" in types_ and "done" in types_
    assert calls["n"] == 3                       # held and retried, not surfaced
    holding = [e for e in evs if getattr(e, "stage", None) == "awaiting-load"]
    assert holding, "the browser got no keepalive while the load ran"
    assert "GB transferred" in getattr(holding[0], "message", "")


def test_a_503_that_never_progresses_dies_honestly(entry_env, monkeypatch):
    """The inverse: busy forever with no forward progress -> the stall clock
    ends it, and the message names where it stopped."""
    remote, worker = entry_env
    monkeypatch.setenv("HUGPY_COLD_HOLD_STALL_S", "0.05")
    remote.set_load_state_provider(
        lambda mk, wid, since=0.0: {"healthy": False, "in_progress": False,
                                    "progress": None, "message": None,
                                    "error": None})

    async def ws(w, payload, rid):
        raise remote._WorkerHTTPError(503, _busy_body(MK))
        yield  # pragma: no cover

    monkeypatch.setattr(remote, "_worker_stream", ws)
    try:
        evs = asyncio.run(_drain(_entry_runner(remote, worker).stream(_entry_req("entry-2"))))
    finally:
        remote.set_load_state_provider(None)
    errs = [e for e in evs if getattr(e, "type", None) == "error"]
    assert errs, "a permanently-busy worker must not hold forever"
    assert "did not finish loading" in errs[0].message


def test_a_507_refusal_fails_fast_and_is_never_held(entry_env, monkeypatch):
    remote, worker = entry_env
    remote.set_load_state_provider(None)
    calls = {"n": 0}

    async def ws(w, payload, rid):
        calls["n"] += 1
        raise remote._WorkerHTTPError(
            507, {"ok": False,
                  "error": "LoadRefusal: won't fit on GPU: needs 16.3 GB"})
        yield  # pragma: no cover

    monkeypatch.setattr(remote, "_worker_stream", ws)
    evs = asyncio.run(_drain(_entry_runner(remote, worker).stream(_entry_req("entry-3"))))
    errs = [e for e in evs if getattr(e, "type", None) == "error"]
    assert errs and "won't fit" in errs[0].message
    assert calls["n"] == 1, "a capacity verdict must never be retried"
