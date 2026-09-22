"""k96 — "no_makeroom": the agent-brain no-evict guarantee.

Operator ruling 2026-08-06 ("evicting a model to allow for [the broker brain]
could create a problem"): a chat flagged ``no_makeroom: true`` must never cost
the fleet a resident model. A WARM model serves exactly as today; a COLD load
runs the whole admission politely (the k56 no_evict rules) and FAILS FAST with
a capacity-class LoadRefusal the agent's brain ladder can walk, instead of
evicting or wedging.

Asserted here, end to end across the seams:
  * the /v1 seam forwards the key (truthy only) and ChatRequest carries it
    without ever dumping a None (the released-worker extra="forbid" landmine);
  * the chat builder forwards it (the silent-drop trap);
  * dispatch's headroom pass under the flag: the in-process LRU yield is
    SKIPPED (nothing evicted), a registered make-room hook is consulted with
    the flag VISIBLE (its polite branch — refusal propagates), and with no
    hook an unfit load refuses fast instead of proceeding unfit;
  * the slot pool neither ceiling-evicts nor bumps a seat under the flag;
  * the relay pops the key from the wire and translates it onto the spill's
    ``no_evict`` — version-gated, stripped LOUDLY for a pre-k56 worker;
  * flag absent -> byte-identical behavior on every one of those paths.

Run: ./venv/bin/python -m pytest tests/test_no_makeroom.py -q
"""
import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-k96-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_engine import alloc_modes as AM
# managers/__init__ star-imports shadow the subpackage attrs — bind the REAL
# module (the dispatch module-shadowing landmine; see test_worker_preference).
D = importlib.import_module("hugpy_engine.dispatch.dispatch")
slots = importlib.import_module("hugpy_engine.serve.slots")

MK = "Qwen~Qwen2.5-3B-Instruct-GGUF"


# ---------------------------------------------------------------------------
# schema + builder
# ---------------------------------------------------------------------------

def test_chat_request_never_dumps_none_and_builder_forwards():
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    from hugpy_engine.resolvers.categories import builders

    plain = ChatRequest(model_key=MK, messages=[{"role": "user", "content": "x"}])
    assert "no_makeroom" not in plain.model_dump()   # the extra="forbid" landmine

    flagged = builders._build_chat_request(
        {"messages": [{"role": "user", "content": "x"}], "no_makeroom": True}, MK)
    assert flagged.no_makeroom is True
    assert flagged.model_dump()["no_makeroom"] is True

    unflagged = builders._build_chat_request(
        {"messages": [{"role": "user", "content": "x"}]}, MK)
    assert "no_makeroom" not in unflagged.model_dump()


# ---------------------------------------------------------------------------
# dispatch headroom pass
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_dispatch():
    """Save/restore every dispatch hook + the contextvar + fake residents."""
    saved = (D._FIT_CHECK, D._EVICTABLE, D._POST_EVICT, D._MAKE_ROOM,
             D._EVICT_REASON, dict(D._INSTANCES), D.evict)
    token = D._NO_MAKEROOM.set(False)
    try:
        yield D
    finally:
        (D._FIT_CHECK, D._EVICTABLE, D._POST_EVICT, D._MAKE_ROOM,
         D._EVICT_REASON) = saved[:5]
        D._INSTANCES.clear()
        D._INSTANCES.update(saved[5])
        D.evict = saved[6]
        D._NO_MAKEROOM.reset(token)


def _seat_fake_resident(key="resident-a"):
    D._INSTANCES[(key, "chat")] = object()
    return key


def test_flag_absent_yields_lru_exactly_as_today(clean_dispatch):
    res = _seat_fake_resident()
    evicted_log = []
    fits = {"ok": False}
    D.set_fit_check(lambda mk: fits["ok"])
    D.set_evictable(lambda mk: True)
    D.set_make_room(None)
    D.set_post_evict_hook(None)
    def fake_evict(mk, task=None):
        evicted_log.append(mk)
        D._INSTANCES.pop((mk, "chat"), None)
        fits["ok"] = True
        return True
    D.evict = fake_evict
    out = D.ensure_headroom_for_load(MK)
    assert out == [res] and evicted_log == [res]


def test_no_makeroom_skips_lru_yield_and_fails_fast_without_hook(clean_dispatch):
    _seat_fake_resident()
    evicted_log = []
    D.set_fit_check(lambda mk: False)          # never fits in free room
    D.set_evictable(lambda mk: True)           # a victim WAS available
    D.set_make_room(None)
    D.evict = lambda mk, task=None: evicted_log.append(mk)
    D._NO_MAKEROOM.set(True)
    with pytest.raises(D.LoadRefusal) as exc:
        D.ensure_headroom_for_load(MK)
    assert evicted_log == []                   # nobody lost their seat
    msg = str(exc.value)
    assert "refusing without evicting" in msg  # the ladder-walkable wording


def test_no_makeroom_consults_hook_politely_and_refusal_propagates(clean_dispatch):
    _seat_fake_resident()
    evicted_log = []
    seen = {}
    D.set_fit_check(lambda mk: False)
    D.set_evictable(lambda mk: True)
    D.evict = lambda mk, task=None: evicted_log.append(mk)
    def hook(mk):
        # The worker's make-room hook must SEE the request flag (its polite
        # branch keys off it) — and its honest refusal must propagate.
        seen["polite"] = D.no_makeroom_active()
        raise D.LoadRefusal({"reason": "won't fit on GPU: free room is short "
                                       "— refusing without evicting",
                             "model_key": mk})
    D.set_make_room(hook)
    D._NO_MAKEROOM.set(True)
    with pytest.raises(D.LoadRefusal):
        D.ensure_headroom_for_load(MK)
    assert seen["polite"] is True
    assert evicted_log == []                   # the LRU yield never ran


def test_no_makeroom_free_room_admit_proceeds(clean_dispatch):
    """The warm/fits case: hook admits into free room -> no raise, no evict."""
    D.set_fit_check(lambda mk: False)          # in-process guard says short...
    D.set_evictable(lambda mk: True)
    D.evict = lambda mk, task=None: (_ for _ in ()).throw(AssertionError)
    D.set_make_room(lambda mk: {"action": "proceed", "evicted": [],
                                "freed_bytes": 0, "reason": None,
                                "note": "already resident / fits free room"})
    D._NO_MAKEROOM.set(True)
    assert D.ensure_headroom_for_load(MK) == []


# ---------------------------------------------------------------------------
# slot pool: no ceiling-evict, no seat-bump
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_slots():
    saved = (slots._FIT_CHECK, slots._EVICTION_POLICY, slots._MAKE_ROOM,
             slots._RESIDENCY_LOOKUP, slots._post)
    token = D._NO_MAKEROOM.set(False)
    try:
        yield slots
    finally:
        (slots._FIT_CHECK, slots._EVICTION_POLICY, slots._MAKE_ROOM,
         slots._RESIDENCY_LOOKUP, slots._post) = saved
        D._NO_MAKEROOM.reset(token)


class _Pool(slots.SlotPool):
    def __init__(self, statuses):
        super().__init__(urls=[s["_control"] for s in statuses])
        self._fake = statuses
        self.unloaded = []

    def statuses(self):
        return [dict(s) for s in self._fake]

    def unload(self, control_url):
        self.unloaded.append(control_url)
        for s in self._fake:
            if s["_control"] == control_url:
                s["model_key"] = None
        return {}


def _occupied(url, mk):
    return {"_control": url, "model_key": mk, "healthy": True,
            "busy": False, "endpoint": url + "/infer", "last_used": 1.0}


def test_polite_seat_over_ceiling_refuses_and_evicts_nobody(clean_slots):
    pool = _Pool([_occupied("http://s1", "other-a"),
                  _occupied("http://s2", "other-b")])
    slots._FIT_CHECK = lambda mk: False        # over the ceiling
    slots._EVICTION_POLICY = lambda mk: True   # victims exist on purpose
    slots._MAKE_ROOM = None
    D._NO_MAKEROOM.set(True)
    with pytest.raises(D.LoadRefusal) as exc:
        pool.endpoint_for(MK)
    assert pool.unloaded == []
    assert "refusing without evicting" in str(exc.value)


def test_polite_seat_never_bumps_an_occupied_seat(clean_slots):
    pool = _Pool([_occupied("http://s1", "other-a"),
                  _occupied("http://s2", "other-b")])
    slots._FIT_CHECK = None                    # ceiling gate not registered
    slots._EVICTION_POLICY = lambda mk: True   # promotion WOULD have bumped
    D._NO_MAKEROOM.set(True)
    with pytest.raises(D.LoadRefusal):
        pool.endpoint_for(MK)
    assert pool.unloaded == []


def test_polite_seat_reuses_a_warm_seat_exactly_as_today(clean_slots):
    pool = _Pool([_occupied("http://s1", MK)])
    slots._FIT_CHECK = lambda mk: False        # even over ceiling: reuse wins
    D._NO_MAKEROOM.set(True)
    assert pool.endpoint_for(MK) == "http://s1/infer"


def test_unflagged_promotion_still_bumps(clean_slots):
    pool = _Pool([_occupied("http://s1", "other-a")])
    slots._FIT_CHECK = None
    slots._EVICTION_POLICY = lambda mk: True
    slots._post = lambda url, body, timeout: {"endpoint": "http://s1/infer"}
    assert pool.endpoint_for(MK) == "http://s1/infer"
    assert pool.unloaded == ["http://s1"]


# ---------------------------------------------------------------------------
# relay: pop from the wire, ride the spill, version-gate
# ---------------------------------------------------------------------------

def _chat_req(**kw):
    from hugpy_engine.schemas.chat_schemas import ChatRequest
    return ChatRequest(model_key=MK,
                       messages=[{"role": "user", "content": "x"}], **kw)


def test_relay_translates_no_makeroom_to_spill_no_evict():
    remote = importlib.import_module(
        "hugpy_engine.resolvers.remote")
    worker = {"id": "w1", "name": "computron",
              "pkg_version": AM.NO_EVICT_MIN_PKG_VERSION}
    payload = remote._worker_payload("chat", _chat_req(no_makeroom=True), MK,
                                     "w1", worker=worker)
    assert payload is not None
    assert "no_makeroom" not in payload        # never on the wire
    assert payload["spill"]["no_evict"] is True


def test_relay_strips_for_pre_k56_worker_and_stays_off_the_wire():
    remote = importlib.import_module(
        "hugpy_engine.resolvers.remote")
    worker = {"id": "w1", "name": "old-box", "pkg_version": "0.1.225"}
    payload = remote._worker_payload("chat", _chat_req(no_makeroom=True), MK,
                                     "w1", worker=worker)
    assert payload is not None
    assert "no_makeroom" not in payload
    assert "no_evict" not in (payload.get("spill") or {})


def test_relay_unflagged_chat_is_byte_identical():
    remote = importlib.import_module(
        "hugpy_engine.resolvers.remote")
    worker = {"id": "w1", "name": "computron",
              "pkg_version": AM.NO_EVICT_MIN_PKG_VERSION}
    payload = remote._worker_payload("chat", _chat_req(), MK, "w1",
                                     worker=worker)
    assert payload is not None
    assert "no_makeroom" not in payload
    assert "no_evict" not in (payload.get("spill") or {})


# ---------------------------------------------------------------------------
# the binding itself: execute_prompt propagates the kwarg into the contextvar
# ---------------------------------------------------------------------------

class _Res:
    def __init__(self):
        self.model_key = MK
        self.task = "chat"
        self.cache_key = (MK, "chat")
        self.builder = lambda kwargs, mk: kwargs


class _Runner:
    def __init__(self, log):
        self.log = log

    def run(self, req=None):
        self.log.append(D.no_makeroom_active())
        return {"ok": True}


def test_execute_prompt_binds_and_resets_the_contextvar(clean_dispatch, monkeypatch):
    log = []
    res = _Res()
    monkeypatch.setattr(D, "resolve", lambda kw: res)
    D._INSTANCES[res.cache_key] = _Runner(log)   # warm: no headroom pass
    D.execute_prompt(messages=[{"role": "user", "content": "x"}],
                     model_key=MK, no_makeroom=True)
    D.execute_prompt(messages=[{"role": "user", "content": "x"}],
                     model_key=MK)
    assert log == [True, False]                  # bound per call, reset after
    assert D.no_makeroom_active() is False
