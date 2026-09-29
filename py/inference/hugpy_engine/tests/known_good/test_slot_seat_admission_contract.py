"""KNOWN-GOOD CONTRACT — the slot seat path admits through plan_fit only.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (F1).
Source under test: hugpy_engine/serve/slots.py (SlotPool.endpoint_for,
_seat_admission, the promotion branch, set_make_room / set_evict_verb),
hugpy_engine/llama/runners/get.py (_build_runner: a LoadRefusal from the
seat is final).

LIVE CASES (2026-09-29, S3c / S6): "VRAM ceiling: evicted idle on-demand
Qwen3-Coder-Next-GGUF ..." — the pool evicted two residents itself, outside
plan_fit: no verdict line, no evict.start/done, vram_evictions unchanged, a
PINNED model labelled on-demand, hollow rows left behind.
"""
from __future__ import annotations

import importlib

import pytest

slots = importlib.import_module("hugpy_engine.serve.slots")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30


class _Pool(slots.SlotPool):
    def __init__(self, statuses):
        super().__init__(urls=[s["_control"] for s in statuses])
        self._statuses = statuses
        self.unloaded = []
        self.loaded = []

    def statuses(self):
        return [dict(s) for s in self._statuses]

    def unload(self, control_url):
        self.unloaded.append(control_url)
        for s in self._statuses:
            if s["_control"] == control_url:
                s["model_key"] = None
                s["healthy"] = True
        return {"ok": True}


def _seat(url, mk, last_used=1.0):
    return {"_control": url, "model_key": mk, "healthy": True, "busy": False,
            "endpoint": url + "/infer", "last_used": last_used}


@pytest.fixture
def hooks(monkeypatch):
    saved = (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
             slots._MAKE_ROOM, slots._EVICT_VERB)
    monkeypatch.setattr(slots, "_get", lambda url, timeout=3.0: {})
    slots.set_eviction_policy(lambda mk: True)
    slots.set_residency_lookup(lambda mk: "on-demand")
    slots.set_make_room(None)
    slots.set_evict_verb(None)
    try:
        D._NO_MAKEROOM.set(False)
    except Exception:  # noqa: BLE001
        pass
    yield
    (slots._EVICTION_POLICY, slots._FIT_CHECK, slots._RESIDENCY_LOOKUP,
     slots._MAKE_ROOM, slots._EVICT_VERB) = saved


def _loader(pool):
    def fake_post(url, body, timeout):
        pool.loaded.append((url, body))
        return {"endpoint": url.replace("/load", "") + "/infer"}
    return fake_post


def test_over_ceiling_seat_asks_the_admission_once_and_never_evicts_itself(hooks, monkeypatch):
    """INVARIANT (F1): over the real-VRAM ceiling the pool consults the
    registered admission (the worker's plan_fit path) EXACTLY once and evicts
    nobody itself — the admission's named evictions are the only evictions.
    Established: core isolation step 2 (2026-09-29)."""
    pool = _Pool([_seat("http://s0", "pinned-resident", 100.0), _seat("http://s1", None)])
    monkeypatch.setattr(slots, "_post", _loader(pool))
    calls = []

    def admission(mk):
        calls.append(mk)
        pool.unload("http://s0")                 # the worker's verb frees the seat
        return {"action": "evicted", "evicted": ["pinned-resident"],
                "freed_bytes": 6 * GIB, "reason": None}
    slots.set_make_room(admission)
    slots.set_fit_check(lambda mk: not any(s["model_key"] == "pinned-resident"
                                           for s in pool._statuses))
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    assert calls == ["NEW"]
    assert pool.unloaded == ["http://s0"]            # by the admission, once
    assert [b.get("model_key") for _u, b in pool.loaded] == ["NEW"]
    assert isinstance(ep, str) and ep


def test_partial_admission_threads_the_plan_into_the_seat_without_looping(hooks, monkeypatch):
    pool = _Pool([_seat("http://s0", None)])
    monkeypatch.setattr(slots, "_post", _loader(pool))
    calls = []
    slots.set_make_room(lambda mk: calls.append(mk) or {
        "action": "partial", "evicted": [], "n_gpu_layers": -1, "n_cpu_moe": 39,
        "reason": None})
    slots.set_fit_check(lambda mk: False)            # the full need never passes
    pool.endpoint_for("moe", load_timeout=1.0)
    assert calls == ["moe"]                          # one admission, no spin
    body = pool.loaded[0][1]
    assert body["n_gpu_layers"] == -1 and body["n_cpu_moe"] == 39
    assert pool.unloaded == []


def test_refusing_admission_raises_load_refusal_and_seats_nothing(hooks, monkeypatch):
    """INVARIANT (F1): a `refuse` verdict is honest on the seat path too —
    LoadRefusal with the plan's structured reason; nothing is loaded, nothing
    is evicted (never admit-then-OOM). Established: step 2."""
    pool = _Pool([_seat("http://s0", "locked", 100.0), _seat("http://s1", None)])
    monkeypatch.setattr(slots, "_post", _loader(pool))
    reason = {"state": "refused", "reason": "won't fit on GPU: needs 20 GiB",
              "model_key": "huge", "fit_failure": {"kind": "vram_fit", "code": "wont_fit"}}
    slots.set_make_room(lambda mk: {"action": "refuse", "evicted": [], "freed_bytes": 0,
                                    "reason": reason})
    slots.set_fit_check(lambda mk: False)
    with pytest.raises(D.LoadRefusal) as exc:
        pool.endpoint_for("huge", load_timeout=1.0)
    assert exc.value.reason["fit_failure"]["kind"] == "vram_fit"
    assert pool.loaded == [] and pool.unloaded == []


def test_no_admission_registered_means_no_eviction_and_an_honest_warning(hooks, monkeypatch, caplog):
    """INVARIANT (F1): with no admission planner (bare engine) the pool evicts
    NOBODY on the ceiling path — nothing may evict outside plan_fit — and
    proceeds with a warning (never hangs a request). Established: step 2."""
    import logging
    pool = _Pool([_seat("http://s0", "A", 100.0), _seat("http://s1", None)])
    monkeypatch.setattr(slots, "_post", _loader(pool))
    slots.set_fit_check(lambda mk: False)
    with caplog.at_level(logging.WARNING, logger=slots.logger.name):
        ep = pool.endpoint_for("NEW", load_timeout=1.0)
    assert pool.unloaded == [] and isinstance(ep, str)
    assert any("no admission planner is registered" in r.getMessage() for r in caplog.records)
    assert not hasattr(slots.SlotPool, "_evict_coldest_on_demand")


def test_seat_promotion_evicts_through_the_registered_verb(hooks, monkeypatch):
    """INVARIANT (F1/F4b): a seat promotion (every seat occupied) evicts the LRU
    on-demand occupant through the worker's registered eviction verb — the
    same verb the plan executor uses — so it is telemetered, counted and
    forgotten everywhere; a verb that frees nothing skips the victim.
    Established: step 2."""
    pool = _Pool([_seat("http://s0", "cold", 10.0), _seat("http://s1", "warm", 90.0)])
    monkeypatch.setattr(slots, "_post", _loader(pool))
    slots.set_fit_check(None)
    verb_calls = []

    def verb(victim, subject):
        verb_calls.append((victim, subject))
        if victim == "cold":
            return {"evicted": False, "reason": "busy after all"}
        pool.unload("http://s1")
        return {"evicted": True, "vram_freed": 4 * GIB, "host_mode": "slot"}
    slots.set_evict_verb(verb)
    ep = pool.endpoint_for("NEW", load_timeout=1.0)
    assert verb_calls == [("cold", "NEW"), ("warm", "NEW")]
    assert pool.unloaded == ["http://s1"]            # via the verb, never the pool directly
    assert pool.loaded[0][0].startswith("http://s1") and isinstance(ep, str)


def test_a_seat_load_refusal_is_final_in_get_llama_runner(monkeypatch):
    """INVARIANT (F1): a LoadRefusal raised by the seat admission propagates out
    of get_llama_runner's slot-first path — the in-process fallback shares the
    card, so falling back would be the admit-then-OOM the refusal prevents (and
    a polite refusal must fail fast). Established: step 2."""
    G = importlib.import_module("hugpy_engine.llama.runners.get")
    policy = importlib.import_module("hugpy_engine.serve.policy")
    monkeypatch.setattr(policy, "no_local_serving", lambda: False)
    monkeypatch.setattr(G, "_require_profile_ready", lambda mk: None)
    monkeypatch.setattr(G, "_resolve_serving_gguf", lambda mk: None)
    monkeypatch.setattr(G, "vision_projector_for", lambda mk, p=None: None)
    monkeypatch.setattr(G, "get_model_config", lambda mk: (_ for _ in ()).throw(RuntimeError("no cfg")))
    spill = importlib.import_module("hugpy_engine.spill")
    monkeypatch.setattr(spill, "rpc_servers", lambda: None)
    monkeypatch.setattr(slots, "slots_enabled", lambda: True)

    class _Refusing(slots.SlotPool):
        def __init__(self, *a, **k):
            pass

        def endpoint_for(self, model_key, **kw):
            raise D.LoadRefusal({"state": "refused", "reason": "won't fit on GPU",
                                 "model_key": model_key})
    monkeypatch.setattr(slots, "SlotPool", _Refusing)
    with pytest.raises(D.LoadRefusal):
        G._build_runner("refused-model")
