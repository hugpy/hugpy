"""Contention-based LRU residency (operator doctrine, locked 2026-07-11).

Replaces the clock-based on-demand eviction. Doctrine: an on-demand model loads
on call and STAYS resident (hot) until a NEW load needs its resources — then the
least-recently-used yieldable resident gives way. 🔒static/gate-busy/slot-backed
residents never yield; 📌PINNED residents DO yield (operator 2026-07-15: pin is
DESIGNATION/routing persistence, not a resource lock — re-affirmed 2026-07-17).
The idle clock (on_demand_ttl_s) becomes OPT-IN.

Covered here:
  * dispatch.ensure_headroom_for_load — the contention mechanism, on fake runners
    with a fake headroom fn:
      (a) idle 10x the old TTL -> STAYS (no idle eviction by default);
      (b) a new load needing room -> the LRU on-demand resident yields, it fits;
      (c) static never yields under pressure, but a PINNED model does
          (pin = designation, not a VRAM lock);
      (d) a model mid-generation (gate permits) is skipped, the next LRU chosen,
          and it yields only after its permits release;
      (f) nothing yieldable + still doesn't fit -> no new error (load proceeds
          exactly as today), zero evictions.
  * _residency_sweep_once — OPT-IN: skipped unless on_demand_ttl_s is set;
      (e) with on_demand_ttl_s explicitly set the clock sweep still evicts idle.
  * _effective_config / settings round-trip — default -> null/off; set -> on.
  * _worker_fit_check / _worker_evictable — the box-side policy registered onto
    dispatch.

Runs like the other tests here: venv/bin/python tests/test_residency_contention.py
"""
import argparse
import importlib
import os
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# See test_residency_static.py: managers/__init__ star-imports shadow the
# subpackage attrs, so go through import_module to bind the REAL modules the
# agent's relative imports use at runtime.
from hugpy_fleet.worker import agent
dispatch = importlib.import_module("hugpy_engine.dispatch.dispatch")
procutil = importlib.import_module("hugpy_platform.procutil")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ---------------------------------------------------------------------------
# A fake box: fixed capacity, per-model sizes, a live resident set. Its .fits
# and .evict close over the same state, so evicting frees room the next fit-check
# sees — exactly the contract dispatch.ensure_headroom_for_load relies on.
# ---------------------------------------------------------------------------
class FakeBox:
    def __init__(self, capacity, sizes):
        self.capacity = capacity
        self.sizes = dict(sizes)
        self.resident = {}                 # mk -> size
        self.trims = 0

    def load(self, mk):
        self.resident[mk] = self.sizes[mk]

    def free(self):
        return self.capacity - sum(self.resident.values())

    def fits(self, mk):                    # dispatch.set_fit_check
        return self.free() >= self.sizes.get(mk, 0)

    def trim(self):                        # dispatch.set_post_evict_hook
        self.trims += 1


def install(box, *, residency=None, in_flight=None):
    """Wire a FakeBox + policy onto dispatch and seed _INSTANCES/_LAST_USED.

    Returns a restore() that undoes everything (dispatch globals are process-wide;
    each test file is its own process, but we stay tidy)."""
    residency = residency or {}
    in_flight = in_flight or {}
    saved_evict = dispatch.evict
    saved_instances = dict(dispatch._INSTANCES)
    saved_last = dict(dispatch._LAST_USED)

    evicted = []

    def fake_evict(mk, task=None):
        evicted.append(mk)
        box.resident.pop(mk, None)
        for k in [k for k in list(dispatch._INSTANCES) if k[0] == mk]:
            dispatch._INSTANCES.pop(k, None)
        return True

    dispatch.evict = fake_evict
    dispatch.set_fit_check(box.fits)
    dispatch.set_post_evict_hook(box.trim)
    dispatch.set_evictable(
        lambda mk: residency.get(mk, "on-demand") == "on-demand"
        and int(in_flight.get(mk, 0)) == 0)

    # seed residents into dispatch's real caches
    dispatch._INSTANCES.clear()
    dispatch._LAST_USED.clear()
    t0 = time.time()
    for i, mk in enumerate(box.resident):
        dispatch._INSTANCES[(mk, "chat")] = object()
        dispatch._LAST_USED[mk] = t0 + i    # insertion order == recency order

    def restore():
        dispatch.evict = saved_evict
        dispatch.set_fit_check(None)
        dispatch.set_evictable(None)
        dispatch.set_post_evict_hook(None)
        dispatch._INSTANCES.clear()
        dispatch._INSTANCES.update(saved_instances)
        dispatch._LAST_USED.clear()
        dispatch._LAST_USED.update(saved_last)

    return evicted, restore


# --- (a) idle far past the old TTL STAYS (no idle eviction by default) --------
_SETTINGS_BACKUP = dict(agent._RUNTIME_SETTINGS)
agent._RUNTIME_SETTINGS.clear()            # no on_demand_ttl_s -> sweep opt-out
_evicted = []
_orig = (dispatch.last_used_snapshot, dispatch.evict, agent.loaded_model_keys)
try:
    old_ttl = 900
    ancient = time.time() - old_ttl * 10   # idle 10x the old default TTL
    dispatch.last_used_snapshot = lambda: {"m-ancient": ancient}
    dispatch.evict = lambda mk: _evicted.append(mk)
    agent.loaded_model_keys = lambda: ["m-ancient"]
    agent._residency_sweep_once(0.0)
    check("(a) idle 10x the old TTL STAYS — no idle eviction when ttl unset",
          _evicted == [])
finally:
    (dispatch.last_used_snapshot, dispatch.evict, agent.loaded_model_keys) = _orig

# --- (b) a new load needing room -> LRU on-demand yields, it fits -------------
box = FakeBox(capacity=10, sizes={"A": 6, "B": 6})
box.load("A")
evicted, restore = install(box)
try:
    check("(b) precondition: B does not fit next to A", not box.fits("B"))
    out = dispatch.ensure_headroom_for_load("B")
    check("(b) the LRU on-demand resident A yielded", out == ["A"] == evicted)
    check("(b) B now fits after the yield", box.fits("B"))
    check("(b) post-evict trim fired once per eviction", box.trims == 1)
finally:
    restore()

# --- (c) static never yields; a PINNED model DOES yield under pressure --------
# Doctrine RE-CLARIFIED by the operator 2026-07-15 (and re-affirmed 2026-07-17):
# 📌 pin = the worker is DESIGNATED that model (routing persistence across
# restarts), NOT a resource lock — a pinned model yields to contention like any
# on-demand resident (its pin/designation is untouched; it reloads on call).
# 🔒 static is the ONLY residency lock and never yields.
# S=static, P=pinned-but-on-demand, OD=plain on-demand. Only S is protected.
box = FakeBox(capacity=12, sizes={"S": 6, "P": 6, "OD": 6, "B": 6})
for mk in ("S", "P", "OD"):
    box.load(mk)                           # 18 > 12: already over — max pressure
# P is pinned but its RESIDENCY is on-demand (pin never sets residency=static),
# so the evictable predicate (residency-only) treats it as a candidate — exactly
# how _worker_evictable treats a pin (it checks static/in-flight/slot, never pin).
# _LAST_USED order == load order (S, P, OD): S is coldest but PROTECTED (static),
# so the coldest CANDIDATE is P — it yields first. Only 1 yield (6 GiB) is needed
# to seat B (12 - 6 static - 6 freed = room), so OD is never touched.
evicted, restore = install(box, residency={"S": "static"})
try:
    out = dispatch.ensure_headroom_for_load("B")
    check("(c) static never yields", "S" not in evicted)
    check("(c) a pinned-but-on-demand model DOES yield under contention "
          "(pin is designation, not a VRAM lock — 2026-07-15)", "P" in evicted)
    check("(c) the coldest candidate (pinned P) yielded first", out[0] == "P")
finally:
    restore()

# --- (f) nothing yieldable + still doesn't fit -> no new error, zero evicts ---
box = FakeBox(capacity=6, sizes={"S1": 6, "S2": 6, "B": 6})
box.load("S1"); box.load("S2")
evicted, restore = install(box, residency={"S1": "static", "S2": "static"})
try:
    raised = None
    try:
        out = dispatch.ensure_headroom_for_load("B")   # must NOT raise
    except Exception as exc:                            # noqa: BLE001
        raised = exc
    check("(f) nothing yieldable -> ensure_headroom returns, never raises",
          raised is None and out == [])
    check("(f) B still doesn't fit (load proceeds/fails as today)",
          not box.fits("B") and evicted == [])
finally:
    restore()

# --- (d) a model mid-generation is skipped; yields only after release --------
box = FakeBox(capacity=10, sizes={"G": 6, "N": 6, "B": 6, "B2": 6})
box.load("G"); box.load("N")               # G is the LRU (seeded first)
_inflight = {"G": 1, "N": 0}               # G has an active gate permit
evicted, restore = install(box, in_flight=_inflight)
try:
    out = dispatch.ensure_headroom_for_load("B")
    check("(d) the mid-generation LRU G is skipped, next LRU N yields",
          out == ["N"])
    check("(d) the busy model G survived (never ripped mid-generation)",
          "G" not in evicted)
    # G releases its permit -> a later load may now yield it.
    _inflight["G"] = 0
    box.load("B")                          # B is resident now; free=10-6=4
    out2 = dispatch.ensure_headroom_for_load("B2")
    check("(d) after release, G becomes the yield candidate", out2 == ["G"])
finally:
    restore()

# --- (e) explicit on_demand_ttl_s -> the clock sweep still evicts idle --------
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"on_demand_ttl_s": 60})
_evicted = []
_orig = (dispatch.last_used_snapshot, dispatch.evict, agent.loaded_model_keys)
try:
    # slots disabled path: no slot occupants to exempt
    import importlib as _il
    slots = _il.import_module("hugpy_engine.serve.slots")
    _slots_en = slots.slots_enabled
    slots.slots_enabled = lambda: False
    dispatch.last_used_snapshot = lambda: {"m-idle": time.time() - 600}
    dispatch.evict = lambda mk: _evicted.append(mk)
    agent.loaded_model_keys = lambda: ["m-idle"]
    agent._residency_sweep_once(0.0)
    check("(e) explicit on_demand_ttl_s -> clock eviction still works",
          _evicted == ["m-idle"])
    slots.slots_enabled = _slots_en
finally:
    (dispatch.last_used_snapshot, dispatch.evict, agent.loaded_model_keys) = _orig
    agent._RUNTIME_SETTINGS.clear()

# --- settings round-trip: default -> null/off; explicit set -> on ------------
agent._RUNTIME_SETTINGS.clear()
eff = agent._effective_config()
check("default: heartbeat on_demand_ttl_s is null (idle reclamation off)",
      eff["on_demand_ttl_s"] is None)
check("default: source reported as 'default'",
      eff.get("on_demand_ttl_s_source") == "default")

agent._RUNTIME_SETTINGS.update({"on_demand_ttl_s": 300})
eff = agent._effective_config()
check("explicit set: heartbeat carries the value", eff["on_demand_ttl_s"] == 300)
check("explicit set: source reported as 'settings'",
      eff.get("on_demand_ttl_s_source") == "settings")
agent._RUNTIME_SETTINGS.clear()

# --- /ops/config round-trip: set, then null clears (idle reclamation off) -----
_reexec_orig = procutil.reexec
procutil.reexec = lambda: None
try:
    tmpd = tempfile.mkdtemp(prefix="hugpy-contention-test-")
    state = agent.WorkerState(name="t", url=None, worker_id="w-c")
    state.args = argparse.Namespace(id_file=os.path.join(tmpd, "agent.id"))
    client = agent.build_app(state).test_client()

    r = client.post("/ops/config", json={"on_demand_ttl_s": 300})
    body = r.get_json()
    check("POST on_demand_ttl_s=300 -> 200 ok",
          r.status_code == 200 and body["settings"]["on_demand_ttl_s"] == 300)

    r = client.post("/ops/config", json={"on_demand_ttl_s": 5})   # below 60
    check("POST below the 60s floor -> 400", r.status_code == 400)

    r = client.post("/ops/config", json={"on_demand_ttl_s": None})
    check("POST null clears it (idle reclamation off, the default)",
          r.status_code == 200 and "on_demand_ttl_s" not in r.get_json()["settings"])
finally:
    procutil.reexec = _reexec_orig

# --- worker-side predicates registered onto dispatch -------------------------
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"residency": {"m-stat": "static"},
                                "pinned": {"m-pin": True}})
_gate_orig = agent.gen_gate.in_flight
_slotkeys = importlib.import_module(
    "hugpy_engine.llama.runners.get")
_sbk_orig = _slotkeys.slot_backed_model_keys
try:
    agent.gen_gate.in_flight = lambda mk: 1 if mk == "m-busy" else 0
    _slotkeys.slot_backed_model_keys = lambda: {"m-slot"}
    check("_worker_evictable: static never yields",
          agent._worker_evictable("m-stat") is False)
    check("_worker_evictable: a pinned model DOES yield (pin = designation, not "
          "a VRAM lock — operator 2026-07-15)",
          agent._worker_evictable("m-pin") is True)
    check("_worker_evictable: mid-generation (gate permits) never yields",
          agent._worker_evictable("m-busy") is False)
    check("_worker_evictable: slot-backed never yields (weights elsewhere)",
          agent._worker_evictable("m-slot") is False)
    check("_worker_evictable: a plain idle on-demand resident yields",
          agent._worker_evictable("m-od") is True)
finally:
    agent.gen_gate.in_flight = _gate_orig
    _slotkeys.slot_backed_model_keys = _sbk_orig

# _worker_fit_check: fits when free VRAM holds the need; contended otherwise.
_need_orig = agent._incoming_need_bytes
_fv_orig = agent._free_vram_bytes
_fr_orig = agent._free_ram_bytes
try:
    agent._incoming_need_bytes = lambda mk: 8 * 2**30      # needs 8 GiB
    agent._free_vram_bytes = lambda: 10 * 2**30            # 10 GiB free VRAM
    agent._free_ram_bytes = lambda: 0
    check("_worker_fit_check: fits when free VRAM >= need",
          agent._worker_fit_check("m") is True)
    agent._free_vram_bytes = lambda: 4 * 2**30             # only 4 GiB free VRAM
    check("_worker_fit_check: contended when free VRAM < need (GPU box)",
          agent._worker_fit_check("m") is False)
    agent._incoming_need_bytes = lambda mk: None           # unknown size
    check("_worker_fit_check: fails OPEN on unknown size (never blocks a load)",
          agent._worker_fit_check("m") is True)
finally:
    agent._incoming_need_bytes = _need_orig
    agent._free_vram_bytes = _fv_orig
    agent._free_ram_bytes = _fr_orig
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update(_SETTINGS_BACKUP)

print(f"\nall {ok} checks passed")
