"""Residency v3 FINAL semantics (operator-locked 2026-07-05, ships as 0.1.133).

Policy axis has exactly TWO tiers: on-demand (the default; no stored entry)
and static (locked seat; permanent with 📌 pin). "Serving" is purely a STATE
(a model in a slot). Truth table covered here:
  * _residency(): default/legacy values read as on-demand; static passes.
  * /ops/config: only "static" stores; null/"on-demand"/"serving"/"warm" all
    clear the override; garbage is rejected.
  * TTL sweep: IN-PROCESS only — yields idle non-static residents (default
    on-demand), spares static, spares SLOT occupants, spares recently-used,
    never unloads a slot.
  * Slot filler (slice 9): fills every empty slot from assigned models —
    static first, then most-recently-used; local GGUF files only; never
    double-books an occupant.
  * Slot promotion: never picks a static or busy occupant; an all-static
    pool fails a load with a clear error instead of evicting.
  * Warm-up on a slot-less box: static always; default on-demand only behind
    the WORKER_PRELOAD gate. On a slots box _kick_provision ALSO kicks the slot
    filler (GGUF seating) AND still warms a STATIC model in-process — the
    transformers-on-slots-box fix, so a static transformers model isn't left a
    hollow 0-VRAM shell; a model already SEATED in a slot is not double-loaded
    (the _slot_occupants guard, not by skipping the warm branch).
  * Prune: residency overrides are assignment-scoped unless pinned.

Runs like the other tests here: venv/bin/python tests/test_residency_static.py
"""
import argparse
import os
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

from hugpy_fleet.worker import agent

# managers/__init__ star-imports shadow the `serve`/`dispatch` subpackage
# attributes (llama's serve.py wins the name), so plain dotted `import x.y.z
# as z` resolves the WRONG module — import_module goes through sys.modules/
# __path__ and gets the real ones (same modules the agent's relative imports
# bind at runtime).
slots = importlib.import_module("hugpy_engine.serve.slots")
dispatch = importlib.import_module("hugpy_engine.dispatch.dispatch")
procutil = importlib.import_module("hugpy_platform.procutil")
provision = importlib.import_module("hugpy_storage.provision")
agent_imports = importlib.import_module("hugpy_fleet.worker.imports")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


_SETTINGS_BACKUP = dict(agent._RUNTIME_SETTINGS)

# --- _residency(): two tiers, on-demand default ------------------------------
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"residency": {
    "m-static": "static", "m-od": "on-demand", "m-warm": "warm"}})
check("static passes through _residency", agent._residency("m-static") == "static")
check("stored on-demand reads as on-demand", agent._residency("m-od") == "on-demand")
check("legacy warm reads as the on-demand default",
      agent._residency("m-warm") == "on-demand")
check("unknown key -> on-demand default", agent._residency("m-x") == "on-demand")
# the registered slot eviction policy is `_residency(mk) == "on-demand"`:
check("eviction policy: default tier is promotable",
      agent._residency("m-x") == "on-demand")
check("eviction policy: static is immovable",
      agent._residency("m-static") != "on-demand")
check("heartbeat passthrough carries static",
      agent._effective_config().get("residency", {}).get("m-static") == "static")

# --- /ops/config: static stores; everything else clears ---------------------
_reexec_orig = procutil.reexec
procutil.reexec = lambda: None          # the route re-execs 0.5s after a POST
try:
    tmpd = tempfile.mkdtemp(prefix="hugpy-residency-test-")
    state = agent.WorkerState(name="t", url=None, worker_id="w-test")
    state.args = argparse.Namespace(id_file=os.path.join(tmpd, "agent.id"))
    client = agent.build_app(state).test_client()

    r = client.post("/ops/config", json={"residency": {"m1": "static"}})
    body = r.get_json()
    check("POST residency static -> 200 ok", r.status_code == 200 and body["ok"] is True)
    check("static persisted in settings", body["settings"]["residency"]["m1"] == "static")
    check("settings file on disk has static",
          '"static"' in open(state.args.id_file + ".settings.json").read())

    r = client.post("/ops/config", json={"residency": {"m1": "banana"}})
    body = r.get_json()
    check("garbage residency -> 400", r.status_code == 400)
    check("error message names static as valid", "static" in body["error"]["message"])

    r = client.post("/ops/config", json={"residency": {"m1": "on-demand"}})
    check("explicit on-demand clears the override (it IS the default)",
          r.status_code == 200 and "residency" not in r.get_json()["settings"])

    r = client.post("/ops/config", json={"residency": {"m1": "static"}})
    check("re-stored static", r.get_json()["settings"]["residency"]["m1"] == "static")
    r = client.post("/ops/config", json={"residency": {"m1": "serving"}})
    check("retired 'serving' residency word is rejected (no legacy alias)",
          r.status_code == 400)

    r = client.post("/ops/config", json={"residency": {"m1": None}})
    check("null clears (idempotent)",
          r.status_code == 200 and "residency" not in r.get_json()["settings"])
finally:
    procutil.reexec = _reexec_orig

# --- TTL sweep: in-process only; slot occupants + static + fresh exempt -----
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"residency": {"m-static": "static"},
                                "on_demand_ttl_s": 60})
_evicted = []
_orig = (dispatch.last_used_snapshot, dispatch.evict,
         agent.loaded_model_keys, slots.slots_enabled, slots.SlotPool)
class _SweepPool:
    def __init__(self, urls=None):
        pass
    def statuses(self):
        return [{"_control": "u1", "model_key": "m-seated", "busy": False,
                 "last_used": 1.0}]
    def unload(self, control):
        raise AssertionError("the sweep must NEVER unload a slot occupant")
try:
    dispatch.last_used_snapshot = lambda: {"m-fresh": time.time()}
    dispatch.evict = lambda mk: _evicted.append(mk)
    agent.loaded_model_keys = lambda: ["m-idle", "m-static", "m-seated", "m-fresh"]
    slots.slots_enabled = lambda: True
    slots.SlotPool = _SweepPool
    agent._residency_sweep_once(0.0)               # everything idle for eons
    check("sweep yields the idle in-process resident (default on-demand)",
          "m-idle" in _evicted)
    check("sweep spares static", "m-static" not in _evicted)
    check("sweep spares SLOT occupants (slots stay filled)",
          "m-seated" not in _evicted)
    check("sweep spares recently-used", "m-fresh" not in _evicted)
    check("exactly one eviction", _evicted == ["m-idle"])

    # no stored overrides at all — the sweep still runs (default on-demand);
    # with its static override gone, m-static is just another idle default
    # resident and rightly yields too.
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update({"on_demand_ttl_s": 60})
    _evicted.clear()
    agent._residency_sweep_once(0.0)
    check("sweep runs with an empty residency map",
          _evicted == ["m-idle", "m-static"])
finally:
    (dispatch.last_used_snapshot, dispatch.evict,
     agent.loaded_model_keys, slots.slots_enabled, slots.SlotPool) = _orig

# --- slot filler (slice 9): static first, MRU next, no double-booking -------
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"residency": {"m-static": "static"}})
_seated = []
_orig = (slots.slots_enabled, slots.SlotPool, agent._models_local,
         agent_imports.get_model_config, dispatch.last_used_snapshot,
         dispatch.runner_for)
class _FillPool:
    def __init__(self, urls=None):
        pass
    def statuses(self):
        return [
            {"_control": "u1", "model_key": "m-occupied", "busy": False},
            {"_control": "u2", "model_key": None},
            {"_control": "u3", "model_key": None},
        ]
_orig_fit = agent._worker_fit_check
_orig_fill_env = os.environ.get("HUGPY_SLOT_FILL")
try:
    os.environ["HUGPY_SLOT_FILL"] = "1"          # background seating is operator opt-in (2026-09-10)
    agent._worker_fit_check = lambda mk: True     # background seats must fit beside residents
    slots.slots_enabled = lambda: True
    slots.SlotPool = _FillPool
    agent._models_local = lambda st: ["m-a", "m-b", "m-static", "m-occupied", "m-comfy"]
    agent_imports.get_model_config = lambda mk: types.SimpleNamespace(
        framework=("comfy" if mk == "m-comfy" else "gguf"))
    dispatch.last_used_snapshot = lambda: {"m-b": 100.0, "m-a": 50.0}
    dispatch.runner_for = lambda model_key=None, **kw: _seated.append(model_key)

    st = agent.WorkerState(name="t", url=None, worker_id="w-fill")
    st.assigned_models = ["m-a", "m-b", "m-static", "m-occupied",
                          "m-comfy", "m-notlocal"]
    agent._fill_empty_slots(st)
    check("filler seats one model per empty slot", len(_seated) == 2)
    check("static seated FIRST", _seated[0] == "m-static")
    check("then the most-recently-used", _seated[1] == "m-b")
    check("never double-books a current occupant", "m-occupied" not in _seated)
    check("skips non-GGUF rows (slots host llama.cpp only)",
          "m-comfy" not in _seated)
    check("skips models whose files aren't local yet",
          "m-notlocal" not in _seated)

    # nothing to do -> no loads
    _seated.clear()
    st.assigned_models = ["m-occupied"]
    agent._fill_empty_slots(st)
    check("no candidates -> no loads", _seated == [])
finally:
    (slots.slots_enabled, slots.SlotPool, agent._models_local,
     agent_imports.get_model_config, dispatch.last_used_snapshot,
     dispatch.runner_for) = _orig
    agent._worker_fit_check = _orig_fit
    if _orig_fill_env is None:
        os.environ.pop("HUGPY_SLOT_FILL", None)
    else:
        os.environ["HUGPY_SLOT_FILL"] = _orig_fill_env

# --- slot promotion: static occupants are immovable --------------------------
_RES = {}
_POSTS = []
_STATUS = {}
_get_orig, _post_orig = slots._get, slots._post
def _fake_get(url, timeout=3.0):
    return dict(_STATUS[url.rsplit("/status", 1)[0]])
def _fake_post(url, body, timeout):
    _POSTS.append((url, dict(body)))
    if url.endswith("/load"):
        return {"endpoint": "ep-" + body.get("model_key", "?")}
    return {}
try:
    slots._get, slots._post = _fake_get, _fake_post
    slots.set_eviction_policy(lambda mk: _RES.get(mk, "on-demand") == "on-demand")
    slots.set_residency_lookup(lambda mk: _RES.get(mk, "on-demand"))
    pool = slots.SlotPool(urls=["http://a", "http://b"])

    # mixed static + default: the default-tier seat is the ONLY victim
    _RES.update({"m-stat": "static"})
    _STATUS.update({
        "http://a": {"model_key": "m-stat", "healthy": True, "busy": False, "last_used": 1},
        "http://b": {"model_key": "m-od",   "healthy": True, "busy": False, "last_used": 2},
    })
    ep = pool.endpoint_for("m-new", load_timeout=5.0)
    check("promotion seats the newcomer", ep == "ep-m-new")
    check("promotion bumped the default-tier occupant",
          ("http://b/unload", {}) in _POSTS)
    check("promotion never touched the static occupant",
          not any(u.startswith("http://a/") for u, _ in _POSTS))

    # every seat static-locked: clear error, never an eviction
    _POSTS.clear()
    _RES.clear(); _RES.update({"m-s1": "static", "m-s2": "static"})
    _STATUS["http://a"] = {"model_key": "m-s1", "healthy": True, "busy": False, "last_used": 1}
    _STATUS["http://b"] = {"model_key": "m-s2", "healthy": True, "busy": False, "last_used": 2}
    err = None
    try:
        pool.endpoint_for("m-new", load_timeout=5.0)
    except RuntimeError as exc:
        err = str(exc)
    check("all-static pool raises a clear error",
          err is not None and "static-locked" in err)
    check("all-static pool made zero unload/load calls", _POSTS == [])

    # a BUSY default-tier occupant is never bumped (promotion skips busy seats)
    _POSTS.clear()
    _RES.clear(); _RES.update({"m-s1": "static"})
    _STATUS["http://b"] = {"model_key": "m-busy", "healthy": True, "busy": True, "last_used": 2}
    check("busy occupants are never evicted (falls back to None/swap)",
          pool.endpoint_for("m-new", load_timeout=5.0) is None and _POSTS == [])

    # bare central (no lookup registered): all-busy stays None, never raises
    slots.set_residency_lookup(None)
    _RES.clear(); _RES.update({"m-s1": "static", "m-s2": "static"})
    _STATUS["http://b"] = {"model_key": "m-s2", "healthy": True, "busy": False, "last_used": 2}
    check("no lookup registered -> historical None fallback",
          pool.endpoint_for("m-new", load_timeout=5.0) is None)
finally:
    slots._get, slots._post = _get_orig, _post_orig
    slots.set_eviction_policy(None)
    slots.set_residency_lookup(None)

# --- warm-up: slot-less box vs slots box (no double-loading) -----------------
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({"residency": {"m-static": "static",
                                              "m-static2": "static"}})
_warmed, _fills = [], []
_prov_orig = (provision.ensure_model_present, provision.model_is_local,
              provision.ensure_model_registered)
_gmc_orig = agent_imports.get_model_config
_runner_orig = dispatch.runner_for
_slots_en_orig = slots.slots_enabled
_fill_orig = agent._fill_empty_slots
_env_orig = os.environ.get("WORKER_PRELOAD")
def _wait_done(st):
    deadline = time.time() + 10.0
    while time.time() < deadline:
        with st._provision_lock:
            if not st._provisioning:
                return True
        time.sleep(0.05)
    return False
try:
    provision.ensure_model_present = lambda mk, url, progress=None: None
    provision.model_is_local = lambda mk: True     # files already on disk
    provision.ensure_model_registered = lambda mk, url: mk
    agent_imports.get_model_config = lambda mk: types.SimpleNamespace(framework=None)
    dispatch.runner_for = lambda model_key=None, **kw: _warmed.append(model_key)

    st = agent.WorkerState(name="t", url=None, worker_id="w-test2")

    # slot-less box
    slots.slots_enabled = lambda: False
    os.environ["WORKER_PRELOAD"] = "0"             # gate OFF
    agent._kick_provision(st, "m-static")
    check("provision thread finished (static)", _wait_done(st))
    check("no slots: static eager-warms with the gate OFF", _warmed == ["m-static"])

    agent._kick_provision(st, "m-default")
    check("provision thread finished (default, gate off)", _wait_done(st))
    check("no slots: default on-demand does NOT warm when gate is OFF",
          "m-default" not in _warmed)

    os.environ["WORKER_PRELOAD"] = "1"             # gate ON
    agent._kick_provision(st, "m-default")
    check("provision thread finished (default, gate on)", _wait_done(st))
    check("no slots: default on-demand DOES warm behind the gate "
          "(old on-demand-never-preloads rule retired)", "m-default" in _warmed)

    # slots box: the slot filler is kicked for slot-eligible (GGUF) seating, AND
    # a STATIC model still warms IN-PROCESS. Doctrine FIX (see _kick_provision,
    # 2026-07): the old `elif _has_slots` skipped the warm branch on a slots box,
    # so a STATIC TRANSFORMERS model (the slot filler only seats GGUF) was left a
    # hollow 0-VRAM shell that read "loaded". Now static warms here regardless of
    # slots; the seated-GGUF double-load is avoided by the _slot_occupants() check,
    # not by skipping the branch. m-static2 is static -> it warms in-process.
    _warmed.clear()
    slots.slots_enabled = lambda: True
    agent._fill_empty_slots = lambda s: _fills.append(True)
    agent._kick_provision(st, "m-static2")
    check("provision thread finished (slots box)", _wait_done(st))
    check("slots box: a STATIC model still warms in-process (transformers-on-"
          "slots-box fix — no hollow 0-VRAM shell)", _warmed == ["m-static2"])
    check("slots box: the slot filler is ALSO kicked (GGUF seating)",
          _fills == [True])

    # no double-loading (the ORIGINAL intent, via the CURRENT mechanism): a model
    # already SEATED IN A SLOT is skipped by the _slot_occupants() guard, so a
    # seated GGUF is never warmed a second time in-process.
    _warmed.clear(); _fills.clear()
    _occ_orig = agent._slot_occupants
    try:
        agent._slot_occupants = lambda *a, **k: {"m-static2"}   # already seated
        agent._kick_provision(st, "m-static2")
        check("provision thread finished (seated static)", _wait_done(st))
        check("slots box: a SEATED model is NOT double-loaded in-process",
              _warmed == [])
    finally:
        agent._slot_occupants = _occ_orig
finally:
    (provision.ensure_model_present, provision.model_is_local,
     provision.ensure_model_registered) = _prov_orig
    agent_imports.get_model_config = _gmc_orig
    dispatch.runner_for = _runner_orig
    slots.slots_enabled = _slots_en_orig
    agent._fill_empty_slots = _fill_orig
    if _env_orig is None:
        os.environ.pop("WORKER_PRELOAD", None)
    else:
        os.environ["WORKER_PRELOAD"] = _env_orig

# --- residency overrides are assignment-scoped unless pinned -----------------
# static-without-pin ends at unassign; pinned overrides survive; assigned
# overrides untouched. Lazy cleanup runs inside assignment adoption
# (_sync_assignment -> _prune_stale_residency), live + persisted, no re-exec.
tmpd2 = tempfile.mkdtemp(prefix="hugpy-residency-prune-")
args2 = argparse.Namespace(id_file=os.path.join(tmpd2, "agent.id"))
_seed = {
    "residency": {"m-keep": "static", "m-gone-static": "static",
                  "m-gone-legacy": "on-demand", "m-gone-pin": "static"},
    "pinned": {"m-gone-pin": True},
}
agent._save_settings(args2, {k: dict(v) for k, v in _seed.items()})
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update({k: dict(v) for k, v in _seed.items()})
st2 = agent.WorkerState(name="t", url=None, worker_id="w-test3")
st2.args = args2
st2.assigned_models = ["m-keep"]     # list unchanged -> no provisioning kicks
try:
    # a response WITHOUT an authoritative models list must never prune
    agent._sync_assignment(st2, {"status": "ok"})
    check("no models list in response -> nothing pruned",
          set(agent._RUNTIME_SETTINGS["residency"]) ==
          {"m-keep", "m-gone-static", "m-gone-legacy", "m-gone-pin"})

    # authoritative list arrives: unassigned+unpinned overrides drop,
    # pinned survives, assigned untouched
    agent._sync_assignment(st2, {"models": ["m-keep"]})
    res_now = agent._RUNTIME_SETTINGS.get("residency") or {}
    check("assigned static override untouched", res_now.get("m-keep") == "static")
    check("unassigned+unpinned static dropped", "m-gone-static" not in res_now)
    check("unassigned stale legacy entry dropped", "m-gone-legacy" not in res_now)
    check("unassigned+PINNED static survives", res_now.get("m-gone-pin") == "static")
    on_disk = agent._load_settings(args2)
    check("prune persisted to the settings file",
          set(on_disk.get("residency") or {}) == {"m-keep", "m-gone-pin"})
    check("pin map itself untouched by the prune",
          on_disk.get("pinned") == {"m-gone-pin": True})
finally:
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update(_SETTINGS_BACKUP)

print(f"\nall {ok} checks passed")
