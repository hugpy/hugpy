"""Central storage-budget + LRU eviction PROPOSAL (read-time) regression.

Covers the central-only pieces of the storage-budget feature:
  * storage_proposal(): budget modes (free-disk reserve default vs explicit
    per-worker disk_cache_gib cap — cap WINS), the over_budget flag, and the
    LRU-ordered greedy proposed_evictions[] (least-recently-picked first, cold
    'never served' = last_picked 0 first, greedy until `need` covered).
  * Guards: static / assigned / loaded / loading / provisioning are NEVER
    proposed — worker-reported protected flag OR the central redundant guard
    (slot-merged loaded_models/loading/provisioning, config residency).
    📌pin is DELIBERATELY NOT a guard (operator, 2026-07-17): pin designates only
    that the allocation/routing survives restarts — it has NO bearing on
    eviction, so a pinned model is a normal LRU candidate (its `pinned` flag is
    attribution info only; `protected` stays False).
  * _public_view spreads the derived `storage` sub-object.
  * heartbeat() stores the worker-reported `storage` verbatim.
  * pick_for_model() stamps the per-(worker,model) `model_last_picked` LRU key.
  * unassign prunes the LRU stamp; set_limits accepts disk_cache_gib.
  * storage_view() recomputes from the RAW record (the /reap-approve 2nd guard).

Runs like the other tests here: venv/bin/python tests/test_storage_budget.py
"""
import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

from worker_store_isolation import isolated_worker_store

W = importlib.import_module(
    "hugpy_fleet.central.workers")

GiB = 1 << 30
ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# Deterministic reserve (default is 50 GiB, but pin it so the env can't drift).
os.environ["HUGPY_WORKER_DISK_RESERVE_GIB"] = "50"


def _worker(**over):
    w = {
        "id": "w", "name": "w", "url": "http://w",
        # ALIVE. Added 2026-07-16: the provisioning guard is now liveness-gated
        # (a dead worker's pull is not in flight), so a fixture with no
        # last_seen reads as OFFLINE and its provisioning entries confer no
        # protection. This fixture means "a healthy worker", so say so.
        "last_seen": time.time(),
        "disk": {"free_bytes": 10 * GiB, "total_bytes": 500 * GiB},
        # pinnedm gets the freshest stamp so it sorts LAST — a pinned model is a
        # candidate now (pin doesn't protect), but keeping it warmest leaves the
        # LRU-order assertions below (never/cold first) intact.
        "model_last_picked": {"warm": 5000.0, "cold": 1000.0,
                              "pinnedm": 9000.0},
        "loaded_models": [], "loading": [], "provisioning": [],
        "config": {}, "limits": {},
        "storage": {
            "cache_used_bytes": 200 * GiB,
            "disk_free": 10 * GiB,          # < 50 GiB reserve -> over budget
            "models": [
                {"model_key": "warm", "bytes": 30 * GiB, "protected": False},
                {"model_key": "cold", "bytes": 30 * GiB, "protected": False},
                {"model_key": "never", "bytes": 25 * GiB, "protected": False},
                # 📌 pin no longer protects files (2026-07-17): the worker
                # reports it as ATTRIBUTION (pinned:true) but NOT protected.
                {"model_key": "pinnedm", "bytes": 100 * GiB,
                 "protected": False, "pinned": True},
                {"model_key": "assignedm", "bytes": 20 * GiB, "assigned": True},
            ],
        },
    }
    w.update(over)
    return w


# --- reserve mode: over budget, LRU greedy proposal --------------------------
p = W.storage_proposal(_worker())
check("reserve mode: budget_basis is reserve", p["budget_basis"] == "reserve")
check("reserve mode: over_budget (disk_free 10 < reserve 50)", p["over_budget"] is True)
check("reserve mode: need = reserve - disk_free = 40 GiB",
      p["need_bytes"] == 40 * GiB)
prop_keys = [e["model_key"] for e in p["proposed_evictions"]]
check("LRU order: a never-served (last_picked=0) model is proposed first",
      prop_keys[0] in ("assignedm", "never"))
# `assigned` is a CANDIDATE too (operator 2026-07-17: allocation = routing
# only, no bearing on eviction). Both never-picked models tie on idle AND on
# calls (0 each), so the SPEC'S KEY ④ — model_key, the stable final tiebreak —
# decides: "assignedm" < "never" alphabetically.
#
# ⚠ THIS ORDER CHANGED (spec 2026-07-25, assets/evictionflow.html). It used to
# be decided by the ``-bytes`` largest-first tiebreak, which put never (25 GiB)
# ahead of assignedm (20 GiB). SIZE IS NO LONGER IN THE SORT KEY: the spec
# replaces it with walk-then-drop, which achieves "fewest deletes" without
# letting size masquerade as a measure of how expendable a model is. The SET is
# unchanged (both still go, 45 GiB covers the 40 GiB need); only the order.
check("key ④ decides the idle+calls tie: model_key, ascending",
      prop_keys[1] == "never")
check("the walk stops once need is covered (assignedm 20 + never 25 >= 40 GiB)",
      prop_keys == ["assignedm", "never"])
check("proposed_free_bytes = sum of proposed (45 GiB)",
      p["proposed_free_bytes"] == 45 * GiB)
check("warm (most-recently-picked) is NOT proposed", "warm" not in prop_keys)
# pinnedm is a CANDIDATE now (pin doesn't protect), but it has the freshest
# last_picked so the greedy cut never reaches it — NOT because it is
# protected. The next check proves it is genuinely unprotected.
check("pinned model NOT proposed here only because it is warmest, not protected",
      "pinnedm" not in prop_keys)
check("cold survives because greedy stopped, not because of protection",
      "cold" not in prop_keys)
by = {m["model_key"]: m for m in p["models"]}
check("pinned model is UNPROTECTED (pin doesn't shield files) but flagged pinned",
      by["pinnedm"]["protected"] is False and by["pinnedm"]["pinned"] is True)
check("pinned model why is attribution-only 'pinned' while protected stays False",
      by["pinnedm"]["why"] == "pinned")
check("assigned-flag model is a CANDIDATE (allocation = routing only, 2026-07-17)",
      by["assignedm"]["protected"] is False)
check("never-served model last_picked is None (no central stamp)",
      by["never"]["last_picked"] is None)


# --- explicit cap WINS over the free-disk reserve ----------------------------
# disk_free 100 GiB is ABOVE the 50 GiB reserve (reserve mode would NOT trip),
# but the 150 GiB cap is BELOW the 200 GiB cache_used -> cap-mode over budget.
# Reserve (50) is carved OUT of the allocation: 200 allocated -> 150 effective.
capw = _worker(limits={"disk_cache_gib": 200},
               disk={"free_bytes": 100 * GiB, "total_bytes": 500 * GiB})
capw["storage"]["disk_free"] = 100 * GiB
pc = W.storage_proposal(capw)
check("cap mode: budget_basis is cap", pc["budget_basis"] == "cap")
check("cap mode: budget == allocation - reserve (200 - 50 = 150 GiB)", pc["budget"] == 150 * GiB)
check("cap mode: allocation_bytes surfaced (200 GiB)", pc["allocation_bytes"] == 200 * GiB)
check("cap mode: budget_sources shows the carve-out",
      pc["budget_sources"]["allocation_gib"] == 200.0
      and pc["budget_sources"]["reserve_gib"] == 50.0
      and pc["budget_sources"]["effective_gib"] == 150.0)
check("cap wins: over budget on cache_used>cap even though disk_free>reserve",
      pc["over_budget"] is True and pc["need_bytes"] == 50 * GiB)
cap_keys = [e["model_key"] for e in pc["proposed_evictions"]]
# SAME shared function in cap mode — and here the DROP PASS visibly bites.
# need = 50 GiB. The walk goes assignedm(20) -> never(25) -> cold(30) = 75 GiB,
# reaching cold because 45 was short. The drop pass then removes assignedm: the
# remaining {never 25, cold 30} = 55 GiB already covers 50 on its own. Two
# models unloaded instead of three — least reaping, exactly as the spec's y1
# example describes ("if y1 is first by order but y2 must go anyway and y2
# alone covers the need, y1 is spared").
#
# ⚠ WAS ["never", "assignedm", "cold"] under the old greedy walk, which had no
# drop pass and so unloaded all three. `assigned` remains a CANDIDATE (operator
# 2026-07-17: allocation = routing only) — it is spared here on merit, not
# protected.
check("cap mode: walk-then-drop spares assignedm (never+cold alone cover 50 GiB)",
      cap_keys == ["never", "cold"])


# --- central redundant guard: slot-merged loaded/loading/provisioning --------
# UPDATED 2026-07-16 (defect: stale `provisioning` never aged out). The
# provisioning guard is now LIVENESS-GATED: it protects only a pull that is
# genuinely in flight (owner alive AND bytes moving). This block always MEANT
# "all candidates are live", so it now supplies the forward progress that makes
# "warm" actually live. The old fixture asserted that a bare flag from a worker
# with no liveness evidence at all protects a model — that was the bug (op sat
# offline 2h+ with 4 immortal entries).
# pinnedm is now an UNPROTECTED candidate (pin doesn't shield files), so for the
# "all candidates live" scenario to hold, mark it loaded too — otherwise it (the
# only remaining unprotected model) would be proposed and this test's premise
# ("nothing left to propose") would no longer be about the live-guard.
# assignedm too: assigned is a candidate since 2026-07-17 (allocation = routing
# only), so it must be live-guarded here for the same reason as pinnedm above.
guardw = _worker(loaded_models=["never", "pinnedm", "assignedm"], loading=["cold"],
                 provisioning=["warm"],
                 provision_progress={"warm": {"done_bytes": 1 << 30,
                                              "total_bytes": 50 << 30,
                                              "progressed_at": time.time()}})
pg = W.storage_proposal(guardw)
check("central guard: nothing proposed when all candidates are live",
      pg["proposed_evictions"] == [])
bg = {m["model_key"]: m for m in pg["models"]}
check("central guard: slot-merged loaded -> protected/why=loaded",
      bg["never"]["protected"] and bg["never"]["why"] == "loaded")
check("central guard: loading -> protected/why=loading",
      bg["cold"]["protected"] and bg["cold"]["why"] == "loading")
check("central guard: LIVE provisioning -> protected/why=provisioning",
      bg["warm"]["protected"] and bg["warm"]["why"] == "provisioning")

# --- NEW CONTRACT (2026-07-16): a DEAD pull protects nothing ----------------
# The converse of the guard above, and the actual defect: a provisioning entry
# whose owner is gone used to grant PERMANENT phantom eviction protection,
# quietly shrinking the reclaimable pool on a full disk. Full coverage lives in
# tests/test_provisioning_liveness.py; these two pin the behaviour here too,
# next to the guard they qualify.
deadw = _worker(provisioning=["warm"])           # alive, but ZERO bytes moving
dg = {m["model_key"]: m for m in W.storage_proposal(deadw)["models"]}
check("central guard: provisioning with NO progress -> NOT protected",
      not dg["warm"]["protected"])

offw = _worker(provisioning=["warm"])
offw["last_seen"] = time.time() - 7750.0         # the op case: offline 2h+
og = {m["model_key"]: m for m in W.storage_proposal(offw)["models"]}
check("central guard: provisioning on an OFFLINE worker -> NOT protected",
      not og["warm"]["protected"])


# --- central guard: config residency=static (pin is NOT a guard) -------------
cfgw = _worker(config={"residency": {"never": "static"}, "pinned": {"cold": True}})
pcfg = W.storage_proposal(cfgw)
kcfg = [e["model_key"] for e in pcfg["proposed_evictions"]]
bcfg = {m["model_key"]: m for m in pcfg["models"]}
check("config static excluded from proposal", "never" not in kcfg)
# 📌 config pin does NOT protect (2026-07-17): `cold` is config-pinned yet must
# remain a normal LRU candidate — pin annotates, never shields.
check("config pinned is a CANDIDATE (proposed), NOT excluded", "cold" in kcfg)
check("config pinned model is flagged pinned but UNprotected",
      bcfg["cold"]["pinned"] is True and bcfg["cold"]["protected"] is False)


# --- 📌 a pinned model IS proposed when the FIFO reaches it (2026-07-17) ------
# The direct proof of the ruling: make the ONLY reclaimable candidate a pinned
# model and force an over-budget state. It must be proposed for eviction.
pinonly = _worker()
pinonly["storage"]["models"] = [
    {"model_key": "pinnedm", "bytes": 100 * GiB, "protected": False,
     "pinned": True},
]
pinonly["model_last_picked"] = {"pinnedm": 1.0}
pp = W.storage_proposal(pinonly)
ppk = [e["model_key"] for e in pp["proposed_evictions"]]
check("pinned model IS proposed for eviction when it's the FIFO candidate",
      pp["over_budget"] is True and ppk == ["pinnedm"])


# --- under budget: no proposal ----------------------------------------------
underw = _worker(disk={"free_bytes": 300 * GiB, "total_bytes": 500 * GiB})
underw["storage"]["disk_free"] = 300 * GiB
pu = W.storage_proposal(underw)
check("under budget (disk_free 300 > reserve 50) -> not over_budget",
      pu["over_budget"] is False)
check("under budget -> empty proposal", pu["proposed_evictions"] == [])


# --- pre-feature agent: no storage field -> monitoring-only, no proposal -----
prew = {"id": "w", "disk": {"free_bytes": 5 * GiB, "total_bytes": 500 * GiB}}
pp = W.storage_proposal(prew)
check("no storage field -> reported False", pp["reported"] is False)
check("no storage field -> empty proposal (no per-model inventory)",
      pp["proposed_evictions"] == [])
check("no storage field -> disk_free still read from worker['disk']",
      pp["disk_free"] == 5 * GiB)


# --- store integration: heartbeat stores storage, _public_view derives -------
# k3 isolation: isolated_worker_store() ALSO redirects the assignment-memory
# sidecar (settings.manifest_path) — a bare WorkerStore(path=tmp) below would
# isolate the worker rows but the later unassign_model() call still writes
# worker_assignments.json beside the real live manifest. See
# tests/worker_store_isolation.py.
store, tmp = isolated_worker_store(prefix="hugpy-storage-test-")
store.register(name="box", url="http://box", worker_id="wid1", models=["m1"])
store.set_admission("wid1", "approved")
view = store.heartbeat(
    "wid1",
    disk={"free_bytes": 10 * GiB, "total_bytes": 500 * GiB},
    storage={"cache_used_bytes": 200 * GiB, "disk_free": 10 * GiB,
             "models": [{"model_key": "leftover", "bytes": 80 * GiB,
                         "protected": False}]},
)
check("heartbeat return derives storage view", "storage" in view)
check("_public_view: over_budget surfaced", view["storage"]["over_budget"] is True)
check("_public_view: leftover proposed for eviction",
      [e["model_key"] for e in view["storage"]["proposed_evictions"]] == ["leftover"])
raw = store._load()["wid1"]
check("heartbeat stored the raw storage survey verbatim",
      raw["storage"]["cache_used_bytes"] == 200 * GiB)
check("get() also derives the storage view",
      store.get("wid1")["storage"]["over_budget"] is True)


# --- pick_for_model stamps the per-(worker,model) LRU key --------------------
before = store._load()["wid1"].get("model_last_picked", {})
check("no per-model stamp before first pick", "m1" not in before)
picked = store.pick_for_model("m1")
check("pick_for_model returns the assigned+approved worker",
      picked is not None and picked["id"] == "wid1")
stamp = store._load()["wid1"]["model_last_picked"]
check("pick_for_model stamped model_last_picked[m1]", "m1" in stamp)
check("stamp is a fresh epoch", abs(stamp["m1"] - time.time()) < 30)
check("per-WORKER last_picked also stamped (round-robin unchanged)",
      "last_picked" in store._load()["wid1"])


# --- storage_view: raw recompute (the /reap-approve second guard) ------------
sv = store.storage_view("wid1")
check("storage_view recomputes from RAW record",
      [e["model_key"] for e in sv["proposed_evictions"]] == ["leftover"])
check("storage_view(unknown) -> None", store.storage_view("nope") is None)
# The module wrapper delegates to the module singleton worker_store (not this
# test's instance); assert it is wired to that store and unknown ids -> None.
check("worker_storage_view wrapper delegates to the singleton store",
      W.worker_storage_view("definitely-not-a-real-worker-id") is None)


# --- set_limits accepts disk_cache_gib; unassign prunes the LRU stamp --------
lv = store.set_limits("wid1", {"disk_cache_gib": 120})
check("set_limits accepts disk_cache_gib (float)",
      lv["limits"]["disk_cache_gib"] == 120.0)
store.unassign_model("wid1", "m1")
check("unassign prunes model_last_picked[m1]",
      "m1" not in store._load()["wid1"].get("model_last_picked", {}))

print(f"\nall {ok} checks passed")
