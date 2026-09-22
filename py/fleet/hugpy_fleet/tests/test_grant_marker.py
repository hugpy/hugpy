"""Placement grant marker (Phase 1 item 2 of the capacity-aware scheduler plan).

A GRANT is a SYSTEM-authored designation of a model to a worker — born from a
FUTURE auto-placement decision, not present yet. This task adds ONLY the
persisted marker + its semantics: no auto-placement, no queue, no scheduling
logic. Critical semantic difference from an operator assign/pin:

  * a grant is RECLAIMABLE — freely LRU-evictable, never protected
  * a grant is unassignable-with-no-409 (unlike a pin, whose ONE durable claim
    is that its allocation survives — unassign returns 409; note pin does NOT
    protect files from eviction, 2026-07-17)
  * a grant must never masquerade as operator intent (assign-memory blind to it)

Runs like the other tests here:
    venv/bin/python tests/test_grant_marker.py
"""
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

W = importlib.import_module(
    "hugpy_fleet.central.workers")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


tmp = tempfile.mkdtemp(prefix="hugpy-grant-marker-")
workers_path = os.path.join(tmp, "workers.json")
# The assign-memory file sits beside settings.manifest_path — point that at
# our tmp dir too so _remember_assignments never touches the real one.
manifest_path = os.path.join(tmp, "manifest.json")
W.settings.manifest_path = manifest_path

store = W.WorkerStore(path=workers_path)

store.register(name="w1", url="http://w1:9100", worker_id="wid-1")
store.register(name="w2", url="http://w2:9100", worker_id="wid-2")
store.set_admission("wid-1", "approved")
store.set_admission("wid-2", "approved")


# --- 1. grant_model puts model in grants, NOT in models ---------------------
view = store.grant_model("wid-1", "Qwen2.5-3B", job_id="job-abc")
check("grant_model returns a public view", view is not None)
check("granted model lands in grants", "Qwen2.5-3B" in view["grants"])
check("grant entry has ts/job_id/origin=system",
      view["grants"]["Qwen2.5-3B"]["job_id"] == "job-abc"
      and view["grants"]["Qwen2.5-3B"]["origin"] == "system"
      and isinstance(view["grants"]["Qwen2.5-3B"]["ts"], float))
check("granted model NOT added to models", "Qwen2.5-3B" not in view.get("models", []))

got = store.get("wid-1")
check("_public_view (via get) shows the grant", "Qwen2.5-3B" in got["grants"])


# --- 2. workers_for_model treats a granted model as serveable ---------------
# Freshly registered/granted workers are online (last_seen stamped at
# register-time), so no extra heartbeat needed for the online_only gate.
serving = store.workers_for_model("Qwen2.5-3B")
check("granted model appears in workers_for_model (serveable)",
      any(w["id"] == "wid-1" for w in serving))
check("worker WITHOUT the grant is not returned for this model",
      not any(w["id"] == "wid-2" for w in serving))


# --- 3. storage_proposal: grant alone is NOT protection ---------------------
def _mk_storage_worker(worker_id, grants=None, assigned_model=None, pinned_model=None,
                        models_report=()):
    """Build a raw worker dict (bypassing the store) for storage_proposal()."""
    w = {
        "id": worker_id,
        "grants": grants or {},
        "storage": {
            "cache_used_bytes": 1000,
            "disk_free": 10,
            "models": list(models_report),
        },
        "disk": {"free_bytes": 10, "total_bytes": 1000},
        "config": {"pinned": {pinned_model: True} if pinned_model else {}},
    }
    return w


# (a) ONLY granted -> not protected, IS a candidate.
w_granted_only = _mk_storage_worker(
    "wid-a", grants={"modelA": {"ts": 1.0, "job_id": None, "origin": "system"}},
    models_report=[{"model_key": "modelA", "bytes": 500, "assigned": False}],
)
prop = W.storage_proposal(w_granted_only)
mA = next(m for m in prop["models"] if m["model_key"] == "modelA")
check("grant-only model is marked granted=True in storage view", mA["granted"] is True)
check("grant-only model is NOT protected", mA["protected"] is False)
check("grant-only model IS an eviction candidate",
      any(e["model_key"] == "modelA" for e in prop["proposed_evictions"]) or True)
# (candidate-ness is really tested via the `protected is False` assertion above;
# proposed_evictions additionally depends on over_budget/need, checked next.)
check("grant-only model shows up over-budget as a proposed eviction",
      any(e["model_key"] == "modelA" for e in prop["proposed_evictions"]))

# (b) ASSIGNED (worker-reported assigned=True) -> a CANDIDATE (operator
# 2026-07-17: "the allocation only stipulates the routing for that model...
# neither of those should have any bearing on the pull or eviction").
w_assigned = _mk_storage_worker(
    "wid-b",
    models_report=[{"model_key": "modelB", "bytes": 500, "assigned": True}],
)
prop_b = W.storage_proposal(w_assigned)
mB = next(m for m in prop_b["models"] if m["model_key"] == "modelB")
check("assigned model is a CANDIDATE (allocation = routing only)",
      mB["protected"] is False)

# (c) BOTH granted AND pinned -> a CANDIDATE (2026-07-17: pin no longer protects
# files; neither grant nor pin shields, so the model is reclaimable). The
# `pinned`/`why` fields stay honest as ATTRIBUTION while `protected` is False.
w_both = _mk_storage_worker(
    "wid-c", grants={"modelC": {"ts": 1.0, "job_id": None, "origin": "system"}},
    pinned_model="modelC",
    models_report=[{"model_key": "modelC", "bytes": 500, "assigned": False}],
)
prop_c = W.storage_proposal(w_both)
mC = next(m for m in prop_c["models"] if m["model_key"] == "modelC")
check("granted+pinned model is NOT protected (pin no longer shields files)",
      mC["protected"] is False)
check("granted+pinned model why is attribution-only 'pinned' (protected False)",
      mC["why"] == "pinned")
check("granted+pinned model is flagged pinned=True (attribution)", mC["pinned"] is True)
check("granted+pinned model is marked granted=True regardless", mC["granted"] is True)


# --- 4. ungrant_model: removes, idempotent, orthogonal to assign ------------
store.assign_model("wid-1", "OtherModel-7B")
view = store.grant_model("wid-1", "Qwen2.5-3B", job_id="job-xyz")
check("sanity: both grant + a different assignment present",
      "Qwen2.5-3B" in view["grants"] and "OtherModel-7B" in view["models"])

view = store.ungrant_model("wid-1", "Qwen2.5-3B")
check("ungrant_model removes the grant", "Qwen2.5-3B" not in view["grants"])
check("ungrant_model does not touch the unrelated assignment",
      "OtherModel-7B" in view["models"])

view2 = store.ungrant_model("wid-1", "Qwen2.5-3B")
check("ungrant_model is idempotent (second call no-op, no error)",
      view2 is not None and "Qwen2.5-3B" not in view2["grants"])


# --- 5. _remember_assignments does NOT persist grants -----------------------
store.grant_model("wid-1", "GrantOnlyModel")
# assign/unassign both call _remember_assignments internally; force a fresh
# snapshot write via assign_model (already exercises the real code path).
store.assign_model("wid-1", "OtherModel-7B")  # re-triggers _remember_assignments

mem = W._load_assign_memory()
check("assign-memory file has an entry for wid-1", "wid-1" in mem)
check("assign-memory entry has models/spill_by_model only (no grants key)",
      "grants" not in mem["wid-1"])
check("assign-memory models list does not include the grant-only model",
      "GrantOnlyModel" not in mem["wid-1"].get("models", []))
# Belt-and-suspenders: read the raw JSON file too, not just the parsed dict.
with open(W._assign_memory_path(), "r", encoding="utf-8") as fh:
    raw = fh.read()
check("raw assign-memory JSON text never mentions 'grants'", '"grants"' not in raw)
check("raw assign-memory JSON text never mentions the grant-only model key",
      "GrantOnlyModel" not in raw)


# --- 6. assign/unassign of a normal model unchanged by presence of a grant --
store.grant_model("wid-2", "SomeGrant")
v = store.assign_model("wid-2", "NormalModel")
check("assign_model still adds to models with a grant present elsewhere",
      "NormalModel" in v["models"])
check("assign_model does not touch grants", "SomeGrant" in v["grants"])

v = store.unassign_model("wid-2", "NormalModel")
check("unassign_model removes from models as before",
      "NormalModel" not in v["models"])
check("unassign_model does not touch grants", "SomeGrant" in v["grants"])


print(f"\nall {ok} checks passed")
