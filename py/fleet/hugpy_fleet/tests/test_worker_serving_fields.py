"""Worker concurrency-capability heartbeat fields (2026-07-11).

Two halves:
  * The worker COMPUTES honest fields — _serving_limits() advertises the
    in-process concurrency cap, and _slot_capability() reports slot_capable
    true/false WITH a reason from the engine-binary truth (a native llama-server
    resolvable vs not — computron's exact silent condition when a box with slots
    configured has no usable engine binary).
  * The fields ROUND-TRIP into the central worker registry (register + heartbeat)
    and _public_view exposes them on /llm/workers rows — legacy-safe: an older
    agent that omits them never has a prior value wiped, and reads as cap-1.

Runs like the other tests here:
    venv/bin/python tests/test_worker_serving_fields.py
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "1"
os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)  # keep slots_enabled honest here

agent = importlib.import_module("hugpy_fleet.worker.agent")
resolve = importlib.import_module("hugpy_engine.native.resolve")
W = importlib.import_module(
    "hugpy_fleet.central.workers")
from worker_store_isolation import isolated_worker_store  # noqa: E402

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --- serving_limits --------------------------------------------------------
check("serving_limits advertises in_process_max_concurrency=1",
      agent._serving_limits() == {"in_process_max_concurrency": 1})
os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "3"
check("serving_limits tracks the env cap",
      agent._serving_limits()["in_process_max_concurrency"] == 3)
os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "1"


# --- slot_capability: engine present -> capable, no reason -----------------
_orig_sb = resolve.server_bin
resolve.server_bin = lambda: "/opt/engine/bin/llama-server"
cap = agent._slot_capability()
check("native engine binary present -> slot_capable True",
      cap["slot_capable"] is True and cap["slot_incapable_reason"] is None)

# --- slot_capability: engine missing -> incapable WITH honest reason -------
resolve.server_bin = lambda: None
os.environ["SLOT_COUNT"] = "0"
cap = agent._slot_capability()
check("no engine binary -> slot_capable False", cap["slot_capable"] is False)
check("reason names the missing binary + the in-process consequence",
      "llama-server" in cap["slot_incapable_reason"]
      and "in-process" in cap["slot_incapable_reason"])

# slots CONFIGURED but no binary = computron's exact silent condition: the
# reason must name the python fallback so central/console can see it.
os.environ["SLOT_COUNT"] = "2"
cap = agent._slot_capability()
check("slots configured + no binary -> reason names the llama_cpp.server fallback",
      cap["slot_capable"] is False
      and "llama_cpp.server" in cap["slot_incapable_reason"])
os.environ.pop("SLOT_COUNT", None)
resolve.server_bin = _orig_sb


# --- registry round-trip ---------------------------------------------------
# k3 isolation: isolated_worker_store() also redirects the assignment-memory
# sidecar (settings.manifest_path) — see tests/worker_store_isolation.py.
store, tmp = isolated_worker_store(prefix="hugpy-serving-fields-")

view = store.register(
    name="computron", url="http://computron:9100", worker_id="wid-cap",
    serving_limits={"in_process_max_concurrency": 1},
    slot_capable=False, slot_incapable_reason="no native llama-server binary")
check("register stores serving_limits",
      view.get("serving_limits") == {"in_process_max_concurrency": 1})
check("register stores slot_capable + reason",
      view.get("slot_capable") is False
      and view.get("slot_incapable_reason") == "no native llama-server binary")

got = store.get("wid-cap")
check("_public_view exposes the fields on the /llm/workers row",
      got.get("slot_capable") is False
      and got.get("serving_limits") == {"in_process_max_concurrency": 1})

# A heartbeat refreshes them (installing the engine flips slot_capable in one beat)
view2 = store.heartbeat(
    "wid-cap", serving_limits={"in_process_max_concurrency": 2},
    slot_capable=True, slot_incapable_reason=None)
check("heartbeat refreshes serving_limits + slot_capable within one beat",
      view2.get("slot_capable") is True
      and view2.get("serving_limits") == {"in_process_max_concurrency": 2}
      and view2.get("slot_incapable_reason") is None)


# --- backward compatibility ------------------------------------------------
# A LEGACY worker (older agent) never sends the fields — its row is untouched
# and reads as cap-1 on the central side (remote._advertised_cap).
store.register(name="legacy", url="http://legacy:9100", worker_id="wid-legacy")
lg = store.get("wid-legacy")
check("legacy worker: serving_limits absent/None",
      lg.get("serving_limits") is None)
check("legacy worker: slot_capable absent/None", lg.get("slot_capable") is None)

remote = importlib.import_module("hugpy_engine.resolvers.remote")
check("central treats a legacy worker as cap 1", remote._advertised_cap(lg) == 1)

# A heartbeat that OMITS the fields must not clobber a previously-set value.
store.heartbeat("wid-cap")  # no serving_limits / slot_capable in this beat
after = store.get("wid-cap")
check("omitted heartbeat fields don't wipe prior values",
      after.get("slot_capable") is True
      and after.get("serving_limits") == {"in_process_max_concurrency": 2})

print(f"\nall {ok} checks passed")
