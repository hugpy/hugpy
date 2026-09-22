"""Worker per-model generation gate (concurrency hardening 2026-07-11).

The failure class: a batch client fired concurrent requests at a model served
IN-PROCESS by llama-cpp-python; the shared, non-reentrant Llama context raced in
native code and the whole worker SEGV/ABRT'd (computron, restart counter 219).
There was no generation lock — only an instance-CREATION lock.

This proves the gate that closes it: threads hammer a FAKE slow runner through
``gen_gate`` and we assert (a) never >1 concurrent entrant, (b) FIFO-ish
progress, (c) a bounded-wait timeout returns the structured busy error, (d) a
streamed response holds the gate to stream end. Plus the slot-skip / disable /
cap-from-env semantics.

Runs like the other tests here:
    venv/bin/python tests/test_worker_gen_gate.py
"""
import importlib
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Deterministic knobs (gen_gate reads env lazily).
os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "1"
os.environ.pop("HUGPY_WORKER_GEN_GATE", None)

gg = importlib.import_module("hugpy_fleet.worker.gen_gate")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# --- config semantics ------------------------------------------------------
check("concurrency_limit default 1 (llama.cpp/transformers truth)",
      gg.concurrency_limit() == 1)
check("gate_timeout_s default 0 (unbounded; HUGPY_WORKER_GEN_GATE_TIMEOUT_S bounds it)",
      gg.gate_timeout_s() == 0.0)


# --- should_gate: default gate, disable switch, slot skip ------------------
check("should_gate default True (no slots seated -> in-process)",
      gg.should_gate("any/Model-GGUF") is True)

os.environ["HUGPY_WORKER_GEN_GATE"] = "off"
check("HUGPY_WORKER_GEN_GATE=off -> should_gate False", gg.should_gate("x") is False)
check("disabled -> acquire returns the shared no-op token",
      gg.acquire_for_payload({"model_key": "x"}) is gg._NULL_TOKEN)
os.environ.pop("HUGPY_WORKER_GEN_GATE", None)

# Slot-backed models schedule themselves in the child -> NOT gated. Guard the
# heavy runner import like test_no_local_serving does.
try:
    get_mod = importlib.import_module("hugpy_engine.llama.runners.get")
except Exception as exc:  # llama_cpp/torch may be absent
    print(f"  skip - slot-backed detection import unavailable ({type(exc).__name__})")
else:
    _orig = get_mod.slot_backed_model_keys
    get_mod.slot_backed_model_keys = lambda: {"slotted/Model-GGUF"}
    try:
        check("slot-backed model is NOT gated (child schedules itself)",
              gg.should_gate("slotted/Model-GGUF") is False)
        check("slot skip is alias-tolerant (tail match)",
              gg.should_gate("Model-GGUF") is False)
        check("a different in-process model is still gated",
              gg.should_gate("other/InProc-GGUF") is True)
    finally:
        get_mod.slot_backed_model_keys = _orig


# --- (a) never more than 1 concurrent entrant; serialized ------------------
MODEL = "gate/text-model"
state = {"cur": 0, "max": 0}
clk = threading.Lock()
exits = []

def fake_run(dwell=0.05):
    with gg.gate_for_payload({"model_key": MODEL}):
        with clk:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(dwell)
        with clk:
            state["cur"] -= 1
            exits.append(1)

N = 8
threads = [threading.Thread(target=fake_run) for _ in range(N)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0
check("never >1 concurrent entrant into the in-process runner", state["max"] == 1)
check("all N gated requests completed", len(exits) == N)
check("gated requests serialized (wall ~ N*dwell, not parallel)",
      wall >= N * 0.05 * 0.85)


# --- (b) FIFO-ish progress -------------------------------------------------
order = []

def fifo_run(tag):
    with gg.gate_for_payload({"model_key": "gate/fifo-model"}):
        order.append(tag)
        time.sleep(0.03)

threads = []
for i in range(5):
    t = threading.Thread(target=fifo_run, args=(i,))
    t.start()
    threads.append(t)
    time.sleep(0.015)  # stagger so waiters queue in start order
for t in threads:
    t.join()
check("gate progresses in FIFO-ish order", order == sorted(order))


# --- (c) bounded-wait timeout -> structured busy error ---------------------
held = threading.Event()
release = threading.Event()

def holder():
    with gg.gate_for_payload({"model_key": "gate/busy-model"}):
        held.set()
        release.wait(5)

h = threading.Thread(target=holder)
h.start()
held.wait(2)
busy = None
try:
    with gg.gate_for_payload({"model_key": "gate/busy-model"}, timeout_s=0.2):
        pass
except gg.ModelBusy as exc:
    busy = exc
release.set()
h.join()
check("bounded-wait timeout raises ModelBusy", busy is not None)
check("ModelBusy reports in_flight (1 already inside)", busy.in_flight == 1)
check("ModelBusy reports the waited time (~timeout)", busy.waited_s >= 0.19)
err = busy.as_error({"id": "w1", "name": "computron"})
check("busy error envelope matches the worker idiom",
      err["ok"] is False
      and err["error"]["code"] == "model_busy"
      and err["error"]["in_flight"] == 1
      and err["error"]["model_key"] == "gate/busy-model"
      and err["error"]["waited_s"] == busy.waited_s
      and err["worker"] == {"id": "w1", "name": "computron"})


# --- (d) a streamed response holds the gate to stream end ------------------
tok = gg.acquire_for_payload({"model_key": "gate/stream-model"})
check("stream acquisition holds the gate (in_flight==1 mid-stream)",
      gg.in_flight("gate/stream-model") == 1)
busy2 = None
try:  # a concurrent request MUST be refused while the stream holds the gate
    gg.acquire_for_payload({"model_key": "gate/stream-model"}, timeout_s=0.15)
except gg.ModelBusy as exc:
    busy2 = exc
check("second request blocked while the stream holds the gate",
      busy2 is not None and busy2.in_flight == 1)
tok.release()
check("gate freed after stream end (in_flight==0)",
      gg.in_flight("gate/stream-model") == 0)
tok.release()  # idempotent
check("token release is idempotent (no over-release)",
      gg.in_flight("gate/stream-model") == 0)
# proven free: a fresh acquire now succeeds immediately
tok2 = gg.acquire_for_payload({"model_key": "gate/stream-model"}, timeout_s=0.2)
check("gate reusable after release", gg.in_flight("gate/stream-model") == 1)
tok2.release()


# --- cap honored: a limit-2 gate admits exactly 2 --------------------------
os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "2"
check("concurrency_limit reads env (2)", gg.concurrency_limit() == 2)
st2 = {"cur": 0, "max": 0}
lk2 = threading.Lock()

def run2():
    with gg.gate_for_payload({"model_key": "gate/limit2-model"}):
        with lk2:
            st2["cur"] += 1
            st2["max"] = max(st2["max"], st2["cur"])
        time.sleep(0.05)
        with lk2:
            st2["cur"] -= 1

ts = [threading.Thread(target=run2) for _ in range(6)]
for t in ts:
    t.start()
for t in ts:
    t.join()
check("limit=2 gate admits exactly 2 concurrent entrants", st2["max"] == 2)
os.environ["HUGPY_INPROCESS_MAX_CONCURRENCY"] = "1"

print(f"\nall {ok} checks passed")
