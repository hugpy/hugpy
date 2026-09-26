"""Central cap-aware relay gating (concurrency hardening 2026-07-11).

Central never FIRES a relay that would enter a busy in-process runner. It tracks
in-flight relays per (worker, model), respects the worker's advertised in-process
cap (absent => 1, the crash-safe legacy assumption), reroutes to another online
holder when the primary is full, waits briefly for a slot to free, and only then
returns an honest busy error. A slot-served model is not gated (its llama-server
child schedules concurrency itself).

Runs like the other tests here:
    venv/bin/python tests/test_central_cap_gating.py
"""
import asyncio
import importlib
import os
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.pop("HUGPY_CENTRAL_GATE", None)
os.environ["HUGPY_CENTRAL_GATE_WAIT_S"] = "0"  # no wait unless a test asks

remote = importlib.import_module("hugpy_engine.resolvers.remote")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_central_cap_gating():
    """Converted from the script-style suite: every `check` is an assert."""


    MODEL = "gate/Model-GGUF"
    W1 = {"id": "w1", "name": "boxA", "serving_limits": {"in_process_max_concurrency": 1}}
    W2 = {"id": "w2", "name": "boxB", "serving_limits": {"in_process_max_concurrency": 1}}
    W_SLOT = {"id": "ws", "name": "slotbox",
              "slots": [{"model_key": MODEL, "healthy": True}]}


    def _reset():
        with remote._INFLIGHT_LOCK:
            remote._INFLIGHT.clear()


    # --- cap math (the legacy-safe assumption) ---------------------------------
    check("legacy worker (no serving_limits) => cap 1", remote._advertised_cap({}) == 1)
    check("advertised cap honored", remote._advertised_cap(W1) == 1)
    check("cap floored to 1 (unlimited in-process concurrency IS the crash)",
          remote._advertised_cap({"serving_limits": {"in_process_max_concurrency": 0}}) == 1)
    check("higher cap honored",
          remote._advertised_cap({"serving_limits": {"in_process_max_concurrency": 4}}) == 4)
    check("slot-served model is uncapped centrally (child schedules itself)",
          remote._effective_cap(W_SLOT, MODEL) is None)
    check("in-process model uses the advertised cap",
          remote._effective_cap(W1, MODEL) == 1)


    # --- ALIAS-TOLERANT slot classification (2026-07-23 incident) --------------
    # A live call for ``Qwen~Qwen3-Coder-Next-GGUF`` returned worker_busy while the
    # model was seated in a SLOT under the BARE key ``Qwen3-Coder-Next-GGUF`` — the
    # ~-spelling missed the slot classification and fell to the in-process cap-1.
    # _model_slot_served must be alias-tolerant via the same ~/-tail unification as
    # routing (workers._match_keys); the cap semantics themselves are unchanged.
    W_SLOT_BARE = {"id": "wsb", "name": "slotbare",
                   "slots": [{"model_key": "Qwen3-Coder-Next-GGUF", "healthy": True}]}
    W_SLOT_QUAL = {"id": "wsq", "name": "slotqual",
                   "slots": [{"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "healthy": True}]}
    check("slot seated under BARE key -> ~-qualified request classifies slot-served",
          remote._model_slot_served(W_SLOT_BARE, "Qwen~Qwen3-Coder-Next-GGUF") is True)
    check("slot seated under BARE key -> ~-qualified request is UNCAPPED (None)",
          remote._effective_cap(W_SLOT_BARE, "Qwen~Qwen3-Coder-Next-GGUF") is None)
    check("REVERSE: slot seated under ~-qualified key -> bare request slot-served",
          remote._model_slot_served(W_SLOT_QUAL, "Qwen3-Coder-Next-GGUF") is True)
    check("REVERSE: slot seated under ~-qualified key -> bare request UNCAPPED",
          remote._effective_cap(W_SLOT_QUAL, "Qwen3-Coder-Next-GGUF") is None)
    check("exact-key slot match still classifies slot-served (regression)",
          remote._model_slot_served(W_SLOT_BARE, "Qwen3-Coder-Next-GGUF") is True)
    check("a DIFFERENT model is NOT slot-served here (no false positive)",
          remote._model_slot_served(W_SLOT_BARE, "Owner~Some-Other-GGUF") is False)
    check("a different model still uses the advertised in-process cap",
          remote._effective_cap(W1, "Owner~Some-Other-GGUF") == 1)
    # allocations-carried slot (the other snapshot source) is alias-tolerant too
    W_SLOT_ALLOC = {"id": "wsa", "name": "slotalloc",
                    "allocations": [{"kind": "slot",
                                     "model_key": "Qwen3-Coder-Next-GGUF"}]}
    check("allocations-carried slot (bare) -> ~-qualified request slot-served",
          remote._model_slot_served(W_SLOT_ALLOC, "Qwen~Qwen3-Coder-Next-GGUF") is True)


    # --- reroute: primary busy -> the second holder ----------------------------
    _reset()
    remote.set_worker_candidates_provider(lambda mk, pool=None: [W1, W2])
    check("occupy W1 (cap 1)", remote._inflight_try_acquire("w1", MODEL, 1) is True)
    slot = remote._acquire_relay_slot(MODEL, None, W1, {"spillA": 1}, wait_s=0)
    check("primary at cap -> reroute to the second holder", slot.worker["id"] == "w2")
    check("reroute reserved an in-flight slot on w2",
          remote._inflight_count("w2", MODEL) == 1)
    check("reroute carries the alternate's spill (not the primary's)",
          slot.spill != {"spillA": 1})
    slot.release()
    check("release frees w2's in-flight slot", remote._inflight_count("w2", MODEL) == 0)


    # --- both holders busy -> honest WorkerBusyError ---------------------------
    _reset()
    remote._inflight_try_acquire("w1", MODEL, 1)
    remote._inflight_try_acquire("w2", MODEL, 1)
    busy = None
    try:
        remote._acquire_relay_slot(MODEL, None, W1, {}, wait_s=0)
    except remote.WorkerBusyError as exc:
        busy = exc
    check("both holders busy -> WorkerBusyError (never fire into a busy runner)",
          busy is not None)
    check("busy names the primary worker + model + in_flight",
          busy.model_key == MODEL and busy.worker_name == "boxA" and busy.in_flight == 1)
    err = busy.as_error()
    check("busy error is honest JSON (the 429/503 envelope)",
          err["error"]["code"] == "worker_busy"
          and err["error"]["model"] == MODEL
          and err["error"]["in_flight"] == 1
          and err["error"]["worker"] == "boxA")


    # --- legacy primary (no serving_limits) is treated as cap 1 ----------------
    _reset()
    LEG = {"id": "wl", "name": "legacy"}
    remote.set_worker_candidates_provider(lambda mk, pool=None: [LEG])
    s1 = remote._acquire_relay_slot(MODEL, None, LEG, {}, wait_s=0)
    check("legacy worker admits the first relay (cap 1)", s1.worker["id"] == "wl")
    busy2 = None
    try:  # a SECOND concurrent relay exceeds the assumed cap of 1
        remote._acquire_relay_slot(MODEL, None, LEG, {}, wait_s=0)
    except remote.WorkerBusyError as exc:
        busy2 = exc
    check("legacy worker's 2nd concurrent relay is refused (cap 1 assumed)",
          busy2 is not None)
    s1.release()


    # --- bounded wait: waits, then admits once a slot frees --------------------
    _reset()
    remote.set_worker_candidates_provider(lambda mk, pool=None: [W1])
    remote._inflight_try_acquire("w1", MODEL, 1)  # occupy the only holder

    def _free_soon():
        time.sleep(0.25)
        remote._inflight_release("w1", MODEL)

    threading.Thread(target=_free_soon, daemon=True).start()
    t0 = time.time()
    s = remote._acquire_relay_slot(MODEL, None, W1, {}, wait_s=2.0)
    waited = time.time() - t0
    check("bounded wait admits once the slot frees (didn't error early)",
          s.worker["id"] == "w1" and 0.2 <= waited <= 1.5)
    s.release()


    # --- slot-served primary: admitted uncapped, not counted -------------------
    _reset()
    remote.set_worker_candidates_provider(lambda mk, pool=None: [W_SLOT])
    s = remote._acquire_relay_slot(MODEL, None, W_SLOT, {}, wait_s=0)
    check("slot-served primary admitted without a cap", s.worker["id"] == "ws")
    check("slot-served relay is NOT counted (uncapped)",
          remote._inflight_count("ws", MODEL) == 0)
    s.release()


    # --- global disable escape hatch -------------------------------------------
    _reset()
    os.environ["HUGPY_CENTRAL_GATE"] = "off"
    remote._inflight_try_acquire("w1", MODEL, 1)  # saturate
    s = remote._acquire_relay_slot(MODEL, None, W1, {}, wait_s=0)  # would busy if gated
    check("HUGPY_CENTRAL_GATE=off -> gate is a pass-through (no admission control)",
          s.worker["id"] == "w1")
    s.release()
    os.environ.pop("HUGPY_CENTRAL_GATE", None)


    # --- DelegatingRunner integration (the real relay surface) -----------------
    # Non-vision (fw, task) so the vision predicate doesn't confound the busy path.
    pair = next((p for p in remote.FRAMEWORK_RUNNERS if p[1] != "image-text-to-text"),
                next(iter(remote.FRAMEWORK_RUNNERS)))
    fw, task = pair
    Runner = remote.make_delegating_runner(fw, task)
    runner = Runner(types.SimpleNamespace(model_key=MODEL))

    _reset()
    remote._select = lambda mk, pool=None, task=None: (dict(W1), {})
    remote.set_worker_candidates_provider(lambda mk, pool=None: [dict(W1)])
    remote._inflight_try_acquire("w1", MODEL, 1)  # saturate the only holder
    # Production's zero default waits until capacity/cancellation. Exercise the
    # bounded refusal surface here with an explicit short ceiling.
    os.environ["HUGPY_CENTRAL_GATE_WAIT_S"] = "0.05"

    req = types.SimpleNamespace(request_id="rid-busy", pool=None,
                                model_dump=lambda: {"model_key": MODEL})
    raised = None
    try:
        asyncio.run(runner.run(req))
    except remote.WorkerBusyError as exc:
        raised = exc
    except Exception as exc:  # any other error would mean the gate leaked
        raised = f"WRONG:{type(exc).__name__}:{exc}"
    check("DelegatingRunner.run raises WorkerBusyError when the only holder is full",
          isinstance(raised, remote.WorkerBusyError))

    async def _collect():
        evs = []
        async for ev in runner.stream(req):
            evs.append(ev)
        return evs

    evs = asyncio.run(_collect())
    os.environ.pop("HUGPY_CENTRAL_GATE_WAIT_S", None)
    check("DelegatingRunner.stream yields exactly one honest busy error event",
          len(evs) == 1
          and getattr(evs[0], "type", None) == "error"
          and "worker_busy" in getattr(evs[0], "message", ""))

    # The saturating permit is still held (we never released it) — prove the gate
    # didn't leak the failed attempts' permits (count stays exactly 1).
    check("failed admissions leak no permits (count still 1)",
          remote._inflight_count("w1", MODEL) == 1)


    # --- async acquire yields the shared loop (no deadlock) --------------------
    # Central drives every relay on ONE long-lived event-loop thread, so the wait
    # MUST yield (await asyncio.sleep) — a blocking sleep would freeze the request
    # holding the slot, which then never finishes to free it. Prove the holder can
    # generate + release WHILE a waiter is blocked, and the waiter is then admitted.
    async def _async_no_deadlock():
        _reset()
        remote.set_worker_candidates_provider(lambda mk, pool=None: [W1])
        held = await remote._acquire_relay_slot_async(MODEL, None, W1, {}, wait_s=0)
        result = {}

        async def gen_and_release():
            for _ in range(5):          # "generation" that needs the loop
                await asyncio.sleep(0.05)
            held.release()
            result["freed"] = True

        async def waiter():
            s = await remote._acquire_relay_slot_async(MODEL, None, W1, {}, wait_s=2.0)
            result["waiter_admitted"] = True
            s.release()

        await asyncio.gather(gen_and_release(), waiter())
        return result

    res = asyncio.run(_async_no_deadlock())
    check("async acquire yields the loop -> holder frees the slot, waiter admitted",
          res.get("waiter_admitted") is True and res.get("freed") is True)

    print(f"\nall {ok} checks passed")
