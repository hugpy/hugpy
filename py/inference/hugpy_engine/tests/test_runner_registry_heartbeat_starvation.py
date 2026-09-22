"""The deaf-worker class (2026-07-29, ae in-VM; same family as k53).

py-spy caught the live shape: the chat thread inside a minutes-long build
(_build_runner -> serve_endpoint -> evict-to-fit -> slot load) HOLDING the
runner-registry lock, and the heartbeat thread blocked at the top of
loaded_runner_detail waiting for a microsecond dict snapshot. Central read the
missed beats as offline while the worker was busily serving.

The contract under test: a build in progress must be INVISIBLE to registry
readers — loaded_runner_detail / slot_backed_model_keys / a cached-get return
promptly while a build runs. Builds still serialize (one at a time).

Run: venv/bin/python -m pytest tests/test_runner_registry_heartbeat_starvation.py -q
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib
G = importlib.import_module("hugpy_engine.llama.runners.get")


class _FakeRunner:
    model_path = None
    n_threads = 4
    n_gpu_layers = 0
    base_url = None
    llm = None


def test_snapshot_readers_do_not_wait_on_a_slow_build(monkeypatch):
    build_started = threading.Event()
    release_build = threading.Event()

    def slow_build(mk):
        build_started.set()
        assert release_build.wait(30), "test wiring: build never released"
        return _FakeRunner()

    monkeypatch.setattr(G, "_build_runner", slow_build)
    # a pre-existing resident the heartbeat wants to report
    with G._LLAMA_LOCK:
        G._LLAMA_INSTANCES["already-resident"] = _FakeRunner()
    try:
        t = threading.Thread(target=G.get_llama_runner, args=("cold-model",),
                             daemon=True)
        t.start()
        assert build_started.wait(10), "build never started"

        # THE assertion: the heartbeat's snapshot returns promptly while the
        # build is still holding whatever it holds. Pre-fix this deadline blew.
        t0 = time.time()
        detail = G.loaded_runner_detail()
        slot_keys = G.slot_backed_model_keys()
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"registry readers starved {elapsed:.1f}s behind a build"
        assert "already-resident" in detail
        assert isinstance(slot_keys, set)

        # a cached get is also unblocked
        t0 = time.time()
        assert G.get_llama_runner("already-resident") is not None
        assert time.time() - t0 < 2.0
    finally:
        release_build.set()
        t.join(10)
        with G._LLAMA_LOCK:
            G._LLAMA_INSTANCES.pop("already-resident", None)
            G._LLAMA_INSTANCES.pop("cold-model", None)


def test_builds_still_serialize_and_coalesce(monkeypatch):
    order = []
    gate = threading.Event()

    def build(mk):
        order.append(("start", mk))
        gate.wait(5)
        order.append(("end", mk))
        return _FakeRunner()

    monkeypatch.setattr(G, "_build_runner", build)
    try:
        t1 = threading.Thread(target=G.get_llama_runner, args=("m1",), daemon=True)
        t2 = threading.Thread(target=G.get_llama_runner, args=("m1",), daemon=True)
        t1.start(); t2.start()
        time.sleep(0.3)
        gate.set()
        t1.join(10); t2.join(10)
        # same key twice -> exactly ONE build (the waiter found the cache)
        assert order.count(("start", "m1")) == 1, order
        # and builds never interleave (start/end strictly paired)
        for i in range(0, len(order), 2):
            assert order[i][0] == "start" and order[i + 1][0] == "end"
    finally:
        with G._LLAMA_LOCK:
            G._LLAMA_INSTANCES.pop("m1", None)
