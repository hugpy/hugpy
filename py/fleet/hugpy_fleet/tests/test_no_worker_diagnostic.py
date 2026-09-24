"""Self-diagnosing refused-local serving (2026-07-15).

A model can be ASSIGNED + pinned + on disk on a worker and still 500 on a real
request when that worker's inference engine is unusable (broken llama-cpp AND no
native llama-server binary — the computron ``GLIBCXX_3.4.30`` / ``slot_capable:
false`` case). ``workers_for_model`` correctly excludes such a worker
(``_engine_unusable``), so with HUGPY_NO_LOCAL_SERVING the request refuses local —
but the message used to be opaque ("no registered worker is available"), hiding
that a DESIGNATED worker was skipped for a fixable reason.

This asserts the two additive hardenings, and that the already-serving / unset
paths stay byte-identical:
  * workers_for_model emits a say-why WARNING when assigned workers are all
    dropped for an unusable engine (parity with the tier/task/id_lock gates).
  * explain_no_worker names the assigned-but-excluded worker + the real reason,
    and returns "" when there is nothing designation-specific to explain.
  * resolvers.remote threads that detail into the refused-local error via the
    optional seam; unset ⇒ the message is exactly local_serving_error(model_key).

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_no_worker_diagnostic.py -q
    venv/bin/python tests/test_no_worker_diagnostic.py
"""
import asyncio
import importlib
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# NB: the heavy modules are imported INSIDE the test, not at module scope —
# importing workers.py registers the live worker provider as a side effect, and a
# sibling test (test_no_local_serving) asserts the "no provider registered"
# standalone posture during COLLECTION. Keeping this file's import a pure no-op
# preserves that isolation regardless of collection order.

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())
    def reset(self):
        self.records = []


MODEL = "Qwen2.5-VL-3B-Instruct-GGUF"
VISION = "image-text-to-text"

# The exact shape a broken box (computron) advertises: llama-cpp not loadable AND
# no native llama-server binary, so serving is impossible by any path.
_BROKEN_ENGINE = {
    "installed": False,
    "error": ("RuntimeError: Failed to load shared library 'libllama.so': "
              "libstdc++.so.6: version `GLIBCXX_3.4.30' not found"),
}
_SLOT_REASON = ("no native llama-server binary resolvable (run "
                "`hugpy install-engine`); the slots fall back to in-process "
                "llama_cpp.server — text only, vision GGUF is refused")


def _fresh_store(W):
    # k3 isolation: function-local import (not module scope) — this file
    # deliberately keeps collection a no-op (see the note above), and
    # worker_store_isolation itself imports workers.py, so importing it here
    # is only safe because W is already imported by the caller at this point.
    # isolated_worker_store() ALSO redirects the assignment-memory sidecar
    # (settings.manifest_path) — a bare WorkerStore(path=tmp) isolates the
    # worker rows but a later register()/assign_model() call on the SAME id
    # would still write worker_assignments.json beside the real live
    # manifest. See tests/worker_store_isolation.py.
    from worker_store_isolation import isolated_worker_store
    store, _tmp = isolated_worker_store(prefix="hugpy-no-worker-diag-")
    return store


def _add(store, wid, *, engine=None, models=(MODEL,), caps=None,
         slot_capable=None, slot_reason=None):
    store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid,
                   models=list(models), engine=engine, task_capabilities=caps,
                   slot_capable=slot_capable, slot_incapable_reason=slot_reason)
    store.set_admission(wid, "approved")


def _run_checks(cap, W, remote, policy):
    _fresh = lambda: _fresh_store(W)
    # --- workers_for_model: engine-unusable assigned worker skipped + say-why ---
    store = _fresh()
    _add(store, "computron", engine=_BROKEN_ENGINE,
         slot_capable=False, slot_reason=_SLOT_REASON)
    cap.reset()
    res = store.workers_for_model(MODEL, task=VISION)
    check("assigned worker with engine.installed=False is excluded", res == [])
    said = any(MODEL in m and "engine" in m and "skipped" in m for m in cap.records)
    check("engine say-why WARNING names the model + that workers were skipped", said)
    check("pick_for_model returns None (caller refuses local under policy)",
          store.pick_for_model(MODEL, task=VISION) is None)

    # A worker NOT reporting engine status is still assumed capable (no regression).
    store2 = _fresh()
    store2.register(name="legacy", url="http://legacy:9100", worker_id="legacy",
                    models=[MODEL])       # no engine field at all
    store2.set_admission("legacy", "approved")
    check("legacy worker (no engine field) is NOT excluded (grandfather)",
          {w["id"] for w in store2.workers_for_model(MODEL, task=VISION)} == {"legacy"})

    # --- explain_no_worker: names the designated-but-broken worker + reason -----
    # explain_no_worker reads the module-global worker_store; point it at ours.
    orig_store = W.worker_store
    try:
        W.worker_store = store
        detail = W.explain_no_worker(MODEL, task=VISION)
        check("explain_no_worker names the assigned worker", "computron" in detail)
        check("explain_no_worker calls it out as engine-unusable",
              "engine unusable" in detail)
        check("explain_no_worker surfaces the actionable native-binary reason",
              "install-engine" in detail)
        check("explain_no_worker names the model", MODEL in detail)

        # No assigned worker at all -> "" (generic 'assign it somewhere' suffices).
        W.worker_store = _fresh()
        check("explain_no_worker returns '' when no worker is assigned",
              W.explain_no_worker(MODEL, task=VISION) == "")

        # A healthy assigned worker that passes every static gate -> "" (a miss
        # there is transient/runtime, not a designation problem).
        healthy = _fresh()
        _add(healthy, "ae", engine={"installed": True, "supports_vision": True})
        W.worker_store = healthy
        check("explain_no_worker returns '' when the assigned worker is eligible",
              W.explain_no_worker(MODEL, task=VISION) == "")

        # A task-incapable assigned worker names the task reason (non-vision task,
        # so the task-capability gate actually applies).
        taskcase = _fresh()
        _add(taskcase, "op", engine={"installed": True},
             models=["org/Embed"], caps={"feature-extraction": False})
        W.worker_store = taskcase
        d2 = W.explain_no_worker("org/Embed", task="feature-extraction")
        check("explain_no_worker names a task-capability exclusion",
              "op" in d2 and "task" in d2)
    finally:
        W.worker_store = orig_store

    # --- resolvers.remote seam: detail threaded in; unset ⇒ byte-identical ------
    # The diagnostic is registered by the fleet's placement wiring (the server
    # calls placement.install() at startup); importing workers.py has no side effects.
    from hugpy_fleet.central import placement as _placement
    _placement.install()
    check("placement.install() registers the no-worker diagnostic seam",
          remote._no_worker_diag is not None)

    orig_diag = remote._no_worker_diag
    try:
        remote.set_no_worker_diagnostic(lambda mk, pool=None, task=None:
                                        f"WHY: {mk} is assigned but broken")
        check("_no_worker_detail returns the registered explanation",
              remote._no_worker_detail(MODEL, None, VISION)
              == f"WHY: {MODEL} is assigned but broken")

        # A provider that raises must degrade to "" (never break a request).
        def _boom(*a, **k):
            raise RuntimeError("diag exploded")
        remote.set_no_worker_diagnostic(_boom)
        check("_no_worker_detail swallows a raising provider -> ''",
              remote._no_worker_detail(MODEL, None, VISION) == "")

        remote.set_no_worker_diagnostic(None)
        check("_no_worker_detail returns '' when the seam is unset",
              remote._no_worker_detail(MODEL, None, VISION) == "")

        # DelegatingRunner.run() under policy ON, no worker selected, WITH a
        # diagnostic registered: the refused-local error carries the detail AND
        # still names the policy flag (the existing contract — no regression).
        framework, task = next(iter(remote.FRAMEWORK_RUNNERS))
        Runner = remote.make_delegating_runner(framework, task)
        runner = Runner(types.SimpleNamespace(model_key="test-model"))
        remote._select = lambda mk, pool=None, task=None, **kw: (None, None)
        req = types.SimpleNamespace(request_id="rid-1", pool=None)
        os.environ["HUGPY_NO_LOCAL_SERVING"] = "true"

        remote.set_no_worker_diagnostic(
            lambda mk, pool=None, task=None: "assigned to computron but engine unusable")
        raised = None
        try:
            asyncio.run(runner.run(req))
        except RuntimeError as exc:
            raised = str(exc)
        check("run() refusal still names HUGPY_NO_LOCAL_SERVING (no regression)",
              raised is not None and "HUGPY_NO_LOCAL_SERVING" in raised)
        check("run() refusal now carries the actionable worker detail",
              raised is not None and "assigned to computron but engine unusable" in raised)

        # Seam UNSET: the refusal is rendered from its stored diagnostics record
        # (operator 2026-09-23: facts, not advice) — gate, predicate naming the
        # policy flag, request id and log_ref; no advisory tail.
        remote.set_no_worker_diagnostic(None)
        raised = None
        try:
            asyncio.run(runner.run(req))
        except RuntimeError as exc:
            raised = str(exc)
        check("run() refusal with the seam UNSET is the factual diagnostics rendering",
              raised is not None and raised.startswith("gate=no_worker")
              and "HUGPY_NO_LOCAL_SERVING" in raised and "request=rid-1" in raised
              and "log_ref=" in raised and "Bring a worker online" not in raised)
    finally:
        remote.set_no_worker_diagnostic(orig_diag)
        os.environ.pop("HUGPY_NO_LOCAL_SERVING", None)


def test_no_worker_diagnostic():
    global ok
    ok = 0
    # Lazy import (see the module-scope note): keeps collecting this file a no-op.
    W = importlib.import_module(
        "hugpy_fleet.central.workers")
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    policy = importlib.import_module("hugpy_engine.serve.policy")
    cap = _Capture()
    wlog = logging.getLogger(W.__name__)
    wlog.addHandler(cap)
    wlog.setLevel(logging.WARNING)
    try:
        _run_checks(cap, W, remote, policy)
    finally:
        wlog.removeHandler(cap)
    print(f"\nall {ok} checks passed")


if __name__ == "__main__":
    test_no_worker_diagnostic()
