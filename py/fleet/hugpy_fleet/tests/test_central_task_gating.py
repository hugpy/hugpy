"""Central per-task capability routing gate (2026-07-11).

Central routes by model ASSIGNMENT, which is why a request for a task the worker
couldn't run reached it and failed at request time. workers_for_model now also
skips a worker that AFFIRMATIVELY advertises it can't run the request's task
(task_capabilities), with the same say-why log the env-tier gate uses. Legacy
workers (no task_capabilities) and unknown tasks are assumed capable, and a None
task never gates — so a pre-feature fleet routes exactly as before.

Runs like the other tests here:
    venv/bin/python tests/test_central_task_gating.py
"""
import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_store_isolation import isolated_worker_store

W = importlib.import_module(
    "hugpy_fleet.central.workers")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# Capture the workers-module warnings so we can assert the say-why log.
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())
    def reset(self):
        self.records = []

_cap = _Capture()
_wlog = logging.getLogger(W.__name__)
_wlog.addHandler(_cap)
_wlog.setLevel(logging.WARNING)


MODEL = "org/Embed-Model"
EMBED = "feature-extraction"


def _fresh_store():
    # k3 isolation: isolated_worker_store() also redirects the assignment-
    # memory sidecar (settings.manifest_path), which a bare
    # WorkerStore(path=tmp) does NOT — see tests/worker_store_isolation.py.
    store, _tmp = isolated_worker_store(prefix="hugpy-task-gate-")
    return store


def _add(store, wid, caps, name=None):
    """Register an APPROVED worker assigned to MODEL with the given caps."""
    store.register(name=name or wid, url=f"http://{wid}:9100", worker_id=wid,
                   models=[MODEL], task_capabilities=caps)
    store.set_admission(wid, "approved")


# --- a worker advertising task:False is skipped for that task --------------
store = _fresh_store()
_add(store, "cap", {EMBED: True})
_add(store, "incap", {EMBED: False})
ids = {w["id"] for w in store.workers_for_model(MODEL, task=EMBED)}
check("worker advertising the task False is skipped", ids == {"cap"})
check("without a task, NEITHER is skipped (task=None never gates)",
      {w["id"] for w in store.workers_for_model(MODEL)} == {"cap", "incap"})

# The pick lands on the capable worker, not the incapable one.
pick = store.pick_for_model(MODEL, task=EMBED)
check("pick_for_model chooses the capable worker", pick and pick["id"] == "cap")


# --- none capable -> empty + the honest say-why log names the reason -------
store = _fresh_store()
_add(store, "incap", {EMBED: False})
_cap.reset()
res = store.workers_for_model(MODEL, task=EMBED)
check("model has a server but it can't do the task -> no eligible worker",
      res == [])
said = any(EMBED in m and "skipped" in m for m in _cap.records)
check("say-why log names the task and that workers were skipped", said)
check("pick_for_model returns None (caller falls back to local)",
      store.pick_for_model(MODEL, task=EMBED) is None)


# --- legacy worker (no task_capabilities) is assumed capable ---------------
store = _fresh_store()
store.register(name="legacy", url="http://legacy:9100", worker_id="legacy",
               models=[MODEL])   # NO task_capabilities
store.set_admission("legacy", "approved")
check("legacy worker serves the task (assumed capable, no regression)",
      {w["id"] for w in store.workers_for_model(MODEL, task=EMBED)} == {"legacy"})


# --- legacy MIXED fleet: legacy served, advertised-False skipped -----------
store = _fresh_store()
store.register(name="legacy", url="http://legacy:9100", worker_id="legacy",
               models=[MODEL])
store.set_admission("legacy", "approved")
_add(store, "incap", {EMBED: False})
_add(store, "cap", {EMBED: True})
ids = {w["id"] for w in store.workers_for_model(MODEL, task=EMBED)}
check("mixed fleet: legacy + advertised-capable served, advertised-False dropped",
      ids == {"legacy", "cap"})


# --- unknown task on an advertising worker = capable -----------------------
store = _fresh_store()
_add(store, "cap", {EMBED: True})   # doesn't enumerate keyword-extraction
check("a task the worker doesn't enumerate is assumed capable",
      {w["id"] for w in store.workers_for_model(MODEL, task="keyword-extraction")}
      == {"cap"})


# --- image-text-to-text is EXCLUDED (vision defers to the engine gate) -----
store = _fresh_store()
_add(store, "novis", {"image-text-to-text": False})
check("image-text-to-text is NOT gated here (engine.supports_vision is authority)",
      {w["id"] for w in store.workers_for_model(MODEL, task="image-text-to-text")}
      == {"novis"})


# --- candidates_for_model (the reroute list) is task-filtered too ----------
store = _fresh_store()
_add(store, "cap", {EMBED: True})
_add(store, "incap", {EMBED: False})
cand_ids = {w["id"] for w in store.candidates_for_model(MODEL, task=EMBED)}
check("candidates_for_model (reroute list) also drops the incapable worker",
      cand_ids == {"cap"})

_wlog.removeHandler(_cap)
print(f"\nall {ok} checks passed")
