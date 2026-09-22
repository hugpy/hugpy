"""Worker per-task capability honesty (2026-07-11).

Yesterday three /ml requests reached workers whose canonical venv lacked an
optional ML dep and failed AT REQUEST TIME (sentence-transformers missing,
whisper NoneType, and numpy>=2.5 breaking `import whisper` outright). The worker
now advertises which tasks it can ACTUALLY run, from the same find_spec probe
central's /ml readiness uses, with a real guarded import for the whisper landmine.

Three halves:
  * task_deps.task_capabilities() maps task -> find_spec(module) (cheap, no imports).
  * agent._task_capabilities() overlays the whisper SPECIAL CASE: find_spec True but
    `import whisper` failing must advertise ASR unavailable (find_spec insufficient),
    TTL-cached so heartbeats stay cheap.
  * The map ROUND-TRIPS into the central registry (register + heartbeat); a LEGACY
    worker that omits it reads as capable (no regression).

Runs like the other tests here:
    venv/bin/python tests/test_worker_task_capabilities.py
"""
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

task_deps = importlib.import_module("hugpy_engine.task_deps")
agent = importlib.import_module("hugpy_fleet.worker.agent")
W = importlib.import_module(
    "hugpy_fleet.central.workers")
from worker_store_isolation import isolated_worker_store  # noqa: E402

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# A fake find_spec probe: only the modules in `present` resolve.
_real_have = task_deps.have
def fake_have(present):
    return lambda mod: mod in present


# --- base map: task -> find_spec(module) -----------------------------------
try:
    # Everything present -> every task True.
    all_mods = {mod for mod, _ in task_deps.TASK_DEPS.values()}
    task_deps.have = fake_have(all_mods)
    caps = task_deps.task_capabilities()
    check("base map advertises every task when all modules resolve",
          all(caps.values()) and set(caps) == set(task_deps.TASK_DEPS))

    # Drop sentence_transformers -> both embed tasks go False, others stay True.
    task_deps.have = fake_have(all_mods - {"sentence_transformers"})
    caps = task_deps.task_capabilities()
    check("missing sentence_transformers -> feature-extraction False",
          caps["feature-extraction"] is False)
    check("missing sentence_transformers -> sentence-similarity False",
          caps["sentence-similarity"] is False)
    check("missing sentence_transformers leaves keyword-extraction True",
          caps["keyword-extraction"] is True)

    # Drop keybert -> keyword-extraction False.
    task_deps.have = fake_have(all_mods - {"keybert"})
    check("missing keybert -> keyword-extraction False",
          task_deps.task_capabilities()["keyword-extraction"] is False)
finally:
    task_deps.have = _real_have


# --- whisper SPECIAL CASE (find_spec insufficient) -------------------------
# find_spec True but `import whisper` failing (numba/numpy landmine) must report
# ASR unavailable — a find_spec-only probe would over-advertise it.
def _reset_whisper_probe():
    agent._WHISPER_PROBE["ok"] = None
    agent._WHISPER_PROBE["at"] = 0.0

_saved_whisper = sys.modules.get("whisper")
try:
    # find_spec False -> False fast, no import attempted.
    task_deps.have = fake_have(set())
    _reset_whisper_probe()
    check("whisper not resolvable -> _whisper_importable False (no import)",
          agent._whisper_importable() is False)

    # find_spec True but import raises (sys.modules[...]=None makes `import` fail).
    task_deps.have = fake_have({"whisper"})
    sys.modules["whisper"] = None
    _reset_whisper_probe()
    check("whisper resolvable but `import whisper` fails -> False (the landmine)",
          agent._whisper_importable() is False)

    # find_spec True and import succeeds -> True.
    sys.modules["whisper"] = types.ModuleType("whisper")
    _reset_whisper_probe()
    check("whisper resolvable and importable -> True",
          agent._whisper_importable() is True)

    # TTL cache: a now-broken import within the TTL still returns the cached True
    # (heartbeats stay cheap; they don't re-import every beat).
    sys.modules["whisper"] = None
    check("whisper probe is TTL-cached (no re-import within the window)",
          agent._whisper_importable() is True)

    # agent._task_capabilities overlays the whisper result onto the base map.
    task_deps.have = fake_have({"whisper", "sentence_transformers", "transformers",
                                "keybert", "diffusers", "llama_cpp", "pdfplumber", "bs4"})
    sys.modules["whisper"] = types.ModuleType("whisper")
    _reset_whisper_probe()
    caps = agent._task_capabilities()
    check("agent map: ASR reflects the real import (True here)",
          caps["automatic-speech-recognition"] is True)
    sys.modules["whisper"] = None
    _reset_whisper_probe()
    caps = agent._task_capabilities()
    check("agent map: ASR False when `import whisper` fails despite find_spec",
          caps["automatic-speech-recognition"] is False)
    check("agent map still advertises the other tasks from find_spec",
          caps["feature-extraction"] is True and caps["keyword-extraction"] is True)
finally:
    task_deps.have = _real_have
    if _saved_whisper is None:
        sys.modules.pop("whisper", None)
    else:
        sys.modules["whisper"] = _saved_whisper
    _reset_whisper_probe()


# --- registry round-trip ---------------------------------------------------
# k3 isolation: isolated_worker_store() also redirects the assignment-memory
# sidecar (settings.manifest_path) — see tests/worker_store_isolation.py.
store, tmp = isolated_worker_store(prefix="hugpy-task-caps-")

CAPS = {"feature-extraction": True, "automatic-speech-recognition": False,
        "keyword-extraction": True}
view = store.register(name="ae", url="http://ae:9100", worker_id="wid-caps",
                      task_capabilities=CAPS)
check("register stores task_capabilities", view.get("task_capabilities") == CAPS)

got = store.get("wid-caps")
check("_public_view exposes task_capabilities on the /llm/workers row",
      got.get("task_capabilities") == CAPS)

# A heartbeat refreshes it (an /ops/pip that adds a dep flips the task in one beat).
CAPS2 = {**CAPS, "automatic-speech-recognition": True}
view2 = store.heartbeat("wid-caps", task_capabilities=CAPS2)
check("heartbeat refreshes task_capabilities within one beat",
      view2.get("task_capabilities") == CAPS2)

# An omitted heartbeat field must not clobber the prior value.
store.heartbeat("wid-caps")
check("omitted heartbeat field doesn't wipe prior task_capabilities",
      store.get("wid-caps").get("task_capabilities") == CAPS2)


# --- backward compatibility ------------------------------------------------
store.register(name="legacy", url="http://legacy:9100", worker_id="wid-legacy")
lg = store.get("wid-legacy")
check("legacy worker: task_capabilities absent/None",
      lg.get("task_capabilities") is None)
check("legacy worker is assumed CAPABLE for any task (no regression)",
      W._task_capable(lg, "feature-extraction") is True
      and W._task_capable(lg, "automatic-speech-recognition") is True)

# _task_capable semantics on an ADVERTISING worker.
adv = store.get("wid-caps")   # task_capabilities == CAPS2
check("_task_capable True for an advertised-True task",
      W._task_capable(adv, "feature-extraction") is True)
store.heartbeat("wid-caps", task_capabilities={"feature-extraction": False})
adv = store.get("wid-caps")
check("_task_capable False for an advertised-False task",
      W._task_capable(adv, "feature-extraction") is False)
check("_task_capable True for a task the worker doesn't enumerate (unknown=capable)",
      W._task_capable(adv, "keyword-extraction") is True)
check("_task_capable True for a None task (non-ML routing never gates)",
      W._task_capable(adv, None) is True)
check("_task_capable True for image-text-to-text (vision defers to engine gate)",
      W._task_capable({"task_capabilities": {"image-text-to-text": False}},
                      "image-text-to-text") is True)

print(f"\nall {ok} checks passed")
