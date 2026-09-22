"""Env-profiles (stage 1) — per-model dependency venvs for slot children.

Operator problem (2026-07-12): a model needing extra pip deps (e.g. optimum) must
be installable WITHOUT version-conflict risk to the shared worker venv. A PROFILE
= a named venv + manifest {name, packages:[...]} materialized at
<worker_root>/envs/<name>/; a profiled model's SLOT CHILD launches from that venv
(process-seam isolation). The agent's own process never imports from it.

Covers:
  * Settings round-trip + /ops/config validation: profiles + model_profiles
    persist (deep-merge), bad shapes rejected with the BadValue envelope, null
    clears; unknown key still 400.
  * Materialization lifecycle (fake pip/venv seam): a fresh profile materializes
    ready; a manifest-hash CHANGE re-materializes; a pip failure records an ERROR
    state as DATA (never raises); idempotent no-op when already ready; the
    background executor is REGISTERED (restart shuts it down).
  * Heartbeat truth: _effective_config carries the per-profile state map +
    the model->profile attribution.
  * Non-ready profile refusal envelope: get._require_profile_ready RAISES
    LocalEngineUnavailable (never a shared-venv fallback) for materializing/error,
    returns the decision for ready, None for unattributed.
  * Spawn-seam: the python-child interpreter swap + child env construction
    (profiles.child_python / child_env) — the real, tested stage-1 consumer — and
    the slot's status surfaces the profile_bin.

Runs like the other tests here: venv/bin/python tests/test_worker_env_profiles.py
"""
import logging
logging.disable(logging.CRITICAL)

import argparse
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent
profiles = importlib.import_module("hugpy_engine.serve.profiles")
getmod = importlib.import_module("hugpy_engine.llama.runners.get")
slot_agent = importlib.import_module("hugpy_engine.serve.slot_agent")
procutil = importlib.import_module("hugpy_platform.procutil")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ── snapshot everything we mutate so the file leaves no residue ──────────────
_SETTINGS_BACKUP = dict(agent._RUNTIME_SETTINGS)
_SOURCE_BACKUP = dict(agent._SETTINGS_SOURCE)
_RESOLVER_BACKUP = profiles._RESOLVER
_RUN_BACKUP = profiles._run
_WR_BACKUP = os.environ.get("HUGPY_WORKER_ROOT")
_ED_BACKUP = os.environ.get("HUGPY_ENGINE_DIR")

# Materialize into a throwaway worker root, never ~/hugpy-worker.
_WR = tempfile.mkdtemp(prefix="hugpy-profiles-wr-")
os.environ["HUGPY_WORKER_ROOT"] = _WR
os.environ.pop("HUGPY_ENGINE_DIR", None)


# A FAKE pip/venv seam: `python -m venv --clear <dir>` mkdirs the venv bin/python
# (so the profile looks real to child_python/idempotency); a pip install SUCCEEDS
# unless one of its packages is in _FAIL_PKGS (simulates a broken manifest). Every
# call is counted so we can assert re-materialization.
_FAIL_PKGS: set = set()
_run_calls: list = []


def _fake_run(cmd, timeout=1800.0):
    _run_calls.append(list(cmd))
    if len(cmd) >= 3 and cmd[1:3] == ["-m", "venv"]:
        d = cmd[-1]                       # the venv dir (last arg)
        binp = os.path.join(d, "Scripts" if os.name == "nt" else "bin")
        os.makedirs(binp, exist_ok=True)
        exe = "python.exe" if os.name == "nt" else "python"
        with open(os.path.join(binp, exe), "w") as fh:
            fh.write("#!fake\n")
        return
    # a pip install line: fail if any target package is marked broken
    if any(p in _FAIL_PKGS for p in cmd):
        raise RuntimeError("pip failed: could not build wheel for a broken pin")
    return


profiles._run = _fake_run


def _reset_profiles_state():
    with profiles._INFLIGHT_LOCK:
        profiles._INFLIGHT.clear()
    profiles._POOL = None
    _run_calls.clear()
    _FAIL_PKGS.clear()


try:
    # ─────────────────────────────────────────────────────────────────────────
    # 1. paths derive from the worker root (never hardcoded)
    # ─────────────────────────────────────────────────────────────────────────
    check("profiles_root lives under the worker root/envs",
          profiles.profiles_root() == os.path.join(_WR, "envs"))
    check("profile_python is <root>/envs/<name>/bin/python",
          profiles.profile_python("p1")
          == os.path.join(_WR, "envs", "p1", "bin", "python"))
    check("slug_ok accepts a normal name", profiles.slug_ok("optimum-edge"))
    check("slug_ok rejects traversal / spaces",
          not profiles.slug_ok("../x") and not profiles.slug_ok("a b")
          and not profiles.slug_ok("") and not profiles.slug_ok(123))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Materialization lifecycle (fake seam)
    # ─────────────────────────────────────────────────────────────────────────
    _reset_profiles_state()
    res = profiles.materialize("edge", ["optimum", "onnxruntime"])
    check("fresh profile materializes ready", res["state"] == "ready")
    st = profiles.read_state("edge")
    check("profile.json stamped ok + hash + packages",
          st["ok"] is True and st["hash"] == profiles.manifest_hash(["optimum", "onnxruntime"])
          and st["packages"] == ["optimum", "onnxruntime"])
    check("venv created + pip install invoked (2 seam calls)", len(_run_calls) == 2)
    check("state_for reads ready for the current manifest",
          profiles.state_for("edge", ["optimum", "onnxruntime"]) == "ready")

    # idempotent: same manifest -> no re-run
    n_before = len(_run_calls)
    res = profiles.materialize("edge", ["optimum", "onnxruntime"])
    check("idempotent: ready profile is a no-op (no new seam calls)",
          res["state"] == "ready" and len(_run_calls) == n_before)

    # manifest hash CHANGE -> re-materialize
    res = profiles.materialize("edge", ["optimum", "onnxruntime", "ftfy"])
    check("manifest change re-materializes", res["state"] == "ready"
          and len(_run_calls) == n_before + 2)
    check("state_for(old manifest) is NOT ready (hash moved)",
          profiles.state_for("edge", ["optimum", "onnxruntime"]) != "ready")

    # failure -> error state as DATA (never raises)
    _FAIL_PKGS.add("brokenpkg")
    res = profiles.materialize("bad", ["brokenpkg"])
    check("pip failure -> error state (no exception)", res["state"] == "error")
    st = profiles.read_state("bad")
    check("error recorded as data in profile.json",
          st["ok"] is False and "pip failed" in (st["error"] or ""))
    check("state_for surfaces error", profiles.state_for("bad", ["brokenpkg"]) == "error")
    rep = profiles.report({"bad": {"packages": ["brokenpkg"]},
                           "edge": {"packages": ["optimum", "onnxruntime", "ftfy"]}})
    check("report carries per-profile state + error message",
          rep["edge"]["state"] == "ready" and rep["bad"]["state"] == "error"
          and "pip failed" in rep["bad"]["error"])
    _FAIL_PKGS.clear()

    # invalid name never crashes materialize
    res = profiles.materialize("../evil", ["x"])
    check("invalid profile name -> error state (no crash)", res["state"] == "error")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. materialize_all kicks a background, REGISTERED executor
    # ─────────────────────────────────────────────────────────────────────────
    _reset_profiles_state()
    registered = []
    pool = profiles.materialize_all(
        {"a": {"packages": ["pkg-a"]}, "b": {"packages": ["pkg-b"]}},
        register=lambda ex: registered.append(ex))
    check("materialize_all registered its executor (restart shuts it down)",
          len(registered) == 1 and pool is registered[0])
    pool.shutdown(wait=True)                       # join the batch
    check("both profiles materialized ready via the pool",
          profiles.state_for("a", ["pkg-a"]) == "ready"
          and profiles.state_for("b", ["pkg-b"]) == "ready")
    # nothing pending when all already ready
    pool2 = profiles.materialize_all({"a": {"packages": ["pkg-a"]}})
    check("materialize_all is a no-op when all ready (no pool)", pool2 is None)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Heartbeat truth: _effective_config carries states + attributions
    # ─────────────────────────────────────────────────────────────────────────
    _reset_profiles_state()
    profiles.materialize("edge", ["optimum"])
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update({
        "profiles": {"edge": {"packages": ["optimum"]},
                     "wip": {"packages": ["transformers"]}},
        "model_profiles": {"some-model": "edge"},
    })
    profiles.set_model_resolver(agent._resolve_model_profile)
    eff = agent._effective_config()
    check("effective config carries the profiles state map",
          eff["profiles"]["edge"]["state"] == "ready"
          and eff["profiles"]["wip"]["state"] == "materializing")
    check("effective config carries the model->profile attribution",
          eff["model_profiles"] == {"some-model": "edge"})
    # absent when no profiles in play (mirrors residency/pinned)
    agent._RUNTIME_SETTINGS.clear()
    eff = agent._effective_config()
    check("no profiles set -> keys omitted (mirrors residency/pinned)",
          "profiles" not in eff and "model_profiles" not in eff)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Resolver + non-ready REFUSAL envelope (never a shared-venv fallback)
    # ─────────────────────────────────────────────────────────────────────────
    _reset_profiles_state()
    profiles.materialize("edge", ["optimum"])
    agent._RUNTIME_SETTINGS.clear()
    agent._RUNTIME_SETTINGS.update({
        "profiles": {"edge": {"packages": ["optimum"]}},
        "model_profiles": {"ready-model": "edge",
                           "wait-model": "notyet",     # profile never declared
                           "err-model": "boom"},
    })
    agent._RUNTIME_SETTINGS["profiles"]["boom"] = {"packages": ["brokenpkg"]}
    _FAIL_PKGS.add("brokenpkg")
    profiles.materialize("boom", ["brokenpkg"])          # -> error state
    _FAIL_PKGS.clear()
    profiles.set_model_resolver(agent._resolve_model_profile)

    # unattributed model -> None (base serving path untouched)
    check("unattributed model resolves to None",
          getmod._require_profile_ready("plain-model") is None)

    # ready -> returns the decision with the venv bin dir
    dec = getmod._require_profile_ready("ready-model")
    check("ready profile returns the decision + venv bin dir",
          dec and dec["state"] == "ready"
          and dec["bin"] == profiles.profile_bin_dir("edge"))

    # materializing -> RAISE (errors-as-data naming the profile + state)
    try:
        getmod._require_profile_ready("wait-model")
        raised = False
    except getmod.LocalEngineUnavailable as exc:
        raised, msg = True, str(exc)
    check("non-ready (materializing) profile REFUSES with a clear envelope",
          raised and "notyet" in msg and "materializing" in msg
          and "shared venv" in msg)

    # error -> RAISE, carrying the recorded error string
    try:
        getmod._require_profile_ready("err-model")
        raised = False
    except getmod.LocalEngineUnavailable as exc:
        raised, msg = True, str(exc)
    check("errored profile REFUSES and surfaces the pip error",
          raised and "boom" in msg and "pip failed" in msg)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Spawn-seam: python-child interpreter swap + child env construction
    # ─────────────────────────────────────────────────────────────────────────
    _reset_profiles_state()
    profiles.materialize("edge", ["optimum"])            # creates a real bin/python
    bind = profiles.profile_bin_dir("edge")
    default_py = "/usr/bin/python-agent-default"

    check("child_python(None,...) is the agent default (no profile)",
          profiles.child_python(None, default_py) == default_py)
    check("child_python(profile) swaps to the profile venv interpreter",
          profiles.child_python(bind, default_py) == profiles.profile_python("edge"))
    try:
        profiles.child_python("/no/such/bin", default_py)
        missing_raised = False
    except RuntimeError as exc:
        missing_raised = "missing" in str(exc)
    check("child_python raises (never silent shared-venv) when venv python absent",
          missing_raised)

    env = profiles.child_env({"PATH": "/usr/bin", "PYTHONHOME": "/x"}, bind)
    check("child_env prepends the profile bin to PATH",
          env["PATH"].split(os.pathsep)[0] == bind and "/usr/bin" in env["PATH"])
    check("child_env sets VIRTUAL_ENV to the venv root + drops PYTHONHOME",
          env["VIRTUAL_ENV"] == profiles.profile_dir("edge") and "PYTHONHOME" not in env)
    base_env = {"PATH": "/usr/bin"}
    check("child_env is a no-op without a profile",
          profiles.child_env(base_env, None) == base_env)

    # the slot surfaces its child's profile_bin in status (observability)
    slot = slot_agent.Slot()
    slot.profile_bin = bind
    check("slot status surfaces the profile_bin", slot.status()["profile_bin"] == bind)
    check("slot status defaults profile_bin to None",
          slot_agent.Slot().status()["profile_bin"] is None)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. /ops/config validation + settings round-trip (deep-merge, null clears)
    # ─────────────────────────────────────────────────────────────────────────
    _sched_orig = agent._schedule_restart
    agent._schedule_restart = lambda *a, **k: None   # no Timer side effects here
    try:
        tmpd = tempfile.mkdtemp(prefix="hugpy-profiles-route-")
        state = agent.WorkerState(name="t", url=None, worker_id="w-prof")
        state.args = argparse.Namespace(id_file=os.path.join(tmpd, "agent.id"))
        client = agent.build_app(state).test_client()

        r = client.post("/ops/config", json={
            "profiles": {"edge": {"packages": ["optimum", "onnxruntime"]}},
            "model_profiles": {"m1": "edge"}})
        body = r.get_json()
        check("POST profiles+model_profiles -> 200 ok",
              r.status_code == 200 and body["ok"] is True)
        check("profile manifest persisted",
              body["settings"]["profiles"]["edge"]["packages"] == ["optimum", "onnxruntime"])
        check("attribution persisted", body["settings"]["model_profiles"] == {"m1": "edge"})
        check("settings file on disk carries the profile",
              "optimum" in open(state.args.id_file + ".settings.json").read())

        # deep-merge: adding a second profile keeps the first
        r = client.post("/ops/config", json={"profiles": {"draft": {"packages": ["diffusers"]}}})
        merged = r.get_json()["settings"]["profiles"]
        check("profiles deep-merge (both kept)", "edge" in merged and "draft" in merged)

        # null clears one profile
        r = client.post("/ops/config", json={"profiles": {"draft": None}})
        check("null clears one profile (deep-merge)",
              "draft" not in r.get_json()["settings"]["profiles"] and
              "edge" in r.get_json()["settings"]["profiles"])

        # bad name rejected
        r = client.post("/ops/config", json={"profiles": {"../evil": {"packages": ["x"]}}})
        check("slug-unsafe profile name -> 400 BadValue",
              r.status_code == 400 and r.get_json()["error"]["code"] == "BadValue"
              and "slug-safe" in r.get_json()["error"]["message"])

        # empty packages rejected
        r = client.post("/ops/config", json={"profiles": {"e": {"packages": []}}})
        check("empty packages -> 400", r.status_code == 400
              and "non-empty" in r.get_json()["error"]["message"])

        # non-list packages rejected
        r = client.post("/ops/config", json={"profiles": {"e": {"packages": "optimum"}}})
        check("non-list packages -> 400", r.status_code == 400)

        # non-string package rejected
        r = client.post("/ops/config", json={"profiles": {"e": {"packages": ["ok", 5]}}})
        check("non-string package element -> 400", r.status_code == 400)

        # model_profiles must be a slug-safe name
        r = client.post("/ops/config", json={"model_profiles": {"m2": "bad name"}})
        check("model_profiles bad name -> 400", r.status_code == 400
              and r.get_json()["error"]["code"] == "BadValue")

        # null clears an attribution
        r = client.post("/ops/config", json={"model_profiles": {"m1": None}})
        check("null clears an attribution", "model_profiles" not in r.get_json()["settings"])

        # unknown key still rejected wholesale
        r = client.post("/ops/config", json={"bogus": 1})
        check("unknown key still 400 UnknownKeys",
              r.status_code == 400 and r.get_json()["error"]["code"] == "UnknownKeys")
    finally:
        agent._schedule_restart = _sched_orig

finally:
    # ── restore all globals/env so the file leaves no residue ────────────────
    agent._RUNTIME_SETTINGS.clear(); agent._RUNTIME_SETTINGS.update(_SETTINGS_BACKUP)
    agent._SETTINGS_SOURCE.clear(); agent._SETTINGS_SOURCE.update(_SOURCE_BACKUP)
    profiles.set_model_resolver(_RESOLVER_BACKUP)
    profiles._run = _RUN_BACKUP
    with profiles._INFLIGHT_LOCK:
        profiles._INFLIGHT.clear()
    profiles._POOL = None
    for k, v in (("HUGPY_WORKER_ROOT", _WR_BACKUP), ("HUGPY_ENGINE_DIR", _ED_BACKUP)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print(f"\nall {ok} checks passed")

# Re-enable logging for the rest of the pytest session (the disable above is import-time only).
logging.disable(logging.NOTSET)
