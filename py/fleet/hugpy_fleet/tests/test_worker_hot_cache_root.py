"""Per-worker HOT-CACHE ROOT attribution (operator ask 2026-07-12).

Makes the box-local NVMe hot tier's root a first-class per-worker setting
instead of env-only: settable from central via /ops/config, projected onto the
HUGPY_HOT_CACHE_ROOT env the tier reads live, reported (value + source) in the
effective config so a /llm/workers row carries the truth. This ONLY attributes
the root per worker — the tier stays an automatic LRU cache and the shared store
stays the source of truth.

Covers:
  * _apply_settings_env projection + precedence: settings > env base > unset,
    with the base-sentinel reversion (a clear reverts to the true drop-in/env
    base, never the last projected value) that survives a simulated re-exec.
  * hot_cache._root() honors the projected settings value over the env base
    (resolution order proven at the tier itself).
  * _effective_config always carries hot_cache_root + hot_cache_root_source.
  * /ops/config validation: absolute path stored (+ trailing-slash normalized);
    relative path rejected with the clear BadValue envelope; null clears.

Runs like the other tests here: venv/bin/python tests/test_worker_hot_cache_root.py
"""
import logging
logging.disable(logging.CRITICAL)          # silence import-time registry chatter

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib

from hugpy_fleet.worker import agent

# import_module gets the REAL serve.hot_cache (managers/__init__ star-imports can
# shadow the subpackage attribute), same module the agent's env projection feeds.
hc = importlib.import_module("hugpy_engine.serve.hot_cache")
procutil = importlib.import_module("hugpy_platform.procutil")

_ENV = agent._ENV_HOT_CACHE_ROOT
_BASE_ENV = agent._HOT_CACHE_ROOT_BASE_ENV

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# Snapshot every global we mutate so the file leaves no residue.
_SETTINGS_BACKUP = dict(agent._RUNTIME_SETTINGS)
_SOURCE_BACKUP = dict(agent._SETTINGS_SOURCE)
_ENV_BACKUP = os.environ.get(_ENV)
_BASE_BACKUP = os.environ.get(_BASE_ENV)

tmp = tempfile.mkdtemp(prefix="hugpy-hcr-")
args = argparse.Namespace(id_file=os.path.join(tmp, "agent.id"))


def _boot():
    """Simulate a FRESH process (execv drops the base sentinel is NOT true — but a
    cold boot has never captured it): clear the sentinel so the next apply
    recaptures the current env as the base."""
    os.environ.pop(_BASE_ENV, None)


# --------------------------------------------------------------------------- #
# 1. no env base — settings set, then cleared (boot + re-exec)
# --------------------------------------------------------------------------- #
_boot()
os.environ.pop(_ENV, None)                              # cold box: tier off
hot_a = os.path.join(tmp, "hotA")
agent._save_settings(args, {"hot_cache_root": hot_a})
agent._apply_settings_env(args)
eff = agent._effective_config()
check("settings projected onto the env", os.environ.get(_ENV) == hot_a)
check("source reported as settings", agent._SETTINGS_SOURCE.get("hot_cache_root") == "settings")
check("effective config carries the value", eff.get("hot_cache_root") == hot_a)
check("effective config carries source=settings", eff.get("hot_cache_root_source") == "settings")
check("best-effort makedirs materialized the root at apply time", os.path.isdir(hot_a))

# clear it — this is a re-exec (sentinel survives): base was "" (no env), so a
# clear reverts to unset == tier off, and the console shows the honest "" .
agent._save_settings(args, {})
agent._apply_settings_env(args)
eff = agent._effective_config()
check("null/cleared: env unset again", os.environ.get(_ENV) is None)
check("null/cleared: effective value is '' (tier off)", eff.get("hot_cache_root") == "")
check("null/cleared: source falls back to default", eff.get("hot_cache_root_source") == "default")
check("effective config ALWAYS carries both keys",
      "hot_cache_root" in eff and "hot_cache_root_source" in eff)


# --------------------------------------------------------------------------- #
# 2. env base present — precedence settings > env, with base-sentinel reversion
# --------------------------------------------------------------------------- #
_boot()
base = os.path.join(tmp, "envbase")
os.makedirs(base, exist_ok=True)
os.environ[_ENV] = base                                # a drop-in / env root
agent._save_settings(args, {})                         # no settings override
agent._apply_settings_env(args)
eff = agent._effective_config()
check("env base only: env preserved", os.environ.get(_ENV) == base)
check("env base only: source reported as env", eff.get("hot_cache_root_source") == "env")
check("env base only: effective shows the base", eff.get("hot_cache_root") == base)

# settings win over the env base (re-exec: sentinel carries the true base)
hot_b = os.path.join(tmp, "hotB")
agent._save_settings(args, {"hot_cache_root": hot_b})
agent._apply_settings_env(args)
eff = agent._effective_config()
check("settings win over env base", os.environ.get(_ENV) == hot_b)
check("settings-win source=settings", eff.get("hot_cache_root_source") == "settings")
# resolution proven at the tier itself: hot_cache._root() reads the projected env
check("hot_cache._root() honors settings over env", hc._root() == hot_b)

# clear -> revert to the TRUE env base (never the last projected hot_b)
agent._save_settings(args, {})
agent._apply_settings_env(args)
eff = agent._effective_config()
check("clear reverts to the true env base (not last projected)", os.environ.get(_ENV) == base)
check("reverted source=env", eff.get("hot_cache_root_source") == "env")
check("hot_cache._root() follows the revert", hc._root() == base)


# --------------------------------------------------------------------------- #
# 3. /ops/config validation: absolute stored, relative rejected, null clears
# --------------------------------------------------------------------------- #
_reexec_orig = procutil.reexec
procutil.reexec = lambda: None             # the route re-execs 0.5s after a POST
try:
    tmpd2 = tempfile.mkdtemp(prefix="hugpy-hcr-route-")
    state = agent.WorkerState(name="t", url=None, worker_id="w-hcr")
    state.args = argparse.Namespace(id_file=os.path.join(tmpd2, "agent.id"))
    client = agent.build_app(state).test_client()

    good = os.path.join(tmpd2, "hot990", "hugpy-hot-cache")
    r = client.post("/ops/config", json={"hot_cache_root": good})
    body = r.get_json()
    check("POST absolute hot_cache_root -> 200 ok", r.status_code == 200 and body["ok"] is True)
    check("absolute path persisted in settings", body["settings"]["hot_cache_root"] == good)
    check("settings file on disk carries the root",
          good in open(state.args.id_file + ".settings.json").read())

    # trailing slash normalized off (matches comfy_url's rstrip idiom)
    r = client.post("/ops/config", json={"hot_cache_root": good + "/"})
    check("trailing slash normalized", r.get_json()["settings"]["hot_cache_root"] == good)

    # relative path rejected with the clear BadValue envelope
    r = client.post("/ops/config", json={"hot_cache_root": "relative/hot"})
    body = r.get_json()
    check("relative path -> 400", r.status_code == 400)
    check("rejection is a BadValue envelope", body["error"]["code"] == "BadValue")
    check("error message names 'absolute'", "absolute" in body["error"]["message"])

    # a non-string is likewise rejected (not an absolute path)
    r = client.post("/ops/config", json={"hot_cache_root": 12345})
    check("non-string -> 400", r.status_code == 400)

    # null clears the override
    r = client.post("/ops/config", json={"hot_cache_root": None})
    body = r.get_json()
    check("null clears the override",
          r.status_code == 200 and "hot_cache_root" not in body["settings"])

    # empty string clears too (idempotent)
    r = client.post("/ops/config", json={"hot_cache_root": ""})
    check("empty string clears (idempotent)",
          r.status_code == 200 and "hot_cache_root" not in r.get_json()["settings"])

    # unknown key still rejected wholesale (the released-worker rejection path)
    r = client.post("/ops/config", json={"bogus_key": "x"})
    check("unknown key still 400 UnknownKeys (0.1.168 rejection path)",
          r.status_code == 400 and r.get_json()["error"]["code"] == "UnknownKeys")
finally:
    procutil.reexec = _reexec_orig


# --------------------------------------------------------------------------- #
# restore globals so this file leaves no residue
# --------------------------------------------------------------------------- #
agent._RUNTIME_SETTINGS.clear()
agent._RUNTIME_SETTINGS.update(_SETTINGS_BACKUP)
agent._SETTINGS_SOURCE.clear()
agent._SETTINGS_SOURCE.update(_SOURCE_BACKUP)
for k, v in ((_ENV, _ENV_BACKUP), (_BASE_ENV, _BASE_BACKUP)):
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v

print(f"\nall {ok} checks passed")

# Re-enable logging for the rest of the pytest session (the disable above is import-time only).
logging.disable(logging.NOTSET)
