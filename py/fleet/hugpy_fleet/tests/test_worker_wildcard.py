"""Per-worker WILDCARD routing opt-in, key-form matching, allocation hard
scope, the blocked-sibling guard and the star ranking tie-break.

Converted from the monolith's script-style ``test_worker_wildcard.py``
sections [3]-[7] (the flag-store CRUD and route sections stay with the
engine/server). Flag stores are redirected into a tmpdir.
"""

from __future__ import annotations

import os

import pytest

from hugpy_control.settings import settings_store
from hugpy_fleet.central import blocklist as bl
from hugpy_fleet.central import workers as W

from worker_store_isolation import isolated_worker_store

mc = pytest.importorskip("hugpy_engine.config.models.models_config",
                         reason="engine model-config store required for the flag stores")

GPU = [{"name": "rtx", "memory_free": 8 * 2**30, "memory_total": 24 * 2**30}]


@pytest.fixture(autouse=True)
def isolated_flag_stores(tmp_path):
    orig_discovery = mc.MODELS_DISCOVERY_PATH
    mc.MODELS_DISCOVERY_PATH = os.path.join(str(tmp_path), "model_discovery.json")
    orig_path, orig_cache = settings_store._path, settings_store._cache
    settings_store._path = str(tmp_path / "settings.json")
    settings_store._cache = None
    try:
        yield
    finally:
        mc.MODELS_DISCOVERY_PATH = orig_discovery
        settings_store._path, settings_store._cache = orig_path, orig_cache


def _store(prefix):
    store, _tmp = isolated_worker_store(prefix=prefix)

    def add(wid, models, **kw):
        store.register(name=wid, url=f"http://{wid}:9100", worker_id=wid, models=models, **kw)
        store.set_admission(wid, "approved")
    return store, add


def test_match_keys_unifies_tilde_and_slash_tails():
    assert W._match_keys("Qwen~X") == {"Qwen~X", "qwen~x", "X", "x"}
    assert W._match_keys("X") == {"X", "x"}
    assert W._match_keys("Qwen~X") & W._match_keys("X")
    assert W._match_keys("X") & W._match_keys("Qwen~X")
    assert W._match_keys("unsloth~X") & W._match_keys("X")
    assert W._match_keys("Qwen~X") & W._match_keys("unsloth~X")
    assert not (W._match_keys("Qwen~X") & W._match_keys("unsloth~Y"))
    assert W._match_keys("Qwen/Qwen2.5-Coder") & W._match_keys("qwen2.5-coder")


def test_sealed_default_wildcard_catch_and_hard_gates():
    ASSIGNED, STRAY, EMBED = "org~Assigned-Model", "org~Stray-Model", "feature-extraction"
    store, add = _store("hugpy-wc-route-")
    add("sealed", [ASSIGNED])
    assert {w["id"] for w in store.workers_for_model(ASSIGNED)} == {"sealed"}
    assert store.workers_for_model(STRAY) == []
    assert store.pick_for_model(STRAY) is None

    mc.set_worker_wildcard("sealed", True)
    cands = store.workers_for_model(STRAY)
    assert [w["id"] for w in cands] == ["sealed"]
    assert cands[0].get("_wildcard_catch") is True
    home = store.workers_for_model(ASSIGNED)
    assert home and not home[0].get("_wildcard_catch")
    assert "_wildcard_catch" not in (store._load().get("sealed") or {})

    add("wild-incap", [], task_capabilities={EMBED: False})
    mc.set_worker_wildcard("wild-incap", True)
    assert "wild-incap" not in {w["id"] for w in store.workers_for_model(STRAY, task=EMBED)}
    assert "wild-incap" in {w["id"] for w in store.workers_for_model(STRAY)}


def test_resident_is_de_facto_designation_and_flag_store_failure_degrades(monkeypatch):
    store, add = _store("hugpy-wc-resident-")
    add("holder", [])
    store.heartbeat("holder", loaded_models=["org~Resident-Model"])
    res = store.workers_for_model("org~Resident-Model")
    assert [w["id"] for w in res] == ["holder"]
    assert not res[0].get("_wildcard_catch")

    def _boom():
        raise RuntimeError("flag store down")
    monkeypatch.setattr(mc, "worker_wildcard_state", _boom)
    assert {w["id"] for w in store.workers_for_model("org~Resident-Model")} == {"holder"}


def test_allocation_is_hard_scope():
    ASSIGNED = "org~Assigned-Model"
    store, add = _store("hugpy-wc-rank-")
    add("home-box", [ASSIGNED])
    add("wild-box", [], gpus=GPU)
    mc.set_worker_wildcard("wild-box", True)
    assert [w["id"] for w in store.candidates_for_model(ASSIGNED)] == ["home-box"]
    assert (store.pick_for_model(ASSIGNED) or {}).get("id") == "home-box"
    store.register(name="home-box", url="http://home-box:9100", worker_id="home-box",
                   models=[ASSIGNED], engine={"installed": False})
    assert store.pick_for_model(ASSIGNED) is None
    assert store.workers_for_model(ASSIGNED) == []


def test_blocked_sibling_guard():
    store, add = _store("hugpy-wc-block-")
    bl.unblock("B~X")
    add("sib", ["B~X"])
    bl.block("B~X", note="test sibling block")
    try:
        assert store.workers_for_model("A~X") == []
        assert store.workers_for_model("B~X") == []
        add("bare", ["X"])
        assert {w["id"] for w in store.workers_for_model("A~X")} == {"bare"}
    finally:
        bl.unblock("B~X")
    assert {w["id"] for w in store.workers_for_model("A~X")} == {"sib", "bare"}


def _clear_stars():
    for wid in list(mc.worker_boot_prewarm_state().keys()):
        mc.set_worker_boot_prewarm(wid, None, False)


def test_star_breaks_ties_but_never_beats_warm_or_home():
    RANKED = "org~Ranked-Model"
    _clear_stars()
    store, add = _store("hugpy-wc-star-a-")
    add("plain-box", [RANKED])
    add("star-box", [RANKED])
    mc.set_worker_boot_prewarm("star-box", RANKED, True)
    assert [w["id"] for w in store.candidates_for_model(RANKED)][0] == "star-box"
    assert (store.pick_for_model(RANKED) or {}).get("id") == "star-box"

    _clear_stars()
    store, add = _store("hugpy-wc-star-b-")
    add("warm-box", [RANKED])
    add("star-cold", [RANKED])
    store.heartbeat("warm-box", loaded_models=[RANKED])
    mc.set_worker_boot_prewarm("star-cold", RANKED, True)
    assert [w["id"] for w in store.candidates_for_model(RANKED)][0] == "warm-box"

    _clear_stars()
    store, add = _store("hugpy-wc-star-c-")
    add("home-plain", [RANKED])
    add("wild-star", [], gpus=GPU)
    mc.set_worker_wildcard("wild-star", True)
    mc.set_worker_boot_prewarm("wild-star", RANKED, True)
    assert [w["id"] for w in store.candidates_for_model(RANKED)] == ["home-plain"]

    _clear_stars()
    store, add = _store("hugpy-wc-star-d-")
    add("plainer", ["Base"])
    add("starrer", ["Base"])
    mc.set_worker_boot_prewarm("starrer", "Owner~Base", True)
    assert [w["id"] for w in store.candidates_for_model("Base")][0] == "starrer"
    _clear_stars()
