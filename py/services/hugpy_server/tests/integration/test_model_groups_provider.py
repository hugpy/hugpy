"""MODEL GROUPS — the CENTRAL-SIDE provider: the kill switch and the seam.

The pure pipeline is tested in test_model_groups.py and the off-path snapshot in
test_model_groups_offpath.py. This file covers the impure half — the thing that
reads the settings store and decides whether the feature runs at all — because
the kill switch is the operator's revert lever and a switch nobody tested is not
a lever.

Settings are redirected to a tmp file via HUGPY_SETTINGS_PATH (honoured by
SettingsStore.path()), so nothing here reads or writes the live
/mnt/llm_storage/projects/settings.json.

    ./venv/bin/pytest tests/test_model_groups_provider.py -q
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MG = importlib.import_module(
    "hugpy_fleet.central.model_groups")
from hugpy_control.settings import settings_store


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """A settings store that is OURS, whatever else the suite did to it.

    ⚠ ``HUGPY_SETTINGS_PATH`` ALONE IS NOT ISOLATION. ``SettingsStore.path()``
    checks the instance's ``_path`` FIRST and only falls back to the env var,
    and at least one other file in this suite (tests/test_comms_wiring.py:98)
    assigns ``settings_store._path`` at MODULE scope and never restores it. Any
    file importing it before this one therefore silently redirects our writes
    into ITS tmp file — where they persist across our per-test tmp_path, so
    test A's ``_enable(True)`` leaks into test B and the kill-switch tests read
    a flag nobody in this file set. That is the k51 cross-file pollution class,
    caught in-sweep: these tests passed standalone and failed behind
    test_comms_wiring.

    So pin ``_path`` too, and restore whatever we found — leaving our own
    redirect behind would make this file the next file's polluter.
    """
    prev_path = settings_store._path
    settings_store._path = str(tmp_path / "settings.json")
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.delenv(MG.ENV_FLAG, raising=False)
    # The store caches reads for 3s and the derived registry for 5s; both must
    # start cold or a previous test's answer leaks into this one.
    settings_store._cache = None
    settings_store._cache_at = 0.0
    MG._cache.update({"at": 0.0, "groups": None, "sig": None})
    yield
    settings_store._path = prev_path
    settings_store._cache = None
    settings_store._cache_at = 0.0
    MG._cache.update({"at": 0.0, "groups": None, "sig": None})


def _enable(on=True):
    settings_store.set(MG.SETTINGS_NS, MG.ENABLED_KEY, on)
    settings_store._cache_at = 0.0        # defeat the 3s read cache


# ---------------------------------------------------------------------------
# THE KILL SWITCH — the operator's revert lever
# ---------------------------------------------------------------------------
def test_default_is_off():
    """Shipped default. Not "off in the fixture" — off because nothing said on."""
    assert MG.enabled_state() == (False, "default")
    assert MG.is_enabled() is False


def test_the_seam_is_a_no_op_when_off():
    """THE OFF-PATH, at the provider. Returns None = 'change nothing' — and it
    returns it BEFORE reading the catalog, the registry or any worker."""
    assert MG.member_for_model("Qwen2.5-7B-Instruct-GGUF") is None
    assert MG.member_for_model("anything", "pool", "text-generation") is None


def test_a_settings_write_turns_it_on():
    _enable(True)
    assert MG.enabled_state() == (True, "settings")
    _enable(False)
    assert MG.enabled_state() == (False, "settings")


@pytest.mark.parametrize("val", ["off", "0", "false", "no", "OFF", " off "])
def test_env_is_a_hard_off_that_outranks_the_setting(monkeypatch, val):
    """HUGPY_MODEL_GROUPS=off wins even with the setting explicitly true —
    the belt to the settings-store braces, for an operator who needs the
    feature gone RIGHT NOW and has a systemd drop-in but no token to hand."""
    _enable(True)
    assert MG.is_enabled() is True
    monkeypatch.setenv(MG.ENV_FLAG, val)
    assert MG.enabled_state() == (False, "env-off")
    assert MG.member_for_model("Qwen2.5-7B-Instruct-GGUF") is None


def test_env_values_that_are_not_an_off_do_not_force_it_on(monkeypatch):
    """The env var is an OFF switch only. It can never enable the feature —
    enabling is an operator action through the gated settings write."""
    monkeypatch.setenv(MG.ENV_FLAG, "on")
    assert MG.is_enabled() is False        # setting still unset => default off


def test_an_unreadable_settings_store_is_an_off_store(monkeypatch):
    """Fail SAFE, not open: if we cannot prove an operator turned it on, it is
    off. The opposite posture would make a corrupt settings file enable a
    routing feature."""
    def _boom(*a, **kw):
        raise RuntimeError("settings unavailable")
    monkeypatch.setattr(settings_store, "get", _boom)
    assert MG.enabled_state() == (False, "default")


def test_a_dict_valued_flag_is_tolerated():
    """A hand-written {"value": true} (the shape the settings ROUTE takes) must
    not read as truthy-dict-therefore-on-forever."""
    settings_store.set(MG.SETTINGS_NS, MG.ENABLED_KEY, {"value": False})
    settings_store._cache_at = 0.0
    assert MG.is_enabled() is False
    settings_store.set(MG.SETTINGS_NS, MG.ENABLED_KEY, {"value": True})
    settings_store._cache_at = 0.0
    assert MG.is_enabled() is True


# ---------------------------------------------------------------------------
# The seam is TOTAL — a policy bug never breaks a request
# ---------------------------------------------------------------------------
def test_a_raising_pipeline_degrades_to_routing_the_requested_key(monkeypatch):
    _enable(True)
    monkeypatch.setattr(MG, "_member_for_model",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert MG.member_for_model("Qwen2.5-7B-Instruct-GGUF") is None


def test_a_broken_catalog_yields_no_groups(monkeypatch):
    _enable(True)
    monkeypatch.setattr(MG, "_catalog",
                        lambda: (_ for _ in ()).throw(RuntimeError("no catalog")))
    assert MG.registry() == {}
    assert MG.member_for_model("Qwen2.5-7B-Instruct-GGUF") is None


def test_single_member_untricked_group_short_circuits(monkeypatch):
    """Nothing to decide => no pipeline, no telemetry. A feed that says 'we
    chose the only option' on every request is a feed nobody reads."""
    _enable(True)
    monkeypatch.setattr(MG, "_catalog", lambda: {
        "Solo-GGUF": {"framework": "gguf", "hub_id": "org/Solo-GGUF"}})
    called = []
    monkeypatch.setattr(MG, "_boxes_for", lambda *a, **kw: called.append(a) or [])
    assert MG.member_for_model("Solo-GGUF") is None
    assert called == [], "expanded a group that had nothing to choose between"


def test_choosing_the_key_already_named_reports_no_change(monkeypatch):
    """Returning the SAME key would make the caller's `_mk != self.model_key`
    guard the only thing preventing a pointless reassignment. Report None."""
    _enable(True)
    monkeypatch.setattr(MG, "_catalog", lambda: {
        "A-GGUF": {"framework": "gguf", "hub_id": "org/Thing-GGUF"},
        "B": {"framework": "transformers", "hub_id": "org/Thing"}})
    monkeypatch.setattr(MG, "_physical", lambda k: {"size_bytes": 1000})
    monkeypatch.setattr(MG, "_boxes_for", lambda mk, *a, **kw: [{
        "id": "w1", "name": "w1", "gpus": [{"memory_total": 1 << 34}],
        "vram_total": 1 << 34, "vram_free": 1 << 34,
        "ram_total": 1 << 34, "free_ram": 1 << 34}])
    # A-GGUF wins (a GGUF member outranks a fixed repo), so asking FOR it is a
    # no-change answer while asking for B is a real swap.
    assert MG.member_for_model("A-GGUF") is None
    assert MG.member_for_model("B") == "A-GGUF"


# ---------------------------------------------------------------------------
# GET /llm/groups reads regardless of the switch
# ---------------------------------------------------------------------------
def test_describe_works_and_reports_the_switch_when_off(monkeypatch):
    monkeypatch.setattr(MG, "_catalog", lambda: {
        "A-GGUF": {"framework": "gguf", "hub_id": "org/Thing-GGUF"},
        "B": {"framework": "transformers", "hub_id": "org/Thing"}})
    monkeypatch.setattr(MG, "_physical", lambda k: {"size_bytes": 1000})
    monkeypatch.setattr(MG, "_boxes_for", lambda *a, **kw: [])
    d = MG.describe_groups()
    assert d["enabled"] is False and d["source"] == "default"
    g = next(x for x in d["groups"] if x["group_key"] == "thing")
    assert {m["model_key"] for m in g["members"]} == {"A-GGUF", "B"}
    assert g["ticks"] == {"quality": False, "speed": False, "priority": False}
