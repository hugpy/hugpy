"""PRIORITY GROUP → WORKER ALLOCATION (2026-08-25).

The explicit priority-group record grows an ordered ``workers`` list: the
group's allocation, in per-worker priority order. It feeds
``placement_policy`` ONLY when the model carries no per-model ``worker_prefs``
(the more specific statement outranks it), and an absent/empty list is the
byte-identical no-op — the compatibility bar every group feature here is held
to (see test_model_groups_offpath.py for the doctrine).

    ./venv/bin/pytest tests/test_priority_group_workers.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PG = importlib.import_module("hugpy_fleet.central.priority_groups")
OV = importlib.import_module("hugpy_engine.serve.overrides")
SETTINGS = importlib.import_module("hugpy_control.settings")


@pytest.fixture(autouse=True)
def _fleet_priority_groups_seam():
    """Engine placement reads groups through hugpy_engine.placement; wire fleet's provider."""
    from hugpy_engine import placement as seam
    from hugpy_fleet.central.placement import FleetPriorityGroups
    seam.set_priority_groups(FleetPriorityGroups())
    yield
    seam.set_priority_groups(None)


@pytest.fixture()
def isolated_stores(tmp_path, monkeypatch):
    """Point the settings store and serve-overrides file at tmp files, and
    drop the settings read-cache so the redirect takes immediately."""
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    SETTINGS.settings_store._cache = None
    SETTINGS.settings_store._cache_at = 0.0
    monkeypatch.setattr(OV, "_OVERRIDES_PATH",
                        str(tmp_path / "serve_overrides.json"))
    yield tmp_path
    SETTINGS.settings_store._cache = None
    SETTINGS.settings_store._cache_at = 0.0


# ---------------------------------------------------------------------------
# Store: the workers field on the record
# ---------------------------------------------------------------------------
def test_put_group_roundtrips_ordered_workers(isolated_stores):
    rec, errors = PG.put_group(
        "vision", name="vision", members=["Qwen2.5-VL-7B-Instruct-GGUF"],
        workers=["ae", "computron"])
    assert errors == []
    assert rec["workers"] == ["ae", "computron"]
    assert PG.get_group("vision")["workers"] == ["ae", "computron"]


def test_workers_field_is_optional_and_defaults_empty(isolated_stores):
    rec, errors = PG.put_group("plain", name="plain", members=["m1"])
    assert errors == []
    assert rec["workers"] == []


def test_duplicate_worker_is_an_error_not_a_silent_dedup(isolated_stores):
    rec, errors = PG.put_group(
        "dup", name="dup", members=["m1"], workers=["ae", "AE"])
    assert rec is None
    assert any("duplicate worker" in e for e in errors)


def test_workers_must_be_a_list(isolated_stores):
    rec, errors = PG.put_group(
        "bad", name="bad", members=["m1"], workers="ae")
    assert rec is None
    assert any("workers must be a list" in e for e in errors)


def test_patch_semantics_replace_never_merge(isolated_stores):
    PG.put_group("g", name="g", members=["m1"], workers=["ae", "computron"])
    rec, errors = PG.put_group("g", name="g", members=["m1"], workers=["op"])
    assert errors == []
    assert rec["workers"] == ["op"]


# ---------------------------------------------------------------------------
# workers_for_key: the placement read
# ---------------------------------------------------------------------------
def test_workers_for_key_matches_alias_tolerantly(isolated_stores):
    PG.put_group("vl", name="vl", members=["Qwen~Qwen2.5-VL-7B-Instruct"],
                 workers=["ae"])
    # The ~-tail spelling must hit the same group (workers._match_keys rule).
    assert PG.workers_for_key("Qwen2.5-VL-7B-Instruct") == ["ae"]


def test_disabled_group_allocates_nothing(isolated_stores):
    PG.put_group("off", name="off", members=["m1"], workers=["ae"],
                 enabled=False)
    assert PG.workers_for_key("m1") == []


def test_ungrouped_key_allocates_nothing(isolated_stores):
    assert PG.workers_for_key("nowhere-model") == []


# ---------------------------------------------------------------------------
# placement_policy: precedence and the no-op path
# ---------------------------------------------------------------------------
def test_group_workers_feed_placement_when_model_has_no_prefs(isolated_stores):
    PG.put_group("g", name="g", members=["m1"], workers=["computron", "ae"])
    prefs, polite, by_worker = OV.placement_policy("m1")
    assert prefs == ["computron", "ae"]
    assert polite is False and by_worker == {}


def test_per_model_prefs_outrank_the_group(isolated_stores):
    PG.put_group("g", name="g", members=["m1"], workers=["computron", "ae"])
    OV.set_override("m1", {"worker_prefs": ["op"]})
    prefs, _, _ = OV.placement_policy("m1")
    assert prefs == ["op"]


def test_no_group_no_prefs_is_byte_identical_empty(isolated_stores):
    assert OV.placement_policy("m1") == ([], False, {})


def test_group_never_supplies_politeness(isolated_stores):
    PG.put_group("g", name="g", members=["m1"], workers=["ae"])
    prefs, polite, by_worker = OV.placement_policy("m1")
    assert prefs == ["ae"]
    assert polite is False and by_worker == {}


# ---------------------------------------------------------------------------
# move_member: the model-table Group column gesture
# ---------------------------------------------------------------------------
def test_move_member_between_groups_appends_last(isolated_stores):
    PG.put_group("a", name="a", members=["m1", "m2"])
    PG.put_group("b", name="b", members=["m3"])
    changed, errors = PG.move_member("m1", "b")
    assert errors == []
    assert PG.get_group("a")["members"] == ["m2"]
    assert PG.get_group("b")["members"] == ["m3", "m1"]


def test_move_member_out_of_all_groups(isolated_stores):
    PG.put_group("a", name="a", members=["m1", "m2"])
    changed, errors = PG.move_member("m1", None)
    assert errors == []
    assert PG.get_group("a")["members"] == ["m2"]


def test_move_member_refuses_to_empty_a_group(isolated_stores):
    PG.put_group("a", name="a", members=["m1"])
    changed, errors = PG.move_member("m1", None)
    assert any("delete the group instead" in e for e in errors)
    assert PG.get_group("a")["members"] == ["m1"]


def test_move_member_into_same_group_is_a_noop(isolated_stores):
    PG.put_group("a", name="a", members=["m1"], workers=["ae"])
    changed, errors = PG.move_member("m1", "a")
    assert errors == [] and changed == []
    assert PG.get_group("a")["members"] == ["m1"]


def test_move_member_preserves_workers(isolated_stores):
    PG.put_group("a", name="a", members=["m1", "m2"], workers=["ae"])
    PG.put_group("b", name="b", members=["m3"], workers=["computron"])
    PG.move_member("m1", "b")
    assert PG.get_group("a")["workers"] == ["ae"]
    assert PG.get_group("b")["workers"] == ["computron"]


def test_move_member_unknown_group(isolated_stores):
    changed, errors = PG.move_member("m1", "nope")
    assert changed == [] and any("no such priority group" in e for e in errors)


# ---------------------------------------------------------------------------
# Nesting: group:<id> members — modules inside a group
# ---------------------------------------------------------------------------
def test_nested_group_expands_in_place_with_provenance(isolated_stores):
    PG.put_group("vl", name="vl", members=["vl-tf", "vl-gguf"])
    PG.put_group("parent", name="parent",
                 members=["m0", "group:vl", "m9"], workers=["ae"])
    walk = PG.expand_members(PG.get_group("parent"))
    assert walk == [("m0", None), ("vl-tf", "vl"), ("vl-gguf", "vl"),
                    ("m9", None)]


def test_nested_expansion_is_cycle_safe(isolated_stores):
    PG.put_group("a", name="a", members=["m1", "group:b"])
    PG.put_group("b", name="b", members=["m2", "group:a"])
    walk = PG.expand_members(PG.get_group("a"))
    assert walk == [("m1", None), ("m2", "b")]


def test_self_reference_is_rejected(isolated_stores):
    rec, errors = PG.put_group("s", name="s", members=["m1", "group:s"])
    assert rec is None and any("cannot contain itself" in e for e in errors)


def test_dangling_group_ref_is_skipped(isolated_stores):
    PG.put_group("p", name="p", members=["m1", "group:ghost"])
    assert PG.expand_members(PG.get_group("p")) == [("m1", None)]


def test_module_inherits_parent_workers(isolated_stores):
    PG.put_group("vl", name="vl", members=["vl-tf"])
    PG.put_group("parent", name="parent", members=["group:vl"],
                 workers=["ae", "computron"])
    # vl has no workers of its own -> the parent's allocation governs.
    assert PG.effective_workers(PG.get_group("vl")) == ["ae", "computron"]
    assert PG.workers_for_key("vl-tf") == ["ae", "computron"]
    # ...and placement sees the inherited order too.
    assert OV.placement_policy("vl-tf")[0] == ["ae", "computron"]


def test_module_own_workers_beat_inherited(isolated_stores):
    PG.put_group("vl", name="vl", members=["vl-tf"], workers=["op"])
    PG.put_group("parent", name="parent", members=["group:vl"], workers=["ae"])
    assert PG.effective_workers(PG.get_group("vl")) == ["op"]
    assert PG.workers_for_key("vl-tf") == ["op"]


def test_disabled_parent_does_not_lend_workers(isolated_stores):
    PG.put_group("vl", name="vl", members=["vl-tf"])
    PG.put_group("parent", name="parent", members=["group:vl"],
                 workers=["ae"], enabled=False)
    assert PG.effective_workers(PG.get_group("vl")) == []


def test_group_ref_exclusivity_between_enabled_parents(isolated_stores):
    PG.put_group("vl", name="vl", members=["vl-tf"])
    PG.put_group("p1", name="p1", members=["group:vl", "x1"])
    rec, errors = PG.put_group("p2", name="p2", members=["group:vl", "x2"])
    assert rec is None
    assert any("at most one enabled group" in e for e in errors)
