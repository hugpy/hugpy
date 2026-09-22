"""TASK TEMPLATES (2026-08-26) — blueprint records over groups + workers.

Operator concept: "Templates => worker groups && => module groups" — map what
groups a task needs, reserve the pool for uninterrupted execution. These tests
cover the record layer and derivations; activation orchestration is
route-layer (worker store) and exercised on dev.

    ./venv/bin/pytest tests/test_task_templates.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PG = importlib.import_module("hugpy_fleet.central.priority_groups")
TT = importlib.import_module("hugpy_fleet.central.task_templates")
SETTINGS = importlib.import_module("hugpy_control.settings")


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    SETTINGS.settings_store._cache = None
    SETTINGS.settings_store._cache_at = 0.0
    yield tmp_path
    SETTINGS.settings_store._cache = None
    SETTINGS.settings_store._cache_at = 0.0


def test_template_roundtrip(isolated):
    PG.put_group("cast", name="cast", members=["m1"])
    rec, errors = TT.put_template("night-shoot", name="night-shoot",
                                  groups=["cast"], workers=["ae"])
    assert errors == []
    assert TT.get_template("night-shoot")["groups"] == ["cast"]
    assert TT.get_template("night-shoot")["workers"] == ["ae"]
    assert TT.get_template("night-shoot")["active"] is False


def test_template_refuses_dangling_group(isolated):
    rec, errors = TT.put_template("t", name="t", groups=["ghost"])
    assert rec is None
    assert any("no such priority group" in e for e in errors)


def test_template_requires_a_task_or_group(isolated):
    # Task-oriented templates (operator 2026-08-28): a task outline alone is a
    # valid template; groups became the legacy/reservation half. Empty both
    # ways is still refused.
    rec, errors = TT.put_template("t", name="t", groups=[], tasks=[])
    assert rec is None
    assert any("at least one task" in e for e in errors)


def test_tasks_only_template_is_valid(isolated):
    rec, errors = TT.put_template(
        "t", name="t",
        tasks=[{"name": "script", "desc": "author the screenplay"}])
    assert errors == []
    assert rec["tasks"] == [
        {"name": "script", "desc": "author the screenplay", "model": None}]


def test_task_outline_validates(isolated):
    rec, errors = TT.put_template(
        "t", name="t",
        tasks=[{"name": "a"}, {"name": "a"}, {"desc": "nameless"},
               {"name": "b", "model": 7}])
    assert rec is None
    assert any("duplicate task" in e for e in errors)
    assert any("needs a name" in e for e in errors)
    assert any("model must be" in e for e in errors)


def test_set_task_model_fills_and_clears_one_slot(isolated):
    TT.put_template("t", name="t", tasks=[
        {"name": "script", "desc": "author"},
        {"name": "render", "desc": "synthesize"}])
    rec, err = TT.set_task_model("t", "SCRIPT", "Qwen2.5-7B-Instruct-GGUF")
    assert err == ""
    assert rec["tasks"][0]["model"] == "Qwen2.5-7B-Instruct-GGUF"
    assert rec["tasks"][1]["model"] is None       # untouched
    rec, err = TT.set_task_model("t", "script", None)
    assert err == "" and rec["tasks"][0]["model"] is None
    _, err = TT.set_task_model("t", "missing", "m")
    assert "no task" in err


def test_groups_only_write_preserves_task_outline(isolated):
    PG.put_group("g", name="g", members=["m1"], workers=["computron"])
    TT.put_template("t", name="t",
                    tasks=[{"name": "script", "model": "m1"}])
    # a legacy write that doesn't mention tasks must not wipe the outline
    rec, errors = TT.put_template("t", name="t", groups=["g"])
    assert errors == []
    assert rec["tasks"][0] == {"name": "script", "desc": "", "model": "m1"}


def test_derive_workers_prefers_own_list(isolated):
    PG.put_group("g", name="g", members=["m1"], workers=["computron"])
    TT.put_template("t", name="t", groups=["g"], workers=["ae"])
    assert TT.derive_workers(TT.get_template("t")) == ["ae"]


def test_derive_workers_unions_group_fences_in_order(isolated):
    PG.put_group("g1", name="g1", members=["m1"], workers=["ae", "computron"])
    PG.put_group("g2", name="g2", members=["m2"], workers=["op", "ae"])
    TT.put_template("t", name="t", groups=["g1", "g2"])
    assert TT.derive_workers(TT.get_template("t")) == ["ae", "computron", "op"]


def test_derive_workers_follows_inheritance(isolated):
    PG.put_group("child", name="child", members=["m1"])
    PG.put_group("parent", name="parent", members=["group:child"],
                 workers=["ae"])
    TT.put_template("t", name="t", groups=["child"])
    # child has no workers of its own; its effective (inherited) fence governs.
    assert TT.derive_workers(TT.get_template("t")) == ["ae"]


def test_mark_active_flips_flag_only(isolated):
    PG.put_group("g", name="g", members=["m1"])
    TT.put_template("t", name="t", groups=["g"])
    rec = TT.mark_active("t", True)
    assert rec["active"] is True and rec["activated_at"] is not None
    rec = TT.mark_active("t", False)
    assert rec["active"] is False


def test_put_preserves_active_state(isolated):
    PG.put_group("g", name="g", members=["m1"])
    TT.put_template("t", name="t", groups=["g"])
    TT.mark_active("t", True)
    rec, errors = TT.put_template("t", name="t", groups=["g"], workers=["ae"])
    assert errors == [] and rec["active"] is True
