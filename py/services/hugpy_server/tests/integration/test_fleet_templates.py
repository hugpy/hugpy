"""Fleet Configuration Templates — Slice 0 (storage + read-only + snapshot + diff).

Covers FLEET-TEMPLATES-DESIGN.md §1/§2/§3a/§3b/§6:
  * schema validation — a valid minimal + full doc, and several malformed docs;
  * CRUD roundtrip through the settings store + the per-save ``revision`` bump;
  * snapshot from a MOCKED live fleet produces a valid, round-tripping template
    (snapshot -> diff against the same unchanged fleet == EMPTY plan);
  * a diff where desired != live yields the expected ordered plan lines;
  * the REST blueprint wiring + the operator-auth gate (writes gated, GET/diff open).

No live fleet is needed: the worker view is a hand-built fixture. Runs both under
pytest and as a script:
    venv/bin/python -m pytest tests/test_fleet_templates.py -v
    venv/bin/python tests/test_fleet_templates.py
"""
import os
import sys
import copy
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolate every store touch to a throwaway dir (never the real settings.json).
_TMP = tempfile.mkdtemp(prefix="hugpy-fleet-templates-test-")
os.environ["PROJECTS_HOME"] = _TMP
os.environ.setdefault("HUGPY_SETTINGS_PATH", os.path.join(_TMP, "settings.json"))

import importlib

ft = importlib.import_module("hugpy_fleet.fleet_manager.templates")
from hugpy_control.settings import SettingsStore


# --------------------------------------------------------------------------
# a mocked live /llm/workers view (the shape list_workers() returns)
# --------------------------------------------------------------------------
def _fleet():
    return [
        {
            "id": "w-aaa", "name": "studio-box",
            "models": ["wan-vace", "qwen-image"],
            "spill_by_model": {"wan-vace": {"n_gpu_layers": -1}},
            "groups": ["studio"],
            "limits": {"gpu_mem_gib": 22, "disk_cache_gib": 400},
            "config": {
                "slot_count": 2,
                "residency": {"wan-vace": "static"},
                "pinned": {"wan-vace": True},
            },
        },
        {
            "id": "w-bbb", "name": "chat-box",
            "models": ["llama3"],
            "spill_by_model": {},
            "config": {"slot_count": 1, "residency": {}, "pinned": {}},
        },
    ]


def _fresh_store():
    fd, path = tempfile.mkstemp(prefix="fleet-store-", suffix=".json", dir=_TMP)
    os.close(fd)
    os.unlink(path)  # start empty; the store creates it on first write
    return SettingsStore(path=path)


# ==========================================================================
# §1 — schema validation
# ==========================================================================
def test_validate_minimal_doc():
    doc = {"schema_version": 1, "name": "empty"}
    assert ft.validate_template(doc) is doc


def test_validate_full_doc():
    doc = {
        "schema_version": 1,
        "name": "studio-night",
        "description": "4x3090 serving wan vace; chat parked",
        "author": "operator",
        "revision": 3,
        "fleet": {"download_budget_per_worker": 3, "required_pkg_version": "0.1.158"},
        "workers": [{
            "select": {"group": "studio"},
            "slot_count": 2,
            "assignments": [
                {"model": "wan-vace", "serving_mode": "static", "pin": True,
                 "spill": {"n_gpu_layers": -1}},
                {"model": "wan-i2v", "serving_mode": "on_demand"},
            ],
            "absent": ["deepcoder-*", "qwen3-*"],
            "limits": {"gpu_mem_gib": 22, "disk_cache_gib": 400},
            "serving_mode": "on_demand",
        }],
    }
    assert ft.validate_template(doc) is doc


@pytest.mark.parametrize("doc, needle", [
    ({"name": "x"}, "schema_version"),                                  # missing version
    ({"schema_version": 2, "name": "x"}, "schema_version"),             # wrong version
    ({"schema_version": 1}, "name"),                                    # missing name
    ({"schema_version": 1, "name": ""}, "name"),                        # empty name
    ({"schema_version": 1, "name": "x", "bogus": 1}, "unknown field"),  # unknown top key
    ({"schema_version": 1, "name": "x",
      "workers": [{"select": {"id": "a", "name": "b"}}]}, "select"),    # 2-key selector
    ({"schema_version": 1, "name": "x",
      "workers": [{"select": {"id": "a"},
                   "assignments": [{"serving_mode": "static"}]}]}, "model"),  # no model
    ({"schema_version": 1, "name": "x",
      "workers": [{"select": {"id": "a"},
                   "assignments": [{"model": "m", "serving_mode": "warm"}]}]},
     "serving_mode"),                                                   # bad serving_mode
    ({"schema_version": 1, "name": "x",
      "workers": [{"select": {"id": "a"}, "slot_count": 99}]}, "slot_count"),  # range
    ({"schema_version": 1, "name": "x",
      "workers": [{"select": {"id": "a"},
                   "assignments": [{"model": "m", "pin": True}],
                   "absent": ["m"]}]}, "pin"),                          # §8.8 pin+absent
])
def test_validate_rejects_malformed(doc, needle):
    with pytest.raises(ft.TemplateError) as exc:
        ft.validate_template(doc)
    assert needle in str(exc.value)


# ==========================================================================
# §2 — storage CRUD + revision bump
# ==========================================================================
def test_crud_roundtrip_through_store():
    store = _fresh_store()
    doc = {"schema_version": 1, "name": "chat-day",
           "description": "chat profile"}
    stored = ft.save_template(doc, store=store)
    assert stored["name"] == "chat-day"

    got = ft.get_template("chat-day", store=store)
    assert got is not None and got["description"] == "chat profile"

    names = [t["name"] for t in ft.list_templates(store=store)]
    assert names == ["chat-day"]

    assert ft.delete_template("chat-day", store=store) is True
    assert ft.get_template("chat-day", store=store) is None
    assert ft.list_templates(store=store) == []


def test_revision_bumps_per_save():
    store = _fresh_store()
    doc = {"schema_version": 1, "name": "prof", "revision": 99}  # supplied rev ignored
    r1 = ft.save_template(doc, store=store)
    assert r1["revision"] == 1
    r2 = ft.save_template({"schema_version": 1, "name": "prof"}, store=store)
    assert r2["revision"] == 2
    r3 = ft.save_template({"schema_version": 1, "name": "prof"}, store=store)
    assert r3["revision"] == 3
    # created_at is preserved across saves; updated_at is present.
    assert r3["created_at"] == r1["created_at"]
    assert "updated_at" in r3


def test_active_pointer():
    store = _fresh_store()
    assert ft.get_active(store=store) is None
    store.set(ft.NS_FLEET, ft.KEY_ACTIVE,
              {"template": "chat-day", "revision": 1, "applied_at": "t"})
    ptr = ft.get_active(store=store)
    assert ptr and ptr["template"] == "chat-day"


# ==========================================================================
# §3b — snapshot: valid doc + the round-trip invariant
# ==========================================================================
def test_snapshot_produces_valid_doc():
    doc = ft.build_snapshot("cap", "captured", _fleet())
    ft.validate_template(doc)  # raises if invalid
    assert doc["name"] == "cap"
    secs = {s["select"]["id"]: s for s in doc["workers"]}
    a = secs["w-aaa"]
    assert a["slot_count"] == 2
    assert a["limits"] == {"gpu_mem_gib": 22, "disk_cache_gib": 400}
    bymodel = {x["model"]: x for x in a["assignments"]}
    assert bymodel["wan-vace"]["serving_mode"] == "static"
    assert bymodel["wan-vace"]["pin"] is True
    assert bymodel["wan-vace"]["spill"] == {"n_gpu_layers": -1}
    assert bymodel["qwen-image"]["serving_mode"] == "on_demand"
    assert "pin" not in bymodel["qwen-image"]


def test_snapshot_diff_roundtrip_is_empty():
    fleet = _fleet()
    doc = ft.build_snapshot("cap", None, fleet)
    diff = ft.compute_diff(doc, copy.deepcopy(fleet))
    assert diff["empty"] is True, diff
    assert all(not w["plan"] for w in diff["workers"])
    assert diff["unmatched_selectors"] == []


# ==========================================================================
# §3a — diff where desired != live yields the expected ordered plan
# ==========================================================================
def _desired_ne_template():
    return {
        "schema_version": 1, "name": "studio-night", "description": "shift",
        "workers": [
            {"select": {"id": "w-aaa"},
             "slot_count": 4,                                   # 2 -> 4
             "assignments": [
                 {"model": "wan-vace", "serving_mode": "static", "pin": True},  # unchanged
                 {"model": "qwen-image", "serving_mode": "static"},             # residency line
                 {"model": "new-model", "serving_mode": "on_demand"},           # assign line
             ],
             "limits": {"gpu_mem_gib": 20}},                    # 22 -> 20
            {"select": {"id": "w-bbb"},
             "absent": ["llama3"]},                             # unassign line
        ],
    }


def test_diff_desired_ne_live_plan_lines():
    diff = ft.compute_diff(_desired_ne_template(), _fleet())
    assert diff["empty"] is False
    byw = {w["worker_id"]: w["plan"] for w in diff["workers"]}

    aaa = byw["w-aaa"]
    ops = [(l["op"], l.get("field"), l.get("model")) for l in aaa]
    assert ("config", "slot_count", None) in ops
    assert ("limits", "gpu_mem_gib", None) in ops
    assert ("config", "residency", "qwen-image") in ops
    assert ("assign", None, "new-model") in ops
    # wan-vace is unchanged (static==static, pin==pin) -> no line mentions it.
    assert all(l.get("model") != "wan-vace" for l in aaa)

    # concrete from/to values.
    sc = next(l for l in aaa if l["op"] == "config" and l["field"] == "slot_count")
    assert sc["from"] == 2 and sc["to"] == 4
    lim = next(l for l in aaa if l["op"] == "limits")
    assert lim["from"] == 22 and lim["to"] == 20
    res = next(l for l in aaa if l["op"] == "config" and l["field"] == "residency")
    assert res["from"] == "on_demand" and res["to"] == "static"

    # §3c ordering: config knobs precede the assign.
    idx_cfg = min(i for i, l in enumerate(aaa) if l["op"] in ("config", "limits"))
    idx_assign = next(i for i, l in enumerate(aaa) if l["op"] == "assign")
    assert idx_cfg < idx_assign

    bbb = byw["w-bbb"]
    un = [l for l in bbb if l["op"] == "unassign"]
    assert len(un) == 1 and un[0]["model"] == "llama3"
    assert un[0]["blocked_by_pin"] is False
    assert un[0]["destructive"] is True


def test_diff_flags_unmatched_selector():
    tmpl = {"schema_version": 1, "name": "typo",
            "workers": [{"select": {"id": "w-ghost"},
                         "assignments": [{"model": "m"}]}]}
    diff = ft.compute_diff(tmpl, _fleet())
    assert {"kind": "id", "value": "w-ghost"} in diff["unmatched_selectors"]
    assert diff["empty"] is False
    # the ghost selector matched nobody; both live workers are out of scope.
    assert {w["worker_id"] for w in diff["out_of_scope_workers"]} == {"w-aaa", "w-bbb"}


# ==========================================================================
# §6 — REST blueprint wiring + operator-auth gate
# ==========================================================================
def _client_and_auth():
    from flask import Flask
    fr = importlib.import_module(
        "hugpy_server.app.routes.fleet_routes")
    oa = importlib.import_module("hugpy_server.app.operator_auth")
    fr.list_workers = lambda: _fleet()          # no live registry
    app = Flask(__name__)
    app.register_blueprint(fr.fleet_bp)
    return app, fr, oa


def test_route_crud_snapshot_diff():
    app, fr, oa = _client_and_auth()
    c = app.test_client()

    # PUT save -> revision 1
    r = c.put("/fleet/templates/chat-day",
              json={"schema_version": 1, "name": "ignored", "description": "d"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["name"] == "chat-day" and r.get_json()["revision"] == 1

    # GET it back
    assert c.get("/fleet/templates/chat-day").get_json()["description"] == "d"
    # list
    names = [t["name"] for t in c.get("/fleet/templates").get_json()["templates"]]
    assert "chat-day" in names
    # unknown -> 404
    assert c.get("/fleet/templates/nope").status_code == 404
    # active pointer null in Slice 0
    assert c.get("/fleet/active").get_json()["active"] is None

    # snapshot the (mocked) live fleet, then diff it -> empty
    snap = c.post("/fleet/templates/snapshot", json={"name": "cap"})
    assert snap.status_code == 200, snap.get_data(as_text=True)
    d = c.post("/fleet/templates/cap/diff", json={})
    assert d.status_code == 200 and d.get_json()["empty"] is True

    # bad doc -> 400
    assert c.put("/fleet/templates/bad",
                 json={"schema_version": 9, "name": "bad"}).status_code == 400

    # delete
    assert c.delete("/fleet/templates/chat-day").get_json()["deleted"] is True


def test_operator_gate_covers_fleet_writes():
    _, _, oa = _client_and_auth()

    def sensitive(method, path):
        for methods, rx in oa._SENSITIVE:
            if method in methods and rx.match(path):
                return True
        return False

    # writes gated
    assert sensitive("PUT", "/fleet/templates/studio-night")
    assert sensitive("DELETE", "/fleet/templates/studio-night")
    assert sensitive("POST", "/fleet/templates/snapshot")
    # reads + dry-run diff OPEN
    assert not sensitive("GET", "/fleet/templates")
    assert not sensitive("GET", "/fleet/templates/studio-night")
    assert not sensitive("GET", "/fleet/active")
    assert not sensitive("POST", "/fleet/templates/studio-night/diff")


def test_operator_gate_enforced_when_token_set(monkeypatch):
    app, fr, oa = _client_and_auth()
    oa.install_operator_gate(app)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    c = app.test_client()
    body = {"schema_version": 1, "name": "x"}
    # gated write without the token -> 401
    assert c.put("/fleet/templates/x", json=body).status_code == 401
    # with the token -> allowed (200)
    assert c.put("/fleet/templates/x", json=body,
                 headers={"X-Operator-Token": "s3cret"}).status_code == 200
    # read-only diff stays open even without a token
    assert c.post("/fleet/templates/x/diff", json={}).status_code == 200


# --------------------------------------------------------------------------
# script runner (mirrors the other tests here)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
