"""Model notes + flags (operator 2026-10-02)."""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

mn = importlib.import_module("hugpy_server.app.model_notes")
oa = importlib.import_module("hugpy_server.app.operator_auth")


def test_validate_flags():
    assert mn.validate(["Broken", "trash", "broken"], "x")[0] == ["broken", "trash"]
    with pytest.raises(ValueError):
        mn.validate(["exploded"], "")
    assert len(mn.validate([], "y" * 9000)[1]) == mn.MAX_NOTE


def test_routes(monkeypatch):
    import hugpy_engine.model_index as mi
    from hugpy_server.app.routes import llm_storage_routes as ls
    store = {}
    monkeypatch.setattr(mi, "resolve_model_id", lambda key: 7 if key == "M" else None)
    monkeypatch.setattr(mn, "get", lambda mid: store.get(mid, {"model_id": mid, "flags": [], "note": ""}))
    monkeypatch.setattr(mn, "put", lambda mid, key, flags, note, by, by_kind: store.setdefault(
        mid, {"model_id": mid, "flags": mn.validate(flags, note)[0], "note": note, "by_kind": by_kind}))
    app = Flask(__name__)
    app.register_blueprint(ls.llm_bp)
    c = app.test_client()
    r = c.post("/models/database/M/notes", json={"flags": ["broken"], "note": "no weights", "by_kind": "agent"})
    assert r.status_code == 200 and r.get_json()["flags"] == ["broken"] and r.get_json()["by_kind"] == "agent"
    assert c.get("/models/database/M/notes").get_json()["note"] == "no weights"
    assert c.get("/models/database/X/notes").status_code == 404


def test_notes_post_is_operator_gated():
    assert any("POST" in m and rx.match("/models/database/M/notes") for m, rx in oa._SENSITIVE)
