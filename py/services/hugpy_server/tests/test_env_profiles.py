"""Per-model environments — store validation, routes, approval (2026-10-02)."""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

ep = importlib.import_module("hugpy_server.app.env_profiles")
er = importlib.import_module("hugpy_server.app.routes.env_profile_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")


def test_validate():
    assert ep.validate("ct-015", ["compressed-tensors>=0.15.0", "transformers==4.57.1"], "worker") == (
        "ct-015", ["compressed-tensors>=0.15.0", "transformers==4.57.1"], "worker")
    for bad in (("../x", [], "worker"), ("ok", ["-e git+https://x"], "worker"),
                ("ok", ["pkg; rm -rf /"], "worker"), ("ok", ["x"], "conda")):
        with pytest.raises(ValueError):
            ep.validate(*bad)


@pytest.fixture
def client(monkeypatch):
    store = {}

    def put(name, packages, base="worker", note="", *, by, by_kind="operator"):
        name, pkgs, base = ep.validate(name, packages, base)
        store[name] = {"name": name, "packages": pkgs, "base": base, "note": note,
                       "status": "approved" if by_kind == "operator" else "proposed", "created_by": by}
        return store[name]
    monkeypatch.setattr(ep, "put", put)
    monkeypatch.setattr(ep, "get", lambda n: store.get(n))
    monkeypatch.setattr(ep, "approve", lambda n, by: store[n].update(status="approved") or store[n] if n in store else None)
    monkeypatch.setattr(ep, "list_all", lambda: list(store.values()))
    tickets = []
    import hugpy_server.app.help_tickets as ht
    monkeypatch.setattr(ht, "file_ticket", lambda kind, title, **kw: tickets.append((kind, title, kw)) or {"id": 1})
    app = Flask(__name__)
    app.register_blueprint(er.env_profile_bp)
    return app.test_client(), store, tickets


def test_operator_profile_is_approved_agent_profile_proposed_with_a_ticket(client):
    c, store, tickets = client
    assert c.post("/llm/env-profiles", json={"name": "a", "packages": ["x==1"]}).get_json()["status"] == "approved"
    assert not tickets
    r = c.post("/llm/env-profiles", json={"name": "b", "packages": ["compressed-tensors>=0.15.0"], "by_kind": "agent"})
    assert r.get_json()["status"] == "proposed" and tickets[0][0] == "env-profile"
    assert c.post("/llm/env-profiles", json={"name": "bad name"}).status_code == 400
    assert c.post("/llm/env-profiles/b/approve").get_json()["status"] == "approved"
    assert len(c.get("/llm/env-profiles").get_json()["profiles"]) == 2


def test_test_requires_approval_and_args(client, monkeypatch):
    c, store, _ = client
    c.post("/llm/env-profiles", json={"name": "p", "packages": [], "by_kind": "agent"})
    assert c.post("/llm/env-profiles/p/test", json={"model": "M", "worker": "w"}).status_code == 409
    assert c.post("/llm/env-profiles/p/test", json={}).status_code == 400
    monkeypatch.setattr(er.threading, "Thread", lambda **kw: type("T", (), {"start": lambda s: None})())
    store["p"]["status"] = "approved"
    assert c.post("/llm/env-profiles/p/test", json={"model": "M", "worker": "w"}).status_code == 202


def test_writes_are_operator_gated():
    gated = lambda p: any("POST" in m and rx.match(p) for m, rx in oa._SENSITIVE)
    assert gated("/llm/env-profiles") and gated("/llm/env-profiles/x/approve") and gated("/llm/env-profiles/x/test")
