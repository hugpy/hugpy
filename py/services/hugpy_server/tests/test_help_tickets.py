"""Help tickets + the help model group (operator 2026-10-02)."""
from __future__ import annotations

import importlib

import pytest
from flask import Flask

ha = importlib.import_module("hugpy_server.app.help_agent")
ht = importlib.import_module("hugpy_server.app.help_tickets")
hr = importlib.import_module("hugpy_server.app.routes.help_routes")


# ── the help model group: a member that errors falls through ────────────────
def test_local_backend_walks_the_group_on_error(monkeypatch):
    b = ha.LocalModelBackend(base="http://x")
    monkeypatch.delenv("HUGPY_HELP_LOCAL_MODEL", raising=False)
    monkeypatch.setattr(b, "models", lambda: ["coder-next", "flux2", "qwen-9b", "coder-3b"])
    seen = []

    def one(model, messages, timeout):
        seen.append(model)
        if model == "coder-next":
            raise RuntimeError("worker refused")
        if model == "flux2":
            return ""
        return f"answer from {model}"
    monkeypatch.setattr(b, "_complete_one", one)
    text, model, failed = b.complete([{"role": "user", "content": "hi"}])
    assert model == "qwen-9b" and text == "answer from qwen-9b"
    assert [f["model"] for f in failed] == ["coder-next", "flux2"] and seen == ["coder-next", "flux2", "qwen-9b"]


def test_local_backend_all_failed_raises(monkeypatch):
    b = ha.LocalModelBackend(base="http://x")
    monkeypatch.setattr(b, "models", lambda: ["a", "b"])
    monkeypatch.setattr(b, "_complete_one", lambda m, msgs, t: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(ha.BackendError):
        b.complete([])


def test_group_designation_and_explicit_override(monkeypatch):
    import hugpy_fleet.central.priority_groups as pg
    monkeypatch.delenv("HUGPY_HELP_LOCAL_MODEL", raising=False)
    monkeypatch.setattr(pg, "get_group", lambda gid: {"id": gid, "members": ["m1", "m2"], "enabled": False})
    monkeypatch.setattr(pg, "expand_members", lambda g, groups=None: [(m, None) for m in g["members"]])
    assert ha.LocalModelBackend(base="http://x").models() == ["m1", "m2"]
    monkeypatch.setenv("HUGPY_HELP_LOCAL_MODEL", "pinned")
    assert ha.LocalModelBackend(base="http://x").models() == ["pinned"]


# ── tickets ─────────────────────────────────────────────────────────────────
def test_only_non_agree_calibrations_file_tickets(monkeypatch):
    filed = []
    monkeypatch.setattr(ht, "file_ticket", lambda kind, title, **kw: filed.append((kind, title, kw)) or {"id": 1})
    assert ht.calibration_ticket({"verdict": "agree"}) is None
    ht.calibration_ticket({"verdict": "disagree", "model_key": "M", "worker_id": "w", "worker_name": "ae",
                           "bnb": True, "detail": {"gate_error_pct": 33.4}})
    kind, title, kw = filed[0]
    assert kind == "calibration" and "+33.4%" in title and kw["detail"]["bnb"] is True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(hr, "_gate", lambda: None)
    monkeypatch.setattr(hr, "_who", lambda: "op")
    store = {1: {"id": 1, "kind": "calibration", "status": "pending", "title": "calibration disagree: M on ae",
                 "model_key": "M", "worker_id": "w", "worker_name": "ae", "detail": {"bnb": True}}}
    monkeypatch.setattr(ht, "get_ticket", lambda tid: store.get(tid))
    monkeypatch.setattr(ht, "set_status", lambda tid, st, res=None: dict(store[tid], status=st, result=res))
    app = Flask(__name__)
    app.register_blueprint(hr.help_bp)
    return app.test_client()


def test_act_calibrate_reruns_with_the_ticket_flags(client, monkeypatch):
    from hugpy_server.app import calibration_run
    from hugpy_server.app.routes import worker_routes
    calls = []
    monkeypatch.setattr(worker_routes, "get_worker", lambda wid: {"id": wid, "name": "ae"})
    monkeypatch.setattr(calibration_run, "start",
                        lambda w, mk, bnb=False, evict_others=False: calls.append((mk, bnb)) or {"job_id": "cal-1"})
    r = client.post("/llm/help/tickets/1/act", json={"action": "calibrate"})
    assert r.status_code == 200 and calls == [("M", True)]
    assert r.get_json()["ticket"]["status"] == "acted"


def test_act_keeper_only_when_available(client, monkeypatch):
    from hugpy_server.app.routes import keeper_help_routes as kh
    monkeypatch.setattr(kh, "file_keeper_report", lambda *a, **k: (None, None))
    assert client.post("/llm/help/tickets/1/act", json={"action": "keeper"}).status_code == 409
    monkeypatch.setattr(kh, "file_keeper_report", lambda *a, **k: ({"id": "m9"}, {"id": "b1"}))
    r = client.post("/llm/help/tickets/1/act", json={"action": "keeper"})
    assert r.status_code == 200 and r.get_json()["ticket"]["status"] == "sent"


def test_act_dismiss_and_unknown(client):
    assert client.post("/llm/help/tickets/1/act", json={"action": "dismiss"}).get_json()["status"] == "dismissed"
    assert client.post("/llm/help/tickets/1/act", json={"action": "explode"}).status_code == 400
    assert client.post("/llm/help/tickets/9/act", json={"action": "dismiss"}).status_code == 404
