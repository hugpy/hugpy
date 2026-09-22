"""k10 hardening: the sanctioned ghost-cleanup route for the assignment-memory
sidecar (``worker_assignments.json``) — ``DELETE /llm/workers/<id>/memory``.

Deleting a live worker ROW deliberately does NOT delete its assignment memory
(designations are worker-lifetime, not row-lifetime, so a re-register under
a known id restores them). That design stays intact: the route only removes
entries whose id is NOT live in workers.json — 200 for a ghost, 404 for an
id never remembered, 409 for a live id — and it is operator-gated in
``operator_auth._SENSITIVE``, same tier as the plain worker DELETE.

The store-level ``forget_assignment_memory`` contract lives in
``py/fleet/hugpy_fleet/tests/test_assignment_memory_forget.py``.

Uses ``swap_worker_store`` so this never touches a live registry or its
sidecar — both are redirected into a fresh tmpdir for the swap's duration.
"""

from __future__ import annotations

import importlib

import pytest
from flask import Flask

from worker_store_isolation import swap_worker_store

W = importlib.import_module("hugpy_fleet.central.workers")
wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")


@pytest.fixture
def client():
    """Bare worker blueprint over an isolated registry seeded with a LIVE
    worker (``live-two``) and a GHOST (``ghost-two``: memory entry survives
    the row removal, by design)."""
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-memory-forget-route-test-"):
        W.worker_store.register(name="live-two", url="http://192.0.2.72:9100", worker_id="live-two")
        W.worker_store.assign_model("live-two", "Some~Model")
        W.worker_store.register(name="ghost-two", url="http://192.0.2.73:9100", worker_id="ghost-two")
        W.worker_store.assign_model("ghost-two", "Ghost~Model")
        W.worker_store.remove("ghost-two")
        assert "ghost-two" in W._load_assign_memory()
        assert W.worker_store.get("ghost-two") is None
        yield app.test_client()


def test_delete_memory_route_status_codes(client):
    r = client.delete("/llm/workers/live-two/memory")
    assert r.status_code == 409
    assert "live-two" in W._load_assign_memory()

    assert client.delete("/llm/workers/totally-unknown-id/memory").status_code == 404

    r = client.delete("/llm/workers/ghost-two/memory")
    assert r.status_code == 200
    assert (r.get_json() or {}).get("forgot") == "ghost-two"
    assert "ghost-two" not in W._load_assign_memory()
    assert "live-two" in W._load_assign_memory()

    # repeating on an already-forgotten id is 404 now (idempotent, honest)
    assert client.delete("/llm/workers/ghost-two/memory").status_code == 404


def _gated(path, method):
    if path == "/api" or path.startswith("/api/"):
        path = path[len("/api"):] or "/"
    return any(method in methods and rx.match(path) for methods, rx in oa._SENSITIVE)


def test_memory_route_is_in_sensitive_tier():
    assert _gated("/llm/workers/some-id/memory", "DELETE")
    assert _gated("/api/llm/workers/some-id/memory", "DELETE")
    assert _gated("/llm/workers/some-id", "DELETE")          # plain row DELETE unaffected
    assert _gated("/llm/workers/some-id", "GET") is False    # reads stay open


def test_memory_route_refuses_anonymous_through_the_gate(monkeypatch):
    """End-to-end: with the gate installed and a token configured, the
    anonymous DELETE is refused (401) before it reaches the sidecar."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    oa.install_operator_gate(app)
    app.config["TESTING"] = True
    with swap_worker_store(prefix="hugpy-memory-forget-gate-test-"):
        W.worker_store.register(name="ghost", url="http://192.0.2.74:9100", worker_id="ghost")
        W.worker_store.assign_model("ghost", "Ghost~Model")
        W.worker_store.remove("ghost")
        c = app.test_client()
        assert c.delete("/llm/workers/ghost/memory").status_code == 401
        assert "ghost" in W._load_assign_memory()
        r = c.delete("/llm/workers/ghost/memory", headers={"X-Operator-Token": "s3cret"})
        assert r.status_code == 200
        assert "ghost" not in W._load_assign_memory()
