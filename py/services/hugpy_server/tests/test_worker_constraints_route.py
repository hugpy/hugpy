"""WP5: ``GET /llm/workers/constraints.txt`` — the lockstep pin set as text.

Public and unauthenticated like ``required-version``: one ``name==<required>``
line per workspace distribution when central pins a version, **204** when it
does not. Register/heartbeat replies carry ``constraints_url`` next to
``required_pkg_version`` so a worker need not derive it.
"""
from __future__ import annotations

import sys

import pytest

from hugpy_fleet.central import workers as W
from hugpy_server.app.routes import worker_routes as wr

URL = "/api/llm/workers/constraints.txt"


@pytest.fixture(autouse=True)
def _no_buildinfo(monkeypatch):
    """Pin the distribution list to the fleet fallback so the body is exact
    whether or not the installed hugpy-platform ships ``buildinfo`` yet."""
    monkeypatch.setitem(sys.modules, "hugpy_platform.buildinfo", None)


def test_constraints_route_lists_every_workspace_distribution(client, monkeypatch):
    monkeypatch.setattr(W, "required_pkg_version", lambda: "0.2.0")
    resp = client.get(URL)
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    body = resp.get_data(as_text=True)
    lines = body.splitlines()
    assert body.endswith("\n") and len(lines) == 13
    assert lines == [f"{n}==0.2.0" for n in W.WORKSPACE_DISTRIBUTIONS_FALLBACK]
    assert "hugpy-fleet==0.2.0" in lines and "hugpy-platform==0.2.0" in lines


def test_constraints_route_204_when_central_pins_no_version(client, monkeypatch):
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    resp = client.get(URL)
    assert resp.status_code == 204
    assert resp.get_data() == b""


def test_constraints_route_is_open_like_required_version(client, monkeypatch):
    """No token, no session — the bootstrap fetches it before a worker exists."""
    monkeypatch.setattr(W, "required_pkg_version", lambda: "0.2.0")
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    assert client.get("/api/llm/workers/required-version").status_code == 200
    assert client.get(URL).status_code == 200


def test_reply_constraints_url_is_proxy_aware(server_app):
    with server_app.test_request_context(
            "/api/llm/workers/register", base_url="http://127.0.0.1:7002",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "dev.hugpy.ai"}):
        assert wr._reply_constraints_url() == \
            "https://dev.hugpy.ai/api/llm/workers/constraints.txt"
    with server_app.test_request_context("/api/llm/workers/register",
                                         base_url="http://10.0.0.5:7002"):
        assert wr._reply_constraints_url() == \
            "http://10.0.0.5:7002/api/llm/workers/constraints.txt"


def test_register_reply_carries_constraints_url(client, monkeypatch):
    monkeypatch.setattr(wr, "required_pkg_version", lambda: "0.2.0")
    monkeypatch.setattr(wr, "_enrollment_ok", lambda: True)
    resp = client.post("/api/llm/workers/register",
                       json={"name": "wp5-test-box", "url": "http://10.0.0.9:9100"},
                       base_url="https://central.test")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    worker = resp.get_json()
    assert worker["required_pkg_version"] == "0.2.0"
    assert worker["constraints_url"] == "https://central.test/api/llm/workers/constraints.txt"
