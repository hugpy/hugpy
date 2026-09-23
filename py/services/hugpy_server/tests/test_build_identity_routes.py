"""Build identity on central (WP2): ``/health`` carries the compact identity,
``/build`` serves the full document, both under ``/api`` too, and neither can
5xx when the identity probe fails."""
from __future__ import annotations

import pytest

from hugpy_platform import buildinfo
from hugpy_server.app.routes import llm_storage_routes as routes

IDENTITY_KEYS = {"version", "sha", "dirty", "editable", "source", "distribution"}
BUILD_KEYS = {"identity", "distributions", "lockstep", "workspace", "python",
              "executable", "hostname", "generated_at"}


@pytest.mark.parametrize("prefix", ["", "/api"])
def test_health_carries_build_identity(client, prefix):
    r = client.get(f"{prefix}/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert set(body["build"]) >= IDENTITY_KEYS
    assert body["build"] == buildinfo.build_identity()


@pytest.mark.parametrize("prefix", ["", "/api"])
def test_build_route_serves_full_document(client, prefix):
    r = client.get(f"{prefix}/build")
    assert r.status_code == 200
    doc = r.get_json()
    assert set(doc) >= BUILD_KEYS
    assert set(doc["identity"]) >= IDENTITY_KEYS
    assert set(doc["distributions"]) == set(buildinfo.WORKSPACE_DISTRIBUTIONS
                                            + buildinfo.EXTERNAL_DISTRIBUTIONS)
    assert isinstance(doc["lockstep"]["ok"], bool)
    assert doc["identity"]["version"] == client.get(f"{prefix}/health").get_json()["build"]["version"]


def test_health_and_build_never_5xx_when_identity_fails(client, monkeypatch):
    def boom():
        raise RuntimeError("no metadata here")
    monkeypatch.setattr(routes._buildinfo, "build_identity", boom)
    monkeypatch.setattr(routes._buildinfo, "build_info", boom)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["build"]["error"].startswith("RuntimeError")
    r = client.get("/build")
    assert r.status_code == 200
    doc = r.get_json()
    assert doc["identity"] is None
    assert doc["error"].startswith("RuntimeError")
