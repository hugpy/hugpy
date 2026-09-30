"""central.toolserver_link — resolves the toolserver via abstract_toolserver
discovery (never starts one) and calls through the shared client (mocked)."""
import json
import os

import pytest

pytest.importorskip("abstract_toolserver.discovery")

from abstract_toolserver import discovery as D  # noqa: E402
from abstract_toolserver import client as C  # noqa: E402
from hugpy_fleet.central import toolserver_link as L  # noqa: E402


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "SYSTEM_DIRS", ())
    monkeypatch.setattr(D, "healthz", lambda url, timeout=None, opener=None: None)
    for k in D.URL_ENV_KEYS + D.TOKEN_ENV_KEYS + ("ABSTRACT_TOOLSERVER_ENDPOINT_FILE",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HUGPY_HOME", raising=False)
    return tmp_path


def test_discover_uses_advertised_endpoint_and_never_starts(home, monkeypatch):
    D.write_advertisement(D.make_advertisement("http://127.0.0.1:7991", "127.0.0.1", 7991,
                                               os.getpid()))
    monkeypatch.setattr(D, "start_local", lambda **kw: pytest.fail("central must not start one"))
    res = L.discover(timeout=5)
    assert res == {"url": "http://127.0.0.1:7991", "source": "advertised"}
    assert L.endpoint()["url"] == "http://127.0.0.1:7991"


def test_discover_none_is_fail_open(home, monkeypatch):
    monkeypatch.setattr(D, "start_local", lambda **kw: pytest.fail("central must not start one"))
    res = L.discover(timeout=5)
    assert res["url"] == "" and res["source"] == "none"


def test_call_goes_through_shared_client(home, monkeypatch):
    D.write_advertisement(D.make_advertisement("http://127.0.0.1:7990", "127.0.0.1", 7990,
                                               os.getpid()))
    L.discover(timeout=5)
    seen = {}

    def fake_post(self, path, body, timeout=None):
        seen["url"], seen["path"], seen["body"] = self.url, path, body
        return {"result": {"ok": True}}

    monkeypatch.setattr(C.ToolserverClient, "_post", fake_post)
    assert L.call("metrics_loads", limit=1) == {"ok": True}
    assert seen == {"url": "http://127.0.0.1:7990", "path": "/ts/call",
                    "body": {"name": "metrics_loads", "arguments": {"limit": 1}}}
