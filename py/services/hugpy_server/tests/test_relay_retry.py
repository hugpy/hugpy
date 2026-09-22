"""_relay_worker_op retry-on-connect (apply-blip fix): config-style ops get
ONE retry after a 3s pause when the worker agent's socket is dead (it re-execs
~0.5s after ACKing a config change), and an honest 503 "agent is restarting"
when the retry also can't connect. Non-config ops keep the single-shot 502.

k59: the relay now goes through worker_http (short connect + per-worker
breaker), so the seam patched here is httpx.request rather than httpx.post,
and each case starts from a clean breaker — a run of failures across UNRELATED
cases would otherwise open it and fail the next case fast, which is correct
behaviour but not what this file is about (see test_worker_http_discipline.py).
The wire contract is unchanged: the error `code` is still the underlying
transport error's class name.
"""
from __future__ import annotations

import importlib
import time

import httpx
import pytest
from flask import Flask

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")
cr = importlib.import_module("hugpy_server.app.routes.comms_routes")
wh = importlib.import_module("hugpy_fleet.central.worker_http")


class _FakeResp:
    status_code = 200

    def json(self):
        return {"ok": True, "restarting": True}


class Relay:
    def __init__(self):
        self.posts: list = []
        self.sleeps: list = []
        self.outcomes: list = []     # per-call: "ok" | exception instance to raise
        self.app = Flask("relay-retry-test")

    def fake_request(self, method, url, **kwargs):
        self.posts.append(url)
        out = self.outcomes.pop(0)
        if out == "ok":
            return _FakeResp()
        raise out

    def run(self, retry):
        wh.reset_breakers()     # each case is its own worker's first contact
        with self.app.app_context():
            resp, status = wr._relay_worker_op(
                "w1", "/ops/config", {"pinned": {"m": True}},
                timeout=15.0, action="config", retry_on_connect=retry)
            return resp.get_json(), status


@pytest.fixture
def relay(monkeypatch):
    r = Relay()
    monkeypatch.setattr(wr, "get_worker", lambda wid: {"name": "t", "url": "http://worker:9999"})
    monkeypatch.setattr(cr, "audit", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "request", r.fake_request)
    monkeypatch.setattr(time, "sleep", lambda s: r.sleeps.append(s))   # _relay imports time as _time
    return r


def test_blip_then_recovery(relay):
    """connect error -> 3s pause -> retry succeeds."""
    relay.outcomes[:] = [httpx.ConnectError("refused"), "ok"]
    body, status = relay.run(retry=True)
    assert status == 200 and body["ok"] is True
    assert relay.sleeps == [3.0], "exactly one 3s pause"
    assert len(relay.posts) == 2, "two POST attempts"


def test_persistent_blip_honest_503(relay):
    """both attempts fail -> honest 503, not a bare 502."""
    relay.outcomes[:] = [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow")]
    body, status = relay.run(retry=True)
    assert status == 503
    assert "restarting" in body["error"]["message"]
    assert body["error"]["code"] == "AgentRestarting"


def test_non_config_ops_single_shot_502(relay):
    """no retry without the flag: generic 502, no sleeping (historical)."""
    relay.outcomes[:] = [httpx.ConnectError("refused")]
    body, status = relay.run(retry=False)
    assert status == 502
    assert body["error"]["code"] == "ConnectError", "generic error shape preserved"
    assert relay.sleeps == [] and len(relay.posts) == 1


def test_non_connect_failures_never_retry(relay):
    """non-connect failures never retry, even for config ops."""
    relay.outcomes[:] = [ValueError("bad json")]
    body, status = relay.run(retry=True)
    assert status == 502
    assert relay.sleeps == [] and len(relay.posts) == 1
