"""The console Help button's help-agent query path (help_agent + help_routes).

* GATE: operator-only in the route AND in operator_auth._SENSITIVE; members 403,
  anonymous 401; HUGPY_AGENT_OPEN never waives it; HUGPY_HELP_AGENT=0 -> 503.
* BACKEND SELECTION: Claude arm preferred; mocked down -> local model; both down
  -> an honest error record. The chosen backend is recorded in the transcript.
* The real ClaudeArmBackend against a fake abstract-claude console API.
* TRANSCRIPTS: append-only JSONL under the sessions dir; syncs never duplicate.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from flask import Flask

from hugpy_server.app import help_agent
from hugpy_server.app import operator_auth as oa
from hugpy_server.app.routes import help_routes


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeClaude:
    name = "claude-arm"
    tools = True
    label = "fake claude arm"
    base = "http://fake-claude"

    def __init__(self, up=True):
        self.up = up
        self.sent = []
        self.events = []
        self.stopped = False

    def available(self):
        return self.up

    def start(self, sid, text):
        self.sent.append(text)
        return {"ref": "cs-fake", "message_ids": ["m1"], "queued": False}

    def send(self, ref, text):
        self.sent.append(text)
        return {"message_ids": [f"m{len(self.sent)}"], "queued": False}

    def pull(self, ref, since):
        return [e for e in self.events if e["seq"] > since]

    def stop(self, ref):
        self.stopped = True


class FakeLocal:
    name = "local-model"
    tools = False
    label = "local model via hugpy /v1 — read-only, no tools"

    def __init__(self, up=True, reply="local answer"):
        self.up, self.reply, self.calls = up, reply, []

    def available(self):
        return self.up

    def complete(self, messages, timeout=240):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    d = tmp_path / "help_sessions"
    monkeypatch.setenv("HUGPY_HELP_SESSIONS_DIR", str(d))
    # the local grounding snapshot must never reach a live central in tests
    monkeypatch.setattr(help_agent, "grounding_snapshot", lambda: "{}")
    return d


def _svc(*backends):
    return help_agent.HelpService(help_agent.HelpStore(), lambda: list(backends))


def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ── app / gate ──────────────────────────────────────────────────────────────
def _app(monkeypatch, svc=None, gate=True):
    if svc is not None:
        monkeypatch.setattr(help_agent, "_SERVICE", svc)
    app = Flask(__name__)
    app.register_blueprint(help_routes.help_bp)
    if gate:
        oa.install_operator_gate(app)
    return app


@pytest.fixture
def external(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "op-secret")
    monkeypatch.setattr(oa, "_fetch_me", lambda: (False, None))
    oa._clear_session_caches()
    yield
    oa._clear_session_caches()


ROUTES = [("GET", "/llm/help/sessions"), ("POST", "/llm/help/sessions"),
          ("GET", "/llm/help/sessions/hs-000000000000"),
          ("POST", "/llm/help/sessions/hs-000000000000/messages"),
          ("GET", "/llm/help/sessions/hs-000000000000/stream"),
          ("POST", "/llm/help/sessions/hs-000000000000/stop"),
          ("GET", "/llm/help/logs?source=central"), ("GET", "/llm/help/status")]


@pytest.mark.parametrize("method,path", ROUTES)
def test_anonymous_gets_401_everywhere(monkeypatch, external, sessions_dir, method, path):
    c = _app(monkeypatch, _svc(FakeClaude())).test_client()
    r = c.open(path, method=method, json={"prompt": "x", "text": "x"})
    assert r.status_code == 401
    # the /api mount is covered by the before_request gate too
    r = c.open("/api" + path, method=method, json={"prompt": "x"})
    assert r.status_code in (401, 404)


@pytest.mark.parametrize("method,path", ROUTES)
def test_route_gate_alone_refuses_anonymous(monkeypatch, external, sessions_dir, method, path):
    """Without the app-wide before_request gate the route still refuses."""
    c = _app(monkeypatch, _svc(FakeClaude()), gate=False).test_client()
    assert c.open(path, method=method, json={"prompt": "x"}).status_code == 401


def test_member_gets_403(monkeypatch, external, sessions_dir):
    monkeypatch.setattr(oa, "principal_role", lambda: oa.ROLE_MEMBER)
    monkeypatch.setattr(oa, "operator_authenticated", lambda: False)
    c = _app(monkeypatch, _svc(FakeClaude()), gate=False).test_client()
    r = c.post("/llm/help/sessions", json={"prompt": "hi"})
    assert r.status_code == 403
    assert r.get_json()["gate"] == "operator"


def test_agent_open_flag_does_not_waive(monkeypatch, external, sessions_dir):
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "1")
    c = _app(monkeypatch, _svc(FakeClaude())).test_client()
    assert c.post("/llm/help/sessions", json={"prompt": "hi"}).status_code == 401
    assert c.get("/llm/help/sessions").status_code == 401


def test_sensitive_inventory_lists_help_paths():
    app = Flask(__name__)
    for m, p in (("GET", "/llm/help/sessions"), ("POST", "/api/llm/help/sessions/x/messages"),
                 ("GET", "/llm/help/logs")):
        with app.test_request_context(p, method=m):
            assert oa._path_is_sensitive(), (m, p)


def test_operator_token_passes_and_kill_switch(monkeypatch, external, sessions_dir):
    fake = FakeClaude()
    c = _app(monkeypatch, _svc(fake)).test_client()
    h = {"X-Operator-Token": "op-secret"}
    r = c.post("/llm/help/sessions", json={"prompt": "why?"}, headers=h)
    assert r.status_code == 201, r.get_json()
    assert r.get_json()["backend"] == "claude-arm"
    monkeypatch.setenv("HUGPY_HELP_AGENT", "0")
    assert c.get("/llm/help/sessions", headers=h).status_code == 503


# ── backend selection ───────────────────────────────────────────────────────
def test_claude_preferred_when_up(sessions_dir):
    fake = FakeClaude()
    t = _svc(fake, FakeLocal()).start("what failed?", {"tab": "compute"})
    assert t["backend"] == "claude-arm" and t["tools"] is True
    # the first turn carries the system context + the context strip
    assert "HUGPY HELP AGENT" in fake.sent[0]
    assert "HARD RULES" in fake.sent[0]
    assert "- tab: compute" in fake.sent[0]
    assert t["status"] == "busy"


def test_fallback_to_local_when_claude_down(sessions_dir):
    local = FakeLocal(reply="it was Echo-Mini")
    svc = _svc(FakeClaude(up=False), local)
    t = svc.start("what failed?", "route: /console")
    assert t["backend"] == "local-model" and t["tools"] is False
    assert "read-only" in t["backend_label"]
    meta = t["records"][0]
    assert [x["available"] for x in meta["tried"]] == [False, True]
    assert _wait(lambda: svc.transcript(t["id"])["status"] == "idle")
    recs = svc.transcript(t["id"])["records"]
    assert any(r.get("type") == "text" and r["text"] == "it was Echo-Mini" for r in recs)
    assert local.calls[0][0]["role"] == "system"
    # follow-up carries history
    svc.send(t["id"], "and why?")
    assert _wait(lambda: len(local.calls) == 2)
    roles = [m["role"] for m in local.calls[1]]
    assert roles[:4] == ["system", "user", "assistant", "user"]


def test_both_backends_down_is_an_honest_error(sessions_dir):
    t = _svc(FakeClaude(up=False), FakeLocal(up=False)).start("hello")
    assert t["backend"] is None
    assert t["status"] == "error"
    assert any(r["kind"] == "error" for r in t["records"])


def test_prefer_local(sessions_dir):
    t = _svc(FakeClaude(), FakeLocal()).start("hi", prefer="local")
    assert t["backend"] == "local-model"


# ── transcript persistence ──────────────────────────────────────────────────
def test_transcript_is_append_only_jsonl_and_sync_is_idempotent(sessions_dir):
    fake = FakeClaude()
    svc = _svc(fake)
    t = svc.start("q1")
    sid = t["id"]
    path = sessions_dir / f"{sid}.jsonl"
    assert path.exists()
    before = path.read_text().splitlines()
    fake.events = [
        {"seq": 1, "type": "status", "state": "compiling"},
        {"seq": 2, "type": "tool", "name": "Bash", "summary": "journalctl -u 7002", "input": "{}"},
        {"seq": 3, "type": "tool_result", "is_error": False, "text": "x" * 5000},
        {"seq": 4, "type": "text", "text": "Echo-Mini failed: wrong tensor shape"},
        {"seq": 5, "type": "done", "rc": 0, "result": "ok", "message_ids": ["m1"]},
    ]
    t1 = svc.transcript(sid)
    t2 = svc.transcript(sid)          # second sync must not re-append
    after = path.read_text().splitlines()
    assert after[:len(before)] == before                  # append-only
    assert t1["count"] == t2["count"] == len(before) + 5
    assert t2["status"] == "idle"
    tr = next(r for r in t2["records"] if r.get("type") == "tool_result")
    assert len(tr["text"]) == 5000                            # stored whole, never cut
    # since= returns only the tail
    tail = svc.transcript(sid, since=t2["count"] - 1)["records"]
    assert len(tail) == 1 and tail[0]["type"] == "done"
    # follow-up: busy until its done arrives
    svc.send(sid, "q2", {"model": "Echo-Mini"})
    assert svc.transcript(sid)["status"] == "busy"
    assert "OPERATOR CONTEXT" in fake.sent[-1] and "HARD RULES" not in fake.sent[-1]
    fake.events.append({"seq": 6, "type": "done", "rc": 0, "message_ids": ["m2"]})
    assert svc.transcript(sid)["status"] == "idle"
    # listing
    rows = svc.store.list()
    assert rows[0]["id"] == sid and rows[0]["title"] == "q1" and rows[0]["backend"] == "claude-arm"
    # stop
    svc.stop(sid)
    assert fake.stopped and svc.transcript(sid)["status"] == "stopped"


def test_bad_ids_are_rejected(sessions_dir):
    svc = _svc(FakeClaude())
    with pytest.raises(KeyError):
        svc.transcript("../../etc/passwd")
    with pytest.raises(KeyError):
        svc.transcript("hs-000000000000")


def test_derive_status_error_on_failed_turn():
    recs = [{"kind": "user", "message_ids": ["a"]},
            {"kind": "event", "type": "done", "rc": 1, "error": "held", "message_ids": ["a"]}]
    assert help_agent.derive_status(recs) == "error"


# ── the real ClaudeArmBackend against a fake abstract-claude console API ───
class _FakeServe(BaseHTTPRequestHandler):
    sessions = {}
    events = {}

    def log_message(self, *a):
        pass

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/console/sessions"):
            return self._json(200, {"sessions": [], "backends": ["claude"]})
        if self.path.startswith("/api/console/events"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            sid, since = q["id"][0], int(q["since"][0])
            return self._json(200, {"events": [e for e in self.events.get(sid, []) if e["seq"] > since]})
        self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/console/sessions":
            assert body["cwd"] == help_agent.SOURCE_ROOT
            self.sessions["cs-1"] = body
            return self._json(200, {"id": "cs-1"})
        if self.path == "/api/console/chat":
            sid = body["session_id"]
            evs = self.events.setdefault(sid, [])
            n = len(evs)
            evs.append({"seq": n + 1, "type": "text", "text": "reply to: " + body["prompt"][-12:]})
            evs.append({"seq": n + 2, "type": "done", "rc": 0, "message_ids": [f"x{n}"]})
            return self._json(200, {"session_id": sid, "message_ids": [f"x{n}"], "queued": False})
        if self.path == "/api/console/interrupt":
            return self._json(200, {"ok": True})
        self._json(404, {})


def test_claude_arm_backend_http_contract(sessions_dir, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServe)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        dead = help_agent.ClaudeArmBackend("http://127.0.0.1:9")   # nothing listens
        live = help_agent.ClaudeArmBackend(url)
        assert dead.available() is False and live.available() is True
        chosen, tried = help_agent.select_backend([dead, live, FakeLocal()])
        assert chosen is live and [x["available"] for x in tried] == [False, True]
        svc = _svc(dead, live, FakeLocal())
        t = svc.start("what failed on aeb?")
        assert t["backend"] == "claude-arm"
        t = svc.transcript(t["id"])
        assert t["status"] == "idle"
        assert any(r.get("type") == "text" and r["text"].endswith("on aeb?") for r in t["records"])
    finally:
        srv.shutdown()
