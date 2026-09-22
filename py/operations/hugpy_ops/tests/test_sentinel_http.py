"""Sentinel against a FAKE HTTP central: the real urllib fetch layer
(``checks._default_get``) hits a local server serving the three read
surfaces; ``run_once`` opens cases, the weight_missing fast path POSTs the
real ``/llm/repos/download`` remedy, and settings resolve central + state
root through the platform."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hugpy_ops.sentinel import checks, runner
from hugpy_ops.sentinel.cases import CaseStore
from hugpy_ops.sentinel.settings import SentinelSettings, default_state_dir, load_settings


@pytest.fixture
def fake_central():
    state = {
        "jobs": {"jobs": [
            {"id": "j-stalled", "status": "processing", "stalled": True,
             "progressed_at": 1.0, "worker": "computron", "model_key": "m"},
            {"id": "j-fine", "status": "processing", "stalled": False}],
            "counts": {}},
        "workers": [
            {"name": "computron", "status": "online", "version_ok": None,
             "allocations": [{"model_key": "Qwen~Pilot", "healthy": True}]},
            {"name": "ae", "status": "offline", "version_ok": None},
            {"name": "op", "status": "online", "version_ok": False,
             "pkg_version": "0.1.1", "required_pkg_version": "0.1.9"}],
        "capabilities": {"ok": True, "count": 0, "capabilities": []},
        "posts": [],
    }

    class Handler(BaseHTTPRequestHandler):
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
            if self.path.startswith("/llm/jobs"):
                return self._json(200, state["jobs"])
            if self.path == "/llm/workers":
                return self._json(200, state["workers"])
            if self.path == "/oracle/capabilities":
                return self._json(200, state["capabilities"])
            return self._json(404, {"error": "no such surface"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state["posts"].append((self.path, body))
            if self.path == "/llm/repos/download":
                return self._json(200, {"id": "job-42", "status": "pending"})
            return self._json(404, {})

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", state
    finally:
        srv.shutdown()


def _settings(tmp_path, central):
    return SentinelSettings(central=central, state_dir=str(tmp_path / "state"),
                            stalled_grace_s=10.0)


def test_detect_over_real_http(fake_central, tmp_path):
    base, _ = fake_central
    s = _settings(tmp_path, base)
    anomalies = checks.detect(s, now=1_000.0, wants_fn=lambda: [])
    kinds = sorted((a.kind, a.fingerprint) for a in anomalies)
    assert ("job_stalled", "job_stalled:j-stalled") in kinds
    assert ("worker_offline", "worker_offline:ae") in kinds
    assert ("worker_version", "worker_version:op") in kinds
    assert not any(k == "job_stalled" and f.endswith("j-fine") for k, f in kinds)
    skew = next(a for a in anomalies if a.kind == "worker_version")
    # the partition-era version evidence: ecosystem distributions, lazily
    assert set(skew.evidence["local_distributions"]) == {"hugpy-fleet", "hugpy-server"}
    assert skew.evidence["required_pkg_version"] == "0.1.9"


def test_run_once_over_real_http_opens_cases_and_posts_download(fake_central, tmp_path):
    base, state = fake_central
    s = _settings(tmp_path, base)
    s.pilot_light = "Qwen~Pilot"          # warm on computron -> no case
    store = CaseStore(s.db_path)
    spawned = []

    def fake_run(cmd, **kw):
        spawned.append(cmd)
        class P:
            returncode = 0
            stdout = json.dumps({"outcome": "done", "run_id": "r1", "answer": "# report"})
            stderr = ""
        return P()

    want = {"registry": "tasks", "name": "llm-gone", "reason": "absent",
            "dest": "/store/models", "hub_id": "Org/Gone", "filename": "gone.gguf",
            "framework": "gguf", "fingerprint": "weight_missing:tasks:llm-gone",
            "resolved": True}
    summary = runner.run_once(s, store=store, run=fake_run, wants_fn=lambda: [want])
    try:
        cases = {c.kind: c for c in store.list()}
        assert set(cases) == {"job_stalled", "worker_offline", "worker_version", "weight_missing"}
        # three agent spawns (one per non-weight case), each documented
        assert len(spawned) == 3 and all(c[1] == "case" for c in spawned)
        assert all(cases[k].state == "documented" for k in ("job_stalled", "worker_offline", "worker_version"))
        # the deterministic remedy went over REAL HTTP to the fake central
        assert cases["weight_missing"].state == "remedied"
        assert state["posts"] == [("/llm/repos/download",
                                   {"hub_id": "Org/Gone", "filename": "gone.gguf",
                                    "name": "llm-gone", "framework": "gguf",
                                    "register": False})]
        assert "job-42" in (cases["weight_missing"].note or "")
        assert len(summary["opened"]) == 4
        # second pass: everything re-detected touches, nothing re-spawns/re-posts
        summary2 = runner.run_once(s, store=store, run=fake_run, wants_fn=lambda: [want])
        assert summary2["opened"] == [] and len(summary2["touched"]) == 4
        assert len(spawned) == 3 and len(state["posts"]) == 1
    finally:
        store.close()


def test_settings_resolve_central_and_state_root_via_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGPY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HUGPY_SENTINEL_DIR", raising=False)
    monkeypatch.delenv("HUGPY_SENTINEL_CENTRAL", raising=False)
    monkeypatch.setenv("HUGPY_BASE_URL", "http://platform-central:7002/")
    s = load_settings()
    assert s.central == "http://platform-central:7002"
    assert s.state_dir == default_state_dir()
    assert s.state_dir == str(tmp_path / "home" / "state" / "sentinel")
    # the legacy alias still resolves through the platform, explicit wins
    monkeypatch.delenv("HUGPY_BASE_URL")
    monkeypatch.setenv("HUGPY_CENTRAL", "http://legacy:7002")
    assert load_settings().central == "http://legacy:7002"
    monkeypatch.setenv("HUGPY_SENTINEL_CENTRAL", "http://explicit:1/")
    assert load_settings().central == "http://explicit:1"
