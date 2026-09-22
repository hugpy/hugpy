"""k29 — Hugging Face credentials store + routes.

Covers: token-store round-trip (0600, precedence, delete), whoami validation
(mocked — no live HF), the GET/POST/DELETE routes end-to-end through the REAL
operator gate, invalid-token rejection (400), and that the routes are in the
operator _SENSITIVE allowlist.
"""
from __future__ import annotations

import importlib
import os
import stat

import pytest
import requests
from flask import Flask

_C = importlib.import_module("hugpy_platform.constants")
hf = importlib.import_module("hugpy_storage.hf_token")
sr = importlib.import_module("hugpy_server.app.routes.search_routes")
oa = importlib.import_module("hugpy_server.app.operator_auth")

_VALID = "hf_valid_token_abcd1234"
OP_HEADERS = {"X-Operator-Token": "op-secret-xyz"}


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _fake_get(url, headers=None, timeout=None, **kw):
    """whoami mock: swap requests.get so no network is touched."""
    assert "whoami-v2" in url
    sent = (headers or {}).get("Authorization", "")
    if sent == f"Bearer {_VALID}":
        return _Resp(200, {"name": "octocat", "fullname": "The Octocat"})
    return _Resp(401, {"error": "Invalid credentials in Authorization header"})


@pytest.fixture(autouse=True)
def hf_env(monkeypatch, tmp_path):
    """Isolate the token file to a throwaway path (the reader keys off the
    constants layer, the writer off the store module — override BOTH so the
    test never touches a real token), start from a clean env (no ambient
    HF_TOKEN, no captured env token), and put the operator gate in enforcing
    mode (open mode + a configured token)."""
    token_path = str(tmp_path / "hf_token")
    monkeypatch.setattr(_C, "HF_TOKEN_PATH", token_path)
    monkeypatch.setattr(hf, "HF_TOKEN_PATH", token_path)
    monkeypatch.setattr(_C, "HF_TOKEN_ENV", False)
    # apply_hf_token_to_env rebuilds these; keep the real ones out of the test.
    monkeypatch.setattr(_C, "HF_TOKEN", getattr(_C, "HF_TOKEN", False), raising=False)
    monkeypatch.setattr(_C, "hfApi", getattr(_C, "hfApi", None), raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "op-secret-xyz")
    monkeypatch.setattr(requests, "get", _fake_get)
    return token_path


@pytest.fixture
def client():
    """The search blueprint behind the REAL operator gate."""
    app = Flask(__name__)
    app.register_blueprint(sr.search_bp)
    oa.install_operator_gate(app)
    return app.test_client()


# ── 1) validation ────────────────────────────────────────────────────────────
def test_validate_valid_token_returns_username():
    st, user, err = hf.validate_hf_token(_VALID)
    assert st == "ok" and user == "octocat"


def test_validate_bad_token_returns_invalid_with_hf_message():
    st, user, err = hf.validate_hf_token("hf_bogus")
    assert st == "invalid" and "Invalid" in (err or "")


# ── 2) store round-trip + 0600 + precedence ──────────────────────────────────
def test_store_round_trip_0600_and_precedence(hf_env):
    assert hf.get_hf_token() is None, "none stored, no env"
    assert hf.token_source() is None

    hf.store_hf_token(_VALID)
    assert hf.get_hf_token() == _VALID
    assert hf.token_source() == "stored"
    assert stat.S_IMODE(os.stat(hf.HF_TOKEN_PATH).st_mode) == 0o600
    # stored file lives at the isolated path (outside any git tree)
    assert hf.HF_TOKEN_PATH == hf_env and os.path.exists(hf_env)
    # apply pushed it into the env seam for implicit call sites
    assert os.environ.get("HF_TOKEN") == _VALID

    # stored wins over env (env source = the genuine captured constants.HF_TOKEN_ENV)
    _C.HF_TOKEN_ENV = "env-fallback-token"
    assert hf.get_hf_token() == _VALID
    assert hf.token_source() == "stored"


# ── 3) delete -> falls back to the genuine env token ─────────────────────────
def test_delete_falls_back_to_env_token():
    hf.store_hf_token(_VALID)
    _C.HF_TOKEN_ENV = "env-fallback-token"
    assert hf.delete_hf_token() is True
    assert not os.path.exists(hf.HF_TOKEN_PATH)
    assert hf.get_hf_token() == "env-fallback-token"
    assert hf.token_source() == "env"
    _C.HF_TOKEN_ENV = False   # back to a truly clean slate
    hf.apply_hf_token_to_env()
    assert hf.get_hf_token() is None


# ── 4) routes end-to-end through the REAL operator gate ──────────────────────
@pytest.mark.parametrize("verb", ["get", "post", "delete"])
def test_gate_refuses_unauthenticated_on_every_verb(client, verb):
    kw = {"json": {"token": _VALID}} if verb == "post" else {}
    assert getattr(client, verb)("/llm/hf/auth", **kw).status_code == 401


def test_get_anonymous_shape(client):
    r = client.get("/llm/hf/auth", headers=OP_HEADERS)
    j = r.get_json()
    assert r.status_code == 200
    assert j["authenticated"] is False
    assert j["username"] is None and j["token_last4"] is None and j["source"] is None


def test_post_invalid_token_400_nothing_stored(client):
    r = client.post("/llm/hf/auth", headers=OP_HEADERS, json={"token": "hf_bogus"})
    assert r.status_code == 400
    assert hf.get_hf_token() is None


def test_post_missing_token_400(client):
    assert client.post("/llm/hf/auth", headers=OP_HEADERS, json={}).status_code == 400


def test_post_valid_then_delete(client):
    r = client.post("/llm/hf/auth", headers=OP_HEADERS, json={"token": _VALID})
    j = r.get_json()
    assert r.status_code == 200 and j["authenticated"] is True and j["username"] == "octocat"
    assert j["token_last4"] == _VALID[-4:], "last4 only"
    assert _VALID not in r.get_data(as_text=True), "the token is never echoed"
    assert j["source"] == "stored"

    r = client.delete("/llm/hf/auth", headers=OP_HEADERS)
    j = r.get_json()
    assert r.status_code == 200 and j.get("removed") is True and j["source"] is None


# ── 5) gating declared in _SENSITIVE (bare + /api-mounted) ───────────────────
def _gated(path, method):
    p = path
    if p == "/api" or p.startswith("/api/"):
        p = p[len("/api"):] or "/"
    return any(method in m and rx.match(p) for m, rx in oa._SENSITIVE)


@pytest.mark.parametrize("verb", ["GET", "POST", "DELETE"])
def test_hf_auth_route_is_operator_gated(verb):
    assert _gated("/llm/hf/auth", verb)


def test_api_mounted_path_also_gated():
    assert _gated("/api/llm/hf/auth", "POST")
