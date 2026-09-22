"""Secure one-time INSTALL LINKS (2026-07-23) — offline tests, NO network.

Four layers:
  * api_keys scope extension: the verification matrix (a scoped key passes its
    scope and fails others, "full" passes everything, LEGACY rows — no new
    fields — behave exactly as before), expiry/disabled, additive list view.
  * install_links store: mint (raw key never in the public view), download
    consumption + scrubbing, wrapper fetches free, revoke kills link AND key.
  * the Flask routes via test_client: operator gate on mint/list/revoke
    (strict — HUGPY_AGENT_OPEN does NOT waive it), the link lifecycle
    (mint → .sh → .py → exhausted 410), the templated download containing the
    key and py_compile-ing, ttl expiry, raw key never in any operator-facing
    response.
  * the central operator allowlist (operator_auth._SENSITIVE) covers exactly
    the management routes and never the download GET.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_install_links.py -q
"""
import io
import json
import os
import py_compile
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-install-links-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", os.path.join(
    tempfile.mkdtemp(prefix="hugpy-install-links-comms-"), "comms.db"))
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

import pytest
from flask import Flask

from hugpy_server.app.functions.imports.utils import api_keys as ak
from hugpy_server.app.functions.imports.utils import install_links as il
from hugpy_server.app.routes import agent_routes
from hugpy_server.app import operator_auth


# ── fixtures: throwaway store files per test ────────────────────────────────
@pytest.fixture(autouse=True)
def scratch_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ak, "_store_path",
                        lambda: str(tmp_path / "api_keys.json"))
    monkeypatch.setattr(il, "_store_path",
                        lambda: str(tmp_path / "install_links.json"))
    yield


@pytest.fixture
def installer_py(tmp_path, monkeypatch):
    """A minimal stand-in installer WITH the EMBEDDED_API_KEY slot, plus the
    REAL repo installer for the compile test."""
    src = tmp_path / "install_hugpy_agent.py"
    src.write_text('#!/usr/bin/env python3\n'
                   'EMBEDDED_API_KEY = ""\n'
                   'if __name__ == "__main__":\n'
                   '    print(bool(EMBEDDED_API_KEY))\n')
    monkeypatch.setenv("HUGPY_AGENT_INSTALLER_PY", str(src))
    return str(src)


OP_TOKEN = "op-secret-token"


@pytest.fixture
def client(monkeypatch, installer_py):
    """Bare app with only the agent blueprint. Operator auth via
    HUGPY_OPERATOR_TOKEN in open mode (the operator_auth contract: open mode +
    token set => the token is required)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", OP_TOKEN)
    monkeypatch.delenv("HUGPY_AGENT_OPEN", raising=False)
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app.test_client()


def _op():
    return {"X-Operator-Token": OP_TOKEN}


def _mint(client, **over):
    body = {"label": "test box"}
    body.update(over)
    r = client.post("/agent/install-links", json=body, headers=_op())
    assert r.status_code == 201, r.get_json()
    return r.get_json()


# ═══════════════════════════════════════════════════════════════════════════
# 1) api_keys scope matrix
# ═══════════════════════════════════════════════════════════════════════════
def test_scope_matrix_scoped_key():
    k = ak.create_api_key(name="n", scopes=["v1"])
    tok = k["key"]
    assert ak.verify_api_key(tok) is True                       # unscoped call: as today
    assert ak.verify_api_key(tok, required_scope="v1") is True  # its scope
    assert ak.verify_api_key(tok, required_scope="ml") is False # others refused
    assert ak.verify_api_key(tok, required_scope="agent-register") is False


def test_scope_matrix_full_passes_all():
    tok = ak.create_api_key(name="n", scopes=["full"])["key"]
    for scope in (None, "v1", "ml", "agent-register", "full"):
        assert ak.verify_api_key(tok, required_scope=scope) is True


def test_scope_matrix_multi_scope():
    tok = ak.create_api_key(name="n", scopes=["v1", "ml"])["key"]
    assert ak.verify_api_key(tok, required_scope="v1") is True
    assert ak.verify_api_key(tok, required_scope="ml") is True
    assert ak.verify_api_key(tok, required_scope="agent-register") is False


def test_legacy_rows_read_as_full_scope():
    """A pre-scope row (no label/scopes/created_by/expires_at/disabled) must
    verify against ANY required_scope and list with the defaults — the lazy
    additive migration."""
    tok = ak.create_api_key(name="old")["key"]
    key_id = ak.key_id_for_token(tok)
    # Strip the new fields to fabricate a genuine legacy row on disk.
    data = ak._load()
    rec = data["keys"][key_id]
    for f in ("label", "scopes", "created_by", "expires_at", "disabled"):
        rec.pop(f, None)
    ak._save(data)
    assert ak.verify_api_key(tok) is True
    assert ak.verify_api_key(tok, required_scope="v1") is True
    assert ak.verify_api_key(tok, required_scope="agent-register") is True
    listed = [k for k in ak.list_api_keys() if k["id"] == key_id][0]
    assert listed["scopes"] == ["full"]
    assert listed["label"] == ""
    assert listed["created_by"] == "operator"
    assert listed["expires_at"] is None and listed["disabled"] is False


def test_unknown_scope_refused_at_mint():
    with pytest.raises(ValueError):
        ak.create_api_key(name="n", scopes=["v2-typo"])


def test_key_expiry_and_disabled():
    tok = ak.create_api_key(name="n", scopes=["v1"],
                            expires_at=time.time() + 3600)["key"]
    assert ak.verify_api_key(tok) is True
    key_id = ak.key_id_for_token(tok)
    data = ak._load()
    data["keys"][key_id]["expires_at"] = time.time() - 5
    ak._save(data)
    assert ak.verify_api_key(tok) is False          # expired
    data = ak._load()
    data["keys"][key_id]["expires_at"] = None
    data["keys"][key_id]["disabled"] = True
    ak._save(data)
    assert ak.verify_api_key(tok) is False          # disabled


def test_mint_response_carries_new_fields():
    k = ak.create_api_key(name="n", label="the label", scopes=["ml"],
                          created_by="install-link")
    assert k["label"] == "the label"
    assert k["scopes"] == ["ml"]
    assert k["created_by"] == "install-link"
    assert "hash" not in k


# ═══════════════════════════════════════════════════════════════════════════
# 2) install_links store
# ═══════════════════════════════════════════════════════════════════════════
def test_store_mint_never_returns_raw_key():
    link = il.create_install_link(label="box A")
    assert "raw_key" not in link
    assert link["status"] == "active"
    assert link["uses_left"] == 1 and link["max_uses"] == 1
    assert link["scopes"] == ["v1"]                 # spec default
    # the key it minted exists, labeled, created_by install-link:
    keys = ak.list_api_keys()
    assert any(k["id"] == link["key_id"]
               and k["created_by"] == "install-link"
               and k["label"] == "box A" for k in keys)
    # listing never leaks the raw key either:
    assert all("raw_key" not in row for row in il.list_install_links())


def test_store_consume_returns_key_then_exhausts_and_scrubs():
    link = il.create_install_link(label="one shot")
    raw = il.consume_download(link["link_id"], remote_addr="1.2.3.4")
    assert raw and raw.startswith("hp_")
    assert ak.verify_api_key(raw, required_scope="v1") is True
    # exhausted now:
    assert il.consume_download(link["link_id"]) is None
    row = il.get_link(link["link_id"])
    assert row["status"] == "exhausted" and row["uses_left"] == 0
    # the raw key is scrubbed from the store file itself:
    with open(il._store_path()) as fh:
        assert raw not in fh.read()
    # audit row recorded:
    assert row["downloads"][0]["remote_addr"] == "1.2.3.4"
    assert row["downloads"][0]["kind"] == "py"


def test_store_multi_use_counts_down():
    link = il.create_install_link(label="triple", max_uses=3)
    for left in (2, 1, 0):
        assert il.consume_download(link["link_id"]) is not None
        assert il.get_link(link["link_id"])["uses_left"] == left
    assert il.consume_download(link["link_id"]) is None


def test_store_wrapper_fetch_does_not_decrement():
    link = il.create_install_link(label="wrapped")
    assert il.peek_active(link["link_id"]) is True
    il.note_wrapper_fetch(link["link_id"], remote_addr="9.9.9.9", kind="sh")
    row = il.get_link(link["link_id"])
    assert row["uses_left"] == 1                    # untouched
    assert any(d["kind"] == "sh" for d in row["downloads"])


def test_store_ttl_expiry_refuses_and_scrubs():
    link = il.create_install_link(label="stale", link_ttl_s=1)
    data = il._load()
    data["links"][link["link_id"]]["expires_at"] = time.time() - 5
    il._save(data)
    assert il.peek_active(link["link_id"]) is False
    assert il.consume_download(link["link_id"]) is None
    assert il.get_link(link["link_id"])["status"] == "expired"
    with open(il._store_path()) as fh:
        stored = json.load(fh)
    assert stored["links"][link["link_id"]]["raw_key"] == ""


def test_store_revoke_kills_link_and_key():
    link = il.create_install_link(label="doomed")
    key_id = link["key_id"]
    assert il.revoke_install_link(link["link_id"]) is True
    assert il.get_link(link["link_id"])["status"] == "revoked"
    assert il.consume_download(link["link_id"]) is None
    # the KEY is revoked too — it verifies for nothing:
    assert all(k["id"] != key_id for k in ak.list_api_keys())
    # and the raw key is gone from disk:
    with open(il._store_path()) as fh:
        stored = json.load(fh)
    assert stored["links"][link["link_id"]]["raw_key"] == ""


def test_store_key_revoked_out_of_band_refuses_download():
    """Operator revokes the KEY directly (from the key list) — the link must
    stop serving even while nominally active."""
    link = il.create_install_link(label="key pulled")
    ak.revoke_api_key(link["key_id"])
    assert il.consume_download(link["link_id"]) is None


def test_store_blank_label_refused():
    with pytest.raises(ValueError):
        il.create_install_link(label="   ")


# ═══════════════════════════════════════════════════════════════════════════
# 3) the Flask routes
# ═══════════════════════════════════════════════════════════════════════════
def test_route_mint_requires_operator(client):
    r = client.post("/agent/install-links", json={"label": "x"})
    assert r.status_code == 401
    r = client.post("/agent/install-links", json={"label": "x"},
                    headers={"X-Operator-Token": "wrong"})
    assert r.status_code == 401


def test_route_list_and_revoke_require_operator(client):
    assert client.get("/agent/install-links").status_code == 401
    assert client.delete("/agent/install-links/abc").status_code == 401


def test_route_agent_open_does_not_waive_install_link_gate(client, monkeypatch):
    """HUGPY_AGENT_OPEN waives the fleet-view operator gates — it must NEVER
    waive a credential-minting surface (the 2026-07-16 register ruling)."""
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    assert client.post("/agent/install-links",
                       json={"label": "x"}).status_code == 401
    assert client.get("/agent/install-links").status_code == 401


def test_route_mint_response_never_contains_raw_key(client):
    import re
    link = _mint(client, label="no leak")
    blob = json.dumps(link)
    assert "raw_key" not in blob
    # no key material of any shape (hp_ + 40 hex — the actual token format;
    # a bare "hp_" substring could occur by chance inside a token_urlsafe id):
    assert not re.search(r"hp_[0-9a-f]{40}", blob)
    assert link["url"].endswith(f"/agent/install/{link['link_id']}")
    assert link["label"] == "no leak"
    assert link["scopes"] == ["v1"]


def test_route_mint_validates_body(client):
    assert client.post("/agent/install-links", json={},
                       headers=_op()).status_code == 400          # no label
    assert client.post("/agent/install-links",
                       json={"label": "x", "scopes": "v1"},
                       headers=_op()).status_code == 400          # not a list
    assert client.post("/agent/install-links",
                       json={"label": "x", "scopes": ["nope"]},
                       headers=_op()).status_code == 400          # bad scope
    assert client.post("/agent/install-links",
                       json={"label": "x", "key_expires_at": "not-a-date"},
                       headers=_op()).status_code == 400


def test_route_mint_accepts_iso_key_expiry(client):
    link = _mint(client, label="iso", key_expires_at="2027-01-01T00:00:00Z")
    keys = ak.list_api_keys()
    rec = [k for k in keys if k["id"] == link["key_id"]][0]
    assert rec["expires_at"] and rec["expires_at"] > time.time()


def test_route_download_lifecycle_and_410(client):
    link = _mint(client, label="lifecycle")
    lid = link["link_id"]

    # .sh wrapper: free, keyless, points at the .py path
    r = client.get(f"/agent/install/{lid}.sh")
    assert r.status_code == 200
    sh = r.get_data(as_text=True)
    import re
    key_rx = re.compile(r"hp_[0-9a-f]{40}")
    assert f"/agent/install/{lid}" in sh
    assert "python3" in sh
    assert not key_rx.search(sh)                    # wrapper has NO key

    # .ps1 wrapper too
    r = client.get(f"/agent/install/{lid}.ps1")
    assert r.status_code == 200
    assert not key_rx.search(r.get_data(as_text=True))

    # wrappers consumed nothing:
    rows = client.get("/agent/install-links", headers=_op()).get_json()["links"]
    row = [x for x in rows if x["link_id"] == lid][0]
    assert row["uses_left"] == 1

    # the .py download: templated key, attachment, compiles
    r = client.get(f"/agent/install/{lid}")
    assert r.status_code == 200
    assert "attachment" in (r.headers.get("Content-Disposition") or "")
    body = r.get_data(as_text=True)
    assert 'EMBEDDED_API_KEY = ""' not in body      # slot was filled
    import re
    m = re.search(r"EMBEDDED_API_KEY = '(hp_[0-9a-f]+)'", body)
    assert m, "templated key missing from the download"
    raw = m.group(1)
    assert ak.verify_api_key(raw, required_scope="v1") is True
    # the served bytes are valid python:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        served = fh.name
    py_compile.compile(served, doraise=True)
    os.unlink(served)

    # exhausted now: the .py AND the wrappers all 410 with a human message
    for path in (f"/agent/install/{lid}",
                 f"/agent/install/{lid}.sh",
                 f"/agent/install/{lid}.ps1"):
        r = client.get(path)
        assert r.status_code == 410
        assert "no longer valid" in r.get_data(as_text=True)


def test_route_download_py_carries_venv_pip_upgrade(client, monkeypatch):
    """The served installer must self-upgrade pip (venv-only) BEFORE the
    package install — old distro pips choke on modern wheels / PEP 517 builds.
    Uses the REAL repo installer; skips if this tree is packaged-only."""
    monkeypatch.delenv("HUGPY_AGENT_INSTALLER_PY", raising=False)
    if agent_routes._find_installer_py() is None:
        pytest.skip("hugpy_agent/install/install_hugpy_agent.py not in this tree")
    link = _mint(client, label="pip upgrade")
    body = client.get(f"/agent/install/{link['link_id']}").get_data(as_text=True)
    # the mandatory -m pip spelling (bare pip.exe cannot self-replace on Windows)
    assert '"-m", "pip", "install", "--upgrade", "pip"' in body
    # guarded to a venv only — never a system/PEP-668 python
    assert "base_prefix" in body
    assert 'upgrade pip (venv only)' in body
    # the served installer CREATES a venv (fixes PEP-668 refusal) BEFORE the
    # pip upgrade, and re-execs into it:
    assert 'create venv (system python only)' in body
    assert '"-m", "venv"' in body
    # the python >= 3.10 FLOOR is gated FIRST, with interpreter discovery and an
    # honest early message (the 3.9 Xcode-CLT Mac field report) — never pip's
    # cryptic "No matching distribution":
    assert 'check python version floor' in body
    assert "MIN_PY = (3, 10)" in body
    assert "brew install python" in body           # macOS remedy
    assert "apt install python3.12" in body        # linux remedy
    assert "Requires-Python" in body               # misleading-fallback fix
    # ordering: floor gate -> venv-create -> pip-upgrade -> package install
    assert (body.index('check python version floor')
            < body.index('create venv (system python only)')
            < body.index('upgrade pip (venv only)')
            < body.index('pip install hugpy_agent'))


def test_route_download_py_carries_launcher_and_workspace_fix(client, monkeypatch):
    """The served installer must (a) register the terminal-console launcher
    (.desktop on Linux with a headless guard, .lnk on Windows via WScript.Shell)
    and (b) write the credential into the venv-parent WORKSPACE — the location
    the console's config chain actually reads (the Windows 'couldn't find the
    .env' fix). All asserted on the served bytes."""
    monkeypatch.delenv("HUGPY_AGENT_INSTALLER_PY", raising=False)
    if agent_routes._find_installer_py() is None:
        pytest.skip("hugpy_agent/install/install_hugpy_agent.py not in this tree")
    link = _mint(client, label="launcher+ws")
    body = client.get(f"/agent/install/{link['link_id']}").get_data(as_text=True)
    # launcher: Linux desktop entry + headless guard
    assert "hugpy-agent.desktop" in body
    assert "Terminal=true" in body
    assert "WAYLAND_DISPLAY" in body           # headless guard signal
    # HOLD-OPEN launcher script: the .desktop Exec points at a script that
    # waits for Enter so an OpenCode-absent exit-1 shows a readable hint, not
    # a blip (operator field report 2026-07-24).
    assert "launch-console.sh" in body
    assert "Press Enter to close" in body
    assert "console exited with status" in body
    # launcher: Windows start-menu shortcut via COM, cmd /k holds the window
    assert "WScript.Shell" in body and "CreateShortcut" in body
    assert "hugpy Agent.lnk" in body
    assert "/k" in body
    # launcher: macOS app bundle opening Terminal.app via the shared script
    assert 'sys.platform == "darwin"' in body
    assert "hugpy Agent.app" in body
    assert "open -a Terminal" in body
    assert "ai.hugpy.agent" in body            # bundle identifier
    assert '"-s", "format", "icns"' in body    # sips icon conversion
    # workspace credential fix: HUGPY_WORKSPACE wiring is present
    assert "HUGPY_WORKSPACE" in body


def test_route_serves_installer_icons_public(client):
    """GET /agent/install/icon.png|.ico serve real image bytes, publicly (no
    operator token), with the right content types — the launcher decoration
    source. Magic-number assertions, not byte-exact."""
    png = client.get("/agent/install/icon.png")
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    assert png.get_data()[:8] == b"\x89PNG\r\n\x1a\n"
    ico = client.get("/agent/install/icon.ico")
    assert ico.status_code == 200
    assert ico.mimetype == "image/x-icon"
    assert ico.get_data()[:4] == b"\x00\x00\x01\x00"      # ICO magic


def test_icon_routes_not_operator_gated():
    """The icon GETs must NOT be in the operator allowlist (an icon is public,
    same posture as the .py/.sh download)."""
    assert not _sensitive("GET", "/agent/install/icon.png")
    assert not _sensitive("GET", "/agent/install/icon.ico")


def test_route_download_py_templates_icon_base(client, monkeypatch):
    """The served .py carries EMBEDDED_ICON_BASE filled with this deployment's
    public base, so the launcher step can fetch the mark from central."""
    import re
    monkeypatch.delenv("HUGPY_AGENT_INSTALLER_PY", raising=False)
    if agent_routes._find_installer_py() is None:
        pytest.skip("hugpy_agent/install/install_hugpy_agent.py not in this tree")
    link = _mint(client, label="icon base")
    body = client.get(f"/agent/install/{link['link_id']}").get_data(as_text=True)
    assert 'EMBEDDED_ICON_BASE = ""' not in body          # slot was filled
    m = re.search(r"EMBEDDED_ICON_BASE = '([^']*)'", body)
    assert m and m.group(1)                                # a real base URL
    assert m.group(1).endswith("/agent/install/icon.png") is False  # base only
    # the installer fetches icon.png / icon.ico from that base:
    assert "icon.png" in body and "icon.ico" in body


def test_route_download_real_installer_compiles(client, monkeypatch):
    """Serve the ACTUAL repo installer (hugpy_agent/install/) templated — it
    must contain the key and py_compile. Skips honestly if the repo layout
    doesn't carry hugpy_agent/ (e.g. a packaged-only deployment)."""
    monkeypatch.delenv("HUGPY_AGENT_INSTALLER_PY", raising=False)
    real = agent_routes._find_installer_py()
    if real is None:
        pytest.skip("hugpy_agent/install/install_hugpy_agent.py not in this tree")
    link = _mint(client, label="real installer")
    r = client.get(f"/agent/install/{link['link_id']}")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "EMBEDDED_API_KEY = 'hp_" in body
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        served = fh.name
    py_compile.compile(served, doraise=True)
    os.unlink(served)


def test_route_expired_link_410(client):
    link = _mint(client, label="ttl")
    data = il._load()
    data["links"][link["link_id"]]["expires_at"] = time.time() - 5
    il._save(data)
    assert client.get(f"/agent/install/{link['link_id']}").status_code == 410
    assert client.get(f"/agent/install/{link['link_id']}.sh").status_code == 410


def test_route_revoke_kills_key_and_download(client):
    link = _mint(client, label="revoked")
    r = client.delete(f"/agent/install-links/{link['link_id']}", headers=_op())
    assert r.status_code == 200
    assert client.get(f"/agent/install/{link['link_id']}").status_code == 410
    assert all(k["id"] != link["key_id"] for k in ak.list_api_keys())
    # unknown id -> 404
    assert client.delete("/agent/install-links/nope",
                         headers=_op()).status_code == 404


def test_route_unknown_link_410(client):
    assert client.get("/agent/install/definitely-not-a-link").status_code == 410


def test_route_list_shows_status_and_counts_never_keys(client):
    a = _mint(client, label="a", max_uses=3)
    client.get(f"/agent/install/{a['link_id']}")     # consume one
    b = _mint(client, label="b")
    client.delete(f"/agent/install-links/{b['link_id']}", headers=_op())
    r = client.get("/agent/install-links", headers=_op())
    assert r.status_code == 200
    blob = r.get_data(as_text=True)
    import re
    assert not re.search(r"hp_[0-9a-f]{40}", blob) and "raw_key" not in blob
    rows = {x["link_id"]: x for x in r.get_json()["links"]}
    assert rows[a["link_id"]]["status"] == "active"
    assert rows[a["link_id"]]["uses_left"] == 2
    assert rows[b["link_id"]]["status"] == "revoked"


# ═══════════════════════════════════════════════════════════════════════════
# 4) the central operator allowlist
# ═══════════════════════════════════════════════════════════════════════════
def _sensitive(method, path):
    return any(method in methods and rx.match(path)
               for methods, rx in operator_auth._SENSITIVE)


def test_allowlist_no_longer_owns_install_links(client):
    """2026-08-06 MEMBER TIER: install-link management left the central
    operator-only allowlist, because a MEMBER may mint a link for their own
    machine and a (methods, path) rule cannot express "operator OR the link's
    creator". The gating moved INTO agent_routes (_require_member_strict +
    the per-link owner check) — so the allowlist must no longer claim these
    routes, and the ROUTES must still fail closed for anonymous."""
    assert not _sensitive("POST", "/agent/install-links")
    assert not _sensitive("GET", "/agent/install-links")
    assert not _sensitive("DELETE", "/agent/install-links/abc123")
    # the download GET was never operator-gated (capability = the link id):
    assert not _sensitive("GET", "/agent/install/abc123")
    assert not _sensitive("GET", "/agent/install/abc123.sh")
    # …and the route layer is the gate now: anonymous is still refused.
    assert client.post("/agent/install-links", json={"label": "x"}).status_code == 401
    assert client.get("/agent/install-links").status_code == 401
    assert client.delete("/agent/install-links/abc").status_code == 401


def test_route_agent_open_does_not_waive_install_links(monkeypatch, client):
    """The HUGPY_AGENT_OPEN waiver waives the fleet-VIEW gates and must NEVER
    waive credential minting. Now that install-links is route-gated, that
    property lives in agent_routes._require_member_strict (which, like
    _require_operator_strict before it, does not consult the flag)."""
    monkeypatch.setenv("HUGPY_AGENT_OPEN", "true")
    app = Flask(__name__)
    with app.test_request_context("/agent/nodes", method="GET"):
        assert operator_auth._path_is_sensitive() is False   # waived, as before
    assert client.post("/agent/install-links", json={"label": "x"}).status_code == 401
    assert client.get("/agent/install-links").status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
# 5) the scope-activated gates (v1/ml/agent-register call sites)
# ═══════════════════════════════════════════════════════════════════════════
def test_v1_scoped_install_key_cannot_register_an_agent(monkeypatch):
    """An install-link key minted with the default ["v1"] scope must NOT pass
    the /agent/register gate — the structural point of scoping."""
    link = il.create_install_link(label="v1 only")
    raw = il.consume_download(link["link_id"])
    assert ak.verify_api_key(raw, required_scope="v1") is True
    assert ak.verify_api_key(raw, required_scope="agent-register") is False

    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    c = app.test_client()
    r = c.post("/agent/register", json={"name": "sneaky"},
               headers={"Authorization": f"Bearer {raw}"})
    assert r.status_code == 401


def test_agent_register_scope_passes(monkeypatch):
    link = il.create_install_link(label="enroller",
                                  scopes=["agent-register"])
    raw = il.consume_download(link["link_id"])
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    c = app.test_client()
    r = c.post("/agent/register", json={"name": "legit"},
               headers={"Authorization": f"Bearer {raw}"})
    # 201 (the store is the real comms sqlite — scratch env) or 500 if the
    # comms db can't init here; the GATE decision is what we assert:
    assert r.status_code != 401


# ═══════════════════════════════════════════════════════════════════════════
# 6) macOS install command (2026-07-25) — mint response `commands` map
# ═══════════════════════════════════════════════════════════════════════════
def test_mint_response_carries_commands_map(client):
    """The mint response gets an additive `commands: {linux, macos, windows}`
    map built from the link's own `url` — a fresh macbook operator field
    report showed no mac guidance (downloaded-.sh Permission denied, then a
    chmod typo) before falling back to the working curl one-liner, so the
    console/API must offer all three explicitly."""
    link = _mint(client, label="mac test")
    assert set(link["commands"].keys()) == {"linux", "macos", "windows"}
    # linux and macos are the SAME command — one POSIX wrapper serves both:
    assert link["commands"]["macos"] == link["commands"]["linux"]
    assert link["commands"]["linux"] == f"curl -fsSL {link['url']}.sh | bash"
    # windows uses the .ps1 wrapper via the iex idiom:
    assert link["commands"]["windows"] == f"irm {link['url']}.ps1 | iex"


def test_mint_response_commands_additive_other_fields_unchanged(client):
    """Adding `commands` must not disturb any existing mint-response field."""
    link = _mint(client, label="additive check", scopes=["ml"], max_uses=3)
    assert link["label"] == "additive check"
    assert link["scopes"] == ["ml"]
    assert link["max_uses"] == 3
    assert link["uses_left"] == 3
    assert link["status"] == "active"
    assert link["url"].endswith(f"/agent/install/{link['link_id']}")
    assert "raw_key" not in link
    assert "commands" in link


# ═══════════════════════════════════════════════════════════════════════════
# 7) macOS zip installer download (2026-07-25)
# ═══════════════════════════════════════════════════════════════════════════
def test_mint_response_carries_downloads_map(client):
    """The mint response gets an additive `downloads: {macos_zip}` map built
    from the link's own `url` — the double-click counterpart to `commands`
    for a macOS field tester who downloaded the raw .sh from a browser and
    hit 'Permission denied' (browsers strip +x on download)."""
    link = _mint(client, label="zip mint")
    assert link["downloads"]["macos_zip"] == f"{link['url']}.zip"
    # additive: every existing field is still present and correct
    assert link["label"] == "zip mint"
    assert "commands" in link
    assert link["url"].endswith(f"/agent/install/{link['link_id']}")
    assert "raw_key" not in link


def test_route_zip_download_headers_and_contents(client):
    """GET .../<link_id>.zip serves a real zip archive with an already-
    executable single .command entry — the double-click installer."""
    link = _mint(client, label="zip contents")
    lid = link["link_id"]
    r = client.get(f"/agent/install/{lid}.zip")
    assert r.status_code == 200
    assert (r.headers.get("Content-Type") or "").startswith("application/zip")
    assert ('attachment; filename="hugpy-agent-installer.zip"'
            in (r.headers.get("Content-Disposition") or ""))

    zf = zipfile.ZipFile(io.BytesIO(r.data))
    assert zf.namelist() == ["Install hugpy Agent.command"]
    info = zf.getinfo("Install hugpy Agent.command")
    assert (info.external_attr >> 16) & 0o111          # executable bits set

    body = zf.read("Install hugpy Agent.command").decode()
    assert f"/agent/install/{lid}.sh" in body           # the curl target
    assert "[hugpy installer exited with status" in body
    assert "read -r _" in body                          # hold-open tail

    # no key material of any shape leaked into the archive — only the .sh
    # wrapper the .command calls does the templating, on a LATER fetch:
    import re
    assert not re.search(r"hp_[0-9a-f]{40}", body)


def test_route_zip_fetch_is_free(client):
    """LOAD-BEARING: the .zip fetch must NOT consume the link's one use — the
    whole point is a double-clickable archive that still costs exactly one
    use when the user later runs it (same as the .sh/.ps1 wrappers)."""
    link = _mint(client, label="zip free")
    lid = link["link_id"]
    before = il.get_link(lid)["uses_left"]
    r = client.get(f"/agent/install/{lid}.zip")
    assert r.status_code == 200
    after = il.get_link(lid)["uses_left"]
    assert after == before                               # untouched
    assert il.peek_active(lid) is True                    # still alive
    row = il.get_link(lid)
    assert any(d["kind"] == "zip" for d in row["downloads"])  # audited


def test_route_zip_dead_link_410(client):
    """A revoked link 410s for .zip too, same as .sh/.ps1."""
    link = _mint(client, label="zip revoked")
    lid = link["link_id"]
    r = client.delete(f"/agent/install-links/{lid}", headers=_op())
    assert r.status_code == 200
    r = client.get(f"/agent/install/{lid}.zip")
    assert r.status_code == 410
    assert "no longer valid" in r.get_data(as_text=True)


# ═══════════════════════════════════════════════════════════════════════════
# 8) macOS .pkg Installer package (tier 2, 2026-07-25)
#
# Tier 2 gives the field tester an installer WINDOW instead of a Terminal. It
# cannot be validated on this VM (no Mac), so the correctness bar is: the
# canonical bomutils recipe followed exactly, the xar structure verified with
# the real `xar -tf`, and EVERY load-bearing string in the generated
# postinstall asserted here.
#
# The build needs mkbom/xar/cpio, which prod central (host ae) does not have —
# the archive-shape tests skip when they are absent so the suite stays
# portable, while the DEGRADE tests (501 + `macos_pkg` omitted from the mint)
# monkeypatch the detection and therefore always run.
# ═══════════════════════════════════════════════════════════════════════════
import gzip
import shutil
import subprocess

_PKG_MISSING = agent_routes._pkg_missing_tools()
needs_pkg_tools = pytest.mark.skipif(
    bool(_PKG_MISSING),
    reason=f"macOS .pkg build tools absent on this host: {_PKG_MISSING}")


def _extract_pkg(pkg_bytes: bytes, dest: Path) -> Path:
    """`xar -xf` the package into dest, returning dest. Shells out to the real
    xar — the structure claim is only worth as much as the tool that reads it."""
    pkg = dest / "hugpy-agent-installer.pkg"
    pkg.write_bytes(pkg_bytes)
    subprocess.run([shutil.which("xar"), "-xf", str(pkg)], cwd=dest,
                   check=True, capture_output=True)
    return dest


def _postinstall_from(pkg_bytes: bytes, dest: Path) -> Path:
    """The `postinstall` file as Installer will see it: out of the xar, out of
    the Scripts cpio.gz, mode preserved."""
    _extract_pkg(pkg_bytes, dest)
    scripts = dest / "scripts_out"
    scripts.mkdir()
    raw = gzip.decompress((dest / "Scripts").read_bytes())
    subprocess.run([shutil.which("cpio"), "-idm"], cwd=scripts, input=raw,
                   check=True, capture_output=True)
    return scripts / "postinstall"


@needs_pkg_tools
def test_route_pkg_is_a_xar_archive_with_pkg_headers(client):
    """GET .../<link_id>.pkg serves a real flat package: xar magic + download
    headers. `application/octet-stream` + an explicit attachment disposition
    is the choice (documented in _serve_install_pkg) — every browser just saves
    it, nothing tries to unarchive it."""
    link = _mint(client, label="pkg headers")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    assert r.status_code == 200
    assert r.data[:4] == b"xar!"                      # xar magic, not a 501 body
    assert (r.headers.get("Content-Type") or "").startswith(
        "application/octet-stream")
    assert ('attachment; filename="hugpy-agent-installer.pkg"'
            in (r.headers.get("Content-Disposition") or ""))
    assert r.headers.get("Cache-Control") == "no-store"


@needs_pkg_tools
def test_route_pkg_archive_members(client, tmp_path):
    """The flat COMPONENT shape: PackageInfo at the xar root (no Distribution /
    product archive), plus Bom, Payload and the Scripts cpio."""
    link = _mint(client, label="pkg members")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    pkg = tmp_path / "hugpy-agent-installer.pkg"
    pkg.write_bytes(r.data)
    listing = subprocess.run([shutil.which("xar"), "-tf", str(pkg)],
                             check=True,
                             capture_output=True).stdout.decode().split()
    assert listing == ["PackageInfo", "Bom", "Payload", "Scripts"]


@needs_pkg_tools
def test_pkg_packageinfo_declares_zero_payload_and_the_postinstall(client, tmp_path):
    """PackageInfo is what makes Installer run a scripts-only package."""
    import xml.etree.ElementTree as ET
    link = _mint(client, label="pkg packageinfo")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    root = ET.fromstring(
        (_extract_pkg(r.data, tmp_path) / "PackageInfo").read_text())
    assert root.tag == "pkg-info"
    assert root.get("identifier") == "ai.hugpy.agent.installer"
    assert root.get("version")                        # some version, declared
    assert root.get("install-location") == "/"
    assert root.get("auth") == "root"
    payload = root.find("payload")
    assert payload.get("numberOfFiles") == "0"        # scripts-only
    assert payload.get("installKBytes") == "0"
    assert root.find("scripts/postinstall").get("file") == "postinstall"


@needs_pkg_tools
def test_pkg_payload_is_empty_and_bom_is_present(client, tmp_path):
    """Zero payload for real: the Payload cpio has NO entries — not even a "."
    that, with install-location="/", would be an ownership change applied to
    the root of the target volume. The Bom is still there (Installer expects
    the member) and non-empty."""
    link = _mint(client, label="pkg payload")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    d = _extract_pkg(r.data, tmp_path)
    entries = subprocess.run([shutil.which("cpio"), "-it"],
                             input=gzip.decompress((d / "Payload").read_bytes()),
                             capture_output=True).stdout.decode().split()
    assert entries == []
    assert (d / "Bom").stat().st_size > 0


@needs_pkg_tools
def test_pkg_postinstall_is_executable_and_valid_shell(client, tmp_path):
    """Mode 0755 must survive the cpio (Installer runs the file), and the
    script must actually parse — a .pkg shows the user no output, so a quoting
    slip would be invisible in the field."""
    link = _mint(client, label="pkg script mode")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    p = _postinstall_from(r.data, tmp_path)
    assert p.stat().st_mode & 0o111                    # executable
    assert oct(p.stat().st_mode)[-3:] == "755"
    assert subprocess.run(["/bin/sh", "-n", str(p)],
                          capture_output=True).returncode == 0


@needs_pkg_tools
def test_pkg_postinstall_installs_for_the_console_user_not_root(client, tmp_path):
    """Installer runs postinstall as ROOT. Every load-bearing string of the
    "install for the logged-in user instead" derivation is asserted here."""
    link = _mint(client, label="pkg console user")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    body = _postinstall_from(r.data, tmp_path).read_text()
    assert "/usr/bin/stat -f%Su /dev/console" in body      # who is at the Mac
    assert "/usr/bin/dscl . -read \"/Users/$u\" NFSHomeDirectory" in body
    assert 'home="/Users/$u"' in body                      # documented fallback
    # the install runs AS that user, with THEIR home:
    assert '/usr/bin/sudo -u "$u" -H' in body
    # …and never for root: a root/absent console user refuses instead.
    assert 'no logged-in user found' in body


@needs_pkg_tools
def test_pkg_postinstall_runs_this_links_one_liner(client, tmp_path):
    """The install itself is the SAME one-liner the console offers / the tier-1
    .command runs, for THIS link — plus --no-launch (an Installer package has
    no terminal for the interactive console TUI). No divergent URL."""
    link = _mint(client, label="pkg one-liner")
    lid = link["link_id"]
    r = client.get(f"/agent/install/{lid}.pkg")
    body = _postinstall_from(r.data, tmp_path).read_text()
    expected = f"{link['commands']['macos']} -s -- --no-launch"
    assert f"ONE_LINER='{expected}'" in body
    assert f"/agent/install/{lid}.sh" in body              # the wrapper, the use
    # a pty, because the shared .sh wrapper re-attaches stdin to /dev/tty and a
    # postinstall has no controlling terminal (open would fail ENXIO):
    assert "/usr/bin/script -q /dev/null /bin/bash" in body
    # no key material of any shape: the key is templated into the .py on the
    # LATER fetch the one-liner performs, never into the package.
    import re
    assert not re.search(r"hp_[0-9a-f]{40}", body)


@needs_pkg_tools
def test_pkg_postinstall_tees_everything_to_a_discoverable_log(client, tmp_path):
    """DIAGNOSTICS MITIGATION: a .pkg hides all output, and terminal output is
    what diagnosed every field bug this installer has had. So the log is the
    feature — created, chown'd to the user, appended, and named in the console
    copy (~/hugpy-agent/install.log)."""
    link = _mint(client, label="pkg log")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    body = _postinstall_from(r.data, tmp_path).read_text()
    assert 'ws="$home/hugpy-agent"' in body                # the workspace dir
    assert 'LOG="$ws/install.log"' in body
    assert 'mkdir -p "$ws"' in body
    assert 'chown "$u" "$ws"' in body                      # the USER's, not root's
    assert 'chown "$u" "$LOG"' in body
    assert 'tee -a "$LOG"' in body                         # append, don't truncate


@needs_pkg_tools
def test_pkg_postinstall_fails_loudly_not_falsely(client, tmp_path):
    """Exit NONZERO on failure so Installer reports a failure instead of a
    false success — including the sneaky one: `curl … | bash` reports BASH's
    status, so a failed curl feeds bash an empty script and looks like a clean
    install. pipefail closes that, and the install's real status is read from a
    file rather than trusted to `script`'s exit code."""
    link = _mint(client, label="pkg failure path")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    body = _postinstall_from(r.data, tmp_path).read_text()
    # the runner the postinstall writes gets pipefail, unconditionally…
    assert "echo 'set -o pipefail'" in body
    # …and NOT the `|| true` form: on a shell without pipefail a failed `set`
    # is fatal to the whole script (dash exits 2, measured), which a .pkg would
    # hide. Hence an explicit /bin/bash for the runner.
    assert "echo 'set -o pipefail 2>/dev/null || true'" not in body
    assert '/bin/bash "$run"' in body
    assert 'echo "$?" > "$0.rc"' in body                   # rc from the install
    assert 'rc=$(cat "$run.rc"' in body
    assert 'exit "$rc"' in body                            # propagate it
    assert "FAILED (status $rc)" in body


@needs_pkg_tools
def test_pkg_build_leaks_no_temp_files(client):
    """The package is built in a mkdtemp and streamed as bytes; nothing may be
    left behind (this route can be hit repeatedly and freely)."""
    import glob
    pattern = os.path.join(tempfile.gettempdir(), "hugpy-agent-pkg-*")
    link = _mint(client, label="pkg no leak")
    before = set(glob.glob(pattern))
    for _ in range(3):
        assert client.get(f"/agent/install/{link['link_id']}.pkg").status_code == 200
    assert set(glob.glob(pattern)) == before


@needs_pkg_tools
def test_route_pkg_fetch_is_free(client):
    """LOAD-BEARING: the .pkg fetch must NOT consume the link's use. The
    package's postinstall fetches the .sh wrapper (which fetches the .py) at
    INSTALL time — a use-eating download would consume the only use and break
    the very install this file exists to enable."""
    link = _mint(client, label="pkg free")
    lid = link["link_id"]
    before = il.get_link(lid)["uses_left"]
    assert client.get(f"/agent/install/{lid}.pkg").status_code == 200
    assert il.get_link(lid)["uses_left"] == before          # untouched
    assert il.peek_active(lid) is True                      # still alive
    assert any(d["kind"] == "pkg" for d in il.get_link(lid)["downloads"])
    # …and the .py it later fetches still works, on the use it was saving:
    assert client.get(f"/agent/install/{lid}").status_code == 200
    assert il.get_link(lid)["uses_left"] == before - 1


@needs_pkg_tools
def test_route_pkg_dead_link_410(client):
    """A revoked link 410s for .pkg too, same as .sh/.ps1/.zip."""
    link = _mint(client, label="pkg revoked")
    lid = link["link_id"]
    assert client.delete(f"/agent/install-links/{lid}", headers=_op()).status_code == 200
    r = client.get(f"/agent/install/{lid}.pkg")
    assert r.status_code == 410
    assert "no longer valid" in r.get_data(as_text=True)


@needs_pkg_tools
def test_mint_offers_macos_pkg_where_central_can_build_one(client):
    """`downloads` gains `macos_pkg` alongside `macos_zip` — additive."""
    link = _mint(client, label="pkg mint")
    assert link["downloads"]["macos_pkg"] == f"{link['url']}.pkg"
    assert link["downloads"]["macos_zip"] == f"{link['url']}.zip"
    assert "commands" in link and "raw_key" not in link


# ── prod parity: the degrade path (always runs — no tools needed) ──────────
def test_route_pkg_501_when_the_toolchain_is_absent(client, monkeypatch):
    """PROD PARITY: prod central (ae) has no mkbom/xar. The route must degrade
    honestly — 501 naming the missing binaries, never a 500 and never a
    truncated/corrupt file."""
    monkeypatch.setattr(agent_routes, "_pkg_missing_tools",
                        lambda: ["mkbom", "xar"])
    link = _mint(client, label="pkg no tools")
    r = client.get(f"/agent/install/{link['link_id']}.pkg")
    assert r.status_code == 501
    text = r.get_data(as_text=True)
    assert "mkbom" in text and "xar" in text            # names both binaries
    assert ".zip" in text                               # points at what works
    assert r.data[:4] != b"xar!"                        # no half-built package


def test_route_pkg_dead_link_still_410s_without_the_toolchain(client, monkeypatch):
    """Link validity is checked BEFORE the toolchain, so a revoked link 410s
    identically on every deployment."""
    link = _mint(client, label="pkg dead no tools")
    lid = link["link_id"]
    assert client.delete(f"/agent/install-links/{lid}", headers=_op()).status_code == 200
    monkeypatch.setattr(agent_routes, "_pkg_missing_tools", lambda: ["mkbom", "xar"])
    assert client.get(f"/agent/install/{lid}.pkg").status_code == 410


def test_pkg_501_does_not_consume_or_audit_a_fetch(client, monkeypatch):
    """A 501 delivered no installer, so it must neither decrement the link nor
    leave an audit line claiming a wrapper was served."""
    link = _mint(client, label="pkg 501 free")
    lid = link["link_id"]
    before = il.get_link(lid)["uses_left"]
    monkeypatch.setattr(agent_routes, "_pkg_missing_tools", lambda: ["mkbom", "xar"])
    assert client.get(f"/agent/install/{lid}.pkg").status_code == 501
    assert il.get_link(lid)["uses_left"] == before
    assert not any(d["kind"] == "pkg" for d in il.get_link(lid)["downloads"])


def test_mint_omits_macos_pkg_when_the_toolchain_is_absent(client, monkeypatch):
    """The console must not be offered a button that cannot work: no tools ->
    no `macos_pkg` key at all (the .zip and the commands are untouched)."""
    monkeypatch.setattr(agent_routes, "_pkg_missing_tools",
                        lambda: ["mkbom", "xar"])
    link = _mint(client, label="pkg omitted")
    assert "macos_pkg" not in link["downloads"]
    assert link["downloads"]["macos_zip"] == f"{link['url']}.zip"
    assert link["commands"]["macos"] == f"curl -fsSL {link['url']}.sh | bash"


def test_pkg_tool_detection_is_probed_once_and_cached(monkeypatch):
    """Detection must not shell out per request. The probe runs once per
    process and the result is cached."""
    monkeypatch.setattr(agent_routes, "_pkg_missing_tools_cache", None)
    calls = []

    def counting_which(binary):
        calls.append(binary)
        return "/usr/bin/" + binary
    monkeypatch.setattr(shutil, "which", counting_which)
    first = agent_routes._pkg_missing_tools()
    n = len(calls)
    assert n == len(agent_routes._PKG_TOOL_BINARIES)
    for _ in range(5):
        assert agent_routes._pkg_missing_tools() == first
    assert len(calls) == n                              # no further probing


def test_install_commands_posix_args_is_additive(client):
    """The .pkg passes --no-launch through `bash -s --` (the only way to give
    the `curl | bash` idiom arguments). Default "" must leave the console's
    copy-paste commands byte-identical."""
    url = "https://example.test/api/agent/install/abc"
    plain = agent_routes._install_commands(url)
    assert plain["macos"] == f"curl -fsSL {url}.sh | bash"
    assert plain["linux"] == plain["macos"]
    with_args = agent_routes._install_commands(url, posix_args="--no-launch")
    assert with_args["macos"] == f"curl -fsSL {url}.sh | bash -s -- --no-launch"
    assert with_args["windows"] == plain["windows"]      # windows untouched
