"""k9 — video SHARE-LINK feature: the key store, the /video gate's share seam,
the STRUCTURAL hardening (a share key opens /video but can NEVER mint another key
or reach any console/operator route), and job attribution.

This is the SIBLING of test_video_gate.py (which pins the k8 gate + its 25
checks). Those 25 stay green untouched — the k8 test overrides the share seam in
every case, so this real implementation never perturbs it. Here we exercise the
real implementation instead.
"""
import importlib
import os
import tempfile
import time

import pytest
from flask import Flask

vsk = importlib.import_module("hugpy_server.app.functions.imports.utils.video_share_keys")
ak = importlib.import_module("hugpy_server.app.functions.imports.utils.api_keys")
va = importlib.import_module("hugpy_server.app.video_auth")
oa = importlib.import_module("hugpy_server.app.operator_auth")


@pytest.fixture(scope="module")
def share_store():
    """Point the share-key store at a throwaway temp file (no dependency on the
    real manifest/settings resolution — the store idiom is what we're testing)."""
    store_file = os.path.join(
        tempfile.mkdtemp(prefix="hugpy-share-store-"), "video_share_keys.json")
    orig = vsk._store_path
    vsk._store_path = lambda: store_file
    try:
        yield store_file
    finally:
        vsk._store_path = orig


# --------------------------------------------------------------------------- #
# 1) the key store: mint / verify / expiry / revoke / wrong-category
# --------------------------------------------------------------------------- #
def test_mint_and_verify(share_store):
    minted = vsk.create_share_key(label="Alex — review", ttl_days=30)
    assert minted["key"].startswith("hpv_") and len(minted["key"]) > 20, \
        "mint: returns the full key ONCE (hpv_ prefix)"
    assert "hash" not in minted, "mint: hash is NOT returned"
    assert minted["id"] and minted["label"] == "Alex — review" and minted["expires_at"]

    good = minted["key"]
    assert vsk.verify_share_key(good) == minted["id"], "a good key resolves to its key_id"
    assert vsk.share_principal(good) == f"share:{minted['id']}"

    assert vsk.verify_share_key("hpv_deadbeef") is None, "garbage => None"
    assert vsk.verify_share_key("") is None
    assert vsk.verify_share_key(None) is None
    # Wrong category: a /v1-style api key (hp_) is NOT in the share store.
    assert vsk.verify_share_key("hp_" + "a" * 40) is None


def test_expired_key_is_rejected(share_store):
    # Expired key: mint with a ttl and force expires_at into the past.
    exp = vsk.create_share_key(label="stale", ttl_days=1)
    data = vsk._load()
    data["keys"][exp["id"]]["expires_at"] = time.time() - 10
    vsk._save(data)
    assert vsk.verify_share_key(exp["key"]) is None, "an EXPIRED key => None"
    assert vsk.share_principal(exp["key"]) is None
    # List: the expired key is present but flagged expired.
    assert any(k["id"] == exp["id"] and k["expired"] for k in vsk.list_share_keys())


def test_non_expiring_key(share_store):
    forever = vsk.create_share_key(label="forever", ttl_days=0)
    assert forever["expires_at"] is None, "ttl_days<=0 => no expiry"
    assert vsk.verify_share_key(forever["key"]) == forever["id"]
    assert forever["id"] in {k["id"] for k in vsk.list_share_keys()}, "active keys listed"


def test_revoke_and_list(share_store):
    minted = vsk.create_share_key(label="revoke me", ttl_days=30)
    assert vsk.revoke_share_key(minted["id"]) is True, "revoke: known id => True"
    assert vsk.verify_share_key(minted["key"]) is None, "a REVOKED key => None"
    assert vsk.revoke_share_key("nope") is False, "revoke: unknown id => False"

    # List: non-revoked only, never leaks the hash.
    listed = vsk.list_share_keys()
    assert minted["id"] not in {k["id"] for k in listed}, "revoked key is excluded"
    assert all("hash" not in k for k in listed), "list never leaks the hash"


def test_stores_are_distinct_files(share_store):
    # The two categories live in DIFFERENT files (structural isolation, not a tag).
    assert os.path.basename(ak._store_path()) == "api_keys.json"
    assert os.path.basename(vsk._store_path()) == "video_share_keys.json"


# --------------------------------------------------------------------------- #
# 2) the /video gate's share seam — the three credential carriers
# --------------------------------------------------------------------------- #
@pytest.fixture
def live(share_store):
    return vsk.create_share_key(label="seam", ttl_days=30)


def test_share_seam_credential_carriers(live):
    from flask import request as _rq
    app = Flask(__name__)
    with app.test_request_context(f"/video/x?share={live['key']}"):
        assert va._video_share_principal(_rq) == f"share:{live['id']}", "?share= query"
    with app.test_request_context("/video/x", headers={"X-Video-Share": live["key"]}):
        assert va._video_share_principal(_rq) == f"share:{live['id']}", "X-Video-Share header"
    with app.test_request_context(
            "/video/x", headers={"Authorization": f"Bearer {live['key']}"}):
        assert va._video_share_principal(_rq) == f"share:{live['id']}", "Bearer hpv_"
    with app.test_request_context("/video/x"):
        assert va._video_share_principal(_rq) is None, "no credential => None"
    with app.test_request_context("/video/x", headers={"X-Video-Share": "hpv_bogus"}):
        assert va._video_share_principal(_rq) is None, "an invalid share key => None"


# --------------------------------------------------------------------------- #
# 3) THE STRUCTURAL PIN (extended): a share key opens /video, but can NEVER
#    reach the mint route (no key-minting-by-key) or any console/operator route.
# --------------------------------------------------------------------------- #
class _ApiPrefixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app
    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def build_app():
    """Mirrors production wiring: /api-strip -> operator gate -> video gate, with
    the console-gated /keys, the OPERATOR-gated /keys/video-share mint route, a
    /video data route, and the SPA shell catch-all."""
    a = Flask(__name__)

    @a.route("/keys", methods=["GET"])
    def _keys():
        return "keys", 200

    @a.route("/keys/video-share", methods=["GET", "POST"])
    def _mint():
        return "minted", 200  # only reachable past the OPERATOR gate

    @a.route("/video/studio/clips", methods=["GET"])
    def _clips():
        return "clips", 200

    def _shell(asset=""):
        return "<html>shell</html>", 200
    a.add_url_rule("/", endpoint="_hugpy_ui", view_func=_shell, defaults={"asset": ""})
    a.add_url_rule("/<path:asset>", endpoint="_hugpy_ui", view_func=_shell)

    oa.install_operator_gate(a)
    va.install_video_gate(a)
    a.wsgi_app = _ApiPrefixMiddleware(a.wsgi_app)
    return a


@pytest.fixture
def external_mode(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    oa._SESSION_CACHE.clear()
    yield monkeypatch
    oa._SESSION_CACHE.clear()


def test_share_key_opens_video_but_never_mints_or_reaches_console(live, external_mode):
    external_mode.setattr(oa, "_validate_session_external", lambda: False)  # NO console session
    c = build_app().test_client()
    hdr = {"X-Video-Share": live["key"]}

    # The share key opens the /video surface.
    assert c.get("/api/video/studio/clips", headers=hdr).status_code == 200
    # ...but is powerless against the mint route (operator-gated + off the /video
    # surface) — this is the "no key-minting-by-key" hardening.
    assert c.get("/api/keys/video-share", headers=hdr).status_code == 401
    assert c.post("/api/keys/video-share", headers=hdr).status_code == 401
    # ...and never satisfies the console gate at large.
    assert c.get("/api/keys", headers=hdr).status_code == 401

    # A bare anon caller is denied the mint route too (baseline).
    assert c.post("/api/keys/video-share").status_code == 401


def test_console_session_opens_video_and_mint(external_mode):
    # A console session opens BOTH the video surface AND the mint route.
    external_mode.setattr(oa, "_validate_session_external", lambda: True)
    oa._SESSION_CACHE.clear()
    c = build_app().test_client()
    assert c.post("/api/keys/video-share").status_code == 200, "operator can mint"
    assert c.get("/api/video/studio/clips").status_code == 200


# --------------------------------------------------------------------------- #
# 4) attribution: a share principal lands on the job, end to end.
# --------------------------------------------------------------------------- #
def test_media_bus_persists_principal(monkeypatch, tmp_path):
    # media_bus persists the principal on the job row + carries it to the bridge.
    import hugpy_video.intel.media_bus as mb
    monkeypatch.setattr(mb, "DB_PATH", os.path.join(tmp_path, "media_jobs.db"))
    monkeypatch.setattr(mb, "_initialized", False)
    monkeypatch.setattr(mb, "serialize_spec", lambda name, spec: "{}")  # bypass real spec serialization
    jid = mb.enqueue("crop", object(), principal="share:xyz789")
    conn = mb._connect()
    try:
        row = conn.execute(
            "SELECT principal FROM media_jobs WHERE job_id=?", (jid,)).fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == "share:xyz789"


def test_bridge_carries_principal_onto_comms_job():
    # the bus->JobStore bridge carries the principal onto the comms Job that
    # GET /llm/jobs reads (running + terminal, in the process that runs the job).
    from hugpy_control.jobs import job_store
    job_bridge = importlib.import_module("hugpy_video.intel.job_bridge")

    job_bridge.on_running("attrib-job-1", "studio_i2v", worker="w", principal="share:xyz789")
    j = job_store.get("attrib-job-1")
    assert j is not None and j.principal == "share:xyz789", "on_running stamps principal"
    assert j.to_dict().get("principal") == "share:xyz789", "principal on the /llm/jobs wire dict"

    job_bridge.on_terminal("attrib-job-1", "studio_i2v", "done", principal="share:xyz789")
    j2 = job_store.get("attrib-job-1")
    assert j2 is not None and j2.principal == "share:xyz789", "survives to the terminal record"

    # A None principal (unattributed job) must never blank an existing attribution.
    job_bridge.on_running("attrib-job-1", "studio_i2v", principal=None)
    assert job_store.get("attrib-job-1").principal == "share:xyz789"
