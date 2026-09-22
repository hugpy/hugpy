"""MEMBER TIER (2026-08-06) — roles, the three gates, and artifact ownership.

THE BUG THIS PINS
-----------------
``operator_authenticated()`` treated ANY valid central session as a full
operator: the upstream ``/me`` payload was fetched and thrown away, so a
registered member could mint API keys and drive worker admission. And the whole
Studio/Media plane (``/media``, ``/ml``, ``/uploads``, ``/session``, ``/chat``)
was gated by nothing at all — anonymous 200s, proven live on :7002.

WHAT IS PINNED HERE
  1. ROLES — operator (token / open-mode / is_admin session), member (approved +
     hugpy|clownworld site grant), anonymous (everything else, incl. an auth
     outage: fail closed).
  2. CONSOLE GATE (operator_auth) — operator allow, member 403 with the exact
     body {"error": "forbidden: read-only member access"}, anonymous 401. Plus
     the mutations added to _SENSITIVE today (models/jobs/repos/phone-brick).
  3. MEMBER GATE (member_auth) — the studio/media surface matcher, anonymous
     401 (redirect for a browser shell), member/operator allowed, OPTIONS free.
  4. OWNERSHIP — media_bus stores/filters `owner`; NULL-owner legacy rows are
     admin-only; the uploads namespace mapping is one shared helper.
  5. INSTALL LINKS — member-callable (amendment): own-links-only listing,
     operator-or-creator delete, and the member scope CLAMP.

No network: the upstream /me is stubbed at operator_auth._fetch_me.

Run: cd .../abstract_hugpy_dev && venv/bin/python -m pytest tests/test_member_tier.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-member-tier-test-"))
os.environ.setdefault("HUGPY_COMMS_DB", os.path.join(
    tempfile.mkdtemp(prefix="hugpy-member-tier-comms-"), "comms.db"))
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

import pytest


@pytest.fixture(autouse=True)
def _external_mode(monkeypatch):
    """Every test here runs the member gate in external mode (set per test,
    never at import, so it cannot leak into the rest of the session)."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")

from flask import Flask

from hugpy_server.app import operator_auth as oa
from hugpy_server.app import member_auth as ma
from hugpy_server.app import video_auth as va


# ── principals the stubbed /me returns ──────────────────────────────────────
ADMIN = {"username": "root", "email": "r@x", "is_admin": True,
         "status": "approved", "sites": ["hugpy"]}
MEMBER = {"username": "alice", "email": "a@x", "is_admin": False,
          "status": "approved", "sites": ["clownworld"]}
MEMBER_HUGPY = {"username": "bob", "is_admin": False,
                "status": "approved", "sites": ["hugpy", "other"]}
PENDING = {"username": "carol", "is_admin": False,
           "status": "pending", "sites": ["hugpy"]}
OUTSIDER = {"username": "dave", "is_admin": False,
            "status": "approved", "sites": ["someotherapp"]}


@pytest.fixture(autouse=True)
def clean_caches():
    oa._clear_session_caches()
    yield
    oa._clear_session_caches()


def _stub_me(monkeypatch, payload, ok=True, outage=False):
    """Stub the ONE upstream call. ``outage=True`` makes _fetch_me report an
    unreachable auth service (its None contract)."""
    def _fake():
        if outage:
            return None
        return (ok, oa._normalize_principal(payload) if payload else None)
    monkeypatch.setattr(oa, "_fetch_me", _fake)


COOKIE = {"Cookie": "auth_session=abc123"}


def _client(app):
    """A test client carrying a session cookie. NOT a `headers={"Cookie": …}`
    call: werkzeug's client owns the Cookie header from its own jar and drops a
    hand-set one, so the gate would see no cookie at all and every case would
    read as anonymous."""
    c = app.test_client()
    c.set_cookie("auth_session", "abc123")
    return c


# ═══════════════════════════════════════════════════════════════════════════
# 1) ROLES
# ═══════════════════════════════════════════════════════════════════════════
def _role_with(monkeypatch, payload, **kw):
    _stub_me(monkeypatch, payload, **kw)
    app = Flask(__name__)
    with app.test_request_context("/keys", method="POST", headers=COOKIE):
        return oa.principal_role()


def test_role_admin_session_is_operator(monkeypatch):
    assert _role_with(monkeypatch, ADMIN) == "operator"


def test_role_approved_member_sites(monkeypatch):
    assert _role_with(monkeypatch, MEMBER) == "member"          # clownworld
    assert _role_with(monkeypatch, MEMBER_HUGPY) == "member"    # hugpy


def test_role_unapproved_is_anonymous(monkeypatch):
    assert _role_with(monkeypatch, PENDING) == "anonymous"


def test_role_wrong_site_is_anonymous(monkeypatch):
    assert _role_with(monkeypatch, OUTSIDER) == "anonymous"


def test_role_no_cookie_is_anonymous(monkeypatch):
    _stub_me(monkeypatch, MEMBER)
    app = Flask(__name__)
    with app.test_request_context("/keys", method="POST"):
        assert oa.principal_role() == "anonymous"
        assert oa.current_principal() is None


def test_role_auth_outage_fails_closed(monkeypatch):
    assert _role_with(monkeypatch, None, outage=True) == "anonymous"


def test_operator_token_is_operator(monkeypatch):
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    _stub_me(monkeypatch, None)
    app = Flask(__name__)
    with app.test_request_context("/keys", method="POST",
                                  headers={"X-Operator-Token": "s3cret"}):
        assert oa.principal_role() == "operator"
        assert oa.operator_authenticated() is True


def test_open_mode_without_token_is_operator(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    app = Flask(__name__)
    with app.test_request_context("/keys", method="POST"):
        assert oa.principal_role() == "operator"


def test_member_is_not_an_operator(monkeypatch):
    _stub_me(monkeypatch, MEMBER)
    app = Flask(__name__)
    with app.test_request_context("/keys", method="POST", headers=COOKIE):
        assert oa.operator_authenticated() is False   # THE BUG, pinned
        assert oa.member_authenticated() is True
        assert oa.principal_username() == "alice"


def test_principal_cached_per_request(monkeypatch):
    calls = []

    def _fake():
        calls.append(1)
        return (True, oa._normalize_principal(MEMBER))
    monkeypatch.setattr(oa, "_fetch_me", _fake)
    app = Flask(__name__)
    with app.test_request_context("/keys", headers=COOKIE):
        for _ in range(5):
            oa.current_principal()
            oa.principal_role()
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2) THE CONSOLE GATE — operator allow / member 403 / anonymous 401
# ═══════════════════════════════════════════════════════════════════════════
def _console_app():
    app = Flask(__name__)

    @app.route("/keys", methods=["GET", "POST"])
    def _keys():
        return "keys", 200

    @app.route("/version")
    def _version():
        return "v", 200

    @app.route("/models/<key>/download", methods=["POST"])
    def _dl(key):
        return "dl", 200

    oa.install_operator_gate(app)
    return app


def test_console_gate_member_gets_403(monkeypatch):
    _stub_me(monkeypatch, MEMBER)
    c = _client(_console_app())
    r = c.post("/keys")
    assert r.status_code == 403
    assert r.get_json() == {"error": "forbidden: read-only member access"}


def test_console_gate_anonymous_gets_401(monkeypatch):
    _stub_me(monkeypatch, None, ok=False)
    c = _client(_console_app())
    assert c.post("/keys").status_code == 401


def test_console_gate_admin_allowed(monkeypatch):
    _stub_me(monkeypatch, ADMIN)
    c = _client(_console_app())
    assert c.post("/keys").status_code == 200


def test_console_gate_open_route_untouched(monkeypatch):
    _stub_me(monkeypatch, None, ok=False)
    c = _client(_console_app())
    assert c.get("/version").status_code == 200


def test_new_sensitive_mutations_cover_the_console_plane():
    def _s(method, path):
        return any(method in m and rx.match(path) for m, rx in oa._SENSITIVE)
    assert _s("POST", "/models/llama-3/download")
    assert _s("DELETE", "/models/llama-3")
    assert _s("POST", "/models/llama-3/prune")
    assert _s("POST", "/models/llama-3/media")
    assert _s("POST", "/models/llama-3/media-default")
    assert _s("POST", "/models/reclassify-images")
    assert _s("POST", "/jobs/j1/cancel")
    assert _s("POST", "/jobs/j1/retry")
    assert _s("POST", "/llm/repos/download")
    assert _s("POST", "/phone-brick/run")
    assert _s("POST", "/phone-brick/runs/r1/cancel")
    assert _s("DELETE", "/phone-brick/phones/p1")
    # …and the STUDIO/MEDIA plane is deliberately NOT operator-only:
    assert not _s("POST", "/uploads")
    assert not _s("POST", "/session/file")
    assert not _s("DELETE", "/session/file")
    assert not _s("POST", "/media/analyze")
    assert not _s("GET", "/ml")
    assert not _s("POST", "/chat/stream")
    assert not _s("POST", "/video/studio/i2v")
    assert not _s("GET", "/video/studio/clips")
    # reads on the console plane stay open
    assert not _s("GET", "/models/llama-3")
    assert not _s("GET", "/jobs")


# ═══════════════════════════════════════════════════════════════════════════
# 3) THE MEMBER GATE — the studio/media plane
# ═══════════════════════════════════════════════════════════════════════════
def _plane_app():
    app = Flask(__name__)
    for rule, ep in (("/media/analyze", "_m"), ("/ml", "_ml"),
                     ("/uploads", "_up"), ("/session/ping", "_sp"),
                     ("/chat/stream", "_cs")):
        app.add_url_rule(rule, ep, lambda: ("ok", 200),
                         methods=["GET", "POST", "OPTIONS"])

    @app.route("/keys", methods=["GET", "POST"])
    def _keys():
        return "keys", 200

    def _shell(asset=""):
        return "<html>", 200
    app.add_url_rule("/", endpoint="_hugpy_ui", view_func=_shell,
                     defaults={"asset": ""})
    app.add_url_rule("/<path:asset>", endpoint="_hugpy_ui", view_func=_shell)

    oa.install_operator_gate(app)
    ma.install_member_gate(app)
    return app


@pytest.mark.parametrize("path,method", [
    ("/media/analyze", "POST"), ("/ml", "GET"), ("/uploads", "POST"),
    ("/session/ping", "POST"), ("/chat/stream", "POST"),
])
def test_member_surface_matches(path, method):
    app = Flask(__name__)
    with app.test_request_context(path, method=method):
        assert ma._is_member_surface() is True
    with app.test_request_context("/api" + path, method=method):
        assert ma._is_member_surface() is True   # after the /api strip


@pytest.mark.parametrize("path", ["/keys", "/version", "/v1/chat/completions",
                                  "/llm/workers", "/mlt", "/mediafoo"])
def test_non_member_surface_untouched(path):
    app = Flask(__name__)
    with app.test_request_context(path):
        assert ma._is_member_surface() is False


def test_plane_anonymous_is_401(monkeypatch):
    _stub_me(monkeypatch, None, ok=False)
    c = _client(_plane_app())
    for path, method in (("/media/analyze", "post"), ("/ml", "get"),
                         ("/uploads", "post"), ("/session/ping", "post"),
                         ("/chat/stream", "post")):
        r = getattr(c, method)(path, headers=COOKIE)
        assert r.status_code == 401, path
        assert "error" in r.get_json()


def test_plane_member_allowed(monkeypatch):
    _stub_me(monkeypatch, MEMBER)
    c = _client(_plane_app())
    assert c.get("/ml").status_code == 200
    assert c.post("/uploads").status_code == 200
    assert c.post("/chat/stream").status_code == 200
    # …while the console plane still refuses that member:
    assert c.post("/keys").status_code == 403


def test_plane_admin_allowed(monkeypatch):
    _stub_me(monkeypatch, ADMIN)
    c = _client(_plane_app())
    assert c.get("/ml").status_code == 200
    assert c.post("/keys").status_code == 200


def test_plane_options_never_blocked(monkeypatch):
    _stub_me(monkeypatch, None, ok=False)
    c = _client(_plane_app())
    assert c.options("/uploads").status_code != 401


def test_plane_browser_shell_redirects(monkeypatch):
    _stub_me(monkeypatch, None, ok=False)
    c = _client(_plane_app())
    r = c.get("/media/", headers={"Sec-Fetch-Dest": "document"})
    assert r.status_code == 302
    assert (r.headers.get("Location") or "").endswith("/")


def test_plane_api_key_is_accepted(monkeypatch):
    """The M2M seam: a valid product key reaches the plane (and NEVER the
    console — that gate does not consult keys)."""
    _stub_me(monkeypatch, None, ok=False)
    monkeypatch.setattr(ma, "_api_key_principal", lambda request: True)
    c = _client(_plane_app())
    assert c.get("/ml", headers={"X-API-Key": "hp_x"}).status_code == 200
    assert c.post("/keys", headers={"X-API-Key": "hp_x"}).status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
# 3b) THE VIDEO GATE now admits a member (and still never the console)
# ═══════════════════════════════════════════════════════════════════════════
def test_video_gate_admits_member(monkeypatch):
    _stub_me(monkeypatch, MEMBER)
    app = Flask(__name__)

    @app.route("/video/studio/clips")
    def _clips():
        return "clips", 200

    @app.route("/keys", methods=["POST"])
    def _keys():
        return "keys", 200
    oa.install_operator_gate(app)
    va.install_video_gate(app)
    c = _client(app)
    assert c.get("/video/studio/clips").status_code == 200
    assert c.post("/keys").status_code == 403


# ═══════════════════════════════════════════════════════════════════════════
# 4) ARTIFACT OWNERSHIP — media_bus
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def bus(tmp_path, monkeypatch):
    from hugpy_video.intel import media_bus as mb
    monkeypatch.setattr(mb, "DB_PATH", str(tmp_path / "media_jobs.db"))
    monkeypatch.setattr(mb, "_initialized", False)
    mb._ensure_db()
    return mb


def _fake_spec(mb, monkeypatch, name="crop"):
    """Insert through enqueue without dragging a real spec in: stub the two
    seams enqueue uses (registry membership + serialization)."""
    monkeypatch.setitem(mb.JOB_REGISTRY, name, object())
    monkeypatch.setattr(mb, "serialize_spec", lambda n, s: '{"stub": true}')
    monkeypatch.setattr(mb, "_bridge", lambda *a, **k: None)


def test_owner_column_migration_is_idempotent(bus):
    import sqlite3
    conn = sqlite3.connect(bus.DB_PATH)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(media_jobs)")}
    idx = {r[1] for r in conn.execute("PRAGMA index_list(media_jobs)")}
    conn.close()
    assert "owner" in cols
    assert "idx_media_jobs_owner" in idx
    bus._initialized = False
    bus._ensure_db()          # re-run: must not raise
    bus._initialized = False
    bus._ensure_db()


def test_enqueue_stores_owner_and_owner_of(bus, monkeypatch):
    _fake_spec(bus, monkeypatch)
    mine = bus.enqueue("crop", object(), principal="user:alice", owner="alice")
    theirs = bus.enqueue("crop", object(), principal="user:bob", owner="bob")
    legacy = bus.enqueue("crop", object())
    assert bus.owner_of(mine) == (True, "alice")
    assert bus.owner_of(theirs) == (True, "bob")
    assert bus.owner_of(legacy) == (True, None)   # NULL owner -> admin-only
    assert bus.owner_of("nope") == (False, None)
    assert bus.get(mine)["owner"] == "alice"


def test_list_jobs_owner_scope(bus, monkeypatch):
    _fake_spec(bus, monkeypatch)
    mine = bus.enqueue("crop", object(), owner="alice")
    bus.enqueue("crop", object(), owner="bob")
    bus.enqueue("crop", object())                       # legacy NULL owner
    everything = {r["job_id"] for r in bus.list_jobs(limit=50)}
    assert len(everything) == 3                          # admin sees all
    alice = bus.list_jobs(limit=50, owner="alice")
    assert [r["job_id"] for r in alice] == [mine]
    assert alice[0]["owner"] == "alice"
    # a NULL-owner row is invisible to every member — an owner filter is an
    # equality, and NULL never satisfies one
    assert bus.list_jobs(limit=50, owner="nobody") == []
    assert [r["owner"] for r in bus.list_jobs(limit=50, owner="bob")] == ["bob"]


def test_route_ownership_predicates(monkeypatch):
    """The route-level policy helpers in video_routes: who may see what."""
    from hugpy_server.app.routes import video_routes as vr
    app = Flask(__name__)

    _stub_me(monkeypatch, MEMBER)
    with app.test_request_context("/video/studio/clips", headers=COOKIE):
        assert vr._viewer() == (False, "alice")
        assert vr._owner_filter() == (True, "alice")     # scoped to themselves
        assert vr._may_view_job("alice") is True
        assert vr._may_view_job("bob") is False
        assert vr._may_view_job(None) is False           # legacy -> admin only
        assert vr._caller_username() == "alice"

    oa._clear_session_caches()
    _stub_me(monkeypatch, ADMIN)
    with app.test_request_context("/video/studio/clips", headers=COOKIE):
        assert vr._owner_filter() == (False, None)       # unscoped
        assert vr._may_view_job("bob") is True
        assert vr._may_view_job(None) is True
    oa._clear_session_caches()
    with app.test_request_context("/video/studio/clips?owner=bob",
                                  headers=COOKIE):
        assert vr._owner_filter() == (True, "bob")       # admin may narrow

    # An ACCOUNTLESS caller that reached the route — in production that is a
    # video-SHARE credential, which the gate admitted by design — keeps the
    # pre-slice, unscoped view. Only MEMBERS are scoped (see _viewer's note):
    # refusing here would kill the share feature, and a member can never mint a
    # share key (/keys/video-share is operator-only).
    oa._clear_session_caches()
    _stub_me(monkeypatch, None, ok=False)
    with app.test_request_context("/video/studio/clips", headers=COOKIE):
        assert vr._viewer() == (True, None)
        assert vr._owner_filter() == (False, None)
        assert vr._may_view_job("alice") is True
        assert vr._caller_username() is None      # owns nothing it creates


# ═══════════════════════════════════════════════════════════════════════════
# 4b) the uploads namespace mapping (one shared helper — writer + reader)
# ═══════════════════════════════════════════════════════════════════════════
def test_upload_namespace_mapping():
    assert oa.upload_namespace(None) is None
    assert oa.upload_namespace("") is None
    assert oa.upload_namespace("alice") == "alice"
    assert oa.upload_namespace("a@b.com") == "a@b.com"
    # no separators, no traversal, no leading dot, never a reserved dir
    assert "/" not in (oa.upload_namespace("a/../b") or "")
    assert not (oa.upload_namespace("...evil") or "").startswith(".")
    assert oa.upload_namespace("generated") == "u_generated"
    assert oa.upload_namespace(".sessions") != ".sessions"


def test_video_routes_uploads_namespace_reader(monkeypatch):
    from hugpy_server.app.routes import video_routes as vr
    from hugpy_platform.constants import UPLOADS_HOME
    root = os.path.realpath(UPLOADS_HOME)
    assert vr._uploads_namespace(os.path.join(root, "alice", "f.png")) == "alice"
    assert vr._uploads_namespace(os.path.join(root, "flat.png")) is None
    assert vr._uploads_namespace(os.path.join(root, "generated", "x.png")) is None
    assert vr._uploads_namespace("/etc/passwd") is None


# ═══════════════════════════════════════════════════════════════════════════
# 5) INSTALL LINKS — the coordinator amendment
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def links(tmp_path, monkeypatch):
    from hugpy_server.app.functions.imports.utils import api_keys as ak, install_links as il
    monkeypatch.setattr(ak, "_store_path", lambda: str(tmp_path / "keys.json"))
    monkeypatch.setattr(il, "_store_path", lambda: str(tmp_path / "links.json"))
    return il


@pytest.fixture
def agent_client(monkeypatch):
    from hugpy_server.app.routes import agent_routes
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return _client(app)


def test_member_can_mint_own_link_and_only_sees_it(monkeypatch, links,
                                                   agent_client):
    _stub_me(monkeypatch, MEMBER)
    r = agent_client.post("/agent/install-links",
                          json={"label": "alice-laptop"})
    assert r.status_code == 201, r.get_data(as_text=True)
    link = r.get_json()
    assert link["owner"] == "alice"
    assert link["scopes"] == ["v1"]          # the product default, not "full"
    assert "raw_key" not in link and "key" not in link
    # an operator-minted link is invisible to that member
    oa._clear_session_caches()
    links.create_install_link(label="ops-box")
    listed = agent_client.get("/agent/install-links").get_json()
    assert [l["label"] for l in listed["links"]] == ["alice-laptop"]


def test_member_scope_clamp(monkeypatch, links, agent_client):
    _stub_me(monkeypatch, MEMBER)
    for bad in (["full"], ["agent-register"], ["v1", "full"]):
        r = agent_client.post("/agent/install-links",
                              json={"label": "x", "scopes": bad})
        assert r.status_code == 403, bad
    ok = agent_client.post("/agent/install-links",
                           json={"label": "x", "scopes": ["v1", "ml"]})
    assert ok.status_code == 201


def test_operator_sees_all_links(monkeypatch, links, agent_client):
    _stub_me(monkeypatch, MEMBER)
    agent_client.post("/agent/install-links", json={"label": "alice-laptop"})
    _stub_me(monkeypatch, ADMIN)
    oa._clear_session_caches()
    links.create_install_link(label="ops-box")
    listed = agent_client.get("/agent/install-links").get_json()
    assert {l["label"] for l in listed["links"]} == {"alice-laptop", "ops-box"}


def test_delete_is_operator_or_creator(monkeypatch, links, agent_client):
    _stub_me(monkeypatch, MEMBER)
    mine = agent_client.post("/agent/install-links", json={"label": "mine"},
                             ).get_json()
    ops = links.create_install_link(label="ops-box")     # owner-less
    # the member may delete their own…
    assert agent_client.delete(
        f"/agent/install-links/{mine['link_id']}").status_code == 200
    # …but never the operator's
    oa._clear_session_caches()
    assert agent_client.delete(
        f"/agent/install-links/{ops['link_id']}").status_code == 403
    # the operator may delete anything
    _stub_me(monkeypatch, ADMIN)
    oa._clear_session_caches()
    assert agent_client.delete(
        f"/agent/install-links/{ops['link_id']}").status_code == 200


def test_install_links_anonymous_still_401(monkeypatch, links, agent_client):
    _stub_me(monkeypatch, None, ok=False)
    assert agent_client.post("/agent/install-links",
                             json={"label": "x"}).status_code == 401
    assert agent_client.get("/agent/install-links").status_code == 401
    assert agent_client.delete("/agent/install-links/x").status_code == 401


def test_console_dist_requires_member(monkeypatch, agent_client):
    _stub_me(monkeypatch, None, ok=False)
    assert agent_client.get("/agent/console/info").status_code == 401
    assert agent_client.get("/agent/console/fleet-console_1.0.0.deb").status_code == 401
    _stub_me(monkeypatch, MEMBER)
    oa._clear_session_caches()
    r = agent_client.get("/agent/console/info")
    assert r.status_code == 200
    body = r.get_json()
    # `install` (2026-08-12) and `artifacts` (2026-08-21, the four packaging
    # formats build-release.sh emits) were both added after this assertion was
    # written as an exact set — hence a subset check plus the keys that matter.
    assert {"deb", "agent_whl", "install", "artifacts"} <= set(body)
    assert set(body["artifacts"]) == {"deb", "rpm", "pacman", "appimage"}
