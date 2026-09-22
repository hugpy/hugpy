"""The /video auth gate (video_auth.install_video_gate) + its structural
separation from the console/operator gate (operator_auth).

k8 (2026-07-18): the video arm went public/ungated 2026-07-15; the operator
directed it behind the SAME boundary as the console. This regresses:

  * the /video + /movie surface is DENIED for an anonymous caller (external
    mode): data/media/XHR -> 401, a browser shell navigation -> 302 to "/";
  * a valid console session (or operator token, or open mode) is ALLOWED through;
  * the SHARE SEAM: a video-scoped share principal satisfies the VIDEO gate but
    NEVER the console gate — share-principal present => /api/video ok while
    /api/keys is STILL 401. This is the load-bearing structural property (the
    share credential is video-scoped by construction);
  * the surface matcher covers /video, /api/video, /movie (and NOT /keys,
    /console, /workers).
"""
import pytest
from flask import Flask

from hugpy_server.app import video_auth as va
from hugpy_server.app import operator_auth as oa


class _ApiPrefixMiddleware:
    """Mirror wsgi_app.ApiPrefixMiddleware: strip a leading /api BEFORE routing,
    exactly as production does (nginx/this middleware), so /api/video/... routes
    to the bare /video/... rule (not the SPA catch-all)."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def build_app():
    """A minimal app that mirrors production wiring: the /api-strip middleware,
    the operator gate, then the video gate; a console-gated route (/keys, in
    operator_auth._SENSITIVE), a video data route, a movie route, an open route,
    and the SPA shell catch-all (endpoint _hugpy_ui) the video gate keys off for
    the redirect branch."""
    app = Flask(__name__)

    @app.route("/keys", methods=["GET"])
    def _keys():
        return "keys", 200

    @app.route("/video/studio/clips", methods=["GET"])
    def _clips():
        return "clips", 200

    @app.route("/movie/presets", methods=["GET"])
    def _movie():
        return "movie", 200

    @app.route("/llm/workers", methods=["GET"])
    def _workers():
        return "workers", 200

    def _shell(asset=""):
        return "<html>shell</html>", 200
    app.add_url_rule("/", endpoint="_hugpy_ui", view_func=_shell, defaults={"asset": ""})
    app.add_url_rule("/<path:asset>", endpoint="_hugpy_ui", view_func=_shell)

    oa.install_operator_gate(app)
    va.install_video_gate(app)
    app.wsgi_app = _ApiPrefixMiddleware(app.wsgi_app)
    return app


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch, tmp_path):
    """Deterministic: no operator token in the env, a fresh session cache, and
    audit/state writes kept out of the real projects tree."""
    monkeypatch.setenv("PROJECTS_HOME", str(tmp_path))
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("HUGPY_AUTH_MODE", raising=False)
    oa._SESSION_CACHE.clear()
    yield
    oa._SESSION_CACHE.clear()


@pytest.fixture
def external(monkeypatch):
    """EXTERNAL mode with no console session and no share credential by default;
    tests flip the two seams via the returned setter."""
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")
    monkeypatch.setattr(oa, "_validate_session_external", lambda: False)
    monkeypatch.setattr(va, "_video_share_principal", lambda request: None)

    def _set(*, session=None, share=None):
        if session is not None:
            monkeypatch.setattr(oa, "_validate_session_external", lambda: session)
        if share is not None:
            monkeypatch.setattr(va, "_video_share_principal", lambda request: share)
        oa._SESSION_CACHE.clear()
        return build_app().test_client()
    return _set


# --- unit: the surface matcher -------------------------------------------------
@pytest.mark.parametrize("path,expected", [
    ("/video/studio/clips", True),
    ("/api/video/studio/clips", True),     # after the /api strip
    ("/movie/presets", True),
    ("/video", True),                       # bare /video
    ("/keys", False),
    ("/videofoo", False),                   # word-boundary
    ("/llm/workers", False),
])
def test_surface_matcher(path, expected):
    with build_app().test_request_context(path):
        assert va._is_video_surface() is expected


# --- EXTERNAL mode: anonymous is denied on video, and on console --------------
def test_external_anonymous_denied(external):
    c = external()
    assert c.get("/api/video/studio/clips").status_code == 401
    assert c.get("/video/studio/clips").status_code == 401
    assert c.get("/movie/presets").status_code == 401
    # Browser shell navigation -> redirect to the console login at "/".
    r = c.get("/video/", headers={"Sec-Fetch-Dest": "document"})
    assert r.status_code == 302
    assert (r.headers.get("Location") or "").endswith("/")
    # Console boundary still works exactly as before (operator gate).
    assert c.get("/api/keys").status_code == 401
    # Non-gated reads stay open.
    assert c.get("/llm/workers").status_code == 200


def test_share_principal_opens_video_only(external):
    """THE STRUCTURAL PROPERTY: a video-share principal opens /video ONLY."""
    c = external(share={"scope": "video", "share": "tok123"})
    assert c.get("/api/video/studio/clips").status_code == 200
    assert c.get("/movie/presets").status_code == 200
    # share cannot satisfy the console gate
    assert c.get("/api/keys").status_code == 401
    assert c.get("/video/", headers={"Sec-Fetch-Dest": "document"}).status_code == 200


def test_console_session_opens_both_surfaces(external):
    c = external(session=True)
    assert c.get("/api/video/studio/clips").status_code == 200
    assert c.get("/api/keys").status_code == 200
    assert c.get("/video/").status_code == 200


# --- OPEN mode (self-hosted product, no token): the arm is open, no login -----
def test_open_mode_no_token(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    c = build_app().test_client()
    assert c.get("/api/video/studio/clips").status_code == 200
    assert c.get("/api/keys").status_code == 200


# --- OPEN mode WITH an operator token: video requires the token too -----------
def test_open_mode_with_operator_token(monkeypatch):
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "s3cret")
    c = build_app().test_client()
    assert c.get("/api/video/studio/clips").status_code == 401
    assert c.get("/api/video/studio/clips",
                 headers={"X-Operator-Token": "s3cret"}).status_code == 200
