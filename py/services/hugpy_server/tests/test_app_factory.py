"""The Flask app builds without the monolith, carries the wiring report, and
serves the packaged console."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from flask import Flask

from hugpy_server import wsgi_app


def test_create_app_is_wired(server_app):
    rep = server_app.extensions["hugpy_wiring"]
    assert rep["ok"] is True, rep["errors"]
    rules = {str(r) for r in server_app.url_map.iter_rules()}
    assert "/health" in rules
    assert "/fleet/runbook" in rules


def test_api_prefix_middleware_and_health(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_fleet_runbook_comes_from_fleet_package(client):
    from hugpy_fleet.doctrine.runbook import load_runbook
    r = client.get("/fleet/runbook")
    assert r.status_code == 200
    assert r.get_json() == load_runbook()


def test_welcome_version_names_this_package(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.get_json()
    assert body["name"] == "hugpy-server"
    assert body["version"] and body["api"] == 1


def test_package_console_dist_is_package_data():
    d = wsgi_app.packaged_console_dir()
    assert d and os.path.isdir(d)
    assert os.path.isfile(os.path.join(d, "index.html"))


def _bare(dist):
    app = Flask("static-mount-test")
    mounted = wsgi_app.mount_console(app, dist)
    return app, mounted


def test_static_mount_serves_index_and_arms(tmp_path):
    (tmp_path / "index.html").write_text("<html>root</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a.js").write_text("console.log(1)")
    for arm in ("fleet", "media", "video"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "index.html").write_text(f"<html>{arm}</html>")
    app, mounted = _bare(str(tmp_path))
    assert mounted is True
    c = app.test_client()
    assert c.get("/").data == b"<html>root</html>"
    assert c.get("/login").data == b"<html>root</html>"          # SPA deep link
    assert c.get("/assets/a.js").data == b"console.log(1)"
    for arm in ("fleet", "media", "video"):
        assert c.get(f"/{arm}").data == f"<html>{arm}</html>".encode()
        assert c.get(f"/{arm}/deep/link").data == f"<html>{arm}</html>".encode()
    r = c.get("/studio/foo")
    assert r.status_code == 302 and r.headers["Location"].endswith("/video/foo")


def test_static_mount_404s_cleanly_without_bundle(tmp_path):
    app, mounted = _bare(str(tmp_path))          # no index.html at all
    assert mounted is False
    c = app.test_client()
    assert c.get("/").status_code == 404
    assert c.get("/fleet").status_code == 404
    app2, mounted2 = _bare(None)
    assert mounted2 is False
    assert app2.test_client().get("/").status_code == 404


def test_console_dist_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_UI_DIST", str(tmp_path))
    assert wsgi_app.console_dist_dir() != str(tmp_path)   # no index there yet
    (tmp_path / "index.html").write_text("x")
    assert wsgi_app.console_dist_dir() == str(tmp_path)


def test_serve_help_and_package_import_without_monolith():
    code = ("import sys; sys.modules['abstract_hugpy_dev']=None; "
            "sys.modules['hugpy_discord']=None; sys.modules['hugpy_ops']=None; "
            "from hugpy_server.wsgi_app import main; main(['--help'])")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--port" in proc.stdout and "--auth" in proc.stdout


def test_main_serves_with_flask_fallback(monkeypatch):
    """main() builds the app, then hands it to the serving glue."""
    calls = {}
    monkeypatch.setattr(wsgi_app, "get_hugpy_flask",
                        lambda **kw: calls.setdefault("app", object()))
    monkeypatch.setattr(wsgi_app, "_serve",
                        lambda app, host, port, threads, workers, debug:
                        calls.update(host=host, port=port, threads=threads, workers=workers) or 0)
    monkeypatch.delenv("HUGPY_AUTH_MODE", raising=False)
    assert wsgi_app.main(["--host", "127.0.0.1", "--port", "7999", "--threads", "3"]) == 0
    assert calls["host"] == "127.0.0.1" and calls["port"] == 7999 and calls["threads"] == 3
    assert os.environ["HUGPY_AUTH_MODE"] == "open"


def test_lazy_module_app_attribute(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(wsgi_app, "_APP", None)
    monkeypatch.setattr(wsgi_app, "get_hugpy_flask", lambda *a, **k: sentinel)
    assert wsgi_app.app is sentinel
