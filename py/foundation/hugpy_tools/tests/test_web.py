"""web.assess_webpage / prescreen_webpage against a real local HTTP server.

Exercises the true urllib path: canned HTML with title/meta/og/twitter tags,
canonical + lang, headings, relative links, repeated boilerplate (dedup),
JSON-LD, a non-UTF8 (cp1252) charset page, and a gzip-encoded page.
"""
from __future__ import annotations

import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hugpy_tools import web

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>Widget Co - Home</title>
<meta name="description" content="We sell widgets.">
<meta property="og:description" content="OG widgets.">
<meta name="twitter:description" content="TW widgets.">
<link rel="canonical" href="https://widget.co/home">
<script type="application/ld+json">{"@type":"Organization","name":"Widget Co"}</script>
<style>.x{color:red}</style>
<script>var boot = 1;</script>
</head>
<body>
<nav>Home About Contact</nav>
<h1>Welcome</h1>
<h2>Our widgets</h2>
<p>Widgets are great.</p>
<p>Our widgets are engineered for reliability and shipped worldwide with a
long, descriptive paragraph of body copy that comfortably clears the empty
render threshold so the cheap stdlib fetch is never second-guessed by a
browser fallback during this deterministic test run on any machine.</p>
<p>Boilerplate line.</p>
<p>Boilerplate line.</p>
<a href="/about">About</a>
<a href="page2.html">Page 2</a>
<a href="https://external.example.com/x">External</a>
<a href="#frag">Frag</a>
</body>
</html>"""

_CP1252 = ('<html><head><title>Café</title>'
           '<meta charset="cp1252"></head><body><p>naïve résumé '
           'text that is long enough to clear the render threshold so no browser '
           'fallback triggers here at all.</p></body></html>')


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/gzip"):
            body = gzip.compress(_HTML.encode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/cp1252"):
            body = _CP1252.encode("cp1252")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield "http://%s:%d" % (host, port)
    httpd.shutdown()


def test_assess_full_shape(server):
    r = web.assess_webpage(server + "/")
    # assessManager-parity keys present
    for k in ("url", "title", "description", "text", "metadata", "jsonld",
              "links", "truncated", "render"):
        assert k in r
    assert r["render"] == "stdlib"
    assert r["title"] == "Widget Co - Home"
    # description prefers name=description
    assert r["description"] == "We sell widgets."
    # additive fields
    assert r["lang"] == "en"
    assert r["canonical"] == "https://widget.co/home"
    assert {h["text"] for h in r["headings"]} == {"Welcome", "Our widgets"}


def test_assess_jsonld_parsed(server):
    r = web.assess_webpage(server + "/")
    assert r["jsonld"] and r["jsonld"][0].get("name") == "Widget Co"


def test_assess_boilerplate_dedup(server):
    r = web.assess_webpage(server + "/")
    # the repeated "Boilerplate line." collapses to one
    assert r["text"].count("Boilerplate line.") == 1


def test_assess_links_absolute_same_domain(server):
    r = web.assess_webpage(server + "/")
    # relative links resolved to absolute, same-domain only, external dropped
    assert (server + "/about") in r["links"]
    assert (server + "/page2.html") in r["links"]
    assert all("external.example.com" not in u for u in r["links"])
    assert all("#frag" not in u for u in r["links"])


def test_assess_truncation_flag(server):
    r = web.assess_webpage(server + "/", max_chars=20)
    assert r["truncated"] is True


def test_assess_gzip(server):
    r = web.assess_webpage(server + "/gzip")
    assert r["title"] == "Widget Co - Home"
    assert "Widgets are great." in r["text"]


def test_assess_cp1252_charset(server):
    r = web.assess_webpage(server + "/cp1252")
    assert r["title"] == "Café"
    assert "résumé" in r["text"]


def test_prescreen_shape(server):
    r = web.prescreen_webpage(server + "/")
    assert set(r) == {"url", "title", "description", "lede", "render"}
    assert r["title"] == "Widget Co - Home"
    assert r["description"] == "We sell widgets."


def test_non_http_scheme_degrades():
    r = web.assess_webpage("file:///etc/hostname")
    assert r["render"] == "failed"
    assert r["error"]


def test_force_render_without_browser_degrades(server):
    # With no headless browser installed, force_render returns a degraded dict
    # whose `error` names the missing dependency; if a browser IS present the
    # page renders instead. Either outcome is acceptable and asserted.
    r = web.assess_webpage(server + "/", force_render=True)
    if r["render"] == "failed":
        assert "browser" in r["error"].lower() or "playwright" in r["error"].lower()
    else:
        assert r["render"] == "rendered"
