"""assess_webpage / prescreen_webpage — compact, LLM-ready webpage assessment,
stdlib only.

A native re-implementation of abstract_webtools' assessManager: it returns the
same dict shape so a caller can switch backends without touching call sites,
but replaces the requests + BeautifulSoup + Selenium chain with ``urllib`` and
``html.parser``. No third-party import on the default path.

Fetch hardening mirrors the agent's http tool: an http(s)-only opener that
refuses redirects to other schemes and disables the ``file://`` / ``ftp://``
handlers outright, a size cap, a timeout, a User-Agent, gzip/deflate handling,
and charset detection (Content-Type header -> BOM -> ``<meta charset>`` ->
utf-8 -> cp1252).

JS rendering is an OPT-IN, lazy path: ``force_render=True`` (or an auto
fall-back when the cheap fetch returns almost no text) tries a headless
browser only if one is installed, and raises a clear "install X" error
otherwise. The default never needs a browser.

Output dict (assessManager parity + additive fields):
    {url, title, description, text, metadata, jsonld, links, truncated, render,
     canonical, lang, headings}
On total failure text/metadata/jsonld/links/headings degrade to empty and
render == "failed" (with an ``error`` string).
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.error
import urllib.request
import zlib
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

TIMEOUT = 20
MAX_BYTES = 2 * 1024 * 1024  # 2 MiB read cap off the wire
USER_AGENT = "hugpy-tools/0.1 (+https://hugpy.ai)"
# Below this many characters of extracted text we assume a JS wall / bot block
# and an (opt-in, if-available) full render is worth trying.
_EMPTY_RENDER_THRESHOLD = 200
_CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-]+)", re.I)


# --------------------------------------------------------------------------- #
# Hardened fetch (mirrors hugpy_agent.tools.http security posture)
# --------------------------------------------------------------------------- #
class _SchemeCheckingRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(("http://", "https://")):
            raise urllib.error.URLError(
                "refusing redirect to non-http(s) URL: %r" % newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _DisabledFile(urllib.request.FileHandler):
    def file_open(self, req):
        raise urllib.error.URLError("file:// scheme is not permitted")


class _DisabledFTP(urllib.request.FTPHandler):
    def ftp_open(self, req):
        raise urllib.error.URLError("ftp:// scheme is not permitted")


_opener = urllib.request.build_opener(
    _SchemeCheckingRedirect, _DisabledFile, _DisabledFTP)


def _open(req, timeout):
    """The single network seam — monkeypatched in tests to avoid the wire."""
    return _opener.open(req, timeout=timeout)


def _decompress(raw: bytes, encoding: str) -> bytes:
    enc = (encoding or "").lower()
    try:
        if enc == "gzip":
            return gzip.decompress(raw)
        if enc == "deflate":
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except (OSError, zlib.error):
        return raw
    return raw


_BOMS = ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16-le"),
         (b"\xfe\xff", "utf-16-be"))


def _decode(raw: bytes, header_charset: str | None) -> str:
    if header_charset:
        try:
            return raw.decode(header_charset, errors="replace")
        except (LookupError, TypeError):
            pass
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return raw.decode(enc, errors="replace")
    m = _CHARSET_RE.search(raw[:4096])
    if m:
        try:
            return raw.decode(m.group(1).decode("ascii"), errors="replace")
        except (LookupError, TypeError):
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _fetch(url: str, timeout: int):
    """GET ``url`` and return (final_url, status, content_type, html_text).
    Raises on transport failure (the caller degrades to render=='failed')."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are allowed, got %r" % url)
    req = urllib.request.Request(
        url, method="GET",
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*",
                 "Accept-Encoding": "gzip, deflate"})
    with _open(req, timeout) as resp:
        final_url = resp.geturl()
        if not final_url.startswith(("http://", "https://")):
            raise ValueError("redirect to non-http(s) URL: %r" % final_url)
        status = getattr(resp, "status", None)
        if status is None and hasattr(resp, "getcode"):
            status = resp.getcode()
        headers = resp.headers
        content_type = headers.get("Content-Type", "") if headers else ""
        content_encoding = headers.get("Content-Encoding", "") if headers else ""
        raw = resp.read(MAX_BYTES)
    raw = _decompress(raw, content_encoding)
    charset = None
    if content_type and "charset=" in content_type.lower():
        charset = content_type.lower().split("charset=", 1)[1].split(";")[0].strip() or None
    return final_url, status, content_type, _decode(raw, charset)


# --------------------------------------------------------------------------- #
# HTML extraction (html.parser, no BeautifulSoup)
# --------------------------------------------------------------------------- #
_SKIP_TEXT = {"script", "style", "noscript", "template"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = None
        self.lang = None
        self.canonical = None
        self.metadata: list[dict] = []
        self.sections: list[str] = []
        self.headings: list[dict] = []
        self.hrefs: list[str] = []
        self.jsonld: list = []
        self._skip = 0
        self._in_title = False
        self._title_buf: list[str] = []
        self._heading_level = 0
        self._heading_buf: list[str] = []
        self._in_jsonld = False
        self._jsonld_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html" and self.lang is None and a.get("lang"):
            self.lang = a["lang"].strip()
        elif tag == "title":
            self._in_title = True
            self._title_buf = []
        elif tag == "meta":
            row = {}
            for k in ("name", "property", "http-equiv", "itemprop",
                      "charset", "content"):
                if a.get(k):
                    row[k] = a[k]
            if row:
                self.metadata.append(row)
        elif tag == "link":
            rel = a.get("rel", "").lower()
            if "canonical" in rel.split() and a.get("href") and not self.canonical:
                self.canonical = a["href"].strip()
        elif tag == "a":
            if a.get("href"):
                self.hrefs.append(a["href"])
        elif tag in _HEADINGS:
            self._heading_level = int(tag[1])
            self._heading_buf = []
        if tag in _SKIP_TEXT:
            self._skip += 1
            if tag == "script" and "ld+json" in a.get("type", "").lower():
                self._in_jsonld = True
                self._jsonld_buf = []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            t = "".join(self._title_buf).strip()
            if t and not self.title:
                self.title = t
        elif tag in _HEADINGS and self._heading_level:
            text = "".join(self._heading_buf).strip()
            if text:
                self.headings.append({"level": self._heading_level, "text": text})
            self._heading_level = 0
        if tag in _SKIP_TEXT and self._skip > 0:
            self._skip -= 1
            if tag == "script" and self._in_jsonld:
                self._in_jsonld = False
                blob = "".join(self._jsonld_buf).strip()
                if blob:
                    try:
                        self.jsonld.append(json.loads(blob))
                    except (ValueError, TypeError):
                        self.jsonld.append({"raw": blob})

    def handle_data(self, data):
        if self._in_jsonld:
            self._jsonld_buf.append(data)
        if self._in_title:
            self._title_buf.append(data)
        if self._heading_level:
            self._heading_buf.append(data)
        if self._skip == 0:
            s = data.strip()
            if s:
                self.sections.append(s)


# --------------------------------------------------------------------------- #
# Budgeting + lookups (assessManager parity)
# --------------------------------------------------------------------------- #
def _budget_text(sections, max_chars):
    """Dedup consecutive boilerplate and cap total length.
    Returns (text, truncated)."""
    parts, seen_last, total, truncated = [], None, 0, False
    for s in sections or []:
        s = (s or "").strip()
        if not s or s == seen_last:
            continue
        if total + len(s) + 1 > max_chars:
            truncated = True
            break
        parts.append(s)
        seen_last = s
        total += len(s) + 1
    return "\n".join(parts), truncated


def _meta_lookup(metadata, *keys):
    wanted = {k.lower() for k in keys}
    for m in metadata:
        for k in ("name", "property", "itemprop"):
            if str(m.get(k, "")).lower() in wanted and m.get("content"):
                return m["content"]
    return None


def _same_domain_links(hrefs, base_url, max_links):
    """Resolve to absolute, keep same-registrable-host http(s) links, dedup,
    cap. Matches assessManager's same-domain link policy."""
    if max_links <= 0:
        return []
    base_host = urlparse(base_url).netloc.lower()
    out: list[str] = []
    for href in hrefs:
        href = (href or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        host = parsed.netloc.lower()
        if base_host and host and host != base_host \
                and not host.endswith("." + base_host) \
                and not base_host.endswith("." + host):
            continue
        if absolute not in out:
            out.append(absolute)
        if len(out) >= max_links:
            break
    return out


# --------------------------------------------------------------------------- #
# Optional JS render (lazy, opt-in)
# --------------------------------------------------------------------------- #
def _render_js(url: str, timeout: int) -> str:
    """Render ``url`` with a headless browser if one is installed. Tries
    Playwright then Selenium; raises ``RuntimeError`` naming what to install
    when neither is available."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        pass
    else:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=timeout * 1000)
                return page.content()
            finally:
                browser.close()
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.options import Options  # type: ignore
    except ImportError:
        raise RuntimeError(
            "JS rendering requires a headless browser, none is installed. "
            "Install Playwright ('pip install playwright' then "
            "'playwright install chromium') or Selenium ('pip install "
            "selenium' with a chromedriver on PATH). The default assess path "
            "does not need either.")
    opts = Options()
    opts.add_argument("--headless=new")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        return driver.page_source
    finally:
        driver.quit()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assess_webpage(url, *, max_chars=12000, max_links=50, force_render=False,
                   timeout=TIMEOUT):
    """Fetch a page and return a compact, LLM-ready structured assessment.

    Same dict shape as abstract_webtools.assessManager.assess_webpage, plus the
    additive keys ``canonical``, ``lang`` and ``headings``.
    """
    render = "failed"
    final_url, status, html = url, None, None
    error = None

    if force_render:
        try:
            html = _render_js(url, timeout)
            render = "rendered"
            final_url = url
        except Exception as exc:  # noqa: BLE001 — surfaced as data
            return _degraded(url, "%s: %s" % (type(exc).__name__, exc))
    else:
        try:
            final_url, status, _ctype, html = _fetch(url, timeout)
            render = "stdlib"
        except Exception as exc:  # noqa: BLE001 — transport failure is data
            return _degraded(url, "%s: %s" % (type(exc).__name__, exc))

    parser = _Extractor()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001 — malformed HTML never crashes the tool
        pass

    # Auto-fallback: cheap fetch returned almost nothing -> try a render, but
    # only if a browser is available; otherwise keep the cheap result honestly.
    if not force_render and sum(len(s) for s in parser.sections) < _EMPTY_RENDER_THRESHOLD:
        try:
            rendered = _render_js(url, timeout)
            p2 = _Extractor()
            p2.feed(rendered)
            p2.close()
            parser, render = p2, "rendered"
        except Exception:  # noqa: BLE001 — no browser / render failed: keep cheap
            pass

    text, truncated = _budget_text(parser.sections, max_chars)
    return {
        "url": url,
        "final_url": final_url,
        "status": status,
        "title": parser.title,
        "description": _meta_lookup(
            parser.metadata, "description", "og:description", "twitter:description"),
        "text": text,
        "metadata": parser.metadata,
        "jsonld": parser.jsonld,
        "links": _same_domain_links(parser.hrefs, final_url, max_links),
        "truncated": truncated,
        "render": render,
        "canonical": parser.canonical,
        "lang": parser.lang,
        "headings": parser.headings,
        "error": error,
    }


def _degraded(url, error):
    return {
        "url": url, "final_url": url, "status": None, "title": None,
        "description": None, "text": "", "metadata": [], "jsonld": [],
        "links": [], "truncated": False, "render": "failed",
        "canonical": None, "lang": None, "headings": [], "error": error,
    }


def prescreen_webpage(url, *, max_chars=600, force_render=False, timeout=TIMEOUT):
    """Cheap relevance pre-screen: title + description + a short lede of body
    text. Same dict shape as assessManager.prescreen_webpage:
    ``{url, title, description, lede, render}``."""
    page = assess_webpage(url, max_chars=max_chars, max_links=0,
                          force_render=force_render, timeout=timeout)
    return {
        "url": url,
        "title": page["title"],
        "description": page["description"],
        "lede": page["text"],
        "render": page["render"],
    }
