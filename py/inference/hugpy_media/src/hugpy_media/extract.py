"""Deterministic media-ingest helpers for the `/ml` amenities that are NOT model
inference — reading text out of an uploaded document (PDF / DOCX / plain text).

These are the EXTRACT stage of the media-intelligence pipeline (extract → enrich:
summarize + keywords), so they live as thin LOCAL handlers rather than riding the
model resolver: there is no model to load, nothing to delegate to a GPU worker. The
heavy parsers are imported lazily so importing this module stays cheap (no torch,
phone-clean) and a missing optional parser degrades to a clear error, not an
ImportError at module load.

Security: reads are JAILED to the storage root (UPLOADS_HOME / DEFAULT_ROOT). The
client only ever passes paths produced by POST /uploads (under UPLOADS_HOME), so a
path that resolves outside the root is rejected — this endpoint never becomes an
arbitrary-file-read primitive.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse, urljoin

from hugpy_platform.constants import UPLOADS_HOME, DEFAULT_ROOT

# Declines used to be silent: a file the chat "couldn't read" left NO trace in the
# server log, so ops could not tell a scanned PDF from a broken parser. Every
# {ok: False} return now logs a warning (operator ask 2026-08-04, k64).
logger = logging.getLogger(__name__)

_MAX_URL_BYTES = 5 * 1024 * 1024  # cap fetched body so a huge/streamed page can't OOM us
_URL_TIMEOUT = 15                 # per-request seconds
_MAX_REDIRECTS = 4

_MAX_ASSESS_CHARS = 12000   # token budget for the structured assessment's body text
_MAX_ASSESS_LINKS = 50      # cap same-domain links carried in an assessment

# Extensions handled by each strategy. Anything else → a clear "unsupported" error.
_PDF_EXT = {".pdf"}
_DOCX_EXT = {".docx"}
_XLSX_EXT = {".xlsx", ".xlsm"}
_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".log", ".text"}

# Formats we genuinely CANNOT read, mapped to an honest, actionable message.
# Previously these fell through to the generic-text reader, whose binary sniff
# declined them with "isn't readable as text" — true but useless: it named no
# format and offered no fix, so the chat's failures read as mush (operator ask
# 2026-08-04, k64). The remedy is a one-step conversion, so say it.
_UNSUPPORTED_EXT = {
    ".doc": "legacy binary .doc isn't supported — save it as .docx and re-attach",
    ".xls": "legacy binary .xls isn't supported — save it as .xlsx and re-attach",
    ".ppt": "legacy binary .ppt isn't supported — save it as .pptx or export a PDF",
}


def _jailed_realpath(path: str) -> str:
    """Resolve ``path`` and require it to live under the storage root.

    Raises PermissionError if it escapes (symlinks resolved via realpath), so a
    crafted path like ``../../etc/passwd`` can't be read through this endpoint.
    """
    rp = os.path.realpath(path)
    roots = [os.path.realpath(r) for r in (UPLOADS_HOME, DEFAULT_ROOT) if r]
    if not any(rp == root or rp.startswith(root + os.sep) for root in roots):
        raise PermissionError("file path is outside the allowed storage root")
    return rp


def _page_reason(exc: Exception) -> str:
    """One short sentence naming why a page failed (never an empty string)."""
    reason = " ".join(str(exc).split()).strip() or exc.__class__.__name__
    return reason[:200]


def _pdf_meta(raw, page_count: int) -> dict:
    """Cheap document info (title / author / page_count) from a PDF's Info dict.

    Every read is guarded: absent, garbage or exploding metadata must NEVER fail
    an extraction (operator ask 2026-08-04, k65) — it degrades to fewer keys.
    Handles both engines' shapes: pdfplumber hands back {"Title": ...} and
    PyPDF2's DocumentInformation is a dict with "/Title" keys.
    """
    meta: dict = {}
    if page_count > 0:
        meta["page_count"] = page_count
    for key, tag in (("title", "Title"), ("author", "Author")):
        try:
            if isinstance(raw, dict):
                value = raw.get(tag) or raw.get("/" + tag)
            else:
                value = getattr(raw, key, None)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if value is None:
                continue
            value = " ".join(str(value).split()).strip()
            if value:
                meta[key] = value[:300]
        except Exception:
            continue  # a hostile/garbage Info entry costs us that one key, nothing more
    return meta


def _pdf_pages_pdfplumber(path: str) -> tuple[list[dict], list[str], dict]:
    """Engine 1. Per-page fault tolerant: a page that throws is RECORDED and
    skipped, so one damaged page can't cost us the other nineteen."""
    pages: list[dict] = []
    warnings: list[str] = []
    meta: dict = {}
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            doc_pages = pdf.pages
            meta = _pdf_meta(getattr(pdf, "metadata", None), len(doc_pages))
            for i, pg in enumerate(doc_pages):
                try:
                    pages.append({"index": i, "text": (pg.extract_text() or "").strip()})
                except Exception as exc:
                    reason = _page_reason(exc)
                    pages.append({"index": i, "text": "", "error": reason})
                    warnings.append(f"page {i + 1} failed: {reason}")
    except Exception as exc:
        # Whole-document failure (parser missing, file unopenable/encrypted, or the
        # page list blew up mid-walk). Keep whatever we already collected — a
        # partial read beats none — and say where it stopped.
        if pages:
            warnings.append(f"stopped after page {len(pages)}: {_page_reason(exc)}")
    return pages, warnings, meta


def _pdf_pages_pypdf2(path: str) -> tuple[list[dict], list[str], dict]:
    """Engine 2 (fallback), same per-page fault tolerance as engine 1."""
    pages: list[dict] = []
    warnings: list[str] = []
    meta: dict = {}
    try:
        import PyPDF2
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            doc_pages = reader.pages
            meta = _pdf_meta(getattr(reader, "metadata", None), len(doc_pages))
            for i, pg in enumerate(doc_pages):
                try:
                    pages.append({"index": i, "text": (pg.extract_text() or "").strip()})
                except Exception as exc:
                    reason = _page_reason(exc)
                    pages.append({"index": i, "text": "", "error": reason})
                    warnings.append(f"page {i + 1} failed: {reason}")
    except Exception as exc:
        if pages:
            warnings.append(f"stopped after page {len(pages)}: {_page_reason(exc)}")
    return pages, warnings, meta


def _extract_pdf(path: str) -> tuple[list[dict], str, dict]:
    """Per-page text via pdfplumber, falling back to PyPDF2 when the document
    yields nothing (scanned/odd PDFs). No OCR — that's the heavy [ocr] path.

    A partly-broken PDF now DEGRADES instead of failing (operator ask 2026-08-04,
    k65): pages that throw are recorded as ``{"index": i, "text": "", "error":
    ...}`` and skipped, and the third return value carries the additive
    ``pages_total`` / ``pages_extracted`` / ``warnings`` / ``meta`` fields so the
    caller can say "Extracted 3/5 pages; page 4 failed: …" instead of silently
    handing back a short document.
    """
    pages, warnings, meta = _pdf_pages_pdfplumber(path)
    if not any(p.get("text") for p in pages):
        # Engine 1 found nothing (or couldn't open it) — try PyPDF2 before giving
        # up. Its result only WINS when it read something, or when engine 1
        # enumerated no pages at all (so a corrupt-vs-scanned verdict keeps the
        # richer page list).
        alt_pages, alt_warnings, alt_meta = _pdf_pages_pypdf2(path)
        if any(p.get("text") for p in alt_pages) or not pages:
            pages, warnings = alt_pages, alt_warnings
            meta = alt_meta or meta
    text = "\n\n".join(p["text"] for p in pages if p.get("text")).strip()
    info = {
        "pages_total": len(pages),
        "pages_extracted": sum(1 for p in pages if p.get("text")),
        "warnings": warnings,
        "meta": meta,
    }
    return pages, text, info


def _extract_docx(path: str) -> tuple[list[dict], str]:
    import docx  # python-docx
    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs if p.text and p.text.strip()).strip()
    return [], text


_MAX_XLSX_CHARS = 400_000  # cap the flattened dump (OOM guard + LLM context budget)


def _extract_xlsx(path: str) -> tuple[list[dict], str]:
    """Flatten an .xlsx/.xlsm workbook to plain text, one block per sheet.

    Spreadsheets used to hit _extract_generic_text, whose binary sniff declined
    them outright — an attached workbook was simply unreadable (operator ask
    2026-08-04, k64). openpyxl is already in the venv, so read it properly:
    ``read_only`` + ``data_only`` streams rows without loading the whole object
    graph and yields CACHED FORMULA VALUES rather than "=SUM(A1:A9)" strings —
    the numbers a reader would actually see.

    Shape mirrors _extract_pdf: ``pages`` is one entry per sheet (so the UI's
    per-page rendering works unchanged), and cells are tab-separated so column
    structure survives into the model's context.
    """
    import openpyxl  # lazy: keeps module import cheap, like the other parsers

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheets: list[dict] = []
        total = 0
        for index, ws in enumerate(wb.worksheets):
            lines: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                while cells and not cells[-1].strip():
                    cells.pop()  # drop trailing empties so blank columns don't pad every row
                if not cells:
                    continue  # wholly empty row
                line = "\t".join(cells)
                lines.append(line)
                total += len(line) + 1
                if total >= _MAX_XLSX_CHARS:
                    lines.append("… (truncated)")
                    break
            body = "\n".join(lines).strip()
            if body:
                sheets.append({"index": index, "name": ws.title,
                               "text": f"# Sheet: {ws.title}\n{body}"})
            if total >= _MAX_XLSX_CHARS:
                break
    finally:
        wb.close()  # read_only workbooks hold an open file handle until closed

    text = "\n\n".join(s["text"] for s in sheets).strip()
    return sheets, text


def _extract_text(path: str) -> tuple[list[dict], str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [], fh.read().strip()


_MAX_TEXT_BYTES = 16 * 1024 * 1024  # cap a generic read so a huge binary can't OOM us


def _extract_generic_text(path: str) -> tuple[list[dict], str]:
    """Reader of last resort: read an UNKNOWN file as UTF-8 text, but only when it
    actually looks textual.

    Mirrors the default in abstract_utilities' read_any_file ("if nothing else
    matches, just read it") — kept stdlib-only here (no pandas/ocr). A cheap
    binary sniff (a NUL byte, or too high a ratio of replacement/control chars)
    declines binary files (images, archives, spreadsheets) instead of handing
    back replacement-character garbage.
    """
    with open(path, "rb") as fh:
        raw = fh.read(_MAX_TEXT_BYTES)
    if not raw or b"\x00" in raw:
        return [], ""
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return [], ""
    bad = sum(1 for ch in text if ch == "�" or (ord(ch) < 32 and ch not in "\t\n\r\f"))
    if bad / len(text) > 0.10:
        return [], ""
    return [], text.strip()


def _decline(name: str, kind: str, error: str, pages: list[dict] | None = None) -> dict:
    """Build a {ok: False} extract result AND log it.

    Single choke point so no decline path can go dark again — the chat surfaces
    ``error`` verbatim to the user, and ops sees the same sentence in the log.
    """
    logger.warning("media extract declined: %s (%s)", name, error)
    return {"ok": False, "kind": kind, "text": "", "pages": pages or [],
            "error": error, "name": name}


def extract_document(path: str) -> dict:
    """Read text from a document under the storage root.

    Returns the same {ok, text, ...} shape the other `/ml` amenities produce, so
    the chat's pickText() narration path treats it identically. ``pages`` is
    populated for PDFs (per-page) and spreadsheets (per-sheet), empty otherwise.

    Every failure is an HONEST, SPECIFIC sentence naming the format and the fix
    where one exists (operator ask 2026-08-04, k64) — the chat renders it
    verbatim, so a vague message here becomes a vague failure for the user.

    A PARTIAL success is just as honest (operator ask 2026-08-04, k65): a PDF
    whose page 4 threw still returns pages 1-3 plus the additive ``pages_total`` /
    ``pages_extracted`` / ``warnings`` / ``meta`` fields, so nothing hands back a
    short document as if it were the whole one.
    """
    rp = _jailed_realpath(path)
    name = os.path.basename(path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"no such file: {name}")

    ext = os.path.splitext(rp)[1].lower()
    # Additive per-format fields (currently PDF-only: pages_total / pages_extracted /
    # warnings / meta). Empty for every other kind, so their shape is untouched.
    extra: dict = {}
    if ext in _UNSUPPORTED_EXT:
        # Named up front rather than after a doomed binary sniff, so the user is
        # told the actual format and the one-step conversion that fixes it.
        return _decline(name, "unsupported", _UNSUPPORTED_EXT[ext])

    if ext in _PDF_EXT:
        pages, text, extra = _extract_pdf(rp)
        kind = "pdf"
        if not text:
            # Distinguish "pages exist but carry no text layer" (a scan — OCR
            # would be needed) from "we couldn't open it at all". The old shared
            # message ("empty or image-only/scanned") left the user guessing.
            # k65: a page that THREW is not evidence of a scan, so the scanned
            # wording needs at least one page that read cleanly and came back empty.
            readable = [p for p in pages if not p.get("error")]
            return _decline(name, kind, (
                "this PDF has no text layer (scanned pages); OCR isn't wired up yet"
                if readable else
                "couldn't read any pages from this PDF — it may be corrupt or "
                "password-protected"
            ), pages)
    elif ext in _DOCX_EXT:
        pages, text = _extract_docx(rp)
        kind = "docx"
        if not text:
            return _decline(name, kind,
                            "this .docx has no readable paragraph text (it may be "
                            "empty, or contain only images/tables)")
    elif ext in _XLSX_EXT:
        pages, text = _extract_xlsx(rp)
        kind = "xlsx"
        if not text:
            return _decline(name, kind, "this workbook has no cells with content", pages)
    elif ext in _TEXT_EXT:
        pages, text = _extract_text(rp)
        kind = "text"
        if not text:
            return _decline(name, kind, "this file is empty")
    else:
        # Unknown extension → default to "just read it" (guarded UTF-8 read),
        # so an unrecognized file is read rather than rejected outright.
        pages, text = _extract_generic_text(rp)
        kind = "text"
        if not text:
            return _decline(name, "unknown",
                            f"file type '{ext or '(none)'}' isn't readable as text "
                            f"(it looks like binary data, not a document)")

    return {
        "ok": True,
        "kind": kind,
        "text": text,
        "pages": pages,
        "chars": len(text),
        "name": os.path.basename(path),
        # k65 — additive only: a PARTIAL read reports what it did and didn't get
        # ("Extracted 3/5 pages; page 4 failed: …") instead of passing off a short
        # document as a whole one. Absent for kinds that don't paginate.
        **extra,
    }


# --- URL fetch (readable text from a webpage) --------------------------------
# Server-side URL fetch is an SSRF surface: a caller could try to make the server
# hit internal services (169.254.169.254 cloud metadata, localhost:7002, LAN
# workers, etc.). We defend by resolving every host (including each redirect hop)
# and REFUSING any address that isn't public — scheme is restricted to http(s),
# redirects are followed manually so each Location is re-validated, and the body
# is size-capped.

def _assert_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise PermissionError(f"only http/https URLs are allowed (got '{p.scheme or 'none'}')")
    host = p.hostname
    if not host:
        raise PermissionError("URL has no host")
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise PermissionError(f"cannot resolve host '{host}'")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise PermissionError(f"refusing to fetch a non-public address ({ip})")


def fetch_url_text(url: str) -> dict:
    """Fetch a public webpage and return its readable text (SSRF-guarded).

    Same {ok, text, ...} shape as extract_document. Strips script/style/nav noise
    via BeautifulSoup's built-in parser (no lxml dependency).
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "kind": "url", "text": "", "error": "no url provided"}
    if "://" not in url:
        url = "https://" + url  # bare domain → https

    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    current = url
    for _ in range(_MAX_REDIRECTS):
        _assert_public_url(current)  # re-validate EVERY hop (defeats redirect-based SSRF)
        resp = session.get(
            current, timeout=_URL_TIMEOUT, allow_redirects=False, stream=True,
            headers={"User-Agent": "hugpy-media-intelligence/1.0"},
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                break
            current = urljoin(current, loc)
            continue
        resp.raise_for_status()
        chunks, total = [], 0
        for chunk in resp.iter_content(8192):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_URL_BYTES:
                break
        html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        text = "\n".join(ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip())
        if not text:
            return {"ok": False, "kind": "url", "url": current, "text": "",
                    "error": "no readable text found at that URL"}
        return {"ok": True, "kind": "url", "url": current, "title": title,
                "text": text, "chars": len(text)}
    return {"ok": False, "kind": "url", "text": "", "error": "too many redirects"}


def assess_url(url: str) -> dict:
    """Structured, LLM-ready webpage assessment, upgrading the plain URL read.

    Delegates to abstract_webtools' assessManager.assess_webpage, which fetches
    cheaply (requests) and only escalates to a full browser render when the page
    comes back JS-walled / near-empty — returning title, meta-description, JSON-LD,
    same-domain links and a token-budgeted body alongside the readable text.

    Posture (consistent with the rest of this module):
      * SSRF — the initial URL is validated at the front door (http/https only,
        public addresses) exactly like fetch_url_text. NOTE: assessManager follows
        redirects internally, so individual redirect HOPS are NOT re-validated the
        way fetch_url_text does — the front-door check is the guarantee here.
      * Cost — abstract_webtools (Selenium / matplotlib) is imported lazily on the
        first call only; a forced browser render happens only as an auto-fallback
        for a near-empty page, never by default.
      * Graceful — if abstract_webtools is absent, errors, or yields no text, this
        degrades to the lightweight requests+bs4 fetch_url_text(), so the amenity
        is never worse than a plain read. Same {ok, kind, text, ...} contract.
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "kind": "url", "text": "", "error": "no url provided"}
    if "://" not in url:
        url = "https://" + url  # bare domain → https
    _assert_public_url(url)  # refuse internal/loopback before any fetch (raises PermissionError)

    try:
        from abstract_webtools.managers.assessManager import assess_webpage as _aw_assess
    except Exception:
        return fetch_url_text(url)  # abstract_webtools not installed → lightweight read

    try:
        page = _aw_assess(url, max_chars=_MAX_ASSESS_CHARS, max_links=_MAX_ASSESS_LINKS)
    except Exception:
        return fetch_url_text(url)  # assessment blew up → lightweight read

    text = (page.get("text") or "").strip() if isinstance(page, dict) else ""
    if not text:
        # assessment produced no usable body (render failed / empty) — plain reader.
        return fetch_url_text(url)

    return {
        "ok": True,
        "kind": "url",
        "url": page.get("url") or url,
        "title": page.get("title"),
        "description": page.get("description"),
        "text": text,
        "metadata": page.get("metadata") or [],
        "jsonld": page.get("jsonld") or [],
        "links": page.get("links") or [],
        "truncated": bool(page.get("truncated")),
        "render": page.get("render"),
        "chars": len(text),
    }
