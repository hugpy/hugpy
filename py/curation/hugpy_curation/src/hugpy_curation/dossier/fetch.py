"""k120 — the ONE way this package talks to the outside world.

Everything external the dossier wants — an arXiv abstract, a subreddit's JSON,
Hacker News' search index — is a nice-to-have on a job that runs unattended at
03:20 and must finish. So there is exactly one fetch helper, and it is built
around three promises:

  IT NEVER RAISES. Every call returns ``FetchResult``; a timeout, a 429, a DNS
  failure and a body that will not parse are all ``ok=False`` with a ``detail``
  string that ends up in the dossier's ``unavailable`` list verbatim. Discovery
  is never blocked on a source.

  IT IS POLITE. A real User-Agent naming the project and a per-HOST minimum
  interval (``MIN_INTERVAL_S``, default 2s) enforced in-process. Public
  endpoints only — nothing here logs in, and nothing here reads a page it was
  not offered as JSON.

  IT CACHES TO DISK. The timer is NIGHTLY, so a 20-hour TTL means a re-run in
  the same night costs nothing and a source that rate-limits us once is not hit
  again in a loop. The cache is also what ``radar.py`` re-reads: the gem scan
  is a SECOND pass over the SAME pulls, never a second round of requests.

Deliberately urllib, not requests: this module is imported by a systemd timer
on a worker box whose venv is not guaranteed to carry anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Identifies us honestly to the endpoints we read. Public API etiquette on
#: Reddit and HN is "say who you are"; an anonymous scraper gets blocked, and
#: deserves to be.
USER_AGENT: str = os.environ.get("DOSSIER_USER_AGENT") or (
    "hugpy-model-discovery/1.0 (self-hosted model reviewer; "
    "contact via the operator of this instance)")

#: Per-host politeness floor, seconds. Reddit's public JSON tolerates roughly
#: one request every couple of seconds unauthenticated; we take that as the
#: rule for everyone.
MIN_INTERVAL_S: float = float(os.environ.get("DOSSIER_MIN_INTERVAL") or 2.0)

#: Hard per-request ceiling. Short on purpose — see the module docstring.
TIMEOUT_S: float = float(os.environ.get("DOSSIER_FETCH_TIMEOUT") or 8.0)

#: Nightly timer, so a cache entry survives comfortably longer than one night's
#: run and shorter than the gap to the next.
CACHE_TTL_S: float = float(os.environ.get("DOSSIER_CACHE_TTL") or 20 * 3600)

#: Refuse to load a body larger than this into memory. A model card can be big;
#: a subreddit listing should not be.
MAX_BYTES: int = int(os.environ.get("DOSSIER_MAX_BYTES") or 4 * 1024 * 1024)

#: Kill switch for boxes with no outbound network at all. When set, every fetch
#: answers ``ok=False`` with an honest reason and nothing is attempted.
OFFLINE_ENV: str = "DOSSIER_OFFLINE"

_last_hit: dict[str, float] = {}
_lock = threading.Lock()


@dataclass(slots=True)
class FetchResult:
    """What came back, or honestly why nothing did."""
    url: str
    ok: bool = False
    status: int | None = None
    text: str = ""
    detail: str = ""
    fetched_at: float = field(default_factory=time.time)
    from_cache: bool = False

    def json(self) -> Any:
        """Parsed body, or None. A body that will not parse is not an error
        the caller has to handle — it is an absent source."""
        if not self.ok or not self.text:
            return None
        try:
            return json.loads(self.text)
        except ValueError:
            return None


def cache_dir() -> str:
    """``DOSSIER_CACHE_DIR``, else ``<cache_dir>/discovery-dossier`` — see
    :mod:`hugpy_curation.config`."""
    from hugpy_curation.config import fetch_cache_dir
    d = fetch_cache_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _cache_path(url: str) -> str:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return os.path.join(cache_dir(), f"{key}.json")


def read_cache(url: str, ttl: float | None = None) -> FetchResult | None:
    """A cached body still inside its TTL, or None. Used by ``radar.py`` to
    re-read the night's pulls without touching the network."""
    path = _cache_path(url)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    when = float(blob.get("fetched_at") or 0)
    if time.time() - when > (CACHE_TTL_S if ttl is None else ttl):
        return None
    return FetchResult(url=url, ok=True, status=blob.get("status"),
                       text=blob.get("text") or "", fetched_at=when,
                       from_cache=True, detail="from cache")


def _write_cache(result: FetchResult) -> None:
    try:
        tmp = _cache_path(result.url) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"url": result.url, "status": result.status,
                       "text": result.text, "fetched_at": result.fetched_at},
                      fh)
        os.replace(tmp, _cache_path(result.url))
    except OSError:
        pass                                     # a cache miss is not a failure


def cached_urls(prefix: str = "") -> list[str]:
    """Every URL currently in the cache (optionally filtered), newest first.
    This is the radar's input: the night's pulls, already paid for."""
    rows: list[tuple[float, str]] = []
    try:
        names = os.listdir(cache_dir())
    except OSError:
        return []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(cache_dir(), name), "r",
                      encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        url = blob.get("url") or ""
        if prefix and not url.startswith(prefix):
            continue
        rows.append((float(blob.get("fetched_at") or 0), url))
    rows.sort(reverse=True)
    return [url for _when, url in rows]


def _wait_for_host(url: str) -> None:
    host = urllib.parse.urlparse(url).netloc
    with _lock:
        last = _last_hit.get(host, 0.0)
        gap = MIN_INTERVAL_S - (time.time() - last)
        if gap > 0:
            time.sleep(min(gap, MIN_INTERVAL_S))
        _last_hit[host] = time.time()


def fetch(url: str, *, timeout: float | None = None,
          headers: Mapping[str, str] | None = None,
          ttl: float | None = None, use_cache: bool = True) -> FetchResult:
    """GET ``url`` politely, through the disk cache. Never raises."""
    if use_cache:
        hit = read_cache(url, ttl)
        if hit is not None:
            return hit
    if os.environ.get(OFFLINE_ENV):
        return FetchResult(url=url, ok=False,
                           detail=f"offline ({OFFLINE_ENV} is set) — not fetched")
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json, text/plain;q=0.8, */*;q=0.5")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    _wait_for_host(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT_S) as resp:
            raw = resp.read(MAX_BYTES)
            status = getattr(resp, "status", None) or resp.getcode()
        text = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return FetchResult(url=url, ok=False, status=exc.code,
                           detail=f"HTTP {exc.code} {exc.reason}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return FetchResult(url=url, ok=False,
                           detail=f"{type(exc).__name__}: {exc}")
    result = FetchResult(url=url, ok=True, status=status, text=text)
    if use_cache:
        _write_cache(result)
    return result


__all__ = ["CACHE_TTL_S", "FetchResult", "MIN_INTERVAL_S", "TIMEOUT_S",
           "USER_AGENT", "cache_dir", "cached_urls", "fetch", "read_cache"]
