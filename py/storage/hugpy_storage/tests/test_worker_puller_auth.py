"""Worker PULLER attaches its enrollment bearer token on central transfers.

Companion to test_worker_transfer_auth_range.py (the SERVER-side gate). Here we
prove the worker-side puller in ``worker_agent/provision.py`` attaches
``Authorization: Bearer <token>`` on EVERY central file-transfer request when an
enrollment token is available, and omits the header entirely when none is — so
provisioning keeps working once ``HUGPY_WORKER_ENROLL_REQUIRED`` is turned on,
and NOTHING changes while it's off (a pure superset of today's behavior).

The token is sourced exactly as CentralClient's is: whatever the agent passes to
``set_enroll_token`` (its ``args.token`` = ``--token`` / ``WORKER_ENROLL_TOKEN``),
falling back to the same ``WORKER_ENROLL_TOKEN`` env var.

Run with the tree venv:
    venv/bin/python -m pytest tests/test_worker_puller_auth.py -v
"""
import os
import sys
import hashlib
import importlib
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep any settings/audit writes out of the real projects tree.
os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-puller-test-"))

prov = importlib.import_module("hugpy_storage.provision")

TOKEN = "hpw_puller_test_token"
# The exact header CentralClient._post builds (f"Bearer {self.token}"); the
# puller must match it byte-for-byte.
EXPECT = f"Bearer {TOKEN}"


class _FakeResp:
    """Minimal urlopen response: context manager + read()/getcode()."""

    def __init__(self, body: bytes = b"", code: int = 200):
        self._buf = body
        self._pos = 0
        self._code = code

    def read(self, n=-1):
        if n is None or n < 0:
            data = self._buf[self._pos:]
            self._pos = len(self._buf)
            return data
        data = self._buf[self._pos:self._pos + n]
        self._pos += len(data)
        return data

    def getcode(self):
        return self._code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def cap(monkeypatch):
    """Patch urllib.request.urlopen (as provision sees it) to record the
    Authorization header of the Request each transfer helper builds, and return a
    canned response. Baseline is tokenless — every test sets its own token."""
    seen = {"auth": "<unset>", "url": None}
    holder = {"resp": lambda: _FakeResp(b"", 200)}

    # Fresh, tokenless baseline (the gradual-rollout default the superset relies
    # on): no explicit token AND no env token.
    monkeypatch.delenv("WORKER_ENROLL_TOKEN", raising=False)
    prov.set_enroll_token(None)

    def _fake_urlopen(req, timeout=None, **kw):
        # Every patched call site passes a urllib Request (never a bare str).
        seen["auth"] = (req.get_header("Authorization")
                        if hasattr(req, "get_header") else None)
        seen["url"] = getattr(req, "full_url", req)
        return holder["resp"]()

    monkeypatch.setattr(prov.urllib.request, "urlopen", _fake_urlopen)
    seen["holder"] = holder
    yield seen
    prov.set_enroll_token(None)


# ── the token-source layer itself ──────────────────────────────────────────
def test_auth_headers_present_and_absent(cap):
    assert prov._auth_headers() == {}          # tokenless baseline
    prov.set_enroll_token(TOKEN)
    assert prov._auth_headers() == {"Authorization": EXPECT}


def test_auth_request_merges_without_clobbering_range(cap):
    prov.set_enroll_token(TOKEN)
    req = prov._auth_request("http://c/file?path=w", {"Range": "bytes=0-9"})
    assert req.get_header("Authorization") == EXPECT
    assert req.get_header("Range") == "bytes=0-9"
    # Tokenless -> Range preserved, NO Authorization added.
    prov.set_enroll_token(None)
    req = prov._auth_request("http://c/file?path=w", {"Range": "bytes=0-9"})
    assert req.get_header("Authorization") is None
    assert req.get_header("Range") == "bytes=0-9"


def test_env_fallback_sources_same_var_as_central_client(cap, monkeypatch):
    # No explicit set_enroll_token, but WORKER_ENROLL_TOKEN set -> header present.
    # This is the same env var CentralClient's --token defaults to.
    monkeypatch.setenv("WORKER_ENROLL_TOKEN", TOKEN)
    prov.set_enroll_token(None)
    assert prov._auth_headers() == {"Authorization": EXPECT}


def test_explicit_token_takes_precedence_over_env(cap, monkeypatch):
    monkeypatch.setenv("WORKER_ENROLL_TOKEN", "hpw_env_tok")
    prov.set_enroll_token("hpw_explicit_tok")
    assert prov._enroll_token() == "hpw_explicit_tok"


# ── _get_json: manifest / model-list / chunksums all funnel through here ────
def test_get_json_token_present(cap):
    prov.set_enroll_token(TOKEN)
    cap["holder"]["resp"] = lambda: _FakeResp(b'{"ok": true}', 200)
    assert prov._get_json("http://c/api/llm/models/m/manifest") == {"ok": True}
    assert cap["auth"] == EXPECT


def test_get_json_token_absent(cap):
    cap["holder"]["resp"] = lambda: _FakeResp(b'{"ok": true}', 200)
    prov._get_json("http://c/api/llm/models/m/manifest")
    assert cap["auth"] is None


# ── _supports_range (Range probe) ──────────────────────────────────────────
def test_supports_range_token_present(cap):
    prov.set_enroll_token(TOKEN)
    cap["holder"]["resp"] = lambda: _FakeResp(b"", 206)
    assert prov._supports_range("http://c/file?path=w") is True
    assert cap["auth"] == EXPECT


def test_supports_range_token_absent(cap):
    cap["holder"]["resp"] = lambda: _FakeResp(b"", 206)
    assert prov._supports_range("http://c/file?path=w") is True
    assert cap["auth"] is None


# ── _download_file (whole-file stream) ─────────────────────────────────────
def test_download_file_token_present(cap, tmp_path):
    prov.set_enroll_token(TOKEN)
    cap["holder"]["resp"] = lambda: _FakeResp(b"", 200)   # empty body -> no-op write
    prov._download_file("http://c/file?path=w", str(tmp_path / "w.bin"), None)
    assert cap["auth"] == EXPECT


def test_download_file_token_absent(cap, tmp_path):
    cap["holder"]["resp"] = lambda: _FakeResp(b"", 200)
    prov._download_file("http://c/file?path=w", str(tmp_path / "w.bin"), None)
    assert cap["auth"] is None


# ── _download_segment (parallel byte-range) ────────────────────────────────
def test_download_segment_token_present(cap, tmp_path):
    prov.set_enroll_token(TOKEN)
    cap["holder"]["resp"] = lambda: _FakeResp(b"abcd", 206)
    dest = tmp_path / "seg.bin"
    dest.write_bytes(b"\x00\x00\x00\x00")
    prov._download_segment("http://c/file?path=w", str(dest), 0, 3)
    assert cap["auth"] == EXPECT
    assert dest.read_bytes() == b"abcd"          # Range semantics still intact


def test_download_segment_token_absent(cap, tmp_path):
    cap["holder"]["resp"] = lambda: _FakeResp(b"abcd", 206)
    dest = tmp_path / "seg.bin"
    dest.write_bytes(b"\x00\x00\x00\x00")
    prov._download_segment("http://c/file?path=w", str(dest), 0, 3)
    assert cap["auth"] is None


# ── _fetch_chunk_verified (content-verified chunk) ─────────────────────────
def test_fetch_chunk_verified_token_present(cap, tmp_path):
    prov.set_enroll_token(TOKEN)
    data = b"abcd"
    cap["holder"]["resp"] = lambda: _FakeResp(data, 206)
    part = tmp_path / "w.part"
    part.write_bytes(b"\x00" * 4)
    prov._fetch_chunk_verified("http://c/file?path=w", str(part), 0, 4, 4,
                               hashlib.sha256(data).hexdigest())
    assert cap["auth"] == EXPECT
    assert part.read_bytes() == data             # hashing/verify untouched


def test_fetch_chunk_verified_token_absent(cap, tmp_path):
    data = b"abcd"
    cap["holder"]["resp"] = lambda: _FakeResp(data, 206)
    part = tmp_path / "w.part"
    part.write_bytes(b"\x00" * 4)
    prov._fetch_chunk_verified("http://c/file?path=w", str(part), 0, 4, 4,
                               hashlib.sha256(data).hexdigest())
    assert cap["auth"] is None


# ── fetch_archive_from_central (streamed tar) ──────────────────────────────
class _Stop(Exception):
    """Sentinel to halt the archive fetch right after the /archive request is
    built — we only need to inspect the header it carried."""


def _drive_archive(cap, monkeypatch, tmp_path):
    """Run fetch_archive_from_central just up to its /archive urlopen, capturing
    the Request header. Stubs manifest + destination so no real registry/network
    is touched."""
    monkeypatch.setattr(prov, "_get_json",
                        lambda url, timeout=30.0: {"files": [], "total_bytes": 0,
                                                   "hub_id": "org/m"})
    monkeypatch.setattr(prov, "_local_destination", lambda meta: str(tmp_path))

    def _capture_then_stop(req, timeout=None, **kw):
        cap["auth"] = (req.get_header("Authorization")
                       if hasattr(req, "get_header") else None)
        cap["url"] = getattr(req, "full_url", req)
        raise _Stop()

    monkeypatch.setattr(prov.urllib.request, "urlopen", _capture_then_stop)
    with pytest.raises(_Stop):
        prov.fetch_archive_from_central("http://c", "m")
    assert cap["url"].endswith("/archive")


def test_archive_token_present(cap, monkeypatch, tmp_path):
    prov.set_enroll_token(TOKEN)
    _drive_archive(cap, monkeypatch, tmp_path)
    assert cap["auth"] == EXPECT


def test_archive_token_absent(cap, monkeypatch, tmp_path):
    _drive_archive(cap, monkeypatch, tmp_path)
    assert cap["auth"] is None
