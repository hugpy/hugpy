"""Worker file-transfer endpoints: authentication + seek-based Range serving.

Covers the security hardening of central's model-weight transfer surface
(worker_routes.py):

  * Range GETs now do a real ``file.seek(start)`` and stream only the requested
    window (206 + correct Content-Range/Content-Length/Accept-Ranges), instead of
    werkzeug's O(offset) ``send_file(conditional=True)`` range wrapper that a few
    large-offset requests could ride into an O(offset^2) CPU-exhaustion DoS.
    We assert CORRECTNESS at a large offset (the exact bytes), not timing.
  * The transfer endpoints (/file, /manifest, /chunksums, /archive) require a
    credential: operator auth OR a valid worker enrollment bearer token. A
    tokenless caller is refused (401) once enrollment is required.
  * register returns 400 on a malformed body (was an unhandled 500).

Run with the tree venv:
    venv/bin/python -m pytest tests/test_worker_transfer_auth_range.py -v
"""
import os
import sys
import importlib
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep any audit / settings writes out of the real projects tree.
os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-transfer-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

from flask import Flask

# ── a temp "model" directory with one real, largeish file ──────────────────
MODEL_DIR = tempfile.mkdtemp(prefix="hugpy-transfer-model-")
FILE_REL = "weights.bin"
SIZE = 5 * 1024 * 1024 + 123          # odd size so suffix/clamp math is exercised
CONTENT = os.urandom(SIZE)
with open(os.path.join(MODEL_DIR, FILE_REL), "wb") as _fh:
    _fh.write(CONTENT)

VALID_TOKEN = "hpw_valid_test_token"
AUTH = {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture()
def client(monkeypatch):
    """Blueprint on a bare Flask app, with the registry + token verifier stubbed.

    ``_model_dir_or_404`` calls the bare names ``get_models_dict`` /
    ``route_destination``; ``_enrollment_ok`` calls the bare name
    ``verify_enrollment_token`` — all module globals, so we patch them on ``wr``
    exactly like the other route tests do.
    """
    # A clean auth baseline: external mode, no operator token, enrollment NOT
    # required (the default gradual-rollout posture). Individual tests override.
    monkeypatch.delenv("HUGPY_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("HUGPY_WORKER_ENROLL_REQUIRED", raising=False)
    monkeypatch.setenv("HUGPY_AUTH_MODE", "external")

    monkeypatch.setattr(
        wr, "get_models_dict",
        lambda dict_return=True: {
            "testmodel": {"key": "testmodel", "hub_id": "org/testmodel",
                          "name": "testmodel"}},
        raising=False)
    monkeypatch.setattr(wr, "route_destination", lambda model: MODEL_DIR,
                        raising=False)
    monkeypatch.setattr(wr, "verify_enrollment_token",
                        lambda tok: tok == VALID_TOKEN, raising=False)

    # Give each test a FRESH global transfer-cap semaphore. worker_routes.py
    # wraps /file and /archive in a module-level BoundedSemaphore (the
    # 2026-07-15 "pin-30 survives" cap) whose permits are released when a
    # streamed Response is closed/drained — but this test file predates that
    # cap and never calls response.close()/get_data() on every 200 it makes
    # (a real WSGI server always closes the iterable it's handed; Flask's
    # bare test client does not, unless you drain or close explicitly). Without
    # this reset, permits "leak" across tests in this file (not in
    # production) and later tests would spuriously 503. Test isolation only —
    # does not change any assertion below.
    import threading as _th
    monkeypatch.setattr(wr, "_transfer_sem", _th.BoundedSemaphore(wr._TRANSFER_CAP))

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


FILE_URL = f"/llm/models/testmodel/file?path={FILE_REL}"


# ── (d) full-file GET still works ──────────────────────────────────────────
def test_full_file_get_200(client):
    r = client.get(FILE_URL, headers=AUTH)
    assert r.status_code == 200
    body = r.get_data()
    assert len(body) == SIZE
    assert body == CONTENT
    assert r.headers.get("Accept-Ranges") == "bytes"


# ── (a) large-offset Range -> 206, exact bytes, correct Content-Range ──────
def test_large_offset_range_206(client):
    start, end = 4_000_000, 4_000_099          # a deep offset: seek, don't scan
    r = client.get(FILE_URL, headers={**AUTH, "Range": f"bytes={start}-{end}"})
    assert r.status_code == 206
    assert r.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
    assert r.headers["Content-Length"] == str(end - start + 1)
    assert r.headers.get("Accept-Ranges") == "bytes"
    assert r.get_data() == CONTENT[start:end + 1]


def test_open_ended_range_to_eof(client):
    start = SIZE - 500
    r = client.get(FILE_URL, headers={**AUTH, "Range": f"bytes={start}-"})
    assert r.status_code == 206
    assert r.headers["Content-Range"] == f"bytes {start}-{SIZE - 1}/{SIZE}"
    assert r.get_data() == CONTENT[start:]


def test_suffix_range(client):
    n = 250
    r = client.get(FILE_URL, headers={**AUTH, "Range": f"bytes=-{n}"})
    assert r.status_code == 206
    assert r.headers["Content-Range"] == f"bytes {SIZE - n}-{SIZE - 1}/{SIZE}"
    assert r.get_data() == CONTENT[-n:]


def test_range_end_overlong_is_clamped(client):
    start = SIZE - 10
    r = client.get(FILE_URL,
                   headers={**AUTH, "Range": f"bytes={start}-{SIZE + 9999}"})
    assert r.status_code == 206
    assert r.headers["Content-Range"] == f"bytes {start}-{SIZE - 1}/{SIZE}"
    assert r.get_data() == CONTENT[start:]


# ── (e) unsatisfiable range -> 416 ─────────────────────────────────────────
def test_unsatisfiable_range_416(client):
    r = client.get(FILE_URL, headers={**AUTH, "Range": f"bytes={SIZE}-{SIZE + 5}"})
    assert r.status_code == 416
    assert r.headers["Content-Range"] == f"bytes */{SIZE}"


def test_multirange_serves_whole_file_200(client):
    # We never emit multipart/byteranges — a multi-range request gets the whole
    # file (200). Crucially it must NOT spin.
    r = client.get(FILE_URL, headers={**AUTH, "Range": "bytes=0-10,20-30"})
    assert r.status_code == 200
    assert r.get_data() == CONTENT


# ── (b) unauthenticated -> 401 when enrollment is required ─────────────────
def test_unauthenticated_transfer_401(client, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    assert client.get(FILE_URL).status_code == 401
    assert client.get("/llm/models/testmodel/manifest").status_code == 401
    assert client.get(
        f"/llm/models/testmodel/chunksums?path={FILE_REL}").status_code == 401
    assert client.get("/llm/models/testmodel/archive").status_code == 401


def test_bad_token_denied_even_when_rollout_off(client):
    # A present-but-invalid token is ALWAYS denied, even during gradual rollout.
    r = client.get(FILE_URL, headers={"Authorization": "Bearer hpw_bogus"})
    assert r.status_code == 401


# ── (c) authenticated -> 200/206 even when enrollment is required ──────────
def test_authenticated_transfer_ok_when_required(client, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    assert client.get(FILE_URL, headers=AUTH).status_code == 200
    assert client.get("/llm/models/testmodel/manifest",
                      headers=AUTH).status_code == 200
    assert client.get(f"/llm/models/testmodel/chunksums?path={FILE_REL}",
                      headers=AUTH).status_code == 200
    r = client.get(FILE_URL, headers={**AUTH, "Range": "bytes=0-99"})
    assert r.status_code == 206


def test_operator_token_passes(client, monkeypatch):
    # Operator token authorizes transfers (open mode + configured token).
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    monkeypatch.setenv("HUGPY_OPERATOR_TOKEN", "op-secret")
    r = client.get(FILE_URL, headers={"X-Operator-Token": "op-secret"})
    assert r.status_code == 200


# ── gradual-rollout: tokenless allowed while enrollment is NOT required ─────
def test_tokenless_allowed_when_rollout_off(client):
    # Documents the deliberate posture: the O(n^2) DoS is fixed unconditionally,
    # but tokenless transfers still work until the keeper flips enrollment on.
    assert client.get(FILE_URL).status_code == 200


# ── manifest correctness (authed) ──────────────────────────────────────────
def test_manifest_lists_file(client):
    r = client.get("/llm/models/testmodel/manifest", headers=AUTH)
    assert r.status_code == 200
    data = r.get_json()
    assert data["total_bytes"] == SIZE
    assert any(f["path"] == FILE_REL and f["size"] == SIZE
               for f in data["files"])


# ── register: malformed body -> 400 (not 500); 401 precedence ──────────────
def test_register_bad_body_400(client):
    # enrollment not required -> tokenless passes the gate; missing "name" is a
    # client error (400), not an unhandled ValidationError (500).
    assert client.post("/llm/workers/register", json={}).status_code == 400


def test_register_401_precedes_body_check(client, monkeypatch):
    monkeypatch.setenv("HUGPY_WORKER_ENROLL_REQUIRED", "1")
    assert client.post("/llm/workers/register", json={}).status_code == 401
