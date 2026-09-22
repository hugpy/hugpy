"""Global central weight-transfer cap (the "pin-30 survives" protection).

Each puller (worker_agent/provision.py) opens up to HUGPY_PULL_CONCURRENCY
(default 8) concurrent segmented Range-GETs PER WORKER. A big assignment/pin
or a reconcile storm across K workers can therefore land ~8*K simultaneous
byte streams on central's two weight-serving endpoints (/file and /archive)
with no shared bound — this is what saturated central's link/CPU in the
2026-07-15 incident.

worker_routes.py now wraps both endpoints in a module-level
``threading.BoundedSemaphore`` (``_transfer_sem``), sized by
``HUGPY_CENTRAL_TRANSFER_MAX`` (default 3, floored to >=1) and read ONCE at
import time. The permit must be held for the STREAM's lifetime (client
finishes / disconnects / errors), not just the handler's — a naive
``with _transfer_sem:`` around the handler body releases the permit before
any bytes are actually sent, since both endpoints return a streaming
Response and the handler function returns immediately. This test proves:

  (a) default cap is 3, and HUGPY_CENTRAL_TRANSFER_MAX overrides it;
  (b) concurrency is actually bounded: with cap=2, two "in-flight" streams
      (held open via a blocking Event, not a sleep) occupy both permits, and
      a 3rd concurrent request is refused with 503 + Retry-After;
  (c) a permit is released when a stream finishes normally AND when it
      errors/disconnects mid-stream — proven by re-acquiring the same count
      afterward with no leak;
  (d) the 503 path carries Retry-After.

Run with the tree venv from .../dev/abstract_hugpy_dev:
    ./venv/bin/python -m pytest tests/test_central_transfer_cap.py -v
"""
import os
import sys
import importlib
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault(
    "PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-transfer-cap-test-"))

wr = importlib.import_module(
    "hugpy_server.app.routes.worker_routes")

from flask import Flask


# ── (a) default cap + env override ──────────────────────────────────────────
def test_default_cap_is_3(monkeypatch):
    monkeypatch.delenv("HUGPY_CENTRAL_TRANSFER_MAX", raising=False)
    assert wr._transfer_cap() == 3


def test_env_override_changes_cap(monkeypatch):
    monkeypatch.setenv("HUGPY_CENTRAL_TRANSFER_MAX", "7")
    assert wr._transfer_cap() == 7


def test_invalid_env_falls_back_to_3(monkeypatch):
    monkeypatch.setenv("HUGPY_CENTRAL_TRANSFER_MAX", "not-a-number")
    assert wr._transfer_cap() == 3


def test_zero_or_negative_floored_to_3(monkeypatch):
    monkeypatch.setenv("HUGPY_CENTRAL_TRANSFER_MAX", "0")
    assert wr._transfer_cap() == 3
    monkeypatch.setenv("HUGPY_CENTRAL_TRANSFER_MAX", "-5")
    assert wr._transfer_cap() == 3


# ── isolated semaphore-wrapping helper tests (no Flask/app needed) ──────────
# These exercise _TransferPermit / _acquire_transfer_permit directly against a
# throwaway semaphore, so the "does a permit leak" question is answered
# without going through HTTP at all.
def _fresh_gate(cap):
    """A private BoundedSemaphore + a short wait, standing in for the module
    globals so tests don't fight each other over the real _transfer_sem."""
    sem = threading.BoundedSemaphore(cap)
    return sem


def test_permit_release_is_idempotent():
    sem = _fresh_gate(1)
    assert sem.acquire(blocking=False) is True   # take the only slot
    permit = wr._TransferPermit(sem)              # wraps the held permit
    assert sem.acquire(blocking=False) is False   # cap=1, already held
    permit.release()
    # Second release must NOT raise (BoundedSemaphore.release over-release
    # would ValueError) and must NOT free a second slot.
    permit.release()
    assert sem.acquire(blocking=False) is True    # exactly one slot freed
    sem.release()


def test_acquire_transfer_permit_blocks_then_times_out(monkeypatch):
    monkeypatch.setattr(wr, "_transfer_sem", threading.BoundedSemaphore(1))
    monkeypatch.setattr(wr, "_TRANSFER_WAIT_S", 0.2)
    p1 = wr._acquire_transfer_permit()
    assert p1 is not None
    start = time.monotonic()
    p2 = wr._acquire_transfer_permit()   # cap exhausted -> times out, not None
    elapsed = time.monotonic() - start
    assert p2 is None
    assert elapsed >= 0.2
    p1.release()
    # No leak: releasing p1 frees the slot for a fresh acquire.
    p3 = wr._acquire_transfer_permit()
    assert p3 is not None
    p3.release()


# ── end-to-end over the real Flask blueprint ────────────────────────────────
MODEL_DIR = tempfile.mkdtemp(prefix="hugpy-transfer-cap-model-")
FILE_REL = "weights.bin"
SIZE = 256 * 1024
CONTENT = os.urandom(SIZE)
with open(os.path.join(MODEL_DIR, FILE_REL), "wb") as _fh:
    _fh.write(CONTENT)

FILE_URL = f"/llm/models/testmodel/file?path={FILE_REL}"
ARCHIVE_URL = "/llm/models/testmodel/archive"


@pytest.fixture()
def client(monkeypatch):
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
    monkeypatch.setattr(wr, "verify_enrollment_token", lambda tok: True,
                        raising=False)

    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    return app.test_client()


def _set_cap(monkeypatch, n):
    """Install a fresh BoundedSemaphore(n) as the module's transfer gate, so
    each test gets an isolated cap instead of racing the real module-level
    one (which other tests / the app's own state also touch)."""
    monkeypatch.setattr(wr, "_transfer_sem", threading.BoundedSemaphore(n))
    monkeypatch.setattr(wr, "_TRANSFER_WAIT_S", 0.3)


def test_bounded_concurrency_and_503_with_retry_after(client, monkeypatch):
    """cap=2: hold 2 streams open via a blocking read, prove a 3rd concurrent
    request gets 503 + Retry-After, then prove releasing frees the gate."""
    _set_cap(monkeypatch, 2)

    release_gate = threading.Event()
    entered = threading.Event()
    entered_count = {"n": 0}
    lock = threading.Lock()

    real_stream = wr._stream_file_window

    def blocking_stream(path, start, end):
        with lock:
            entered_count["n"] += 1
            if entered_count["n"] <= 2:
                entered.set()
        # Block the generator mid-stream (after first chunk) until told to
        # proceed — this holds the permit "in flight" without a sleep race.
        gen = real_stream(path, start, end)
        first = next(gen)
        yield first
        release_gate.wait(timeout=5)
        yield from gen

    monkeypatch.setattr(wr, "_stream_file_window", blocking_stream)

    results = {}

    def do_range_request(name, start, end):
        r = client.get(FILE_URL, headers={"Range": f"bytes={start}-{end}"})
        results[name] = r
        r.get_data()  # fully drain -> fires the generator's finally/release

    t1 = threading.Thread(target=do_range_request, args=("t1", 0, 99))
    t2 = threading.Thread(target=do_range_request, args=("t2", 100, 199))
    t1.start()
    t2.start()

    # Wait until both in-flight requests have actually entered the streaming
    # generator (i.e. both permits are held), not just "thread started".
    deadline = time.monotonic() + 5
    while entered_count["n"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entered_count["n"] == 2, "both in-flight streams must hold a permit"

    # A 3rd concurrent request must be refused: both permits are held by t1/t2.
    r3 = client.get(FILE_URL, headers={"Range": "bytes=200-210"})
    assert r3.status_code == 503
    assert r3.headers.get("Retry-After") is not None

    # Let the two in-flight streams finish and release their permits.
    release_gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert results["t1"].status_code == 206
    assert results["t2"].status_code == 206

    # Gate is free again -> a 4th request now succeeds.
    r4 = client.get(FILE_URL, headers={"Range": "bytes=0-9"})
    assert r4.status_code == 206
    r4.get_data()


def test_503_has_retry_after_header_value(client, monkeypatch):
    _set_cap(monkeypatch, 1)
    # Exhaust the single permit directly (no need to spin up threads for this
    # one — just hold it open) then confirm the response body + header.
    permit = wr._acquire_transfer_permit()
    assert permit is not None
    try:
        r = client.get(FILE_URL)
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "2"
    finally:
        permit.release()


# ── (c) no leak on NORMAL finish (full-file send_file path) ────────────────
def test_permit_released_on_full_file_finish(client, monkeypatch):
    _set_cap(monkeypatch, 1)

    r1 = client.get(FILE_URL)   # full-file GET -> send_file + call_on_close
    assert r1.status_code == 200
    assert r1.get_data() == CONTENT
    # Simulate what the WSGI server does when it's done writing the response:
    # close the iterable. This is what fires call_on_close(permit.release).
    r1.close()

    # Permit must be free again -> immediate second full-file GET succeeds
    # rather than 503ing (proves no leak on the send_file/call_on_close path).
    r2 = client.get(FILE_URL)
    assert r2.status_code == 200
    r2.close()


# ── (c) no leak on ERROR / DISCONNECT mid-stream (ranged + archive paths) ──
def test_permit_released_on_ranged_stream_error(client, monkeypatch):
    _set_cap(monkeypatch, 1)
    real_stream = wr._stream_file_window

    def boom(path, start, end):
        yield b"partial-bytes"
        raise ConnectionError("simulated client disconnect mid-stream")

    monkeypatch.setattr(wr, "_stream_file_window", boom)

    r1 = client.get(FILE_URL, headers={"Range": "bytes=0-99"})
    assert r1.status_code == 206
    with pytest.raises(ConnectionError):
        r1.get_data()   # draining the generator surfaces the raised error

    # Despite the mid-stream error, the generator's `finally: permit.release()`
    # must have run -> the single permit is free again immediately. Restore
    # the real (non-raising) streamer for this second request so we're
    # checking permit availability, not re-triggering the injected error.
    monkeypatch.setattr(wr, "_stream_file_window", real_stream)
    r2 = client.get(FILE_URL, headers={"Range": "bytes=0-9"})
    assert r2.status_code == 206
    r2.get_data()


def test_permit_released_on_archive_stream_disconnect(client, monkeypatch):
    _set_cap(monkeypatch, 1)

    r1 = client.get(ARCHIVE_URL)
    assert r1.status_code == 200
    # Simulate a client disconnecting partway through: close the response
    # without fully draining it. werkzeug's TestResponse.close() closes the
    # underlying app_iter, which for a generator triggers GeneratorExit ->
    # the generator's `finally: thread.join(); permit.release()` still runs.
    r1.close()

    r2 = client.get(ARCHIVE_URL)
    assert r2.status_code == 200
    r2.close()


def test_archive_bounded_concurrency_503(client, monkeypatch):
    _set_cap(monkeypatch, 1)
    permit = wr._acquire_transfer_permit()
    assert permit is not None
    try:
        r = client.get(ARCHIVE_URL)
        assert r.status_code == 503
        assert r.headers.get("Retry-After") is not None
    finally:
        permit.release()
