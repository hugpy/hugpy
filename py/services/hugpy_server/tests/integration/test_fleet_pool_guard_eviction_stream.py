"""Pool-guard checks that drive hugpy_server.app.routes.eviction_routes (moved
from hugpy_fleet/tests/test_pool_guard.py: they import the server, which the
fleet package must not)."""

"""k59 — the fast-read reserve: SSE streams may not take the last threads.

Central is `gunicorn --workers 1 --threads N`. A long-lived stream holds one of
those N threads for its whole life, so enough open feeds starve every ordinary
request — the operator's "blips over just a few calls". STREAM_MAX_S bounds how
LONG one stream holds a thread; this reserve bounds how MANY do.

The property under test: past the cap a NEW stream is refused immediately and
honestly (503 + Retry-After, a message that says why), the slot is returned when
the stream ends OR when the client walks away mid-stream, and a refusal never
blocks — a stream waiting for a slot would still be holding the thread it was
trying not to monopolize.

Run: venv/bin/python -m pytest tests/test_pool_guard.py -q
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("HUGPY_COMMS_DB", "off")

from hugpy_fleet.central import pool_guard


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HUGPY_STREAM_SLOTS", "HUGPY_GUNICORN_THREADS"):
        monkeypatch.delenv(var, raising=False)
    pool_guard.reset()
    yield
    pool_guard.reset()


def test_eviction_stream_refuses_past_the_cap_and_still_serves_reads(monkeypatch):
    """End to end on the endpoint the k59 task names: past the cap the STREAM
    is refused, and the ordinary read beside it still answers."""
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    monkeypatch.setenv("HUGPY_AUTH_MODE", "open")
    pool_guard.reset()

    import importlib
    from flask import Flask
    er = importlib.import_module(
        "hugpy_server.app.routes.eviction_routes")
    monkeypatch.setattr(er, "_operator_or_worker", lambda: True)

    app = Flask("pool-guard-test")
    app.register_blueprint(er.eviction_bp)
    client = app.test_client()

    held = pool_guard.stream_slot(); held.__enter__()
    try:
        r = client.get("/llm/evictions/stream")
        assert r.status_code == 503
        assert r.headers.get("Retry-After")
        assert r.get_json()["error"]["code"] == "StreamCapacity"
        # ...and the fast read on the same blueprint is unaffected.
        assert client.get("/llm/evictions").status_code == 200
    finally:
        held.__exit__(None, None, None)


def test_eviction_stream_returns_its_slot_when_the_client_leaves(monkeypatch):
    monkeypatch.setenv("HUGPY_STREAM_SLOTS", "1")
    pool_guard.reset()

    import importlib
    from flask import Flask
    er = importlib.import_module(
        "hugpy_server.app.routes.eviction_routes")
    monkeypatch.setattr(er, "_operator_or_worker", lambda: True)

    app = Flask("pool-guard-test-2")
    app.register_blueprint(er.eviction_bp)
    client = app.test_client()

    r = client.get("/llm/evictions/stream")
    assert r.status_code == 200
    assert pool_guard.snapshot()["held"] == 1
    r.close()                       # the tab goes away mid-stream
    assert pool_guard.snapshot()["held"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
