"""or-k16 — producer attribution persisted in the reliability ledger, shared
across processes, written through to central, served by oracle routes.

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  ./venv/bin/python -m pytest tests/test_oracle_selection_producers.py -q
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from hugpy_oracle import selection
from hugpy_oracle.selection import ReliabilityLedger, Selector


@pytest.fixture()
def db(tmp_path):
    return os.path.join(str(tmp_path), "reliability.sqlite")


@pytest.fixture()
def bound(db, monkeypatch):
    """Bind the process selector to a fresh ledger at ``db``; clear the cache."""
    monkeypatch.setenv(selection.LEDGER_PATH_ENV, db)
    monkeypatch.delenv(selection.LEDGER_REMOTE_ENV, raising=False)
    monkeypatch.delenv(selection.SELECTION_DISABLE_ENV, raising=False)
    ledger = ReliabilityLedger(db)
    sel = Selector(ledger=ledger, get_view=lambda cap: None, get_matrix=lambda: None)
    monkeypatch.setattr(selection, "_PROCESS_SELECTOR", sel)
    selection._PRODUCERS.clear()
    yield ledger
    ledger.close()


# --------------------------------------------------------------------------- #
# ledger table
# --------------------------------------------------------------------------- #


def test_ledger_producers_table_roundtrip(db):
    led = ReliabilityLedger(db)
    assert led.producer("mct://x") is None
    led.remember_producer("mct://x", "image.generate", "flux", worker="w1")
    row = led.producer("mct://x")
    assert (row["capability"], row["model_id"], row["worker"]) == ("image.generate", "flux", "w1")
    assert row["ts"]
    led.remember_producer("mct://x", "image.generate", "sdxl", worker="w2")   # re-attribution upserts
    assert led.producer("mct://x")["model_id"] == "sdxl"
    assert led.producer_count() == 1
    s = led.summary()
    assert s["producers"] == 1 and s["outcomes"] == 0
    assert s["producers_by_model"][0]["model_id"] == "sdxl"
    assert s["workers"] == [{"worker": "w2", "n": 1}]
    led.close()


def test_two_ledger_instances_same_file_see_each_other(db):
    a = ReliabilityLedger(db)
    b = ReliabilityLedger(db)
    a.remember_producer("ref-a", "audio.tts", "kokoro", worker="A")
    assert b.producer("ref-a")["model_id"] == "kokoro"
    b.remember_producer("ref-b", "audio.tts", "chatterbox", worker="B")
    assert a.producer("ref-b")["worker"] == "B"
    assert a.producer_count() == b.producer_count() == 2
    a.close(); b.close()


def test_cross_process_producer_visible(db):
    """A real second interpreter writes; this process reads (and vice versa)."""
    child = textwrap.dedent(f"""
        from hugpy_oracle.selection import ReliabilityLedger
        led = ReliabilityLedger({db!r})
        led.remember_producer("from-child", "video.generate", "wan", worker="child")
        row = led.producer("from-parent")
        print(row["model_id"] if row else "MISSING")
    """)
    parent = ReliabilityLedger(db)
    parent.remember_producer("from-parent", "text.generate", "qwen", worker="parent")
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "qwen"
    assert parent.producer("from-child")["model_id"] == "wan"
    parent.close()


# --------------------------------------------------------------------------- #
# module-level API: cache in front of the ledger
# --------------------------------------------------------------------------- #


def test_remember_producer_writes_ledger_and_cache(bound):
    selection.remember_producer("r1", "image.generate", "flux")
    assert selection._PRODUCERS["r1"] == ("image.generate", "flux")
    row = bound.producer("r1")
    assert row["model_id"] == "flux" and "@" in row["worker"]
    assert selection.producer_stamp("r1") == {"capability": "image.generate", "model_id": "flux"}


def test_producer_of_reads_through_from_ledger_on_cache_miss(bound):
    bound.remember_producer("r2", "audio.tts", "kokoro", worker="other-proc")
    assert "r2" not in selection._PRODUCERS
    assert selection.producer_of("r2") == ("audio.tts", "kokoro")
    assert selection._PRODUCERS["r2"] == ("audio.tts", "kokoro")   # filled
    assert selection.producer_of("nope") is None


def test_verdict_for_ref_lands_against_persisted_producer(bound):
    bound.remember_producer("r3", "audio.tts", "kokoro", worker="other-proc")
    assert selection.note_verdict_for_ref("r3", hard_pass=False, repair_code="clipping") is True
    rows = bound.recent("audio.tts")
    assert rows and rows[0]["model_id"] == "kokoro" and rows[0]["hard_pass"] == 0
    assert selection.note_verdict_for_ref("unknown", hard_pass=True) is False


def test_remember_producer_survives_ledger_failure(bound, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(bound, "remember_producer", boom)
    selection.remember_producer("r4", "x", "m")          # must not raise
    assert selection.producer_of("r4") == ("x", "m")     # cache still serves


# --------------------------------------------------------------------------- #
# ORACLE_LEDGER_REMOTE write-through
# --------------------------------------------------------------------------- #


def test_remote_write_through_posts_and_keeps_local(bound, monkeypatch):
    calls = []

    def fake(method, path, body=None, *, base=None):
        calls.append((method, path, body))
        return {"ok": True}
    monkeypatch.setenv(selection.LEDGER_REMOTE_ENV, "http://central:7860/")
    monkeypatch.setattr(selection, "_remote_request", fake)
    selection.remember_producer("r5", "image.generate", "flux", worker="w")
    assert calls == [("POST", "/api/oracle/producers",
                      {"ref": "r5", "capability": "image.generate", "model_id": "flux", "worker": "w"})]
    assert bound.producer("r5")["model_id"] == "flux"


def test_remote_read_fills_local_on_miss(bound, monkeypatch):
    monkeypatch.setenv(selection.LEDGER_REMOTE_ENV, "http://central:7860")

    def fake(method, path, body=None, *, base=None):
        assert method == "GET" and path == "/api/oracle/producers?ref=r%206"
        return {"ok": True, "producer": {"ref": "r 6", "capability": "audio.tts",
                                         "model_id": "kokoro", "worker": "w9", "ts": "t"}}
    monkeypatch.setattr(selection, "_remote_request", fake)
    assert selection.producer_of("r 6") == ("audio.tts", "kokoro")
    assert bound.producer("r 6")["worker"] == "w9"     # cached into the local ledger


def test_remote_unreachable_never_blocks(bound, monkeypatch):
    monkeypatch.setenv(selection.LEDGER_REMOTE_ENV, "http://127.0.0.1:9")   # discard port: refused
    monkeypatch.setattr(selection, "LEDGER_REMOTE_TIMEOUT_S", 0.5)
    selection.remember_producer("r7", "x", "m")
    assert selection.producer_of("r7") == ("x", "m")
    assert selection.producer_of("never-anywhere") is None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(bound):
    from flask import Flask
    from hugpy_server.app.routes.oracle_routes import oracle_bp
    app = Flask("oracle-producers-test")
    app.register_blueprint(oracle_bp, url_prefix="/api")
    return app.test_client()


def test_routes_post_get_and_summary(client, bound):
    r = client.post("/api/oracle/producers", json={"ref": "mct://a", "capability": "image.generate",
                                                  "model_id": "flux", "worker": "gpu-1"})
    assert r.status_code == 200 and r.get_json()["producer"]["worker"] == "gpu-1"
    r = client.get("/api/oracle/producers?ref=mct://a")
    assert r.get_json()["producer"]["model_id"] == "flux"
    r = client.get("/api/oracle/producers?ref=missing")
    assert r.status_code == 200 and r.get_json()["producer"] is None
    r = client.get("/api/oracle/producers")
    assert r.get_json()["count"] == 1 and r.get_json()["producers"][0]["ref"] == "mct://a"
    r = client.post("/api/oracle/producers", json={"ref": "x"})
    assert r.status_code == 400 and "capability" in r.get_json()["error"]
    bound.record("image.generate", "flux", ok=True, hard_pass=True)
    r = client.get("/api/oracle/ledger/summary")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] and body["outcomes"] == 1 and body["producers"] == 1
    assert body["by_model"][0]["passed"] == 1
    assert body["remote"] is None


def test_routes_503_when_selection_disabled(client, monkeypatch):
    monkeypatch.setenv(selection.SELECTION_DISABLE_ENV, "1")
    assert client.get("/api/oracle/producers?ref=a").status_code == 503
    assert client.get("/api/oracle/ledger/summary").status_code == 503


def test_worker_to_central_end_to_end(client, bound, tmp_path, monkeypatch):
    """A 'worker' ledger in another file writes through to central over HTTP
    (the flask test client stands in for the network), and central's ledger
    has it."""
    monkeypatch.setenv(selection.LEDGER_REMOTE_ENV, "http://central")

    def via_client(method, path, body=None, *, base=None):
        resp = client.open(path.replace("/api", "/api", 1), method=method, json=body)
        return resp.get_json()
    monkeypatch.setattr(selection, "_remote_request", via_client)
    # the worker's own local ledger lives elsewhere; central's is ``bound``
    worker_ledger = ReliabilityLedger(os.path.join(str(tmp_path), "worker.sqlite"))
    monkeypatch.setattr(selection, "_ledger", lambda: worker_ledger)
    selection.remember_producer("shared-ref", "video.generate", "wan", worker="gpu-2")
    assert worker_ledger.producer("shared-ref")["model_id"] == "wan"
    assert bound.producer("shared-ref")["worker"] == "gpu-2"
    worker_ledger.close()
