"""Central transfer ledger: central's OWN accounting of the weight bytes it
serves (manifest -> ranged /file segments), the authoritative source of the
per-worker "downloading from central" state."""
from __future__ import annotations

import json
import os
import tempfile

import pytest
from flask import Flask

from hugpy_server.app import transfer_ledger as tl
from hugpy_server.app.routes import model_status_routes as ms


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture()
def led(tmp_path):
    return tl.TransferLedger(clock=Clock(), jsonl_path=str(tmp_path / "transfers.jsonl"))


def only(led):
    return led.snapshot(include_done=False)[0]


def test_bytes_accounted_per_range_request_and_completion(led, tmp_path):
    led.manifest("M", "wid-aeb", [{"path": "a.gguf", "size": 100}, {"path": "b.json", "size": 10}], 110)
    t1 = led.begin("M", "wid-aeb", "a.gguf", 100)
    t2 = led.begin("M", "wid-aeb", "a.gguf", 100)
    led.served(t1, 40)
    led.served(t2, 30)
    v = only(led)
    assert (v["status"], v["bytes_served"], v["total_bytes"], v["segments_in_flight"]) == ("active", 70, 110, 2)
    assert v["pct"] == pytest.approx(63.6) and v["source"] == "central-ledger"
    led.clock.t += 10
    led.served(t1, 60)
    led.served(t2, 999)                      # a retried segment never overcounts a file
    led.end(t1)
    led.end(t2)
    assert only(led)["bytes_served"] == 100 and only(led)["files_done"] == 1
    t3 = led.begin("M", "wid-aeb", "b.json", 10)
    led.served(t3, 10)
    led.end(t3)
    v = only(led)
    assert v["status"] == "complete" and v["files_done"] == 2 and v["pct"] == 100.0
    assert v["elapsed_s"] == 10 and v["mb_per_s"] is not None
    lines = (tmp_path / "transfers.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["status"] == "complete"
    led.end(t3)                              # a late close never re-persists
    assert len((tmp_path / "transfers.jsonl").read_text().splitlines()) == 1


def test_stalled_then_aborted_into_history(led, tmp_path):
    led.manifest("M", "w", [{"path": "a", "size": 100}], 100)
    tok = led.begin("M", "w", "a", 100)
    led.served(tok, 10)
    led.end(tok)
    led.clock.t += 31
    v = only(led)
    assert v["status"] == "stalled" and v["idle_s"] == 31
    led.clock.t += 600
    assert led.snapshot(include_done=False) == []
    hist = led.snapshot(include_done=True)
    assert hist[0]["status"] == "aborted"
    assert json.loads((tmp_path / "transfers.jsonl").read_text().splitlines()[-1])["status"] == "aborted"


def test_new_manifest_after_completion_starts_a_new_transfer(led):
    led.manifest("M", "w", [{"path": "a", "size": 5}], 5)
    tok = led.begin("M", "w", "a", 5)
    led.served(tok, 5)
    led.end(tok)
    led.clock.t += 5
    led.manifest("M", "w", [{"path": "a", "size": 5}], 5)
    # A manifest read alone is not a transfer (nothing served yet): no live row
    # until the first byte-range request opens it.
    assert led.snapshot(include_done=False) == []
    led.begin("m", "w", "a.bin", 10)
    assert only(led)["bytes_served"] == 0 and only(led)["status"] == "active"
    assert [h["status"] for h in led.snapshot()][-1] == "complete"


def test_worker_state_prefers_the_ledger_over_the_heartbeat():
    w = {"id": "wid-aeb", "name": "aeb", "status": "online", "provisioning": ["M"],
         "provision_progress": {"M": {"done_bytes": 1, "total_bytes": 2}}}
    row = {"worker_id": "wid-aeb", "model_key": "M", "status": "active", "bytes_served": 12.4e9,
           "total_bytes": 21.6e9, "pct": 57.4, "mb_per_s": 135.2, "started_at": 900.0, "idle_s": 0.1,
           "files_done": 3, "files_total": 5, "last_request_at": 999.9}
    s = ms.model_worker_state({"model_key": "M"}, w, None, now=1000.0, transfers=[row])
    assert s["state"] == "downloading from central"
    assert s["detail"] == ("12.4/21.6 GB 57% · 135.2 MB/s · 1m40s elapsed · files 3/5 (central transfer ledger)")
    assert s["progress"]["source"] == "central-ledger"
    fb = ms.model_worker_state({"model_key": "M"}, w, None, now=1000.0, transfers=[])
    assert fb["progress"]["source"] == "worker-heartbeat" and "no central transfer-ledger entry" in fb["detail"]
    stalled = ms.model_worker_state({"model_key": "M"}, w, None, now=1000.0,
                                    transfers=[{**row, "status": "stalled", "idle_s": 45}])
    assert "STALLED: no request for 45s" in stalled["detail"]


# ── through the real transfer routes ────────────────────────────────────────

@pytest.fixture()
def routes(monkeypatch, tmp_path):
    from hugpy_server.app.routes import worker_routes as wr
    d = tmp_path / "model"
    d.mkdir()
    (d / "w.gguf").write_bytes(os.urandom(3 * 1024 * 1024 + 7))
    monkeypatch.setattr(wr, "_transfer_authorized", lambda: True)
    monkeypatch.setattr(wr, "_model_dir_or_404", lambda k: ({"framework": "gguf"}, str(d)))
    monkeypatch.setattr(wr, "_transfer_selection", lambda raw, model, dest: raw)
    monkeypatch.setattr(wr, "_no_weights_reason", lambda *a: None)
    monkeypatch.setattr(wr, "_budget_refusal_for_transfer", lambda *a: None)
    fresh = tl.TransferLedger(jsonl_path=str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(tl, "ledger", fresh)
    monkeypatch.setattr(ms, "load_workers", lambda: ([{"id": "wid-aeb", "name": "aeb", "status": "online"}],
                                                     {"error": None}))
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    app.register_blueprint(ms.model_status_bp)
    return app.test_client(), fresh, (d / "w.gguf").stat().st_size


def test_routes_account_every_byte_and_expose_llm_transfers(routes):
    c, led, size = routes
    h = {"X-Worker-Id": "wid-aeb"}
    assert c.get("/llm/models/M/manifest", headers=h).status_code == 200
    half = size // 2
    r = c.get("/llm/models/M/file?path=w.gguf", headers={**h, "Range": f"bytes=0-{half - 1}"})
    assert r.status_code == 206 and len(r.data) == half
    v = led.snapshot(include_done=False)[0]
    assert v["bytes_served"] == half and v["status"] == "active" and v["segments_in_flight"] == 0
    r = c.get("/llm/models/M/file?path=w.gguf", headers={**h, "Range": f"bytes={half}-"})
    assert r.status_code == 206 and len(r.data) == size - half
    t = c.get("/llm/transfers").get_json()
    row = next(x for x in t["transfers"] if x["model_key"] == "M")
    assert row["status"] == "complete" and row["bytes_served"] == size and row["worker"] == "aeb"
