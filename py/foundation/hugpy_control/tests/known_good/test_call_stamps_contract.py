"""KNOWN-GOOD CONTRACT — stage stamps on the call record.

Operator (2026-09-29): "this is why the processing and processed attributes in
the calls db are necessary" — a call written only on completion is invisible
while it is stuck. Source under test: hugpy_control.jobs (Job.processing_ts,
JobStore.update/on_output/finish/_force_cancel_terminal) and
hugpy_control.calllog (phases start/processing/first_token/end, merged reader).

Deterministic: in-memory JobStore (no mirror), the call log on a tmp file.
"""
from __future__ import annotations

import json
import os

import pytest

from hugpy_control import calllog
from hugpy_control.jobs import JobStore


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "calls.jsonl"
    monkeypatch.setenv("HUGPY_CALL_LOG", str(p))
    return p


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_stamps_are_written_at_each_transition_not_at_the_end(log):
    """INVARIANT: queued_ts at create, processing_ts at the FIRST entry into
    processing (begin_dispatch), first_token_ts at the first output,
    processed_ts + processed_status at the terminal transition — each stamped
    when it happens, each mirrored to the call log as its own phase row; the
    request body rides only on the start row. Established: 2026-09-29."""
    js = JobStore(mirror=None)
    js.create("m", id="j1", kind="v1", request={"messages": [{"role": "user", "content": "x" * 5000}]})
    d = js.get("j1").to_dict()
    assert d["queued_ts"] and d["processing_ts"] is None and d["first_token_ts"] is None
    assert d["processed_ts"] is None and d["processed_status"] is None
    assert [r["phase"] for r in _lines(log)] == ["start"]

    js.begin_dispatch("j1", worker="ae-worker")
    d = js.get("j1").to_dict()
    assert d["processing_ts"] is not None and d["status"] == "processing"
    js.update("j1", stage="prefill", progress=0.5)      # a later processing write
    js.update("j1", status="processing")               # does not re-stamp
    assert js.get("j1").processing_ts == d["processing_ts"]
    assert [r["phase"] for r in _lines(log)] == ["start", "processing"]

    js.on_output("j1"); js.on_output("j1")
    d = js.get("j1").to_dict()
    assert d["first_token_ts"] is not None and d["status"] == "streaming"
    assert [r["phase"] for r in _lines(log)] == ["start", "processing", "first_token"]

    js.finish("j1")
    d = js.get("j1").to_dict()
    assert d["processed_ts"] is not None and d["processed_status"] == "done"
    rows = _lines(log)
    assert [r["phase"] for r in rows] == ["start", "processing", "first_token", "end"]
    assert "request" in rows[0] and all("request" not in r for r in rows[1:])
    assert rows[0]["ts"] <= rows[1]["ts"] <= rows[2]["ts"] <= rows[3]["ts"]

    merged = calllog.read()
    assert len(merged) == 1
    m = merged[0]
    assert m["queued_ts"] == pytest.approx(rows[0]["ts"])
    assert m["processing_ts"] == pytest.approx(rows[1]["ts"])
    assert m["first_token_ts"] == pytest.approx(rows[2]["ts"])
    assert m["processed_ts"] == pytest.approx(rows[3]["ts"]) and m["processed_status"] == "done"
    assert m["status"] == "done" and m["worker"] == "ae-worker" and m["tokens"] == 2


def test_cancelled_and_refused_calls_still_get_a_processed_stamp(log):
    """INVARIANT: a job cancelled through the AUTHORITATIVE cancel — relayed to
    a live owner OR force-marked in the store — and a job refused/failed
    before any token both end with processed_ts + a terminal processed_status
    on the row and in the merged call log. Established: 2026-09-29."""
    js = JobStore(mirror=None)
    # (a) owner-less pending job: store-mode cancel
    js.create("m", id="c1", kind="v1")
    res = js.cancel_authoritative("c1", "operator")
    assert res == {"cancelled": True, "mode": "store", "status": "cancelled"}
    d = js.get("c1").to_dict()
    assert d["processed_ts"] is not None and d["processed_status"] == "cancelled"
    # (b) live owner: the handle fires, the owner's teardown finishes it
    fired = []
    js.create("m", id="c2", kind="v1")
    js.begin_dispatch("c2", worker="w")
    js.attach_cancel("c2", lambda: fired.append(True))
    assert js.cancel_authoritative("c2", "operator")["mode"] == "relayed"
    assert fired and js.get("c2").cancel_requested
    js.finish("c2")                      # what streaming's finally does
    assert js.get("c2").to_dict()["processed_status"] == "cancelled"
    # (c) refused before a token
    js.create("m", id="r1", kind="v1")
    js.begin_dispatch("r1", worker="w")
    js.finish("r1", error="worker_busy: refused")
    assert js.get("r1").to_dict()["processed_status"] == "failed"

    by_id = {m["id"]: m for m in calllog.read()}
    assert by_id["c1"]["processed_status"] == "cancelled" and by_id["c1"]["processed_ts"]
    assert by_id["c2"]["processed_status"] == "cancelled" and by_id["c2"]["processing_ts"]
    assert by_id["r1"]["processed_status"] == "failed" and by_id["r1"]["error"].startswith("worker_busy")


def test_restart_orphan_reads_interrupted_with_a_processed_stamp(log, monkeypatch):
    """INVARIANT: a call the process restart orphaned (start/processing rows,
    no end row, started before this process) reads status=interrupted with
    processed_status=interrupted and a processed_ts — never an immortal
    'processing' row. Established: calllog interrupted rule, widened
    2026-09-29 to processing/streaming."""
    old = calllog._PROCESS_STARTED_AT
    js = JobStore(mirror=None)
    js.create("m", id="o1", kind="v1")
    js.begin_dispatch("o1", worker="w")
    monkeypatch.setattr(calllog, "_PROCESS_STARTED_AT", old + 10_000)
    m = calllog.read()[0]
    assert m["status"] == "interrupted" and m["processed_status"] == "interrupted"
    assert m["processed_ts"] == old + 10_000 and m["processing_ts"]
