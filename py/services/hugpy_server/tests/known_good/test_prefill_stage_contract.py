"""KNOWN-GOOD CONTRACT — prefill honesty on the job row and the queue view.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "call queue / relay", 2026-09-29
Coder-Next "stuck answering" incident). Source under test:
hugpy_server.app.functions.chat.streaming._feed_job_from_status (stage
"prefill") and hugpy_engine.dispatch.activity.snapshot (stage/progress/
input_tokens on the queue row).

Deterministic: one JobStore over a tmp SQLite mirror, no worker, no network.
"""
from __future__ import annotations

import importlib

import pytest

J = importlib.import_module("hugpy_control.jobs")
activity = importlib.import_module("hugpy_engine.dispatch.activity")
streaming = importlib.import_module("hugpy_server.app.functions.chat.streaming")
from hugpy_control.shared import SqliteMirror  # noqa: E402
from hugpy_engine.schemas.event_schemas import StatusEvent  # noqa: E402

MODEL = "Qwen3-Coder-Next-GGUF"


@pytest.fixture
def store(tmp_path, monkeypatch):
    js = J.JobStore(mirror=SqliteMirror(path=str(tmp_path / "comms.db")))
    monkeypatch.setattr(J, "job_store", js)
    monkeypatch.setattr(activity, "job_store", js)
    monkeypatch.setattr(streaming, "job_store", js)
    return js


def test_prefill_status_sets_stage_real_prompt_size_and_feeds_the_stall_clock(store):
    """INVARIANT: a worker/relay ``status`` event with stage="prefill" moves the
    job to processing + stage "prefill", records the ENGINE's prompt size
    (n_prompt -> input_tokens) and its progress (n_past/n_prompt), and every
    progress advance feeds progressed_at — a legitimately prefilling 30k-token
    prompt never reads `stalled`, and never reads "answering, 0 tokens".
    Established: 2026-09-29 (live: 29,451-token reducer prompt prefilled 104 s
    at 282 tok/s with the row at "processing, 0 tokens, stalled")."""
    store.create(MODEL, id="p1", kind="v1")
    store.begin_dispatch("p1", worker="ae-worker")
    job = store.get("p1")
    job.progressed_at -= 200          # pretend 200 s of silence
    assert job.to_dict()["stalled"] is True
    streaming._feed_job_from_status("p1", StatusEvent(
        request_id="p1", stage="prefill", n_prompt=29451, n_past=12436,
        progress=0.42, message="prefill 12436/29451 prompt tokens"))
    job = store.get("p1")
    assert job.status == "processing" and job.stage == "prefill"
    assert job.input_tokens == 29451 and job.progress == pytest.approx(0.42)
    assert job.to_dict()["stalled"] is False, "engine progress is forward progress"
    t1 = job.progressed_at
    job.progressed_at -= 200
    # A repeated figure (same progress) is NOT movement; an advance is.
    streaming._feed_job_from_status("p1", StatusEvent(
        request_id="p1", stage="prefill", n_prompt=29451, n_past=12436, progress=0.42))
    assert store.get("p1").to_dict()["stalled"] is True
    streaming._feed_job_from_status("p1", StatusEvent(
        request_id="p1", stage="prefill", n_prompt=29451, n_past=20628, progress=0.70))
    job = store.get("p1")
    assert job.to_dict()["stalled"] is False and job.progressed_at >= t1 - 1
    # The relay's pre-engine estimate is labelled and never written as truth.
    store.create(MODEL, id="p2", kind="v1")
    streaming._feed_job_from_status("p2", StatusEvent(
        request_id="p2", stage="prefill", n_prompt_est=7000,
        message="prefill on ae-worker: ~7000 prompt tokens (estimate)"))
    j2 = store.get("p2")
    assert j2.stage == "prefill" and j2.status == "processing" and j2.input_tokens is None


def test_queue_row_carries_stage_progress_and_prompt_tokens(store):
    """INVARIANT: /api/llm/queue rows expose stage / progress / input_tokens so
    the console can render "prefill 42% of 29451" for a processing call and
    plain "processing"/"active" otherwise. Established: 2026-09-29."""
    store.create(MODEL, id="q1", kind="v1", prompt="hi")
    store.begin_dispatch("q1", worker="ae-worker")
    row = activity.snapshot()[0]
    assert row["state"] == "processing" and row["stage"] == "" and row["input_tokens"] is None
    streaming._feed_job_from_status("q1", StatusEvent(
        request_id="q1", stage="prefill", n_prompt=100, n_past=50, progress=0.5))
    row = activity.snapshot()[0]
    assert (row["state"], row["stage"], row["progress"], row["input_tokens"]) == \
        ("processing", "prefill", 0.5, 100)
    activity.on_token("q1")
    assert activity.snapshot()[0]["state"] == "active"
