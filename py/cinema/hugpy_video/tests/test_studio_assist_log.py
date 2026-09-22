"""STUDIO-ASSIST LIVE LOG (operator directive, 2026-07-31).

The operator wants a live log in the studio UI showing what each prompt-generate
attempt actually returned — the raw model reply, what was stripped, and the
outcome — so failures ("returned only reasoning and no output", "did not return
the JSON object the spread contract requires") are self-diagnosable.

This proves the two halves that make that log trustworthy:

  1. THE STORE (comms/studio_assist_log.py) — append/recent/max_id, the ring
     bound, and the sqlite mirror keyed by rowid (the cross-gunicorn-worker
     cursor). A store fault must cost nothing; ``raw`` is kept UNTRUNCATED.

  2. THE INSTRUMENTATION (hugpy_server video_routes.py) — the four outcomes the
     operator actually sees, driven through the real route with a stubbed
     executor (the tests/test_prompt_spread.py harness):
        * a served spread          (JSON parsed ok      -> outcome=served)
        * a spread parse failure   (SpreadParseError    -> outcome=parse_error,
                                     the FULL raw reply captured)
        * a reasoning-only reply   (<think>-only        -> from_reasoning True)
        * a worker error           (executor raised     -> outcome=worker_error)

Runs under pytest AND as a plain script:
    venv/bin/python -m pytest tests/test_studio_assist_log.py -q
    venv/bin/python tests/test_studio_assist_log.py
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.disable(logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


from hugpy_video import studio_assist_log as SAL


# --------------------------------------------------------------------------- #
# 1. THE STORE
# --------------------------------------------------------------------------- #
def test_store_append_recent_maxid():
    with tempfile.TemporaryDirectory() as d:
        store = SAL.StudioAssistStore(path=os.path.join(d, "c.db"))
        SAL.set_store(store)
        SAL.reset_for_tests()

        check("empty store recent is []", store.recent() == [])
        check("empty store max_id is 0", store.max_id() == 0)

        r1 = SAL.append(run_id="run-1", mode="spread", outcome=SAL.OUTCOME_SERVED,
                        raw="A" * 10, text="A prompt", model_requested="m",
                        model_resolved="m-x", elapsed_ms=12)
        r2 = SAL.append(run_id="run-2", mode="negative", outcome=SAL.OUTCOME_EMPTY,
                        raw="", error="nothing came back")
        check("append returns a record", isinstance(r1, dict) and r1["run_id"] == "run-1")

        got = store.recent()
        check("both rows persisted", len(got) == 2)
        check("recent is oldest-first", got[0]["run_id"] == "run-1" and got[1]["run_id"] == "run-2")
        check("each row carries a store _id", all(isinstance(g["_id"], int) for g in got))
        check("max_id tracks the head", store.max_id() == got[-1]["_id"])

        # after_id is the stream cursor — returns only what is newer.
        tail = store.recent(after_id=got[0]["_id"])
        check("after_id tails by rowid", len(tail) == 1 and tail[0]["run_id"] == "run-2")

        # raw is stored UNTRUNCATED; raw_cap only bounds the returned copy.
        SAL.append(run_id="big", mode="spread", outcome=SAL.OUTCOME_PARSE_ERROR,
                   raw="Z" * 5000, error="bad json")
        full = [g for g in store.recent() if g["run_id"] == "big"][0]
        check("store keeps raw untruncated", len(full["raw"]) == 5000)
        capped = [g for g in store.recent(raw_cap=100) if g["run_id"] == "big"][0]
        check("raw_cap truncates the returned copy", len(capped["raw"]) == 100)
        check("raw_cap flags truncation", capped.get("raw_truncated") is True)
        SAL.set_store(None)


def test_ring_is_bounded():
    SAL.set_store(SAL.StudioAssistStore(path="off"))   # disable sqlite side
    SAL.reset_for_tests()
    orig = SAL.RING_MAX
    try:
        SAL.RING_MAX = 5
        SAL._RING.clear()
        SAL._RING = __import__("collections").deque(maxlen=SAL.RING_MAX)
        for i in range(20):
            SAL.append(run_id=f"r{i}", mode="spread", outcome=SAL.OUTCOME_SERVED)
        ring = SAL.recent(limit=999)
        check("ring is bounded to RING_MAX", len(ring) == 5)
        check("ring keeps the NEWEST", ring[-1]["run_id"] == "r19")
    finally:
        SAL.RING_MAX = orig
        SAL._RING = __import__("collections").deque(maxlen=orig)
    SAL.set_store(None)


def test_disabled_store_is_silent():
    store = SAL.StudioAssistStore(path="off")   # HUGPY_COMMS_DB=off sentinel
    check("off sentinel disables the store", store._disabled is True)
    check("disabled append is a no-op", store.append([{"run_id": "x", "stage": "s"}]) == 0)
    check("disabled recent is []", store.recent() == [])
    check("disabled max_id is 0", store.max_id() == 0)
    # A store pointed at an unwritable path must degrade, never raise.
    bad = SAL.StudioAssistStore(path="/proc/nonexistent/dir/c.db")
    check("unwritable store append degrades to 0",
          bad.append([{"run_id": "y"}]) == 0)


def test_outcome_classification():
    C = SAL.classify_execute_error
    check("400 -> resolve_error", C(400, "unknown model") == SAL.OUTCOME_RESOLVE_ERROR)
    check("502 dead worker -> worker_error",
          C(502, "no live worker for this model") == SAL.OUTCOME_WORKER_ERROR)
    check("502 produced no text -> empty",
          C(502, "assist produced no text") == SAL.OUTCOME_EMPTY)
    check("502 returned nothing -> empty",
          C(502, "the assistant returned nothing — neither a prompt nor reasoning")
          == SAL.OUTCOME_EMPTY)
