"""STUDIO-ASSIST LIVE LOG — the ROUTE half (hugpy_server instrumentation),
split out of hugpy_video/tests/test_studio_assist_log.py (the store half stays
with the store). Drives the real /video studio routes with a stubbed executor
and asserts the four outcomes the operator sees are recorded.
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

# --------------------------------------------------------------------------- #
# 2. THE INSTRUMENTATION — driven through the real route, stubbed executor.
#
# Reuses the tests/test_prompt_spread.py harness (_patch_executor patches the
# ONE execute_prompt the route reaches through functions.imports; patching
# managers.dispatch would silently run live inference — see that file's landmine
# note).
# --------------------------------------------------------------------------- #
def _load_route_harness():
    import json  # noqa: F401 — used by the spread reply builder
    from flask import Flask

    vr = importlib.import_module(
        "hugpy_server.app.routes.video_routes")
    imports_mod = sys.modules[
        "hugpy_server.app.functions.imports"]
    app = Flask(__name__)
    app.register_blueprint(vr.video_bp)
    return vr, imports_mod, app.test_client()


def _patch_executor(imports_mod, reply_text, ok=True, error=None, raises=None):
    import types
    def fake_execute_prompt(*a, **kw):
        if raises is not None:
            raise raises
        return {"ok": ok, "text": reply_text, "error": error,
                "model_key": "resolved-model-x"}
    imports_mod.execute_prompt = fake_execute_prompt
    return types.SimpleNamespace()


def _spread_body(ids=("segment-0", "segment-1")):
    return {
        "mode": "spread",
        "movie_query": "a diver finds something under the ice",
        "style_bible": {"world": "arctic station", "subject": "a lone diver",
                        "visual_style": "grainy 16mm"},
        "fixed_segments": [],
        "target_segments": [
            {"segment_id": sid, "direction": "make it colder",
             "joint_mode": "vace_extend", "index": i}
            for i, sid in enumerate(ids)],
        "global_negative": ["watermark"],
        "steering_seed": 184392,
    }


def _segments_reply(ids):
    import json
    return json.dumps({"segments": [
        {"segment_id": sid, "operation": "generate_from_direction",
         "prompt": f"A shot for {sid}.", "negative": "blurry",
         "continuity_note": f"after {sid}", "directions_used": [0]}
        for sid in ids], "invented_identity_attributes": [], "warnings": []})


def _fresh_store():
    d = tempfile.mkdtemp()
    SAL.set_store(SAL.StudioAssistStore(path=os.path.join(d, "c.db")))
    SAL.reset_for_tests()
    return SAL.get_store()


def test_served_spread_records_served():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    ids = ("segment-0", "segment-1")
    _patch_executor(imports_mod, _segments_reply(ids))
    r = client.post("/video/prompt/assist", json=_spread_body(ids))
    check("served spread returns 200", r.status_code == 200)
    rows = store.recent()
    check("served spread logged exactly one record", len(rows) == 1)
    rec = rows[0]
    check("served spread outcome=served", rec["outcome"] == SAL.OUTCOME_SERVED)
    check("served spread captured raw", "segment-0" in rec.get("raw", ""))
    check("served spread labelled mode=spread", rec["mode"] == "spread")
    check("served spread records model_resolved", rec["model_resolved"] == "resolved-model-x")
    SAL.set_store(None)


def test_spread_parse_error_records_raw():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    # A reply that is NOT the spread JSON contract -> SpreadParseError.
    _patch_executor(imports_mod, "here are some nice shots for your movie, enjoy!")
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("parse failure returns 502", r.status_code == 502)
    rows = store.recent()
    check("parse failure logged one record", len(rows) == 1)
    rec = rows[0]
    check("parse failure outcome=parse_error", rec["outcome"] == SAL.OUTCOME_PARSE_ERROR)
    check("parse failure captured the FULL raw reply",
          rec.get("raw") == "here are some nice shots for your movie, enjoy!")
    check("parse failure recorded the contract error", bool(rec.get("error")))
    SAL.set_store(None)


def test_reasoning_only_reply_flags_from_reasoning():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    # The whole answer lives inside <think>…</think>: _studio_no_think salvages
    # the reasoning as the prompt and flags from_reasoning. Drive it through the
    # detail/generate inline handler (mode=generate), which serves prose.
    _patch_executor(imports_mod, "<think>A neon city at dusk, rain-slicked streets</think>")
    r = client.post("/video/prompt/assist",
                    json={"mode": "generate", "context": {"kind": "image"}})
    check("reasoning-only generate returns 200", r.status_code == 200)
    body = r.get_json()
    check("reasoning-only served the reasoning as prompt", bool(body.get("prompt")))
    rows = store.recent()
    check("reasoning-only logged one record", len(rows) == 1)
    rec = rows[0]
    check("reasoning-only outcome=served", rec["outcome"] == SAL.OUTCOME_SERVED)
    check("reasoning-only flags from_reasoning True", rec.get("from_reasoning") is True)
    check("reasoning-only captured the raw <think> reply", "<think>" in rec.get("raw", ""))
    check("reasoning-only stripped text present", bool(rec.get("text")))
    SAL.set_store(None)


def test_worker_error_records_worker_error():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    _patch_executor(imports_mod, "", raises=RuntimeError("no live worker for this model"))
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("worker error returns 502", r.status_code == 502)
    rows = store.recent()
    check("worker error logged one record", len(rows) == 1)
    rec = rows[0]
    check("worker error outcome=worker_error", rec["outcome"] == SAL.OUTCOME_WORKER_ERROR)
    check("worker error recorded the message", bool(rec.get("error")))
    SAL.set_store(None)


def test_resolve_error_records_resolve_error():
    vr, imports_mod, client = _load_route_harness()
    store = _fresh_store()
    _patch_executor(imports_mod, "", raises=ValueError("unknown model_key 'nope'"))
    r = client.post("/video/prompt/assist", json=_spread_body())
    check("resolve error returns 400", r.status_code == 400)
    rows = store.recent()
    rec = rows[0]
    check("resolve error outcome=resolve_error", rec["outcome"] == SAL.OUTCOME_RESOLVE_ERROR)
    SAL.set_store(None)


def test_backfill_and_stream_routes_exist():
    vr, imports_mod, client = _load_route_harness()
    _fresh_store()
    # No auth gate installed on this bare test app, so the routes resolve; we only
    # assert they are wired and shaped (the video gate provides auth in prod).
    r = client.get("/video/prompt/assist/log?limit=10")
    check("backfill route resolves", r.status_code == 200)
    body = r.get_json()
    check("backfill returns events+cursor", "events" in body and "cursor" in body)
    SAL.set_store(None)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print(f"\n{ok} checks passed")


if __name__ == "__main__":
    _run_all()
