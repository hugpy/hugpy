"""LOAD-STATE-LATENCY (2026-09-29) — the slot child says "loading" while it loads.

The /load route stamps ``loading_model_key`` on the Slot for the whole
preflight/spawn/wait-healthy window and nudges the agent's heartbeat at BOTH
edges (start, and done or refused), reusing the inflight-edge nudge. /status
reports ``loading`` / ``loading_since`` so the agent can attribute the load to
this seat from its first beat.

Run: venv/bin/python -m pytest tests/test_slot_agent_loading_state.py -q
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sa = importlib.import_module("hugpy_engine.serve.slot_agent")


def _app(monkeypatch):
    monkeypatch.setattr(sa, "SLOT_ID", "1", raising=False)
    nudges = []
    monkeypatch.setattr(sa, "_nudge_agent_heartbeat", lambda slot=None: nudges.append(slot))
    app, slot = sa.build_app()
    monkeypatch.setattr(sa, "no_local_serving", lambda: False, raising=False)
    return app, slot, nudges


def test_load_route_claims_loading_for_the_whole_load_and_nudges_both_edges(monkeypatch):
    app, slot, nudges = _app(monkeypatch)
    seen = {}

    def fake_load(model_key, *a, **kw):
        # mid-load: the claim is visible on the status row, one nudge fired
        seen["loading"] = slot.loading_model_key
        seen["since"] = slot.loading_since
        seen["nudges_at_start"] = len(nudges)
        slot.model_key = model_key
        return {"model_key": model_key, "healthy": True}

    slot.load = fake_load
    r = app.test_client().post("/load", json={"model_key": "coder"})
    assert r.status_code == 200
    assert seen["loading"] == "coder" and isinstance(seen["since"], float)
    assert seen["nudges_at_start"] == 1
    assert len(nudges) == 2                      # start + done
    assert slot.loading_model_key is None and slot.loading_since is None


def test_refused_load_clears_the_claim_and_still_nudges(monkeypatch):
    app, slot, nudges = _app(monkeypatch)

    def refuse(model_key, *a, **kw):
        raise RuntimeError("CPU-resident share exceeds this model's RAM budget")

    slot.load = refuse
    r = app.test_client().post("/load", json={"model_key": "coder"})
    assert r.status_code == 500
    assert "RAM budget" in r.get_json()["error"]
    assert slot.loading_model_key is None
    assert len(nudges) == 2                      # start + refused


def test_status_reports_the_loading_claim(monkeypatch):
    app, slot, _ = _app(monkeypatch)
    # status() probes the child; stub the process-facing bits of a bare slot
    monkeypatch.setattr(slot, "_self_heal", lambda: None)
    monkeypatch.setattr(slot, "verify_identity", lambda force=False: None)
    monkeypatch.setattr(slot, "healthy", lambda: False)
    monkeypatch.setattr(sa, "free_vram_bytes", lambda: None, raising=False)
    st = slot.status()
    assert st["loading"] is None and st["loading_since"] is None
    slot.loading_model_key, slot.loading_since = "coder", 42.0
    st = slot.status()
    assert st["loading"] == "coder" and st["loading_since"] == 42.0
