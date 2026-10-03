"""Failed/cancelled calls reach model_calls via the call log's end hook."""
from __future__ import annotations

import importlib

fc = importlib.import_module("hugpy_server.app.failed_calls")
cl = importlib.import_module("hugpy_control.calllog")


def test_done_rows_are_ignored_failures_written(monkeypatch):
    import hugpy_engine.model_index as mi
    seen = []
    monkeypatch.setattr(mi, "record_call_if_absent", lambda rid, m, w, **kw: seen.append((rid, m, w, kw)) or True)
    assert fc.write({"id": "j1", "model_key": "M", "worker": None, "status": "failed",
                     "error": "LoadRefusal: won't fit", "duration_ms": 1500, "stage": "loading"})
    rid, m, w, kw = seen[0]
    assert (rid, m, w) == ("j1", "M", "unassigned") and kw["elapsed_s"] == 1.5
    assert kw["state"]["outcome"] == {"ok": False, "status": "failed", "error": "LoadRefusal: won't fit"}
    assert kw["state"]["stage"] == "loading"
    assert fc.write({"id": "j2", "status": "failed"}) is False              # no model: unattributable


def test_end_hook_runs_for_end_rows_only(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGPY_CALL_LOG", str(tmp_path / "calls.jsonl"))
    got = []
    monkeypatch.setattr(cl, "_END_HOOKS", [got.append])

    class Job:
        id, kind, model_key, model_name, status, worker = "j", "chat", "M", "M", "failed", "w"
        error, started_ts, stage = "boom", None, None
    cl.record("start", Job())
    cl.record("end", Job())
    assert [r["phase"] for r in got] == ["end"] and got[0]["status"] == "failed"


def test_on_call_end_skips_done(monkeypatch):
    started = []
    monkeypatch.setattr(fc.threading, "Thread", lambda **kw: started.append(kw) or type("T", (), {"start": lambda s: None})())
    fc.on_call_end({"status": "done", "model_key": "M", "id": "j"})
    fc.on_call_end({"status": "cancelled", "model_key": "M", "id": "j"})
    assert len(started) == 1
