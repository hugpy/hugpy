"""KNOWN-GOOD CONTRACT — the remote resolver keeps a worker's structured refusal.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (F3).
Source under test: hugpy_engine/resolvers/remote.py (_event_from_worker_line,
_LoadFailed, RemoteLoadError, _ColdRetry, structured_load_failure,
fit_failure_of, _cold_progress, _HonestError), hugpy_engine/schemas/
event_schemas.py (ErrorEvent.code / .load_failure).

LIVE CASE (2026-09-29, S2/S5): every failed fire returned prose only —
{"error": {"code": 500, "type": "api_error", "message": "<prose>"}} — even
where the worker had a structured FitFailure (kind=vram_fit code=wont_fit
protected=1). The structure was dropped at _event_from_worker_line and at
_LoadFailed.
"""
from __future__ import annotations

import importlib

import pytest

remote = importlib.import_module("hugpy_engine.resolvers.remote")
ES = importlib.import_module("hugpy_engine.schemas.event_schemas")
D = importlib.import_module("hugpy_engine.dispatch.dispatch")

GIB = 1 << 30

LF = {"class": "vram_fit", "loader_stderr": None, "path": None,
      "message": "LoadRefusal: won't fit on GPU: needs 10.5 GB, 82.2 MB free of 23.6 GB",
      "fit_failure": {"kind": "vram_fit", "code": "wont_fit",
                      "need_bytes": 11_253_969_110, "budget_bytes": 82_182_144,
                      "plan_n_cpu_moe": None, "contract_n_cpu_moe": None,
                      "permanent": False, "state_dependent": True},
      "refusal": {"needs_bytes": 11_253_969_110, "free_vram_bytes": 82_182_144,
                  "free_vram_device_bytes": 1_155_923_968,
                  "external_floor_bytes": 1_073_741_824, "ceiling_reserve_bytes": 0,
                  "protected_count": 1,
                  "protected": [{"model_key": "Qwen3.8_4B_Distilled_GGUF",
                                 "why": "static (locked residency)",
                                 "vram_bytes": 21_200_000_000}],
                  "evicted": []}}


def test_worker_error_line_keeps_the_structure_and_the_old_shape_still_parses():
    """INVARIANT (F3): a worker SSE error dict's `load_failure` / `code` ride
    into the ErrorEvent; a worker that sends neither yields the pre-F3 event
    (None, None). Established: core isolation step 2 (2026-09-29)."""
    ev = remote._event_from_worker_line(
        {"type": "error", "message": "LoadRefusal: won't fit on GPU", "code": "vram_fit",
         "load_failure": LF}, "r1")
    assert isinstance(ev, ES.ErrorEvent)
    assert ev.message.startswith("LoadRefusal") and ev.code == "vram_fit"
    assert ev.load_failure["fit_failure"]["code"] == "wont_fit"
    old = remote._event_from_worker_line({"type": "error", "message": "boom"}, "r1")
    assert old.load_failure is None and old.code is None
    assert ES.ErrorEvent(request_id="r", message="m").model_dump()["load_failure"] is None


def test_structured_load_failure_walks_events_exceptions_and_chains():
    """INVARIANT (F3): the structure is found on a _LoadFailed / _ColdRetry /
    RemoteLoadError, on an ErrorEvent, behind a RuntimeError raised FROM a
    _WorkerHTTPError whose body carried it, and on a local LoadRefusal (via
    load_failure_of). Nothing structured -> None, never a raise."""
    lf = remote._LoadFailed("The 'ae' worker could not complete this request: ...",
                            load_failure=LF, code="vram_fit")
    assert lf.message.startswith("The 'ae' worker") and lf.code == "vram_fit"
    assert remote.structured_load_failure(lf) == LF
    assert remote.structured_load_failure(
        remote._ColdRetry("x", busy=False, load_failure=LF)) == LF
    assert remote.structured_load_failure(remote.RemoteLoadError("x", load_failure=LF)) == LF
    assert remote.structured_load_failure(
        ES.ErrorEvent(request_id="r", message="m", load_failure=LF)) == LF
    http = remote._WorkerHTTPError(500, {"ok": False, "error": "won't fit", "load_failure": LF},
                                   "http://w/infer")
    try:
        raise RuntimeError("worker ae failed for m: HTTP 500") from http
    except RuntimeError as chained:
        assert remote.structured_load_failure(chained) == LF
    local = D.LoadRefusal({"state": "refused", "model_key": "m",
                           "reason": "won't fit on GPU: needs 1 GiB",
                           "fit_failure": LF["fit_failure"],
                           "protected": LF["refusal"]["protected"],
                           "external_floor_bytes": GIB, "free_vram_device_bytes": GIB + 1})
    out = remote.structured_load_failure(local)
    assert out["class"] == "vram_fit" and out["fit_failure"]["kind"] == "vram_fit"
    assert out["refusal"]["external_floor_bytes"] == GIB
    assert remote.structured_load_failure(RuntimeError("plain prose")) is None
    assert remote.structured_load_failure(None) is None
    assert remote._LoadFailed("legacy single-arg").load_failure is None     # old callers


def test_fit_failure_of_hoists_kind_code_need_budget_and_the_blockers():
    ff = remote.fit_failure_of(LF)
    assert ff["kind"] == "vram_fit" and ff["code"] == "wont_fit"
    assert ff["need_bytes"] == 11_253_969_110 and ff["budget_bytes"] == 82_182_144
    assert ff["blocked_by"] == ["Qwen3.8_4B_Distilled_GGUF"]
    assert ff["protected"][0]["why"] == "static (locked residency)"
    assert ff["external_floor_bytes"] == 1_073_741_824
    assert ff["free_vram_device_bytes"] == 1_155_923_968
    assert ff["free_vram_bytes"] == 82_182_144 and ff["evicted"] == []
    assert remote.fit_failure_of({"class": "hard_load_failure", "message": "x"}) is None
    assert remote.fit_failure_of(None) is None


def test_cold_progress_never_reads_a_hollow_seat_as_ready(monkeypatch):
    """INVARIANT (F3/F4): a slot row measured hollow (materialized False) is not
    "loaded and idle"; the provider's healthy is the materialized-gated
    verdict; an honest error carries the worker's structured load_failure.
    Established: step 2."""
    worker = {"id": "w1", "name": "ae-worker",
              "slots": [{"model_key": "Qwen3-Coder-Next-GGUF", "healthy": True,
                         "busy": False, "materialized": False}]}
    monkeypatch.setattr(remote, "_load_state", lambda mk, wid, since: None)
    moved, prog, msg, honest, ready = remote._cold_progress("Qwen3-Coder-Next-GGUF", worker, 0.0)
    assert ready is False and honest is None
    worker["slots"][0]["materialized"] = True
    moved, prog, msg, honest, ready = remote._cold_progress("Qwen3-Coder-Next-GGUF", worker, 0.0)
    assert ready is True and "loaded and idle" in msg
    # provider verdicts: materialized-gated healthy, structured honest error
    monkeypatch.setattr(remote, "_load_state",
                        lambda mk, wid, since: {"healthy": False, "materialized": False,
                                                "in_progress": False, "progress": None,
                                                "message": None, "error": None})
    assert remote._cold_progress("Other", worker, 0.0)[4] is False
    monkeypatch.setattr(remote, "_load_state",
                        lambda mk, wid, since: {"healthy": False, "in_progress": False,
                                                "progress": None, "message": None,
                                                "error": "LoadRefusal: won't fit on GPU",
                                                "load_failure": LF})
    moved, prog, msg, honest, ready = remote._cold_progress("Other", worker, 0.0)
    assert isinstance(honest, str) and honest.startswith("LoadRefusal")
    assert honest.load_failure == LF and ready is False
    assert f"{honest}" == "LoadRefusal: won't fit on GPU"       # still a plain str for every consumer
