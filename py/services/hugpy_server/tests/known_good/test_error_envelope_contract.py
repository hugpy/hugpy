"""KNOWN-GOOD CONTRACT — the v1 error envelope carries the structured refusal,
and central consults MEASURED residency, never loaded_models membership.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation");
design: notes/core-isolation-step2-2026-09-29.md (F3).
Source under test: hugpy_server/app/routes/v1_routes.py (_openai_error,
_structured_error_fields), messages_routes.py (_anthropic_error),
hugpy_fleet/central/workers.py (_resident_materialized, _polite_admits,
load_state_for_model), hugpy_fleet/worker/agent.py (_event_to_dict).
"""
from __future__ import annotations

import importlib
import types

import pytest

flask = pytest.importorskip("flask")
V1 = importlib.import_module("hugpy_server.app.routes.v1_routes")
MSG = importlib.import_module("hugpy_server.app.routes.messages_routes")
W = importlib.import_module("hugpy_fleet.central.workers")
A = importlib.import_module("hugpy_fleet.worker.agent")
remote = importlib.import_module("hugpy_engine.resolvers.remote")
ES = importlib.import_module("hugpy_engine.schemas.event_schemas")
AM = importlib.import_module("hugpy_engine.alloc_modes")

GIB = 1 << 30
LF = {"class": "vram_fit", "loader_stderr": None, "path": None,
      "message": "LoadRefusal: won't fit on GPU: needs 10.5 GB, 82.2 MB free of 23.6 GB",
      "fit_failure": {"kind": "vram_fit", "code": "wont_fit",
                      "need_bytes": 11_253_969_110, "budget_bytes": 82_182_144},
      "refusal": {"free_vram_bytes": 82_182_144, "free_vram_device_bytes": 1_155_923_968,
                  "external_floor_bytes": 1_073_741_824, "protected_count": 1,
                  "protected": [{"model_key": "Qwen3.8_4B_Distilled_GGUF",
                                 "why": "static (locked residency)", "vram_bytes": 21 * GIB}],
                  "evicted": []}}
PROSE = "The 'ae-worker' worker could not complete this request: LoadRefusal: won't fit on GPU"


@pytest.fixture
def ctx():
    app = flask.Flask(__name__)
    with app.test_request_context("/v1/chat/completions"):
        yield app


def _body(resp):
    return resp[0].get_json(), resp[1]


def test_openai_envelope_carries_type_fit_failure_and_keeps_the_prose(ctx, monkeypatch):
    """INVARIANT (F3): with a structured cause the envelope is
    error.type = the load class, error.code = the HTTP status (unchanged),
    error.message = the human sentence (unchanged), plus error.fit_failure
    {kind, code, need_bytes, budget_bytes, blocked_by, protected,
    external_floor_bytes, free_vram_device_bytes} and error.load_failure.
    Without a cause the envelope is byte-identical to before.
    Established: core isolation step 2 (2026-09-29)."""
    monkeypatch.setattr(V1, "_request_diagnostics", lambda *a, **k: None)
    ev = ES.ErrorEvent(request_id="r", message=PROSE, load_failure=LF, code="vram_fit")
    body, status = _body(V1._openai_error(PROSE, 500, "api_error", cause=ev))
    err = body["error"]
    assert status == 500 and err["code"] == 500 and err["message"] == PROSE
    assert err["type"] == "vram_fit"
    ff = err["fit_failure"]
    assert ff["kind"] == "vram_fit" and ff["code"] == "wont_fit"
    assert ff["need_bytes"] == 11_253_969_110 and ff["budget_bytes"] == 82_182_144
    assert ff["blocked_by"] == ["Qwen3.8_4B_Distilled_GGUF"]
    assert ff["protected"][0]["why"] == "static (locked residency)"
    assert ff["external_floor_bytes"] == 1_073_741_824
    assert ff["free_vram_device_bytes"] == 1_155_923_968
    assert err["load_failure"]["class"] == "vram_fit"
    # an exception cause (the non-streaming path) threads the same way
    exc = remote.RemoteLoadError("worker ae-worker failed for m: ...", load_failure=LF)
    body, _ = _body(V1._openai_error(f"{type(exc).__name__}: {exc}", 500, "api_error", cause=exc))
    assert body["error"]["fit_failure"]["code"] == "wont_fit"
    # no cause / prose-only cause: byte-identical envelope
    body, _ = _body(V1._openai_error(PROSE, 500, "api_error"))
    assert set(body["error"]) == {"message", "type", "code"} and body["error"]["type"] == "api_error"
    body, _ = _body(V1._openai_error(PROSE, 500, "api_error",
                                     cause=ES.ErrorEvent(request_id="r", message=PROSE)))
    assert set(body["error"]) == {"message", "type", "code"}


def test_anthropic_envelope_carries_the_same_structure(ctx, monkeypatch):
    monkeypatch.setattr(V1, "_request_diagnostics", lambda *a, **k: None)
    ev = ES.ErrorEvent(request_id="r", message=PROSE, load_failure=LF)
    body, status = _body(MSG._anthropic_error(PROSE, 500, "api_error", cause=ev))
    assert body["type"] == "error" and status == 500
    err = body["error"]
    assert err["message"] == PROSE and err["type"] == "vram_fit"
    assert err["fit_failure"]["blocked_by"] == ["Qwen3.8_4B_Distilled_GGUF"]
    body, _ = _body(MSG._anthropic_error(PROSE, 500, "api_error"))
    assert set(body["error"]) == {"type", "message"}


def test_worker_error_dict_reaches_the_envelope_through_central(ctx, monkeypatch):
    """The whole wire: worker _event_to_dict -> central _event_from_worker_line ->
    _LoadFailed -> ErrorEvent -> envelope; the prose is humanised, the
    structure is intact."""
    monkeypatch.setattr(V1, "_request_diagnostics", lambda *a, **k: None)
    worker_ev = ES.ErrorEvent(request_id="w", message="LoadRefusal: won't fit on GPU",
                              load_failure=LF, code="vram_fit")
    line = A._event_to_dict(worker_ev)
    assert line["load_failure"] == LF and line["code"] == "vram_fit"
    assert A._event_to_dict(ES.ErrorEvent(request_id="w", message="m")) == {"type": "error",
                                                                              "message": "m"}
    central_ev = remote._event_from_worker_line(line, "r1")
    lf = remote._LoadFailed(remote._humanize_worker_error("ae-worker", central_ev.message),
                            load_failure=central_ev.load_failure, code=central_ev.code)
    out = ES.ErrorEvent(request_id="r1", message=lf.message, load_failure=lf.load_failure,
                        code=lf.code)
    body, _ = _body(V1._openai_error(out.message, 500, "api_error", cause=out))
    assert body["error"]["message"].startswith("The 'ae-worker' worker could not complete")
    assert body["error"]["fit_failure"]["need_bytes"] == 11_253_969_110


def _worker(loaded, allocations, slots=()):
    return {"id": "w1", "name": "ae-worker", "pkg_version": AM.NO_EVICT_MIN_PKG_VERSION,
            "loaded_models": list(loaded), "allocations": list(allocations),
            "slots": list(slots), "models_local": [], "loading": [], "provisioning": [],
            "provision_progress": {}, "load_reports": {}, "refused": {}}


def test_central_reads_measured_residency_not_membership(monkeypatch):
    """INVARIANT (F3): _polite_admits and load_state_for_model consult the
    allocation row's `materialized`; a loaded_models entry whose row says
    materialized:false is NOT "already resident" and NOT healthy (so the hold
    can never say "loaded and idle" for a hollow runner). A materialized row
    or a healthy slot child reads resident as before.
    LIVE CASE: coder-next 2026-09-29 12:24-12:32 (three identical cycles)."""
    MK = "Qwen3-Coder-Next-GGUF"
    hollow = _worker([MK], [{"kind": "ram", "model_key": MK, "materialized": False}])
    assert W._resident_materialized(hollow, MK) is False
    monkeypatch.setattr(W, "_free_room_probe", None)
    admits, why = W._polite_admits(hollow, MK)
    assert "already resident" not in why                      # falls to the worker's own admission
    monkeypatch.setattr(W, "_live_health", lambda w: {"loaded_models": [MK]})
    monkeypatch.setattr(W, "worker_store", types.SimpleNamespace(get=lambda wid: hollow))
    ls = W.load_state_for_model(MK, "w1", 0.0)
    assert ls["healthy"] is False and ls["materialized"] is False

    live = _worker([MK], [{"kind": "ram", "model_key": MK, "materialized": True}])
    assert W._resident_materialized(live, MK) is True
    assert W._polite_admits(live, MK) == (True, "already resident on worker")
    monkeypatch.setattr(W, "worker_store", types.SimpleNamespace(get=lambda wid: live))
    assert W.load_state_for_model(MK, "w1", 0.0)["healthy"] is True

    seated = _worker([], [{"kind": "slot", "model_key": MK, "healthy": True, "materialized": True}],
                     slots=[{"model_key": MK, "healthy": True, "materialized": True}])
    assert W._polite_admits(seated, MK)[1] == "already seated in a healthy native slot"
    unknown = _worker([MK], [])                                # an older worker: no row speaks
    assert W._resident_materialized(unknown, MK) is None
    assert W._polite_admits(unknown, MK) == (True, "already resident on worker")


def test_load_state_carries_the_structured_load_failure(monkeypatch):
    import time
    MK = "Qwen3.8-9B-Distill-GGUF"
    w = _worker([], [])
    w["load_reports"] = {MK: {"ok": False, "error": "LoadRefusal: won't fit on GPU",
                              "ts": time.time(), "load_failure": LF}}
    monkeypatch.setattr(W, "_live_health", lambda w: None)
    monkeypatch.setattr(W, "worker_store", types.SimpleNamespace(get=lambda wid: w))
    ls = W.load_state_for_model(MK, "w1", 0.0)
    assert ls["error"].startswith("LoadRefusal") and ls["load_failure"]["fit_failure"]["code"] == "wont_fit"
    assert ls["healthy"] is False and ls["materialized"] is None
