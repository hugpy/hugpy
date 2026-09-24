"""The resolver refuses a model the post-download admission HELD (alloc.force
bypasses for diagnosis); every refusal is rendered from a stored structured
diagnostics record (per-worker rows, failed predicate, log_ref) with no
advisory text; and resolving/auditing an installed fixture model makes zero
network attempts (the Hub resolver step is gone)."""
from __future__ import annotations

import json
import socket
import types

import pytest

from hugpy_engine import placement as P
from hugpy_engine import routing_diagnostics as RD
from hugpy_engine.resolvers import remote as R

HELD = ("'M' is held from the serving pool by admission: faulty_model: check_tensor_dims "
        "[integrity=faulty_model, grade=None, at=t, job=j]")


class Blocklist:
    def __init__(self, held=None, blocked=None):
        self.held, self.blocked = held, blocked

    def blocked_keys(self):
        return ()

    def block_reason(self, k):
        return self.blocked

    def admission_reason(self, k):
        return self.held


class Metrics(P.NullModelMetrics):
    def __init__(self):
        self.rows = []

    def record_refusal(self, diag):
        self.rows.append(json.loads(json.dumps(diag, default=str)))
        return len(self.rows)

    def find_refusal(self, rid):
        return next((r for r in self.rows if r.get("request_id") == rid), None)


WORKERS = [
    {"id": "w-aeb", "name": "aeb", "status": "online", "last_seen": 1.0, "admission": "approved",
     "serving_limits": {}, "models_local": ["M"], "loaded_models": ["M"], "models": ["M"],
     "vram_free": 12 * 2**30, "vram_total": 24 * 2**30, "slots": [],
     "load_reports": {"M": {"ok": False, "ts": 5.0, "error": "ValueError: Failed to load\nmore",
                            "load_failure": {"class": "faulty_model",
                                             "loader_stderr": "check_tensor_dims: wrong shape\n..."}}}},
    {"id": "w-comp", "name": "computron", "status": "online", "last_seen": 1.0, "admission": "approved",
     "serving_limits": {"in_process_max_concurrency": 2}, "models_local": [], "loaded_models": [],
     "models": [], "slots": []},
    {"id": "w-op", "name": "op", "status": "offline", "last_seen": 0.0, "slots": []},
]


class Registry(P.NullWorkerRegistry):
    def list_workers(self, *, online_only=True):
        return [w for w in WORKERS if not online_only or w["status"] == "online"]


@pytest.fixture
def seams(monkeypatch):
    m = Metrics()
    P.set_worker_registry(Registry())
    P.set_model_metrics(m)
    monkeypatch.setattr(R, "_worker_candidates_provider", None)
    RD._RECENT.clear()
    yield m
    P.reset_providers()


def req(**alloc):
    return types.SimpleNamespace(request_id="rq-1", model="M", alloc=alloc or None)


def test_held_model_is_refused_and_force_bypasses(seams):
    P.set_blocklist(Blocklist(held=HELD))
    assert R._blocked_reason("M", req()) == HELD
    assert R._blocked_reason("M", req(force=True)) is None
    assert R._gate_of(HELD) == "admission_held"
    # the admission refusal is permanent for the cold-hold classifier
    assert any(m in HELD.lower() for m in R._PERMANENT_LOAD_MARKERS)


def test_operator_block_outranks_force(seams):
    P.set_blocklist(Blocklist(blocked="'M' is blocked from the serving pool by the operator"))
    assert "blocked" in R._blocked_reason("M", req(force=True))


def test_null_blocklist_never_holds(seams):
    P.set_blocklist(None)
    assert R._blocked_reason("M", req()) is None


def test_held_refusal_is_rendered_from_a_stored_record(seams):
    msg = R._refusal_message("M", req(), "admission_held", HELD)
    assert "gate=admission_held" in msg and "held from the serving pool by admission" in msg
    assert "log_ref=compute_actions#1" in msg
    stored = RD.lookup("rq-1")
    assert stored["gate"] == "admission_held" and stored["log_ref"] == "compute_actions#1"
    assert seams.rows[0]["request_id"] == "rq-1"


def test_worker_busy_carries_structured_diagnostics(seams, monkeypatch):
    monkeypatch.setattr(R, "_inflight_count", lambda wid, mk: 1 if wid == "w-aeb" else 0)
    err = R._busy(WORKERS[0], "M", req=req(worker="aeb"), pool=None, task="text-generation")
    d = err.diagnostics
    assert d["gate"] == "worker_busy" and d["request_id"] == "rq-1"
    assert d["model"] == {"requested": "M", "resolved": "M"}
    assert d["alloc"] == {"worker": "aeb"}
    assert "in_flight_central >= concurrency_limit" in d["predicate"] and "aeb 1/1" in d["predicate"]
    rows = {r["name"]: r for r in d["candidates"]}
    assert set(rows) == {"aeb", "computron", "op"}
    aeb = rows["aeb"]
    assert aeb["holds"] == {"on_disk": True, "loaded": True, "designated": True}
    assert aeb["concurrency_limit"] == 1 and "absent" in aeb["limit_source"]
    assert aeb["in_flight_central"] == 1 and "in_flight_central=1 >= limit=1" in aeb["skipped"]
    assert aeb["last_load_failure"]["class"] == "faulty_model"
    assert aeb["last_load_failure"]["loader_stderr"] == "check_tensor_dims: wrong shape"
    assert rows["computron"]["concurrency_limit"] == 2
    assert "not a routing candidate" in rows["computron"]["skipped"]
    assert rows["op"]["skipped"].startswith("status=offline")
    assert "HUGPY_CENTRAL_GATE_WAIT_S=" in d["rule"]
    assert d["log_ref"] == "compute_actions#1" and seams.rows[0]["gate"] == "worker_busy"
    msg = err.stream_message()
    assert msg.startswith("worker_busy: gate=worker_busy")
    assert "predicate: no candidate admitted" in msg and "log_ref=compute_actions#1" in msg
    for advice in ("retry shortly", "try again", "please"):
        assert advice not in msg.lower()
    assert err.as_error()["error"]["diagnostics"]["request_id"] == "rq-1"


def test_diagnostics_store_failure_still_yields_message(monkeypatch):
    class Boom(P.NullModelMetrics):
        def record_refusal(self, diag):
            raise RuntimeError("store down")
    P.set_model_metrics(Boom())
    try:
        d = RD.record(RD.build(request_id="rq-x", requested="M", resolved="M",
                               gate="no_worker", predicate="p"))
        assert d["log_ref"].startswith("memory:") and RD.lookup("rq-x") is d
    finally:
        P.reset_providers()


def test_installed_model_resolves_offline(tmp_path, monkeypatch):
    """C: resolving an installed fixture model never touches the network."""
    from hugpy_storage import hugpy_marker as hm
    from hugpy_engine.apis import get_module as gm

    monkeypatch.setattr("hugpy_storage.model_metadata.model_metadata_store",
                        type("S", (), {"get_repo_info": lambda self, h: None})())

    def no_net(*a, **k):
        raise AssertionError("network attempted")
    monkeypatch.setattr(socket, "socket", no_net)
    d = tmp_path / "gguf" / "owner" / "M-GGUF"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    hm.write_hugpy_marker(str(d), hub_id="owner/M-GGUF", name="M-GGUF", framework="gguf",
                          tasks=["text-generation"], source="backfill",
                          hub_meta={"pipeline_tag": "text-generation", "license": "mit"})
    chain = gm.build_resolver_chain()
    assert "hub_model_info" not in [n for n, _ in chain] and "hub_meta" in [n for n, _ in chain]
    meta, sources = gm.enrich(str(d), "owner/M-GGUF", chain)
    assert meta.pipeline_tag == "text-generation" and sources["pipeline_tag"] == "hub_meta"
    assert meta.license == "mit" and meta.model_type == "llama"
    # comfy/local namespaces and uncaptured repos: nothing, no fetch
    assert gm.resolve_hub_meta(str(d), "comfy/x") == {}
    bare = tmp_path / "bare"
    bare.mkdir()
    assert gm.resolve_hub_meta(str(bare), "owner/other") == {}
