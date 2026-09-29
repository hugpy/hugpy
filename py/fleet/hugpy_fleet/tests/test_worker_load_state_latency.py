"""LOAD-STATE-LATENCY (2026-09-29) — the worker must SAY "loading" while it loads.

Operator: "how is the ui always so slow attributing a model to resources? it can
never say loading because it goes through a full response to a query before it
even gets onto the vram ui."

Root cause: the heartbeat's ``loading`` list came ONLY from dispatch's
``_BUILDING`` set, which brackets the lazy runner-WRAPPER construction
(milliseconds). The real load — slot child spawn + wait-healthy, or an
in-process ``ensure_loaded`` — happens on first ``.runner`` access, outside
that bracket, so ``loading`` was empty for the entire cold load and the row
flipped straight from "not loaded" to "serving" after the first answer. And
nothing woke the 15 s beat at the load edges.

These tests pin the worker side:
  * ``_loading_model_keys`` unions dispatch-building, the slot rows' own
    ``loading`` claim, and a cold request's LOAD INTENT — minus anything that is
    already resident (healthy slot / materialized in-process).
  * ``_load_intent_begin`` wakes the beat only for a COLD model; ``_load_intent_end``
    wakes it again only when the begin did.
  * ``_allocations`` marks the loading rows and synthesizes a row for a seat
    that is loading a model it has not claimed as ``model_key`` yet.
  * the central liveness row carries the per-allocation load state.

Run: venv/bin/python -m pytest tests/test_worker_load_state_latency.py -q
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ag = importlib.import_module("hugpy_fleet.worker.agent")
hb = importlib.import_module("hugpy_fleet.central.heartbeat_db")

GIB = 1 << 30
MIB = 1024 * 1024


def _slot_row(**over):
    row = {
        "slot_id": "1", "model_key": "coder", "healthy": True, "busy": False,
        "endpoint": "http://x:8101", "rss_bytes": 2 * GIB,
        "n_gpu_layers": 17, "ctx": 16384, "child_pid": 90004242,
        "last_used": 0.0,
    }
    row.update(over)
    return row


def _clear_intent():
    with ag._LOAD_INTENT_LOCK:
        ag._LOAD_INTENT.clear()


def _stub_residency(monkeypatch, *, materialized=(), slot_backed=(), building=()):
    monkeypatch.setattr(ag, "_is_materialized",
                        lambda mk: True if mk in materialized else None)
    get = importlib.import_module("hugpy_engine.llama.runners.get")
    monkeypatch.setattr(get, "slot_backed_model_keys", lambda: set(slot_backed))
    disp = importlib.import_module("hugpy_engine.dispatch.dispatch")
    monkeypatch.setattr(disp, "loading_model_keys", lambda: sorted(building))


# ═══════════ 1. the loading union ═══════════════════════════════════════════
def test_slot_load_claim_is_reported_as_loading(monkeypatch):
    """A slot whose /load is in flight says so on its status row; the beat
    reports that model as loading from the first second, not after healthy."""
    _clear_intent()
    _stub_residency(monkeypatch)
    rows = [_slot_row(model_key=None, healthy=False, loading="coder", loading_since=1.0)]
    assert ag._loading_model_keys(rows) == ["coder"]


def test_cold_request_intent_reads_as_loading_until_resident(monkeypatch):
    """The admission window BEFORE any slot claims the model (evict-to-fit,
    MoE planning) counts as loading; the moment a healthy slot seats it, the
    key drops out — even while the request is still answering."""
    _clear_intent()
    _stub_residency(monkeypatch)
    assert ag._load_intent_begin("coder") is True
    try:
        assert ag._loading_model_keys([]) == ["coder"]
        assert ag._loading_model_keys([_slot_row(model_key="coder", healthy=False)]) == ["coder"]
        assert ag._loading_model_keys([_slot_row(model_key="coder", healthy=True)]) == []
    finally:
        ag._load_intent_end("coder", True)
    assert ag._loading_model_keys([]) == []


def test_warm_request_never_reads_as_loading(monkeypatch):
    """A request for an already-materialized in-process model (or a cached
    slot-backed runner) must not flash "loading"."""
    _clear_intent()
    _stub_residency(monkeypatch, materialized={"warm"}, slot_backed={"seated"})
    assert ag._load_intent_begin("warm") is False
    assert ag._load_intent_begin("seated") is False
    try:
        assert ag._loading_model_keys([]) == []
        assert ag._loading_model_keys(None) == []
        # ...but a slot row that has CLAIMED the model without being healthy
        # contradicts the cached runner: that seat is (re)loading it.
        assert ag._loading_model_keys([_slot_row(model_key="seated", healthy=False)]) == ["seated"]
    finally:
        ag._load_intent_end("warm", False)
        ag._load_intent_end("seated", False)


def test_dispatch_building_still_counts(monkeypatch):
    _clear_intent()
    _stub_residency(monkeypatch, building={"transformers-x"})
    assert ag._loading_model_keys(None) == ["transformers-x"]


# ═══════════ 2. the beat is woken at the cold edges only ════════════════════
def test_cold_intent_wakes_the_beat_at_both_edges(monkeypatch):
    _clear_intent()
    _stub_residency(monkeypatch)
    ag._HB_WAKE.clear()
    cold = ag._load_intent_begin("coder")
    assert cold is True and ag._HB_WAKE.is_set()
    ag._HB_WAKE.clear()
    ag._load_intent_end("coder", cold)
    assert ag._HB_WAKE.is_set()
    assert ag._load_intent_keys() == []
    ag._HB_WAKE.clear()


def test_warm_intent_never_wakes_the_beat(monkeypatch):
    _clear_intent()
    _stub_residency(monkeypatch, materialized={"warm"})
    ag._HB_WAKE.clear()
    cold = ag._load_intent_begin("warm")
    assert cold is False and not ag._HB_WAKE.is_set()
    ag._load_intent_end("warm", cold)
    assert not ag._HB_WAKE.is_set()


def test_materialize_marks_intent_and_wakes(monkeypatch):
    """The probe/warm path (_materialize) is a load window too."""
    _clear_intent()
    _stub_residency(monkeypatch)
    monkeypatch.setattr(ag, "_evt", None, raising=False)
    seen = {}

    class Runner:
        model_key = "coder"

        def ensure_loaded(self):
            seen["during"] = ag._load_intent_keys()
            seen["woken"] = ag._HB_WAKE.is_set()

    ag._HB_WAKE.clear()
    ag._materialize(Runner())
    assert seen == {"during": ["coder"], "woken": True}
    assert ag._load_intent_keys() == []
    ag._forget_materialized("coder")
    ag._HB_WAKE.clear()


# ═══════════ 3. allocation rows carry the load state ════════════════════════
def _alloc(monkeypatch, rows, *, loading=()):
    monkeypatch.setattr(ag, "_gpu_process_vram", lambda: {90004242: {"mib": 4096}})
    monkeypatch.setattr(ag, "_loaded_detail", lambda: {})
    monkeypatch.setattr(ag, "_inprocess_gpu_bytes", lambda: {})
    monkeypatch.setattr(ag, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(ag, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(ag, "_slot_total_layers_fallback", lambda mk: None)
    monkeypatch.setattr(ag, "_loading_model_keys", lambda slots=None: sorted(loading))
    return ag._allocations(slot_statuses=rows)


def test_slot_row_mid_load_is_loading_not_materialized(monkeypatch):
    """Child spawned (model_key claimed, VRAM growing) but not healthy yet."""
    out = _alloc(monkeypatch, [_slot_row(healthy=False, loading="coder", loading_since=5.0)],
                 loading={"coder"})
    assert len(out) == 1
    row = out[0]
    assert row["kind"] == "slot" and row["model_key"] == "coder"
    assert row["loading"] is True
    assert row["materialized"] is False
    assert row["loading_since"] == 5.0
    assert row["vram_bytes"] == 4096 * MIB     # measured so far, still counted


def test_pre_claim_load_gets_a_synthetic_row(monkeypatch):
    """The slot is in /load preflight for `coder` while still reporting the
    previous occupant: both rows exist, the incoming one flagged loading."""
    out = _alloc(monkeypatch, [_slot_row(model_key="old", healthy=True, loading="coder")],
                 loading={"coder"})
    keys = {r["model_key"]: r for r in out}
    assert set(keys) == {"old", "coder"}
    assert keys["coder"]["loading"] is True and keys["coder"]["materialized"] is False
    assert keys["coder"]["healthy"] is False and keys["coder"]["slot_id"] == "1"
    assert "loading" not in keys["old"]


def test_cold_intent_without_any_row_gets_a_synthetic_ram_row(monkeypatch):
    out = _alloc(monkeypatch, [], loading={"newbie"})
    assert out == [{"kind": "ram", "model_key": "newbie", "device": None,
                    "vram_bytes": None, "serving": False, "last_used": None,
                    "loading": True, "materialized": False}]


def test_healthy_slot_row_carries_no_loading_key(monkeypatch):
    """A healthy seat never reads as loading. Since core-isolation step 2 (F4)
    every slot row states ``materialized`` from the slot's own ``healthy``
    (known_good/test_hollow_rows_contract.py), so a healthy row says True."""
    out = _alloc(monkeypatch, [_slot_row()])
    assert "loading" not in out[0]
    assert out[0]["materialized"] is True


# ═══════════ 4. central: the liveness row carries the per-allocation state ══
def test_liveness_row_carries_slim_allocation_load_state():
    rec = {"id": "w", "name": "ae", "allocations": [
        {"kind": "slot", "slot_id": "1", "model_key": "coder", "healthy": False, "busy": False,
         "loading": True, "materialized": False, "vram_bytes": 4 * GIB, "device": "cuda",
         "rss_bytes": 1, "model_bytes": 99, "alloc_source": {"kind": "x"}},
        {"kind": "ram", "model_key": "qwythos", "materialized": True, "vram_bytes": None},
        {"junk": True},
    ], "loading": ["coder"]}
    live = hb.liveness_from_record(rec, 100.0, 45.0)
    assert live["loading"] == ["coder"]
    assert live["allocations"] == [
        {"model_key": "coder", "kind": "slot", "slot_id": "1", "healthy": False, "busy": False,
         "loading": True, "materialized": False, "vram_bytes": 4 * GIB, "device": "cuda"},
        {"model_key": "qwythos", "kind": "ram", "materialized": True},
    ]
