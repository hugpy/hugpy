"""Allocation rows must say what they actually know — operator 2026-07-25:
"i need for the state to accurately show what it is. it shows missing for
nearly everything, even models it's serving".

Three defects in ``worker_agent.agent._allocations``:

  1. SLOT rows never set ``serving`` at all — a slot that had just answered an
     inference on ae reported ``serving: null``. Fixed: slot rows compute
     ``serving`` (and surface ``last_used``) on the SAME ``_SERVING_WINDOW_S``
     terms the ram rows use, plus ``busy`` = answering right now.
  2. ``device`` was derived ONLY from nvidia-smi's per-PID compute-app map. On
     computron nvidia-smi fails outright ("Failed to initialize NVML: Driver/
     library version mismatch") so EVERY row read ``device: null`` — even a live
     GPU seat — while torch on the same box saw the card fine.
  3. RAM rows had the same hole via torch: an in-process GGUF ``Llama`` handle
     is not a torch module, so ``device``/``vram_bytes`` were null even though
     the load DECLARED its placement in n_gpu_layers/gpu_pct.

Fix for 2+3 respects the residency doctrine ("measured, never inferred from
membership"): the fallback is the placement the worker ITSELF declared at
launch, and it is LABELED ``device_source: 'inferred'`` vs ``'measured'`` so
nothing can launder a guess into a measurement. ``vram_bytes`` is never
inferred. ``device_source`` is OMIT-WHEN-UNSET (extra=forbid wire landmine +
old central sees an unchanged shape).

Run: venv/bin/python -m pytest tests/test_allocation_state_honesty.py -q
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ag = importlib.import_module("hugpy_fleet.worker.agent")

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


def _alloc(monkeypatch, slot_row=None, *, gpu_procs=None, loaded=(),
           detail=None, inproc=None, last_used=None):
    """Drive _allocations with every external read stubbed."""
    monkeypatch.setattr(ag, "_gpu_process_vram", lambda: dict(gpu_procs or {}))
    monkeypatch.setattr(ag, "_loaded_detail", lambda: dict(detail or {}))
    monkeypatch.setattr(ag, "_inprocess_gpu_bytes", lambda: dict(inproc or {}))
    monkeypatch.setattr(ag, "loaded_model_keys", lambda: list(loaded))
    monkeypatch.setattr(ag, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(ag, "_slot_total_layers_fallback", lambda mk: None)
    lu = dict(last_used or {})
    mod = importlib.import_module("hugpy_engine.dispatch.dispatch")
    monkeypatch.setattr(mod, "last_used_snapshot", lambda: lu, raising=False)
    return ag._allocations(slot_statuses=[slot_row] if slot_row else [])


# ═══════════ 1. slot rows report serving / last_used ════════════════════════
def test_busy_slot_is_serving():
    """A slot with in-flight work is answering RIGHT NOW — measured directly,
    no clock window involved (this is the ae live-proof row)."""
    assert ag._slot_serving(_slot_row(busy=True)) is True


def test_recently_used_slot_is_serving():
    now = time.time()
    assert ag._slot_serving(_slot_row(last_used=now - 5.0), now) is True


def test_cold_slot_is_not_serving():
    """Seated since yesterday's test churn -> idle, same as an idle ram row."""
    now = time.time()
    stale = now - (ag._SERVING_WINDOW_S + 60)
    assert ag._slot_serving(_slot_row(last_used=stale), now) is False


def test_never_used_slot_is_not_serving_and_last_used_is_none():
    """The slot seeds last_used=0.0 at construction; 0.0 means NEVER, which is
    None on the wire (matching the ram rows' None=never)."""
    row = _slot_row(last_used=0.0)
    assert ag._slot_last_used(row) is None
    assert ag._slot_serving(row, time.time()) is False


def test_slot_allocation_row_carries_serving_and_last_used(monkeypatch):
    """The regression itself: the row used to omit `serving` entirely."""
    now = time.time()
    out = _alloc(monkeypatch, _slot_row(busy=True, last_used=now - 2))
    a = out[0]
    assert a["kind"] == "slot"
    assert a["serving"] is True            # was ABSENT -> read as null
    assert a["last_used"] == now - 2


def test_slot_and_ram_serving_agree_on_the_same_window(monkeypatch):
    """Slot and ram rows must MEAN the same thing by `serving`. Same age either
    side of the window boundary -> same verdict on both kinds."""
    now = time.time()
    fresh, stale = now - 1.0, now - (ag._SERVING_WINDOW_S + 60)
    for age, want in ((fresh, True), (stale, False)):
        out = _alloc(monkeypatch, _slot_row(last_used=age),
                     loaded=["ram-model"], last_used={"ram-model": age})
        slot = [r for r in out if r["kind"] == "slot"][0]
        ram = [r for r in out if r["kind"] == "ram"][0]
        assert slot["serving"] is want
        assert ram["serving"] is want


# ═══════════ 2. device: measured vs inferred ════════════════════════════════
def test_inferred_device_from_declared_placement():
    assert ag._inferred_device(17) == "cuda"      # partial offload
    assert ag._inferred_device(-1) == "cuda"      # all layers on GPU
    assert ag._inferred_device(0) == "cpu"
    assert ag._inferred_device(None) is None      # no basis -> say nothing
    assert ag._inferred_device(None, 42.0) == "cuda"   # engine with no ngl
    assert ag._inferred_device(None, 0) == "cpu"
    assert ag._inferred_device("junk") is None    # never crash a heartbeat


def test_measured_device_wins_and_is_labeled(monkeypatch):
    """nvidia-smi joined on child_pid = ground truth: device + real bytes,
    stamped 'measured'."""
    out = _alloc(monkeypatch, _slot_row(),
                 gpu_procs={90004242: {"name": "llama-server", "mib": 3000}})
    a = out[0]
    assert a["device"] == "cuda"
    assert a["vram_bytes"] == 3000 * MIB
    assert a["device_source"] == "measured"


def test_broken_nvidia_smi_infers_gpu_seat_and_labels_it(monkeypatch):
    """computron: nvidia-smi fails -> gpu_procs {}. A slot child launched with
    ngl=17 IS GPU-resident; reporting device:null read as "missing". Now it
    reports cuda, EXPLICITLY inferred, with vram_bytes still null (placement is
    knowable without nvidia-smi; a byte count is not)."""
    out = _alloc(monkeypatch, _slot_row(n_gpu_layers=17), gpu_procs={})
    a = out[0]
    assert a["device"] == "cuda"
    assert a["device_source"] == "inferred"
    assert a["vram_bytes"] is None         # never a fabricated number


def test_broken_nvidia_smi_infers_cpu_slot(monkeypatch):
    out = _alloc(monkeypatch, _slot_row(n_gpu_layers=0), gpu_procs={})
    a = out[0]
    assert a["device"] == "cpu"
    assert a["device_source"] == "inferred"


def test_no_basis_at_all_stays_null_with_no_provenance(monkeypatch):
    """NEVER report a confident device you cannot justify. No nvidia-smi AND no
    declared placement -> device null and device_source ABSENT (there is no
    claim to label)."""
    out = _alloc(monkeypatch, _slot_row(n_gpu_layers=None), gpu_procs={})
    a = out[0]
    assert a["device"] is None
    assert "device_source" not in a


def test_measured_cpu_from_live_non_gpu_child_stays_measured(monkeypatch):
    """nvidia-smi WORKS but the child isn't a compute app -> genuinely CPU, and
    that is a measurement, not an inference."""
    out = _alloc(monkeypatch, _slot_row(n_gpu_layers=0),
                 gpu_procs={9999: {"name": "other", "mib": 100}})
    a = out[0]
    assert (a["device"], a["vram_bytes"]) == ("cpu", 0)
    assert a["device_source"] == "measured"


# ═══════════ 3. ram rows get the same treatment ═════════════════════════════
def test_ram_row_torch_measured_device_labeled(monkeypatch):
    out = _alloc(monkeypatch, loaded=["vl"],
                 inproc={"vl": {"vram_bytes": 7 * GIB, "device": "cuda"}})
    a = out[0]
    assert (a["device"], a["vram_bytes"]) == ("cuda", 7 * GIB)
    assert a["device_source"] == "measured"


def test_ram_row_inprocess_gguf_infers_from_declared_placement(monkeypatch):
    """An in-process GGUF Llama handle is not a torch module, so torch sees
    nothing — but the load declared n_gpu_layers. Previously device:null."""
    out = _alloc(monkeypatch, loaded=["gguf-inproc"],
                 detail={"gguf-inproc": {"n_gpu_layers": 33, "gpu_pct": 100}},
                 inproc={})
    a = out[0]
    assert a["kind"] == "ram"
    assert a["device"] == "cuda"
    assert a["device_source"] == "inferred"
    assert a["vram_bytes"] is None


def test_ram_row_infers_from_gpu_pct_when_no_ngl(monkeypatch):
    out = _alloc(monkeypatch, loaded=["m"],
                 detail={"m": {"n_gpu_layers": None, "gpu_pct": 60}}, inproc={})
    assert out[0]["device"] == "cuda"
    assert out[0]["device_source"] == "inferred"


def test_ram_row_with_nothing_knowable_omits_provenance(monkeypatch):
    out = _alloc(monkeypatch, loaded=["m"], detail={"m": {}}, inproc={})
    a = out[0]
    assert a["device"] is None
    assert "device_source" not in a


# ═══════════ wire compatibility ═════════════════════════════════════════════
def test_device_source_is_the_only_new_key(monkeypatch):
    """WIRE LANDMINE: central->worker relay is pydantic extra=forbid on released
    workers, and central (newer) must tolerate rows WITHOUT the new keys. Assert
    exactly which keys are new so nothing else sneaks onto the wire."""
    baseline = {
        "kind", "model_key", "slot_id", "healthy", "busy", "endpoint",
        "rss_bytes", "n_gpu_layers", "ctx", "vram_bytes", "device",
    }
    out = _alloc(monkeypatch, _slot_row(), gpu_procs={})
    assert set(out[0]) - baseline == {"device_source", "serving", "last_used"}


def test_existing_field_meanings_unchanged(monkeypatch):
    """`device` must keep meaning 'the device the weights live on' and rss_bytes
    stays VmRSS verbatim — the fallback only fills a hole, never re-defines."""
    out = _alloc(monkeypatch, _slot_row(),
                 gpu_procs={90004242: {"name": "llama-server", "mib": 3000}})
    a = out[0]
    assert a["device"] in ("cuda", "cpu")
    assert a["rss_bytes"] == 2 * GIB
    assert a["n_gpu_layers"] == 17 and a["ctx"] == 16384


def test_rows_are_json_serializable(monkeypatch):
    import json
    out = _alloc(monkeypatch, _slot_row(busy=True), loaded=["m"],
                 detail={"m": {"n_gpu_layers": 0}}, inproc={},
                 last_used={"m": time.time()})
    json.dumps(out)          # heartbeat body must serialize
    assert len(out) == 2
