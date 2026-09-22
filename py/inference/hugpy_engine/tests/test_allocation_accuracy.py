"""Honest allocation accuracy — the operator's "17/undefined layers · 42.9 GB RSS".

Two accuracy defects in the slot allocation payload, verified on ae 2026-07-22
(Qwen3-Coder-Next, 17/48 offload):

  1. no total-layer field — the console rendered the literal "17/undefined".
     Fix: the slot reads the GGUF header block_count where it already resolves
     the model path (_build_cmd) and reports ``total_layers`` in status(); the
     agent's unified allocation row passes it through, with a CACHED
     _served_gguf_geometry fallback for adopted old-build slot children.
  2. rss_bytes is VmRSS, which counts the mmap'd GGUF's FILE-BACKED pages as
     resident — reclaimable page cache, not pinned RAM (ae: VmRSS 45.2G vs
     RssAnon 1.5G — ~28x overstated). Fix: /proc/<pid>/status's RssAnon/RssFile/
     RssShmem ride ALONGSIDE (rss_anon_bytes / rss_file_bytes /
     rss_shmem_bytes); ``rss_bytes`` keeps its VmRSS meaning verbatim for wire
     back-compat, and the new fields are OMIT-WHEN-UNSET so old central/UI see
     an unchanged shape.

Run: venv/bin/python -m pytest tests/test_allocation_accuracy.py -q
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sa = importlib.import_module("hugpy_engine.serve.slot_agent")

GIB = 1 << 30

_PROC_STATUS = """\
Name:\tllama-server
VmPeak:\t 47342592 kB
VmRSS:\t 45182016 kB
RssAnon:\t  1523712 kB
RssFile:\t 43610112 kB
RssShmem:\t   48192 kB
Threads:\t32
"""


# ═══════════ _proc_rss_detail: the honest split ═════════════════════════════
def test_proc_rss_detail_reads_anon_file_shmem(tmp_path, monkeypatch):
    fake = tmp_path / "status"
    fake.write_text(_PROC_STATUS)
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(fake if str(p).startswith("/proc/4242/") else p, *a, **k))
    d = sa._proc_rss_detail(4242)
    assert d == {
        "rss_anon_bytes": 1523712 * 1024,
        "rss_file_bytes": 43610112 * 1024,
        "rss_shmem_bytes": 48192 * 1024,
    }
    # the honest figure is ~28x smaller than raw VmRSS — the whole point.
    assert d["rss_anon_bytes"] * 20 < 45182016 * 1024


def test_proc_rss_detail_degrades_to_empty_on_unreadable_proc():
    # A vanished pid (or non-Linux) must yield {} — callers then OMIT the
    # fields; the heartbeat never crashes on the read.
    assert sa._proc_rss_detail(2**31 - 7) == {}


def test_proc_rss_detail_missing_split_keys_partial(tmp_path, monkeypatch):
    # An old kernel without RssAnon/RssFile: whatever IS present is returned,
    # nothing invented.
    fake = tmp_path / "status"
    fake.write_text("Name:\tx\nVmRSS:\t 100 kB\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(fake if str(p).startswith("/proc/7/") else p, *a, **k))
    assert sa._proc_rss_detail(7) == {}


# ═══════════ slot status(): fields present / omitted honestly ═══════════════
def _bare_slot():
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = "coder"
    s.ngl = 17
    s.total_layers = 48
    s.ctx = 16384
    s.threads = 6
    s.cpus = None
    s.gpu = "0"
    s.profile_bin = None
    s.expected_bytes = 46 * GIB
    s.loaded_at = s.last_used = 1.0
    s.last_load_error = None
    s.inflight = 0
    s.healthy = lambda: True
    s._self_heal = lambda: None
    s.model_path = None
    s.verify_identity = lambda force=False: (True, None)   # no child to probe here
    return s


def test_status_reports_total_layers_and_rss_split(monkeypatch):
    s = _bare_slot()
    s.proc = type("P", (), {"pid": 4242, "poll": lambda self: None})()
    monkeypatch.setattr(sa, "_proc_rss_bytes", lambda pid: 45182016 * 1024)
    monkeypatch.setattr(sa, "_proc_rss_detail",
                        lambda pid: {"rss_anon_bytes": 1 * GIB,
                                     "rss_file_bytes": 43 * GIB})
    monkeypatch.setattr(sa, "_allowed_cpus", lambda: None)
    import hugpy_engine.spill as spill
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: None)
    st = s.status()
    assert st["n_gpu_layers"] == 17
    assert st["total_layers"] == 48                     # the "of 48"
    assert st["rss_bytes"] == 45182016 * 1024           # VmRSS meaning UNCHANGED
    assert st["rss_anon_bytes"] == 1 * GIB              # the honest pinned figure
    assert st["rss_file_bytes"] == 43 * GIB             # mmap'd/cache share


def test_status_omits_split_when_proc_unreadable(monkeypatch):
    s = _bare_slot()
    s.proc = type("P", (), {"pid": 4242, "poll": lambda self: None})()
    monkeypatch.setattr(sa, "_proc_rss_bytes", lambda pid: 45182016 * 1024)
    monkeypatch.setattr(sa, "_proc_rss_detail", lambda pid: {})   # /proc failed
    monkeypatch.setattr(sa, "_allowed_cpus", lambda: None)
    import hugpy_engine.spill as spill
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: None)
    st = s.status()
    assert "rss_anon_bytes" not in st                   # OMITTED, never null-spam
    assert "rss_file_bytes" not in st
    assert st["rss_bytes"] == 45182016 * 1024           # legacy field intact


def test_status_dead_child_no_split_no_crash(monkeypatch):
    s = _bare_slot()
    s.model_key = None
    s.total_layers = None
    s.proc = None
    monkeypatch.setattr(sa, "_allowed_cpus", lambda: None)
    import hugpy_engine.spill as spill
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: None)
    st = s.status()
    assert st["rss_bytes"] == 0
    assert st["total_layers"] is None
    assert "rss_anon_bytes" not in st


# ═══════════ agent _allocations(): pass-through + fallbacks ═════════════════
ag = importlib.import_module("hugpy_fleet.worker.agent")


def _slot_status_row(**over):
    row = {
        "slot_id": "1", "model_key": "coder", "healthy": True, "busy": False,
        "endpoint": "http://x:8101", "rss_bytes": 45182016 * 1024,
        "n_gpu_layers": 17, "ctx": 16384, "child_pid": 4242,
    }
    row.update(over)
    return row


def _alloc_rows(monkeypatch, slot_row):
    # No GPU procs, no in-process residents — isolate the slot row logic.
    monkeypatch.setattr(ag, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(ag, "_loaded_detail", lambda: {})
    monkeypatch.setattr(ag, "_inprocess_gpu_bytes", lambda: {})
    monkeypatch.setattr(ag, "loaded_model_keys", lambda: [])
    return ag._allocations(slot_statuses=[slot_row])


def test_allocation_row_passes_through_new_fields(monkeypatch):
    row = _slot_status_row(total_layers=48,
                           rss_anon_bytes=1 * GIB, rss_file_bytes=43 * GIB)
    out = _alloc_rows(monkeypatch, row)
    assert len(out) == 1
    a = out[0]
    assert a["total_layers"] == 48
    assert a["rss_anon_bytes"] == 1 * GIB
    assert a["rss_file_bytes"] == 43 * GIB
    assert a["rss_bytes"] == 45182016 * 1024            # unchanged meaning


def test_allocation_row_old_slot_falls_back_to_agent_reads(monkeypatch):
    """An ADOPTED old-build slot child omits total_layers + the rss split; the
    agent fills both itself — geometry via the cached _served_gguf_geometry
    read, the split via /proc/<child_pid>/status on the same box."""
    ag._TOTAL_LAYERS_CACHE.clear()
    monkeypatch.setattr(ag, "_served_gguf_geometry",
                        lambda mk: ("/models/coder.gguf", 48))
    monkeypatch.setattr(sa, "_proc_rss_detail",
                        lambda pid: {"rss_anon_bytes": 2 * GIB,
                                     "rss_file_bytes": 40 * GIB})
    out = _alloc_rows(monkeypatch, _slot_status_row())   # no new fields reported
    a = out[0]
    assert a["total_layers"] == 48                       # geometry fallback
    assert a["rss_anon_bytes"] == 2 * GIB                # agent-side /proc read
    assert a["rss_file_bytes"] == 40 * GIB


def test_allocation_row_omits_when_nothing_knowable(monkeypatch):
    """Non-GGUF / unreadable everything: the new fields are ABSENT (wire shape
    unchanged for an old central), never null-stuffed, never a crash."""
    ag._TOTAL_LAYERS_CACHE.clear()
    monkeypatch.setattr(ag, "_served_gguf_geometry", lambda mk: (None, None))
    monkeypatch.setattr(sa, "_proc_rss_detail", lambda pid: {})
    out = _alloc_rows(monkeypatch, _slot_status_row())
    a = out[0]
    assert "total_layers" not in a
    assert "rss_anon_bytes" not in a
    assert "rss_file_bytes" not in a
    assert a["rss_bytes"] == 45182016 * 1024


def test_total_layers_fallback_is_cached_per_model(monkeypatch):
    """The geometry fallback parses a GGUF header — once per model, NOT once
    per heartbeat. Misses (None) are cached too."""
    ag._TOTAL_LAYERS_CACHE.clear()
    calls = {"n": 0}

    def _geom(mk):
        calls["n"] += 1
        return ("/m.gguf", 48)
    monkeypatch.setattr(ag, "_served_gguf_geometry", _geom)
    assert ag._slot_total_layers_fallback("coder") == 48
    assert ag._slot_total_layers_fallback("coder") == 48
    assert calls["n"] == 1                               # cached
    monkeypatch.setattr(ag, "_served_gguf_geometry", lambda mk: (None, None))
    assert ag._slot_total_layers_fallback("other") is None
    assert ag._slot_total_layers_fallback("other") is None
    assert "other" in ag._TOTAL_LAYERS_CACHE             # miss cached as well
    ag._TOTAL_LAYERS_CACHE.clear()


# ═══════════ _build_cmd return: total_layers rides the load ═════════════════
def test_load_stores_total_layers_from_build_cmd(monkeypatch):
    import threading
    s = sa.Slot.__new__(sa.Slot)
    s.model_key = None
    s.lock = threading.Lock()
    s.proc = None
    s.ngl = s.ctx = s.threads = s.cpus = s.gpu = None
    s.child_kind = None
    s.total_layers = None
    s.expected_bytes = None
    s.loaded_at = s.last_used = 0.0
    s.profile_bin = None
    s._load_failures = {}
    s._load_backoff_until = {}
    s.last_load_error = None
    s.child_base = "http://127.0.0.1:9101"
    monkeypatch.setattr(sa, "_build_cmd",
                        lambda *a, **k: (["true"], 17, 16384, 6, None, "cpp", 48, None))
    monkeypatch.setattr(sa, "_model_expected_bytes", lambda mk: 46 * GIB)
    monkeypatch.setattr(sa.subprocess, "Popen", lambda *a, **k: type(
        "P", (), {"pid": 1, "poll": lambda self: None})())
    s._kill = lambda: None
    s._wait_healthy = lambda: True
    s.status = lambda: {"model_key": s.model_key, "total_layers": s.total_layers}
    out = s.load("coder")
    assert s.total_layers == 48
    assert out["total_layers"] == 48
