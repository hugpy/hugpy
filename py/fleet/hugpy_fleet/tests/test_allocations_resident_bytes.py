"""Measured host-RAM occupancy on ``kind:'ram'`` allocation rows.

Operator ruling 2026-07-28: worker-side MEASUREMENTS are the truth for
residency/occupancy. RAM rows previously carried only ``model_bytes`` (the
model's ON-DISK size), so ae reported 77 GB of "resident" models against 26 GB
physically used and the console had to widen its RAM denominator.

``_allocations`` now also emits, OMIT-WHEN-UNSET:
  ram_resident_bytes  — measured host RAM the model occupies now
  ram_resident_source — 'smaps' (page residency of this process's mappings of the
                    model's own files) | 'torch' (parameter+buffer bytes on
                    device 'cpu')
A file size is never promoted into ram_resident_bytes: with no measurement BOTH keys
are absent.

Run: venv/bin/python -m pytest tests/test_allocations_resident_bytes.py -q
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ag = importlib.import_module("hugpy_fleet.worker.agent")

MIB = 1 << 20

# Two mappings of the SAME weight file (r-- and r-x segments), one mapping of a
# second file in the same model dir, an anonymous mapping, a pseudo-path, a
# deleted-suffix mapping, and an UNRELATED file — exactly the shapes /proc emits.
_SMAPS = """\
7f0000000000-7f0000100000 r--p 00000000 08:01 101   /store/models/qwen/model-00001.gguf
Size:               1024 kB
Rss:                 512 kB
Pss:                 512 kB
VmFlags: rd mr mw me
7f0000100000-7f0000200000 r-xp 00100000 08:01 101   /store/models/qwen/model-00001.gguf
Size:               1024 kB
Rss:                 256 kB
7f0000200000-7f0000300000 r--p 00000000 08:01 102   /store/models/qwen/mmproj.gguf
Size:               1024 kB
Rss:                 128 kB
7f0000300000-7f0000400000 rw-p 00000000 00:00 0
Size:               1024 kB
Rss:                1024 kB
7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0     [stack]
Size:                132 kB
Rss:                 132 kB
7f0000400000-7f0000500000 r--p 00000000 08:01 103   /store/models/other/weights.safetensors (deleted)
Size:               1024 kB
Rss:                  64 kB
7f0000500000-7f0000600000 r-xp 00000000 08:01 104   /usr/lib/libc.so.6
Size:               1024 kB
Rss:                  32 kB
"""


# ═══════════ the parse: group by pathname, sum Rss ══════════════════════════
def test_parse_groups_and_sums_rss_per_path():
    d = ag._parse_smaps_rss_by_path(_SMAPS)
    # two segments of the same file collapse into one summed entry
    assert d["/store/models/qwen/model-00001.gguf"] == (512 + 256) * 1024
    assert d["/store/models/qwen/mmproj.gguf"] == 128 * 1024
    # "(deleted)" is stripped so a replaced file still groups with its path
    assert d["/store/models/other/weights.safetensors"] == 64 * 1024
    assert d["/usr/lib/libc.so.6"] == 32 * 1024


def test_parse_skips_anonymous_and_pseudo_mappings():
    d = ag._parse_smaps_rss_by_path(_SMAPS)
    assert all(p.startswith("/") for p in d)
    assert "[stack]" not in d
    # the 1024 kB anonymous mapping must not have been attributed to the file
    # mapped just before it.
    assert d["/store/models/qwen/mmproj.gguf"] == 128 * 1024


def test_parse_degrades_on_garbage():
    assert ag._parse_smaps_rss_by_path("") == {}
    assert ag._parse_smaps_rss_by_path("not smaps at all\nRss: wat kB\n") == {}


# ═══════════ the join: only mappings under the model's store dir ════════════
def test_resident_bytes_sums_only_the_models_own_files():
    rss = ag._parse_smaps_rss_by_path(_SMAPS)
    got = ag._resident_bytes_under_dir(rss, "/store/models/qwen")
    assert got == (512 + 256 + 128) * 1024      # libc / other model excluded


def test_resident_bytes_is_none_when_nothing_is_mapped():
    """None, never 0 — no mapping means no measurement, so the caller OMITS the
    field rather than claiming a measured zero."""
    rss = ag._parse_smaps_rss_by_path(_SMAPS)
    assert ag._resident_bytes_under_dir(rss, "/store/models/absent") is None
    assert ag._resident_bytes_under_dir({}, "/store/models/qwen") is None
    assert ag._resident_bytes_under_dir(rss, "") is None


def test_resident_bytes_prefix_match_is_directory_bounded():
    """/store/models/qwe must not swallow /store/models/qwen's files."""
    rss = ag._parse_smaps_rss_by_path(_SMAPS)
    assert ag._resident_bytes_under_dir(rss, "/store/models/qwe") is None


# ═══════════ the row: measured-first, omit-when-unset ═══════════════════════
def _ram_rows(monkeypatch, *, store_dirs, smaps, inproc):
    monkeypatch.setattr(ag, "_slot_statuses", lambda: [])
    monkeypatch.setattr(ag, "loaded_model_keys", lambda: list(store_dirs))
    monkeypatch.setattr(ag, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(ag, "_is_materialized", lambda mk: True)
    monkeypatch.setattr(ag, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(
        ag, "_loaded_detail",
        lambda: {mk: {"model_bytes": 77 * (1 << 30)} for mk in store_dirs})
    monkeypatch.setattr(ag, "_inprocess_gpu_bytes", lambda: inproc)
    monkeypatch.setattr(ag, "_model_store_dir", lambda mk: store_dirs[mk])
    monkeypatch.setattr(ag, "_smaps_rss_by_path",
                        lambda: ag._parse_smaps_rss_by_path(smaps))
    return {r["model_key"]: r for r in ag._allocations()
            if r["kind"] == "ram"}


def test_row_reports_measured_smaps_bytes(monkeypatch):
    rows = _ram_rows(monkeypatch,
                     store_dirs={"qwen": "/store/models/qwen"},
                     smaps=_SMAPS, inproc={})
    r = rows["qwen"]
    assert r["ram_resident_bytes"] == (512 + 256 + 128) * 1024
    assert r["ram_resident_source"] == "smaps"
    # the on-disk size is untouched and stays the upper bound
    assert r["model_bytes"] == 77 * (1 << 30)
    assert r["ram_resident_bytes"] < r["model_bytes"]


def test_row_falls_back_to_torch_cpu_bytes(monkeypatch):
    rows = _ram_rows(
        monkeypatch,
        store_dirs={"dan": "/store/models/dan"},          # nothing mapped
        smaps=_SMAPS,
        inproc={"dan": {"vram_bytes": 0, "device": "cpu",
                        "cpu_bytes": 3 * MIB}})
    r = rows["dan"]
    assert r["ram_resident_bytes"] == 3 * MIB
    assert r["ram_resident_source"] == "torch"


def test_row_omits_both_keys_with_no_measurement(monkeypatch):
    rows = _ram_rows(
        monkeypatch,
        store_dirs={"dan": "/store/models/dan"},
        smaps=_SMAPS,
        inproc={"dan": {"vram_bytes": 4 * MIB, "device": "cuda",
                        "cpu_bytes": 0}})
    r = rows["dan"]
    assert "ram_resident_bytes" not in r and "ram_resident_source" not in r
    assert r["model_bytes"] == 77 * (1 << 30)      # labeled fallback survives


def test_row_survives_a_proc_hiccup(monkeypatch):
    def boom():
        raise OSError("/proc went away")
    monkeypatch.setattr(ag, "_slot_statuses", lambda: [])
    monkeypatch.setattr(ag, "loaded_model_keys", lambda: ["qwen"])
    monkeypatch.setattr(ag, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(ag, "_is_materialized", lambda mk: None)
    monkeypatch.setattr(ag, "_gpu_process_vram", lambda: {})
    monkeypatch.setattr(ag, "_loaded_detail", lambda: {"qwen": {}})
    monkeypatch.setattr(ag, "_inprocess_gpu_bytes", lambda: {})
    monkeypatch.setattr(ag, "_model_store_dir", lambda mk: "/store/models/qwen")
    monkeypatch.setattr(ag, "_smaps_rss_by_path", boom)
    rows = [r for r in ag._allocations() if r["kind"] == "ram"]
    assert len(rows) == 1                       # the heartbeat still produced a row
    assert "ram_resident_bytes" not in rows[0]


def test_smaps_read_degrades_to_empty_off_proc(monkeypatch):
    monkeypatch.setattr(ag, "_SMAPS_CACHE", {"at": 0.0, "value": {}})
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda p, *a, **k: (_ for _ in ()).throw(OSError("nope"))
        if str(p) == "/proc/self/smaps" else real_open(p, *a, **k))
    assert ag._smaps_rss_by_path() == {}


# ═══════════ torch cpu_bytes is carried out of the introspection ════════════
def test_inprocess_gpu_bytes_emits_cpu_bytes_key():
    """The CPU analog must ride alongside vram_bytes/device (it is what the
    'torch' ram_resident_source reads)."""
    import inspect
    src = inspect.getsource(ag._inprocess_gpu_bytes)
    assert '"cpu_bytes": cpu' in src
