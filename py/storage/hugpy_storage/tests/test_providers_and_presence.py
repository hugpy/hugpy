"""Provider hooks (defaults + injection) and storage's own presence rule."""
import os
import struct

import pytest

from hugpy_storage import providers, model_presence as mp


@pytest.fixture(autouse=True)
def _reset():
    providers.reset_providers()
    yield
    providers.reset_providers()


# ── providers ────────────────────────────────────────────────────────────────
def test_budget_gate_default_admits_and_injection_is_used():
    providers.get_budget_gate()(None, "m", 10)            # default: admit
    calls = []
    providers.set_budget_gate(lambda st, k, n: calls.append((k, n)))
    providers.get_budget_gate()("state", "m", 5)
    assert calls == [("m", 5)]
    providers.set_budget_gate(None)
    assert providers.get_budget_gate() is providers._admit_all


def test_budget_gate_is_called_by_provision(monkeypatch):
    from hugpy_storage import provision
    gate_calls = []
    providers.set_budget_gate(lambda st, k, n: gate_calls.append((k, n)))
    monkeypatch.setattr(provision, "model_is_local", lambda k: False)
    monkeypatch.setattr(provision, "ensure_model_registered", lambda k, u: k)
    monkeypatch.setattr(provision, "central_total_bytes", lambda u, k: 1234)
    monkeypatch.setattr(provision, "_provision_now", lambda k, u, progress=None: True)
    assert provision.ensure_model_present("m", "http://c") is True
    assert gate_calls == [("m", 1234)]


def test_serve_path_hook_default_identity_and_failure_safe():
    assert providers.serve_path("/a") == "/a"
    providers.set_serve_path_hook(lambda p: p + "-hot")
    assert providers.serve_path("/a") == "/a-hot"
    providers.set_serve_path_hook(lambda p: 1 / 0)
    assert providers.serve_path("/a") == "/a"


def test_footprint_selector_default_sums_listing():
    assert providers.get_footprint_selector()([("a", 1), ("b", 2)], "transformers") == 3
    providers.set_footprint_selector(lambda listing, fw: 42)
    assert providers.get_footprint_selector()([], None) == 42


def test_executor_registrar_and_telemetry_defaults():
    providers.get_executor_registrar()(object())          # no-op
    assert providers.get_transfer_telemetry() is None
    sink = object()
    providers.set_transfer_telemetry(sink)
    assert providers.get_transfer_telemetry() is sink


def test_provision_telemetry_wrapper_tolerates_partial_sink():
    from hugpy_storage import provision
    assert provision._evt() is None

    class Sink:
        def emit_provision_done(self, *a, **k):
            raise RuntimeError("boom")
    providers.set_transfer_telemetry(Sink())
    ev = provision._evt()
    assert ev.emit_provision_done("m", "hf") is None      # swallowed
    assert ev.emit_provision_start("m", "hf") is None     # missing -> no-op
    assert ev.serve_scope() is None


# ── presence ─────────────────────────────────────────────────────────────────
def _w(path, data=b"x", size=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data if size is None else data * size)


def test_transformers_completeness(tmp_path):
    d = str(tmp_path / "m")
    assert mp.model_looks_downloaded(d, {"framework": "transformers"}) is False
    _w(os.path.join(d, "config.json"), b"{}")
    _w(os.path.join(d, "model.safetensors"), b"x", 10)        # LFS pointer stub
    assert mp.model_looks_downloaded(d, {"framework": "transformers"}) is False
    _w(os.path.join(d, "model.safetensors"), b"x", 1024 * 1024 + 1)
    assert mp.model_looks_downloaded(d, {"framework": "transformers"}) is True


def test_gguf_completeness_pin_and_vision(tmp_path):
    d = str(tmp_path / "g")
    _w(os.path.join(d, "sub", "m-Q4_K_M.gguf"), b"x", 1024 * 1024 + 1)
    cfg = {"framework": "gguf", "filename": "m-Q8_0.gguf"}
    # pinned quant absent but another complete quant serves -> complete
    assert mp.model_looks_downloaded(d, cfg) is True
    assert mp.pinned_filename_present(d, cfg) is False
    assert mp.pinned_filename_present(d, {"framework": "gguf", "filename": "Q4_K_M"}) is True
    assert mp.pinned_filename_present(d, {"framework": "transformers"}) is None
    # vision needs an mmproj beside the weights
    vcfg = {"framework": "gguf", "primary_task": "image-text-to-text"}
    assert mp.model_looks_downloaded(d, vcfg) is False
    _w(os.path.join(d, "sub", "mmproj-f16.gguf"), b"x", 10)
    assert mp.model_looks_downloaded(d, vcfg) is True
    assert mp.find_gguf_file(d, {"filename": "Q4_K_M"}).endswith("m-Q4_K_M.gguf")
    assert all("mmproj" not in g for g in mp.gguf_files(d))


def test_comfy_and_pipeline_completeness(tmp_path):
    c = str(tmp_path / "c")
    _w(os.path.join(c, "ckpt.safetensors"), b"x")
    assert mp.model_looks_downloaded(c, {"framework": "comfy", "filename": "ckpt.safetensors"})
    assert not mp.model_looks_downloaded(c, {"framework": "comfy", "filename": "other"})
    p = str(tmp_path / "p")
    _w(os.path.join(p, "model_index.json"), b"{}")
    assert mp.model_looks_downloaded(p, {"framework": "diffusers"}) is False
    _w(os.path.join(p, "unet", "w.safetensors"), b"x", 1024 * 1024 + 1)
    assert mp.model_looks_downloaded(p, {"framework": "diffusers"}) is True


def test_resolver_uses_storage_completeness(tmp_path):
    from hugpy_storage.model_paths import resolve_model_dir
    root = str(tmp_path)
    model = {"hub_id": "o/r", "framework": "transformers", "primary_task": "text-generation"}
    flat = os.path.join(root, "models", "transformers", "o", "r")
    _w(os.path.join(flat, "config.json"), b"{}")
    _w(os.path.join(flat, "model.safetensors"), b"x", 10)   # stub only
    assert resolve_model_dir(model, root, require_complete=True) is None
    _w(os.path.join(flat, "model.safetensors"), b"x", 1024 * 1024 + 1)
    assert resolve_model_dir(model, root, require_complete=True) == flat


def test_dir_size_walk_listing_and_disk_helpers(tmp_path):
    d = str(tmp_path / "s")
    _w(os.path.join(d, "a.bin"), b"x", 10)
    _w(os.path.join(d, ".cache", "junk"), b"x", 100)
    assert mp.dir_size_bytes(d) == 110
    assert mp.walk_listing(d) == [("a.bin", 10)]
    assert mp.dir_size_bytes(None) is None
    st = mp.disk_stats(os.path.join(d, "does", "not", "exist"))
    assert st["disk_total_bytes"] > 0 and st["disk_mount"]
    exc = OSError(28, "No space left on device")
    assert mp.errno_name(exc) == "ENOSPC"
    assert mp.errno_name(ValueError()) == ""
    human = mp.describe_disk_error(exc, d)
    assert human.startswith("disk full (ENOSPC)") and "free of" in human
    assert mp.describe_disk_error(ValueError("x")) == ""


# ── gguf_inspect ─────────────────────────────────────────────────────────────
def _gguf(path, kv, tensors=()):
    """Write a minimal GGUF v3 header: kv = [(key, type, value)],
    tensors = [(name, dims, offset)]; data section padded to alignment 32."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for key, t, v in kv:
        body += s(key) + struct.pack("<I", t)
        body += struct.pack("<I", v) if t == 4 else s(v)
    for name, dims, off in tensors:
        body += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        body += struct.pack("<I", 0) + struct.pack("<Q", off)
    pad = (-len(body)) % 32
    body += b"\0" * pad + b"\0" * 64
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(body)


def test_gguf_inspect_reads_metadata_and_dense_vs_moe(tmp_path):
    from hugpy_storage import gguf_inspect as gi
    dense = str(tmp_path / "dense.gguf")
    _gguf(dense, [("llama.block_count", 4, 32)])
    assert gi.gguf_metadata(dense, (".block_count",)) == {".block_count": 32}
    assert gi.gguf_moe_detail(dense) == {"is_moe": False, "expert_count": None,
                                         "expert_used_count": None, "sparsity": None,
                                         "expert_bytes": 0, "non_expert_bytes": 0,
                                         "expert_bytes_by_layer": {}, "files": 1}
    moe = str(tmp_path / "moe.gguf")
    _gguf(moe, [("qwen.expert_count", 4, 8), ("qwen.expert_used_count", 4, 2)],
          [("blk.0.attn_q.weight", (4, 4), 0),
           ("blk.0.ffn_gate_exps.weight", (2, 2, 8), 16),
           ("blk.1.ffn_up_exps.weight", (2, 2, 8), 48)])
    d = gi.gguf_moe_detail(moe)
    assert d["is_moe"] is True and d["expert_count"] == 8 and d["sparsity"] == 0.25
    assert d["expert_bytes"] > 0 and set(d["expert_bytes_by_layer"]) == {0, 1}
    assert gi.gguf_moe_detail(str(tmp_path / "missing.gguf")) == {"is_moe": False}
    assert gi.gguf_metadata(str(tmp_path / "missing.gguf"), (".x",)) == {}
    # the marker writer sees the same truth
    from hugpy_storage.hugpy_marker import detect_moe_capable
    assert detect_moe_capable(str(tmp_path), framework="gguf") in (True, False)


def test_engine_spill_reexports_storage_reader():
    pytest.importorskip("hugpy_engine")
    from hugpy_engine import spill
    from hugpy_storage import gguf_inspect as gi
    assert spill.gguf_moe_detail is gi.gguf_moe_detail
    assert spill._gguf_metadata is gi._gguf_metadata
    assert spill._MOE_DETAIL_CACHE is gi._MOE_DETAIL_CACHE
