"""hugpy-model-audit: synthetic GGUF / safetensors files in tmp_path exercise
every verdict path offline (no central, no fleet), plus the worker
load-error classifier. The GGUF writer here emits a real v3 file: KV table,
tensor-info table, aligned F32 data section."""
from __future__ import annotations

import json
import os
import struct

import pytest

from hugpy_ops import model_audit as ma


# ── tiny GGUF writer ─────────────────────────────────────────────────────────

def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_u32(key, val):
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_str(key, val):
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_str_array(key, vals):
    return (_s(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(vals))
            + b"".join(_s(v) for v in vals))


def write_gguf(path, *, arch="llama", n_embd=8, vocab=4, embd_dims=None, truncate=0, magic=b"GGUF"):
    """One F32 ``token_embd.weight`` whose stored dims are ``embd_dims``
    (default ``[n_embd, vocab]`` — consistent with the metadata)."""
    dims = list(embd_dims or [n_embd, vocab])
    kvs = [_kv_str("general.architecture", arch), _kv_u32("general.alignment", 32),
           _kv_u32(f"{arch}.embedding_length", n_embd),
           _kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab)])]
    tinfo = (_s("token_embd.weight") + struct.pack("<I", len(dims))
             + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<I", 0) + struct.pack("<Q", 0))
    head = magic + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kvs)) + b"".join(kvs) + tinfo
    pad = (-len(head)) % 32
    n = 1
    for d in dims:
        n *= d
    data = b"\x01" * (n * 4)
    blob = head + b"\x00" * pad + data
    if truncate:
        blob = blob[:-truncate]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def write_safetensors(path, n_floats=4, truncate=0, zero_head=False):
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [n_floats], "data_offsets": [0, n_floats * 4]}}).encode()
    blob = struct.pack("<Q", len(hdr)) + hdr + b"\x01" * (n_floats * 4)
    if truncate:
        blob = blob[:-truncate]
    if zero_head:
        blob = b"\x00" * len(blob)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def ctx(overrides=None, workers=None):
    return ma.Context("http://central.invalid", workers or [], overrides or {}, "/dev/null/serve_overrides.json",
                      {}, set())


def row(key, dest, fw="gguf", eff=None, task="text-generation"):
    return {"model_key": key, "framework": fw, "destination": str(dest), "hub_id": f"owner/{key}",
            "primary_task": task, "effective_gguf": eff, "workers": [], "hot_workers": []}


def audit(r, c=None):
    c = c or ctx()
    return ma.finalize(ma.audit_model(r, c), r, c)


# ── GGUF verdicts ────────────────────────────────────────────────────────────

def test_valid_gguf_is_static_ok(tmp_path):
    write_gguf(tmp_path / "m" / "m-Q8_0.gguf")
    rep = audit(row("m", tmp_path / "m", eff="m-Q8_0.gguf"))
    assert rep.verdict == ma.STATIC_OK and not rep.eliminate
    assert any("magic=GGUF version=3 arch=llama" in line for line in rep.log)
    assert "static checks passed; never loaded on any worker" in rep.log


def test_truncated_gguf_is_broken_download_not_faulty(tmp_path):
    write_gguf(tmp_path / "m" / "m.gguf", truncate=20)
    rep = audit(row("m", tmp_path / "m", eff="m.gguf"))
    assert rep.verdict == ma.BROKEN
    assert "truncated by 20" in rep.why
    assert not any(f.verdict == ma.FAULTY for f in rep.findings)


def test_inconsistent_token_embd_is_faulty_and_eliminated(tmp_path):
    # the Echo-Mini shape: metadata says 4096 x 32005, the tensor is 384 x 32000
    write_gguf(tmp_path / "e" / "e.gguf", n_embd=16, vocab=6, embd_dims=[4, 5])
    rep = audit(row("e", tmp_path / "e", eff="e.gguf"))
    assert rep.verdict == ma.FAULTY and rep.eliminate
    assert "tensor 'token_embd.weight' has wrong shape; expected 16, 6, got 4, 5, 1, 1" in rep.why


def test_bad_magic_is_faulty(tmp_path):
    write_gguf(tmp_path / "b" / "b.gguf", magic=b"GGUX")
    rep = audit(row("b", tmp_path / "b", eff="b.gguf"))
    assert rep.verdict == ma.FAULTY and "bad magic" in rep.why


def test_zero_filled_gguf_is_broken(tmp_path):
    p = tmp_path / "z" / "z.gguf"
    write_gguf(p)
    size = os.path.getsize(p)
    with open(p, "wb") as fh:
        fh.write(b"\x00" * size)
    rep = audit(row("z", tmp_path / "z", eff="z.gguf"))
    assert rep.verdict == ma.BROKEN and "zero" in rep.why


def test_clip_pin_is_misconfigured(tmp_path):
    d = tmp_path / "v"
    write_gguf(d / "v-Q4_K_M.gguf")
    write_gguf(d / "mmproj" / "v-vision-Q6_K.gguf", arch="clip")
    worker = {"name": "w1", "id": "abc", "status": "online", "gpu_total_bytes_known": 8 << 30}
    c = ctx(overrides={"v": {"gguf_file_by_worker": {"abc": "v-vision-Q6_K.gguf"}}}, workers=[worker])
    rep = audit(row("v", d, eff="v-Q4_K_M.gguf"), c)
    assert rep.verdict == ma.MISCONFIG and not rep.eliminate
    assert "vision projector" in rep.why and "arch=clip" in rep.why
    assert "gguf_file_by_worker" in (rep.suggested_fix or "")


def test_projector_only_dir_is_misconfigured(tmp_path):
    write_gguf(tmp_path / "p" / "mmproj" / "p-vision.gguf", arch="clip")
    rep = audit(row("p", tmp_path / "p", eff="p-vision.gguf"))
    assert rep.verdict == ma.MISCONFIG and "every GGUF on central is a vision projector" in rep.why


def test_missing_shard_is_broken(tmp_path):
    write_gguf(tmp_path / "s" / "s-00002-of-00002.gguf")
    rep = audit(row("s", tmp_path / "s", eff="s.gguf"))
    assert rep.verdict == ma.BROKEN and "s-00001-of-00002.gguf" in rep.why


def test_unserved_variant_defect_does_not_decide_verdict(tmp_path):
    d = tmp_path / "u"
    write_gguf(d / "u-Q8_0.gguf")
    write_gguf(d / "u-Q4_0.gguf", truncate=8)
    rep = audit(row("u", d, eff="u-Q8_0.gguf"))
    assert rep.verdict == ma.STATIC_OK
    assert any(f.detail.startswith("unserved variant:") for f in rep.findings)


def test_video_arch_labelled_text_is_mislabeled(tmp_path):
    write_gguf(tmp_path / "w" / "w.gguf", arch="wan")
    rep = audit(row("w", tmp_path / "w", eff="w.gguf"))
    assert rep.verdict == ma.MISLABELED and rep.detected_task == "text-to-video"


def test_not_downloaded(tmp_path):
    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "hugpy.json").write_text("{}")
    assert audit(row("n", tmp_path / "n")).verdict == ma.NOT_DOWNLOADED


# ── safetensors / transformers ───────────────────────────────────────────────

def test_safetensors_checks(tmp_path):
    ok = tmp_path / "a.safetensors"
    write_safetensors(ok)
    assert ma.safetensors_check(str(ok))["kind"] == "ok"
    short = tmp_path / "b.safetensors"
    write_safetensors(short, truncate=3)
    assert ma.safetensors_check(str(short))["kind"] == "truncated"
    zero = tmp_path / "c.safetensors"
    write_safetensors(zero, zero_head=True)
    assert ma.safetensors_check(str(zero))["kind"] == "zeroed"


def test_transformers_index_naming_absent_shard(tmp_path):
    d = tmp_path / "t"
    write_safetensors(d / "model-00001-of-00002.safetensors")
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    rep = audit(row("t", d, fw="transformers"))
    assert rep.verdict == ma.BROKEN and "model-00002-of-00002.safetensors" in rep.why


def test_transformers_unparsable_config_is_faulty(tmp_path):
    d = tmp_path / "c"
    write_safetensors(d / "model.safetensors")
    (d / "config.json").write_text('{"architectures": ["X"], "model_type": ')
    rep = audit(row("c", d, fw="transformers"))
    assert rep.verdict == ma.FAULTY


def test_incomplete_maps_to_its_file(tmp_path):
    name = "model-00001-of-00002.safetensors"
    h = ma.hf_short_hash(name + ".metadata")
    rel = os.path.join(".cache", "huggingface", "download", f"{h}.deadbeef.incomplete")
    assert ma.map_incomplete(rel, [name, "other.safetensors"]) == name


# ── load-error classifier ────────────────────────────────────────────────────

@pytest.mark.parametrize("text, cls", [
    ("llama_model_load: error loading model: error loading model architecture: "
     "unsupported model architecture: 'clip'", ma.MISCONFIG),
    ("RuntimeError: load failed. Loader stderr: llama_model_load: error loading model: check_tensor_dims: "
     "tensor 'token_embd.weight' has wrong shape; expected  4096, 32005, got   384, 32000,     1,     1", ma.FAULTY),
    ("gguf_init_from_file_impl: invalid magic characters 'GGUX'", ma.FAULTY),
    ("LoadRefusal: won't fit on GPU: needs 22.8 GB, 21.1 GB free of 23.6 GB", "fit"),
    ("model needs ~14.2 GB VRAM for n_gpu_layers=-1 but only ~6.8 GB free", "fit"),
    ("WorkerUnreachable: RemoteProtocolError: Server disconnected without sending a response.", "transient"),
    ("not local — probe does not download (lazy doctrine 2026-07-17); files arrive on first real call", "info"),
    ("vision model loaded in-process (text-only — the python binding cannot load the mmproj projector)", "info"),
    ("KeyError: \"Unknown transformers sub-module 'AutoModelForImageTextToText'.\"", "info"),
    ("RuntimeError: x: in-process load failed - ValueError: Failed to load model from file: /m.gguf", "load_failed"),
])
def test_classify_load_error(text, cls):
    assert ma.classify_load_error(text)["class"] == cls


def test_fit_need_bytes_and_stderr_split():
    c = ma.classify_load_error("won't fit on GPU: needs 14.2 GB, 6.8 GB free")
    assert c["need_bytes"] == int(14.2e9)
    head, tail = ma.split_loader_stderr("RuntimeError: boom Loader stderr: line1\nline2")
    assert head == "RuntimeError: boom" and tail == "line1\nline2"
    assert ma.classify_load_error("") is None


def test_worker_fit_refusal_beyond_card_total_is_misconfigured(tmp_path):
    write_gguf(tmp_path / "f" / "f.gguf")
    worker = {"name": "small", "id": "id1", "status": "online", "gpu_total_bytes_known": 8 << 30,
              "load_reports": {"f": {"ok": False, "error": "won't fit on GPU: needs 14.2 GB, 6.8 GB free"}}}
    rep = audit(row("f", tmp_path / "f", eff="f.gguf"), ctx(workers=[worker]))
    assert rep.verdict == ma.MISCONFIG and rep.evidence["workers"]["small"]["load_error_class"]["class"] == "fit"


def test_loader_signature_on_two_workers_corroborates_faulty(tmp_path):
    write_gguf(tmp_path / "g" / "g.gguf")
    err = "boom Loader stderr: check_tensor_dims: tensor 'token_embd.weight' has wrong shape"
    ws = [{"name": n, "id": n, "status": "online", "gpu_total_bytes_known": 8 << 30,
           "load_reports": {"g": {"ok": False, "error": err}}} for n in ("a", "b")]
    rep = audit(row("g", tmp_path / "g", eff="g.gguf"), ctx(workers=ws))
    assert rep.verdict == ma.FAULTY and "identical on 2 workers" in rep.why
    assert rep.evidence["workers"]["a"]["loader_stderr"].startswith("check_tensor_dims")


# ── install manifest (hugpy.json) — the only source of expected values ───────

def _manifest(dest, files, **kw):
    """Write hugpy.json with an install manifest ``files`` = [(rel, bytes, sha)]."""
    blob = {"hub_id": "owner/x", "framework": "gguf",
            "manifest": {"revision": "abc123", "captured_at": "2026-09-23T00:00:00+00:00",
                         "source": "huggingface",
                         "files": [{"path": r, "bytes": n, "sha256": h} for r, n, h in files]}, **kw}
    os.makedirs(dest, exist_ok=True)
    (dest / "hugpy.json").write_text(json.dumps(blob))


def test_manifest_match_is_static_ok(tmp_path):
    d = tmp_path / "m"
    n = write_gguf(d / "m-Q8_0.gguf")
    _manifest(d, [("m-Q8_0.gguf", n, None)])
    rep = audit(row("m", d, eff="m-Q8_0.gguf"))
    assert rep.verdict == ma.STATIC_OK and not rep.eliminate
    assert rep.evidence["manifest"]["revision"] == "abc123"
    assert any("== install manifest" in line for line in rep.log)
    assert not any(f.check == "no_manifest" for f in rep.findings)


def test_manifest_size_mismatch_is_broken_download(tmp_path):
    d = tmp_path / "t"
    n = write_gguf(d / "t.gguf", truncate=16)
    _manifest(d, [("t.gguf", n + 16, None)])
    rep = audit(row("t", d, eff="t.gguf"))
    assert rep.verdict == ma.BROKEN and not rep.eliminate
    assert "install manifest says" in rep.why
    assert "re-provision" in (rep.suggested_fix or "")


def test_manifest_file_missing_on_disk_is_broken(tmp_path):
    d = tmp_path / "g"
    n = write_gguf(d / "g-Q8_0.gguf")
    _manifest(d, [("g-Q8_0.gguf", n, None), ("mmproj/g-f16.gguf", 999, None)])
    rep = audit(row("g", d, eff="g-Q8_0.gguf"))
    assert rep.verdict == ma.BROKEN and "mmproj/g-f16.gguf" in rep.why and not rep.eliminate


def test_complete_file_differing_from_manifest_is_info(tmp_path):
    d = tmp_path / "r"
    n = write_gguf(d / "r.gguf")
    (d / "README.md").write_text("longer than one byte")
    _manifest(d, [("r.gguf", n + 32, None), ("README.md", 1, None)])
    rep = audit(row("r", d, eff="r.gguf"))
    assert rep.verdict == ma.STATIC_OK
    assert any(f.detail.startswith("changed since install") for f in rep.findings)


def test_no_manifest_is_info_and_uses_legacy_quants(tmp_path):
    d = tmp_path / "q"
    n = write_gguf(d / "q-Q4_K_M.gguf", truncate=8)
    (d / "hugpy.json").write_text(json.dumps({"hub_id": "owner/q", "quants": [
        {"file": "q-Q4_K_M.gguf", "quant": "q4_k_m", "bytes": n + 8, "shards": 1}]}))
    rep = audit(row("q", d, eff="q-Q4_K_M.gguf"))
    nm = [f for f in rep.findings if f.check == "no_manifest"]
    assert nm and nm[0].verdict == "info"
    assert rep.verdict == ma.BROKEN and "hugpy.json quants" in rep.why and not rep.eliminate


def test_no_manifest_zero_filled_comfy_is_never_eliminated(tmp_path):
    # the 2026-09-23 audit escalated zero-filled comfy files to eliminate on a
    # Hub 404 for placeholder ids (comfy/<name>): a broken download is re-provision.
    d = tmp_path / "c"
    write_safetensors(d / "c.safetensors", zero_head=True)
    (d / "hugpy.json").write_text(json.dumps({"hub_id": "comfy/c", "framework": "comfy"}))
    r = row("c", d, fw="comfy", task="text-to-image")
    r["hub_id"] = "comfy/c"
    rep = audit(r)
    assert rep.verdict == ma.BROKEN and not rep.eliminate
    assert any(f.check == "no_manifest" and f.verdict == "info" for f in rep.findings)


def test_hash_optin_detects_bit_rot_under_cap(tmp_path):
    import hashlib
    d = tmp_path / "h"
    n = write_gguf(d / "h.gguf")
    good = hashlib.sha256((d / "h.gguf").read_bytes()).hexdigest()
    _manifest(d, [("h.gguf", n, good)])
    r = row("h", d, eff="h.gguf")
    c = ctx()
    c.hash_check = True
    assert audit(r, c).verdict == ma.STATIC_OK
    blob = bytearray((d / "h.gguf").read_bytes())
    blob[-1] ^= 0xFF                                   # same size, one flipped byte
    (d / "h.gguf").write_bytes(bytes(blob))
    assert audit(r).verdict == ma.STATIC_OK           # hashing is off by default
    c2 = ctx()
    c2.hash_check = True
    rep = audit(r, c2)
    assert rep.verdict == ma.BROKEN and "sha256" in rep.why
    c3 = ctx()
    c3.hash_check, c3.hash_max_bytes = True, 10       # over the cap -> skipped
    rep3 = audit(r, c3)
    assert rep3.verdict == ma.STATIC_OK and any("skipped" in line for line in rep3.log)


def test_hub_flag_is_a_hard_error(capsys):
    assert ma.main(["--hub"]) == 2
    assert "not supported" in capsys.readouterr().err


def test_audit_makes_no_hub_call(tmp_path, monkeypatch):
    calls = []
    real = ma._request

    def spy(url, **kw):
        calls.append(url)
        return real(url, **kw)

    monkeypatch.setattr(ma, "_request", spy)
    d = tmp_path / "m"
    n = write_gguf(d / "m.gguf")
    _manifest(d, [("m.gguf", n, None)])
    audit(row("m", d, eff="m.gguf"))
    assert not any("huggingface" in u for u in calls)
    assert not hasattr(ma, "hub_listing")
