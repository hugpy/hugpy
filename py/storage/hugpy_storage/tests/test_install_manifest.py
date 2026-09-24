"""Install manifest (hugpy.json ``manifest``): captured ONCE at download time
from what the download chose + the HF download metadata already on disk (no
Hub call, no hashing), preserved across re-stamps, written atomically, and
read back by the worker-side presence checks."""
from __future__ import annotations

import json
import os

import pytest

from hugpy_storage import hugpy_marker as hm

SHA = "7843eaa90bb1f37aaf92bcf6b13c595a67c71c81a5681829539b50028aeba4bc"
GIT_SHA1 = "c8da8a824911cf3e31d6a272130e547ffe305221"
REV = "0123456789abcdef0123456789abcdef01234567"


def _hf_file(local_dir, rel, data, etag, commit=REV):
    """What huggingface_hub leaves for one local-dir download."""
    p = os.path.join(local_dir, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)
    meta = os.path.join(local_dir, ".cache", "huggingface", "download", rel + ".metadata")
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    with open(meta, "w") as fh:
        fh.write(f"{commit}\n{etag}\n{os.path.getmtime(p) + 0.5}\n")


def test_build_manifest_from_hf_download_metadata(tmp_path):
    d = str(tmp_path)
    _hf_file(d, "m-Q4_K_M.gguf", b"GGUF" + b"\1" * 100, SHA)
    _hf_file(d, "mmproj/m-f16.gguf", b"GGUF" + b"\2" * 10, SHA.replace("7", "8"))
    _hf_file(d, "config.json", b"{}", GIT_SHA1)          # non-LFS: git sha1, not recorded
    (tmp_path / "hugpy.json").write_text("{}")
    (tmp_path / "x.gguf.part").write_bytes(b"partial")
    m = hm.build_install_manifest(d)
    assert m["revision"] == REV and m["source"] == "huggingface" and m["captured_at"]
    by = {e["path"]: e for e in m["files"]}
    assert set(by) == {"m-Q4_K_M.gguf", "mmproj/m-f16.gguf", "config.json"}   # no marker/.cache/.part
    assert by["m-Q4_K_M.gguf"] == {"path": "m-Q4_K_M.gguf", "bytes": 104, "sha256": SHA}
    assert by["config.json"]["sha256"] is None


def test_stale_metadata_sha_is_not_trusted(tmp_path):
    d = str(tmp_path)
    _hf_file(d, "w.gguf", b"GGUF", SHA)
    later = os.path.getmtime(tmp_path / "w.gguf") + 100
    os.utime(tmp_path / "w.gguf", (later, later))       # file rewritten after the download
    (e,) = hm.build_install_manifest(d)["files"]
    assert e["sha256"] is None and e["bytes"] == 4


def test_write_marker_with_manifest_keeps_quants_and_is_atomic(tmp_path, monkeypatch):
    import hugpy_platform.atomic_json as aj
    calls = []
    real = aj.save_json
    monkeypatch.setattr(aj, "save_json", lambda p, d: (calls.append(p), real(p, d)))
    d = str(tmp_path)
    _hf_file(d, "m-Q8_0.gguf", b"GGUF" + b"\0" * 20, SHA)
    man = hm.build_install_manifest(d)
    hm.write_hugpy_marker(d, hub_id="o/m", framework="gguf", manifest=man)
    assert calls == [os.path.join(d, "hugpy.json")]
    blob = json.load(open(tmp_path / "hugpy.json"))
    assert blob["manifest"] == man
    assert blob["quants"] == [{"file": "m-Q8_0.gguf", "quant": "q8_0", "bytes": 24, "shards": 1}]
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_restamp_without_manifest_preserves_it(tmp_path):
    d = str(tmp_path)
    (tmp_path / "w.safetensors").write_bytes(b"x" * 8)
    man = hm.build_install_manifest(d, source="local")
    hm.write_hugpy_marker(d, hub_id="o/w", framework="transformers", manifest=man)
    hm.write_hugpy_marker(d, hub_id="o/w", framework="transformers", tasks=["text-generation"])
    blob = json.load(open(tmp_path / "hugpy.json"))
    assert blob["manifest"] == man and blob["tasks"] == ["text-generation"]
    hm.sync_marker_quants(d)
    assert json.load(open(tmp_path / "hugpy.json"))["manifest"] == man


def test_merge_keeps_prior_entries_for_untouched_files():
    prior = {"revision": "r1", "files": [{"path": "a.gguf", "bytes": 1, "sha256": None},
                                         {"path": "b.gguf", "bytes": 2, "sha256": None}]}
    new = {"revision": "r2", "captured_at": "t", "source": "huggingface",
           "files": [{"path": "b.gguf", "bytes": 3, "sha256": SHA}]}
    out = hm.merge_manifests(prior, new)
    assert out["revision"] == "r2"
    assert out["files"] == [{"path": "a.gguf", "bytes": 1, "sha256": None, "revision": "r1"},
                            {"path": "b.gguf", "bytes": 3, "sha256": SHA}]


class _FakeHF:
    """hf_hub_download / snapshot_download that behave like huggingface_hub's
    local-dir mode: file + .cache metadata, no network."""

    def __init__(self, files):
        self.files = files

    def hf_hub_download(self, repo_id, filename, local_dir, subfolder=None, **kw):
        _hf_file(local_dir, filename, self.files[filename], SHA)
        return os.path.join(local_dir, filename)

    def snapshot_download(self, repo_id, local_dir, allow_patterns=None, **kw):
        return local_dir                                  # (no sidecars in this repo)


def _patch_download(monkeypatch, tmp_path, files):
    from hugpy_storage import download_models as dm
    monkeypatch.setattr(dm, "_hf", lambda: _FakeHF(files))
    monkeypatch.setattr(dm, "publish_catalog_changed", lambda *a, **k: None)
    monkeypatch.setattr(dm, "resolve_model_dir", lambda *a, **k: None)
    dest = tmp_path / "models" / "gguf" / "o" / "m-GGUF"
    monkeypatch.setattr(dm, "flat_destination", lambda model, root=None: str(dest))
    return dm, dest


def test_download_one_writes_manifest_of_chosen_files(tmp_path, monkeypatch):
    dm, dest = _patch_download(monkeypatch, tmp_path, {"m-Q4_K_M.gguf": b"GGUF" + b"\1" * 40})
    model = {"hub_id": "o/m-GGUF", "framework": "gguf", "filename": "m-Q4_K_M.gguf",
             "primary_task": "text-generation"}
    dm.download_one(model, root=str(tmp_path), model_key="m-GGUF")
    blob = json.load(open(dest / "hugpy.json"))
    assert blob["hub_id"] == "o/m-GGUF" and blob["source"] == "download"
    assert blob["manifest"]["files"] == [{"path": "m-Q4_K_M.gguf", "bytes": 44, "sha256": SHA}]
    assert blob["manifest"]["revision"] == REV and blob["manifest"]["source"] == "huggingface"
    assert blob["quants"][0]["file"] == "m-Q4_K_M.gguf"


def test_download_into_existing_dir_merges_manifest(tmp_path, monkeypatch):
    dm, dest = _patch_download(monkeypatch, tmp_path, {"m-Q8_0.gguf": b"GGUF" + b"\2" * 60})
    os.makedirs(dest)
    (dest / "m-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\1" * 40)
    hm.write_hugpy_marker(str(dest), hub_id="o/m-GGUF", framework="gguf",
                          manifest=hm.build_install_manifest(str(dest), source="huggingface"))
    monkeypatch.setattr(dm, "resolve_model_dir", lambda *a, **k: str(dest))
    model = {"hub_id": "o/m-GGUF", "framework": "gguf", "filename": "m-Q8_0.gguf"}
    dm.download_one(model, root=str(tmp_path), model_key="m-GGUF")
    files = json.load(open(dest / "hugpy.json"))["manifest"]["files"]
    assert [(e["path"], e["bytes"]) for e in files] == [("m-Q4_K_M.gguf", 44), ("m-Q8_0.gguf", 64)]


# ── worker side: the manifest travels in hugpy.json with the transfer ────────

def test_worker_check_against_install_manifest(tmp_path):
    from hugpy_storage import provision as pv
    d = str(tmp_path)
    (tmp_path / "m-Q4.gguf").write_bytes(b"GGUF" + b"\1" * 12)
    (tmp_path / "config.json").write_text('{"edited": "after install"}')
    man = {"revision": REV, "captured_at": "t", "source": "huggingface",
           "files": [{"path": "m-Q4.gguf", "bytes": 16, "sha256": None},
                     {"path": "m-Q8.gguf", "bytes": 99, "sha256": None},     # other quant: not on worker
                     {"path": "config.json", "bytes": 2, "sha256": None}]}
    (tmp_path / "hugpy.json").write_text(json.dumps({"hub_id": "o/m", "manifest": man}))
    transfer = [{"path": "m-Q4.gguf", "size": 16}, {"path": "config.json", "size": 27},
                {"path": "hugpy.json", "size": 1}]
    pv._check_install_manifest("m", d, transfer)          # in-scope weights match
    assert pv.local_copy_matches_manifest(d)
    (tmp_path / "m-Q4.gguf").write_bytes(b"GGUF" + b"\1" * 4)   # central served a short copy
    with pytest.raises(pv.CentralCopyBroken, match="m-Q4.gguf is 8 bytes, manifest says 16"):
        pv._check_install_manifest("m", d, transfer)
    assert not pv.local_copy_matches_manifest(d)


def test_worker_check_without_manifest_is_a_noop(tmp_path):
    from hugpy_storage import provision as pv
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    (tmp_path / "hugpy.json").write_text(json.dumps({"hub_id": "o/m"}))
    pv._check_install_manifest("m", str(tmp_path), [{"path": "m.gguf", "size": 4}])
    assert pv.local_copy_matches_manifest(str(tmp_path))
