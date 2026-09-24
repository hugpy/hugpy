"""hugpy-model-manifest-backfill: dry-run plans without touching the Hub or
the files; --apply writes hub-sourced or local manifests into hugpy.json."""
from __future__ import annotations

import json
import os

from hugpy_ops import model_manifest_backfill as bf


def _model(root, name, hub_id, files, framework="gguf", manifest=None):
    d = root / name
    d.mkdir(parents=True)
    for rel, data in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    marker = {"hub_id": hub_id, "framework": framework}
    if manifest is not None:
        marker["manifest"] = manifest
    (d / "hugpy.json").write_text(json.dumps(marker))
    return {"model_key": name, "hub_id": hub_id, "framework": framework, "destination": str(d)}


def _rows(tmp_path):
    return [
        _model(tmp_path, "real", "owner/real-GGUF", {"real-Q4.gguf": b"GGUF" + b"\1" * 60}),
        _model(tmp_path, "ckpt", "comfy/ckpt", {"ckpt.safetensors": b"\0" * 32}, framework="comfy"),
        _model(tmp_path, "done", "owner/done", {"w.gguf": b"GGUF"},
               manifest={"files": [{"path": "w.gguf", "bytes": 4, "sha256": None}]}),
        {"model_key": "gone", "hub_id": "owner/gone", "destination": str(tmp_path / "nope")},
    ]


def test_placeholder_ids_are_not_hub_repos():
    assert bf.is_hub_repo_id("owner/repo-GGUF")
    assert not bf.is_hub_repo_id("comfy/dreamshaper")
    assert not bf.is_hub_repo_id("owner/repo", framework="comfy")
    assert not bf.is_hub_repo_id("owner/repo", marker_source="custom")
    assert not bf.is_hub_repo_id("just-a-name")
    assert not bf.is_hub_repo_id(None)


def test_dry_run_plans_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "hub_listing", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hub called")))
    rows = _rows(tmp_path)
    before = {r["destination"]: open(os.path.join(r["destination"], "hugpy.json")).read()
              for r in rows if os.path.isdir(r["destination"])}
    res = bf.run(rows, apply=False)
    s = res["summary"]
    assert s["mode"] == "dry-run" and s["would_backfill"] == 2
    assert s["from_hub"] == 1 and s["from_local"] == 1
    # The model that already has a manifest is either sized in this pass (no
    # size stamped yet) or skipped as "manifest + size present"; never blank.
    assert s["skipped"].get("destination absent") == 1
    assert s.get("would_size", 0) + s["skipped"].get("manifest + size present", 0) == 1
    after = {d: open(os.path.join(d, "hugpy.json")).read() for d in before}
    assert before == after


def test_apply_hub_and_local(tmp_path, monkeypatch):
    rows = _rows(tmp_path)
    sha = "a" * 64

    def fake_listing(hub_id, revision=None, token=None, timeout=None):
        assert hub_id == "owner/real-GGUF"
        return {"status": 200, "revision": "rev1",
                "files": {"real-Q4.gguf": {"bytes": 64, "sha256": sha},
                          "other-Q8.gguf": {"bytes": 999, "sha256": "b" * 64}}}

    monkeypatch.setattr(bf, "hub_listing", fake_listing)
    res = bf.run(rows, apply=True)
    assert res["summary"]["written_source"] == {"huggingface": 1, "local": 1}
    real = json.load(open(tmp_path / "real" / "hugpy.json"))
    m = real["manifest"]
    assert m["source"] == "huggingface" and m["revision"] == "rev1"
    # only the file actually on disk — never the whole repo listing
    assert m["files"] == [{"path": "real-Q4.gguf", "bytes": 64, "sha256": sha}]
    assert real["hub_id"] == "owner/real-GGUF"               # identity kept
    ck = json.load(open(tmp_path / "ckpt" / "hugpy.json"))["manifest"]
    assert ck["source"] == "local" and ck["revision"] is None
    assert ck["files"] == [{"path": "ckpt.safetensors", "bytes": 32, "sha256": None}]
    # idempotent: a second run finds nothing to do
    assert bf.run(rows, apply=True)["summary"]["backfilled"] == 0


def test_hub_404_falls_back_to_local(tmp_path, monkeypatch):
    rows = [_model(tmp_path, "x", "owner/x", {"x.gguf": b"GGUF1234"})]
    monkeypatch.setattr(bf, "hub_listing", lambda *a, **k: {"status": 404, "files": {}})
    bf.run(rows, apply=True)
    m = json.load(open(tmp_path / "x" / "hugpy.json"))["manifest"]
    assert m["source"] == "local" and m["backfill"]["hub_status"] == 404
    assert m["files"] == [{"path": "x.gguf", "bytes": 8, "sha256": None}]
