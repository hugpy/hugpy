"""Behavioural checks for hugpy_platform.utils (the bits with logic)."""
from __future__ import annotations

import base64

import pytest

from hugpy_platform import utils


def test_slugify():
    assert utils.slugify("  My Clip / v2.mp4 ") == "My_Clip___v2.mp4"   # "/" -> "_", runs of spaces -> "_"
    assert utils.slugify("///") == "media"
    assert utils.slugify("", fallback="x") == "x"


def test_unique_path(tmp_path):
    p = tmp_path / "a.txt"
    assert utils.unique_path(str(p)) == str(p)
    p.write_text("x")
    assert utils.unique_path(str(p)) == str(tmp_path / "a_1.txt")


def test_mmproj_detection(tmp_path):
    model = tmp_path / "Qwen-VL"
    (model / "shards").mkdir(parents=True)
    (model / "model-Q4_K_M.gguf").write_bytes(b"0")
    (model / "shards" / "mmproj-f16.gguf").write_bytes(b"0")
    assert utils.is_mmproj_file("mmproj-f16.gguf")
    assert not utils.is_mmproj_file("model-Q4_K_M.gguf")
    assert utils.find_mmproj(str(model)) == str(model / "shards" / "mmproj-f16.gguf")
    assert utils.find_mmproj(str(model / "model-Q4_K_M.gguf")).endswith("mmproj-f16.gguf")
    # a missing FILE falls back to its parent dir; a missing TREE yields None
    assert utils.find_mmproj(str(tmp_path / "nowhere" / "m.gguf")) is None
    ggufs = utils.get_guffs_in_dir(str(model))
    assert len(ggufs) == 2
    assert utils.extract_gguf_filename(ggufs, str(model)) == "model-Q4_K_M.gguf"


def test_infer_framework(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(b"0")
    assert utils.infer_framework(str(tmp_path)) == "transformers"
    (tmp_path / "m.gguf").write_bytes(b"0")
    assert utils.infer_framework(str(tmp_path)) == "gguf"
    assert utils.infer_framework(str(tmp_path / "nope")) is None


def test_base64_image(tmp_path):
    f = tmp_path / "img.bin"
    f.write_bytes(b"\x89PNG")
    assert utils.get_base_64_image(str(f)) == base64.b64encode(b"\x89PNG").decode("ascii")


def test_messages_helpers():
    assert utils.get_messages("hi") == [{"role": "user", "content": "hi"}]
    assert utils.message_to_dict({"role": "system", "content": 1}) == {"role": "system", "content": "1"}

    class M:
        role, content = "assistant", "ok"

    assert utils.messages_to_dicts([M()]) == [{"role": "assistant", "content": "ok"}]


def test_merge_disk_over_registry_uses_disk_authoritative_fields():
    merged, prov = utils.merge_disk_over_registry(
        {"name": "on-disk", "filename": "a.gguf", "task": None, "port": 1},
        {"name": "registry", "task": "chat", "port": 2},
    )
    assert merged["name"] == "on-disk" and prov["name"] == "disk"        # DISK_AUTHORITATIVE
    assert merged["filename"] == "a.gguf" and prov["filename"] == "disk"
    assert merged["task"] == "chat" and prov["task"] == "registry"        # registry fills gaps
    assert merged["port"] == 2 and prov["port"] == "registry"             # not disk-authoritative


def test_require_file(tmp_path):
    f = tmp_path / "x"
    with pytest.raises(FileNotFoundError):
        utils.require_file(str(f))
    f.write_text("1")
    assert utils.require_file(str(f)) == str(f)
