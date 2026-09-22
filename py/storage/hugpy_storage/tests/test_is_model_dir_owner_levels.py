"""is_model_dir's owner-level exemption across BOTH store layouts.

Flat (2026-07-11): models/<runtime>/<owner>/<repo>  — owner at depth 2.
Legacy task:       models/<runtime>/<task>/<owner>/<repo> — owner at depth 3.

An owner dir whose child repo keeps its ggufs at repo top level matches the
quant-subfolder probe ("a subdir holds .gguf"), so without the exemption the
org is swallowed into one phantom model (2026-09-10) or reported as a
stale-dir orphan (the legacy shape). A flat-layout REPO at depth 3 that holds
quant SUBFOLDERS must still be a leaf (2026-09-02).
"""
import os

import pytest

from hugpy_storage import model_paths as mp


def _gguf(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF" + b"\0" * 64)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(mp, "get_model_home", lambda *a, **k: root)
    return root


def test_flat_owner_at_depth_2_is_not_a_model(home):
    repo = os.path.join(home, "gguf", "ponpoke", "flux2-klein-GGUF")
    _gguf(os.path.join(repo, "flux2-Q4_K_M.gguf"))
    assert mp.is_model_dir(os.path.join(home, "gguf", "ponpoke")) is False
    assert mp.is_model_dir(repo) is True


def test_legacy_owner_at_depth_3_is_not_a_model(home):
    repo = os.path.join(home, "gguf", "text-generation", "Qwen", "Qwen3-Coder-Next-GGUF")
    _gguf(os.path.join(repo, "model.gguf"))
    assert mp.is_model_dir(os.path.join(home, "gguf", "text-generation")) is False
    assert mp.is_model_dir(os.path.join(home, "gguf", "text-generation", "Qwen")) is False
    assert mp.is_model_dir(repo) is True
    assert mp.get_model_dirs(home) == [repo]


def test_flat_repo_with_quant_subfolders_at_depth_3_is_a_leaf(home):
    # gguf/<owner>/<repo>/<quant>/*.gguf: the repo (depth 3) is the model.
    repo = os.path.join(home, "gguf", "Qwen", "Qwen3-Coder-Next-GGUF")
    _gguf(os.path.join(repo, "Qwen3-Coder-Next-Q4_K_M", "shard-00001-of-00002.gguf"))
    assert mp.is_model_dir(repo) is True
    assert mp.get_model_dirs(home) == [repo]
    # a hyphenated OWNER is not mistaken for a task segment
    repo2 = os.path.join(home, "gguf", "black-forest-labs", "FLUX.1-dev-GGUF")
    _gguf(os.path.join(repo2, "q8", "flux1-dev-Q8_0.gguf"))
    assert mp.is_model_dir(repo2) is True


@pytest.mark.parametrize("name,expected", [
    ("text-generation", True), ("image-text-to-text", True),
    ("automatic-speech-recognition", True), ("misc", True), ("dataset", True),
    ("video-to-audio", True),                       # unlisted tag, task word
    ("black-forest-labs", False), ("Qwen", False), ("TheBloke", False),
    ("stable-diffusion", False), ("", False), ("Text-Generation", False),
])
def test_looks_like_task_dir(name, expected):
    assert mp._looks_like_task_dir(name) is expected
