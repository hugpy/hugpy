"""Env-profiles stage 3: the transformers slot argv and the runner routing."""
import os

import pytest

from hugpy_engine.serve import slot_agent as S


def test_build_tf_cmd_uses_profile_python(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    pbin = tmp_path / "envs" / "ct" / "bin"
    pbin.mkdir(parents=True)
    (pbin / "python").write_text("#!fake\n")
    out = S._build_tf_cmd("M", n_gpu_layers=None, ctx=4096, path=str(model), profile_bin=str(pbin),
                          tf_opts={"bnb4": True, "trust_remote_code": True,
                                   "max_memory": {"0": "20GiB", "cpu": "60GiB"}})
    argv = out[0]
    assert argv[0] == str(pbin / "python") and argv[1].endswith("tf_child.py")
    for flag in ("--model-dir", "--ctx", "--bnb4", "--trust-remote-code", "--max-memory"):
        assert flag in argv
    assert argv[argv.index("--device") + 1] == "auto"
    cpu = S._build_tf_cmd("M", n_gpu_layers=0, ctx=None, path=str(model), profile_bin=str(pbin))[0]
    assert cpu[cpu.index("--device") + 1] == "cpu" and "--max-memory" not in cpu


def test_build_tf_cmd_refuses_without_a_dir_or_ready_venv(tmp_path):
    with pytest.raises(FileNotFoundError):
        S._build_tf_cmd("M", path=str(tmp_path / "nope"))
    model = tmp_path / "m"
    model.mkdir()
    with pytest.raises(RuntimeError):     # profile bin without its python: never the shared venv
        S._build_tf_cmd("M", path=str(model), profile_bin=str(tmp_path / "missing" / "bin"))


def _runner(monkeypatch, prof):
    from hugpy_engine.generate import generate_runner as G
    from hugpy_engine.serve import profiles
    monkeypatch.setattr(profiles, "resolve_model", lambda mk: prof)
    r = G.DeepCoderChatRunner.__new__(G.DeepCoderChatRunner)
    r.model_key = "M"

    class Cfg:
        model_dir, adapter_dir, trust_remote_code, use_quantization, device = "/m", None, False, False, "cuda"
    r._cfg = Cfg()
    return r


def test_routing_no_profile_ready_and_not_ready(monkeypatch):
    from hugpy_engine.llama.runners.get import LocalEngineUnavailable
    assert _runner(monkeypatch, None)._profile_delegate() is None
    with pytest.raises(LocalEngineUnavailable) as ei:
        _runner(monkeypatch, {"name": "ct", "state": "materializing", "bin": None})._profile_delegate()
    assert "ct" in str(ei.value) and "materializing" in str(ei.value)
    d = _runner(monkeypatch, {"name": "ct", "state": "ready", "bin": "/envs/ct/bin"})._profile_delegate()
    from hugpy_engine.llama.runners.chat_runner import ProfileChildChatRunner
    assert isinstance(d, ProfileChildChatRunner)
