"""The REAL reason a GGUF fails to load must survive to central (2026-09-23).

Echo-Mini (token_embd.weight 384x32000 vs metadata 4096x32005): the slot child
rejected it HARD, get_llama_runner logged that and fell back in-process, and
python_runner raised the generic "Failed to load model from file" ``from None``
— the tensor-shape verdict survived only in the worker journal.

Pinned here:
  (a) a HARD slot rejection never reaches the in-process loader and the raised
      error carries key, path, class and the loader stderr verbatim;
  (b) a SOFT refusal still falls back, and the fallback's error is CHAINED and
      names the slot reason;
  (c) python_runner's in-process failure keeps ``__cause__``.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

getmod = importlib.import_module("hugpy_engine.llama.runners.get")
prmod = importlib.import_module("hugpy_engine.llama.runners.src.python_runner")
slots = importlib.import_module("hugpy_engine.serve.slots")
lf = importlib.import_module("hugpy_engine.serve.load_failure")
slot_agent = importlib.import_module("hugpy_engine.serve.slot_agent")

ECHO_STDERR = ("0.00.132.623 E llama_model_load: error loading model: check_tensor_dims: "
               "tensor 'token_embd.weight' has wrong shape; expected 4096, 32005, "
               "got 384, 32000, 1, 1")
CLIP_STDERR = ("0.00.101.004 E llama_model_load: error loading model: error loading "
               "model architecture: unknown model architecture: 'clip'")


def _hard_reply(key, path, stderr):
    """What the slot agent's /load route answers for a child that exited 1."""
    exc = slot_agent.Slot.__new__(slot_agent.Slot)
    exc.last_load_class = "hard_load_failure"
    exc.last_load_stderr = " ".join(stderr.split())
    exc.last_load_stderr_raw = stderr
    exc.model_path = path
    exc.last_load_error = (
        "hard load failure: the llama-server child exited (code 1) after 1.0s "
        "without ever serving — the loader rejected the load, not stalled; "
        f"retrying cannot fix it. Loader stderr: {stderr} Attempt 1, backing off 30s")
    err = exc._load_failure_exc(f"slot 1: {key} {exc.last_load_error}", key)
    return {"error": f"{type(err).__name__}: {err}", "load_failure": err.load_failure}


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "Echo-Mini.Q4_K_M.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 64)
    return str(p)


@pytest.fixture
def env(monkeypatch, gguf):
    """get_llama_runner with every box-dependent seam stubbed; the in-process
    loader is the REAL LlamaCppPythonRunner over a fake llama_cpp whose Llama()
    fails exactly like llama-cpp-python does on a rejected file."""
    calls = {"llama": 0}

    class _Llama:
        def __init__(self, model_path, **kw):
            calls["llama"] += 1
            raise ValueError(f"Failed to load model from file: {model_path}")

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)

    for mod in (getmod, prmod):
        monkeypatch.setattr(mod, "get_model_config", lambda k: types.SimpleNamespace(name=k, tasks=[]))
        monkeypatch.setattr(mod, "ensure_serving_weights", lambda k: str(Path(gguf).parent))
        monkeypatch.setattr(mod, "get_gguf_file", lambda d, c, prefer=None: gguf)
    spill = importlib.import_module("hugpy_engine.spill")
    monkeypatch.setattr(spill, "llama_kwargs", lambda p: {})
    monkeypatch.setattr(spill, "cpu_resident_bytes", lambda p, n: 0)
    monkeypatch.setattr(spill, "gguf_moe_detail", lambda p: {"is_moe": False})
    monkeypatch.setattr(spill, "rpc_servers", lambda: None)
    shard = importlib.import_module("hugpy_engine.llama.runners.src.shard_server")
    monkeypatch.setattr(shard, "ensure_vision_server", lambda k: None)
    monkeypatch.setattr(getmod, "_require_profile_ready", lambda k: None)

    class _NoHttp:
        def __init__(self, *a, **k):
            raise ConnectionError("no llama-server on the default port")
    monkeypatch.setattr(getmod, "LlamaCppRunner", _NoHttp)
    monkeypatch.setattr(slots, "slots_enabled", lambda: True)
    monkeypatch.setattr(getmod, "_REFUSED", {})
    monkeypatch.setattr(getmod, "_LLAMA_INSTANCES", {})
    return calls


def _pool_raising(exc_factory):
    class _Pool:
        def endpoint_for(self, key, opts=None):
            raise exc_factory(key, (opts or {}).get("path"))
    return _Pool


@pytest.mark.parametrize("stderr,needle", [
    (ECHO_STDERR, "check_tensor_dims: tensor 'token_embd.weight' has wrong shape"),
    (CLIP_STDERR, "unknown model architecture: 'clip'"),
])
def test_hard_slot_rejection_never_falls_back(env, monkeypatch, gguf, stderr, needle):
    monkeypatch.setattr(slots, "SlotPool", _pool_raising(
        lambda k, p: lf.from_slot_reply(_hard_reply(k, p, stderr)["error"],
                                        _hard_reply(k, p, stderr), model_key=k, path=p)))
    ctor = []
    monkeypatch.setattr(getmod, "LlamaCppPythonRunner",
                        lambda *a, **k: ctor.append(a) or pytest.fail("in-process loader invoked"))
    with pytest.raises(lf.HardLoadFailure) as ei:
        getmod.get_llama_runner("Echo-Mini")
    err = ei.value
    print("\nHARD:", err)
    assert not ctor and env["llama"] == 0
    msg = str(err)
    assert msg.startswith("Echo-Mini: hard_load_failure")
    assert gguf in msg and needle in msg and "hard load failure" in msg
    lf_rec = err.load_failure
    assert lf_rec["class"] == "hard_load_failure" and lf_rec["loader_stderr"] == stderr and lf_rec["path"] == gguf
    assert lf_rec["model_key"] == "Echo-Mini" and lf_rec["message"] == str(err)
    assert err.__cause__ is not None
    # The refuse-backoff re-raise keeps type, text and the structured verdict.
    with pytest.raises(lf.HardLoadFailure) as again:
        getmod.get_llama_runner("Echo-Mini")
    assert needle in str(again.value)
    assert lf.load_failure_of(again.value)["class"] == "hard_load_failure"


def test_legacy_slot_wording_still_classifies_hard():
    """A slot process still running pre-fix code answers only the text."""
    text = ("RuntimeError: " + _hard_reply("Echo-Mini", "/m/e.gguf", ECHO_STDERR)["error"]
            .split(": ", 1)[1])
    err = lf.from_slot_reply(text, {"error": text}, model_key="Echo-Mini", path="/m/e.gguf")
    assert err.hard and err.path == "/m/e.gguf"
    assert err.loader_stderr == ECHO_STDERR


def test_soft_refusal_falls_back_and_chains(env, monkeypatch, gguf):
    reason = ("slot 1: Echo-Mini needs ~42.0 GB VRAM but only 7.9 GB free — "
              "offload fewer layers or free the card")
    monkeypatch.setattr(slots, "SlotPool", _pool_raising(
        lambda k, p: RuntimeError(f"slot load failed: RuntimeError: {reason}")))
    with pytest.raises(lf.ModelLoadFailure) as ei:
        getmod.get_llama_runner("Echo-Mini")
    err = ei.value
    print("\nSOFT:", err)
    assert env["llama"] == 1                       # the fallback DID run
    assert isinstance(err.__cause__, ValueError)   # chained, not severed
    msg = str(err)
    assert msg.startswith(f"Echo-Mini: in-process load failed - ValueError: "
                          f"Failed to load model from file: {gguf} | slot refusal: ")
    assert reason in msg
    assert err.load_failure["class"] == "vram_fit" and err.load_failure["path"] == gguf


def test_python_runner_keeps_cause(env, gguf):
    with pytest.raises(RuntimeError) as ei:
        prmod.LlamaCppPythonRunner("Echo-Mini")
    assert isinstance(ei.value.__cause__, ValueError)
    assert not ei.value.__suppress_context__ or ei.value.__cause__ is not None
    assert str(ei.value) == (f"Echo-Mini: in-process load failed - ValueError: "
                             f"Failed to load model from file: {gguf}")


def test_stderr_kept_whole():
    big = "x" * 5000 + "check_tensor_dims: wrong shape"
    e = lf.HardLoadFailure("m", loader_stderr=big, path="/p")
    assert lf.LOADER_STDERR_MAX is None                 # logs are never cut
    assert e.loader_stderr == big


def test_load_failure_of_classifies_without_marker():
    assert lf.load_failure_of(RuntimeError("generation blew up")) is None
    assert lf.load_failure_of(RuntimeError("x"), classify=True)["class"] == "other"
    LEU = getmod.LocalEngineUnavailable
    assert lf.load_failure_of(LEU("no engine"))["class"] == "engine_unavailable"
    d = lf.load_failure_of(lf.HardLoadFailure("boom", loader_stderr="s", path="/p"),
                           message=True)
    assert d["class"] == "hard_load_failure" and d["loader_stderr"] == "s" and d["path"] == "/p"
    assert d["message"] == "HardLoadFailure: boom"
