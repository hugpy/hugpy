"""Vision GGUFs load WITH their projector or not at all (2026-09-23).

aeb's load_reports for Qwen2.5-VL-7B / Qwen3.6-*-A3B / unsloth~Qwen3.8-27B read
"vision model loaded in-process (text-only — the python binding cannot load the
mmproj projector)": when the slot pool did not seat the model, get_llama_runner
fell through to llama-cpp-python, which loaded the language weights only and
answered every image turn blind with ok:true.

Pinned here:
  (a) a model dir with an mmproj -> the slot child's argv carries
      ``--mmproj <that file>``;
  (b) the in-process fallback is REFUSED for a projector model with the typed
      ``vision_needs_slot`` load failure (fit numbers include the projector);
  (c) a seated slot gives an HTTP runner with ``is_vision`` forced on, so
      /infer image_url parts and /ml/vision ``images`` fold in on the SAME
      runner; a child that came up without the projector is refused;
  (d) the slot autofit / serve-spec fit reserve the projector bytes.
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
sa = importlib.import_module("hugpy_engine.serve.slot_agent")
spill = importlib.import_module("hugpy_engine.spill")
chat_schemas = importlib.import_module("hugpy_engine.schemas.chat_schemas")

GIB = 1 << 30
RED_PNG = "data:image/png;base64,iVBORw0KGgo="


@pytest.fixture
def vl_dir(tmp_path):
    d = tmp_path / "Qwen2.5-VL-3B-Instruct-GGUF"
    d.mkdir()
    model = d / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 4096)
    proj = d / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
    with open(proj, "wb") as fh:
        fh.truncate(3 * 1024 * 1024)
    return types.SimpleNamespace(dir=str(d), model=str(model), proj=str(proj))


@pytest.fixture
def env(monkeypatch, vl_dir):
    calls = {"llama": 0}

    class _Llama:
        def __init__(self, model_path, **kw):
            calls["llama"] += 1

    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    cfg = types.SimpleNamespace(name="vl", tasks=["text-generation"])  # task row lacks vision
    for mod in (getmod, prmod):
        monkeypatch.setattr(mod, "get_model_config", lambda k: cfg)
        monkeypatch.setattr(mod, "ensure_serving_weights", lambda k: vl_dir.dir)
        monkeypatch.setattr(mod, "get_gguf_file", lambda d, c, prefer=None: vl_dir.model)
    monkeypatch.setattr(spill, "llama_kwargs", lambda p: {})
    monkeypatch.setattr(spill, "cpu_resident_bytes", lambda p, n: 0)
    monkeypatch.setattr(spill, "gguf_moe_detail", lambda p: {"is_moe": False})
    monkeypatch.setattr(spill, "rpc_servers", lambda: None)
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 2 * GIB)
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: 8 * GIB)
    shard = importlib.import_module("hugpy_engine.llama.runners.src.shard_server")
    monkeypatch.setattr(shard, "ensure_vision_server", lambda k: None)
    monkeypatch.setattr(getmod, "_require_profile_ready", lambda k: None)
    monkeypatch.setattr(getmod, "_slot_serves_vision", lambda base: True)

    real_http = getmod.LlamaCppRunner

    class _Http(real_http):
        def __init__(self, model_key, *, env_path=None, base_url=None):
            if not base_url:
                raise ConnectionError("no llama-server on the default port")
            super().__init__(model_key, base_url=base_url)
    monkeypatch.setattr(getmod, "LlamaCppRunner", _Http)
    monkeypatch.setattr(slots, "slots_enabled", lambda: True)
    monkeypatch.setattr(getmod, "_REFUSED", {})
    monkeypatch.setattr(getmod, "_LLAMA_INSTANCES", {})
    monkeypatch.delenv("HUGPY_INPROCESS_VISION", raising=False)
    return calls


def _pool(endpoint=None, exc=None, seen=None):
    class _Pool:
        def endpoint_for(self, key, opts=None):
            if seen is not None:
                seen.append(dict(opts or {}))
            if exc is not None:
                raise exc
            return endpoint
    return _Pool


# (a) ---------------------------------------------------------------------
def test_slot_argv_carries_mmproj(monkeypatch, vl_dir):
    serve = importlib.import_module("hugpy_engine.serve.serve")
    monkeypatch.setattr(serve, "LLAMA_SERVER_BIN", "/bin/echo")
    monkeypatch.setattr(sa, "_server_supports_flag", lambda b, f: True)
    seen = {}

    def _autofit(p, free_vram=None, extra_reserve_bytes=0, **_kw):
        seen["reserve"] = extra_reserve_bytes
        return -1
    monkeypatch.setattr(spill, "autofit_gpu_layers", _autofit)
    monkeypatch.setattr(spill, "gguf_moe_detail", lambda path: {"is_moe": False})
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 20 * GIB)
    monkeypatch.setattr(spill, "total_vram_bytes", lambda: 24 * GIB)
    for e in ("HUGPY_ALLOC_MODE", "HUGPY_N_CPU_MOE", "HUGPY_N_GPU_LAYERS",
              "HUGPY_HOT_CACHE_ROOT", "HUGPY_MODEL_CACHE"):
        monkeypatch.delenv(e, raising=False)
    argv, *_ = sa._build_cmd("Qwen2.5-VL-3B-Instruct-GGUF", path=vl_dir.model, ctx=4096)
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1] == vl_dir.proj
    # (d) the slot autofit reserved the projector the child loads.
    assert seen["reserve"] == 3 * 1024 * 1024


# (b) ---------------------------------------------------------------------
def test_inprocess_refused_when_slots_busy(env, monkeypatch, vl_dir):
    monkeypatch.setattr(slots, "SlotPool", _pool(endpoint=None))
    with pytest.raises(lf.ModelLoadFailure) as ei:
        getmod.get_llama_runner("Qwen2.5-VL-3B-Instruct-GGUF")
    err = ei.value
    print("\nVISION:", err)
    assert env["llama"] == 0                       # never loaded text-only
    assert err.load_class == "vision_needs_slot"
    msg = str(err)
    assert "every slot is busy" in msg
    assert "projector 3 MiB" in msg and "mmproj-Qwen2.5-VL-3B" in msg
    assert "free VRAM 2.00 GiB of 8.00 GiB" in msg
    assert lf.load_failure_of(err)["class"] == "vision_needs_slot"
    # The refuse-backoff re-raise keeps the class.
    with pytest.raises(lf.ModelLoadFailure) as again:
        getmod.get_llama_runner("Qwen2.5-VL-3B-Instruct-GGUF")
    assert lf.load_failure_of(again.value)["class"] == "vision_needs_slot"


def test_inprocess_refused_on_soft_slot_refusal_names_it(env, monkeypatch):
    monkeypatch.setattr(slots, "SlotPool", _pool(exc=RuntimeError(
        "slot load failed: needs ~9.1 GB VRAM but only 2.0 GB free")))
    with pytest.raises(lf.ModelLoadFailure) as ei:
        getmod.get_llama_runner("vl")
    assert ei.value.load_class == "vision_needs_slot"
    assert "needs ~9.1 GB VRAM" in str(ei.value)
    assert env["llama"] == 0


def test_python_runner_direct_refuses_projector_model(env, vl_dir):
    with pytest.raises(lf.ModelLoadFailure) as ei:
        prmod.LlamaCppPythonRunner("vl")
    assert ei.value.load_class == "vision_needs_slot"
    assert env["llama"] == 0


def test_text_model_still_falls_back_in_process(env, monkeypatch, vl_dir):
    Path(vl_dir.proj).unlink()                     # no projector -> text model
    monkeypatch.setattr(slots, "SlotPool", _pool(endpoint=None))
    r = getmod.get_llama_runner("text-model")
    assert isinstance(r, prmod.LlamaCppPythonRunner) and env["llama"] == 1


def test_classify_text_knows_vision_needs_slot():
    assert lf.classify_text("slot load failed: RuntimeError: m: vision_needs_slot — "
                            "vision model (mmproj sidecar present)") == "vision_needs_slot"


# (c) ---------------------------------------------------------------------
def test_slot_seat_forces_vision_and_both_routes_share_runner(env, monkeypatch, vl_dir):
    seen = []
    monkeypatch.setattr(slots, "SlotPool", _pool(endpoint="http://127.0.0.1:9201", seen=seen))
    r = getmod.get_llama_runner("vl")
    assert r.base_url == "http://127.0.0.1:9201"
    assert r.is_vision is True                     # registry row lacked the task
    assert seen and seen[0]["path"] == vl_dir.model
    # /infer (OpenAI image_url parts) and /ml/vision (images list) both become
    # ChatRequest.images and both land on this ONE cached runner.
    infer_req = chat_schemas.ChatRequest(model_key="vl", messages=[{"role": "user", "content": [
        {"type": "text", "text": "what color is the square? one word"},
        {"type": "image_url", "image_url": {"url": RED_PNG}}]}])
    ml_req = chat_schemas.ChatRequest(model_key="vl", messages=[
        {"role": "user", "content": "what color is the square? one word"}], images=[RED_PNG])
    assert infer_req.images == ml_req.images == [RED_PNG]
    assert getmod.get_llama_runner("vl") is r
    for req in (infer_req, ml_req):
        msgs = r._attach_image([dict(m) for m in req.messages], req)
        parts = msgs[-1]["content"]
        assert {"type": "image_url", "image_url": {"url": RED_PNG}} in parts


def test_seated_child_without_projector_is_refused(env, monkeypatch):
    monkeypatch.setattr(getmod, "_slot_serves_vision", lambda base: False)
    monkeypatch.setattr(slots, "SlotPool", _pool(endpoint="http://127.0.0.1:9201"))
    with pytest.raises(lf.ModelLoadFailure) as ei:
        getmod.get_llama_runner("vl")
    assert ei.value.load_class == "vision_needs_slot"
    assert "without --mmproj" in str(ei.value)
    assert env["llama"] == 0


def test_no_slot_pool_uses_native_mmproj_server(env, monkeypatch):
    monkeypatch.setattr(slots, "slots_enabled", lambda: False)
    shard = importlib.import_module("hugpy_engine.llama.runners.src.shard_server")
    monkeypatch.setattr(shard, "ensure_vision_server", lambda k: "http://127.0.0.1:8300")
    r = getmod.get_llama_runner("vl")
    assert r.base_url == "http://127.0.0.1:8300" and r.is_vision is True
    assert env["llama"] == 0


# (d) ---------------------------------------------------------------------
def test_projector_bytes_is_the_loaded_projector(vl_dir):
    # A second, larger precision beside it: the reserve must size the ONE the
    # child loads (find_mmproj's pick), which is also what --mmproj gets.
    from hugpy_platform.utils import find_mmproj
    big = Path(vl_dir.dir) / "mmproj-Qwen2.5-VL-3B-Instruct-f32.gguf"
    with open(big, "wb") as fh:
        fh.truncate(6 * 1024 * 1024)
    pick = find_mmproj(vl_dir.model)
    assert spill.vision_projector_bytes(vl_dir.model) == Path(pick).stat().st_size


def test_serve_spec_autofit_reserves_projector(monkeypatch, vl_dir):
    serve = importlib.import_module("hugpy_engine.serve.serve")
    seen = {}
    monkeypatch.setattr(spill, "free_vram_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(spill, "autofit_gpu_layers",
                        lambda p, free_vram=None, extra_reserve_bytes=0, **_k:
                        seen.setdefault("reserve", extra_reserve_bytes) and 7)
    serve._ngl_for_alloc_mode("max-gpu", vl_dir.model, {})
    assert seen["reserve"] == 3 * 1024 * 1024
