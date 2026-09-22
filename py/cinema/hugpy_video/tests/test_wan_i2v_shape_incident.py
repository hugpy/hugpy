"""2026-08-21 INCIDENT — "[io_error] wan i2v inference failed: The size of tensor a
(36) must match the size of tensor b (16) at non-singleton dimension 1" marked
retryable, looping on the same spec.

ROOT CAUSE (media_jobs.db job 73727ed4…, and 18 identical rows since 2026-07-21):
``capability=i2v`` with ``start_image=null`` routed to wan2.1-i2v-14b-720p; with no
image ``run_wan_i2v`` takes the t2v ``WanPipeline`` branch, whose latents are sized
from ``transformer.config.in_channels`` (36 for the i2v DiT) while the DiT emits 16
-> the scheduler step dies. Deterministic; a retry can never fix it.

What this suite locks (no GPU, no weights, no diffusers import):
  * geometry snapping (frames 4k+1, width/height multiples of 16) + exact image fit
  * checkpoint <-> conditioning pairing refusal (CONFIG_ERROR, before any load)
  * 480P/720P variant pick by requested resolution
  * exception classification: shape -> SHAPE_ERROR, OOM -> OOM, API -> CONFIG_ERROR
  * bus translation: SHAPE_ERROR/CONFIG_ERROR never retryable; context veto honoured
  * per-spec retry budget (HUGPY_WAN_RETRY_BUDGET, default 3)
  * multi-GPU layout planner (component / layers / single, never pooled) + applier
    on CPU-resident fake modules

Run:
  cd /srv/share/projects/hugpy/dev/abstract_hugpy_dev
  venv/bin/python -m pytest -q tests/studio/test_wan_i2v_shape_incident.py
"""
from __future__ import annotations

import pytest
import json
import logging
import os
import sys
import tempfile

os.environ.setdefault("STUDIO_ALLOW_UNPINNED", "1")

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugpy_video.intel.studio.enums import Capability, DeterminismClass, Framework, Precision, Task
from hugpy_video.intel.studio.errors import ErrorCode, StageError
from hugpy_video.intel.studio.schemas import RenderManifest, Resolution, SamplerConfig, SeedBundle
from hugpy_video.intel.studio.runners import wan_i2v as W
from hugpy_video.intel.runners.studio_i2v import _stage_error_to_job_error

INCIDENT_MSG = ("The size of tensor a (36) must match the size of tensor b (16) at "
                "non-singleton dimension 1")


def _manifest(width=832, height=480, fps=16, frames=None, model_id="wan2.1-i2v-14b-720p",
              precision=Precision.BF16, task=Task.I2V, capability=Capability.I2V):
    return RenderManifest(
        render_id="r-incident", capability=capability, model_id=model_id,
        weight_hash=None, framework=Framework.WAN, task=task, precision=precision,
        seeds=SeedBundle(global_seed=0, stage_seeds=(("base", 0),)),
        sampler=SamplerConfig(sampler="unipc", scheduler="flow_match", steps=32, cfg=5.0),
        resolution_ladder=(Resolution(width, height, fps),),
        determinism_class=DeterminismClass.SEEDED_APPROX, requested_frames=frames)


# ---------------------------------------------------------------- geometry ----
def test_snap_geometry_pure():
    assert W._snap_geometry(832, 480, 81) == (832, 480, 81)
    assert W._snap_geometry(1280, 720, 81) == (1280, 720, 81)
    assert W._snap_geometry(1000, 500, 30) == (992, 496, 29)
    assert W._snap_geometry(5, 5, 0) == (16, 16, 1)
    assert W._snap_geometry(848, 480, 82) == (848, 480, 81)


def test_wan_geometry_snaps_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger=W.logger.name)
    w, h, fps, n = W._wan_geometry(_manifest(width=1000, height=500, frames=30))
    assert (w, h, fps, n) == (992, 496, 16, 29)
    assert any("snapped" in r.getMessage() for r in caplog.records)


def test_wan_geometry_no_log_when_on_grid(caplog):
    caplog.set_level(logging.WARNING, logger=W.logger.name)
    assert W._wan_geometry(_manifest(frames=81)) == (832, 480, 16, 81)
    assert not any("snapped" in r.getMessage() for r in caplog.records)


def test_fit_image_exact_size_with_center_crop():
    from PIL import Image
    img = Image.new("RGB", (1000, 400), (10, 20, 30))
    out = W._fit_image(img, 832, 480)
    assert out.size == (832, 480)
    tall = W._fit_image(Image.new("L", (400, 1000)), 1280, 720)
    assert tall.size == (1280, 720) and tall.mode == "RGB"
    same = Image.new("RGB", (832, 480))
    assert W._fit_image(same, 832, 480).size == (832, 480)


# ---------------------------------------------------- checkpoint pairing ----
def test_checkpoint_pairing_refuses_i2v_without_image():
    msg = W._checkpoint_pairing_error(36, False, "Wan-AI/Wan2.1-I2V-14B-720P")
    assert msg and "36-vs-16" in msg and "start_image" in msg
    assert W._checkpoint_pairing_error(36, True, "x") is None
    msg2 = W._checkpoint_pairing_error(16, True, "Wan-AI/Wan2.1-T2V-1.3B")
    assert msg2 and "TEXT-to-video" in msg2
    assert W._checkpoint_pairing_error(16, False, "x") is None
    assert W._checkpoint_pairing_error(None, False, "x") is None


def test_transformer_in_channels_reads_config():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "transformer"))
        with open(os.path.join(d, "transformer", "config.json"), "w") as fh:
            json.dump({"in_channels": 36, "out_channels": 16}, fh)
        assert W._transformer_in_channels(d) == 36
        assert W._transformer_in_channels(os.path.join(d, "nope")) is None


def test_variant_for_geometry():
    assert W._variant_for_geometry("Wan-AI/Wan2.1-I2V-14B-720P", 832, 480) == \
        "Wan-AI/Wan2.1-I2V-14B-480P"
    assert W._variant_for_geometry("Wan-AI/Wan2.1-I2V-14B-720P", 1280, 720) == \
        "Wan-AI/Wan2.1-I2V-14B-720P"
    assert W._variant_for_geometry("Wan-AI/Wan2.1-I2V-14B-480P", 1280, 720) == \
        "Wan-AI/Wan2.1-I2V-14B-720P"
    assert W._variant_for_geometry("Wan-AI/Wan2.1-T2V-1.3B", 832, 480) == \
        "Wan-AI/Wan2.1-T2V-1.3B"


# ------------------------------------------------------- classification ----
class OutOfMemoryError(RuntimeError):  # torch.OutOfMemoryError look-alike
    pass


def test_classify_exception():
    assert W._classify_exception(RuntimeError(INCIDENT_MSG))[0] is ErrorCode.SHAPE_ERROR
    assert W._classify_exception(RuntimeError(
        "Given groups=1, weight of size [5120, 36, 1, 2, 2], expected input[1, 16, 21, "
        "60, 104] to have 36 channels"))[0] is ErrorCode.SHAPE_ERROR
    assert W._classify_exception(OutOfMemoryError("CUDA out of memory"))[0] is ErrorCode.OOM
    assert W._classify_exception(RuntimeError("CUDA error: out of memory"))[0] is ErrorCode.OOM
    assert W._classify_exception(TypeError(
        "__call__() got an unexpected keyword argument 'callback_on_step_end'"))[0] \
        is ErrorCode.CONFIG_ERROR
    assert W._classify_exception(OSError("No space left on device"))[0] is ErrorCode.IO_ERROR
    assert W._classify_exception(RuntimeError("ffmpeg died"))[0] is ErrorCode.IO_ERROR


def test_bus_translation_retryability():
    shape = StageError(ErrorCode.SHAPE_ERROR, INCIDENT_MSG, (("retryable", "false"),))
    assert _stage_error_to_job_error(shape).retryable is False
    cfg = StageError(ErrorCode.CONFIG_ERROR, "i2v checkpoint without start_image")
    assert _stage_error_to_job_error(cfg).retryable is False
    io = StageError(ErrorCode.IO_ERROR, "disk hiccup")
    assert _stage_error_to_job_error(io).retryable is True
    assert _stage_error_to_job_error(io.with_context(retryable="false")).retryable is False
    assert _stage_error_to_job_error(StageError(ErrorCode.OOM, "oom")).retryable is True
    # the context can never UPGRADE a deterministic code
    assert _stage_error_to_job_error(
        shape.with_context(retryable="true")).retryable is False


def test_retry_budget_per_spec(monkeypatch):
    monkeypatch.delenv(W._RETRY_BUDGET_ENV, raising=False)
    base = (("content_hash", "72b1ad00"),)
    with tempfile.TemporaryDirectory() as d:
        c1 = dict(W._budgeted_context(d, ErrorCode.IO_ERROR, "x", base))
        c2 = dict(W._budgeted_context(d, ErrorCode.IO_ERROR, "x", base))
        c3 = dict(W._budgeted_context(d, ErrorCode.IO_ERROR, "x", base))
        assert c1.get("retryable") is None and c2.get("retryable") is None
        assert c3["retryable"] == "false" and c3["failures"] == "3"
        with open(os.path.join(d, W._FAILURES_NAME)) as fh:
            ledger = json.load(fh)
        assert ledger["io_error"]["count"] == 3
        # a different code has its own count
        assert dict(W._budgeted_context(d, ErrorCode.OOM, "x", base))["failures"] == "1"
    monkeypatch.setenv(W._RETRY_BUDGET_ENV, "1")
    with tempfile.TemporaryDirectory() as d:
        assert dict(W._budgeted_context(d, ErrorCode.OOM, "x", base))["retryable"] == "false"
    # an unwritable dir never breaks error reporting
    ctx = dict(W._budgeted_context("/proc/definitely/not/writable", ErrorCode.IO_ERROR,
                                   "x", base))
    assert ctx["failures"] == "1"


# --------------------------------------------------------- multi-GPU plan ----
TWO_3090 = [(24.0, 23.0), (24.0, 23.0)]
DIT14_BF16, DIT14_NF4, AUX_I2V, ACT = 30.54, 8.59, 12.23, 4.6   # 14B i2v @480p81f


def test_plan_auto_single_gpu():
    lay = W.plan_device_layout("auto", [(24.0, 23.0)], DIT14_BF16, DIT14_NF4, AUX_I2V, ACT)
    assert lay.mode == "single" and lay.requested == "auto"


def test_plan_auto_two_gpus_is_component_and_quantizes_14b():
    lay = W.plan_device_layout("auto", TWO_3090, DIT14_BF16, DIT14_NF4, AUX_I2V, ACT)
    assert lay.mode == "component"
    assert lay.transformer_device == "cuda:0" and lay.aux_device == "cuda:1"
    assert lay.quantize is True           # 30.5 GiB bf16 > 23-1.5-4.6 on cuda:0 ALONE
    assert "nf4" in lay.describe()


def test_plan_never_pools_vram():
    # pooled 46 GiB would hold a 30 GiB bf16 DiT; per-device it must not
    lay = W.plan_device_layout("component", TWO_3090, 30.0, 8.0, 5.0, 1.0)
    assert lay.quantize is True


def test_plan_component_bf16_when_it_fits_one_card():
    lay = W.plan_device_layout("component", TWO_3090, 2.64, 0.8, 11.1, 4.6)   # 1.3B t2v
    assert lay.mode == "component" and lay.quantize is False


def test_plan_component_aux_must_fit_cuda1():
    lay = W.plan_device_layout("component", [(24.0, 23.0), (24.0, 6.0)],
                               2.64, 0.8, 11.1, 4.6)
    assert lay.mode == "single" and "cuda:1" in lay.reason


def test_plan_layers_shards_with_per_card_budgets():
    lay = W.plan_device_layout("layers", TWO_3090, 20.0, 6.0, 12.23, 2.0)
    assert lay.mode == "layers"
    assert lay.max_memory == {0: "19.50GiB", 1: "7.27GiB"}
    assert lay.quantize is False


def test_plan_layers_falls_back_to_component_when_shards_too_small():
    lay = W.plan_device_layout("layers", TWO_3090, DIT14_BF16, DIT14_NF4, AUX_I2V, ACT)
    assert lay.mode == "component" and lay.quantize is True
    assert lay.reason.startswith("layers:")


def test_plan_explicit_single_and_requested_multi_without_gpus():
    assert W.plan_device_layout("single", TWO_3090, 1, 1, 1, 1).mode == "single"
    lay = W.plan_device_layout("layers", [(24.0, 23.0)], 1, 1, 1, 1)
    assert lay.mode == "single" and "needs 2+" in lay.reason
    assert W.plan_device_layout("bogus", [], 1, 1, 1, 1).requested == "auto"


def test_device_mode_from_env():
    assert W._device_mode_from_env({}) == "auto"
    assert W._device_mode_from_env({W._DEVICE_MODE_ENV: " Layers "}) == "layers"
    assert W._device_mode_from_env({W._DEVICE_MODE_ENV: "pooled"}) == "auto"


class _FakeCuda:
    def __init__(self, frees):
        self._frees = frees

    def device_count(self):
        return len(self._frees)

    def mem_get_info(self, i):
        gib = 1024 ** 3
        return int(self._frees[i] * gib), int(24.0 * gib)


class _FakeTorch:
    def __init__(self, frees):
        self.cuda = _FakeCuda(frees)


def test_plan_layout_for_manifest_with_fake_torch(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=W.logger.name)
    monkeypatch.setenv(W._DEVICE_MODE_ENV, "auto")
    lay = W._plan_layout_for(_manifest(frames=81), _FakeTorch([23.0, 23.0]), 832, 480, 81)
    assert lay.mode == "component" and lay.quantize is True
    assert any("wan device layout" in r.getMessage() and "cuda:1 24.0G" in r.getMessage()
               for r in caplog.records)
    lay1 = W._plan_layout_for(_manifest(frames=81), _FakeTorch([23.0]), 832, 480, 81)
    assert lay1.mode == "single"
    assert W._cuda_inventory(object()) == []


def test_component_sizes_match_footprint_table():
    sizes = W._component_sizes_gib("wan2.1-i2v-14b-720p", 832, 480, 81)
    assert sizes is not None
    dit_bf16, dit_nf4, aux, act = sizes
    assert 30.0 < dit_bf16 < 31.0 and 8.0 < dit_nf4 < 9.0 and 12.0 < aux < 12.5
    assert act > W._DECODE_WS_GIB
    assert W._component_sizes_gib("not-a-model", 1, 1, 1) is None


# ------------------------------------------------------ multi-GPU applier ----
def test_apply_device_layout_on_cpu_modules():
    torch = pytest.importorskip("torch")

    class _Out(dict):              # transformers ModelOutput look-alike
        pass

    class _Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x, mask=None):
            return _Out(last_hidden_state=self.lin(x))

    class _Dist:
        def __init__(self, parameters):
            self.parameters = parameters

    class _EncOut:
        def __init__(self, p):
            self.latent_dist = _Dist(p)

    class _Vae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def encode(self, x):
            self.seen.append(("encode", x.device.type))
            return _EncOut(x * 2)

        def decode(self, z, return_dict=True):
            self.seen.append(("decode", z.device.type))
            return (z,)

    class _Pipe:
        def __init__(self):
            self.transformer = torch.nn.Linear(2, 2)
            self.text_encoder = _Enc()
            self.image_encoder = None
            self.vae = _Vae()
            self.levers = []

        def enable_attention_slicing(self):
            self.levers.append("attn")

        @property
        def _execution_device(self):
            return torch.device("meta")

    pipe = _Pipe()
    lay = W.DeviceLayout("component", "auto", "cpu", "cpu", False, None, "test")
    engaged = W._apply_device_layout(pipe, lay, torch)
    assert "attention_slicing" in engaged
    assert pipe._execution_device == torch.device("cpu")
    assert pipe.device == torch.device("cpu")
    out = pipe.text_encoder(torch.zeros(1, 2))
    assert out["last_hidden_state"].device.type == "cpu"
    enc = pipe.vae.encode(torch.ones(1, 2))
    assert torch.equal(enc.latent_dist.parameters, torch.full((1, 2), 2.0))
    assert pipe.vae.decode(torch.ones(1, 2), return_dict=False)[0].device.type == "cpu"
    assert pipe.vae.seen == [("encode", "cpu"), ("decode", "cpu")]


def test_apply_device_layout_unwinds_on_failure():
    torch = pytest.importorskip("torch")

    class _Vae(torch.nn.Module):
        def encode(self, x):
            return x

    class _Pipe:
        transformer = torch.nn.Linear(2, 2)
        text_encoder = torch.nn.Linear(2, 2)
        image_encoder = None
        vae = _Vae()

    pipe = _Pipe()
    orig_encode = pipe.vae.encode
    lay = W.DeviceLayout("component", "auto", "cpu", "cuda:7", False, None, "test")
    try:
        W._apply_device_layout(pipe, lay, torch)
    except Exception:
        pass
    else:
        raise AssertionError("placing on cuda:7 must fail on a GPU-less box")
    assert pipe.vae.encode == orig_encode or "encode" not in pipe.vae.__dict__
    assert not pipe.text_encoder._forward_pre_hooks


def test_move_to_device_structures():
    torch = pytest.importorskip("torch")
    t = torch.zeros(1)
    moved = W._move_to_device({"a": t, "b": (t, [t])}, "cpu", torch)
    assert moved["a"].device.type == "cpu" and isinstance(moved["b"], tuple)
    assert W._move_to_device("str", "cpu", torch) == "str"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
