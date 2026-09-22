"""One-shot VRAM-class retry across the comfy dispatch seam (k71).

ComfyUI generations intermittently fail on the FIRST attempt with an
OOM/allocation error caused by stale pre-eviction VRAM/pool state; an identical
second try succeeds. The seams retry ONCE — and ONLY for that class.

Covered here:
  * vram_retry.is_retryable_vram_failure — positive-marker classification:
    the VRAM/eviction/OOM/allocation class retries; workflow-validation,
    missing-model/checkpoint, timeout, no-images, connection errors do NOT.
  * ComfyRunner.run — a retryable first failure triggers EXACTLY one retry
    (cancel first submission -> re-drive headroom hook -> settle -> resubmit);
    a second failure surfaces exactly as today; non-retryable never retries;
    a success needs no retry machinery at all.
  * ImageGenRunner.run — the in-process twin (evict-idle + release + settle).

The identity_mesh (video) half of the original script lives in
``py/services/hugpy_server/tests/integration/test_identity_mesh_vram_retry.py``
because it needs ``hugpy_video``, which media does not depend on.
"""
from __future__ import annotations

import asyncio
import os
import types

import pytest

from hugpy_media.comfy import comfy_runner
from hugpy_media.imagegen import imagegen_runner, vram_retry
from hugpy_media.imagegen.schemas import GeneratedImage, ImageGenRequest

is_retryable = vram_retry.is_retryable_vram_failure


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setenv("HUGPY_VRAM_RETRY_SETTLE_S", "0")   # no real sleeping in tests


# ===========================================================================
# Part 1 — classification: retryable ONLY for the VRAM/OOM/allocation class
# ===========================================================================
RETRYABLE = [
    # comfy history execution_error, torch >= 2.4 wording (exception_type +
    # exception_message ride the JSON detail):
    RuntimeError('ComfyUI execution error: {"exception_type": '
                 '"torch.OutOfMemoryError", "exception_message": "Allocation '
                 'on device 0 would exceed allowed memory. (out of memory)"}'),
    # torch < 2.4 wording (also what the wan runners' is_oom matches):
    RuntimeError("ComfyUI execution error: CUDA out of memory. Tried to "
                 "allocate 2.50 GiB (GPU 0; 23.56 GiB total capacity)"),
    # in-process exception TYPE name carries the class even with a terse message
    type("OutOfMemoryError", (RuntimeError,), {})("CUDA error"),
    # allocator-starved library calls one step later:
    RuntimeError("cuBLAS failure: CUBLAS_STATUS_ALLOC_FAILED"),
    RuntimeError("RuntimeError: CUDA error: out of memory"),
    # comfy model_management / host-side wording for the same condition:
    RuntimeError("Not enough memory to load the model"),
    RuntimeError("Unable to allocate 512.0 MiB for output tensor"),
]

NOT_RETRYABLE = [
    # workflow validation (the /prompt 400)
    RuntimeError("ComfyUI rejected the workflow (400): {\"error\": {\"type\": "
                 "\"prompt_outputs_failed_validation\"}}"),
    # missing model/checkpoint
    RuntimeError("ComfyUI rejected the workflow (400): value not in list: "
                 "ckpt_name: 'sd15.safetensors' not in []"),
    # timeout — excluded by TYPE even though a message could be anything
    TimeoutError("ComfyUI did not finish prompt abc within 600s"),
    # finished-but-empty
    RuntimeError("ComfyUI finished but produced no images"),
    # transport failure
    ConnectionError("connection refused: 127.0.0.1:8188"),
    # generic execution error with no VRAM wording
    RuntimeError("ComfyUI execution error: mat1 and mat2 shapes cannot be "
                 "multiplied"),
    # a timeout whose text happens to mention memory is STILL excluded by type
    TimeoutError("out of memory while waiting"),
]


@pytest.mark.parametrize("exc", RETRYABLE, ids=[str(e)[:40] for e in RETRYABLE])
def test_vram_class_is_retryable(exc):
    assert is_retryable(exc)


@pytest.mark.parametrize("exc", NOT_RETRYABLE, ids=[str(e)[:40] for e in NOT_RETRYABLE])
def test_other_failures_are_not_retryable(exc):
    assert not is_retryable(exc)


def test_settle_delay_is_env_clamped(monkeypatch):
    monkeypatch.setenv("HUGPY_VRAM_RETRY_SETTLE_S", "9999")
    assert vram_retry.settle_delay_s() == 10.0
    monkeypatch.setenv("HUGPY_VRAM_RETRY_SETTLE_S", "garbage")
    assert vram_retry.settle_delay_s() == 3.0


# ===========================================================================
# Part 2 — ComfyRunner.run: exactly ONE retry, and only for the VRAM class
# ===========================================================================
REQ = ImageGenRequest(request_id="req-1", model_key="comfy-ckpt", prompt="a cat")
IMAGES = [GeneratedImage(path="/tmp/x.png", width=64, height=64)]


def _oom():
    exc = RuntimeError('ComfyUI execution error: {"exception_type": '
                       '"torch.OutOfMemoryError", "exception_message": '
                       '"Allocation on device 0"}')
    exc.comfy_prompt_id = "prompt-first"
    return exc


def make_runner(script):
    """A ComfyRunner whose _generate pops outcomes off `script` (an exception
    -> raise, anything else -> return); records the attempt count."""
    cfg = types.SimpleNamespace(model_key="comfy-ckpt", filename="ckpt.safetensors")
    runner = comfy_runner.ComfyRunner(cfg)
    calls = {"generate": 0, "cancel": []}

    def fake_generate(req):
        calls["generate"] += 1
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    runner._generate = fake_generate
    runner._cancel_prompt = lambda pid: calls["cancel"].append(pid)
    return runner, calls


@pytest.fixture
def headroom_hook(monkeypatch):
    calls = []
    monkeypatch.setattr(comfy_runner, "_COMFY_HEADROOM_HOOK",
                        lambda mk, job_id=None: calls.append((mk, job_id)))
    return calls


def test_comfy_retryable_then_success_retries_exactly_once(headroom_hook):
    runner, calls = make_runner([_oom(), IMAGES])
    res = asyncio.run(runner.run(REQ))
    assert res.ok is True
    assert calls["generate"] == 2, "exactly one retry (two _generate calls)"
    assert calls["cancel"] == ["prompt-first"], "first submission cancelled by its prompt id"
    assert headroom_hook == [("comfy-ckpt", "req-1")], "headroom hook re-driven before the retry"


def test_comfy_second_failure_surfaces_without_third_attempt(headroom_hook):
    oom2 = RuntimeError("ComfyUI execution error: CUDA out of memory. Tried "
                        "to allocate 2.50 GiB")
    runner, calls = make_runner([_oom(), oom2])
    res = asyncio.run(runner.run(REQ))
    assert res.ok is False
    assert calls["generate"] == 2, "never a third attempt"
    assert "Tried to allocate" in (res.error or ""), "the SECOND failure's wording surfaces"


@pytest.mark.parametrize("exc", [
    TimeoutError("ComfyUI did not finish prompt p1 within 600s"),
    RuntimeError("ComfyUI rejected the workflow (400): value not in list: ckpt_name"),
    RuntimeError("ComfyUI finished but produced no images"),
], ids=["timeout", "validation", "no-images"])
def test_comfy_non_retryable_never_retries(headroom_hook, exc):
    runner, calls = make_runner([exc, IMAGES])
    res = asyncio.run(runner.run(REQ))
    assert calls["generate"] == 1
    assert res.ok is False
    assert calls["cancel"] == [] and headroom_hook == []


def test_comfy_plain_success_touches_no_retry_machinery(headroom_hook):
    runner, calls = make_runner([IMAGES])
    res = asyncio.run(runner.run(REQ))
    assert res.ok and calls["generate"] == 1 and calls["cancel"] == []
    assert headroom_hook == []


# ===========================================================================
# Part 3 — ImageGenRunner.run: the in-process twin of the same seam
# ===========================================================================
def make_imagegen_runner(script):
    cfg = types.SimpleNamespace(model_key="sd15")
    runner = imagegen_runner.ImageGenRunner(cfg)
    calls = {"generate": 0}

    def fake_generate(req):
        calls["generate"] += 1
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    runner._generate = fake_generate
    return runner, calls


@pytest.fixture
def settles(monkeypatch):
    seen = []
    monkeypatch.setattr(imagegen_runner, "_settle_for_vram_retry",
                        lambda cache, lock, keep: seen.append(keep))
    return seen


def test_imagegen_retryable_then_success(settles):
    torch_oom = type("OutOfMemoryError", (RuntimeError,), {})(
        "CUDA out of memory. Tried to allocate 20.00 MiB")
    runner, calls = make_imagegen_runner([torch_oom, IMAGES])
    res = asyncio.run(runner.run(
        ImageGenRequest(request_id="r2", model_key="sd15", prompt="a dog")))
    assert res.ok is True and calls["generate"] == 2
    assert settles == ["sd15"], "settle re-drove eviction for the generating model"


def test_imagegen_non_retryable_single_attempt(settles):
    runner, calls = make_imagegen_runner(
        [ValueError("image-to-image requires an init image"), IMAGES])
    res = asyncio.run(runner.run(
        ImageGenRequest(request_id="r3", model_key="sd15", prompt="a dog")))
    assert res.ok is False and calls["generate"] == 1 and settles == []


def test_real_settle_helper_noops_without_gpu():
    # The REAL settle helper is safe on a no-GPU box (empty cache, settle=0).
    imagegen_runner._settle_for_vram_retry({}, imagegen_runner.threading.Lock(), "sd15")
