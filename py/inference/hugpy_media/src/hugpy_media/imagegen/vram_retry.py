"""One-shot VRAM-class retry classification (k71).

ComfyUI-driven generations (and their in-process diffusers siblings)
intermittently fail on the FIRST attempt with an OOM/allocation error caused by
stale pre-eviction VRAM/pool state — the headroom eviction hadn't settled when
the gen committed — while an IDENTICAL second try succeeds. The dispatch seams
(``ComfyRunner.run``, ``ImageGenRunner.run``, ``Img2ImgRunner.run``,
``video_intel.runners.identity_mesh.run``) retry ONCE when, and only when, the
first failure classifies as that VRAM/eviction/OOM/allocation class.

Classification is POSITIVE-MARKER based and deliberately narrow: workflow
validation ("ComfyUI rejected the workflow (400): …"), a missing model/
checkpoint ("value not in list: ckpt_name …"), a poll timeout ("did not finish
prompt … within …s"), "finished but produced no images", connection errors and
every other failure class carry none of these markers and surface exactly as
before, on the first attempt. A TimeoutError is additionally excluded by TYPE:
a timed-out gen may still be running server-side, and re-running it is a
duplication hazard the eviction class doesn't have (its first attempt already
died).

This module is pure stdlib and import-light on purpose: the comfy runner, the
imagegen runners and the video_intel comfy stations all import it without
dragging in httpx/torch/anything heavy.
"""
from __future__ import annotations

import os

# The REAL wording of the VRAM/allocation failure class as it reaches these
# seams (derived from torch + ComfyUI + the house classifier vocabulary in
# managers/resolvers/remote.py's _PERMANENT_LOAD_MARKERS and the wan runners'
# is_oom check — this list is comfy/diffusers-side, so it stays narrower):
#   * "CUDA out of memory. Tried to allocate …"  — torch < 2.4 OOM text (also
#     the wording the wan runners' is_oom check matches on).
#   * "Allocation on device 0 …"                 — torch >= 2.4
#     OutOfMemoryError text; this is what ComfyUI's execution_error history
#     entry carries as exception_message on a VRAM-starved gen.
#   * "torch.OutOfMemoryError"/"torch.cuda.OutOfMemoryError" — the
#     exception_type field in that same history JSON (and the exception class
#     name when the failure is in-process).
#   * "HIP out of memory"                        — the ROCm twin.
#   * CUBLAS/CUDNN workspace-allocation failures — the same starvation one
#     library call later.
#   * "not enough memory"/"unable to allocate"   — ComfyUI model_management /
#     host-side allocator wording for the identical condition.
RETRYABLE_VRAM_MARKERS = (
    "out of memory",              # covers "CUDA out of memory", "HIP out of memory"
    "outofmemory",                # exception TYPE names, incl. comfy's exception_type
    "allocation on device",       # torch >= 2.4 OutOfMemoryError message
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "not enough memory",
    "insufficient memory",
    "unable to allocate",
)

_SETTLE_ENV = "HUGPY_VRAM_RETRY_SETTLE_S"
_SETTLE_DEFAULT_S = 3.0
_SETTLE_MAX_S = 10.0


def is_retryable_vram_failure(exc: BaseException) -> bool:
    """Whether a FIRST-attempt failure is the VRAM/eviction/OOM/allocation class
    that one retry (after the pool settles) is allowed to absorb.

    Matches the exception's type name + message against the positive markers
    above. TimeoutError is excluded by type regardless of wording: a timeout is
    not the eviction class, and its gen may still be running server-side."""
    if isinstance(exc, TimeoutError):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in RETRYABLE_VRAM_MARKERS)


def settle_delay_s() -> float:
    """The bounded pool-settle sleep between the retryable first failure and the
    one retry — the FINAL settle step, after the real eviction/headroom hooks
    have been re-driven. Default 3s, operator-tunable via
    ``HUGPY_VRAM_RETRY_SETTLE_S``, clamped to [0, 10] so a typo can never turn
    the retry seam into a hang."""
    raw = os.environ.get(_SETTLE_ENV)
    if raw is None or not str(raw).strip():
        return _SETTLE_DEFAULT_S
    try:
        val = float(raw)
    except ValueError:
        return _SETTLE_DEFAULT_S
    return max(0.0, min(_SETTLE_MAX_S, val))
