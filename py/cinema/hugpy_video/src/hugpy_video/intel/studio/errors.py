"""Errors as data (INV-3).

Two disciplines, deliberately separated:

  * Runtime/policy failures (no model fits, license mismatch, identity drift, OOM)
    are *values*: `Err(StageError(...))`. They flow into the job ledger and drive
    retry/regen policy. Nothing in the render path raises them into the void.

  * Programmer error caught at import (a malformed registry, a missing runner) is
    NOT data. It raises `RegistryError` loudly at boot so the process never starts
    in a broken state. Same for `ConfigError` on missing env wiring (INV-5).

`Result` is a tiny stdlib Ok|Err. No external deps.
"""

from __future__ import annotations

# TODO(P0-1): reconcile with result_schema.JobError / comms.jobs.JobError — do not
# let this become a third error vocab. Studio's Ok/Err/StageError/ErrorCode stay
# self-contained for this slice; the bind-reconciliation slice unifies them.

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


class ErrorCode(str, Enum):
    NO_CAPABLE_MODEL = "no_capable_model"
    VRAM_EXCEEDED = "vram_exceeded"
    LICENSE_VIOLATION = "license_violation"
    RESOLUTION_UNSUPPORTED = "resolution_unsupported"
    RUNNER_MISSING = "runner_missing"
    NATIVE_AUDIO_UNAVAILABLE = "native_audio_unavailable"
    LATENCY_BUDGET_EXCEEDED = "latency_budget_exceeded"
    # STR-6: locked identity + a real-time latency budget is a forbidden combo.
    CAPABILITY_STREAMING_CONFLICT = "capability_streaming_conflict"
    MISSING_CONSENT = "missing_consent"           # LEGAL-1
    IDENTITY_DRIFT = "identity_drift"             # ID-5
    OOM = "oom"
    NAN_IN_VAE = "nan_in_vae"
    IO_ERROR = "io_error"                 # frame/artifact write failed (P0-B1 runner)
    ASSEMBLY_FAILED = "assembly_failed"   # ffmpeg mux failed (P0-B1 runner)
    # Cooperative mid-render cancel (Task 1): the runner polled a caller-supplied
    # should_cancel() and aborted BEFORE writing a clip. INTENTIONAL, so NEVER
    # retryable — the same spec re-run should regenerate, not auto-retry. The
    # string matches media_bus's terminal vocab so the bus adapter maps it straight
    # to a "cancelled" job status.
    CANCELLED = "cancelled"               # aborted mid-render via should_cancel()
    # P0-6 real-model runner preflight (errors-as-data on a box that can't run yet)
    DEPS_MISSING = "deps_missing"         # torch/diffusers/bitsandbytes not installed
    NO_GPU = "no_gpu"                     # no CUDA device available for inference
    WEIGHTS_MISSING = "weights_missing"   # model weights not on disk under weights root
    # B-3 VACE v2v preflight: a video-to-video render is DEFINED by the clip it
    # restyles/enhances, so a v2v request whose manifest carries no (or a
    # nonexistent) source_video is a SPEC error — malformed on ANY box, GPU or not.
    # It is checked BEFORE deps/GPU/weights so the spec error is reported here rather
    # than masked by a GPU-less box's DEPS_MISSING. INTENTIONAL / not retryable (the
    # same source-less spec re-run fails identically).
    SOURCE_MISSING = "source_missing"     # v2v render has no source clip to enhance
    # IDENTITY LOCK (id_lock) preflight: an identity-locked render is DEFINED by the
    # reference image(s) of the subject it preserves (Wan VACE reference-to-video). An
    # id_lock request whose manifest carries no (or a nonexistent) reference_images is a
    # SPEC error — malformed on ANY box, GPU or not. Checked BEFORE deps/GPU/weights so
    # the spec error is reported here rather than masked by a GPU-less DEPS_MISSING.
    # INTENTIONAL / not retryable (the same reference-less spec re-run fails identically).
    REFERENCE_MISSING = "reference_missing"   # id_lock render has no reference image(s)
    # DIRECT MODEL CHOICE (pin): a request pinned a specific model_id (the caller wants
    # THAT model, not the router's auto-pick). This code is returned as DATA (never a
    # silent fallback to another model) when the pin cannot be honored: the model_id is
    # unknown to the studio registry, or the model exists but does not declare the
    # requested capability. A pinned model that DOES serve the capability but fails a
    # live gate (resolution / VRAM budget / license) surfaces the normal sharpened code
    # (RESOLUTION_UNSUPPORTED / VRAM_EXCEEDED / LICENSE_VIOLATION) with the single
    # rejected reason — so the caller always learns exactly why the pin didn't bind.
    # Deterministic (the same pin fails identically) -> the bus classifies it not-retryable.
    PINNED_MODEL_UNAVAILABLE = "pinned_model_unavailable"
    # DETERMINISTIC INFERENCE FAILURES (2026-08-21 incident: a studio render looped on
    # "[io_error] ... The size of tensor a (36) must match the size of tensor b (16)"
    # marked retryable — the SAME spec re-run fails identically, so a retry loop only
    # burns hours of GPU time). Neither is retryable (runners/studio_i2v's
    # _RETRYABLE_CODES does not list them).
    #   SHAPE_ERROR  — a tensor shape/size mismatch raised by the framework mid-denoise
    #                  (channels / frames / resolution disagree between spec, checkpoint
    #                  and pipeline class). Fix the spec or the checkpoint pairing.
    #   CONFIG_ERROR — a spec/checkpoint/pipeline pairing known to be wrong BEFORE or
    #                  AFTER the render (i2v checkpoint with no start image, a t2v
    #                  checkpoint asked to condition on an image, an API mismatch with
    #                  the installed diffusers). Fix the request or the install.
    SHAPE_ERROR = "shape_error"
    CONFIG_ERROR = "config_error"


@dataclass(frozen=True)
class StageError:
    """A failure, as a value. `context` is a frozen tuple of (key, value) pairs
    rather than a dict so the whole thing stays hashable and immutable."""
    code: ErrorCode
    message: str
    context: tuple[tuple[str, str], ...] = ()

    def with_context(self, **kv: str) -> "StageError":
        extra = tuple((str(k), str(v)) for k, v in kv.items())
        return StageError(self.code, self.message, self.context + extra)

    def __str__(self) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in self.context)
        return f"[{self.code.value}] {self.message}" + (f" ({ctx})" if ctx else "")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, _default: T) -> T:
        return self.value

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        return Ok(fn(self.value))

    def and_then(self, fn: "Callable[[T], Result[U, E]]") -> "Result[U, E]":
        return fn(self.value)


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> "T":
        # Unwrapping an Err is programmer error, not a recoverable runtime state.
        raise AssertionError(f"unwrap() on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def map(self, _fn: Callable[[T], U]) -> "Result[U, E]":
        return self  # type: ignore[return-value]

    def and_then(self, _fn: "Callable[[T], Result[U, E]]") -> "Result[U, E]":
        return self  # type: ignore[return-value]


Result = Union[Ok[T], Err[E]]


class RegistryError(RuntimeError):
    """Raised by validate_registry() at import. Programmer error, fail-loud."""


class ConfigError(RuntimeError):
    """Raised by load_env() when required environment wiring is missing (INV-5)."""
