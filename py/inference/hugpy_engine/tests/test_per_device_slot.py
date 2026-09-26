"""Worker honors central's per-GPU device pin on the llama.cpp SLOT path.

Central rides the choice on the spill wire (HUGPY_MAIN_GPU / HUGPY_TENSOR_SPLIT).
The slot seam must thread it into the /load body the slot child consumes:
  * a SINGLE-card pin  -> body["gpu"] -> the child's CUDA_VISIBLE_DEVICES;
  * a tensor SPLIT     -> --tensor-split + --main-gpu on the llama-server argv,
    with CUDA_VISIBLE_DEVICES left unrestricted (all cards visible for the split).

Run:  python3 -m pytest tests/test_per_device_slot.py -q
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

slots = importlib.import_module("hugpy_engine.serve.slots")
sa = importlib.import_module("hugpy_engine.serve.slot_agent")


def _clear():
    for k in ("HUGPY_MAIN_GPU", "HUGPY_TENSOR_SPLIT", "HUGPY_GPU_MEM_GIB",
              "HUGPY_CPU_MEM_GIB", "HUGPY_N_CPU_MOE", "HUGPY_ALLOC_MODE",
              "HUGPY_N_GPU_LAYERS", "DEFAULT_LLAMA_THREADS"):
        os.environ.pop(k, None)


# ── env -> /load body threading ──────────────────────────────────────────────
def test_env_request_opts_threads_main_gpu_to_the_slot_child():
    _clear()
    os.environ["HUGPY_MAIN_GPU"] = "2"
    try:
        opts = slots.env_request_opts({})
        assert opts.get("gpu") == "2"          # -> CUDA_VISIBLE_DEVICES on the child
        assert opts.get("main_gpu") == "2"     # -> --main-gpu when splitting
        assert "tensor_split" not in opts
    finally:
        _clear()


def test_env_request_opts_threads_tensor_split():
    _clear()
    os.environ["HUGPY_MAIN_GPU"] = "0"
    os.environ["HUGPY_TENSOR_SPLIT"] = "0.5,0.5"
    try:
        opts = slots.env_request_opts({})
        assert opts.get("tensor_split") == "0.5,0.5"
        assert opts.get("gpu") == "0"
    finally:
        _clear()


def test_env_request_opts_no_device_keys_when_unset():
    _clear()
    opts = slots.env_request_opts({})
    assert "gpu" not in opts and "main_gpu" not in opts and "tensor_split" not in opts


# ── slot_agent parse helpers ─────────────────────────────────────────────────
def test_parse_tensor_split_forms():
    assert sa._parse_tensor_split("0.5,0.5") == [0.5, 0.5]
    assert sa._parse_tensor_split([0.25, 0.75]) == [0.25, 0.75]
    assert sa._parse_tensor_split(None) is None
    assert sa._parse_tensor_split("") is None
    assert sa._parse_tensor_split("bogus") is None


def test_as_int_or_none():
    assert sa._as_int_or_none("3") == 3
    assert sa._as_int_or_none(3) == 3
    assert sa._as_int_or_none(None) is None
    assert sa._as_int_or_none("") is None
    assert sa._as_int_or_none("x") is None


def test_split_detection_needs_two_nonzero_shares():
    # A single-card pin expressed as a share vector is NOT a split (one nonzero) —
    # it must not suppress the CUDA_VISIBLE_DEVICES single-card pin.
    def _is_split(ts):
        parsed = sa._parse_tensor_split(ts)
        return parsed is not None and sum(1 for x in parsed if x and x > 0) >= 2
    assert _is_split("0.5,0.5") is True
    assert _is_split("0,0,1,0") is False
    assert _is_split([1.0]) is False


# ── the argv actually carries the split flags ────────────────────────────────
def test_build_cmd_emits_tensor_split_and_main_gpu():
    src = inspect.getsource(sa._build_cmd)
    assert "--tensor-split" in src
    assert "--main-gpu" in src
    # gated on server support so an older binary degrades, never crashes
    assert "_server_supports_flag" in src


def test_load_accepts_tensor_split_and_main_gpu():
    params = inspect.signature(sa.Slot.load).parameters
    assert "tensor_split" in params and "main_gpu" in params


# ── plain-script runner ─────────────────────────────────────────────────────
def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
