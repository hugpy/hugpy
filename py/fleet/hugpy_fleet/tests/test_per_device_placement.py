"""CENTRAL per-GPU-device placement (operator 2026-09-25: "is it actually going
to take into account the gpus or self-manage on nvidia's end?").

On a-brain (4x 24 GiB = 96 GiB) the box-sum said a 30 GiB model "fits" and
admission/feasibility judged it against the sum. But a diffusers/comfy pipeline
lives on ONE card, so 30 GiB fits NO card there — while a 30 GiB GGUF tensor-
splits across the cards and does. These tests pin that engine-aware, per-device
behavior in central AND prove single-GPU boxes stay byte-identical.

Run:  venv/bin/python -m pytest tests/test_per_device_placement.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PROJECTS_HOME"] = tempfile.mkdtemp(prefix="hugpy-perdev-test-")
os.environ.setdefault("HUGPY_COMMS_DB", "off")
os.environ["HUGPY_COLD_HOLD_HEALTH"] = "off"

from hugpy_fleet.central import workers as W  # noqa: E402
from hugpy_engine.alloc_modes import feasible_modes  # noqa: E402

GIB = 2 ** 30


def _gpus(n, free_each=24 * GIB, total_each=24 * GIB):
    return [{"index": i, "memory_free": free_each, "memory_total": total_each}
            for i in range(n)]


# ── engine splittability ─────────────────────────────────────────────────────
def test_engine_splittable():
    assert W._engine_splittable("gguf") is True
    assert W._engine_splittable("llama_cpp") is True
    assert W._engine_splittable("diffusers") is False
    assert W._engine_splittable("transformers") is False


# ── engine-aware capacity: the 4x24 headline ─────────────────────────────────
def test_capacity_4x24_gguf_is_box_sum_diffusers_is_largest_card():
    w = {"id": "abrain", "gpus": _gpus(4)}
    assert W._worker_gpu_capacity_for_engine(w, "gguf", use="total") == 96 * GIB
    assert W._worker_gpu_capacity_for_engine(w, "diffusers", use="total") == 24 * GIB


def test_capacity_single_gpu_is_identical_for_both_engines():
    w = {"id": "ae", "gpus": _gpus(1)}
    assert W._worker_gpu_capacity_for_engine(w, "gguf", use="total") == 24 * GIB
    assert W._worker_gpu_capacity_for_engine(w, "diffusers", use="total") == 24 * GIB


def test_capacity_degrades_to_pooled_total_without_per_device_data():
    # No gpus[] -> the durable/box total path (never a manufactured smaller limit).
    w = {"id": "fresh", "gpu_total_bytes_known": 48 * GIB}
    assert W._worker_gpu_capacity_for_engine(w, "diffusers", use="total") == 48 * GIB


# ── the admission consequence: gpu-only feasibility on 4x24 ──────────────────
def test_30g_diffusers_loses_gpu_only_on_4x24_but_30g_gguf_keeps_it():
    w = {"id": "abrain", "gpus": _gpus(4)}
    ram = 128 * GIB
    diff_cap = W._worker_gpu_capacity_for_engine(w, "diffusers", use="total")  # 24G
    gguf_cap = W._worker_gpu_capacity_for_engine(w, "gguf", use="total")       # 96G
    # 30 GiB DIFFUSERS priced against the largest single card (24) -> gpu-only gone.
    diff_modes = feasible_modes("diffusers", 30 * GIB, diff_cap, ram)
    assert "gpu-only" not in diff_modes, diff_modes
    # 30 GiB GGUF priced against the box sum (96) -> gpu-only stays (it splits).
    gguf_modes = feasible_modes("gguf", 30 * GIB, gguf_cap, ram)
    assert "gpu-only" in gguf_modes, gguf_modes


def test_single_gpu_box_behavior_unchanged_for_diffusers():
    # On a lone 24 GiB card a 30 GiB diffusers model was never gpu-only feasible,
    # and still isn't — the per-device path didn't change single-GPU verdicts.
    w = {"id": "ae", "gpus": _gpus(1)}
    cap = W._worker_gpu_capacity_for_engine(w, "diffusers", use="total")
    assert "gpu-only" not in feasible_modes("diffusers", 30 * GIB, cap, 128 * GIB)
    assert "gpu-only" in feasible_modes("diffusers", 20 * GIB, cap, 128 * GIB)


# ── _device_spill_for: the wire the worker honors ────────────────────────────
def _store_with(worker):
    store = W.WorkerStore(path=os.path.join(
        tempfile.mkdtemp(prefix="hugpy-store-"), "workers.json"))
    store._load = lambda: {worker["id"]: worker}     # type: ignore[assignment]
    return store


def test_device_spill_pins_a_single_card_for_diffusers_on_multi_gpu(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 20 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "diffusers")
    w = {"id": "abrain", "caps": {"device_pin": 1}, "allocations": [],
         # card 0 nearly full, card 2 the smallest that still fits 20 GiB
         "gpus": [{"index": 0, "memory_free": 2 * GIB, "memory_total": 24 * GIB},
                  {"index": 1, "memory_free": 24 * GIB, "memory_total": 24 * GIB},
                  {"index": 2, "memory_free": 23 * GIB, "memory_total": 24 * GIB}]}
    spill = _store_with(w)._device_spill_for("abrain", "sd-xl")
    assert spill == {"main_gpu": 2}, spill                 # best-fit smallest that fits


def test_device_spill_splits_a_gguf_across_cards(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 30 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "gguf")
    w = {"id": "abrain", "caps": {"device_pin": 1}, "allocations": [],
         "gpus": _gpus(4)}
    spill = _store_with(w)._device_spill_for("abrain", "big-gguf")
    assert spill.get("main_gpu") == 0
    assert isinstance(spill.get("tensor_split"), list) and len(spill["tensor_split"]) == 2


def test_device_spill_empty_without_capability(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 20 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "diffusers")
    w = {"id": "abrain", "allocations": [], "gpus": _gpus(4)}   # no caps.device_pin
    assert _store_with(w)._device_spill_for("abrain", "sd-xl") == {}


def test_device_spill_empty_on_single_gpu(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 20 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "diffusers")
    w = {"id": "ae", "caps": {"device_pin": 1}, "allocations": [], "gpus": _gpus(1)}
    assert _store_with(w)._device_spill_for("ae", "sd-xl") == {}


def test_device_spill_none_when_no_card_fits_a_diffusers(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 30 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "diffusers")
    w = {"id": "abrain", "caps": {"device_pin": 1}, "allocations": [],
         "gpus": _gpus(4)}                                   # 30 fits no 24 card
    assert _store_with(w)._device_spill_for("abrain", "big-sd") == {}


def test_device_spill_anchors_to_the_resident_card(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda mk: 20 * GIB)
    monkeypatch.setattr(W, "_model_engine", lambda mk: "diffusers")
    # Model already resident on card 3 -> stay there (no churn), even though
    # best-fit would otherwise choose a different card.
    w = {"id": "abrain", "caps": {"device_pin": 1},
         "allocations": [{"kind": "ram", "model_key": "sd-xl", "gpu_index": 3}],
         "gpus": _gpus(4)}
    assert _store_with(w)._device_spill_for("abrain", "sd-xl") == {"main_gpu": 3}


# ── plain-script runner ─────────────────────────────────────────────────────
def _main() -> int:
    import types
    class _MP:
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for o, n, v in reversed(self._undo):
                setattr(o, n, v)
            self._undo = []
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for name, fn in tests:
        mp = _MP()
        try:
            if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(mp)
            else:
                fn()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {name}")
        finally:
            mp.undo()
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
