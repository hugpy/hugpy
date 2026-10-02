"""4-bit re-price touches the weights only (2026-10-02): bitsandbytes does not
quantize the KV cache. The old whole-need multiply priced MN-GRAND's 3.2 GiB of
KV as 0.96 GiB and printed full-precision weights in refusals."""
from hugpy_engine.alloc_modes import BNB_4BIT_SIZE_RATIO
from hugpy_fleet.worker import agent as A

G = 1 << 30


def test_kv_is_not_quantized():
    det = {"weights": int(50.2 * G), "kv": int(3.2 * G), "total": int(53.4 * G),
           "calibration_correction": 1.0}
    need, out, ratio = A._bnb_reprice(det)
    w4 = int(int(50.2 * G) * BNB_4BIT_SIZE_RATIO)
    assert ratio == float(BNB_4BIT_SIZE_RATIO)
    assert out["weights"] == w4 and out["weights_full_precision"] == int(50.2 * G)
    assert out["kv"] == int(3.2 * G)
    assert need == out["total"] == w4 + int(3.2 * G)
    assert det["weights"] == int(50.2 * G)          # caller's dict untouched


def test_correction_still_applies_to_the_repriced_total():
    det = {"weights": 10 * G, "kv": 2 * G, "total": 12 * G, "calibration_correction": 1.1}
    need, out, _ = A._bnb_reprice(det)
    assert need == int((int(10 * G * BNB_4BIT_SIZE_RATIO) + 2 * G) * 1.1)


def test_no_weights_is_a_noop():
    det = {"weights": None, "kv": 0, "total": None}
    need, out, ratio = A._bnb_reprice(det)
    assert need is None and ratio is None and out is det
