"""4-bit margins (2026-10-02): a bitsandbytes 4-bit residency learns its OWN
margin under <key>#bnb-4bit against the 4-bit file figure; the full-precision
record is untouched, and the 4-bit re-price uses the measured margin."""
from hugpy_engine.alloc_modes import BNB_4BIT_SIZE_RATIO
from hugpy_fleet.worker import agent as A

G = 1 << 30
FILE = int(50.2 * G)


def _patch(monkeypatch):
    monkeypatch.setattr(A, "_model_framework", lambda mk: "transformers")
    monkeypatch.setattr(A, "_device_class", lambda i=None: "RTX 3090")
    monkeypatch.setattr(A, "_persist_weights_margins", lambda: None)
    for k in ("M", "M#bnb-4bit"):
        monkeypatch.setitem(A._WEIGHTS_MARGINS, k, None)
        A._WEIGHTS_MARGINS.pop(k, None)


def test_bnb_file_figure_and_tag():
    w4, tag = A._bnb_file(FILE, None)
    assert w4 == int(FILE * float(BNB_4BIT_SIZE_RATIO)) and tag == "bnb-4bit"
    assert A._bnb_file(FILE, "m.safetensors")[1] == "m.safetensors+bnb-4bit"


def test_4bit_measurement_is_its_own_record(monkeypatch):
    _patch(monkeypatch)
    A._WEIGHTS_MARGINS["M"] = {"model_key": "M", "file": None, "file_bytes": FILE,
                              "backend": "transformers", "margin": 1.08, "samples": 2}
    w4, tag = A._bnb_file(FILE, None)
    delta = int(w4 * 1.05)
    rec = A._margin_record(A._bnb_margin_key("M"), delta, w4, tag, origin="calibrate",
                           framework="transformers")
    assert rec is not None and abs(rec["margin"] - 1.05) < 0.01 and rec["kv_measured_bytes"] == 0
    assert A._WEIGHTS_MARGINS["M"]["margin"] == 1.08          # full precision untouched
    assert A._weights_margin_for("M", None)["margin"] == 1.08
    assert A._weights_margin_for("M#bnb-4bit", tag)["source"] == "measured"


def test_reprice_uses_measured_4bit_margin(monkeypatch):
    _patch(monkeypatch)
    w4, tag = A._bnb_file(FILE, None)
    A._WEIGHTS_MARGINS["M#bnb-4bit"] = {"model_key": "M#bnb-4bit", "file": tag, "file_bytes": w4,
                                       "backend": "transformers", "margin": 1.05, "samples": 1}
    det = {"weights": int(FILE * 1.15), "kv": 3 * G, "total": int(FILE * 1.15) + 3 * G,
           "calibration_correction": 1.0, "weights_file_bytes": FILE, "weights_file": None}
    need, out, _ = A._bnb_reprice(det, "M")
    assert out["weights"] == int(w4 * 1.05) and out["bnb_4bit_margin"] == 1.05
    assert need == out["weights"] + 3 * G


def test_reprice_without_measurement_keeps_ratio_path(monkeypatch):
    _patch(monkeypatch)
    det = {"weights": int(FILE * 1.15), "kv": 0, "total": int(FILE * 1.15),
           "calibration_correction": 1.0, "weights_file_bytes": FILE, "weights_file": None}
    _need, out, _ = A._bnb_reprice(det, "M")
    assert out["weights"] == int(int(FILE * 1.15) * float(BNB_4BIT_SIZE_RATIO))
    assert "bnb_4bit_margin" not in out


def test_central_never_crosses_4bit_and_full_records(monkeypatch):
    from hugpy_fleet.central import workers as W
    rec = {"margin": 1.05, "samples": 1, "measured_at": 1}
    full = {"margin": 1.10, "samples": 3, "measured_at": 1}
    monkeypatch.setattr(W.worker_store, "raw_all",
                        lambda: [{"id": "w", "name": "w", "weights_margins": {"M": full, "M#bnb-4bit": rec}}])
    assert W.weights_margin_for("M")["margin"] == 1.10
    assert W.weights_margin_for("M#bnb-4bit")["margin"] == 1.05
