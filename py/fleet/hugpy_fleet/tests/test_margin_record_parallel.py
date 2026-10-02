"""_margin_record (2026-10-02): the one delta -> margin arithmetic, shared by
the heartbeat learner and calibrate, nets out KV x llama-server --parallel."""
from hugpy_fleet.worker import agent as A

G = 1 << 30


def _patch(monkeypatch):
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(A, "_kv_bytes_at_ctx", lambda mk, ctx, cfg=None: 2 * G)
    monkeypatch.setattr(A, "_device_class", lambda i=None: "RTX 3090")
    monkeypatch.setattr(A, "_persist_weights_margins", lambda: None)
    monkeypatch.setitem(A._WEIGHTS_MARGINS, "M", None)
    A._WEIGHTS_MARGINS.pop("M", None)


def test_extra_sequences_are_kv_not_weights(monkeypatch):
    _patch(monkeypatch)
    # 10 GiB file, 3 sequences x 2 GiB KV, 16 GiB measured -> weights 10 GiB
    rec = A._margin_record("M", 16 * G, 10 * G, "m.gguf", ctx=4096, parallel=3, origin="calibrate")
    assert rec["margin"] == 1.0 and rec["kv_measured_bytes"] == 6 * G and rec["parallel"] == 3


def test_single_sequence_default(monkeypatch):
    _patch(monkeypatch)
    rec = A._margin_record("M", 11 * G, 10 * G, "m.gguf", ctx=4096)
    assert rec["margin"] == 0.9 and rec["origin_kind"] == "heartbeat"


def test_implausible_ratio_not_recorded(monkeypatch):
    _patch(monkeypatch)
    assert A._margin_record("M", 4 * G, 10 * G, "m.gguf", ctx=4096) is None
    assert "M" not in A._WEIGHTS_MARGINS
