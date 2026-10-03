"""Opt-in context yield (operator 2026-10-02): a loaded model with a range and
ctx_yield comes back at a smaller ctx instead of staying evicted."""
from hugpy_fleet.worker import agent as A

G = 1 << 30


def test_yield_pct():
    # 4 GiB weights + 4 GiB KV at 50%; 6 GiB of room -> 2 GiB KV -> 25%
    assert A._yield_pct(4 * G, 4 * G, 50, 10, 6 * G) == 25
    assert A._yield_pct(4 * G, 4 * G, 50, 30, 6 * G) is None      # min not reachable
    assert A._yield_pct(4 * G, 4 * G, 50, 0, 20 * G) == 50       # fits whole: no shrink needed
    assert A._yield_pct(4 * G, 4 * G, 50, None, 6 * G) is None   # no range: no yield
    assert A._yield_pct(4 * G, 4 * G, 50, 0, 3 * G) is None      # not even the weights


def test_only_opted_in_gguf_residents_yield(monkeypatch):
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "ctx_yield_db", {"Y"})
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "ctx_min_pct_db", {"Y": 10, "N": 10})
    monkeypatch.setattr(A, "_model_framework", lambda mk: "gguf")
    monkeypatch.setattr(A, "_incoming_need_detail", lambda mk: {"weights": 4 * G, "kv": 4 * G, "ctx_pct": 50})
    assert A._plan_ctx_yields("S", ["N", "Y"], 6 * G) == [("Y", 25)]
    monkeypatch.setattr(A, "_model_framework", lambda mk: "transformers")
    assert A._plan_ctx_yields("S", ["Y"], 6 * G) == []
