"""Transformers 'auto' ctx = the largest ctx that fits (2026-10-02 test fire:
models refused at a 131k-262k loader default whose weights fit)."""
from hugpy_fleet.worker import agent as A

G = 1 << 30


def test_transformers_auto_ctx_is_bounded_by_the_fit(monkeypatch):
    geo = {"n_kv_heads": 8, "head_dim": 128, "n_layers": 48, "block_count": 48,
           "ctx_train": 262144, "dtype": "bfloat16"}
    monkeypatch.setattr(A, "_fresh_admission_ticket", lambda mk: None)
    monkeypatch.setattr(A, "_ctx_pct", lambda mk: None)
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: 262144)
    monkeypatch.setattr(A, "_model_kv_geometry", lambda mk, cfg=None: geo)
    monkeypatch.setattr(A, "_fit_weights_and_corr", lambda mk: (10 * G, None))
    monkeypatch.setattr(A, "_total_vram_bytes", lambda: 24 * G)
    monkeypatch.setattr(A, "_free_vram_bytes", lambda: 23 * G)
    out = A._effective_ctx("T", {"framework": "transformers"})
    assert out["source"] == "loader-default"
    assert 0 < out["ctx"] < 262144
    kv = A._kv_at_ctx(geo, out["ctx"])
    assert 10 * G + kv <= 23 * G


def test_engine_without_kv_geometry_keeps_model_max(monkeypatch):
    monkeypatch.setattr(A, "_fresh_admission_ticket", lambda mk: None)
    monkeypatch.setattr(A, "_ctx_pct", lambda mk: None)
    monkeypatch.setattr(A, "_model_max_ctx", lambda mk, cfg=None: 4096)
    monkeypatch.setattr(A, "_model_kv_geometry", lambda mk, cfg=None: {})
    out = A._effective_ctx("D", {"framework": "transformers"})
    assert out == {"ctx": 4096, "pct": None, "max": 4096, "source": "model-max"}
