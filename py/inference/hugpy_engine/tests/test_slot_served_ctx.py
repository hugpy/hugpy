"""The slot /load carries the ctx the worker's admission priced (2026-10-02):
the child is a separate process and cannot read the admission ticket, so it
guessed its own -c (MN-GRAND priced 20480, launched 29696)."""
from hugpy_engine.serve import slots


def test_priced_ctx_rides_the_load_body(monkeypatch):
    monkeypatch.setattr(slots, "_SERVED_CTX", lambda mk: 20480 if mk == "M" else None)
    assert slots._with_served_ctx({"model_key": "M"}, "M")["ctx"] == 20480
    assert "ctx" not in slots._with_served_ctx({"model_key": "X"}, "X")


def test_explicit_ctx_wins_and_no_resolver_is_a_noop(monkeypatch):
    monkeypatch.setattr(slots, "_SERVED_CTX", lambda mk: 20480)
    assert slots._with_served_ctx({"model_key": "M", "ctx": 4096}, "M")["ctx"] == 4096
    monkeypatch.setattr(slots, "_SERVED_CTX", None)
    assert "ctx" not in slots._with_served_ctx({"model_key": "M"}, "M")


def test_resolver_error_falls_back_to_the_child(monkeypatch):
    def boom(mk):
        raise RuntimeError("db down")
    monkeypatch.setattr(slots, "_SERVED_CTX", boom)
    assert "ctx" not in slots._with_served_ctx({"model_key": "M"}, "M")
