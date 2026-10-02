"""Slot concurrency is priced from the served file's real per-sequence KV
(2026-10-02): a flat 160 KiB/token guess gave MN-GRAND (324 KiB/token) N=3 and
llama-server OOM'd allocating 3x the KV the gate had admitted."""
import hugpy_engine.spill as sp
from hugpy_engine.serve import slot_agent as sa

GIB = 1 << 30
PER_TOK = 331_776                       # MN-GRAND f16 KV bytes/token


def _setup(monkeypatch, free_gib):
    monkeypatch.delenv("HUGPY_SLOT_PARALLEL", raising=False)
    monkeypatch.setattr(sp, "free_vram_bytes", lambda: int(free_gib * GIB))
    monkeypatch.setattr(sp, "_gguf_kv_geometry", lambda p: {"gguf_path": p})
    monkeypatch.setattr(sp, "kv_bytes_for_geo",
                        lambda geo, ctx, dtype_bytes=2.0: int(ctx * PER_TOK * dtype_bytes / 2))


def test_one_sequence_when_a_second_would_not_fit(monkeypatch):
    _setup(monkeypatch, 23.3)
    n = sa._slot_parallel(ctx=20480, model_bytes=int(13.37 * GIB), path="/x.gguf")
    assert n == 1                       # old guess: 3 -> 19,440 MiB KV -> OOM


def test_extras_only_when_whole_sequences_fit(monkeypatch):
    _setup(monkeypatch, 48.0)
    n = sa._slot_parallel(ctx=30720, model_bytes=int(13.37 * GIB), path="/x.gguf")
    kv = 30720 * PER_TOK
    assert n >= 2 and int(13.37 * GIB) + n * kv <= 48 * GIB


def test_quantized_cache_halves_the_sequence(monkeypatch):
    _setup(monkeypatch, 23.3)
    f16 = sa._slot_parallel(ctx=8192, model_bytes=int(13.37 * GIB), path="/x.gguf")
    q8 = sa._slot_parallel(ctx=8192, model_bytes=int(13.37 * GIB), path="/x.gguf",
                           kv_cache_type="q8_0")
    assert q8 >= f16
