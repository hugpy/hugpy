"""512 MiB compute cushion only for a NEW process (operator ruling 2026-10-02):
a llama-server child (GGUF) pays it, a torch load inside the worker does not."""
from hugpy_fleet.worker import agent as A

G = 1 << 30
MIB = 1 << 20


def _fw(monkeypatch, fw):
    monkeypatch.delenv("HUGPY_VRAM_CEILING_FRAC", raising=False)
    monkeypatch.delenv("HUGPY_VRAM_CEILING_CUSHION_GIB", raising=False)
    monkeypatch.setattr(A, "_external_vram_floor_bytes", lambda: 0)
    monkeypatch.setattr(A, "_model_framework", lambda mk: fw)


def test_gguf_pays_the_cushion(monkeypatch):
    _fw(monkeypatch, "gguf")
    assert A._vram_ceiling_reserve_bytes(24 * G, "M") == 512 * MIB


def test_in_worker_transformers_load_pays_nothing(monkeypatch):
    _fw(monkeypatch, "transformers")
    assert A._vram_ceiling_reserve_bytes(24 * G, "M") == 0
    assert A._fit_policy(24 * G, "M").ceiling_reserve_bytes == 0


def test_unknown_model_or_framework_stays_conservative(monkeypatch):
    _fw(monkeypatch, None)
    assert A._vram_ceiling_reserve_bytes(24 * G, "M") == 512 * MIB
    assert A._vram_ceiling_reserve_bytes(24 * G) == 512 * MIB


def test_explicit_frac_override_is_verbatim(monkeypatch):
    _fw(monkeypatch, "transformers")
    monkeypatch.setenv("HUGPY_VRAM_CEILING_FRAC", "0.9")
    assert A._vram_ceiling_reserve_bytes(10 * G, "M") == int(10 * G * (1 - 0.9))
