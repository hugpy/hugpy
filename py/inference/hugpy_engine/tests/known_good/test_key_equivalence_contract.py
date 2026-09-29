"""KNOWN-GOOD CONTRACT — model-key identity on the worker's serving seams
(engine side): the slot seat match, the cached-runner seat check, the
in-process registry resolve.

Operator rule (board 2026-09-29): a ``-GGUF`` suffix versus no suffix are
NECESSARILY the same model — a format variant of one key. Any other differing
name token (``-Distill``, a quant marker, a finetune tag…) makes a DIFFERENT
model that must not be loaded or served unless that exact key was called,
even when it is backed by the same file on the worker's drive. Same-file
residency never implies key equivalence.

Incident: eviction test S3 / F7 (notes/eviction-test-2026-09-29.md) — a
request for ``Qwen3.8-9B-GGUF`` routed to ae-worker was resolved to the
resident ``Qwen3.8-9B-Distill-GGUF`` (both ``Qwen3.8-9B-Q8_0.gguf``), which
evicted Coder-Next to seat it, and the client got a 200 for another key.

Source under test: hugpy_engine/fit/types.py (canonical_key, key_equivalent),
hugpy_engine/serve/slots.py (seat_serves), hugpy_engine/llama/runners/get.py
(_slot_still_holds), hugpy_engine/config/main.py (_resolve_model_key).
Deterministic: no slot child, no HTTP (httpx.get is stubbed), no registry.
"""
from __future__ import annotations

import importlib

import httpx
import pytest

fit = importlib.import_module("hugpy_engine.fit")
slots = importlib.import_module("hugpy_engine.serve.slots")
G = importlib.import_module("hugpy_engine.llama.runners.get")
CM = importlib.import_module("hugpy_engine.config.main")

DISTILL = "Qwen3.8-9B-Distill-GGUF"
PLAIN = "Qwen3.8-9B-GGUF"


def test_seat_serves_is_key_equivalence_not_file_identity():
    """INVARIANT: SlotPool.endpoint_for's "already serving" step reuses a seat
    ONLY for the same canonical key. The Distill seat never answers for the
    plain key (S3 shape); a seat under ``X`` answers for ``X-GGUF``.
    Established: operator rule 2026-09-29 (slots.seat_serves)."""
    assert slots.seat_serves({"model_key": DISTILL}, PLAIN) is False
    assert slots.seat_serves({"model_key": PLAIN}, DISTILL) is False
    assert slots.seat_serves({"model_key": "Qwen3.8-9B"}, PLAIN) is True
    assert slots.seat_serves({"model_key": PLAIN}, "Qwen3.8-9B") is True
    assert slots.seat_serves({"model_key": None}, PLAIN) is False
    assert slots.seat_serves({}, PLAIN) is False and slots.seat_serves(None, PLAIN) is False


class _HttpRunner:
    base_url = "http://127.0.0.1:9201"
    _slot_backed = True
    llm = None


def test_cached_runner_seat_check_drops_a_seat_holding_a_different_key(monkeypatch):
    """INVARIANT: ``_slot_still_holds`` treats the seat's ``model_key`` as
    identity: Distill under a request for the plain key is a POSITIVE
    mismatch (False -> the cache entry is dropped and re-resolved), while
    ``X`` vs ``X-GGUF`` is the same seat and proceeds to the contract check.
    The resident-preferring canonical resolve cannot launder the mismatch.
    Established: operator rule 2026-09-29."""
    held = {"model_key": DISTILL}

    class _Resp:
        def json(self):
            return {"slot_id": 1, "healthy": True, **held}
    monkeypatch.setattr(httpx, "get", lambda url, timeout=1.5: _Resp())
    monkeypatch.setattr(G, "get_model_config", lambda mk, prefer=None: (_ for _ in ()).throw(KeyError(mk)))
    assert G._slot_still_holds(_HttpRunner(), PLAIN) is False
    # even when the resolver would "prefer" the resident, identity decides
    monkeypatch.setattr(G, "get_model_config",
                        lambda mk, prefer=None: type("C", (), {"model_key": DISTILL})())
    assert G._slot_still_holds(_HttpRunner(), PLAIN) is False
    # same seat, other spelling: passes the identity check into the contract check
    held["model_key"] = "Qwen3.8-9B"
    monkeypatch.setattr(slots, "alloc_mismatch", lambda st, req, src: None)
    monkeypatch.setattr(slots, "status_satisfies_opts", lambda st, opts: True)
    for leak in ("HUGPY_N_GPU_LAYERS", "HUGPY_N_CPU_MOE", "HUGPY_GGUF_FILE", "HUGPY_ALLOC_MODE"):
        monkeypatch.delenv(leak, raising=False)
    assert G._slot_still_holds(_HttpRunner(), PLAIN) is True


def test_in_process_registry_resolve_crosses_only_the_gguf_suffix():
    """INVARIANT: ``config.main._resolve_model_key`` (get_model_config /
    serve_endpoint's canonicalisation) resolves ``X`` <-> ``X-GGUF`` and never
    ``X-GGUF`` -> ``X-Distill-GGUF``; the owner-qualified row is still found
    through its bare tail. Established: operator rule 2026-09-29."""
    reg = {PLAIN: 1, DISTILL: 2}
    assert CM._resolve_model_key("Qwen3.8-9B", reg) == PLAIN
    assert CM._resolve_model_key("Qwen3.8-9B-Distill", reg) == DISTILL
    assert CM._resolve_model_key(PLAIN, {DISTILL: 2}) is None
    assert CM._resolve_model_key("Qwen3.8-9B", {DISTILL: 2}) is None
    assert CM._resolve_model_key("Qwen3-Coder-Next", {"Qwen~Qwen3-Coder-Next-GGUF": 1}) \
        == "Qwen~Qwen3-Coder-Next-GGUF"
    # a resident spelling is preferred among equivalent candidates, never a sibling
    assert CM._resolve_model_key("Qwen3.8-9B", {PLAIN: 1, "AtomicChat~Qwen3.8-9B-GGUF": 2},
                                 prefer=["AtomicChat~Qwen3.8-9B-GGUF"]) == "AtomicChat~Qwen3.8-9B-GGUF"
