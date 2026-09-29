"""KNOWN-GOOD CONTRACT — model-key identity in central routing and the
worker's resident match (fleet side).

Operator rule (board 2026-09-29): ``X-GGUF`` and ``X`` are one key (a format
variant); any other differing name token (``-Distill``, a quant marker, a
finetune tag…) is a DIFFERENT model that must not be loaded or served unless
that exact key was called — same file on the worker's drive or not.

Incident: eviction test S3 / F7 (notes/eviction-test-2026-09-29.md): a request
for ``Qwen3.8-9B-GGUF`` was served on ae-worker by the resident
``Qwen3.8-9B-Distill-GGUF`` (same ``Qwen3.8-9B-Q8_0.gguf``), evicting Coder-Next.

Source under test: hugpy_fleet/central/workers.py (_serveable_match,
_resident_on, _allocated_on, _on_disk_match, _routing_rank,
_canonical_registry_key), hugpy_fleet/worker/agent.py (_worker_has_resident,
_partition_residents, _subject_resident_vram_bytes). Deterministic: worker
records are literals, every live seam is stubbed on the module.
"""
from __future__ import annotations

import importlib

import pytest

W = importlib.import_module("hugpy_fleet.central.workers")
A = importlib.import_module("hugpy_fleet.worker.agent")
gen_gate = importlib.import_module("hugpy_fleet.worker.gen_gate")

GIB = 1 << 30
DISTILL = "Qwen3.8-9B-Distill-GGUF"
PLAIN = "Qwen3.8-9B-GGUF"


class _State:
    pass


def _w(wid, *, loaded=(), allocations=(), grants=None, local=(), wildcard=True,
       last_picked=0.0):
    return {
        "id": wid, "name": wid, "url": f"http://{wid}:9100",
        "loaded_models": list(loaded), "allocations": list(allocations),
        "grants": dict(grants or {}), "models_local": list(local),
        "_wildcard_catch": wildcard,
        "gpus": [{"memory_total": 24 * GIB, "memory_free": 20 * GIB}],
        "last_picked": last_picked,
    }


@pytest.fixture(autouse=True)
def _no_blocklist(monkeypatch):
    monkeypatch.setattr(W, "_model_blocked", lambda m: False)


def _rank(w, mk, **kw):
    return W._routing_rank(w, mk, W._match_keys(mk), starred=False, **kw)


def test_routing_never_credits_a_worker_with_a_key_it_holds_under_another_name():
    """INVARIANT (S3 / F7): a worker resident/allocated/on-disk for
    ``Qwen3.8-9B-Distill-GGUF`` is NOT resident, allocated or on-disk for
    ``Qwen3.8-9B-GGUF`` — the rank's residency, allocation and on-disk terms
    all read 1 (cold), exactly like an empty box.
    Established: operator rule 2026-09-29 (_serveable_match identity)."""
    wanted = W._match_keys(PLAIN)
    holder = _w("ae", loaded=[DISTILL], local=[DISTILL],
                allocations=[{"model_key": DISTILL, "kind": "slot", "healthy": True}])
    cold = _w("cold")
    assert W._serveable_match(PLAIN, wanted, [DISTILL]) is False
    assert W._resident_on(holder, PLAIN, wanted) is False
    assert W._allocated_on(holder, PLAIN, wanted) is False
    assert W._on_disk_match(holder, PLAIN, wanted) is False
    assert _rank(holder, PLAIN)[3:5] == (1, 1) == _rank(cold, PLAIN)[3:5]
    assert _rank(holder, PLAIN, feasible_ordering=True)[7] == 1   # on-disk term: cold
    # and the mirror: holding the plain key earns nothing for the Distill key
    assert W._resident_on(_w("b", loaded=[PLAIN]), DISTILL, W._match_keys(DISTILL)) is False


def test_routing_credits_the_same_canonical_key_across_the_gguf_suffix():
    """INVARIANT: ``X`` and ``X-GGUF`` are ONE key, so residency, allocation
    and on-disk presence under either spelling credit a request for the other
    (the scorer still credits file presence for the SAME canonical key), and
    the k67 ``Owner~X`` tail unification is preserved.
    Established: operator rule 2026-09-29."""
    wanted = W._match_keys(PLAIN)
    w = _w("ae", loaded=["Qwen3.8-9B"], local=["Qwen3.8-9B"],
           allocations=[{"model_key": "Qwen3.8-9B", "kind": "ram"}])
    assert W._resident_on(w, PLAIN, wanted) and W._allocated_on(w, PLAIN, wanted)
    assert W._on_disk_match(w, PLAIN, wanted) is True
    assert _rank(w, PLAIN)[3:5] == (0, 0)
    assert _rank(w, PLAIN, feasible_ordering=True)[7] == 0
    assert _rank(w, PLAIN) < _rank(_w("cold"), PLAIN)
    assert W._on_disk_match(_w("c", local=[PLAIN]), "Qwen3.8-9B", W._match_keys("Qwen3.8-9B"))
    assert W._on_disk_match(_w("d", local=["Qwen~Qwen3-Coder-Next-GGUF"]),
                            "Qwen3-Coder-Next-GGUF", W._match_keys("Qwen3-Coder-Next-GGUF"))
    assert W._on_disk_match(_w("e", local=["Qwen~Qwen3-Coder-Next-GGUF"]),
                            "Qwen3-Coder-Next", W._match_keys("Qwen3-Coder-Next"))


def test_central_size_facts_resolve_across_the_gguf_suffix_only(monkeypatch):
    """INVARIANT: ``_canonical_registry_key`` maps ``X`` <-> ``X-GGUF`` (and the
    ``Owner~X`` tail) to the row on record, never to a sibling; an unknown key
    stays unknown. Established: operator rule 2026-09-29."""
    mc = importlib.import_module("hugpy_engine.config.models.models_config")
    monkeypatch.setattr(mc, "get_models_dict",
                        lambda dict_return=True: {PLAIN: {}, DISTILL: {}})
    assert W._canonical_registry_key("Qwen3.8-9B") == PLAIN
    assert W._canonical_registry_key("Qwen3.8-9B-Distill") == DISTILL
    assert W._canonical_registry_key("empero-ai~Qwen3.8-9B-GGUF") == PLAIN
    assert W._canonical_registry_key("Qwen3.8-9B-heretic") == "Qwen3.8-9B-heretic"
    assert W._canonical_registry_key(PLAIN) == PLAIN


def test_worker_resident_check_is_key_identity(monkeypatch):
    """INVARIANT (S3 / F7): the worker's "already have a live copy" test never
    answers a request for key A from resident B: Distill resident -> the plain
    key is NOT resident; ``X`` resident -> ``X-GGUF`` IS.
    Established: operator rule 2026-09-29 (_worker_has_resident)."""
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [DISTILL])
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: set())
    ext = importlib.import_module("hugpy_fleet.worker.external_residents")
    monkeypatch.setattr(ext, "keys", lambda: [])
    pidreg = importlib.import_module("hugpy_fleet.worker.pid_registry")
    monkeypatch.setattr(pidreg, "snapshot_for_heartbeat", lambda: {"models": []})
    assert A._worker_has_resident(PLAIN) is False
    assert A._worker_has_resident(DISTILL) is True
    assert A._worker_has_resident("Qwen3.8-9B-Distill") is True
    monkeypatch.setattr(A, "loaded_model_keys", lambda: [])
    monkeypatch.setattr(A, "_slot_occupants", lambda strict=False: {"Qwen3.8-9B"})
    assert A._worker_has_resident(PLAIN) is True
    assert A._worker_has_resident(DISTILL) is False


@pytest.fixture
def residents(monkeypatch):
    rows = [{"model_key": DISTILL, "vram_bytes": 10 * GIB, "host_mode": "slot", "alive": True},
            {"model_key": "Qwen3.8-9B", "vram_bytes": 2 * GIB, "host_mode": "slot", "alive": True}]
    monkeypatch.setattr(A, "_vram_residents", lambda s: list(rows))
    monkeypatch.setattr(A, "_busy_slot_models", lambda: set())
    monkeypatch.setattr(A, "_queued_ahead_of", lambda subject: set())
    monkeypatch.setattr(A, "_residency", lambda mk: "on-demand")
    monkeypatch.setattr(A, "_comfy_busy_reason", lambda s: None)
    monkeypatch.setattr(A, "detect_gpus", lambda: [])
    monkeypatch.setattr(A, "_target_device_index", lambda: None)
    monkeypatch.setattr(gen_gate, "in_flight", lambda mk: 0)
    return rows


def test_admission_partition_and_subject_credit_use_key_identity(residents):
    """INVARIANT: for an admission of ``Qwen3.8-9B-GGUF`` the resident
    ``Qwen3.8-9B`` IS the subject (excluded from eviction, its bytes credited)
    and ``Qwen3.8-9B-Distill-GGUF`` is a DISTINCT evictable resident whose bytes
    are never credited to the subject — same file or not.
    Established: operator rule 2026-09-29 (_partition_residents,
    _subject_resident_vram_bytes)."""
    cands, prot = A._partition_residents(_State(), PLAIN)
    assert [r["model_key"] for r in cands] == [DISTILL] and prot == []
    assert A._subject_resident_vram_bytes(_State(), PLAIN) == 2 * GIB
    cands, _ = A._partition_residents(_State(), DISTILL)
    assert [r["model_key"] for r in cands] == ["Qwen3.8-9B"]
    assert A._subject_resident_vram_bytes(_State(), DISTILL) == 10 * GIB
    assert A._subject_resident_vram_bytes(_State(), "Qwen3.8-9B-heretic-GGUF") == 0
