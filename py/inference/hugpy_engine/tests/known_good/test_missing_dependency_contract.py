"""KNOWN-GOOD CONTRACT — a PEFT adapter with no base in the store is a FINAL,
no-spend refusal (load class ``missing_dependency``), never a retried load.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "evict + fit / allocation").
Source under test: hugpy_engine/peft_adapters.py (resolve_adapter_pair ->
AdapterBaseUnavailable carrying ``fix``), hugpy_engine/generate/config.py
(adapter_base_refusal -> MissingDependencyFailure), hugpy_engine/serve/
load_failure.py (MISSING_DEPENDENCY / FINAL_LOAD_CLASSES / load_failure_of),
hugpy_engine/resolvers/remote.py (_is_permanent_load_error /
_is_final_load_class / _is_cacheable_load_verdict).

Deterministic: a synthetic adapter dir under tmp_path, an empty store root; no
torch, no worker, no network.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

pa = importlib.import_module("hugpy_engine.peft_adapters")
LF = importlib.import_module("hugpy_engine.serve.load_failure")
remote = importlib.import_module("hugpy_engine.resolvers.remote")

BASE = "unsloth/Llama-3.2-3B-Instruct"
KEY = "veeraragavan410~Llama-3.2-3B-sentiment"


@pytest.fixture
def orphan_adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / pa.ADAPTER_CONFIG_NAME).write_text(json.dumps(
        {"base_model_name_or_path": BASE, "peft_type": "LORA"}))
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    store = tmp_path / "store"
    store.mkdir()
    return str(d), str(store)


def _raise(orphan_adapter):
    d, store = orphan_adapter
    with pytest.raises(pa.AdapterBaseUnavailable) as ei:
        pa.resolve_adapter_pair(d, root=store)
    return ei.value


def test_adapter_refusal_is_structured_final_and_carries_base_and_fix(orphan_adapter):
    """INVARIANT: the live RuntimeError ("Llama-3.2-3B-sentiment: PEFT adapter
    (base 'unsloth/Llama-3.2-3B-Instruct') — the adapter is on disk but its
    base model is NOT in this store ...") is now a MissingDependencyFailure:
    load class ``missing_dependency`` (a FINAL class), ``base_id`` = the base to
    acquire, ``fix`` = the operator instruction, the original chained as the
    cause, the prose unchanged. Established: 2026-09-29 (central retried this
    load indefinitely: "retrying ... worker load state is not confirmed")."""
    cfg = importlib.import_module("hugpy_engine.generate.config")
    exc = _raise(orphan_adapter)
    assert exc.base_model == BASE and exc.fix and BASE in exc.fix
    ref = cfg.adapter_base_refusal(KEY, exc)
    assert isinstance(ref, LF.MissingDependencyFailure)
    assert isinstance(ref, LF.ModelLoadFailure)
    assert ref.load_class == LF.MISSING_DEPENDENCY == "missing_dependency"
    assert LF.MISSING_DEPENDENCY in LF.LOAD_FAILURE_CLASSES
    assert LF.MISSING_DEPENDENCY in LF.FINAL_LOAD_CLASSES
    assert str(ref).startswith(f"{KEY}: PEFT adapter (base {BASE!r})")
    assert "base model is NOT in this store" in str(ref) and "FIX:" in str(ref)
    d = ref.load_failure
    assert d["class"] == "missing_dependency"
    assert d["base_id"] == BASE and d["fix"] == exc.fix
    assert d["model_key"] == KEY and d["path"] == orphan_adapter[0]
    # the additive wire payload (what rides the worker's error -> central ->
    # the v1 envelope as error.type = the class)
    try:
        raise ref from exc
    except LF.ModelLoadFailure as chained:
        lf = LF.load_failure_of(chained)
    assert lf["class"] == "missing_dependency" and lf["base_id"] == BASE
    assert lf["fix"] == exc.fix and lf["message"]


def test_bare_adapter_exception_still_classifies_as_missing_dependency(orphan_adapter):
    """GUARD: an AdapterBaseUnavailable that escapes without the wrapper (an
    older call site) still classifies as missing_dependency, never 'other'.
    Established: 2026-09-29."""
    exc = _raise(orphan_adapter)
    lf = LF.load_failure_of(exc, classify=True)
    assert lf["class"] == "missing_dependency" and lf["base_id"] == BASE


def test_central_treats_missing_dependency_as_final_not_transient(orphan_adapter):
    """INVARIANT: central's hold loop classifies a missing_dependency as
    PERMANENT for the attempt (fail fast, job terminal) by its STRUCTURED class
    and by its prose, and NEVER caches the verdict (a download repairs it —
    state-dependent, like 'could not fetch model'). A plain transient message
    is still transient. Established: 2026-09-29."""
    cfg = importlib.import_module("hugpy_engine.generate.config")
    ref = cfg.adapter_base_refusal(KEY, _raise(orphan_adapter))
    # structured: the class alone decides, whatever the wording
    assert remote._is_final_load_class(ref.load_failure)
    assert remote._is_final_load_class(ref)
    assert remote._is_final_load_class({"class": "missing_dependency"})
    assert not remote._is_final_load_class({"class": "other"})
    assert not remote._is_final_load_class(None)
    # prose (a worker still emitting the old RuntimeError text)
    old_prose = (f"{KEY}: PEFT adapter (base {BASE!r}) — the adapter is on disk "
                 f"but its base model is NOT in this store, and an adapter cannot "
                 f"be loaded without it. FIX: acquire {BASE!r}")
    assert remote._is_permanent_load_error(old_prose)
    assert remote._is_permanent_load_error(ref)
    assert remote._is_permanent_load_error(remote._HonestError(str(ref), ref.load_failure))
    assert not remote._is_permanent_load_error("worker has not confirmed model readiness")
    # fast refusal, never a cached verdict
    assert not remote._is_cacheable_load_verdict(str(ref))
    assert not remote._is_cacheable_load_verdict(old_prose)
