"""KNOWN-GOOD CONTRACT — model key resolution.

Catalogue: docs/KNOWN-GOOD-CORE.md (area "model key resolution").
Source under test: hugpy_engine/resolvers/assure_model_key.py and
hugpy_engine/resolvers/model_resolver.py (resolve_model_key).

Every test pins ONE invariant that was established as known-good and names
where it came from. These tests are deterministic: the registry is a fake
dict installed on the module, the worker/serving facts are stubbed, and no
network, GPU or live worker is consulted. If one of these fails after a
refactor, the refactor changed a contract the fleet relies on — read the
docstring before "fixing" the test.

Run:
    PYTHONPATH="$(find py -mindepth 3 -maxdepth 3 -type d -path '*/src' -printf '%p:' | sed 's/:$//')" \
      python -m pytest -q --import-mode=importlib py/inference/hugpy_engine/tests/known_good
"""
from __future__ import annotations

import importlib
import types

import pytest

AMK = importlib.import_module("hugpy_engine.resolvers.assure_model_key")
MR = importlib.import_module("hugpy_engine.resolvers.model_resolver")

GGUF = "Qwen~Qwen3-Coder-Next-GGUF"
TF = "Qwen~Qwen3-Coder-Next"
UNSLOTH = "unsloth~Qwen3-Coder-Next-GGUF"


def _cfg(hub_id=None, folder=None, tasks=("text-generation",)):
    return types.SimpleNamespace(hub_id=hub_id, folder=folder,
                                 tasks=list(tasks), meta={}, name=hub_id)


@pytest.fixture(autouse=True)
def _no_alias_file(tmp_path, monkeypatch):
    """Operator aliases are opt-in; keep them out of these contracts."""
    monkeypatch.setenv("HUGPY_MODEL_ALIASES_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(AMK, "_ALIAS_PATH_CACHED", None)
    monkeypatch.setattr(AMK, "_ALIAS_MTIME", -1.0)
    monkeypatch.setattr(AMK, "_ALIAS_CACHE", {})


@pytest.fixture
def registry(monkeypatch):
    """Install a fake MODEL_REGISTRY on both the resolver modules."""
    def _install(entries: dict):
        reg = dict(entries)
        monkeypatch.setattr(AMK, "MODEL_REGISTRY", reg)
        monkeypatch.setattr(MR, "MODEL_REGISTRY", reg)
        return reg
    return _install


@pytest.fixture
def facts(monkeypatch):
    """Stub the DB/worker seams the tie pipeline reads: nothing blocked or
    starred, every key servable, ``serving`` per the test's choice, no usage."""
    state = {"serving": set()}
    monkeypatch.setattr(AMK, "_servable_facts",
                        lambda k: (True, k in state["serving"]))
    monkeypatch.setattr(AMK, "_blocked_fact", lambda k: False)
    monkeypatch.setattr(AMK, "_starred_fact", lambda k: False)
    monkeypatch.setattr(AMK, "_adjusted_use", lambda k: 0)
    # The winner's tie-pick counter is advisory; never touch a settings store.
    monkeypatch.setattr(AMK, "_pipeline_pick",
                        _pipeline_pick_without_settings(AMK._pipeline_pick))
    return state


def _pipeline_pick_without_settings(real):
    """Run the real pipeline pick but swallow the settings_store increment."""
    def _pick(requested, keys):
        import sys
        blocked = {}
        for name in ("hugpy_control.settings",):
            blocked[name] = sys.modules.get(name)
            sys.modules[name] = None          # import inside the try → ImportError
        try:
            return real(requested, keys)
        finally:
            for name, mod in blocked.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
    return _pick


# ---------------------------------------------------------------------------
# Owner-aware specificity (frontier handoff 2026-09-28)
# ---------------------------------------------------------------------------
def test_qualified_key_never_collapses_to_another_owners_sibling(registry):
    """INVARIANT: ``unsloth~Qwen3-Coder-Next-GGUF`` can NOT resolve to
    ``Qwen~Qwen3-Coder-Next-GGUF`` (or vice versa) just because the bare tail
    matches. A qualified request names a publisher; resolution returns None
    (→ honest "unknown model" at intake) rather than another owner's model.
    Established: frontier handoff 2026-09-28, assure_model_key._match_tier."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key(GGUF) == GGUF, "explicit spelling stays exact"
    assert AMK.assure_model_key(UNSLOTH) is None, \
        "another publisher's bare-tail sibling must not be substituted"
    # And the mirror image: the registry holds ONLY the unsloth variant.
    registry({UNSLOTH: _cfg(hub_id="unsloth/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key(GGUF) is None


def test_qualified_owner_may_be_carried_by_hub_id_of_a_bare_key(registry):
    """INVARIANT: the owner of a bare registry key is read from its hub_id, so
    ``unsloth~X`` matches bare key ``X`` whose hub_id is ``unsloth/X`` and does
    NOT match bare ``X`` whose hub_id is ``Qwen/X``.
    Established: frontier handoff 2026-09-28 (owner-aware bare-tail tier)."""
    bare = "Qwen3-Coder-Next-GGUF"
    registry({bare: _cfg(hub_id="unsloth/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key(UNSLOTH) == bare
    assert AMK.assure_model_key(GGUF) is None


def test_unqualified_request_resolves_into_qualified_family(registry):
    """INVARIANT: an UNQUALIFIED request (``Qwen3-Coder-Next-GGUF``) still
    resolves to the single qualified row that carries that bare tail — the
    k67 bare/qualified unification is preserved by the owner-aware tier.
    Established: k67 (bare-tail tier), re-pinned 2026-09-28."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key("Qwen3-Coder-Next-GGUF") == GGUF


def test_resolve_model_key_refuses_cross_owner_with_honest_error(registry):
    """INVARIANT: resolve_model_key (the intake resolver v1_routes/streaming
    use) raises ValueError naming the request when the owner does not match;
    it never silently routes to the sibling.
    Established: frontier handoff 2026-09-28 + slice 9 defect 3 (reject at
    intake)."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    with pytest.raises(ValueError) as ei:
        MR.resolve_model_key(model_key=UNSLOTH)
    assert UNSLOTH in str(ei.value)
    assert MR.resolve_model_key(model_key=GGUF) == GGUF


# ---------------------------------------------------------------------------
# Unqualified duplicates follow the SERVING representation
# ---------------------------------------------------------------------------
def test_unqualified_duplicate_picks_the_serving_representation(registry, facts):
    """INVARIANT: for an unqualified family name that matches BOTH a GGUF and a
    Transformers row, the representation a live worker is serving wins:
    GGUF loaded → GGUF; Transformers loaded → Transformers.
    Established: frontier handoff 2026-09-28 ("the serving representation
    wins"); assure_model_key._pipeline_pick via name_match 'serving' rank."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
              TF: _cfg(hub_id="Qwen/Qwen3-Coder-Next")})
    facts["serving"] = {GGUF}
    assert AMK.assure_model_key("coder-next") == GGUF
    facts["serving"] = {TF}
    assert AMK.assure_model_key("coder-next") == TF


def test_package_specific_request_is_never_redirected_by_serving(registry, facts,
                                                                monkeypatch):
    """INVARIANT: a request that names a packaging (``…-GGUF``, ``…-AWQ``,
    ``…-4bit``…) is DELIBERATE specificity. A same-tier tie is broken by the
    deterministic total order (usage, quant rank, canonical key), never by the
    serving-aware pipeline, so a hot Transformers sibling cannot hijack an
    explicit GGUF request. strict=True surfaces a true tie as None.
    Established: frontier handoff 2026-09-28 (_package_specific gate)."""
    a, b = "Qwen~Qwen3-Coder-Next-GGUF", "unsloth~Qwen3-Coder-Next-GGUF"
    registry({a: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
              b: _cfg(hub_id="unsloth/Qwen3-Coder-Next-GGUF")})
    facts["serving"] = {b}

    def _must_not_run(requested, keys):
        raise AssertionError("serving pipeline must not decide a package-specific request")
    monkeypatch.setattr(AMK, "_pipeline_pick", _must_not_run)

    assert AMK._package_specific("Qwen3-Coder-Next-GGUF") is True
    assert AMK._package_specific("coder-next") is False
    # Same tier (bare tail), same usage, same quant rank → canonical key asc.
    assert AMK.assure_model_key("Qwen3-Coder-Next-GGUF") == a
    assert AMK.assure_model_key("Qwen3-Coder-Next-GGUF", strict=True) is None


def test_exact_registry_membership_always_wins(registry, facts):
    """INVARIANT: a literal registry key resolves to itself before any tier,
    alias or serving consideration — explicit spelling remains exact.
    Established: assure_model_key contract (pre-dates 2026-09; re-pinned)."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
              TF: _cfg(hub_id="Qwen/Qwen3-Coder-Next")})
    facts["serving"] = {GGUF}
    assert AMK.assure_model_key(TF) == TF
    assert AMK.assure_model_key(GGUF + "/") == GGUF   # trailing slash tolerated


def test_unknown_key_resolves_to_none_and_intake_error_names_it(registry):
    """INVARIANT: a key that matches nothing resolves to None, and the intake
    resolver turns that into ValueError('Unknown model_key=…') — never a
    pending-forever job (slice 9 defect 3, reject at intake)."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key("zzz-nonexistent-model-xyz-999") is None
    with pytest.raises(ValueError) as ei:
        MR.resolve_model_key(model_key="zzz-nonexistent-model-xyz-999")
    assert "Unknown model_key" in str(ei.value)
