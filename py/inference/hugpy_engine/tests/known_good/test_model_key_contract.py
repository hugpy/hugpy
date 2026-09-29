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


# ---------------------------------------------------------------------------
# Key equivalence: ``-GGUF`` is a format suffix, everything else is identity
# (operator rule, board 2026-09-29; eviction test S3 / F7)
# ---------------------------------------------------------------------------
def _fw(framework, hub_id=None, tasks=("text-generation",)):
    return types.SimpleNamespace(framework=framework, hub_id=hub_id, folder=None,
                                 tasks=list(tasks), meta={}, name=hub_id)


def test_key_equivalent_strips_only_the_gguf_format_suffix():
    """INVARIANT: ``X-GGUF`` == ``X`` (case-insensitive suffix, ``-`` or ``_``
    separator); ANY other differing token — ``-Distill``, a quant marker, an
    ``-i1``, a finetune tag — is a different model. Same-file residency is
    not consulted: the rule is purely about the key.
    Established: operator rule 2026-09-29 (key_equivalent / canonical_key)."""
    eq, canon = AMK.key_equivalent, AMK.canonical_key
    assert eq("Qwen3.8-9B-GGUF", "Qwen3.8-9B")
    assert eq("Qwen3.8-9B", "Qwen3.8-9B-gguf")
    assert eq("Qwen3.8_4B_Distilled_GGUF", "Qwen3.8_4B_Distilled")
    assert eq("Qwen3.8-9B-Distill-GGUF", "Qwen3.8-9B-Distill")
    assert canon("Qwen3.8-9B-GGUF") == "Qwen3.8-9B"
    assert canon("Qwen3.8-9B-Distill-GGUF") == "Qwen3.8-9B-Distill"
    # the S3 pair: same file on ae's drive, DIFFERENT keys
    assert not eq("Qwen3.8-9B-GGUF", "Qwen3.8-9B-Distill-GGUF")
    assert not eq("Qwen3.8-9B", "Qwen3.8-9B-Distill")
    # quant / variant markers are identity, not format
    assert not eq("Qwen3.8-9B-GGUF", "Qwen3.8-9B-GGUF-Q8_0")
    assert not eq("Qwen3.8-9B-GGUF", "Qwen3.8-9B-Q4_K_M-GGUF")
    assert not eq("Qwen3.8-9B-heretic-uncensored-GGUF", "Qwen3.8-9B-heretic-uncensored-i1-GGUF")
    assert not eq("Qwen3-Coder-Next-GGUF", "Qwen3-Coder-Next-AWQ")
    # the suffix is stripped once, at the end only
    assert canon("GGUF-Something-GGUF") == "GGUF-Something"
    assert not eq("", "") and not eq(None, "X")


def test_format_equivalence_tier_resolves_across_the_gguf_suffix_never_to_a_sibling(registry):
    """INVARIANT: a request for ``X`` resolves to the ``X-GGUF`` row (and
    ``X-GGUF`` to ``X``) DETERMINISTICALLY, ahead of the fuzzy tier — so the
    presence of ``X-Distill-GGUF`` in the same registry can never win. The
    format tier is weaker than the bare-tail tier: an exact-tail row still
    beats a format sibling.
    Established: operator rule 2026-09-29 (_TIER_FORMAT)."""
    plain, distill = "Qwen3.8-9B-GGUF", "Qwen3.8-9B-Distill-GGUF"
    registry({plain: _cfg(hub_id="empero-ai/Qwen3.8-9B-GGUF"),
              distill: _cfg(hub_id="empero-ai/Qwen3.8-9B-Distill-GGUF")})
    assert AMK.assure_model_key("Qwen3.8-9B") == plain
    assert AMK.assure_model_key("Qwen3.8-9B-Distill") == distill
    assert AMK.assure_model_key(plain) == plain and AMK.assure_model_key(distill) == distill
    # the mirror: only the plain (transformers-style) row is on record
    registry({"Qwen3.8-9B": _cfg(hub_id="Qwen/Qwen3.8-9B"), distill: _cfg()})
    assert AMK.assure_model_key("Qwen3.8-9B-GGUF") == "Qwen3.8-9B"
    # exact tail outranks the format sibling when both exist
    registry({"Qwen3.8-9B": _cfg(), plain: _cfg()})
    assert AMK.assure_model_key("Qwen3.8-9B") == "Qwen3.8-9B"
    assert AMK.assure_model_key(plain) == plain


def test_s3_shape_a_request_for_the_plain_key_never_resolves_to_the_distill_sibling(registry):
    """INVARIANT (eviction test S3 / F7, 2026-09-29): with ONLY
    ``Qwen3.8-9B-Distill-GGUF`` on record (ae-worker's registry), a request for
    ``Qwen3.8-9B-GGUF`` resolves to None — for the default resolver (the
    request is package-specific, so the fuzzy tier is skipped) and for the
    identity-only mode a worker uses (``fuzzy=False``). Intake refuses with an
    honest "did you mean" naming the sibling; nothing is silently substituted.
    Established: operator rule 2026-09-29."""
    distill = "Qwen3.8-9B-Distill-GGUF"
    registry({distill: _cfg(hub_id="empero-ai/Qwen3.8-9B-Distill-GGUF")})
    assert AMK.assure_model_key("Qwen3.8-9B-GGUF") is None
    assert AMK.assure_model_key("Qwen3.8-9B-GGUF", fuzzy=False) is None
    assert AMK.assure_model_key("Qwen3.8-9B", fuzzy=False) is None, \
        "identity-only mode never binds a bare name to a superset sibling"
    with pytest.raises(ValueError) as ei:
        MR.resolve_model_key(model_key="Qwen3.8-9B-GGUF")
    assert "Qwen3.8-9B-GGUF" in str(ei.value) and distill in str(ei.value)
    # the worker's catalog seam (ensure_model_registered -> _assure_local_key)
    # is identity-only by construction
    from hugpy_engine.catalog_bridge import EngineCatalogSource
    assert EngineCatalogSource().canonical_key("Qwen3.8-9B-GGUF") is None
    assert EngineCatalogSource().canonical_key(distill) == distill


def test_qualified_owner_keys_are_unchanged_by_format_equivalence(registry):
    """INVARIANT: the format tier honours the owner discipline exactly as the
    bare-tail tier does — ``unsloth~X`` never resolves to ``Qwen~X-GGUF``; the
    same-owner and unqualified spellings do.
    Established: operator rule 2026-09-29 on top of frontier handoff 2026-09-28."""
    registry({GGUF: _cfg(hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key("unsloth~Qwen3-Coder-Next") is None
    assert AMK.assure_model_key("Qwen~Qwen3-Coder-Next") == GGUF
    assert AMK.assure_model_key("Qwen3-Coder-Next") == GGUF
    assert AMK.assure_model_key(UNSLOTH) is None
    bare = "Qwen3-Coder-Next"
    registry({bare: _cfg(hub_id="unsloth/Qwen3-Coder-Next")})
    assert AMK.assure_model_key("unsloth~Qwen3-Coder-Next-GGUF") == bare
    assert AMK.assure_model_key("Qwen~Qwen3-Coder-Next-GGUF") is None


def test_format_redirect_pairs_only_gguf_suffix_siblings(registry, facts):
    """INVARIANT: an explicit model_format redirect pairs representation
    siblings by ``key_equivalent`` (owner-stripped): ``X-GGUF`` <-> ``X`` only.
    ``-AWQ`` / a quant tag are NOT format variants, so with no plain row the
    resolved key is kept (warning), never an AWQ/quant sibling.
    Established: operator rule 2026-09-29 (replaces the _logical_id strip)."""
    g, awq, tf = "Qwen~Qwen3-Coder-Next-GGUF", "Qwen~Qwen3-Coder-Next-AWQ", "Qwen~Qwen3-Coder-Next"
    registry({g: _fw("gguf", hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
              awq: _fw("transformers", hub_id="Qwen/Qwen3-Coder-Next-AWQ")})
    assert AMK.assure_model_key(g, fmt="transformers") == g
    registry({g: _fw("gguf", hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
              awq: _fw("transformers", hub_id="Qwen/Qwen3-Coder-Next-AWQ"),
              tf: _fw("transformers", hub_id="Qwen/Qwen3-Coder-Next")})
    assert AMK.assure_model_key(g, fmt="transformers") == tf
    assert AMK.assure_model_key(tf, fmt="gguf") == g
