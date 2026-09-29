"""KNOWN-GOOD CONTRACT — explicit representation variable ``model_format``.

Catalogue: notes/KNOWN-GOOD-CORE.md (area "model key resolution").
Source under test: hugpy_engine/resolvers/assure_model_key.py (fmt path,
_redirect_to_format, _representation_of) and resolvers/model_resolver.py
(resolve_model_key model_format kwarg + validation).

Operator intent (2026-09-29): a call may carry ``model_format`` = "gguf" |
"transformers" | "auto" (default). Explicit gguf/transformers is AUTHORITATIVE:
that representation is chosen even if the other is already loaded and even if the
name suffix says otherwise. "auto" is byte-identical to legacy resolution.

Deterministic: fake MODEL_REGISTRY, stubbed worker/serving facts, no network/GPU.
"""
from __future__ import annotations

import importlib
import types

import pytest

AMK = importlib.import_module("hugpy_engine.resolvers.assure_model_key")
MR = importlib.import_module("hugpy_engine.resolvers.model_resolver")

GGUF = "Qwen~Qwen3-Coder-Next-GGUF"
TF = "Qwen~Qwen3-Coder-Next"


def _cfg(framework, hub_id=None, tasks=("text-generation",)):
    return types.SimpleNamespace(framework=framework, hub_id=hub_id, folder=None,
                                 tasks=list(tasks), meta={}, name=hub_id)


@pytest.fixture(autouse=True)
def _no_alias_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_MODEL_ALIASES_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(AMK, "_ALIAS_PATH_CACHED", None)
    monkeypatch.setattr(AMK, "_ALIAS_MTIME", -1.0)
    monkeypatch.setattr(AMK, "_ALIAS_CACHE", {})


@pytest.fixture
def registry(monkeypatch):
    def _install(entries: dict):
        reg = dict(entries)
        monkeypatch.setattr(AMK, "MODEL_REGISTRY", reg)
        monkeypatch.setattr(MR, "MODEL_REGISTRY", reg)
        return reg
    return _install


@pytest.fixture
def facts(monkeypatch):
    """Stub the DB/worker seams; ``serving`` per the test's choice."""
    state = {"serving": set()}
    monkeypatch.setattr(AMK, "_servable_facts",
                        lambda k: (True, k in state["serving"]))
    monkeypatch.setattr(AMK, "_blocked_fact", lambda k: False)
    monkeypatch.setattr(AMK, "_starred_fact", lambda k: False)
    monkeypatch.setattr(AMK, "_adjusted_use", lambda k: 0)

    def _pick(requested, keys):
        import sys
        prev = sys.modules.get("hugpy_control.settings")
        sys.modules["hugpy_control.settings"] = None
        try:
            return AMK._pipeline_pick.__wrapped__(requested, keys) \
                if hasattr(AMK._pipeline_pick, "__wrapped__") else _real(requested, keys)
        finally:
            if prev is None:
                sys.modules.pop("hugpy_control.settings", None)
            else:
                sys.modules["hugpy_control.settings"] = prev

    _real = AMK._pipeline_pick
    monkeypatch.setattr(AMK, "_pipeline_pick", lambda r, k: _real(r, k))
    return state


def _both(registry):
    return registry({GGUF: _cfg("gguf", hub_id="Qwen/Qwen3-Coder-Next-GGUF"),
                     TF: _cfg("transformers", hub_id="Qwen/Qwen3-Coder-Next")})


# ---------------------------------------------------------------------------
# Precedence #1 — explicit model_format is AUTHORITATIVE
# ---------------------------------------------------------------------------
def test_format_gguf_forces_gguf_even_when_transformers_loaded(registry, facts):
    """INVARIANT: model_format='gguf' selects the GGUF row for a bare family
    name even though the Transformers row is the live serving representation.
    The explicit format overrides the serving tie-break (precedence #1)."""
    _both(registry)
    facts["serving"] = {TF}
    assert AMK.assure_model_key("Qwen3-Coder-Next") == TF            # auto → serving-side
    assert AMK.assure_model_key("Qwen3-Coder-Next", fmt="gguf") == GGUF


def test_format_transformers_forces_transformers_even_when_gguf_loaded(registry, facts):
    """INVARIANT: model_format='transformers' selects the Transformers row for a
    name whose suffix says GGUF and while GGUF is the loaded representation —
    the explicit format overrides both the name suffix and serving."""
    _both(registry)
    facts["serving"] = {GGUF}
    assert AMK.assure_model_key(GGUF) == GGUF                        # name suffix, auto
    assert AMK.assure_model_key(GGUF, fmt="transformers") == TF


def test_resolved_representation_gates_specialcasing_not_a_name_substring(registry, facts):
    """INVARIANT: after a format redirect the RESOLVED key's framework is the
    requested representation, so the mmproj/MoE/ctx special-casing (which gates
    on framework/is_gguf_engine, never on a name substring) fires for the right
    representation: a '-GGUF' NAME with model_format='transformers' yields a key
    whose framework is 'transformers', not 'gguf'."""
    _both(registry)
    got = AMK.assure_model_key(GGUF, fmt="transformers")
    assert AMK._framework_of(got) == "transformers"
    assert AMK._representation_of(got) == "transformers"
    got2 = AMK.assure_model_key("Qwen3-Coder-Next", fmt="gguf")
    assert AMK._framework_of(got2) == "gguf"


def test_format_with_no_such_representation_keeps_resolved(registry, facts):
    """INVARIANT: when the registry holds no row of the requested representation
    the base resolution is KEPT (a resolver cannot invent an absent row); the
    request is not failed. Only the GGUF row exists → transformers request stays
    on GGUF."""
    registry({GGUF: _cfg("gguf", hub_id="Qwen/Qwen3-Coder-Next-GGUF")})
    assert AMK.assure_model_key(GGUF, fmt="transformers") == GGUF


def test_format_redirect_never_crosses_owner(registry, facts):
    """INVARIANT: a representation redirect stays within the resolved model's
    publisher — an unsloth GGUF request forced to transformers does not jump to
    Qwen's transformers row when no unsloth transformers row exists."""
    UNS = "unsloth~Qwen3-Coder-Next-GGUF"
    registry({UNS: _cfg("gguf", hub_id="unsloth/Qwen3-Coder-Next-GGUF"),
              TF: _cfg("transformers", hub_id="Qwen/Qwen3-Coder-Next")})
    assert AMK.assure_model_key(UNS, fmt="transformers") == UNS     # no unsloth TF → kept


# ---------------------------------------------------------------------------
# Precedence #3 + auto == legacy
# ---------------------------------------------------------------------------
def test_auto_bare_name_follows_serving_representation(registry, facts):
    """INVARIANT (re-pin): model_format absent/'auto' leaves the serving-aware
    tie-break in charge for a bare (non-package-specific) family name."""
    _both(registry)
    facts["serving"] = {GGUF}
    assert AMK.assure_model_key("coder-next") == GGUF
    assert AMK.assure_model_key("coder-next", fmt="auto") == GGUF
    facts["serving"] = {TF}
    assert AMK.assure_model_key("coder-next") == TF
    assert AMK.assure_model_key("coder-next", fmt="auto") == TF


def test_auto_equals_legacy_for_package_specific(registry, facts):
    """INVARIANT: fmt='auto' is byte-identical to no fmt — a package-specific
    name resolves to its own representation with or without the auto marker."""
    _both(registry)
    facts["serving"] = {TF}
    assert AMK.assure_model_key(GGUF) == AMK.assure_model_key(GGUF, fmt="auto") == GGUF


def test_open_item_auto_name_qualifier_stays_hard(registry, facts):
    """OPEN ITEM (documented, NOT precedence #2): with model_format='auto' a
    name qualifier stays HARD — a '-GGUF' name resolves to the GGUF row even
    while Transformers is the loaded representation. Honoring precedence #2
    (loaded rep overriding a name suffix) would require serving to override the
    package-specific contract and is deferred; the explicit model_format is the
    supported override. This test pins the CURRENT behaviour so the deferral is
    visible and any future change is deliberate."""
    _both(registry)
    facts["serving"] = {TF}
    assert AMK.assure_model_key(GGUF) == GGUF        # NOT redirected to loaded TF


# ---------------------------------------------------------------------------
# resolve_model_key — kwarg + validation
# ---------------------------------------------------------------------------
def test_resolve_model_key_threads_and_validates_model_format(registry, facts):
    """INVARIANT: resolve_model_key forwards model_format to the resolver and
    rejects an unknown value with a ValueError that names the field."""
    _both(registry)
    facts["serving"] = {TF}
    assert MR.resolve_model_key(model_key="Qwen3-Coder-Next",
                                model_format="gguf") == GGUF
    assert MR.resolve_model_key(model_key=GGUF,
                                model_format="transformers") == TF
    assert MR.resolve_model_key(model_key=GGUF, model_format="auto") == GGUF
    assert MR.resolve_model_key(model_key=GGUF, model_format=None) == GGUF
    with pytest.raises(ValueError) as ei:
        MR.resolve_model_key(model_key=GGUF, model_format="ollama")
    assert "model_format" in str(ei.value)
