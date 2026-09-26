"""Operator model aliases (2026-09-24) — opt-in vendor/harness name -> hugpy key.

A harness that hardcodes its own default model name ("anthropic/claude-opus-4.6",
"gpt-6-luna", ...) resolves through hugpy when the operator registers an alias,
WITHOUT editing the harness. Rules pinned here:
  * empty/absent alias file -> behaviour byte-identical (no aliases).
  * a registered alias whose TARGET resolves wins (before fuzzy matching).
  * literal registry membership of the requested name always wins over an alias.
  * a BROKEN alias (target not resolvable) is IGNORED -> the honest refusal for
    the original name still fires (never a silent substitution to a missing model).
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AMK = importlib.import_module("hugpy_engine.resolvers.assure_model_key")


@pytest.fixture()
def registry(monkeypatch):
    reg = {"Real-Model": {}}
    monkeypatch.setattr(AMK, "MODEL_REGISTRY", reg)
    return reg


@pytest.fixture()
def alias_file(tmp_path, monkeypatch):
    p = tmp_path / "model_aliases.json"
    monkeypatch.setenv("HUGPY_MODEL_ALIASES_PATH", str(p))
    # reset the module's mtime cache so each test's file is re-read
    monkeypatch.setattr(AMK, "_ALIAS_PATH_CACHED", None)
    monkeypatch.setattr(AMK, "_ALIAS_MTIME", -1.0)
    monkeypatch.setattr(AMK, "_ALIAS_CACHE", {})

    def _write(mapping):
        p.write_text(json.dumps(mapping), encoding="utf-8")
        return p
    return _write


def test_no_alias_file_is_a_noop(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("HUGPY_MODEL_ALIASES_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(AMK, "_ALIAS_PATH_CACHED", None)
    monkeypatch.setattr(AMK, "_ALIAS_MTIME", -1.0)
    monkeypatch.setattr(AMK, "_ALIAS_CACHE", {})
    assert AMK._load_aliases() == {}
    assert AMK.assure_model_key("Real-Model") == "Real-Model"


def test_registered_alias_resolves_to_target(registry, alias_file):
    alias_file({"anthropic/claude-opus-4.6": "Real-Model"})
    assert AMK.assure_model_key("anthropic/claude-opus-4.6") == "Real-Model"


def test_alias_matches_slug_form(registry, alias_file):
    alias_file({"anthropic/claude-opus-4.6": "Real-Model"})
    # slugified spelling of the same alias key still hits.
    assert AMK.assure_model_key("anthropic_claude-opus-4.6") == "Real-Model"


def test_literal_registry_membership_wins_over_alias(registry, alias_file):
    alias_file({"Real-Model": "something-else"})
    assert AMK.assure_model_key("Real-Model") == "Real-Model"


def test_broken_alias_is_ignored_not_substituted(registry, alias_file):
    # Target does not exist -> the alias is ignored and the original name gets
    # the honest unresolved result (None), never a silent redirect.
    alias_file({"vendor/ghost": "Nonexistent-Model"})
    assert AMK.assure_model_key("vendor/ghost") is None


def test_bad_alias_file_is_fail_open(registry, tmp_path, monkeypatch):
    p = tmp_path / "model_aliases.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("HUGPY_MODEL_ALIASES_PATH", str(p))
    monkeypatch.setattr(AMK, "_ALIAS_PATH_CACHED", None)
    monkeypatch.setattr(AMK, "_ALIAS_MTIME", -1.0)
    monkeypatch.setattr(AMK, "_ALIAS_CACHE", {})
    assert AMK._load_aliases() == {}
    assert AMK.assure_model_key("Real-Model") == "Real-Model"
