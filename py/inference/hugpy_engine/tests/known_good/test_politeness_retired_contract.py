"""KNOWN-GOOD CONTRACT — per-model politeness is RETIRED (direction d1136,
2026-09-29: no model-specific backend behaviour). ``no_evict`` /
``no_evict_by_worker`` are unknown to the serve overrides; residency
(static / on-demand) is the only protection vocabulary.

Source under test: hugpy_engine/serve/overrides.py (ALLOWED_FIELDS,
_RETIRED_FIELDS, set_override, _load/_strip_retired, placement_policy).
Deterministic: a private serve_overrides.json under tmp_path.
"""
from __future__ import annotations

import importlib
import json
import logging

import pytest

OV = importlib.import_module("hugpy_engine.serve.overrides")
MK = "FLUX.2-klein-9B"


@pytest.fixture
def ov(monkeypatch, tmp_path):
    monkeypatch.setattr(OV, "_OVERRIDES_PATH", str(tmp_path / "serve_overrides.json"))
    OV._RETIRED_LOGGED.clear()
    return OV


def test_a_post_with_no_evict_is_ignored_with_a_warning(ov, caplog):
    """INVARIANT: the fields are not in ALLOWED_FIELDS; a write carrying them
    stores nothing for them (warned) and keeps the rest of the body.
    Established: 2026-09-29."""
    assert "no_evict" not in ov.ALLOWED_FIELDS and "no_evict_by_worker" not in ov.ALLOWED_FIELDS
    with caplog.at_level(logging.WARNING):
        row = ov.set_override(MK, {"worker_prefs": ["ae"], "no_evict": True,
                                   "no_evict_by_worker": {"ae": True}})
    assert row == {"worker_prefs": ["ae"]} == ov.get_override(MK)
    assert any("retired" in r.getMessage() for r in caplog.records)


def test_placement_policy_no_longer_reads_politeness(ov):
    """INVARIANT: (prefs, False, {}) whatever the file says. Established: 2026-09-29."""
    with open(ov._OVERRIDES_PATH, "w", encoding="utf-8") as fh:
        json.dump({MK: {"worker_prefs": ["ae"], "no_evict": True,
                        "no_evict_by_worker": {"ae": True}}}, fh)
    assert ov.placement_policy(MK) == (["ae"], False, {})
    assert ov.placement_prefs(MK) == (["ae"], False)
    assert ov.polite_on_worker(MK, "ae") is False


def test_a_stored_entry_with_the_field_still_loads(ov, caplog):
    """INVARIANT: a pre-retirement file loads without error; the retired field
    is dropped (logged once), every other field intact, and the next save
    writes the file without it. Established: 2026-09-29."""
    with open(ov._OVERRIDES_PATH, "w", encoding="utf-8") as fh:
        json.dump({MK: {"gpu_mem_gib": 8.0, "no_evict": True},
                   "Other": {"strict": True, "no_evict_by_worker": {"ae": False}}}, fh)
    with caplog.at_level(logging.WARNING):
        assert ov.get_override(MK) == {"gpu_mem_gib": 8.0}
        assert ov.get_override("Other") == {"strict": True}
        ov.get_override(MK)                                   # logged ONCE per (model, field)
    assert sum("retired" in r.getMessage() for r in caplog.records) == 2
    ov.set_override(MK, {"gpu_mem_gib": 9.0})
    with open(ov._OVERRIDES_PATH, encoding="utf-8") as fh:
        assert json.load(fh) == {MK: {"gpu_mem_gib": 9.0}, "Other": {"strict": True}}
