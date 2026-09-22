"""STUDIO min_rank_floor (2026-08-28) — the k120 landmine's enforcement.

The registry cannot express a rank floor, so the STUDIO dispatch route refuses
a plan authored by a writer-chain member below rank 3 (the resolve chain's
fallthrough bottoms out at the 3B, ruled "no good" for script work).

    ./venv/bin/pytest tests/test_studio_floor.py -q
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PG = importlib.import_module("hugpy_fleet.central.priority_groups")
VR = importlib.import_module(
    "hugpy_server.app.routes.video_routes")

_CHAIN = ["Fable-5-Distill-35B", "9B-DFlash", "Qwen2-7B-Instruct-GGUF",
          "Qwen2.5-3B-Instruct-GGUF", "Coder-Next"]


@pytest.fixture()
def writer_chain(monkeypatch):
    group = {"id": VR._STUDIO_WRITER_GROUP, "members": list(_CHAIN)}
    monkeypatch.setattr(PG, "get_group",
                        lambda gid: group if gid == group["id"] else None)
    # Real expand_members yields (member_key, source) TUPLES — the live shape
    # (a string-shaped mock here once hid a fail-open bug in production).
    monkeypatch.setattr(PG, "expand_members",
                        lambda g, *a, **kw: [(m, None) for m in g["members"]])
    return group


def test_bare_string_members_also_match(monkeypatch):
    # Tolerated legacy/future shape: expand_members returning bare strings.
    group = {"id": VR._STUDIO_WRITER_GROUP, "members": list(_CHAIN)}
    monkeypatch.setattr(PG, "get_group", lambda gid: group)
    monkeypatch.setattr(PG, "expand_members",
                        lambda g, *a, **kw: list(g["members"]))
    assert VR._studio_floor_violation("Qwen2.5-3B-Instruct-GGUF") is not None


def test_below_floor_member_is_refused(writer_chain):
    err = VR._studio_floor_violation("Qwen2.5-3B-Instruct-GGUF")
    assert err is not None and "min_rank_floor" in err


def test_chain_tail_is_refused_too(writer_chain):
    assert VR._studio_floor_violation("Coder-Next") is not None


def test_at_or_above_floor_passes(writer_chain):
    assert VR._studio_floor_violation("Fable-5-Distill-35B") is None
    assert VR._studio_floor_violation("Qwen2-7B-Instruct-GGUF") is None


def test_outside_model_is_a_pin_not_a_fallthrough(writer_chain):
    # An explicitly pinned model outside the chain is the caller's choice.
    assert VR._studio_floor_violation("some-other-model") is None


def test_registry_trouble_fails_open(monkeypatch):
    def _boom(_gid):
        raise RuntimeError("registry down")
    monkeypatch.setattr(PG, "get_group", _boom)
    assert VR._studio_floor_violation("Qwen2.5-3B-Instruct-GGUF") is None


def test_no_resolved_model_is_not_a_violation(writer_chain):
    assert VR._studio_floor_violation(None) is None
    assert VR._studio_floor_violation("") is None
