"""Central's admission gate reads hugpy.json["admission"] through the
persisted marker aspect: held -> refusal text (with the permanent marker),
pending/admitted/absent -> routable; unreadable -> fail open."""
from __future__ import annotations

import pytest

from hugpy_fleet.central import admission_gate as G
from hugpy_fleet.central import workers as W


@pytest.fixture
def marker(monkeypatch):
    box = {"marker": {}}
    monkeypatch.setattr(W, "_canonical_registry_key", lambda k: k)
    monkeypatch.setattr(W, "_registry_row", lambda k: {"model_key": k})
    monkeypatch.setattr("hugpy_storage.model_physical.lookup_physical",
                        lambda k, row, aspect: ({"hugpy_marker": box["marker"]}, "fresh"))
    return box


def test_held_refuses_with_reason(marker):
    marker["marker"] = {"admission": {"status": "held", "reason": "broken_download: short file",
                                      "integrity": "broken_download", "grade": None,
                                      "at": "t", "job": "j1"}}
    r = G.admission_reason("M")
    assert G.HELD_MARKER in r and "broken_download: short file" in r and "job=j1" in r
    assert G.is_held("M")


@pytest.mark.parametrize("block", [None, {"status": "pending"}, {"status": "admitted", "grade": 80}])
def test_not_held_routes(marker, block):
    marker["marker"] = {"admission": block} if block else {}
    assert G.admission_reason("M") is None


def test_fail_open(monkeypatch):
    def boom(k):
        raise RuntimeError("registry down")
    monkeypatch.setattr(W, "_canonical_registry_key", boom)
    assert G.admission_reason("M") is None
