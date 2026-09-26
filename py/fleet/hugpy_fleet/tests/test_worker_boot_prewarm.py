"""Regression coverage for the retired boot-prewarm protocol.

Heartbeat adoption may carry the legacy field during a rolling upgrade, but it
must never download or create a VRAM/RAM resident. Residency is call-driven.
"""

import importlib


A = importlib.import_module("hugpy_fleet.worker.agent")


def test_boot_prewarm_protocol_is_retired():
    assert not hasattr(A, "_adopt_boot_prewarm")
    assert not hasattr(A, "_BOOT_PREWARM_DONE")


def test_assignment_adoption_does_not_restore_boot_prewarm(monkeypatch):
    kicked = []
    monkeypatch.setattr(A, "_kick_provision",
                        lambda *args, **kwargs: kicked.append((args, kwargs)))
    state = A.WorkerState(name="t", url=None, worker_id="w-no-prewarm")
    A._sync_assignment(state, {"models": [], "boot_prewarm": "M~star"})
    assert kicked == []
