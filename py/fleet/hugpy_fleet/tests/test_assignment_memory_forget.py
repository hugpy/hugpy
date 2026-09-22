"""Sanctioned removal of GHOST entries from the assignment-memory sidecar.

Designations are worker-lifetime, not row-lifetime (deleting a live row keeps
its memory so a re-register restores it). ``forget_assignment_memory`` only
removes ids that are NOT live. Converted from the monolith's script-style
``test_worker_memory_forget.py`` section [1]; the route/operator-gating
sections stay with the server.
"""

from __future__ import annotations

import os

import pytest

from hugpy_fleet.central import workers as W

from worker_store_isolation import swap_worker_store


def test_forget_ghost_refuses_live_and_reports_unknown():
    with swap_worker_store(prefix="hugpy-memory-forget-test-"):
        W.worker_store.register(name="live-one", url="http://192.0.2.70:9100", worker_id="live-one")
        W.worker_store.assign_model("live-one", "Some~Model")
        assert "live-one" in W._load_assign_memory()

        W.worker_store.register(name="ghost-one", url="http://192.0.2.71:9100", worker_id="ghost-one")
        W.worker_store.assign_model("ghost-one", "Ghost~Model")
        assert W.worker_store.remove("ghost-one")
        assert W.worker_store.get("ghost-one") is None
        assert "ghost-one" in W._load_assign_memory()      # survives by design

        with pytest.raises(ValueError):
            W.forget_assignment_memory("live-one")
        assert "live-one" in W._load_assign_memory()
        assert W.forget_assignment_memory("never-existed-id") == "unknown"

        mem_path = W._assign_memory_path()
        before = os.stat(mem_path).st_mtime_ns
        assert W.forget_assignment_memory("ghost-one") == "forgot"
        assert "ghost-one" not in W._load_assign_memory()
        assert "live-one" in W._load_assign_memory()
        assert os.stat(mem_path).st_mtime_ns >= before
        assert not os.path.isfile(mem_path + ".tmp")
        assert W.forget_assignment_memory("ghost-one") == "unknown"
