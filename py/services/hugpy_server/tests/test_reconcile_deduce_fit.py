"""The central background warm planner was retired by load-only-on-call.

Keep a small boundary regression so a future refactor cannot accidentally
restore assignment/reconcile-driven VRAM residency.
"""

from hugpy_server.app.routes import worker_routes as wr


def test_background_warm_planner_is_absent():
    assert not hasattr(wr, "_reconcile_warm_set")
    assert not hasattr(wr, "_fit_skip_last")


def test_background_warm_dispatch_is_absent():
    assert not hasattr(wr, "_kick_warm")
    assert not hasattr(wr, "_polite_warm_ok")
