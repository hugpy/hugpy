"""Fake render-child targets for ``_studio_subproc.run_render_subprocess``.

A ``spawn`` child unpickles its target BY REFERENCE (module + qualname), so the
target must live in a module the child can import by that name. Under pytest's
importlib import mode a test module's own name is derived from its path and is
NOT importable from a child interpreter — this helper module is, because the
package conftest puts ``tests/integration`` on ``sys.path`` (the same idiom as
``worker_store_isolation``), which the spawn child inherits.
"""
from __future__ import annotations


def step_progress_target(spec_dict, conn, cancel_event):
    """Streams three denoise-step frames, then the settled payload."""
    from hugpy_fleet.worker import _studio_subproc
    for step in (1, 2, 3):
        conn.send({_studio_subproc._PROGRESS_KEY: {
            "phase": "rendering", "step": step, "steps": 3}})
    conn.send({"ok": True, "path": "/shared/clip.mp4", "frames": 81})
    conn.close()
