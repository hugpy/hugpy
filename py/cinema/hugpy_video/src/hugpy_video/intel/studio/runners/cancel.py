"""Cooperative diffusion-step cancellation and progress callback."""

from __future__ import annotations


def cancel_step_callback(should_cancel, on_step, steps):
    """Build the callback accepted by Wan diffusion pipelines."""
    def callback(pipe_ref, step_index, timestep, cb_kwargs):
        if should_cancel is not None and should_cancel():
            pipe_ref._interrupt = True
        if on_step is not None:
            try:
                on_step(int(step_index) + 1, int(steps))
            except Exception:  # noqa: BLE001 — telemetry never breaks a render
                pass
        return cb_kwargs

    return callback
