"""Console-side manager for phone-brick orchestration.

Wraps the standalone :class:`~hugpy.phone_brick.ChainOrchestrator` so the console
can fan one image across the registered phone pool and track the run, instead of
the operator shelling out to ``python -m ...phone_brick orchestrate``. The
detection logic itself is unchanged — this only adapts it to the registry + the
run store the UI polls.
"""
from hugpy_fleet.phone_brick_orchestrator.runner import start_run, run_phases_summary

__all__ = ["start_run", "run_phases_summary"]
