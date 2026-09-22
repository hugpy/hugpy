"""hugpy-fleet: central fleet state plus the full, GGUF and phone workers.

Public surface (kept light — no engine/HTTP imports at package import):

* ``hugpy_fleet.wire`` — the DTOs central and workers exchange (re-exported).
* ``hugpy_fleet.central.placement`` — ``install()``/``uninstall()`` wiring the
  fleet into ``hugpy_engine.placement`` (the server calls ``install()``).
* ``hugpy_fleet.central.config`` — state roots (``settings``, ``configure``).
* Entry points: ``hugpy-worker``, ``hugpy-gguf-worker``, ``hugpy-phone-brick``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from hugpy_fleet.wire import (  # noqa: E402 — stdlib-only DTOs
    OPERATION_VERBS,
    EnrollmentRequest,
    ModelAssignment,
    WorkerHeartbeat,
    WorkerOperation,
    WorkerRegistration,
    WireDTO,
)

__all__ = [
    "__version__",
    "OPERATION_VERBS",
    "EnrollmentRequest",
    "ModelAssignment",
    "WireDTO",
    "WorkerHeartbeat",
    "WorkerOperation",
    "WorkerRegistration",
    "install",
    "uninstall",
    "configure",
]


def __getattr__(name: str):
    # Lazy: keeps ``import hugpy_fleet`` free of engine imports.
    if name in ("install", "uninstall"):
        from hugpy_fleet.central import placement
        return getattr(placement, name)
    if name == "configure":
        from hugpy_fleet.central.config import configure
        return configure
    raise AttributeError(f"module 'hugpy_fleet' has no attribute {name!r}")
