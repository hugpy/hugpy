"""Per-box serving policy gate.

One knob, read at CALL TIME (never cached at import), so it responds to a
systemd ``Environment=`` line, a drop-in, or a ``FLAG=… cmd`` shell prefix the
same way the sibling gates do (``HUGPY_LOCAL_FALLBACK`` in resolvers.remote,
``HUGPY_VIDEOGEN_LOCAL`` in the gpu guard).

    HUGPY_NO_LOCAL_SERVING=true

When set, THIS box must never host or serve a model in its own process/memory:
no local slot pool, no llama-swap/systemd local endpoint, no in-process
llama-cpp-python / transformers load, no local diffusion generation. Requests
route to a registered worker or fail with a clear, actionable error.

CRITICAL — this is a PER-BOX opt-in, default OFF. The very same package runs on
the worker boxes (ae/computron/op), where local serving is the whole job; a
hardcoded-off would kill the fleet on the next release. Default off === today's
behavior, everywhere. Only a box that explicitly sets the flag (the central
API/UI/dev station) refuses local serving.

Read ``os.environ`` only (like the sibling gates) so a systemd/env change takes
effect without touching the .env file.
"""
from __future__ import annotations

import os

_TRUE = ("1", "true", "yes", "on")


def no_local_serving() -> bool:
    """True when this box is configured to never serve/host models locally."""
    return os.environ.get("HUGPY_NO_LOCAL_SERVING", "").strip().lower() in _TRUE


def local_serving_error(model_key: str | None = None, *, detail: str = "") -> str:
    """A uniform, actionable message for a refused local-serving attempt."""
    who = f" for {model_key!r}" if model_key else ""
    tail = f" ({detail})" if detail else ""
    return (
        f"local model serving is disabled on this box (HUGPY_NO_LOCAL_SERVING); "
        f"no registered worker is available to serve{who}{tail}. Bring a worker "
        f"online for this model, or unset HUGPY_NO_LOCAL_SERVING to allow local "
        f"serving here."
    )
