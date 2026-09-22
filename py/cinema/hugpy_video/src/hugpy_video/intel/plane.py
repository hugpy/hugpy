"""The one place video drives the inference plane (``hugpy_engine``).

Every runner that needs a model call (image generation, the vision judge,
the studio tester) goes through :func:`execute_prompt` here rather than
importing the engine's dispatch module itself, so:

* the engine's internal layout is one seam away (``hugpy_engine.dispatch`` is
  a package whose public verb lives in ``hugpy_engine.dispatch.dispatch``);
* tests fake the plane by monkeypatching ``hugpy_video.intel.plane.execute_prompt``
  or by passing an ``executor`` where a runner accepts one.

``execute_prompt(**kwargs)`` is the engine's one-shot request -> result verb
(sync; the result may be awaitable — callers wrap it in
``hugpy_platform.async_runtime.run`` which tolerates plain values).
"""

from __future__ import annotations

from typing import Any

__all__ = ["execute_prompt", "engine_execute_prompt"]


def engine_execute_prompt():
    """Resolve the engine's ``execute_prompt`` lazily (never at import time).

    Prefers a public re-export on ``hugpy_engine.dispatch``; falls back to the
    defining module ``hugpy_engine.dispatch.dispatch``."""
    import importlib

    pkg = importlib.import_module("hugpy_engine.dispatch")
    fn = getattr(pkg, "execute_prompt", None)
    if fn is None:
        fn = importlib.import_module("hugpy_engine.dispatch.dispatch").execute_prompt
    return fn


def execute_prompt(*args: Any, **kwargs: Any):
    return engine_execute_prompt()(*args, **kwargs)
