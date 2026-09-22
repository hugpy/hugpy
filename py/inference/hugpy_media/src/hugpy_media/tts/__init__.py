"""Text-to-speech manager — the fleet-side seat for the k98 chatterbox adapter.

LAZY ON PURPOSE. ``managers.tts.seat`` is consulted on every worker HEARTBEAT,
and importing this package eagerly would drag the runner (and pydantic, and the
constants tree) into that path for nothing. PEP 562 module ``__getattr__`` keeps
``from ...tts import ChatterboxTtsRunner`` working for the resolver's category
tables while ``import abstract_hugpy_dev.managers.tts.seat`` stays stdlib-cheap.
"""
from __future__ import annotations

__all__ = ["ChatterboxTtsRunner", "SynthesizedAudio", "TtsRequest", "TtsResult",
           "seat"]

_LAZY = {
    "ChatterboxTtsRunner": (".tts_runner", "ChatterboxTtsRunner"),
    "TtsRequest": (".schemas", "TtsRequest"),
    "TtsResult": (".schemas", "TtsResult"),
    "SynthesizedAudio": (".schemas", "SynthesizedAudio"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
