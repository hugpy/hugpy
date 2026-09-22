"""Canonical ML-amenity task -> dependency map + a cheap capability probe.

Single source of truth shared by:
  * central's ``/ml`` readiness (``flask_app.../ml_routes`` builds its route-keyed
    ``ML_DEP`` from this), and
  * the worker agent's advertised ``task_capabilities`` heartbeat field
    (``worker_agent.agent._task_capabilities``),
so the two can never DRIFT on which import module powers which dispatch task.

Deliberately dependency-free (stdlib ``importlib`` only): central imports it in a
phone-clean ``GET /ml`` and the worker imports it to build EVERY heartbeat, so it
must never drag a heavy ML import. The probe is ``find_spec`` ONLY — it resolves
whether a module is importable without importing it. The one task where
``find_spec`` is insufficient (whisper: resolvable yet ``import whisper`` can die
under the numba/numpy>=2.5 landmine, 2026-07-11) keeps its guarded real import in
the worker agent, on top of this map.
"""
from __future__ import annotations

import importlib.util
from typing import Dict, Tuple

# task (the dispatch/`/ml` task key) -> (import module, pip extra that provides it).
# Mirrors — and is the source for — ml_routes.ML_DEP (which is keyed by ROUTE name
# and joins through ml_routes.ML_TASKS). Keep the two key spaces in sync via that
# join, never by hand-copying values.
TASK_DEPS: Dict[str, Tuple[str, str]] = {
    "automatic-speech-recognition": ("whisper", "audio"),
    # SPEECH OUT (chatterbox seat). The distribution is ``chatterbox-tts`` and
    # the import name is ``chatterbox`` — they differ, and this table's probe is
    # by MODULE, so the module name is what belongs here. A worker that seats the
    # backend in a per-model env-PROFILE venv (its torch pins collide with the
    # agent's) answers False to this find_spec and True to the agent's overlay —
    # see managers.tts.seat, the same shape as the whisper special case below.
    "text-to-speech":               ("chatterbox", "tts"),
    "text-summarization":           ("transformers", "transformers"),
    "keyword-extraction":           ("keybert", "keywords"),
    "feature-extraction":           ("sentence_transformers", "embed"),
    "sentence-similarity":          ("sentence_transformers", "embed"),
    "image-text-to-text":           ("llama_cpp", "engine"),
    "text-to-image":                ("diffusers", "imagegen"),
    "document-extraction":          ("pdfplumber", "extract"),
    "url-extraction":               ("bs4", "web"),
    "depth-estimation":             ("transformers", "transformers"),
    "object-detection":             ("transformers", "transformers"),
    "image-classification":         ("transformers", "transformers"),
    "image-segmentation":           ("transformers", "transformers"),
}


def have(mod: str) -> bool:
    """Whether ``mod`` is importable — ``find_spec`` only, never imports it."""
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def task_capabilities() -> Dict[str, bool]:
    """``{task: bool}`` from the ``find_spec`` probe (cheap, no heavy imports).

    The worker overlays TWO special cases on top of this base map before
    advertising it (see worker_agent.agent._task_capabilities): whisper (a
    guarded real import) and text-to-speech (a seat that may live in a separate
    env-profile venv, which no probe of THIS interpreter can see).
    """
    return {task: have(mod) for task, (mod, _extra) in TASK_DEPS.items()}
