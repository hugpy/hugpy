"""Grading-suite registry: which suite grades a model, keyed by its task.

``SUITES = {task: Suite}``.  ``suite_for_model(row)`` picks by the catalog
row's ``primary_task`` (falling back to its ``tasks`` list) and returns None
for task types with no suite (video, ASR, TTS, embeddings, detection ...), so
the benchmark SKIPS those with a recorded reason instead of grading them 0 with
the text suite.

A Suite's ``call(client, lane, variation, prompt_spec, tokens)`` returns a
dict ``{answer, output, error, elapsed_s, tok_s, ctx_in, ctx_out, ...}`` and
``score(response, checker)`` returns the pass/fail bool.  ``tasks`` has the
text suite's shape: ``{task: ((tier, prompt_spec, checker), x3)}`` so the
recorded ``grade_detail`` stays ``{task: {tier, max, history:[{tier, pass}]}}``.

Submodules import lazily: registering a suite must not import PIL/numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional, Tuple

__all__ = ["Suite", "SUITES", "SUITES_BY_NAME", "suite_for_model", "suite_by_name", "model_tasks"]


@dataclass(frozen=True)
class Suite:
    name: str
    module: str                       # submodule under this package
    grade_task: str                   # model_metrics.task written with the grade
    modes: Optional[Tuple[str, ...]] = None   # alloc modes to run (None = all)
    pinned: bool = True               # calls honor alloc.worker -> one lane per worker
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def _mod(self):
        if "m" not in self._cache:
            self._cache["m"] = import_module("." + self.module, __name__)
        return self._cache["m"]

    @property
    def tasks(self):
        return self._mod().TASKS

    @property
    def max(self):
        return 3 * len(self.tasks)

    @property
    def seat(self):
        return self._mod().SEAT

    @property
    def speed(self):
        return getattr(self._mod(), "SPEED", None)

    @property
    def judge(self) -> Optional[Callable[..., Any]]:
        return getattr(self._mod(), "judge", None)

    def call(self, client, lane, variation, prompt_spec, tokens):
        return self._mod().call(client, lane, variation, prompt_spec, tokens)

    def score(self, response, checker):
        return self._mod().score(response, checker)


TEXT = Suite("hugpy-native-v2", "text", "text-generation")
VISION = Suite("hugpy-vision-v1", "vision", "image-text-to-text")
# ImageGenRequest carries no ``alloc``: a worker pin would be silently dropped,
# so imagegen is graded once per model on whatever seat hugpy places it.
IMAGEGEN = Suite("hugpy-imagegen-v1", "imagegen", "text-to-image", pinned=False)

SUITES = {
    "text-generation": TEXT,
    "text2text-generation": TEXT,
    "text-summarization": TEXT,
    "image-text-to-text": VISION,
    "text-to-image": IMAGEGEN,
}
SUITES_BY_NAME = {s.name: s for s in (TEXT, VISION, IMAGEGEN)}


def model_tasks(model_row):
    """The row's tasks, primary first."""
    row = model_row or {}
    tasks = [row.get("primary_task")] + list(row.get("tasks") or [])
    return [t for i, t in enumerate(tasks) if isinstance(t, str) and t and t not in tasks[:i]]


def suite_for_model(model_row):
    """The Suite for this catalog row, or None when no suite covers its task."""
    row = model_row or {}
    primary = row.get("primary_task")
    if primary:
        return SUITES.get(primary)
    for task in row.get("tasks") or []:
        if task in SUITES:
            return SUITES[task]
    return None


def suite_by_name(name):
    if name in SUITES_BY_NAME:
        return SUITES_BY_NAME[name]
    if name in SUITES:
        return SUITES[name]
    raise KeyError(f"unknown grading suite {name!r}; known: {sorted(SUITES_BY_NAME)}")
