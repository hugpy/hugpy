"""Task-runner registry: how media/video runners plug into the engine.

The engine dispatches by ``(framework, task)`` (``("gguf", "text-generation")``,
``("transformers", "automatic-speech-recognition")``, ...). Runners for
anything beyond plain chat live in other packages (``hugpy_media``,
``hugpy_video``) which the engine must not import. They register themselves
here at composition time::

    from hugpy_engine.tasks import register_task
    register_task(
        "automatic-speech-recognition",
        runner=WhisperRunner,                  # class, or a zero-arg factory
        build_request=_build_whisper_request,  # (kwargs, model_key) -> request
        frameworks=("transformers",),          # which registry frameworks it serves
        extra="audio",                         # pip extra that provides it
        source="hugpy_media",
    )

``frameworks`` may be left empty to serve the task for *every* framework
(stored under the ``"*"`` wildcard). The same task may be registered under
different frameworks by different packages: the engine registers
``("gguf", "image-text-to-text")`` (a vision GGUF rides the llama.cpp chat
path) and media registers ``("transformers", "image-text-to-text")``.

``runner`` is either the runner class itself or a zero-argument factory that
imports and returns it (so registration never pays for torch/diffusers);
``TaskSpec.runner_class()`` resolves it once. ``build_request`` is optional:
when a plugin registers only the runner, the engine's legacy builder for that
pair (``hugpy_engine.resolvers.categories.builders``) is used if one exists.

``hugpy_media.plugin.register()`` and ``hugpy_video.plugin.register()`` do this
for their tasks; the server and the worker entry points call them. Packages may
also expose a ``hugpy_engine.tasks`` entry point (a zero-arg callable) and the
engine loads those lazily the first time the runner table is consulted
(:func:`ensure_plugins_loaded`). Chat/text tasks that belong to the engine
itself register in ``hugpy_engine.resolvers.categories`` the same way, so
there is exactly one table. Unknown tasks resolve to ``None`` and callers
report "no runner installed for task" instead of importing anything.

Live views over the registry (``FRAMEWORK_RUNNERS``, ``MODEL_REQUEST_BUILDERS``,
``KNOWN_TASKS_REGISTRY``) are built with :class:`RunnerTable`,
:class:`BuilderTable` and :class:`TaskNames` so existing call sites keep their
dict/set semantics (``key in table``, ``table[key]``, ``sorted(table)``).
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping as _MappingABC
from collections.abc import Set as _SetABC
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

__all__ = [
    "ANY_FRAMEWORK",
    "TaskSpec",
    "register_task",
    "register_runner",
    "unregister_task",
    "task_spec",
    "runner_for_task",
    "runner_for",
    "request_builder_for_task",
    "registered_tasks",
    "registered_pairs",
    "load_entry_points",
    "ensure_plugins_loaded",
    "reset_tasks",
    "RunnerTable",
    "BuilderTable",
    "TaskNames",
]

RunnerFactory = Callable[[], Any]
RequestBuilder = Callable[[Mapping[str, Any], str], Any]

ANY_FRAMEWORK = "*"
ENTRY_POINT_GROUP = "hugpy_engine.tasks"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    runner: Any  # class or zero-arg factory
    build_request: Optional[RequestBuilder] = None
    frameworks: Sequence[str] = field(default_factory=tuple)
    extra: Optional[str] = None  # pip extra providing the heavy dependency
    source: str = ""  # package that registered it, for diagnostics

    def runner_class(self) -> Any:
        """The runner class: ``runner`` itself when it is a class, else the
        result of calling the zero-arg factory (memoised per spec)."""
        cached = _RESOLVED.get(id(self))
        if cached is not None:
            return cached
        target = self.runner
        if target is not None and not inspect.isclass(target) and callable(target):
            target = target()
        _RESOLVED[id(self)] = target
        return target


_lock = threading.RLock()
# task -> {framework | "*" -> TaskSpec}
_tasks: dict[str, dict[str, TaskSpec]] = {}
_RESOLVED: dict[int, Any] = {}
_entry_points_loaded = False


def register_task(
    task: str,
    *,
    runner: Any,
    build_request: Optional[RequestBuilder] = None,
    frameworks: Sequence[str] = (),
    extra: Optional[str] = None,
    source: str = "",
    replace: bool = True,
) -> TaskSpec:
    """Register ``runner`` (and optionally ``build_request``) for ``task``.

    ``frameworks`` names the registry frameworks the runner serves; empty means
    every framework (the ``"*"`` wildcard). With ``replace=False`` an existing
    registration for the same (framework, task) is kept and returned."""
    if not task:
        raise ValueError("task name is required")
    if isinstance(frameworks, str):
        frameworks = (frameworks,)
    fws = tuple(frameworks) or (ANY_FRAMEWORK,)
    spec = TaskSpec(task, runner, build_request, fws, extra, source)
    with _lock:
        slot = _tasks.setdefault(task, {})
        for fw in fws:
            if not replace and fw in slot:
                spec = slot[fw]
                continue
            slot[fw] = spec
    return spec


register_runner = register_task  # naming from EXTRACTION_GUIDE section 4


def unregister_task(task: str, framework: Optional[str] = None) -> None:
    with _lock:
        if framework is None:
            _tasks.pop(task, None)
            return
        slot = _tasks.get(task)
        if slot:
            slot.pop(framework, None)
            if not slot:
                _tasks.pop(task, None)


def task_spec(task: str, framework: Optional[str] = None) -> Optional[TaskSpec]:
    """The registration for ``task`` under ``framework`` (exact match first,
    then the ``"*"`` wildcard). With no framework: the wildcard registration if
    any, else the first one registered."""
    with _lock:
        slot = _tasks.get(task)
        if not slot:
            return None
        if framework is not None:
            return slot.get(framework) or slot.get(ANY_FRAMEWORK)
        if ANY_FRAMEWORK in slot:
            return slot[ANY_FRAMEWORK]
        return next(iter(slot.values()))


def runner_for_task(task: str, framework: Optional[str] = None) -> Optional[Any]:
    """The runner *class* for ``task`` (factories are resolved), or None."""
    spec = task_spec(task, framework)
    return spec.runner_class() if spec else None


runner_for = runner_for_task  # naming from EXTRACTION_GUIDE section 4


def request_builder_for_task(task: str, framework: Optional[str] = None) -> Optional[RequestBuilder]:
    spec = task_spec(task, framework)
    return spec.build_request if spec else None


def registered_tasks() -> Mapping[str, TaskSpec]:
    """task -> representative spec (see :func:`task_spec`)."""
    with _lock:
        return {task: task_spec(task) for task in list(_tasks)}


def registered_pairs() -> Mapping[tuple[str, str], TaskSpec]:
    """(framework, task) -> spec, wildcard registrations under ``"*"``."""
    with _lock:
        return {(fw, task): spec for task, slot in _tasks.items() for fw, spec in slot.items()}


def load_entry_points(group: str = ENTRY_POINT_GROUP) -> int:
    """Import every ``hugpy_engine.tasks`` entry point and call it.

    Each entry point is a zero-arg callable that registers its tasks. Returns
    the number of entry points loaded. Failures are isolated per plugin so one
    broken optional stack cannot take the engine down.
    """
    from importlib.metadata import entry_points

    loaded = 0
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - python < 3.10 shape
        eps = entry_points().get(group, [])
    for ep in eps:
        try:
            ep.load()()
            loaded += 1
        except Exception:  # noqa: BLE001 - plugin isolation
            import logging

            logging.getLogger(__name__).exception("task plugin %s failed to load", ep.name)
    return loaded


def ensure_plugins_loaded() -> None:
    """Load entry-point plugins once (idempotent, never raises).

    Called lazily by the registry views the first time a caller consults the
    runner/builder table, so ``import hugpy_engine`` never imports a plugin."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    with _lock:
        if _entry_points_loaded:
            return
        _entry_points_loaded = True
    try:
        load_entry_points()
    except Exception:  # noqa: BLE001 - discovery is best effort
        import logging

        logging.getLogger(__name__).debug("task entry-point discovery failed", exc_info=True)


def reset_tasks(*, reload_entry_points: bool = False) -> None:
    """Forget every registration (tests)."""
    global _entry_points_loaded
    with _lock:
        _tasks.clear()
        _RESOLVED.clear()
        if reload_entry_points:
            _entry_points_loaded = False


# ---------------------------------------------------------------------------
# Live views — keep the historical ``FRAMEWORK_RUNNERS`` /
# ``MODEL_REQUEST_BUILDERS`` / ``KNOWN_TASKS_REGISTRY`` call sites working
# against the registry without a static table.
# ---------------------------------------------------------------------------

class _PairTable(_MappingABC):
    """Mapping ``(framework, task) -> value`` over the registry.

    Exact pairs win; a ``("*", task)`` wildcard answers any framework. Iteration
    lists the registered pairs (wildcards as ``("*", task)``)."""

    def _value(self, spec: TaskSpec, key: tuple[str, str]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _lookup(self, key: Any) -> Optional[tuple[TaskSpec, tuple[str, str]]]:
        if not isinstance(key, tuple) or len(key) != 2:
            return None
        ensure_plugins_loaded()
        framework, task = key
        spec = task_spec(task, framework)
        if spec is None:
            return None
        return spec, (framework, task)

    def __getitem__(self, key: Any) -> Any:
        found = self._lookup(key)
        if found is None:
            raise KeyError(key)
        value = self._value(*found)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: Any) -> bool:
        found = self._lookup(key)
        return found is not None and self._value(*found) is not None

    def __iter__(self) -> Iterator[tuple[str, str]]:
        ensure_plugins_loaded()
        return iter([k for k, spec in registered_pairs().items() if self._value(spec, k) is not None])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self)})"


class RunnerTable(_PairTable):
    """``(framework, task) -> runner class`` view (``FRAMEWORK_RUNNERS``)."""

    def _value(self, spec: TaskSpec, key: tuple[str, str]) -> Any:
        try:
            return spec.runner_class()
        except Exception:  # noqa: BLE001 - a broken optional stack is "not installed"
            import logging

            logging.getLogger(__name__).warning(
                "runner for %s (%s) failed to import", key, spec.source or "?", exc_info=True)
            return None


class BuilderTable(_PairTable):
    """``(framework, task) -> request builder`` view (``MODEL_REQUEST_BUILDERS``).

    A registration without ``build_request`` falls back to ``fallback`` (the
    engine's legacy builders keyed by exact pair, then by ``("*", task)``)."""

    def __init__(self, fallback: Optional[Mapping[tuple[str, str], RequestBuilder]] = None) -> None:
        self._fallback = fallback or {}

    def _value(self, spec: TaskSpec, key: tuple[str, str]) -> Any:
        if spec.build_request is not None:
            return spec.build_request
        framework, task = key
        return self._fallback.get((framework, task)) or self._fallback.get((ANY_FRAMEWORK, task))


class TaskNames(_SetABC):
    """Set of registered task names (``KNOWN_TASKS_REGISTRY``)."""

    def _names(self) -> set[str]:
        ensure_plugins_loaded()
        return set(registered_tasks())

    def __contains__(self, item: Any) -> bool:
        return item in self._names()

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._names()))

    def __len__(self) -> int:
        return len(self._names())

    def __repr__(self) -> str:
        return f"TaskNames({sorted(self._names())})"
