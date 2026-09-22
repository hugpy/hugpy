"""Task categories: the (framework, task) runner and request-builder tables.

Both are live views over :mod:`hugpy_engine.tasks`; importing this package
registers the engine's own chat runners (lazily imported) and nothing else.
"""
from hugpy_engine.resolvers.categories.builders import (
    LEGACY_BUILDERS,
    MODEL_REQUEST_BUILDERS,
)
from hugpy_engine.resolvers.categories.frameworks import (
    FRAMEWORK_RUNNERS,
    KNOWN_TASKS_REGISTRY,
    register_builtin_tasks,
)

__all__ = [
    "FRAMEWORK_RUNNERS",
    "KNOWN_TASKS_REGISTRY",
    "LEGACY_BUILDERS",
    "MODEL_REQUEST_BUILDERS",
    "register_builtin_tasks",
]
