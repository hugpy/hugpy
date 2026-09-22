"""Test helper: an isolated, tmpdir-backed worker registry.

``hugpy_fleet.central.config.settings`` is process-wide; redirecting its
``manifest_path`` (and thereby ``state_dir``) moves ``workers.json``, the
assignment-memory sidecar and every other fleet state file into a tmpdir so
no test can reach a live registry.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from contextlib import contextmanager
from typing import Iterator, Tuple

W = importlib.import_module("hugpy_fleet.central.workers")


def _isolated_paths(prefix: str) -> Tuple[str, str, str]:
    tmp = tempfile.mkdtemp(prefix=prefix)
    return tmp, os.path.join(tmp, "workers.json"), os.path.join(tmp, "model_manifest.json")


def isolated_worker_store(prefix: str = "hugpy-test-workers-"):
    """A fresh ``WorkerStore`` in a tmpdir; also redirects the assignment-memory
    sidecar there for the rest of the process (not restored). Returns
    ``(store, tmp_dir)``."""
    tmp, workers_path, manifest_path = _isolated_paths(prefix)
    W.settings.manifest_path = manifest_path
    return W.WorkerStore(path=workers_path), tmp


@contextmanager
def swap_worker_store(prefix: str = "hugpy-test-workers-") -> Iterator["W.WorkerStore"]:
    """Isolate AND swap the module-level ``W.worker_store`` singleton (what the
    module-level wrappers and the placement registry use); restores on exit."""
    tmp, workers_path, manifest_path = _isolated_paths(prefix)
    orig_store = W.worker_store
    orig_manifest_path = W.settings.manifest_path
    W.settings.manifest_path = manifest_path
    W.worker_store = W.WorkerStore(path=workers_path)
    try:
        yield W.worker_store
    finally:
        W.worker_store = orig_store
        W.settings.manifest_path = orig_manifest_path
