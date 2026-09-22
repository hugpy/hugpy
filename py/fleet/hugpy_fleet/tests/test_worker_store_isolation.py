"""The tmpdir isolation helper really isolates the registry AND its sidecar.

Converted from the monolith's script-style ``test_worker_store_isolation.py``:
sections [2]-[5] (the ``PROJECTS_HOME`` idiom section is moot now that fleet
roots come from ``hugpy_fleet.central.config``).
"""

from __future__ import annotations

import json
import os

from hugpy_fleet.central import workers as W

from worker_store_isolation import isolated_worker_store, swap_worker_store


def _snapshot(path):
    if not os.path.isfile(path):
        return (False, None, None, None)
    st = os.stat(path)
    ids = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            ids = sorted((json.load(fh) or {}).keys())
    except Exception:  # noqa: BLE001
        pass
    return (True, st.st_mtime_ns, st.st_size, ids)


def test_isolated_worker_store_redirects_registry_and_sidecar():
    real_workers = W._default_workers_path()
    real_mem = W._assign_memory_path()
    before_w, before_m = _snapshot(real_workers), _snapshot(real_mem)

    store, tmp = isolated_worker_store(prefix="hugpy-isolation-proof-")
    assert store._path.startswith(tmp) and store._path != real_workers
    assert W._assign_memory_path().startswith(tmp)
    assert W._assign_memory_path() != real_mem

    v1 = store.register(name="ghost-a", url="http://192.0.2.50:9100",
                        worker_id="ghost-a", models=["Some~Model"])
    assert v1 is not None
    v1b = store.register(name="ghost-a", url="http://192.0.2.50:9100",
                         worker_id="ghost-a", models=["Some~Model", "Other~Model"])
    assert "Other~Model" in v1b["models"]
    store.register(name="ghost-b", url="http://192.0.2.51:9100", worker_id="ghost-b")
    av = store.assign_model("ghost-b", "Assigned~Model")
    assert av is not None and "Assigned~Model" in av["models"]
    uv = store.unassign_model("ghost-b", "Assigned~Model")
    assert uv is not None and "Assigned~Model" not in uv["models"]
    assert store.heartbeat("ghost-b", loaded_models=["Assigned~Model"]) is not None

    mem_path = W._assign_memory_path()
    assert os.path.isfile(mem_path) and mem_path.startswith(tmp)
    with open(mem_path, "r", encoding="utf-8") as fh:
        assert "ghost-a" in json.load(fh)

    # the real files (if any) were never touched
    after_w, after_m = _snapshot(real_workers), _snapshot(real_mem)
    assert before_w[0] == after_w[0]
    if before_w[0]:
        assert not ({"ghost-a", "ghost-b"} & set(after_w[3] or []))
    assert before_m[0] == after_m[0]


def test_swap_worker_store_drives_module_wrappers_and_restores():
    orig_store = W.worker_store
    orig_manifest = W.settings.manifest_path
    with swap_worker_store(prefix="hugpy-isolation-proof-swap-") as swapped:
        assert W.worker_store is swapped
        assert W.settings.manifest_path != orig_manifest
        W.worker_store.register(name="w1", url="http://192.0.2.60:9100", worker_id="w1")
        view = W.assign_model("w1", "Routed~Model")
        assert view is not None and "Routed~Model" in view["models"]
    assert W.worker_store is orig_store
    assert W.settings.manifest_path == orig_manifest
    assert W._default_workers_path() == os.path.join(
        os.path.dirname(orig_manifest), "workers.json")
