"""Catalog bridge: the engine's registry becomes storage's catalog source,
the hot cache becomes storage's serve-path hook, and a ``catalog.changed``
event refreshes discovery. ``install()`` is idempotent and the facade's
default backend installs it lazily.
"""
from __future__ import annotations

import time

import pytest

from hugpy_engine import catalog_bridge as CB


@pytest.fixture
def clean_bridge():
    CB.uninstall()
    yield
    CB.uninstall()


def test_install_wires_storage_and_control(clean_bridge):
    from hugpy_control.bus import TOPIC_CATALOG_CHANGED, bus
    from hugpy_storage import providers
    from hugpy_storage.catalog_source import get_catalog_source

    assert CB.install() is True
    assert CB.installed()
    assert CB.install() is True                      # idempotent
    assert isinstance(get_catalog_source(), CB.EngineCatalogSource)
    hook = providers.get_serve_path_hook()
    assert hook("/nonexistent/model.gguf") == "/nonexistent/model.gguf"   # hot cache off -> identity

    # a catalog.changed event reaches the listener and triggers ONE refresh
    seen = []
    orig = CB._source.refresh
    CB._source.refresh = lambda: seen.append(time.time())
    try:
        bus.publish(TOPIC_CATALOG_CHANGED, source="test", payload={"change": "download", "model_key": "m"})
        for _ in range(50):
            if seen:
                break
            time.sleep(0.05)
        assert len(seen) == 1
    finally:
        CB._source.refresh = orig

    CB.uninstall()
    assert not CB.installed()
    assert not isinstance(get_catalog_source(), CB.EngineCatalogSource)


def test_engine_catalog_source_answers_from_the_live_registry(clean_bridge):
    from hugpy_engine.config.models import models_config as mc

    src = CB.EngineCatalogSource()
    rows = src.rows()
    assert isinstance(rows, dict)
    assert set(rows) == set(mc.MODEL_REGISTRY_DICT)
    assert src.canonical_key("zz-definitely-not-a-model") is None
    assert src.get("zz-definitely-not-a-model") is None
    assert src.resolve_dir("zz-definitely-not-a-model") is None
    assert src.register("", {}) is False
    if rows:
        key = next(iter(rows))
        assert src.canonical_key(key) == key
        cfg = src.get(key)
        assert cfg is not None and cfg.model_key == key


def test_register_inserts_a_central_row(clean_bridge):
    from hugpy_engine.config.models import models_config as mc

    src = CB.EngineCatalogSource()
    key = "zz-bridge-probe"
    row = {"hub_id": "zz/zz-bridge-probe", "framework": "gguf", "primary_task": "text-generation",
           "tasks": ["text-generation"], "name": key, "folder": key, "model_key": key,
           "filename": "zz-bridge-probe.Q4_K_M.gguf"}
    try:
        ok = src.register(key, row)
        assert ok is True
        assert key in mc.MODEL_REGISTRY and key in mc.MODEL_REGISTRY_DICT
        assert src.canonical_key(key) == key
    finally:
        mc.MODEL_REGISTRY.pop(key, None)
        mc.MODEL_REGISTRY_DICT.pop(key, None)


def test_local_backend_is_the_default_and_installs_the_bridge(clean_bridge):
    import hugpy_engine as engine
    from hugpy_engine.backends import LocalBackend

    engine.configure_backend(None)
    backend = engine.get_backend()
    assert isinstance(backend, LocalBackend)
    assert CB.installed()
    rows = backend.catalog_rows()
    assert isinstance(rows, dict)
    assert isinstance(backend.supported_tasks(), tuple)
    assert ("gguf", "text-generation") in backend.supported_tasks()
