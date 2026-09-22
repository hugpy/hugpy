"""The catalog seam: null default, install/reset, and the storage code paths
that must degrade honestly when no registry is installed."""
import os

import pytest

from hugpy_storage import catalog_source as cs


@pytest.fixture(autouse=True)
def _reset():
    cs.reset_catalog_source()
    yield
    cs.reset_catalog_source()


def test_null_default_knows_nothing():
    src = cs.get_catalog_source()
    assert isinstance(src, cs.NullCatalogSource)
    assert isinstance(src, cs.CatalogSource)          # runtime-checkable Protocol
    assert cs.catalog_rows() == {}
    assert cs.catalog_get("anything") is None
    assert cs.catalog_canonical_key("anything") is None
    assert cs.catalog_resolve_dir("anything") is None
    assert cs.catalog_register("k", {"hub_id": "o/r"}) is False
    cs.catalog_refresh()                              # no-op, never raises


def test_set_and_reset_catalog_source():
    d = cs.DictCatalogSource({"m": {"hub_id": "org/m", "framework": "gguf"}})
    cs.set_catalog_source(d)
    assert cs.get_catalog_source() is d
    assert cs.catalog_canonical_key("org/m") == "m"
    assert cs.catalog_canonical_key("m") == "m"
    assert cs.catalog_get("m").hub_id == "org/m"
    assert cs.catalog_rows() == {"m": {"hub_id": "org/m", "framework": "gguf"}}
    assert cs.catalog_register("n", {"hub_id": "org/n"}) is True
    assert "n" in cs.catalog_rows()
    cs.catalog_refresh()
    assert d.refreshed == 1
    cs.set_catalog_source(None)
    assert isinstance(cs.get_catalog_source(), cs.NullCatalogSource)


def test_guarded_accessors_swallow_implementation_errors():
    class Broken:
        name = "broken"
        def rows(self): raise RuntimeError("x")
        def get(self, k): raise RuntimeError("x")
        def canonical_key(self, k): raise RuntimeError("x")
        def resolve_dir(self, k): raise RuntimeError("x")
        def register(self, k, r): raise RuntimeError("x")
        def refresh(self): raise RuntimeError("x")
    cs.set_catalog_source(Broken())
    assert cs.catalog_rows() == {}
    assert cs.catalog_get("k") is None
    assert cs.catalog_canonical_key("k") is None
    assert cs.catalog_resolve_dir("k") is None
    assert cs.catalog_register("k", {}) is False
    cs.catalog_refresh()


def test_dict_source_resolves_dir_through_storage_layout(tmp_path):
    root = str(tmp_path)
    d = cs.DictCatalogSource({"m": {"hub_id": "org/m", "framework": "transformers",
                                    "primary_task": "text-generation"}}, root=root)
    cs.set_catalog_source(d)
    flat = os.path.join(root, "models", "transformers", "org", "m")
    # nothing on disk: the flat write target (require_complete=False semantics)
    assert cs.catalog_resolve_dir("m") == flat


def test_routing_of_reads_objects_and_dicts():
    from types import SimpleNamespace
    obj = SimpleNamespace(hub_id="o/r", framework="gguf", filename="q4.gguf",
                          primary_task="text-generation", tasks=None, include=None,
                          folder=None, dir=None, name="r")
    r = cs.routing_of(obj)
    assert r["hub_id"] == "o/r" and r["filename"] == "q4.gguf"
    r2 = cs.routing_of({"hub_id": "o/r", "task": "misc"}, framework="misc")
    assert r2["primary_task"] == "misc" and r2["framework"] == "misc"


def test_provision_degrades_without_catalog(tmp_path, monkeypatch):
    """With the null source, provision's registry helpers answer 'unknown'
    rather than raising, and ensure_model refuses an unknown key loudly."""
    from hugpy_storage import provision, download_models
    assert provision._assure_local_key("x") is None
    assert provision._register_local_model("x", {"hub_id": "o/x"}) is False
    assert provision.model_is_local("x") is False
    assert provision._dest_hint("x") is None
    with pytest.raises(KeyError):
        download_models.ensure_model("x", root=str(tmp_path))


def test_ensure_model_registered_uses_central_row_when_local_unknown(monkeypatch):
    from hugpy_storage import provision
    monkeypatch.setattr(provision, "_fetch_central_model_row",
                        lambda url, key: {"key": "m", "hub_id": "org/m"})
    # null catalog: the row cannot be registered -> None (honest), no raise
    assert provision.ensure_model_registered("m", "http://central") is None
    # with a registry, the row lands and the canonical key comes back
    cs.set_catalog_source(cs.DictCatalogSource())
    assert provision.ensure_model_registered("m", "http://central") == "m"
    assert cs.catalog_get("m").hub_id == "org/m"
