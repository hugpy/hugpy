"""Central external model metadata store — fetch-once, permanent,
cache-never-a-gate. (Renamed from test_hf_metadata_cache.py when the store
widened from HF-only to HF + Civitai — comms/model_metadata.py.)

Covers the operator-ratified policy (a cache miss queries the provider ONCE;
the response lands in the central SQLite DB; that model/repo is theoretically
never pinged again) plus the revised 07-22 ruling that /search's list_models
stays LIVE on every query (search caching was dropped mid-build):

  1. miss -> ONE live fetch -> hit (live called exactly once across N reads),
  2. force=True refresh overwrites in place (second live call, cache updated),
  3. degradation: an unwritable DB path never breaks the live path (cache is
     a cache, not a gate),
  4. forget() re-arms a live re-fetch (the operator refresh hatch),
  5. repo_files read-through (the _fp16_ignore_patterns backing),
  6. model_size() rides the permanent cache (one live call, then zero),
  7. /search calls list_models LIVE even on an identical repeat query,
  8. serialize_model_info flattens an HF-shaped object defensively,
  9. civitai_meta roundtrip + fetch_civitai_meta fetch-once semantics
     (miss-with-id fetches once; miss-without-id = NO network EVER;
     failures not cached; provenance-only rows upgrade),
 10. sidecar-driven comfy-sweep enrichment (with sidecar -> decorated row;
     without -> row byte-identical to today's),
 11. rename shims (old import path, old env var, old-db-file migration).

NO test touches the network — every live surface is a fake throughout.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_storage.model_metadata import (
    ModelMetadataStore,
    checkpoint_stem,
    fetch_civitai_meta,
    fetch_repo_info,
    model_metadata_store,
    serialize_civitai_model,
    serialize_model_info,
    sum_sibling_sizes,
)


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeSibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


class FakeInfo:
    def __init__(self, hub_id="owner/repo", sizes=((("a.safetensors", 100),
                                                    ("b.gguf", 50)))):
        self.id = hub_id
        self.modelId = hub_id
        self.sha = "abc123"
        self.author = hub_id.split("/")[0]
        self.pipeline_tag = "text-generation"
        self.library_name = "transformers"
        self.tags = ["gguf"]
        self.card_data = {"license": "apache-2.0", "language": "en"}
        self.gated = False
        self.private = False
        self.downloads = 7
        self.likes = 3
        self.last_modified = None
        self.created_at = None
        self.safetensors = SimpleNamespace(total=1234)
        self.transformers_info = SimpleNamespace(auto_model="AutoModelForCausalLM")
        self.siblings = [FakeSibling(n, s) for n, s in sizes]


class FakeApi:
    """Counts model_info calls so 'live called exactly once' is provable."""

    def __init__(self, info=None, fail=False):
        self.calls = 0
        self.info = info or FakeInfo()
        self.fail = fail

    def model_info(self, hub_id, files_metadata=True):
        self.calls += 1
        if self.fail:
            raise RuntimeError("HF down")
        return self.info


def _store():
    return ModelMetadataStore(os.path.join(
        tempfile.mkdtemp(prefix="modelmeta-"), "model_metadata.db"))


def _use(store, monkeypatch):
    """Point the module singleton (which fetch_repo_info / fetch_civitai_meta
    funnel through) at a throwaway store for one test."""
    import hugpy_storage.model_metadata as m
    monkeypatch.setattr(m, "model_metadata_store", store)
    return store


# ── 1. miss -> fetch -> hit: live called exactly once ────────────────────────
def test_miss_fetches_once_then_hits(monkeypatch):
    _use(_store(), monkeypatch)
    api = FakeApi()
    first = fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert first is not None and first["id"] == "owner/repo"
    assert api.calls == 1
    for _ in range(5):
        again = fetch_repo_info("owner/repo", purpose="discovery", api=api)
        assert again["id"] == "owner/repo"
        assert again["siblings"] == first["siblings"]
    assert api.calls == 1, "cache hit must cost ZERO live HF calls"
    assert sum_sibling_sizes(first) == 150


# ── 2. force refresh overwrites in place ─────────────────────────────────────
def test_force_refresh_hits_live_and_overwrites(monkeypatch):
    store = _use(_store(), monkeypatch)
    api = FakeApi()
    fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert api.calls == 1
    api.info = FakeInfo(sizes=(("a.safetensors", 999),))
    forced = fetch_repo_info("owner/repo", purpose="discovery", api=api, force=True)
    assert api.calls == 2
    assert sum_sibling_sizes(forced) == 999
    # the overwrite persisted — a plain read now serves the NEW row, no live
    cached = fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert api.calls == 2
    assert sum_sibling_sizes(cached) == 999
    assert store.get_repo_info("owner/repo")["siblings"][0]["size"] == 999


# ── 3. degradation: broken cache never breaks the live path ──────────────────
def test_unwritable_db_degrades_to_live(monkeypatch):
    # a path under a file (not a dir) can never be created -> every store op
    # fails -> the cache must be a no-op, never an exception surface
    bad_parent = os.path.join(tempfile.mkdtemp(prefix="hfmeta-bad-"), "f")
    with open(bad_parent, "w") as fh:
        fh.write("not a dir")
    store = _use(ModelMetadataStore(os.path.join(bad_parent, "x", "meta.db")),
                 monkeypatch)
    api = FakeApi()
    out = fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert out is not None and out["id"] == "owner/repo"
    assert api.calls == 1
    # every call goes live (no cache), but ALWAYS succeeds
    out2 = fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert out2 is not None
    assert api.calls == 2
    # store surface itself stays inert, not raising
    assert store.get_repo_info("owner/repo") is None
    assert store.get_repo_files("owner/repo") is None
    store.put_repo_files("owner/repo", ["a"])          # swallowed
    assert store.forget("owner/repo") == 0
    stats = store.stats()
    assert stats["repos"] == 0 and "db_path" in stats


# ── 4. forget -> refetch ─────────────────────────────────────────────────────
def test_forget_rearms_live_fetch(monkeypatch):
    store = _use(_store(), monkeypatch)
    api = FakeApi()
    fetch_repo_info("owner/repo", purpose="discovery", api=api)
    store.put_repo_files("owner/repo", ["a", "b"])
    assert api.calls == 1
    removed = store.forget("owner/repo")
    assert removed == 2, "forget drops BOTH repo_info and repo_files rows"
    fetch_repo_info("owner/repo", purpose="discovery", api=api)
    assert api.calls == 2, "post-forget access re-fetches live"
    # other repos untouched
    fetch_repo_info("other/repo", purpose="discovery", api=api)
    assert store.forget("owner/repo") >= 1
    assert store.get_repo_info("other/repo") is not None


# ── 5. repo_files read-through ───────────────────────────────────────────────
def test_repo_files_roundtrip():
    store = _store()
    assert store.get_repo_files("o/r") is None
    store.put_repo_files("o/r", ["model_index.json", "x.safetensors"])
    assert store.get_repo_files("o/r") == ["model_index.json", "x.safetensors"]
    stats = store.stats()
    assert stats["file_lists"] == 1


# ── 6. model_size rides the permanent cache ──────────────────────────────────
def test_model_size_uses_cache(monkeypatch):
    _use(_store(), monkeypatch)
    from hugpy_server.app.functions.imports.options import search as opt_search
    from hugpy_server.app.functions.imports.utils import constants as hf_const
    api = FakeApi()
    monkeypatch.setattr(hf_const, "hfApi", api)
    assert opt_search.model_size("owner/repo") == 150
    assert api.calls == 1
    assert opt_search.model_size("owner/repo") == 150
    assert api.calls == 1, "second model_size must be served from the cache"
    # live failure on a MISS -> None (old contract), never an exception
    failing = FakeApi(fail=True)
    monkeypatch.setattr(hf_const, "hfApi", failing)
    assert opt_search.model_size("never/seen") is None


# ── 7. /search stays LIVE on repeat queries (operator ruling 07-22) ──────────
def test_search_route_always_calls_list_models_live(monkeypatch):
    _use(_store(), monkeypatch)
    from hugpy_server.app.routes import search_routes as sr

    class FakeListModel:
        modelId = "owner/repo"
        author = "owner"
        downloads = 1
        likes = 1
        tags = []
        pipeline_tag = "text-generation"
        library_name = "transformers"
        private = False
        last_modified = ""
        created_at = ""

    class FakeSearchApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.list_calls = 0

        def list_models(self, **kw):
            self.list_calls += 1
            return [FakeListModel()]

    api = FakeSearchApi()
    monkeypatch.setattr(sr, "api", api)
    from hugpy_server.wsgi_app import get_hugpy_flask
    client = get_hugpy_flask().test_client()
    r1 = client.get("/search?q=repo&with_size=0")
    r2 = client.get("/search?q=repo&with_size=0")
    assert r1.status_code == 200 and r2.status_code == 200
    assert api.list_calls == 2, "search must ping HF live on EVERY query"
    body = r1.get_json()
    assert body and body[0]["hub_id"] == "owner/repo"
    assert "cached_at" not in body[0], "search responses carry no cache marker"


# ── 8. defensive serialization ───────────────────────────────────────────────
def test_serialize_model_info_defensive():
    p = serialize_model_info(FakeInfo())
    json.dumps(p)                                       # JSON-native
    assert p["license"] == "apache-2.0"
    assert p["languages"] == ["en"]
    assert p["safetensors_params"] == 1234
    assert p["auto_model_class"] == "AutoModelForCausalLM"
    assert p["siblings"] == [{"rfilename": "a.safetensors", "size": 100},
                             {"rfilename": "b.gguf", "size": 50}]
    # a hollow object degrades to Nones, never raises
    hollow = serialize_model_info(SimpleNamespace())
    assert hollow["id"] is None and hollow["siblings"] == []


# ── 9. civitai_meta: roundtrip + fetch-once semantics ────────────────────────
_CIVITAI_MODEL_PAYLOAD = {
    "id": 4201, "name": "DreamShaper", "type": "Checkpoint", "nsfw": False,
    "tags": ["base model", "art"],
    "modelVersions": [
        {"id": 128713, "name": "v8", "baseModel": "SD 1.5",
         "trainedWords": [], "files": [
             {"name": "dreamshaper_8.safetensors", "sizeKB": 2082666,
              "type": "Model"}]},
    ],
}


class ExplodingFetcher:
    """A network surface that must never be touched."""

    def __call__(self, url, headers=None, timeout=None):
        raise AssertionError(f"NETWORK CALL ATTEMPTED: {url}")


class CountingFetcher:
    def __init__(self, payload=None, fail=False):
        self.calls = 0
        self.payload = payload if payload is not None else _CIVITAI_MODEL_PAYLOAD
        self.fail = fail

    def __call__(self, url, headers=None, timeout=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("civitai down")
        return self.payload


def test_civitai_meta_roundtrip():
    store = _store()
    assert store.get_civitai_meta("dreamshaper-8") is None
    store.put_civitai_meta("dreamshaper-8", {"civitai_id": 4201, "name": "DS"})
    got = store.get_civitai_meta("dreamshaper-8")
    assert got == {"civitai_id": 4201, "name": "DS"}
    assert store.stats()["civitai_models"] == 1
    # overwrite in place
    store.put_civitai_meta("dreamshaper-8", {"civitai_id": 4201, "name": "DS8"})
    assert store.get_civitai_meta("dreamshaper-8")["name"] == "DS8"
    assert store.stats()["civitai_models"] == 1


def test_fetch_civitai_miss_with_id_fetches_once_then_hits(monkeypatch):
    _use(_store(), monkeypatch)
    f = CountingFetcher()
    first = fetch_civitai_meta("dreamshaper-8", civitai_id=4201, fetcher=f)
    assert f.calls == 1
    assert first["civitai_id"] == 4201
    assert first["name"] == "DreamShaper"
    assert first["base_model"] == "SD 1.5"
    assert first["version_id"] == 128713
    assert first["page_url"] == "https://civitai.com/models/4201"
    for _ in range(5):
        again = fetch_civitai_meta("dreamshaper-8", civitai_id=4201,
                                   fetcher=ExplodingFetcher())
        assert again == first
    assert f.calls == 1, "cache hit must cost ZERO live Civitai calls"


def test_fetch_civitai_miss_without_id_no_network(monkeypatch):
    _use(_store(), monkeypatch)
    # NEVER search by filename guess — a miss without any held id is None and
    # the network surface is provably untouched (the fake would explode).
    assert fetch_civitai_meta("mystery-checkpoint",
                              fetcher=ExplodingFetcher()) is None


def test_fetch_civitai_failure_not_cached(monkeypatch):
    _use(_store(), monkeypatch)
    failing = CountingFetcher(fail=True)
    assert fetch_civitai_meta("ds", civitai_id=4201, fetcher=failing) is None
    assert failing.calls == 1
    # second call RETRIES (failure was not cached), then succeeds and caches
    ok = CountingFetcher()
    got = fetch_civitai_meta("ds", civitai_id=4201, fetcher=ok)
    assert ok.calls == 1 and got["name"] == "DreamShaper"
    again = fetch_civitai_meta("ds", civitai_id=4201,
                               fetcher=ExplodingFetcher())
    assert again == got


def test_provenance_only_row_upgrades_once(monkeypatch):
    store = _use(_store(), monkeypatch)
    # the download-time stamp (thin, provenance_only) is a hit for id-less
    # readers but upgrades via ONE live call when enrichment runs
    store.put_civitai_meta("ds", {"civitai_id": 4201, "name": "DreamShaper",
                                  "provenance_only": True})
    f = CountingFetcher()
    got = fetch_civitai_meta("ds", fetcher=f)   # ids ride the stamp itself
    assert f.calls == 1
    assert got["base_model"] == "SD 1.5" and not got.get("provenance_only")
    # upgraded row is now the cached truth — no further calls
    assert fetch_civitai_meta("ds", fetcher=ExplodingFetcher()) == got


def test_serialize_civitai_version_shape():
    # /model-versions/<vid> shape (bare version with embedded model)
    ver = {"id": 128713, "name": "v8", "baseModel": "SD 1.5", "modelId": 4201,
           "model": {"name": "DreamShaper", "type": "Checkpoint",
                     "nsfw": False},
           "files": [{"name": "a.safetensors", "sizeKB": 1024}]}
    out = serialize_civitai_model(ver, version_id=128713)
    assert out["civitai_id"] == 4201 and out["version_id"] == 128713
    assert out["name"] == "DreamShaper" and out["base_model"] == "SD 1.5"
    assert out["files"] == [{"name": "a.safetensors", "size_bytes": 1048576,
                             "type": None}]
    # garbage degrades, never raises, and stays JSON-native
    assert serialize_civitai_model(None) == {}
    json.dumps(out)
    json.dumps(serialize_civitai_model({"modelVersions": "not-a-list"}))


def test_checkpoint_stem_matches_sweep_slug():
    assert checkpoint_stem("DreamShaper_8 (fp16).safetensors") == \
        "dreamshaper-8-fp16"
    assert checkpoint_stem("/x/y/Model.CKPT".lower()) == "model"


# ── 10. sidecar-driven sweep enrichment ──────────────────────────────────────
def _run_sweep(tmp_path, monkeypatch, sidecar=None, fetcher=None):
    """Run _sweep_comfy_checkpoints against a throwaway root with one fake
    checkpoint (+ optional sidecar), all path/marker plumbing stubbed to stay
    inside tmp_path, and the civitai fetch pointed at ``fetcher`` (an
    ExplodingFetcher proves the no-sidecar path performs NO network)."""
    import importlib
    import hugpy_engine.config.models.models_config as mc
    import hugpy_storage.model_metadata as mm
    # `import pkg.mod as x` trips over this package's attribute shadowing —
    # importlib resolves the actual module objects reliably.
    consts = importlib.import_module(
        "hugpy_platform.constants")
    paths = importlib.import_module(
        "hugpy_storage.model_paths")
    marker = importlib.import_module(
        "hugpy_storage.hugpy_marker")

    root = tmp_path / "store"
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "coolmodel_v2.safetensors").write_bytes(b"\x00fake")
    if sidecar is not None:
        (ckpt_dir / "coolmodel_v2.safetensors.civitai.json").write_text(
            sidecar if isinstance(sidecar, str) else json.dumps(sidecar))

    monkeypatch.setattr(consts, "DEFAULT_ROOT", str(root))
    monkeypatch.setattr(
        paths, "route_destination",
        lambda row, root_=None: str(tmp_path / "layout" / row["name"]))
    monkeypatch.setattr(marker, "write_hugpy_marker",
                        lambda directory, **kw: None)
    if fetcher is not None:
        real = mm.fetch_civitai_meta
        monkeypatch.setattr(
            mm, "fetch_civitai_meta",
            lambda stem, **kw: real(stem, fetcher=fetcher,
                                    **{k: v for k, v in kw.items()
                                       if k != "fetcher"}))
    return mc._sweep_comfy_checkpoints({})


def test_sweep_without_sidecar_row_identical_to_today(tmp_path, monkeypatch):
    _use(_store(), monkeypatch)
    rows = _run_sweep(tmp_path, monkeypatch, sidecar=None,
                      fetcher=ExplodingFetcher())   # NO network, ever
    row = rows["comfy-coolmodel-v2"]
    assert row == {"model_max_length": 77, "include": None,
                   "name": "comfy-coolmodel-v2", "framework": "comfy",
                   "hub_id": "comfy/coolmodel-v2",
                   "filename": "coolmodel_v2.safetensors",
                   "folder": "comfy/coolmodel-v2",
                   "tasks": ["text-to-image", "image-to-image"],
                   "primary_task": "text-to-image", "port": None}


def test_sweep_with_sidecar_decorates_row(tmp_path, monkeypatch):
    store = _use(_store(), monkeypatch)
    sidecar = {"civitai_id": 4201, "version_id": 128713,
               "name": "Cool Model", "base_model": "SDXL 1.0",
               "provenance_only": True}
    f = CountingFetcher()
    rows = _run_sweep(tmp_path, monkeypatch, sidecar=sidecar, fetcher=f)
    row = rows["comfy-coolmodel-v2"]
    # identity unchanged; decoration added
    assert row["name"] == "comfy-coolmodel-v2"
    assert row["hub_id"] == "comfy/coolmodel-v2"
    assert row["display_name"] == "Cool Model"
    assert row["civitai_id"] == 4201
    assert row["civitai_version_id"] == 128713
    assert row["civitai_base_model"] == "SDXL 1.0"
    assert "base_model" not in row, "base_model means PEFT adapter — must never be set by decoration"
    assert row["model_max_length"] == 77, "token length deliberately untouched"
    # the sweep warmed the central table (one fetch)
    assert f.calls == 1
    assert store.get_civitai_meta("coolmodel-v2")["name"] == "DreamShaper"


def test_sweep_with_corrupt_sidecar_degrades(tmp_path, monkeypatch):
    _use(_store(), monkeypatch)
    rows = _run_sweep(tmp_path, monkeypatch, sidecar="{not json!!",
                      fetcher=ExplodingFetcher())
    row = rows["comfy-coolmodel-v2"]
    assert "display_name" not in row and "civitai_base_model" not in row


def test_sweep_fetch_failure_degrades_to_unenriched(tmp_path, monkeypatch):
    _use(_store(), monkeypatch)
    sidecar = {"civitai_id": 4201, "name": "Cool Model",
               "base_model": "SD 1.5"}
    rows = _run_sweep(tmp_path, monkeypatch, sidecar=sidecar,
                      fetcher=CountingFetcher(fail=True))
    row = rows["comfy-coolmodel-v2"]
    # sidecar decoration survives; the failed warm-up is invisible
    assert row["display_name"] == "Cool Model"
    assert row["civitai_base_model"] == "SD 1.5"


# ── 11. rename shims ─────────────────────────────────────────────────────────
def test_old_import_path_still_works():
    import hugpy_storage.hf_metadata as legacy
    import hugpy_storage.model_metadata as new
    assert legacy.HfMetadataStore is new.ModelMetadataStore
    assert legacy.hf_metadata_store is new.model_metadata_store
    assert legacy.fetch_repo_info is new.fetch_repo_info
    assert legacy.fetch_civitai_meta is new.fetch_civitai_meta


def test_legacy_env_var_honored_and_new_wins(monkeypatch):
    from hugpy_storage.model_metadata import default_db_path
    monkeypatch.delenv("HUGPY_MODEL_METADATA_DB", raising=False)
    monkeypatch.setenv("HUGPY_HF_CACHE_DB", "/tmp/legacy.db")
    assert default_db_path() == "/tmp/legacy.db"
    monkeypatch.setenv("HUGPY_MODEL_METADATA_DB", "/tmp/new.db")
    assert default_db_path() == "/tmp/new.db", "new env var must win"
    monkeypatch.delenv("HUGPY_HF_CACHE_DB", raising=False)
    assert default_db_path() == "/tmp/new.db"


def test_legacy_db_file_migrates_on_init(tmp_path):
    legacy = tmp_path / "hf_metadata.db"
    # seed a real legacy DB with one row via the store itself
    seed = ModelMetadataStore(str(legacy))
    seed.put_repo_info("owner/repo", {"id": "owner/repo"})
    assert legacy.exists()
    # a store pointed at the NEW default name beside it migrates the file
    store = ModelMetadataStore(str(tmp_path / "model_metadata.db"))
    assert store.get_repo_info("owner/repo") == {"id": "owner/repo"}
    assert not legacy.exists(), "old file renamed in place (one-shot)"
    assert (tmp_path / "model_metadata.db").exists()


def test_no_migration_when_new_db_exists(tmp_path):
    # the new DB exists FIRST — a legacy file appearing beside it later must
    # never clobber it (migration is strictly absent-new + present-old)
    new = ModelMetadataStore(str(tmp_path / "model_metadata.db"))
    new.put_repo_info("new/repo", {"id": "new/repo"})
    old = ModelMetadataStore(str(tmp_path / "hf_metadata.db"))
    old.put_repo_info("old/repo", {"id": "old/repo"})
    again = ModelMetadataStore(str(tmp_path / "model_metadata.db"))
    assert again.get_repo_info("new/repo") is not None
    assert again.get_repo_info("old/repo") is None
    assert (tmp_path / "hf_metadata.db").exists(), "legacy file left alone"
