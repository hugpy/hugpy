import json
import hashlib
from pathlib import Path


def test_host_scan_records_external_pointer_without_moving_weights(tmp_path, monkeypatch):
    from hugpy_fleet.worker import system_models
    from hugpy_platform import constants

    host = tmp_path / "host"
    model = host / "elsewhere" / "Example-GGUF"
    model.mkdir(parents=True)
    weight = model / "example-Q4_K_M.gguf"
    with weight.open("wb") as fh:
        fh.truncate(2 * 1024 * 1024)
    managed = host / "hugpy" / "models"
    managed.mkdir(parents=True)
    (managed / "managed.gguf").write_bytes(b"x")
    inventory = tmp_path / "inventory"
    monkeypatch.setenv("HUGPY_WORKER_MODEL_SCAN_ROOTS", str(host))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_INVENTORY_DIR", str(inventory))
    monkeypatch.setattr(constants, "MODELS_HOME", str(managed))

    rows = system_models.scan_system_models()
    assert set(rows) == {"Example-GGUF"}
    row = rows["Example-GGUF"]
    assert row["external_location"] == str(model)
    assert row["size_bytes"] == weight.stat().st_size
    pointers = list(inventory.glob("*/hugpy.json"))
    assert len(pointers) == 1
    assert json.loads(pointers[0].read_text())["location"] == str(model)
    assert not (model / "hugpy.json").exists()


def test_host_scan_excludes_archives_and_incomplete_gguf_splits(tmp_path, monkeypatch):
    from hugpy_fleet.worker import system_models
    from hugpy_platform import constants

    host = tmp_path / "host"
    complete = host / "live" / "Complete-GGUF" / "Q4_K_M"
    partial = host / "partial" / "Partial-GGUF" / "Q4_K_M"
    archived = host / "models" / "_archive" / "dedupes" / "Archived-GGUF" / "Q4_K_M"
    for directory in (complete, partial, archived):
        directory.mkdir(parents=True)
    for number in (1, 2):
        (complete / f"complete-Q4_K_M-{number:05d}-of-00002.gguf").write_bytes(b"x" * (2 * 1024 * 1024))
    (partial / "partial-Q4_K_M-00001-of-00002.gguf").write_bytes(b"x" * (2 * 1024 * 1024))
    (archived / "archived-Q4_K_M-00001-of-00002.gguf").write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_SCAN_ROOTS", str(host))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_INVENTORY_DIR", str(tmp_path / "inventory"))
    monkeypatch.setattr(constants, "MODELS_HOME", str(tmp_path / "managed"))

    rows = system_models.scan_system_models()
    assert list(rows) == ["Complete-GGUF"]
    assert rows["Complete-GGUF"]["external_location"] == str(complete.parent)


def test_host_scan_qualifies_name_collision_with_different_hub(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hugpy_fleet.worker import system_models
    from hugpy_engine.config.models import models_config as mc
    from hugpy_platform import constants

    host = tmp_path / "host"
    model = host / "Other" / "Same-GGUF"
    model.mkdir(parents=True)
    with (model / "model.gguf").open("wb") as fh:
        fh.truncate(2 * 1024 * 1024)
    (model / "hugpy.json").write_text(json.dumps({
        "name": "Same-GGUF", "hub_id": "Other/Same-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation"}))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_SCAN_ROOTS", str(host))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_INVENTORY_DIR", str(tmp_path / "inventory"))
    monkeypatch.setattr(constants, "MODELS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {
        "Same-GGUF": SimpleNamespace(hub_id="Central/Same-GGUF")})

    rows = system_models.scan_system_models()
    assert list(rows) == ["Other~Same-GGUF"]
    pointer = next((tmp_path / "inventory").glob("*/hugpy.json"))
    assert json.loads(pointer.read_text())["model_key"] == "Other~Same-GGUF"


def test_ollama_manifest_is_discovered_and_served_through_adapter(tmp_path, monkeypatch):
    from hugpy_fleet.worker import agent, ollama_adapter, system_models
    from hugpy_engine.config.models import models_config as mc
    from hugpy_platform import constants
    from hugpy_storage import provision

    store = tmp_path / "ollama"
    manifest = store / "manifests/registry.ollama.ai/library/example/latest"
    manifest.parent.mkdir(parents=True)
    blobs = store / "blobs"
    blobs.mkdir()
    layer = b"weights"
    config = b"{}"
    def blob(data):
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        (blobs / digest.replace(":", "-")).write_bytes(data)
        return {"digest": digest, "size": len(data)}
    model_layer = {**blob(layer), "mediaType": "application/vnd.ollama.image.model"}
    manifest.write_text(json.dumps({"config": blob(config), "layers": [model_layer]}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setenv("HUGPY_WORKER_MODEL_SCAN_ROOTS", str(store))
    monkeypatch.setenv("HUGPY_WORKER_MODEL_INVENTORY_DIR", str(tmp_path / "inventory"))
    monkeypatch.setattr(constants, "MODELS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(system_models, "_ollama_tags", lambda: {
        "example:latest": {"name": "example:latest", "digest": digest,
                           "capabilities": ["completion"]}})
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", {})
    monkeypatch.setattr(agent, "_SYSTEM_MODELS", {"at": 0.0, "rows": {}, "running": False})
    monkeypatch.setattr(agent, "_SYSTEM_ADDED_KEYS", set())

    rows = system_models.scan_system_models()
    key = "ollama~example~latest"
    assert rows[key]["ollama_model"] == "example:latest"
    assert rows[key]["external_location"] == str(manifest)
    agent._refresh_system_models()
    assert ollama_adapter.model_name(key) == "example:latest"
    assert key in agent._system_models_snapshot()
    assert agent._models_local(type("State", (), {"assigned_models": []})()) == [key]
    monkeypatch.setattr(provision, "catalog_get", lambda mk: mc.MODEL_REGISTRY.get(mk))
    monkeypatch.setattr(provision, "catalog_resolve_dir",
                        lambda mk: str(manifest) if mk == key else None)
    assert provision.model_is_local(key)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"message":{"content":"hello"},"done_reason":"stop","eval_count":1}'
    calls = []
    def request(route, body, *, stream=False):
        calls.append((route, body, stream))
        return Response()
    monkeypatch.setattr(ollama_adapter, "_request", request)
    result = agent._run_once({"model_key": key, "prompt": "hi", "max_tokens": 12})
    assert result["text"] == "hello"
    assert calls[0][0] == "/api/chat"
    assert calls[0][1]["model"] == "example:latest"
    assert calls[0][1]["options"]["num_predict"] == 12

    # A stale tag digest or missing blob must remove the model from routing.
    (blobs / model_layer["digest"].replace(":", "-")).unlink()
    assert key not in system_models.scan_system_models()
    agent._refresh_system_models()
    assert key not in agent._system_models_snapshot()
    assert key not in mc.MODEL_REGISTRY


def test_ollama_stream_and_unload(monkeypatch):
    from hugpy_fleet.worker import ollama_adapter

    monkeypatch.setattr(ollama_adapter, "model_name", lambda _key: "example:latest")
    calls = []
    class Response:
        def __init__(self, lines): self.lines = lines
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter(self.lines)
        def read(self): return b'{}'
    def request(route, body, *, stream=False):
        calls.append((route, body, stream))
        return Response([
            b'{"message":{"content":"hi"}}\n',
            b'{"done":true,"done_reason":"length","prompt_eval_count":2,"eval_count":1}\n',
        ])
    monkeypatch.setattr(ollama_adapter, "_request", request)
    events = list(ollama_adapter.stream({"model_key": "key", "prompt": "hello"}))
    assert events == [
        {"type": "token", "text": "hi"},
        {"type": "done", "finish_reason": "max_tokens",
         "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
    ]
    monkeypatch.setattr(ollama_adapter, "running", lambda: {})
    assert ollama_adapter.unload("example:latest")
    assert calls[-1][1]["keep_alive"] == 0


def test_external_path_cannot_be_reaped_even_with_cache_reaping_enabled(tmp_path, monkeypatch):
    from hugpy_storage import provision
    from hugpy_platform import constants

    managed = tmp_path / "hugpy" / "models"
    external = tmp_path / "foreign" / "model"
    managed.mkdir(parents=True)
    external.mkdir(parents=True)
    (external / "weights.gguf").write_bytes(b"weights")
    monkeypatch.setattr(constants, "MODELS_HOME", str(managed))
    monkeypatch.setenv("HUGPY_MODEL_STORE_REAPABLE", "1")
    monkeypatch.delenv("HUGPY_SHARED_MODEL_STORE", raising=False)

    assert provision._model_store_reapable(str(managed / "cached"))
    assert not provision._model_store_reapable(str(external))
    assert not provision.wipe_model("external", path=str(external))
    assert (external / "weights.gguf").exists()


def test_worker_loads_external_pointer_until_managed_copy_exists(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hugpy_engine.config import main

    managed = tmp_path / "hugpy-models"
    external = tmp_path / "host-model"
    external.mkdir()
    cfg = SimpleNamespace(folder="gguf/local/model", extra={"external_location": str(external)})
    monkeypatch.setattr(main, "MODELS_HOME", str(managed))
    monkeypatch.setattr(main, "get_model_config", lambda _key: cfg)
    monkeypatch.setattr(main, "model_looks_downloaded",
                        lambda path, _cfg: Path(path).is_dir())
    assert main.get_model_path("model") == str(external)
    local = managed / cfg.folder
    local.mkdir(parents=True)
    assert main.get_model_path("model") == str(local)


def test_worker_registers_scanned_model_for_local_loading(tmp_path, monkeypatch):
    from hugpy_engine.config.models import models_config as mc
    from hugpy_fleet.worker import agent, system_models

    key = "Scanned-GGUF"
    model = tmp_path / key
    model.mkdir()
    with (model / "model.gguf").open("wb") as fh:
        fh.truncate(2 * 1024 * 1024)
    monkeypatch.setattr(system_models, "scan_system_models", lambda: {key: {
        "model_key": key, "name": key, "hub_id": "local/Scanned-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation", "filename": "model.gguf",
        "dir": str(model), "external_location": str(model),
    }})
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", {})
    monkeypatch.setattr(agent, "_SYSTEM_MODELS", {"at": 0.0, "rows": {}, "running": False})
    monkeypatch.setattr(agent, "_SYSTEM_ADDED_KEYS", set())
    agent._refresh_system_models()
    assert mc.MODEL_REGISTRY[key].extra["external_location"] == str(model)
    assert key in agent._system_models_snapshot()
    monkeypatch.setattr(system_models, "scan_system_models", lambda: {})
    agent._refresh_system_models()
    assert key not in mc.MODEL_REGISTRY
    assert key not in agent._system_models_snapshot()


def test_worker_catalog_keeps_remote_path_out_of_central_config(tmp_path, monkeypatch):
    from hugpy_engine.config.models import models_config as mc

    key = "System-Only-GGUF"
    path = tmp_path / "worker" / key
    path.mkdir(parents=True)
    catalog = tmp_path / "worker_model_catalog.json"
    monkeypatch.setattr(mc, "_WORKER_CATALOG_PATH", str(catalog))
    monkeypatch.setattr(mc, "_WORKER_CATALOG_SEEN", 0)
    monkeypatch.setattr(mc, "_WORKER_ONLY_KEYS", set())
    monkeypatch.setattr(mc, "_WORKER_TOUCHED_KEYS", set())
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", {})

    mc.record_worker_models("worker-a", {key: {
        "model_key": key, "name": key, "hub_id": "local/System-Only-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation", "filename": "model.gguf",
        "size_bytes": 2048, "external_location": str(path),
    }})
    row = mc.get_models_dict(dict_return=True)[key]
    assert row["worker_only"] is True
    assert row["worker_locations"] == {"worker-a": str(path)}
    assert row["size_bytes"] == 2048
    assert "external_location" not in mc.MODEL_REGISTRY[key].extra


def test_existing_catalog_key_without_central_files_is_worker_only(tmp_path, monkeypatch):
    from hugpy_engine.config.models import models_config as mc

    key = "Known-GGUF"
    base = mc.get_assessed_model_config({
        "model_key": key, "name": key, "hub_id": "local/Known-GGUF",
        "folder": "gguf/local/Known-GGUF", "framework": "gguf",
        "tasks": ["text-generation"], "primary_task": "text-generation"})
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {key: base})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", {key: base.to_dict()})
    monkeypatch.setattr(mc, "_WORKER_ONLY_KEYS", set())
    monkeypatch.setattr(mc, "_WORKER_TOUCHED_KEYS", set())
    monkeypatch.setattr(mc, "_WORKER_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setattr(mc, "_WORKER_CATALOG_SEEN", 0)
    monkeypatch.setattr(mc, "_central_copy_present", lambda _key: False)
    mc.record_worker_models("w1", {key: {
        "model_key": key, "name": key, "hub_id": "local/Known-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation", "external_location": "/host/known",
    }})
    assert mc.MODEL_REGISTRY_DICT[key]["worker_only"] is True
    assert mc.MODEL_REGISTRY[key].extra["worker_only"] is True


def test_central_catalog_keeps_same_name_different_hubs_separate(tmp_path, monkeypatch):
    from hugpy_engine.config.models import models_config as mc

    base = mc.get_assessed_model_config({
        "model_key": "Same-GGUF", "name": "Same-GGUF",
        "hub_id": "Central/Same-GGUF", "folder": "gguf/Central/Same-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation"})
    monkeypatch.setattr(mc, "MODEL_REGISTRY", {"Same-GGUF": base})
    monkeypatch.setattr(mc, "MODEL_REGISTRY_DICT", {"Same-GGUF": base.to_dict()})
    monkeypatch.setattr(mc, "_WORKER_ONLY_KEYS", set())
    monkeypatch.setattr(mc, "_WORKER_TOUCHED_KEYS", set())
    monkeypatch.setattr(mc, "_WORKER_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setattr(mc, "_WORKER_CATALOG_SEEN", 0)
    mc.record_worker_models("w1", {"Same-GGUF": {
        "name": "Same-GGUF", "hub_id": "Other/Same-GGUF",
        "framework": "gguf", "tasks": ["text-generation"],
        "primary_task": "text-generation", "external_location": "/host/model",
    }})
    assert mc.MODEL_REGISTRY_DICT["Same-GGUF"].get("worker_locations") is None
    assert mc.MODEL_REGISTRY_DICT["Other~Same-GGUF"]["worker_only"] is True


def test_worker_only_model_routes_only_to_host_with_files(tmp_path, monkeypatch):
    from hugpy_engine.config.models import models_config as mc
    from hugpy_fleet.central import workers as W

    key = "Host-Only-GGUF"
    store = W.WorkerStore(path=str(tmp_path / "workers.json"))
    monkeypatch.setattr(W, "_assign_memory_path", lambda: str(tmp_path / "assign.json"))
    monkeypatch.setattr(W, "required_pkg_version", lambda: None)
    monkeypatch.setattr(W, "worker_can_hold", lambda _w, _mk: None)
    monkeypatch.setattr(mc, "get_models_dict", lambda **_kw: {
        key: {"worker_only": True, "worker_locations": {"owner-id": "/models/host"}}})

    owner = store.register(name="owner", url="http://owner:9100", worker_id="owner-id")
    other = store.register(name="other", url="http://other:9100")
    for worker in (owner, other):
        store.set_admission(worker["id"], "approved")
        store.heartbeat(worker["id"], pkg_version="0.1.241",
                        models_local=[key] if worker is owner else [])
    store.assign_model(other["id"], key)  # stale assignment cannot trigger a pull
    assert [w["name"] for w in store.workers_for_model(key)] == ["owner"]


def test_grade_lanes_and_cold_reset_respect_worker_owned_model(monkeypatch):
    from unittest.mock import Mock
    from hugpy_curation.review import fleet_grading as grading

    model = {"model_key": "host-model", "framework": "gguf",
             "worker_only": True, "worker_locations": {"w1": "/host/model"},
             "size_bytes": 2048}
    workers = [{"id": "w1", "name": "owner"}, {"id": "w2", "name": "other"}]
    monkeypatch.setattr(grading, "eligible", lambda _w: True)
    monkeypatch.setattr(grading, "_serving", lambda _client, _mk: {})
    lanes = grading._lanes(Mock(), workers, [model])
    assert [lane["worker_id"] for lane in lanes] == ["w1"]
    client = Mock()
    assert "skipped" in grading._cold_reset(client, lanes[0])
    client.request.assert_not_called()
