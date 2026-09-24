"""Placement preview distinguishes free VRAM from total VRAM, without writes."""
import ast
import importlib.util
import types
from pathlib import Path


def isolated_route():
    """Execute the real route with injected collaborators, without booting
    package-wide services from Flask's import graph."""
    from flask import jsonify, abort
    spec = importlib.util.find_spec("hugpy_server.app.routes.worker_routes")
    path = Path(spec.origin)
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model_placement_preview")
    function.decorator_list = []
    module = types.ModuleType("isolated_placement")
    module.jsonify, module.abort = jsonify, abort
    for name in ("_transfer_authorized", "_model_blocked", "_archive_refusal", "_model_gguf_bytes", "list_workers", "_worker_already_has", "_disk_preflight_reason", "_worker_fit"):
        setattr(module, name, None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def test_preview_exposes_two_gpu_budgets(monkeypatch):
    from flask import Flask
    wr = isolated_route()
    worker = {"id": "w", "name": "gpu", "status": "online", "vram_free": 2, "vram_total": 24, "free_ram": 64}
    monkeypatch.setattr(wr, "_transfer_authorized", lambda: True)
    monkeypatch.setattr(wr, "_model_blocked", lambda key: False)
    monkeypatch.setattr(wr, "_archive_refusal", lambda key: None)
    monkeypatch.setattr(wr, "_model_gguf_bytes", lambda key: 8)
    monkeypatch.setattr(wr, "list_workers", lambda: [worker])
    monkeypatch.setattr(wr, "_worker_already_has", lambda w, key: True)
    monkeypatch.setattr(wr, "_disk_preflight_reason", lambda w, key: None)
    monkeypatch.setattr(wr, "_worker_fit", lambda key, w: {
        "fit": True, "need": 10, "gpu_resident": w["vram_free"] >= 10,
        "vram_free": w["vram_free"], "ram_free": w["free_ram"], "reason": None})
    with Flask(__name__).test_request_context():
        result = wr.model_placement_preview("m").get_json()["workers"][0]
    assert result["fits_free_vram"] is False
    assert result["fits_total_vram"] is True
    assert result["required_vram_bytes"] == 10
    assert worker["vram_free"] == 2  # preview never changes worker state


def test_unknown_model_size_stays_unknown(monkeypatch):
    from flask import Flask
    wr = isolated_route()
    monkeypatch.setattr(wr, "_transfer_authorized", lambda: True)
    monkeypatch.setattr(wr, "_model_blocked", lambda key: False)
    monkeypatch.setattr(wr, "_archive_refusal", lambda key: None)
    monkeypatch.setattr(wr, "_model_gguf_bytes", lambda key: None)
    monkeypatch.setattr(wr, "list_workers", lambda: [{"id": "w", "vram_total": 24}])
    monkeypatch.setattr(wr, "_worker_already_has", lambda w, key: False)
    monkeypatch.setattr(wr, "_disk_preflight_reason", lambda w, key: None)
    monkeypatch.setattr(wr, "_worker_fit", lambda key, w: {"fit": None, "need": None, "gpu_resident": None})
    with Flask(__name__).test_request_context():
        result = wr.model_placement_preview("m").get_json()["workers"][0]
    assert result["fits_free_vram"] is None
    assert result["fits_total_vram"] is None
