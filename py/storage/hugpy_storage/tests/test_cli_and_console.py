"""CLI entry points and the Flask-free console helpers."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hugpy_storage"


def test_cli_help_and_subcommands(capsys):
    from hugpy_storage.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("daemon", "sync", "status"):
        assert cmd in out
    assert main([]) == 2                                   # no subcommand -> usage


def test_cli_status_runs_without_daemon_or_catalog(capsys):
    from hugpy_storage.cli import main
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "models home" in out and "catalog" in out


def test_console_scripts_declared():
    import tomllib
    data = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["hugpy-storage"] == "hugpy_storage.cli:main"
    assert scripts["hugpy-downloader"] == "hugpy_storage.downloader.daemon:main"
    for dep in ("huggingface_hub", "requests", "pydantic"):
        assert any(d.startswith(dep) for d in data["project"]["dependencies"])


def test_daemon_module_entry_is_importable_without_running():
    from hugpy_storage.downloader import daemon
    from hugpy_storage.downloader import __main__ as dunder
    assert dunder.main is daemon.main


def test_deploy_unit_shipped_and_points_at_new_entry():
    unit = SRC / "deploy" / "hugpy-downloader-dev.service"
    text = unit.read_text()
    assert "hugpy-storage daemon" in text
    assert "abstract_hugpy_dev" not in text.split("[Service]")[1].split("Restart=")[0]


def test_console_helpers_are_flask_free():
    for rel in ("console/downloader.py", "console/downloads.py",
                "console/cancelable_downloads.py", "console/model_physical.py",
                "schemas/download_schemas.py", "hf_token.py"):
        tree = ast.parse((SRC / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else [node.module or ""]
                for n in names:
                    assert not n.split(".")[0].startswith(("flask", "abstract_flask")), (rel, n)
            if isinstance(node, ast.Name):
                assert node.id not in ("request", "jsonify"), (rel, node.id)


def test_model_status_from_disk_truth(tmp_path, monkeypatch):
    from hugpy_storage.console import downloader as cd
    model = {"hub_id": "o/r", "framework": "gguf", "primary_task": "text-generation",
             "filename": "r-Q4_K_M.gguf"}
    flat = tmp_path / "models" / "gguf" / "o" / "r"
    monkeypatch.setattr(cd, "route_destination", lambda m: str(flat))
    assert cd.model_status(model)["status"] == "not_installed"
    flat.mkdir(parents=True)
    (flat / "r-Q8_0.gguf").write_bytes(b"x" * (1024 * 1024 + 1))
    st = cd.model_status(model)
    assert st["status"] == "installed"
    assert "filename_warning" in st                     # pinned quant absent
    cd.write_install_marker(str(flat), "r", model)
    assert os.path.isfile(str(flat / "hugpy.json"))


def test_download_schemas_are_self_contained():
    from hugpy_storage.schemas.download_schemas import DownloadRequest, Runtime, DownloadStatus
    req = DownloadRequest(hub_id="o/r", framework=Runtime.gguf)
    assert req.framework == "gguf" and DownloadStatus.queued == "queued"


def test_hf_token_listener_seam(tmp_path, monkeypatch):
    from hugpy_storage import hf_token
    monkeypatch.setattr(hf_token, "HF_TOKEN_PATH", str(tmp_path / "hf_token"))
    monkeypatch.setattr(hf_token, "read_stored_hf_token",
                        lambda: (tmp_path / "hf_token").read_text()
                        if (tmp_path / "hf_token").exists() else False)
    seen = []
    hf_token.add_token_listener(seen.append)
    try:
        hf_token.store_hf_token("hf_abc123")
        assert seen[-1] == "hf_abc123"
        assert hf_token.get_hf_token() == "hf_abc123"
        assert hf_token.token_source() == "stored"
        assert hf_token.delete_hf_token() is True
        assert seen[-1] is None or seen[-1] == hf_token._env_token()
    finally:
        hf_token.remove_token_listener(seen.append)


def test_package_init_is_light():
    code = ("import sys; sys.modules['abstract_hugpy_dev']=None; import hugpy_storage; "
            "heavy=[m for m in ('huggingface_hub','requests','pydantic','sqlite3') if m in sys.modules]; "
            "print(sorted(hugpy_storage.__all__)[:3], heavy)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "[]" in proc.stdout.strip().rsplit(" ", 1)[-1] or proc.stdout.strip().endswith("[]"), proc.stdout
