"""GPU-worker installation must never silently register a CPU-only engine."""
from __future__ import annotations

from hugpy_fleet.worker import install, setup


def test_cuda_binding_rebuild_is_scoped_to_llama_cpp(monkeypatch):
    probes = iter([
        {"installed": True, "supports_gpu_offload": False,
         "version": "0.3.35", "error": None},
        {"installed": True, "supports_gpu_offload": True,
         "version": "0.3.35", "error": None},
    ])
    calls = []

    monkeypatch.setattr(setup, "nvidia_gpus", lambda: [{"name": "GPU"}])
    monkeypatch.setattr(setup, "has_nvcc", lambda: True)
    monkeypatch.setattr(setup, "_native_engine", lambda: {
        "found": True, "path": "/engine/llama-server",
        "source": "test", "spawn_ok": True,
    })
    monkeypatch.setattr(setup, "_llama_offload", lambda: next(probes))

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return 0, "built"

    monkeypatch.setattr(setup, "_run", fake_run)
    result = setup.provision_cuda_engine(dry_run=False)

    assert result.status == setup.OK
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert "--no-deps" in argv
    assert argv[argv.index("--no-binary") + 1] == "llama-cpp-python"
    assert ":all:" not in argv
    assert argv[-1] == "llama-cpp-python==0.3.35"
    assert kwargs["env"]["CMAKE_ARGS"] == "-DGGML_CUDA=on"


def test_required_cuda_failure_prevents_service_registration(monkeypatch, tmp_path):
    required_failure = setup.Check(
        "cuda-engine", setup.FAIL, "GPU engine is CPU-only", required=True)
    monkeypatch.setattr(
        install, "_turnkey_converge", lambda _opts, _setup: ([required_failure], {}))

    registered = []
    monkeypatch.setattr(
        install, "_install_systemd_user", lambda _opts: registered.append(True))

    rc = install.main([
        "--central", "https://central.invalid",
        "--name", "gpu-worker",
        "--storage", str(tmp_path),
        "--service", "systemd",
        "--no-preflight",
    ])

    assert rc != 0
    assert registered == []


def test_probe_json_accepts_diagnostics_after_result(monkeypatch):
    def fake_run(argv, **kwargs):
        if "llama_cpp" in argv[-1]:
            return 0, ('{"installed": true, "supports_gpu_offload": true, '
                       '"version": "0.3.35", "error": null}\n'
                       "ggml_cuda_init: found 4 CUDA devices\n")
        return 0, ('{"ok": true, "version": "2.14.0", "error": null}\n'
                   "FutureWarning: pynvml is deprecated\n")

    monkeypatch.setattr(setup, "_run", fake_run)
    assert setup._llama_offload()["supports_gpu_offload"] is True
    assert setup._import_probe("torch")["ok"] is True
