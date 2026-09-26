"""ComfyUI pinned to a chosen GPU via its own --cuda-device flag.

Comfy is ONE managed instance per worker, so its card is chosen at provision
(HUGPY_COMFY_CUDA_DEVICE) and baked into the spawn launch string; comfy_process
carries it straight onto the argv it spawns. (Running one comfy per card is a
documented follow-up.)

Run:  python3 -m pytest tests/test_comfy_cuda_device.py -q
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

cp = importlib.import_module("hugpy_fleet.worker.comfy_process")
setup = importlib.import_module("hugpy_fleet.worker.setup")


def test_parse_launch_keeps_cuda_device_in_spawn_args():
    spec = cp.parse_launch(
        "spawn:/venv/bin/python /comfy/main.py --port 8188 --listen 127.0.0.1 "
        "--cuda-device 2")
    assert spec["kind"] == "spawn"
    assert "--cuda-device" in spec["args"] and "2" in spec["args"]


def test_manager_spawns_argv_with_cuda_device():
    spec = cp.parse_launch(
        "spawn:/venv/bin/python /comfy/main.py --port 8188 --cuda-device 3")
    seen = {}

    class _Child:
        pid = 4242
        def poll(self): return None

    def _popen(argv, **kw):
        seen["argv"] = argv
        return _Child()

    # not running until AFTER we launch, so start() actually spawns.
    mgr = cp.ComfyManager(spec, "http://127.0.0.1:8188",
                          popen=_popen,
                          readiness=lambda url, timeout=2.0: "argv" in seen,
                          clock=lambda: 0.0, sleep=lambda s: None)
    res = mgr.start(ready_timeout=1.0)
    assert res["ok"] is True, res
    assert "--cuda-device" in seen["argv"] and "3" in seen["argv"], seen["argv"]


def test_setup_builds_cuda_device_from_env():
    src = inspect.getsource(setup.provision_comfy)
    assert "HUGPY_COMFY_CUDA_DEVICE" in src
    assert "--cuda-device" in src


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())
