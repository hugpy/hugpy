"""Portable always-on supervisor for per-model ``llama-server`` children.

The systemd driver gives each model a unit; that's Linux-only. This driver-side
helper does the same job with no init system: it launches ``llama-server`` as a
**detached background process** (via :func:`hugpy._platform.procutil.popen_detached`,
which picks the right OS primitive), records the pid + port in a JSON file under
the per-OS data dir, and can stop/status it later. Works identically on Windows,
macOS, and Linux.

Invoked as a subcommand so it fits the ServePlan (RunCmd) model the console
already drives::

    python -m hugpy.managers.serve.supervisor start  <model_key>
    python -m hugpy.managers.serve.supervisor stop   <model_key>
    python -m hugpy.managers.serve.supervisor status <model_key>
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from hugpy_platform.app_dirs import data_dir
from hugpy_platform.procutil import popen_detached, terminate_tree


def _state_dir() -> str:
    d = os.path.join(data_dir(), "supervised")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(model_key: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in model_key)


def _pidfile(model_key: str) -> str:
    return os.path.join(_state_dir(), _slug(model_key) + ".json")


def _read(model_key: str) -> Optional[dict]:
    try:
        with open(_pidfile(model_key), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)            # signal 0 = liveness probe (works on Windows too)
        return True
    except (OSError, ProcessLookupError):
        return False
    except PermissionError:
        return True                # exists but not ours


def _server_argv(spec) -> list[str]:
    """llama-server argv for a ServeSpec — mirrors the systemd ExecStart line."""
    argv = [
        spec.server_bin,
        "-m", spec.model_file,
        "--host", spec.host,
        "--port", str(spec.port),
        "-c", str(spec.ctx_size),
        "-t", str(spec.threads),
        "--n-gpu-layers", str(spec.n_gpu_layers),
    ]
    argv += list(spec.extra_args)
    return argv


def start(model_key: str) -> dict:
    from hugpy_engine.serve.serve import serve_spec_for
    from hugpy_engine.native.resolve import server_bin

    existing = status(model_key)
    if existing.get("running"):
        return existing

    spec = serve_spec_for(model_key)
    if not server_bin():
        raise RuntimeError(
            "no llama-server binary — run `hugpy install-engine` (or set "
            "LLAMA_SERVER_BIN) before using supervised serving.")
    if not spec.model_file or not os.path.isfile(spec.model_file):
        raise RuntimeError(f"{model_key}: model file not found ({spec.model_file!r})")

    argv = _server_argv(spec)
    log_path = os.path.join(_state_dir(), _slug(model_key) + ".log")
    log = open(log_path, "ab")
    proc = popen_detached(argv, stdout=log, stderr=log)

    record = {
        "model_key": model_key,
        "pid": proc.pid,
        "port": spec.port,
        "host": spec.host,
        "endpoint": f"http://{spec.host}:{spec.port}",
        "log": log_path,
        "started": time.time(),
    }
    with open(_pidfile(model_key), "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    return {**record, "running": True}


def stop(model_key: str) -> dict:
    rec = _read(model_key)
    if rec and rec.get("pid") and _alive(rec["pid"]):
        class _P:  # terminate_tree only needs a .pid / .terminate
            pid = rec["pid"]
            def terminate(self):
                try: os.kill(self.pid, 15)
                except OSError: pass
        terminate_tree(_P())
    try:
        os.remove(_pidfile(model_key))
    except OSError:
        pass
    return {"model_key": model_key, "running": False, "stopped": True}


def status(model_key: str) -> dict:
    rec = _read(model_key)
    if not rec:
        return {"model_key": model_key, "running": False}
    running = bool(rec.get("pid") and _alive(rec["pid"]))
    return {**rec, "running": running}


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("start", "stop", "status"):
        print("usage: python -m hugpy.managers.serve.supervisor {start|stop|status} <model_key>",
              file=sys.stderr)
        return 2
    action, model_key = argv[0], argv[1]
    result = {"start": start, "stop": stop, "status": status}[action](model_key)
    print(json.dumps(result))
    return 0 if result.get("running") or action == "stop" else 1


if __name__ == "__main__":
    raise SystemExit(main())
