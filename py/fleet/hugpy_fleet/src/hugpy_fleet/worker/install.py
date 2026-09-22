"""Cross-platform worker bootstrap.

The original installer was a bash script (served at ``/llm/workers/install.sh``)
that only ran on Linux: it hunted for a Python with the package, then wired up a
systemd *user* unit. This module does the same job on Windows, macOS, and Linux
from one place, because it runs *inside* an interpreter that already has hugpy
importable::

    python -m hugpy.worker_agent.install --central https://your-hugpy/ --name box-1

Auto-start registration is best-effort and OS-appropriate:

    Linux (systemd present)  ~/.config/systemd/user/hugpy-worker.service + enable
    macOS                    ~/Library/LaunchAgents/ai.hugpy.worker.plist + load
    Windows                  schtasks /create /sc onlogon
    otherwise / --service none   just run the worker in the foreground

Every knob has an env fallback mirroring the agent's own (WORKER_CENTRAL_URL,
WORKER_NAME, WORKER_PORT, WORKER_MODELS, DEFAULT_ROOT).
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from typing import List, Optional

from hugpy_platform.platform_facade import IS_LINUX, IS_MACOS, IS_WINDOWS
from hugpy_platform.app_dirs import config_dir, data_dir
from hugpy_platform.central import central_base_url

_SERVICE_NAME = "hugpy-worker"
_LAUNCHD_LABEL = "ai.hugpy.worker"


def _worker_argv(opts) -> List[str]:
    argv = [sys.executable, "-m", "hugpy_fleet.worker", "--central", opts.central]
    if opts.name:
        argv += ["--name", opts.name]
    if opts.port:
        argv += ["--port", str(opts.port)]
    if opts.models:
        argv += ["--models", opts.models]
    # Role / RPC-backend wiring — without these a cross-machine shard backend
    # (--role rpc) can't be persisted by the installer (it would silently come
    # up as role=worker). Forward only non-default values to keep ExecStart lean.
    if getattr(opts, "role", None) and opts.role != "worker":
        argv += ["--role", opts.role]
    if getattr(opts, "rpc_host", None):
        argv += ["--rpc-host", opts.rpc_host]
    if getattr(opts, "rpc_port", None):
        argv += ["--rpc-port", str(opts.rpc_port)]
    if getattr(opts, "rpc_bin", None):
        argv += ["--rpc-bin", opts.rpc_bin]
    # Spill / GPU placement.
    if getattr(opts, "spill", None):
        argv += ["--spill", opts.spill]
    if getattr(opts, "n_gpu_layers", None) is not None:
        argv += ["--n-gpu-layers", str(opts.n_gpu_layers)]
    if getattr(opts, "gpu_mem", None) is not None:
        argv += ["--gpu-mem", str(opts.gpu_mem)]
    if getattr(opts, "cpu_mem", None) is not None:
        argv += ["--cpu-mem", str(opts.cpu_mem)]
    if getattr(opts, "tensor_split", None):
        argv += ["--tensor-split", opts.tensor_split]
    if getattr(opts, "main_gpu", None) is not None:
        argv += ["--main-gpu", str(opts.main_gpu)]
    return argv


def _systemd_available() -> bool:
    return IS_LINUX and os.path.isdir("/run/systemd/system")


# --------------------------------------------------------------------------- #
# per-OS service registration                                                 #
# --------------------------------------------------------------------------- #
def _install_systemd_user(opts) -> str:
    unit_dir = os.path.join(os.path.expanduser("~"), ".config", "systemd", "user")
    os.makedirs(unit_dir, exist_ok=True)
    # Establish ~/.hugpy/{state,config,logs,run} at install time so the very
    # first write lands in the right place — "dictated at install".
    try:
        from hugpy_platform import app_dirs as _hp
        _hp.ensure_hugpy_home()
    except Exception:  # noqa: BLE001 — never fail an install over the skeleton
        pass
    unit_path = os.path.join(unit_dir, _SERVICE_NAME + ".service")
    exec_start = " ".join(_worker_argv(opts))
    # Use systemd's %h home specifier for the engine-dir default so the unit is
    # portable across users (no baked-in /home/<user> path).
    env_lines = "\n".join(f'Environment="{k}={v}"'
                          for k, v in _env_for(opts, home="%h").items())
    unit = (
        "[Unit]\n"
        f"Description=hugpy worker ({opts.name})\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"{env_lines}\n"
        f"ExecStart={exec_start}\n"
        # on-failure (NOT always): a deliberate operator block/revoke makes the
        # agent exit 0 and it must STAY stopped — Restart=always would fight the
        # operator by respawning a blocked worker. A non-zero crash still restarts.
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    with open(unit_path, "w", encoding="utf-8") as fh:
        fh.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", _SERVICE_NAME + ".service"],
                   check=False)
    # Linger keeps the per-user systemd manager (and this unit) alive after logout
    # and across reboots without an active login session. Without it the worker
    # dies at logout — the classic "it stopped overnight" failure.
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    linger_cmd = ["loginctl", "enable-linger"] + ([user] if user else [])
    linger = subprocess.run(linger_cmd, check=False)
    if linger.returncode != 0:
        print("WARNING: `loginctl enable-linger` failed — this worker will STOP at "
              "logout and will NOT survive a reboot. Fix it with:\n"
              f"  sudo loginctl enable-linger {user or '$USER'}", file=sys.stderr)
    return f"systemd user unit installed: {unit_path} (systemctl --user status {_SERVICE_NAME})"


def _install_launchd(opts) -> str:
    agents = os.path.join(os.path.expanduser("~"), "Library", "LaunchAgents")
    os.makedirs(agents, exist_ok=True)
    plist_path = os.path.join(agents, _LAUNCHD_LABEL + ".plist")
    args_xml = "".join(f"    <string>{a}</string>\n" for a in _worker_argv(opts))
    env_xml = "".join(
        f"    <key>{k}</key><string>{v}</string>\n" for k, v in _env_for(opts).items())
    plist = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f"  <key>Label</key><string>{_LAUNCHD_LABEL}</string>\n"
        "  <key>ProgramArguments</key><array>\n" + args_xml + "  </array>\n"
        "  <key>EnvironmentVariables</key><dict>\n" + env_xml + "  </dict>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "  <key>KeepAlive</key><true/>\n"
        "</dict></plist>\n"
    )
    with open(plist_path, "w", encoding="utf-8") as fh:
        fh.write(plist)
    subprocess.run(["launchctl", "unload", plist_path], check=False,
                   stderr=subprocess.DEVNULL)
    subprocess.run(["launchctl", "load", plist_path], check=False)
    return f"launchd agent installed: {plist_path} (launchctl list | grep {_LAUNCHD_LABEL})"


def _install_schtasks(opts) -> str:
    # Wrap the worker command in a .cmd so env vars + quoting survive Task Scheduler.
    launcher = os.path.join(data_dir(), "hugpy-worker.cmd")
    env_lines = "\n".join(f"set {k}={v}" for k, v in _env_for(opts).items())
    cmd_body = "@echo off\n" + env_lines + "\n" + subprocess.list2cmdline(_worker_argv(opts)) + "\n"
    with open(launcher, "w", encoding="utf-8") as fh:
        fh.write(cmd_body)
    subprocess.run(
        ["schtasks", "/create", "/tn", _SERVICE_NAME, "/sc", "onlogon",
         "/tr", launcher, "/f", "/rl", "limited"],
        check=False,
    )
    return f"scheduled task '{_SERVICE_NAME}' created (Task Scheduler, runs at logon): {launcher}"


def _env_for(opts, home: Optional[str] = None) -> dict:
    """The unit/agent environment block. ``home`` is the home-dir token used for
    defaulted paths — an expanded path on launchd/schtasks, systemd's ``%h``
    specifier for the systemd user unit (portable across users)."""
    if home is None:
        home = os.path.expanduser("~")
    env = {"WORKER_CENTRAL_URL": opts.central}
    # Single base dir for hugpy runtime files (worker id/settings, model
    # metadata, …). Setting it in the unit means a fresh install lands them
    # under ~/.hugpy instead of strewing $HOME — ``_platform/paths.py`` honors
    # this exact env. ``home`` is systemd's ``%h`` for the user unit.
    env["HUGPY_HOME"] = os.path.join(home, ".hugpy")
    if getattr(opts, "name", None):
        env["WORKER_NAME"] = opts.name
    if getattr(opts, "port", None):
        env["WORKER_PORT"] = str(opts.port)
    # Native llama.cpp engine location: llama-server (vision GGUFs) resolves here
    # and the slot child derives its LD_LIBRARY_PATH from it, so ALWAYS set it —
    # a bare install otherwise silently uses the per-OS data dir the setup doc
    # never mentions. Flag-overridable via --engine-dir.
    env["HUGPY_ENGINE_DIR"] = getattr(opts, "engine_dir", None) or os.path.join(
        home, "hugpy-worker", "engine")
    # Register but don't auto-serve: the operator turns models on from the
    # console. ``off`` is the fleet default; --serve-mode overrides.
    if getattr(opts, "serve_mode", None):
        env["DEFAULT_SERVE_MODE"] = opts.serve_mode
    # Enrollment token (single-purpose per box, revocable) — lets the worker
    # register when central requires enrollment.
    if getattr(opts, "enroll_token", None):
        env["WORKER_ENROLL_TOKEN"] = opts.enroll_token
    # Zero-copy model storage root (optional): a box mounting central's volume
    # points here and provisioning becomes a no-op file check.
    if opts.storage:
        env["DEFAULT_ROOT"] = opts.storage
    return env


def _run_foreground(opts) -> int:
    for k, v in _env_for(opts).items():
        os.environ.setdefault(k, v)
    print("hugpy worker (foreground):", " ".join(_worker_argv(opts)))
    return subprocess.call(_worker_argv(opts))


# --------------------------------------------------------------------------- #
# k118 install-time preflight                                                 #
# --------------------------------------------------------------------------- #
def preflight(*, force: bool = False) -> int:
    """Assess THIS box against the fleet doctrine BEFORE it first heartbeats.

    The point is the ordering. Every failure this exists for was found the same
    way — the worker installed cleanly, registered, advertised a task, took a
    job, and only then discovered it had no ``ffmpeg`` / ``bitsandbytes`` /
    ``diffusers``. Running the diff here moves that discovery to the one moment
    when someone is already looking at the terminal.

    Returns 0 when clean or only warnings, 1 when a BLOCKER was found (unless
    ``force``). Never raises and never installs anything: it prints the repair
    plan and lets the operator decide. A box with no resolvable doctrine prints
    that plainly and returns 0 — an install must not be gated by a document the
    box has no way to fetch.
    """
    try:
        from hugpy_fleet.doctrine import doctrine as _doctrine
        from hugpy_fleet.doctrine.doctor import assess, render
        from hugpy_fleet.worker.environment_report import build_report
    except Exception as exc:  # noqa: BLE001 — an older tree simply has no gate
        print(f"  preflight: unavailable in this build ({exc}) — skipped")
        return 0
    current = _doctrine.latest()
    if current is None:
        print(f"  preflight: no doctrine found in {_doctrine.doctrine_dir()} "
              f"— environment NOT assessed (this is honest, not clean)")
        return 0
    try:
        assessment = assess(build_report(), current)
    except Exception as exc:  # noqa: BLE001 — a preflight never breaks an install
        print(f"  preflight: could not assess this box ({exc}) — skipped")
        return 0
    print()
    print(render(assessment))
    if assessment.blockers:
        print()
        print("  repair plan (NOT run — copy, review, then run):")
        for line in assessment.repair_plan():
            print(f"    {line}")
        if force:
            print()
            print("  --force: continuing with blockers present. This worker "
                  "will advertise tasks it cannot complete.")
            return 0
        print()
        print("  refusing to continue with blockers present; re-run with "
              "--force to install anyway.")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _resolve_service(choice: str) -> str:
    if choice != "auto":
        return choice
    if _systemd_available():
        return "systemd"
    if IS_MACOS:
        return "launchd"
    if IS_WINDOWS:
        return "schtasks"
    return "foreground"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="hugpy-worker-install",
        description="Install/run this machine as a hugpy GPU worker (cross-platform).")
    p.add_argument("--central", default=central_base_url(default=None),
                   help="central hugpy base URL (e.g. https://your-hugpy/); "
                        "env HUGPY_BASE_URL, legacy WORKER_CENTRAL_URL honoured")
    p.add_argument("--name", default=os.environ.get("WORKER_NAME") or socket.gethostname())
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "9100")))
    p.add_argument("--models", default=os.environ.get("WORKER_MODELS"))
    p.add_argument("--storage", default=os.environ.get("DEFAULT_ROOT"),
                   help="local model storage root (default: per-OS data dir)")
    p.add_argument("--storage-root", dest="storage_root", default=None,
                   help="alias for --storage: DEFAULT_ROOT / zero-copy model root")
    # Env-block completeness (mirrors the systemd unit the setup doc documents).
    p.add_argument("--engine-dir", default=os.environ.get("HUGPY_ENGINE_DIR"),
                   help="native llama.cpp engine dir where llama-server + slot-child "
                        "libs resolve (default: ~/hugpy-worker/engine, %%h in the unit)")
    p.add_argument("--serve-mode", default=os.environ.get("DEFAULT_SERVE_MODE", "off"),
                   choices=("off", "on"),
                   help="DEFAULT_SERVE_MODE: register but don't auto-serve (off, default)")
    p.add_argument("--enroll-token", default=os.environ.get("WORKER_ENROLL_TOKEN"),
                   help="single-purpose enrollment token minted on central "
                        "(POST /llm/enroll-tokens)")
    # Role / RPC backend — forwarded to the agent so a shard backend can be
    # persisted by the installer (mirrors the agent's own WORKER_* env fallbacks).
    p.add_argument("--role", default=os.environ.get("WORKER_ROLE", "worker"),
                   help="worker | rpc (rpc = cross-machine shard backend)")
    p.add_argument("--rpc-host", default=os.environ.get("WORKER_RPC_HOST"),
                   help="bind host for rpc-server (role=rpc; default 0.0.0.0)")
    p.add_argument("--rpc-port", type=int,
                   default=(int(os.environ["WORKER_RPC_PORT"])
                            if os.environ.get("WORKER_RPC_PORT") else None),
                   help="rpc-server port advertised to the lead (role=rpc)")
    p.add_argument("--rpc-bin", default=os.environ.get("WORKER_RPC_BIN"),
                   help="path to the llama.cpp rpc-server binary (role=rpc)")
    # Spill / GPU placement (forwarded as-is to the agent).
    p.add_argument("--spill", choices=("auto", "off"),
                   default=os.environ.get("WORKER_SPILL"),
                   help="GPU/CPU spill: auto-fit layers on the GPU, or off")
    p.add_argument("--n-gpu-layers", type=int,
                   default=(int(os.environ["WORKER_N_GPU_LAYERS"])
                            if os.environ.get("WORKER_N_GPU_LAYERS") else None))
    p.add_argument("--gpu-mem", type=float,
                   default=(float(os.environ["WORKER_GPU_MEM_GIB"])
                            if os.environ.get("WORKER_GPU_MEM_GIB") else None))
    p.add_argument("--cpu-mem", type=float,
                   default=(float(os.environ["WORKER_CPU_MEM_GIB"])
                            if os.environ.get("WORKER_CPU_MEM_GIB") else None))
    p.add_argument("--tensor-split", default=os.environ.get("WORKER_TENSOR_SPLIT"),
                   help="multi-GPU split, e.g. '0.7,0.3'")
    p.add_argument("--main-gpu", type=int,
                   default=(int(os.environ["WORKER_MAIN_GPU"])
                            if os.environ.get("WORKER_MAIN_GPU") else None))
    p.add_argument("--service", choices=("auto", "systemd", "launchd", "schtasks",
                                         "foreground", "none"),
                   default=os.environ.get("WORKER_SERVICE", "auto"),
                   help="auto-start mechanism (default: auto-detect for this OS)")
    # k118: the environment doctrine preflight. ON by default — an install that
    # silently skips the check is the state this slice exists to end.
    p.add_argument("--no-preflight", dest="preflight", action="store_false",
                   default=True,
                   help="skip the k118 environment-doctrine preflight")
    p.add_argument("--force", action="store_true",
                   help="install even when the preflight finds blockers")
    opts = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    if not opts.central:
        p.error("--central is required (or set WORKER_CENTRAL_URL)")
    # --storage-root is the canonical alias; when given it wins over --storage.
    if getattr(opts, "storage_root", None):
        opts.storage = opts.storage_root
    if not opts.storage:
        opts.storage = os.path.join(data_dir(), "worker_storage")

    print("hugpy worker installer")
    print(f"  python  : {sys.executable}")
    print(f"  central : {opts.central}")
    print(f"  name    : {opts.name}")
    print(f"  storage : {opts.storage}")
    print(f"  role    : {opts.role}"
          + (f" (rpc-server on {opts.rpc_host or '0.0.0.0'}:{opts.rpc_port or 50052})"
             if opts.role == "rpc" else ""))

    if getattr(opts, "preflight", True):
        rc = preflight(force=bool(getattr(opts, "force", False)))
        if rc != 0:
            return rc

    service = _resolve_service(opts.service)
    if service in ("foreground", "none"):
        if service == "none":
            print("worker command:", " ".join(_worker_argv(opts)))
            return 0
        return _run_foreground(opts)

    try:
        if service == "systemd":
            print(_install_systemd_user(opts))
        elif service == "launchd":
            print(_install_launchd(opts))
        elif service == "schtasks":
            print(_install_schtasks(opts))
    except Exception as exc:
        print(f"service registration failed ({exc}); run the worker manually:\n  "
              + " ".join(_worker_argv(opts)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
