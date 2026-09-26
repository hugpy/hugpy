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


def _fleet_default_serve_mode() -> str:
    """The fleet's own default serve mode — the SAME value the engine falls back
    to when DEFAULT_SERVE_MODE is unset (hugpy_engine.serve.serve, currently
    'swap'). Read it from there so the installer never drifts from the fleet;
    'swap' is the honest literal fallback if the engine isn't importable."""
    try:
        from hugpy_engine.serve.serve import _default_serve_mode
        return _default_serve_mode()
    except Exception:  # noqa: BLE001
        return "swap"


def _worker_argv(opts) -> List[str]:
    argv = [sys.executable, "-m", "hugpy_fleet.worker", "--central", opts.central]
    if opts.name:
        argv += ["--name", opts.name]
    if getattr(opts, "advertise", None):
        argv += ["--advertise", opts.advertise]
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
def _render_unit(opts) -> str:
    """The canonical systemd user unit text for ``opts`` (pure — no I/O). Factored
    out so drift detection can diff the existing unit against what we WOULD write
    without touching the filesystem."""
    exec_start = " ".join(_worker_argv(opts))
    # Use systemd's %h home specifier for the engine-dir default so the unit is
    # portable across users (no baked-in /home/<user> path).
    env_lines = "\n".join(f'Environment="{k}={v}"'
                          for k, v in _env_for(opts, home="%h").items())
    return (
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


def _install_systemd_user(opts) -> str:
    unit_dir = os.path.join(os.path.expanduser("~"), ".config", "systemd", "user")
    unit_path = os.path.join(unit_dir, _SERVICE_NAME + ".service")
    unit = _render_unit(opts)
    dry_run = bool(getattr(opts, "dry_run", False))

    # Report CONVERGENCE: what an existing (hand-built / drifted) unit had that the
    # canonical unit corrects — so re-running the installer names the drift it
    # fixed instead of silently overwriting (item 7: DEFAULT_SERVE_MODE=off + a
    # retired central URL were both hiding in the a-brain unit).
    existing = ""
    if os.path.isfile(unit_path):
        try:
            with open(unit_path, "r", encoding="utf-8") as fh:
                existing = fh.read()
        except Exception:  # noqa: BLE001
            existing = ""
    if existing:
        try:
            from hugpy_fleet.worker.setup import unit_drift
            drift = unit_drift(existing, unit)
        except Exception:  # noqa: BLE001
            drift = []
        if drift:
            print(f"  unit drift corrected in {unit_path}:")
            for d in drift:
                print(f"    {d}")
        else:
            print(f"  existing unit at {unit_path} already canonical")

    if dry_run:
        print(f"  DRY-RUN: would write {unit_path} and enable "
              f"{_SERVICE_NAME}.service (--now). Rendered unit:")
        for line in unit.splitlines():
            print(f"    | {line}")
        return f"DRY-RUN: systemd user unit NOT written ({unit_path})"

    os.makedirs(unit_dir, exist_ok=True)
    # Establish ~/.hugpy/{state,config,logs,run} at install time so the very
    # first write lands in the right place — "dictated at install".
    try:
        from hugpy_platform import app_dirs as _hp
        _hp.ensure_hugpy_home()
    except Exception:  # noqa: BLE001 — never fail an install over the skeleton
        pass
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
    # Advertise URL (WORKER_URL): the callback address central dials. A
    # WireGuard-joined worker sets this to its tunnel address (http://10.66.0.x:
    # 9100) so central reaches it over the tunnel; baking it into the unit keeps
    # that across restarts. Absent -> the agent derives the local IP toward
    # central at runtime (unchanged behaviour).
    if getattr(opts, "advertise", None):
        env["WORKER_URL"] = opts.advertise
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
    # Extra unit env produced by the setup convergence (e.g. the ComfyUI launcher
    # + shared-checkpoint wiring), baked in so the worker comes up owning comfy
    # with no hand-written drop-in.
    for k, v in (getattr(opts, "extra_env", None) or {}).items():
        if v is not None and str(v) != "":
            env[k] = str(v)
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
    p.add_argument("--advertise", default=os.environ.get("WORKER_URL"),
                   help="URL central should call back on (baked as WORKER_URL in "
                        "the unit); default env WORKER_URL. A WireGuard worker "
                        "sets this to http://10.66.0.x:9100.")
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
    p.add_argument("--serve-mode",
                   default=os.environ.get("DEFAULT_SERVE_MODE") or _fleet_default_serve_mode(),
                   choices=("off", "systemd", "supervised", "swap"),
                   help="DEFAULT_SERVE_MODE (fleet default: swap = on-demand, "
                        "slot-first). 'off' registers but never auto-serves. The "
                        "engine's ServeMode enum is off|systemd|supervised|swap; "
                        "the legacy 'on' is not a valid mode.")
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
    # ── turnkey convergence + self-check (the by-hand steps, now in code) ──────
    p.add_argument("--dry-run", action="store_true",
                   help="inspect + report only: converge NOTHING, write no unit, "
                        "start no service — every step prints the exact command it "
                        "WOULD run")
    p.add_argument("--self-check-only", action="store_true",
                   help="run the self-check against this box and print the report "
                        "without installing/converging anything (implies no unit write)")
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                   help="skip the reachability self-check (worker->central, "
                        "central->worker)")
    p.add_argument("--no-provision-engine", dest="provision_engine",
                   action="store_false", default=True,
                   help="do NOT build/install the CUDA llama engine when a GPU is "
                        "present (only detect + report)")
    p.add_argument("--no-retire-legacy", dest="retire_legacy",
                   action="store_false", default=True,
                   help="do NOT disable legacy hugpy/abstract_hugpy worker units on "
                        "this box (only detect + report)")
    p.add_argument("--no-open-firewall", dest="open_firewall",
                   action="store_false", default=True,
                   help="do NOT add a ufw rule opening the worker port to central "
                        "(only report)")
    p.add_argument("--install-comfy", dest="install_comfy", action="store_true",
                   default=(os.environ.get("HUGPY_INSTALL_COMFY", "").lower()
                            in ("1", "true", "yes")),
                   help="install/converge ComfyUI (the worker owns its lifecycle) "
                        "and wire it to the shared model store")
    p.add_argument("--no-comfy", dest="install_comfy", action="store_false",
                   help="skip ComfyUI install/convergence")
    p.add_argument("--comfy-version", default=os.environ.get("HUGPY_COMFY_VERSION"),
                   help="ComfyUI git tag/ref to pin (default: current default branch, "
                        "recorded by commit)")
    p.add_argument("--comfy-start-check", action="store_true",
                   help="after install, actually START comfy and verify "
                        "/system_stats + a checkpoint (slower)")
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

    from hugpy_fleet.worker import setup as _setup
    service = _resolve_service(opts.service)
    dry = bool(opts.dry_run or opts.self_check_only)

    # ── pre-service convergence + checks (the by-hand steps, now in code) ──────
    checks, comfy_env = _turnkey_converge(opts, _setup)
    opts.extra_env = comfy_env  # baked into the unit by _env_for

    # --self-check-only: report the box's state and stop (write no unit).
    if opts.self_check_only:
        print(_setup.render(checks))
        return _setup.overall_rc(checks)

    # ── write + enable the service (dry-run writes nothing) ───────────────────
    if service in ("foreground", "none"):
        print(_setup.render(checks))
        rc = _setup.overall_rc(checks)
        if rc != 0:
            print("self-check found required failures; NOT starting the worker.",
                  file=sys.stderr)
            return rc
        if service == "none":
            print("worker command:", " ".join(_worker_argv(opts)))
            return 0
        return _run_foreground(opts)

    try:
        if service == "systemd":
            print(_install_systemd_user(opts))
        elif service == "launchd":
            if dry:
                print(f"  DRY-RUN: would install launchd agent for {opts.name}")
            else:
                print(_install_launchd(opts))
        elif service == "schtasks":
            if dry:
                print(f"  DRY-RUN: would create scheduled task for {opts.name}")
            else:
                print(_install_schtasks(opts))
    except Exception as exc:
        print(f"service registration failed ({exc}); run the worker manually:\n  "
              + " ".join(_worker_argv(opts)), file=sys.stderr)
        return 1

    # ── post-service checks (only when the worker was actually started) ────────
    if not dry and service == "systemd" and opts.verify:
        home = os.path.expanduser("~")
        hugpy_home = os.environ.get("HUGPY_HOME") or os.path.join(home, ".hugpy")
        worker_id = _setup.wait_for_worker_id(hugpy_home, timeout=30.0)
        if worker_id:
            checks.append(_setup.Check("registered", _setup.OK,
                                       f"worker registered with central; id={worker_id}",
                                       required=False))
        else:
            checks.append(_setup.Check("registered", _setup.WARN,
                                       "worker did not persist a registered id within "
                                       "30s — check `journalctl --user -u "
                                       f"{_SERVICE_NAME} -f`", required=False))
        checks.append(_setup.check_central_can_reach_worker(
            opts.central, worker_id, getattr(opts, "enroll_token", None)))

    if opts.install_comfy:
        checks.append(_setup.comfy_self_check(
            comfy_env, do_start=bool(opts.comfy_start_check), dry_run=dry))

    print(_setup.render(checks))
    return _setup.overall_rc(checks)


def _turnkey_converge(opts, _setup):
    """Run every convergence + inspection step and return (checks, comfy_env).

    Each step is gated by its own flag AND the global dry-run/self-check-only:
    ``dry`` means inspect + record the exact command, change nothing on the box.
    """
    dry = bool(opts.dry_run or opts.self_check_only)
    checks = []
    # item 1 — worker -> central reachability.
    if opts.verify:
        checks.append(_setup.check_central_reachable(opts.central))
    # item 1 / 10 — advertise URL (explicit wins, else derived; report clearly).
    checks.append(_setup.check_advertise(opts.central, opts.port,
                                         getattr(opts, "advertise", None)))
    # item 2 — slot port clearance (defense-in-depth atop slot_agent's guard).
    try:
        from hugpy_engine.serve.slots import _slot_count, _slot_port_base
        sc, spb = _slot_count(), _slot_port_base()
    except Exception:  # noqa: BLE001
        sc = int(os.environ.get("SLOT_COUNT", "2") or 2)
        spb = int(os.environ.get("SLOT_PORT_BASE", "8101") or 8101)
    checks.append(_setup.check_slot_ports(opts.port, sc, spb))
    # item 6 — legacy worker units on this box.
    checks.append(_setup.retire_legacy_workers(dry_run=dry or not opts.retire_legacy))
    # item 4 — CUDA engine (native llama-server + llama-cpp-python offload).
    checks.append(_setup.provision_cuda_engine(
        dry_run=dry or not opts.provision_engine))
    # item 5 — torch + torchvision matched set, importable.
    checks.append(_setup.check_torch_companions())
    # item 11 — ComfyUI (worker owns its lifecycle).
    comfy_env = {}
    if opts.install_comfy:
        c, comfy_env = _setup.provision_comfy(
            storage_root=opts.storage, worker_port=opts.port,
            slot_port_base=spb, slot_count=sc,
            version=opts.comfy_version, dry_run=dry)
        checks.append(c)
    else:
        checks.append(_setup.Check(
            "comfyui", _setup.SKIP,
            "ComfyUI install not requested (--install-comfy / HUGPY_INSTALL_COMFY=1 "
            "to enable)", required=False))
    # item 8 — firewall: open the worker port to central.
    checks.append(_setup.open_firewall_to_central(
        opts.central, opts.port, dry_run=dry or not opts.open_firewall))
    # item 9 — stale central record (central-side; recorded, not changed here).
    checks.append(_setup.note_stale_record())
    return checks, comfy_env


if __name__ == "__main__":
    raise SystemExit(main())
