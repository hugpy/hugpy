"""hugpy command line — the friendly front door of the Hugpy ecosystem.

    hugpy serve      [--host 0.0.0.0] [--port 7002] [--auth open|external] ...
    hugpy worker     --central https://your-hugpy/ [worker agent args...]
    hugpy bot        [--central http://127.0.0.1:7002] [--env PATH] [--guild ID]
    hugpy download   MODEL_KEY [MODEL_KEY ...]
    hugpy chat       "prompt"            (stdlib HTTP client of a central)
    hugpy keeper / sentinel / chaos / provision / drift / oracle / curation / video / storage
    hugpy install-engine | install-deps | reclassify-images | version | build

Every feature lives in the package that owns it; this module only maps a
subcommand to that package's entry point and imports it on demand. A bare
``pip install hugpy`` therefore stays tiny, and a feature whose package is not
installed answers with ``pip install "hugpy[<extra>]"`` instead of a traceback.

The only ecosystem package imported eagerly is ``hugpy_platform``; ``chat``
and ``install-deps`` are stdlib-only so they work on the thinnest install.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional


class DispatchError(RuntimeError):
    """A subcommand's owner package (or its entry point) is unavailable.

    The message is user-facing: ``main`` prints it and exits 1, never with a
    traceback.
    """


@dataclass(frozen=True)
class Target:
    """Where a subcommand's implementation lives.

    ``module``  dotted module inside the owning package
    ``attrs``   candidate callables, first present wins
    ``extra``   the ``hugpy[<extra>]`` profile that installs the owner
    ``script``  console script of the owner to fall back on when ``module`` has
                none of ``attrs`` (older/newer owner versions)
    """

    module: str
    attrs: tuple[str, ...]
    extra: str
    script: Optional[str] = None

    @property
    def package(self) -> str:
        return self.module.split(".")[0]

    @property
    def distribution(self) -> str:
        return self.package.replace("_", "-")


# Subcommand -> owning entry point. ``chat``, ``install-deps`` and ``version``
# are stdlib-only and have no target. Passthrough commands (``PASSTHROUGH``)
# hand every argument after the subcommand to the target's own parser.
DISPATCH: dict[str, Target] = {
    "serve": Target("hugpy_server.wsgi_app", ("main", "get_hugpy_flask"), "server"),
    "worker": Target("hugpy_fleet.worker.agent", ("main",), "worker", "hugpy-worker"),
    "gguf-worker": Target("hugpy_fleet.gguf_worker.agent", ("main",), "worker", "hugpy-gguf-worker"),
    "phone-brick": Target("hugpy_fleet.phone_brick.__main__", ("main",), "worker", "hugpy-phone-brick"),
    "bot": Target("hugpy_discord.bot", ("main",), "bot", "hugpy-bot"),
    "download": Target("hugpy_storage.download_models", ("ensure_model",), "storage"),
    "storage": Target("hugpy_storage.cli", ("main",), "storage", "hugpy-storage"),
    "install-engine": Target("hugpy_engine.native.fetch", ("install",), "native-engine"),
    "reclassify-images": Target("hugpy_engine.apis.reclassify", ("reclassify_images",), "engine"),
    "keeper": Target("hugpy_ops.keeper", ("main",), "ops"),
    "sentinel": Target("hugpy_ops.sentinel.__main__", ("main",), "ops"),
    "chaos": Target("hugpy_ops.chaos.runner", ("main",), "ops"),
    "provision": Target("hugpy_ops.provisioner", ("main",), "ops"),
    "drift": Target("hugpy_ops.drift", ("main",), "ops", "hugpy-drift-check"),
    "oracle": Target("hugpy_oracle.cli", ("main",), "oracle", "hugpy-oracle"),
    "curation": Target("hugpy_curation.cli", ("main",), "curation", "hugpy-curation"),
    "video": Target("hugpy_video.cli", ("main",), "video", "hugpy-video"),
}

PASSTHROUGH: dict[str, str] = {
    "worker": "join a hugpy central as a worker (full-stack GPU/CPU worker)",
    "gguf-worker": "join a central as a GGUF-only worker (llama.cpp, no torch)",
    "phone-brick": "run the phone-brick arm (ONNX detection / rpc shard donor)",
    "storage": "model store: download daemon, sync from a central, status",
    "keeper": "terminal keeper REPL - a model keeps this machine (or an LXD instance)",
    "sentinel": "sentinel: bound-exceeded detection -> case -> diagnosis agent",
    "chaos": "chaos runner: adversarial allocation/eviction sweeps against a central",
    "provision": "provisioner: fetch declared-but-missing weights across registries",
    "drift": "drift check: prove checkout, installed packages, fleet and PyPI are one build",
    "oracle": "oracle: steward self-check, hook installation",
    "curation": "curation: discovery dossiers and the review pipeline",
    "video": "video intelligence: media job bus, self tests, state roots",
}


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _install_hint(target: Target) -> str:
    return f'install it with:  pip install "hugpy[{target.extra}]"'


def _import(module: str, target: Target):
    """Import ``module`` (a module of ``target``'s package) or raise DispatchError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if not missing or missing.split(".")[0] == target.package:
            raise DispatchError(
                f"hugpy: the '{target.distribution}' package is not installed "
                f"(needed for this command)\n  {_install_hint(target)}"
            ) from None
        raise DispatchError(
            f"hugpy: '{target.distribution}' is installed but its dependency "
            f"'{missing}' is missing ({exc})\n  {_install_hint(target)}"
        ) from None
    except Exception as exc:  # broken owner module: still no traceback
        raise DispatchError(
            f"hugpy: importing {module} failed: {type(exc).__name__}: {exc}"
        ) from None


def _script_runner(script: str) -> Callable[[Optional[list[str]]], int]:
    def run(argv: Optional[list[str]] = None) -> int:
        return subprocess.call([script, *(argv or [])])

    run.__name__ = f"console_script:{script}"
    return run


def resolve_target(command: str) -> Callable:
    """Return the callable that implements ``command``.

    Imports the owning module lazily; if it has none of the expected
    attributes, falls back to the owner's console script when one is on
    ``PATH``. Raises :class:`DispatchError` with an install hint otherwise.
    """
    target = DISPATCH[command]
    try:
        mod = _import(target.module, target)
    except DispatchError:
        script = target.script and shutil.which(target.script)
        if script:
            return _script_runner(script)
        raise
    for attr in target.attrs:
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    script = target.script and shutil.which(target.script)
    if script:
        return _script_runner(script)
    raise DispatchError(
        f"hugpy: {target.module} exposes none of {', '.join(target.attrs)}; "
        f"the installed '{target.distribution}' does not match this hugpy "
        f"({_install_hint(target)})"
    )


def _passthrough(command: str, argv: list[str]) -> int:
    """Hand ``argv`` to the owner's own CLI. ``--help`` never fails: when the
    owner is missing, the meta-level description and install hint print
    instead of an error."""
    target = DISPATCH[command]
    try:
        fn = resolve_target(command)
    except DispatchError as exc:
        if argv and argv[0] in ("-h", "--help") and len(argv) == 1:
            print(f"usage: hugpy {command} [args...]\n\n{PASSTHROUGH[command]}\n\n"
                  f"All arguments go to {target.module} (distribution "
                  f"'{target.distribution}'), which is not installed.\n"
                  f"  {_install_hint(target)}")
            return 0
        raise exc
    return int(fn(argv) or 0)


# --------------------------------------------------------------------------- #
# subcommands with their own flags
# --------------------------------------------------------------------------- #
def _serve(args: argparse.Namespace, raw: list[str]) -> int:
    # Distribution default: single-operator instance, no login wall. The
    # /v1 API-key system still gates programmatic access. Deployments that
    # front a real auth service set --auth external (or HUGPY_AUTH_MODE).
    if args.auth:
        os.environ["HUGPY_AUTH_MODE"] = args.auth
    else:
        os.environ.setdefault("HUGPY_AUTH_MODE", "open")

    fn = resolve_target("serve")
    if getattr(fn, "__name__", "") != "get_hugpy_flask":
        # The server ships its own runner (``hugpy_server.wsgi_app:main``).
        return int(fn(raw) or 0)

    origins = [o.strip() for o in (args.origins or "").split(",") if o.strip()] or None

    # Pass the app factory as a zero-arg THUNK — never call it here, in the
    # process that becomes the gunicorn arbiter (master). The factory opens the
    # media_jobs.db / comms / reservation sqlite stores (each with its WAL/SHM
    # sidecars) and starts background threads (the media-bus runner pool, the
    # admission runner, the benchmark-resume hook). All of that is fork-hostile:
    # POSIX advisory locks are NOT inherited across fork, sqlite handles are not
    # fork-safe, and threads do not survive fork. gunicorn calls this thunk from
    # load() — i.e. INSIDE the forked worker — so every connection and thread is
    # born in the process that actually serves. (Incident 2026-09-24: a
    # master-built app left the single worker holding media_jobs.db-wal/-shm fds
    # marked "(deleted)", yielding "database is locked" / "disk I/O error".)
    def _build_app():
        return fn(name="hugpy", allowed_origins=origins, debug=args.debug)

    return _run_wsgi(_build_app, args.host, args.port, args.threads, args.debug)


def _run_wsgi(build_app, host: str, port: int, threads: int, debug: bool) -> int:
    """Serve a WSGI app: gunicorn on POSIX, waitress elsewhere, Flask dev
    server as the last resort. ``build_app`` is a zero-arg factory thunk; for
    gunicorn it is called from load(), INSIDE the forked worker, so the app's
    sqlite connections and background threads are created post-fork (incident
    2026-09-24). waitress and the Flask dev server do not fork, so they build the
    app in-process. Server-agnostic glue, no routes live here."""
    bind = f"{host}:{port}"
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError:
        try:
            from waitress import serve as _waitress_serve
        except ImportError:
            print(f"hugpy: gunicorn/waitress not installed; using the Flask dev server on {bind}",
                  file=sys.stderr)
            build_app().run(host=host, port=port, debug=debug)
            return 0
        print(f"hugpy serving on http://{bind}  (console at /, API at /api/v1)  [waitress]")
        print(f"  first run? finish setup at  http://{bind}/welcome")
        _waitress_serve(build_app(), host=host, port=port, threads=threads)
        return 0

    class _App(BaseApplication):
        def load_config(self):
            self.cfg.set("bind", bind)
            self.cfg.set("workers", 1)          # singleton registries/job store
            self.cfg.set("threads", threads)
            self.cfg.set("timeout", 300)

        def load(self):
            # gunicorn calls this in the WORKER (after fork). Building the app
            # here — not in the arbiter — is the fix: the media_jobs.db handles
            # and the runner/admission threads belong to the serving process.
            return build_app()

    print(f"hugpy serving on http://{bind}  (console at /, API at /api/v1)")
    print(f"  first run? finish setup at  http://{bind}/welcome")
    _App().run()
    return 0


def _bot(args: argparse.Namespace) -> int:
    """Run the discord bot arm. It drives a hugpy central over HTTP, so it can
    point at this machine or a remote central. Flags map onto ``hugpy-bot``."""
    fn = resolve_target("bot")
    argv: list[str] = []
    if args.central:
        argv += ["--base-url", args.central]
    if args.env:
        argv += ["--env", args.env]
    if args.guild:
        argv += ["--guild", str(args.guild)]
    if args.log_level:
        argv += ["--log-level", args.log_level]
    return int(fn(argv) or 0)


def _download(args: argparse.Namespace) -> int:
    """Pull one or more catalog models to the local model store."""
    ensure_model = resolve_target("download")
    # The store resolves catalog rows through the engine's registry when it
    # is installed; without it only models the store already knows resolve.
    try:
        bridge = importlib.import_module("hugpy_engine.catalog_bridge")
        bridge.install()
    except Exception:  # noqa: BLE001 — optional richer catalog
        pass
    rc = 0
    for key in args.model_key:
        try:
            path = ensure_model(key, args.root) if args.root else ensure_model(key)
        except KeyError as exc:
            print(f"hugpy download: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            rc = 1
            continue
        except Exception as exc:  # noqa: BLE001 — surface, don't trace
            print(f"hugpy download: {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
            continue
        print(f"{key}  ->  {path}")
    return rc


def _chat(args: argparse.Namespace) -> int:
    """The terminal is just another transport on the same substrate: streams
    /chat/stream with a request_id (so the run shows in the unified jobs view
    and the console can cancel it), authenticates with a real credential
    (principal token or API key) and Ctrl-C cancels SERVER-SIDE via
    /llm/chat/cancel before exiting. Stdlib-only, like hpy."""
    import json as _json
    import signal
    import urllib.request
    import uuid as _uuid

    from hugpy_platform.central import central_base_url

    base = (args.central or central_base_url()).rstrip("/")
    token = (args.token or os.environ.get("HUGPY_TOKEN")
             or os.environ.get("HUGPY_API_KEY") or "").strip()

    def _post(path: str, payload: dict, stream: bool = False):
        req = urllib.request.Request(
            base + path, data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {token}"} if token else {})},
            method="POST")
        return urllib.request.urlopen(req, timeout=None if stream else 15)

    prompt = " ".join(args.prompt) if args.prompt else None
    if not prompt:
        try:
            prompt = input("hugpy> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
    if not prompt:
        return 0

    rid = _uuid.uuid4().hex
    body = {"prompt": prompt, "request_id": rid, "transport": "cli"}
    if args.model:
        body["model_key"] = args.model

    def _cancel(_sig=None, _frm=None):
        print("\n[cancelling server-side…]", file=sys.stderr)
        try:
            _post(f"/llm/chat/cancel/{rid}", {})
        except Exception:
            pass
        raise SystemExit(130)
    signal.signal(signal.SIGINT, _cancel)

    try:
        resp = _post("/chat/stream", body, stream=True)
    except Exception as exc:
        print(f"hugpy chat: cannot reach central at {base}: {exc}", file=sys.stderr)
        return 1
    finish = "stop"
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            event = _json.loads(line[5:].strip())
        except ValueError:
            continue
        etype = event.get("type")
        if etype == "token":
            print(event.get("text") or "", end="", flush=True)
        elif etype == "error":
            print(f"\nhugpy chat: {event.get('message')}", file=sys.stderr)
            return 1
        elif etype == "done":
            finish = event.get("finish_reason") or "stop"
            break
    print()
    if finish == "cancelled":
        print("[cancelled]", file=sys.stderr)
    return 0


def _install_engine(args: argparse.Namespace) -> int:
    """Provision the native llama.cpp binaries (llama-server / rpc-server).

    The in-process engine (llama-cpp-python, ``hugpy[gguf]``) needs none of
    this; the always-on serve drivers and the GPU shard fleet do.
    """
    target = DISPATCH["install-engine"]
    install = resolve_target("install-engine")
    build = _import("hugpy_engine.native.build", target)
    resolve = _import("hugpy_engine.native.resolve", target)

    try:
        if args.build_from_source:
            info = build.build_from_source(cuda=args.cuda, tag=args.tag, jobs=args.jobs)
        else:
            try:
                info = install(cuda=args.cuda, tag=args.tag, force=args.force)
            except Exception as exc:
                print(f"hugpy: prebuilt fetch failed ({exc}); trying source build…",
                      file=sys.stderr)
                info = build.build_from_source(cuda=args.cuda, tag=args.tag, jobs=args.jobs)
    except Exception as exc:
        print(f"hugpy install-engine: failed: {exc}", file=sys.stderr)
        return 1

    print(f"hugpy engine ready — {info.get('note')}")
    print(f"  engine dir : {info.get('engine_dir')}")
    print(f"  llama-server: {info.get('server_bin') or resolve.server_bin()}")
    print(f"  rpc-server  : {info.get('rpc_bin') or '(not built)'}")
    if info.get("persisted_to"):
        print(f"  persisted   : {info['persisted_to']}")
    return 0


def _reclassify_images(args: argparse.Namespace) -> int:
    """Re-derive image-model tasks from each model dir's own declaration and
    re-stamp the wrong ones. Dry by default — pass --apply to write."""
    reclassify_images = resolve_target("reclassify-images")

    report = reclassify_images(apply=args.apply)
    changed = report["changed"]
    print(f"hugpy reclassify-images: scanned {report['scanned']} dir(s), "
          f"{len(changed)} to correct "
          f"({'APPLIED' if report['applied'] else 'dry run — nothing written'})")
    for c in changed:
        why = c.get("pipeline_class") or ("adapter-only dir" if c["adapter"] else c["source"])
        print(f"  {c['name']}: {c['from']} -> {c['to']}   [{why}]")
    if changed and not report["applied"]:
        print("  re-run with --apply to write the sidecars + discovery report")
    return 0


def _detect_profile() -> tuple[str, str]:
    """Pick cpu-worker/gpu-worker by probing local hardware. Returns
    (profile, reason) — the reason is always printed, detection is never silent."""
    try:
        from hugpy_platform.hardware import detect_gpus
    except Exception as exc:
        return "cpu", f"GPU detection unavailable ({type(exc).__name__}: {exc}); defaulting to cpu-worker"

    try:
        gpus = detect_gpus()
    except Exception as exc:
        return "cpu", f"GPU detection failed ({type(exc).__name__}: {exc}); defaulting to cpu-worker"

    if not gpus:
        return "cpu", "no usable NVIDIA GPU detected; choosing cpu-worker"

    names = ", ".join(f"{g.get('name')} (index {g.get('index')})" for g in gpus)
    return "gpu", f"detected {len(gpus)} GPU(s): {names}; choosing gpu-worker"


def _install_deps(args: argparse.Namespace) -> int:
    """Install a worker box's dependency set based on what the box actually is.

    Most fleet boxes have a GPU, so gpu-worker is the default profile — --cpu
    (or --profile cpu) is the explicit opt-out. --profile auto runs hardware
    detection instead of trusting the operator's say-so."""
    if args.cpu and args.profile not in (None, "cpu"):
        print(f"hugpy install-deps: --cpu conflicts with --profile {args.profile}", file=sys.stderr)
        return 2
    requested = "cpu" if args.cpu else (args.profile or "gpu")

    if requested == "auto":
        profile, reason = _detect_profile()
        print(f"hugpy install-deps: {reason}")
    else:
        profile = requested
        print(f"hugpy install-deps: profile = {profile}-worker "
              f"({'--cpu' if args.cpu else 'default' if args.profile is None else f'--profile {profile}'})")
        if profile == "gpu":
            _, detect_reason = _detect_profile()
            if not detect_reason.startswith("detected"):
                why = "it is the default" if args.profile is None else "you asked for it"
                print(f"hugpy install-deps: WARNING — {detect_reason.split('; ')[0]}; "
                      f"proceeding with gpu-worker anyway ({why}). "
                      f"Pass --cpu if this box has no GPU.")

    pkg = f"hugpy[{profile}-worker]"
    if args.version:
        pkg += f"=={args.version}"

    cmd = [sys.executable, "-m", "pip", "install", pkg]
    print("hugpy install-deps: would run:" if args.dry_run else "hugpy install-deps: will run:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0

    if not args.yes:
        try:
            reply = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nhugpy install-deps: aborted", file=sys.stderr)
            return 1
        if reply not in ("y", "yes"):
            print("hugpy install-deps: aborted", file=sys.stderr)
            return 1

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"hugpy install-deps: pip install failed (exit {proc.returncode})", file=sys.stderr)
        return proc.returncode
    print(f"hugpy install-deps: installed {pkg}")
    return 0


ECOSYSTEM_DISTRIBUTIONS = (
    "hugpy-platform", "hugpy-control", "hugpy-storage", "hugpy-engine",
    "hugpy-media", "hugpy-video", "hugpy-oracle", "hugpy-fleet",
    "hugpy-curation", "hugpy-ops", "hugpy-discord", "hugpy-server",
)


def _buildinfo():
    """``hugpy_platform.buildinfo`` when the installed platform ships it, else
    None. Lazy: a thin ``pip install hugpy`` may carry an older platform."""
    try:
        return importlib.import_module("hugpy_platform.buildinfo")
    except Exception:  # noqa: BLE001 — absent or broken: callers degrade
        return None


def _build(args: argparse.Namespace) -> int:
    """Print this interpreter's build identity (which commit is running)."""
    import json as _json

    bi = _buildinfo()
    if bi is None:
        print("hugpy build: hugpy_platform.buildinfo is not available in this install "
              "(hugpy-platform predates build identity; upgrade hugpy-platform)",
              file=sys.stderr)
        return 1
    if args.json:
        print(_json.dumps(bi.build_info(), indent=2, sort_keys=True))
    else:
        print(bi.identity_line())
    return 0


def _version(_args: argparse.Namespace) -> int:
    """Print the installed hugpy distributions (a thin install lists few),
    headed by the build identity line when the platform can produce one."""
    from importlib.metadata import PackageNotFoundError, version

    from hugpy import __version__

    bi = _buildinfo()
    if bi is not None:
        try:
            print(bi.identity_line())
        except Exception:  # noqa: BLE001 — identity is a courtesy, never a failure
            pass
    print(f"hugpy {__version__}")
    for dist in ECOSYSTEM_DISTRIBUTIONS:
        try:
            print(f"  {dist:<16} {version(dist)}")
        except PackageNotFoundError:
            print(f"  {dist:<16} (not installed)")
    return 0


# --------------------------------------------------------------------------- #
# parser + main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hugpy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the hugpy console + API in one process")
    s.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    s.add_argument("--port", type=int, default=7002, help="bind port (default: 7002)")
    s.add_argument("--threads", type=int, default=8, help="server threads (default: 8)")
    s.add_argument("--auth", choices=("open", "external"),
                   help="auth mode (default: open, or HUGPY_AUTH_MODE)")
    s.add_argument("--origins", help="comma-separated CORS origins (default: same-origin only)")
    s.add_argument("--debug", action="store_true")

    for name, help_text in PASSTHROUGH.items():
        # The owner's parser handles --help; add_help=False keeps ours quiet.
        sub.add_parser(name, help=help_text, add_help=False)

    b = sub.add_parser("bot", help="run the hugpy discord bot (drives a central over HTTP)")
    b.add_argument("--central",
                   help="hugpy central base URL the bot calls "
                        "(default: http://127.0.0.1:7002 or HUGPY_BASE_URL)")
    b.add_argument("--env", help="path to a .env with DISCORD_TOKEN/settings (sets HUGPY_BOT_ENV)")
    b.add_argument("--guild", type=int, help="restrict slash-command sync to one guild id")
    b.add_argument("--log-level", help="bot log level (default: INFO)")

    d = sub.add_parser("download", help="pull catalog model(s) into the local model store")
    d.add_argument("model_key", nargs="+", help="model key(s) as listed by the catalog")
    d.add_argument("--root", help="model store root override (default: DEFAULT_ROOT)")

    c = sub.add_parser("chat", help="stream a chat from a hugpy central in the "
                       "terminal (tracked + cancellable like every transport)")
    c.add_argument("prompt", nargs="*", help="the prompt (omit for interactive)")
    c.add_argument("--central", help="central base URL (default: HUGPY_BASE_URL "
                   "or http://127.0.0.1:7002)")
    c.add_argument("--model", help="model_key (default: central decides)")
    c.add_argument("--token", help="principal token (hpp_…) or API key "
                   "(default: HUGPY_TOKEN / HUGPY_API_KEY)")

    e = sub.add_parser("install-engine",
                       help="download/build the native llama.cpp server binary")
    e.add_argument("--cuda", action="store_true", help="fetch/build a CUDA-enabled engine")
    e.add_argument("--build-from-source", action="store_true",
                   help="cmake build instead of a prebuilt release (needs git+cmake)")
    e.add_argument("--tag", help="llama.cpp release tag (default: latest)")
    e.add_argument("--jobs", type=int, help="parallel build jobs (source build only)")
    e.add_argument("--force", action="store_true", help="re-download even if already installed")

    i = sub.add_parser("install-deps",
                       help="install a worker box's pip extras (gpu-worker by default, "
                            "--cpu for CPU-only boxes)")
    i.add_argument("--profile", choices=("gpu", "cpu", "auto"),
                   help="gpu-worker/cpu-worker, or auto to detect (default: gpu)")
    i.add_argument("--cpu", action="store_true", help="shorthand for --profile cpu")
    i.add_argument("--version", help="pin hugpy to this version (default: unpinned)")
    i.add_argument("--dry-run", action="store_true",
                   help="print the pip command and exit without running it")
    i.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    r = sub.add_parser("reclassify-images",
                       help="re-derive image-model tasks from each model dir's own "
                            "declaration (diffusers model_index.json / adapter-only "
                            "dirs) and re-stamp the wrong ones")
    r.add_argument("--apply", action="store_true",
                   help="write the corrected tasks (default: dry run)")

    sub.add_parser("version", help="show installed hugpy distributions")

    bld = sub.add_parser("build", help="show which build (version + commit) this "
                                       "interpreter runs")
    bld.add_argument("--json", action="store_true",
                     help="the full build document (all distributions, lockstep, workspace)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        # Everything after a passthrough subcommand belongs to the owner's parser.
        if argv and argv[0] in PASSTHROUGH:
            return _passthrough(argv[0], argv[1:])
        args = parser.parse_args(argv)
        if args.cmd == "serve":
            return _serve(args, argv[1:])
        if args.cmd == "bot":
            return _bot(args)
        if args.cmd == "download":
            return _download(args)
        if args.cmd == "chat":
            return _chat(args)
        if args.cmd == "install-engine":
            return _install_engine(args)
        if args.cmd == "install-deps":
            return _install_deps(args)
        if args.cmd == "reclassify-images":
            return _reclassify_images(args)
        if args.cmd == "version":
            return _version(args)
        if args.cmd == "build":
            return _build(args)
    except DispatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
