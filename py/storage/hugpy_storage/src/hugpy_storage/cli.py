"""``hugpy-storage`` — the storage package's command line.

    hugpy-storage daemon [--max-concurrent N] [--poll SECONDS]
        Run the model-download daemon (claims download jobs off the shared
        comms queue). The ``hugpy-downloader`` script is an alias for this.

    hugpy-storage sync --central URL (--model KEY ... | --all | --list)
        Pull model directories from a central node (``model_sync``).

    hugpy-storage status
        Where this box keeps models and whether a daemon is alive.

Heavy imports happen inside the subcommands so ``--help`` is instant and
never needs huggingface_hub or a comms database.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hugpy-storage",
        description="Hugpy storage: download daemon, central sync, inventory status.")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("daemon", help="run the model-download daemon")
    d.add_argument("--max-concurrent", type=int, default=None,
                   help="transfers at once (default: HUGPY_DOWNLOADER_MAX_CONCURRENT or 2)")
    d.add_argument("--poll", type=float, default=None,
                   help="queue poll interval in seconds (default: HUGPY_DOWNLOADER_POLL_SECONDS or 1.5)")

    s = sub.add_parser("sync", help="pull model directories from a central node")
    s.add_argument("--central", required=True, help="central base URL, e.g. https://hugpy.ai")
    s.add_argument("--model", action="append", dest="models", help="model_key to pull (repeatable)")
    s.add_argument("--all", action="store_true", help="pull every model central lists")
    s.add_argument("--root", default=None, help="local storage root override")
    s.add_argument("--list", action="store_true", help="just list central's models and exit")

    sub.add_parser("status", help="show the model store roots and daemon liveness")
    return parser


def _cmd_daemon(args: argparse.Namespace) -> int:
    if args.max_concurrent is not None:
        os.environ["HUGPY_DOWNLOADER_MAX_CONCURRENT"] = str(args.max_concurrent)
    if args.poll is not None:
        os.environ["HUGPY_DOWNLOADER_POLL_SECONDS"] = str(args.poll)
    from hugpy_storage.downloader.daemon import main as daemon_main
    return int(daemon_main([]) or 0)


def _cmd_sync(args: argparse.Namespace) -> int:
    from hugpy_storage.model_sync import main as sync_main
    argv = ["--central", args.central]
    for m in args.models or []:
        argv += ["--model", m]
    if args.all:
        argv.append("--all")
    if args.list:
        argv.append("--list")
    if args.root:
        argv += ["--root", args.root]
    return int(sync_main(argv) or 0)


def _cmd_status(_args: argparse.Namespace) -> int:
    from hugpy_platform.constants import DEFAULT_ROOT, MODELS_HOME
    from hugpy_storage.catalog_source import get_catalog_source
    from hugpy_storage.downloader.presence import downloader_alive, heartbeat_path
    print(f"default root : {DEFAULT_ROOT}")
    print(f"models home  : {MODELS_HOME}")
    print(f"comms db     : {os.environ.get('HUGPY_COMMS_DB') or '(default)'}")
    print(f"daemon       : {'alive' if downloader_alive() else 'not running'} "
          f"({heartbeat_path()})")
    src = get_catalog_source()
    print(f"catalog      : {getattr(src, 'name', type(src).__name__)}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "daemon":
        return _cmd_daemon(args)
    if args.command == "sync":
        return _cmd_sync(args)
    if args.command == "status":
        return _cmd_status(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
