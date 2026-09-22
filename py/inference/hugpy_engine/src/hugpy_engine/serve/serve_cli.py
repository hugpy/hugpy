#!/usr/bin/env python3
"""serve_cli.py — drive the serving plans from the shell.

Dry-run is the default and needs no privileges. --apply executes, which for
systemd/swap means writing under /etc and calling systemctl, so run that with
sudo. Local execution only; for a peer node, import apply_plan and pass an
SSH-backed run/write instead (the plan is host-agnostic argv).

    python3 -m abstract_hugpy_dev.managers.serve.serve_cli status
    python3 -m abstract_hugpy_dev.managers.serve.serve_cli install            # dry run
    sudo python3 -m abstract_hugpy_dev.managers.serve.serve_cli install --apply
    sudo python3 -m abstract_hugpy_dev.managers.serve.serve_cli install --only flux2 --apply
    sudo python3 -m abstract_hugpy_dev.managers.serve.serve_cli start flux2 --apply
    sudo python3 -m abstract_hugpy_dev.managers.serve.serve_cli stop flux2 --apply
"""

import argparse
import os
import subprocess

from hugpy_engine.serve.serve import (
    install_serving,
    start_serving,
    stop_serving,
    serving_overview,
    apply_plan,
)


def _run(argv):
    return subprocess.run(argv, check=True)


def _write(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def main():
    parser = argparse.ArgumentParser(description="Stand up / control model serving.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print serving_overview(); no side effects")

    p_install = sub.add_parser("install", help="generate units / swap config")
    p_install.add_argument("--only", nargs="*", default=None, help="limit to these model keys")
    p_install.add_argument("--apply", action="store_true", help="execute (needs sudo)")

    p_start = sub.add_parser("start")
    p_start.add_argument("key")
    p_start.add_argument("--apply", action="store_true")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("key")
    p_stop.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    if args.cmd == "status":
        for row in serving_overview():
            print(row)
        return

    if args.cmd == "install":
        plan = install_serving(only=args.only)
    elif args.cmd == "start":
        plan = start_serving(args.key)
    else:
        plan = stop_serving(args.key)

    if not args.apply:
        print("# dry run — pass --apply to execute")
        for line in plan.describe():
            print(line)
        return

    apply_plan(plan, run=_run, write=_write)
    print("applied.")


if __name__ == "__main__":
    main()

