"""``hugpy-video`` command line (minimal; the console script in pyproject).

    hugpy-video --help
    hugpy-video jobs list [--all] [--limit N] [--json]
    hugpy-video jobs registry
    hugpy-video selftest            # in-process checks, no GPU/service

Everything heavy is imported inside the sub-command that needs it.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _cmd_jobs_list(args) -> int:
    from hugpy_video import jobs

    rows = jobs.list_jobs(include_terminal=args.all, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        print("no jobs")
        return 0
    for r in rows:
        print(f"{r.get('job_id','?'):34} {r.get('name','?'):26} {r.get('status','?')}")
    return 0


def _cmd_jobs_registry(args) -> int:
    from hugpy_video import jobs

    for name, spec in sorted(jobs.registered_jobs().items()):
        print(f"{name:26} {spec.runner_key[0]}/{spec.runner_key[1]:18} "
              f"queue={spec.queue} timeout={spec.timeout_s}s")
    return 0


def _cmd_selftest(args) -> int:
    from hugpy_video.selftest import run_selftest

    failures = run_selftest(verbose=True)
    return 1 if failures else 0


def _cmd_state(args) -> int:
    from dataclasses import asdict

    from hugpy_video.state import get_state_roots

    print(json.dumps(asdict(get_state_roots()), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from hugpy_video import __version__

    p = argparse.ArgumentParser(prog="hugpy-video",
                                description="Hugpy video: media job bus, runners, studio.")
    p.add_argument("--version", action="version", version=f"hugpy-video {__version__}")
    sub = p.add_subparsers(dest="cmd")

    jobs = sub.add_parser("jobs", help="media job bus")
    jsub = jobs.add_subparsers(dest="jobs_cmd")
    ls = jsub.add_parser("list", help="list jobs")
    ls.add_argument("--all", action="store_true", help="include terminal jobs")
    ls.add_argument("--limit", type=int, default=50)
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_cmd_jobs_list)
    reg = jsub.add_parser("registry", help="registered job names and runner keys")
    reg.set_defaults(func=_cmd_jobs_registry)

    st = sub.add_parser("selftest", help="in-process self checks (no GPU, no service)")
    st.set_defaults(func=_cmd_selftest)

    state = sub.add_parser("state", help="print the resolved state roots")
    state.set_defaults(func=_cmd_state)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
