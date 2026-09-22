"""hugpy-sentinel — run-once | status | record-scorecard.

`run-once` is what the systemd timer calls: one detect -> case -> spawn
pass, then exit. `status` lists the case table for a human.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys

from hugpy_ops.sentinel.cases import CaseStore
from hugpy_ops.sentinel.checks import append_scorecard
from hugpy_ops.sentinel.runner import run_once
from hugpy_ops.sentinel.settings import load_settings


def _fmt_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def cmd_run_once(args) -> int:
    settings = load_settings()
    summary = run_once(settings)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_status(args) -> int:
    settings = load_settings()
    store = CaseStore(settings.db_path)
    try:
        cases = store.list(state=args.state)
        if not cases:
            print("no cases")
            return 0
        for c in cases:
            print("case %4d  %-13s %-8s %-22s opened %s  last %s  %s"
                  % (c.id, c.state, c.severity, c.kind,
                     _fmt_ts(c.opened_at), _fmt_ts(c.last_seen),
                     c.fingerprint))
            if c.report_path:
                print("           report: %s" % c.report_path)
            if c.note:
                print("           note:   %s" % c.note)
        return 0
    finally:
        store.close()


def cmd_record_scorecard(args) -> int:
    settings = load_settings()
    detail = json.loads(args.detail) if args.detail else {}
    append_scorecard(settings, args.capability, args.model,
                     hard_pass=args.hard_pass == "1", detail=detail)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="hugpy-sentinel",
        description="hugpy sentinel: bound-exceeded detection -> case -> "
                    "one-shot diagnosis agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run-once", help="one detect->case->spawn pass "
                                        "(what the timer calls)")
    p.set_defaults(fn=cmd_run_once)

    p = sub.add_parser("status", help="list cases")
    p.add_argument("--state", help="filter by state")
    p.set_defaults(fn=cmd_status)

    # The sentinel never calls POST /oracle/route itself (route EXECUTES);
    # route callers feed observed scorecards into the streak history here.
    p = sub.add_parser("record-scorecard",
                       help="append one observed /oracle/route scorecard "
                            "to the sentinel's streak history")
    p.add_argument("--capability", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hard-pass", dest="hard_pass", choices=["0", "1"],
                   required=True)
    p.add_argument("--detail", help="JSON detail (e.g. worker, repair_code)")
    p.set_defaults(fn=cmd_record_scorecard)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
