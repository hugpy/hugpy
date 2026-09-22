"""CLI: hugpy-curation review <command>

  screen  <hub_id> [...]        metadata-only verdict, downloads nothing
  review  <hub_id> [...]        screen, then download + load-test if it passes
  run     <criteria>            full pipeline over a saved criteria's search
  criteria list|show|set        manage saved criteria
  reports <criteria>            what previous runs concluded
  push    [--run N|--all]       re-send finished runs to central

`push` replays what the automatic post-run hand-off could not deliver (central
down, box offline). Same code path as the automatic one — see push.py for the
REVIEW_CENTRAL_URL / REVIEW_CENTRAL_TOKEN environment.
"""
from __future__ import annotations

import argparse
import json
import sys

from hugpy_curation.review.criteria import (
    ReviewCriteria,
    list_criteria,
    load_criteria,
    save_criteria,
)


def _crit_from_args(args) -> ReviewCriteria:
    if getattr(args, "criteria", None):
        return load_criteria(args.criteria)
    c = ReviewCriteria(name="adhoc", query=getattr(args, "query", "") or "")
    if getattr(args, "vram_gib", None):
        c.vram_bytes = int(args.vram_gib * 1024**3)
    if getattr(args, "context", None):
        c.target_context = args.context
    if getattr(args, "no_judge", False):
        c.judge = False
    if getattr(args, "incumbent", None):
        c.incumbents = list(args.incumbent)
    return c


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hugpy-curation review", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--criteria", help="saved criteria name")
    common.add_argument("--vram-gib", type=float, help="override VRAM budget")
    common.add_argument("--context", type=int, help="override target context")
    common.add_argument("--incumbent", action="append",
                        help="a model you already run (repeatable)")
    common.add_argument("--no-judge", action="store_true",
                        help="skip the hugpy-agent verdict")
    common.add_argument("--json", action="store_true", help="raw JSON output")

    sp = sub.add_parser("screen", parents=[common], help="metadata only")
    sp.add_argument("hub_id", nargs="+")

    rp = sub.add_parser("review", parents=[common],
                        help="screen, then download + load-test survivors")
    rp.add_argument("hub_id", nargs="+")

    up = sub.add_parser("run", parents=[common], help="run a saved criteria")
    up.add_argument("criteria_name")
    up.add_argument("--force", action="store_true",
                    help="re-screen repos reviewed recently")
    up.add_argument("--report", help="write a markdown report to this path")

    cp = sub.add_parser("criteria", help="manage saved criteria")
    cp.add_argument("action", choices=["list", "show", "set"])
    cp.add_argument("name", nargs="?")
    cp.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE", help="field to set (repeatable)")

    pu = sub.add_parser("push", help="send finished runs to central")
    pu.add_argument("--run", type=int, help="one local run id")
    pu.add_argument("--all", action="store_true",
                    help="every run with no successful push yet")
    pu.add_argument("--limit", type=int, default=50,
                    help="with --all, how many runs to replay (default 50)")

    rep = sub.add_parser("reports", help="previous conclusions")
    rep.add_argument("criteria_name", nargs="?")
    rep.add_argument("--limit", type=int, default=20)
    rep.add_argument("--best", action="store_true",
                     help="leaderboard instead of chronological")

    args = p.parse_args(argv)

    if args.cmd == "criteria":
        return _criteria_cmd(args)
    if args.cmd == "reports":
        return _reports_cmd(args)
    if args.cmd == "push":
        return _push_cmd(args)

    from hugpy_curation.review import pipeline

    if args.cmd in ("screen", "review"):
        crit = _crit_from_args(args)
        out = []
        for hub_id in args.hub_id:
            rv = pipeline.review_one(hub_id, crit,
                                     download=(args.cmd == "review"))
            out.append(rv.to_dict())
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        return 0

    # run
    crit = load_criteria(args.criteria_name)
    result = pipeline.run(crit, force=args.force)
    md = pipeline.report_markdown(result)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"report written to {args.report}")
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print()
        print(md)
    return 0


def _criteria_cmd(args) -> int:
    if args.action == "list":
        names = list_criteria()
        print("\n".join(names) if names else "(no saved criteria)")
        return 0
    if not args.name:
        print("a criteria name is required", file=sys.stderr)
        return 2
    if args.action == "show":
        print(json.dumps(load_criteria(args.name).to_dict(), indent=2))
        return 0

    # set: create or update, one KEY=VALUE at a time
    try:
        c = load_criteria(args.name)
    except OSError:
        c = ReviewCriteria(name=args.name)
    for pair in args.set:
        if "=" not in pair:
            print(f"expected KEY=VALUE, got {pair!r}", file=sys.stderr)
            return 2
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in ReviewCriteria.__dataclass_fields__:
            print(f"unknown field {key!r}", file=sys.stderr)
            return 2
        cur = getattr(c, key)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, list):
            val = [v.strip() for v in raw.split(",") if v.strip()]
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif cur is None:
            val = None if raw in ("", "none", "null") else raw
        else:
            val = raw
        setattr(c, key, val)
    path = save_criteria(c)
    print(f"saved {path}")
    return 0


def _push_cmd(args) -> int:
    """Manual replay of the automatic post-run push. Exit 1 when nothing
    landed, so a cron wrapper can notice; exit 0 when at least one run went."""
    from hugpy_curation.review import push as push_mod
    if not push_mod.central_url():
        print("REVIEW_CENTRAL_URL is not set — nothing to push to",
              file=sys.stderr)
        return 2
    if args.run is None and not args.all:
        print("give --run <id> or --all", file=sys.stderr)
        return 2
    results = ([push_mod.push_run(args.run, log=print)] if args.run is not None
               else push_mod.push_pending(limit=args.limit, log=print))
    if not results:
        print("(nothing pending)")
        return 0
    ok = [r for r in results if r.get("ok")]
    print(f"{len(ok)}/{len(results)} run(s) pushed")
    return 0 if ok else 1


def _reports_cmd(args) -> int:
    from hugpy_curation.review import store
    if args.best:
        if not args.criteria_name:
            print("--best needs a criteria name", file=sys.stderr)
            return 2
        rows = store.leaderboard(args.criteria_name, limit=args.limit)
    else:
        rows = store.recent(args.criteria_name, limit=args.limit)
    if not rows:
        print("(nothing recorded yet)")
        return 0
    for r in rows:
        s = (r.get("payload") or {}).get("screen") or {}
        mark = "✓" if r.get("passed") else "✗"
        extra = f" verdict={r['verdict']}" if r.get("verdict") else ""
        print(f"{mark} {r['hub_id']}  score={r.get('score')} "
              f"stage={r['stage']} quant={s.get('best_quant')}{extra}")
        if not r.get("passed") and s.get("reasons"):
            print(f"    {'; '.join(s['reasons'][:2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
