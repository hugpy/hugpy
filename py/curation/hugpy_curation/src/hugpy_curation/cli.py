"""``hugpy-curation`` — the package's command line.

Subcommands:

``review ...``
    The review pipeline's own CLI (``hugpy_curation.review.__main__``):
    ``screen``, ``review``, ``run <criteria>``, ``criteria``, ``reports``,
    ``push``. Everything after ``review`` is handed to it untouched, so the
    systemd units run ``hugpy-curation review run %i --report ...``.

``dossier list [criteria]``
    The discovery dossiers the store holds — every criteria, or one — as the
    compact summary rows the console lists, newest first.

``install-providers``
    Wire this process's seams: the dossier store into
    ``hugpy_oracle.providers`` (``set_dossier_source``) and, when a fleet
    doctrine source is importable, the doctrine into
    ``hugpy_curation.providers``. Prints what was wired as JSON.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    from hugpy_curation import __version__

    parser = argparse.ArgumentParser(
        prog="hugpy-curation",
        description="Hugpy curation: discovery dossiers and the HF model review pipeline.")
    parser.add_argument("--version", action="version",
                        version=f"hugpy-curation {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser(
        "review", help="the review pipeline (screen / review / run / criteria / reports / push)",
        description="Arguments are passed to the review pipeline's CLI; "
                    "`hugpy-curation review --help` lists them.")
    review.add_argument("args", nargs=argparse.REMAINDER,
                        help="review pipeline arguments")
    review.set_defaults(func=_cmd_review)

    dossier = sub.add_parser("dossier", help="discovery dossiers")
    dsub = dossier.add_subparsers(dest="dossier_command", required=True)
    dlist = dsub.add_parser("list", help="list stored dossiers (summary rows)")
    dlist.add_argument("criteria", nargs="?", default=None,
                       help="one criteria name (default: every criteria)")
    dlist.add_argument("--limit", type=int, default=50,
                       help="rows per criteria (default 50)")
    dlist.add_argument("--json", action="store_true", help="JSON instead of a table")
    dlist.set_defaults(func=_cmd_dossier_list)

    providers = sub.add_parser(
        "install-providers",
        help="install curation's dossier source in the oracle (and a doctrine source when available)")
    providers.set_defaults(func=_cmd_install_providers)
    return parser


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def _cmd_review(args: argparse.Namespace) -> int:
    from hugpy_curation.review.__main__ import main as review_main
    rest = list(args.args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return int(review_main(rest or ["--help"]) or 0)


# ---------------------------------------------------------------------------
# dossier list
# ---------------------------------------------------------------------------

def dossier_rows(criteria: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
    """Summary rows for the stored dossiers, newest first per criteria."""
    import os
    from hugpy_curation.dossier import store as dstore

    if criteria:
        names = [criteria]
    else:
        root = dstore.root_dir()
        try:
            names = sorted(n for n in os.listdir(root)
                           if os.path.isdir(os.path.join(root, n)))
        except OSError:
            names = []
    rows: list[dict[str, Any]] = []
    for name in names:
        for path in dstore.list_for(name)[:max(1, int(limit))]:
            dossier = dstore.load_path(path)
            if dossier is not None:
                rows.append(dstore.summary(dossier, path))
    return rows


def _cmd_dossier_list(args: argparse.Namespace) -> int:
    rows = dossier_rows(args.criteria, args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        from hugpy_curation.dossier import store as dstore
        print(f"(no dossiers under {dstore.root_dir()})")
        return 0
    for r in rows:
        verdict = r.get("verdict") or "-"
        print(f"{r.get('criteria')}\t{r.get('hub_id')}\t{verdict}\t"
              f"{r.get('best_quant') or '-'}\t{r.get('generated_at') or ''}")
    return 0


# ---------------------------------------------------------------------------
# install-providers
# ---------------------------------------------------------------------------

def _cmd_install_providers(args: argparse.Namespace) -> int:
    from hugpy_curation import install_providers
    report = install_providers()
    print(json.dumps(report, indent=2, default=str))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
