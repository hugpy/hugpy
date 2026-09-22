"""``hugpy-oracle`` — the package's command line (steward pass, hook wiring).

Subcommands:

``steward``
    One steward pass (k113c): the selection system checks itself and,
    with ``--apply``, applies BOUNDED rebalancing to the selection policy.

    Two modes, chosen by ``--api`` / ``$HUGPY_API``:

    * remote — POST (``--apply``) or GET the running central's
      ``/api/oracle/steward`` with the stdlib only (no Flask, no requests),
      so the rebalance lands on the LIVE process selector. This is what the
      ``deploy/hugpy-steward.service`` unit does on a box that runs central.
    * local — open the shared reliability ledger
      (``hugpy_oracle.config.ledger_path``), build the newest routing matrix
      and the catalog's eligible-model sets, and run :class:`Steward`
      in-process. ``--apply`` rebalances this process's selector only (the
      report is what matters here; a persisted policy is TODO-18).

    The report is JSON on stdout. Exit status: 0 when the report is ``ok``,
    2 when it carries an alarm, 1 on a transport/ledger failure — so a
    systemd oneshot shows failed and that IS the alert.

No pathlib; os.path only (project discipline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

__all__ = ["main", "build_parser", "run_steward_local", "run_steward_remote"]

API_ENV = "HUGPY_API"
STEWARD_PATH = "/api/oracle/steward"


def build_parser() -> argparse.ArgumentParser:
    from hugpy_oracle import __version__

    parser = argparse.ArgumentParser(
        prog="hugpy-oracle",
        description="Hugpy oracle: creative planning, model selection and "
                    "the selection steward.")
    parser.add_argument("--version", action="version", version=f"hugpy-oracle {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    steward = sub.add_parser("steward", help="run one steward self-check pass")
    steward.add_argument("--apply", action="store_true",
                         help="apply bounded rebalancing (default: report only)")
    steward.add_argument("--api", default=None,
                         help=f"central base URL; default ${API_ENV}. Unset = run in-process")
    steward.add_argument("--local", action="store_true",
                         help=f"run in-process even when ${API_ENV} is set")
    steward.add_argument("--ledger", default=None,
                         help="reliability ledger path (local mode; default: ORACLE_LEDGER_PATH)")
    steward.add_argument("--timeout", type=float, default=120.0,
                         help="HTTP timeout in seconds (remote mode)")
    steward.add_argument("--compact", action="store_true",
                         help="single-line JSON instead of indented")
    steward.set_defaults(func=_cmd_steward)

    hooks = sub.add_parser("install-hooks",
                           help="install the oracle's video hooks in this process and report what was wired")
    hooks.set_defaults(func=_cmd_install_hooks)
    return parser


# ---------------------------------------------------------------------------
# steward
# ---------------------------------------------------------------------------

def run_steward_remote(base_url: str, *, apply: bool, timeout: float = 120.0) -> dict[str, Any]:
    """POST/GET ``<base_url>/api/oracle/steward``; the decoded JSON body."""
    import urllib.request

    url = base_url.rstrip("/") + STEWARD_PATH
    req = urllib.request.Request(url, method="POST" if apply else "GET",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-supplied base URL
        raw = resp.read().decode("utf-8", "replace")
    body = json.loads(raw) if raw.strip() else {}
    if not isinstance(body, dict):
        body = {"ok": False, "error": "non-object steward response", "body": body}
    return body


def run_steward_local(*, apply: bool, ledger_path: Optional[str] = None) -> dict[str, Any]:
    """The steward pass in-process over the shared reliability ledger."""
    from hugpy_oracle import selection
    from hugpy_oracle.steward import Steward

    path = ledger_path or selection.default_ledger_path()
    ledger = selection.ReliabilityLedger(path)
    try:
        selector = None
        matrix = None
        if apply:
            selector = selection.process_selector()
        if selector is not None:
            matrix = selector.matrix()
        else:
            try:
                from hugpy_oracle import routing_matrix
                matrix = routing_matrix.load_matrix()
            except Exception:  # noqa: BLE001 — a missing matrix is a staleness finding, not a crash
                matrix = None
        eligible: dict[str, tuple[str, ...]] = {}
        catalog_error: Optional[str] = None
        try:
            from hugpy_oracle import catalog
            for view in catalog.list_capabilities():
                if view.model_ids:
                    eligible[view.name] = tuple(view.model_ids)
        except Exception as exc:  # noqa: BLE001 — starvation checks degrade, the rest runs
            catalog_error = f"{type(exc).__name__}: {exc}"
        steward = Steward(ledger, selector=selector, matrix=matrix, eligible_models=eligible)
        report = steward.check()
        body = report.to_dict()
        body["applied"] = bool(apply and selector is not None)
        body["ledger_path"] = path
        body["ledger_rows"] = ledger.count()
        if catalog_error:
            body["catalog_error"] = catalog_error
        return body
    finally:
        ledger.close()


def _cmd_steward(args: argparse.Namespace) -> int:
    api = None if args.local else (args.api or os.environ.get(API_ENV, "").strip() or None)
    try:
        if api:
            body = run_steward_remote(api, apply=args.apply, timeout=args.timeout)
        else:
            body = run_steward_local(apply=args.apply, ledger_path=args.ledger)
    except Exception as exc:  # noqa: BLE001 — the unit must show failed, with the reason
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    if args.compact:
        print(json.dumps(body, default=str))
    else:
        print(json.dumps(body, indent=2, default=str))
    return 0 if body.get("ok", True) else 2


# ---------------------------------------------------------------------------
# install-hooks
# ---------------------------------------------------------------------------

def _cmd_install_hooks(args: argparse.Namespace) -> int:
    from hugpy_oracle import install_hooks
    print(json.dumps(install_hooks(), indent=2, default=str))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
