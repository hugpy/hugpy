"""``hugpy-fleet`` — the operator's fleet CLI.

Today it fronts the WireGuard join flow (the one-step remote-worker onboarding),
so the operator can run, on the hub::

    hugpy-fleet join-code a-brain        # -> self-contained join script on stdout
    hugpy-fleet list                     # -> registered wg peers
    hugpy-fleet revoke a-brain           # -> remove a wg peer

The heavy lifting lives in ``hugpy_fleet.central.wg_join``; this is the console
entry point that pyproject wires to the ``hugpy-fleet`` command. New fleet
subcommands can hang off ``build_parser`` here later without disturbing the
existing per-arm commands (``hugpy-worker`` etc.).
"""
from __future__ import annotations

import sys
from typing import Optional

from hugpy_fleet.central import wg_join


def main(argv: Optional[list] = None) -> int:
    # The join verbs (join-code/list/revoke) ARE the fleet CLI's surface today,
    # so we reuse wg_join's parser under the `hugpy-fleet` program name.
    return wg_join.main(argv, prog="hugpy-fleet")


if __name__ == "__main__":
    raise SystemExit(main())
