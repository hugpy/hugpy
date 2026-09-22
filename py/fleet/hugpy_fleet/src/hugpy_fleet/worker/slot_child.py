"""Entry point for a worker's slot child: ``python -m hugpy_fleet.worker.slot_child``.

The slot supervisor in ``worker.agent`` spawns one of these per slot. It
installs the fleet's storage providers (the child calls
``ensure_model_present`` itself) and then runs the engine's slot agent
unchanged. Keeping the wrapper in fleet means the engine never imports the
fleet to get its gate/telemetry.
"""

from __future__ import annotations

import sys

__all__ = ["main"]


def main() -> int:
    from hugpy_fleet.worker.storage_hooks import install_storage_providers
    # No executor registrar in a child: there is no agent restart path to drain.
    install_storage_providers(with_registrar=False)
    from hugpy_engine.serve.slot_agent import main as _slot_main
    return int(_slot_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
