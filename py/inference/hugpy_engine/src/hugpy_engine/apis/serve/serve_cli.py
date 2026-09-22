"""Deprecated location — re-export of the canonical serve CLI.

The implementation now lives in ``hugpy_engine.serve.serve_cli``; this shim
keeps the old import path valid.
"""
from hugpy_engine.serve.serve_cli import main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
