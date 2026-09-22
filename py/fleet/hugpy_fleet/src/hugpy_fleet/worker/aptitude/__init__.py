"""Worker-side aptitude self-test — the MECHANICAL half of the studio bench.

``score.py`` and ``cases.py`` are ported VERBATIM from
``evaluations/studio-aptitude/``; ``parse.py`` carries only that bench's
reply-parsing helper. Deliberately NOT ported: the bench's HTTP client, its
sweep driver, and its LLM judge — a worker must never make a network call or
ask a model to grade another model in order to report its own health.

Everything in this package is pure except ``selftest.maybe_run``, which is
gated OFF by default (``HUGPY_WORKER_SELFTEST=on``). Importing this package
pulls no runner, no torch, no dispatch.
"""
from __future__ import annotations

__all__ = ["cases", "parse", "score", "selftest"]
