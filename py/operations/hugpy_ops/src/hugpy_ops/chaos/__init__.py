"""chaos — the chaos-and-learn exerciser (p1, EPOCH CLOSER).

Randomly exercises the live assortment (models x cards x alloc modes x ctx%)
through the REAL public paths and records ONE predicted-vs-measured observation
per trial, so the t28 learner can learn placement templates from measured
reality. See ``schema.py`` (the contract) and ``SCHEMA.md``.

Entry point:  hugpy-chaos [--dry-run] [--rounds N] ...  /  hugpy-chaos sweep ...
              (console script; ``python -m hugpy_ops.chaos`` is equivalent)

This module is SELF-CONTAINED: it never edits worker-agent internals, flex.py,
or need-pricing (the learner's / worker's turf). It only drives HTTP and reads
/models/<key>/meta for central's own cheap prediction."""

__all__ = ["SCHEMA_VERSION"]


def __getattr__(name):
    # Lazy: schema.py shares its alloc-mode vocabulary with hugpy_engine, so
    # `import hugpy_ops.chaos` alone must not pull the engine in.
    if name == "SCHEMA_VERSION":
        from hugpy_ops.chaos.schema import SCHEMA_VERSION
        return SCHEMA_VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
