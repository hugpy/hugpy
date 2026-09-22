"""Declarative eliminate-then-rank resolution — one framework, many pipelines.

A *pipeline* is an ordered tuple of named *stages*.  Each stage is one of:

  - ``eliminate`` — a keep-predicate ``fn(candidate, call) -> bool``.  Candidates
    that fail are dropped and recorded (``rejected_at`` = the stage name), so the
    decision log says *where* each loser fell out.
  - ``rank`` — a sort-key ``fn(candidate, call) -> comparable``.  Applied as a
    STABLE sort, so **the LAST ranker in the pipeline dominates** and earlier
    rankers only break ties beneath it.  This is how precedence is spelled out:
    put the weakest tiebreaker first and the strongest preference last.

The framework is stdlib-only and emits the existing oracle records
(:class:`~hugpy_oracle.selection.CandidateVerdict` /
:class:`SelectionDecision`) as its envelope, so oracle callers are unchanged.
Two pipelines are expressed on it:

  * **name resolution** (``pipelines.name``) — fuzzy model-NAME → canonical key,
    used by ``assure_model_key`` (closest-string / blocked / fit / servable /
    adjusted-usage / starred).
  * **oracle selection** (``pipelines.oracle``) — the per-capability selector,
    a stage-for-stage mirror of ``selection.select`` steps 1-9, validated at
    parity before anything switches to it.

Nothing here reaches the network or a fleet; every fact a stage needs arrives
on the ``call`` object or the ``Candidate`` record, both injectable for tests.
"""
# Implementation moved DOWN to hugpy_engine.name_match (2026-09-22): the
# engine's assure_model_key needs it and must not import the oracle.
from hugpy_engine.name_match import Candidate, Stage, StageRegistry, Unresolvable, run_pipeline  # noqa: F401

__all__ = [
    "Candidate",
    "Stage",
    "StageRegistry",
    "Unresolvable",
    "run_pipeline",
]
