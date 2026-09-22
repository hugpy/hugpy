"""The two concrete pipelines built on :mod:`oracle.pipeline`.

* :data:`NAME_PIPELINE` / :data:`name_registry` — resolve a fuzzy model NAME to
  a canonical registry key.  This is the pipeline ``assure_model_key`` calls
  once its exact/slug/hub/folder/bare-tail cascade has failed.  Precedence is
  spelled out by ordering (last ranker dominates):

      closest-string   (weakest rank: token/edit distance — pure tiebreaker)
      exact            (eliminate: alloc.exact -> keep only the requested ns)
      blocked          (eliminate: operator-blocked / unclassified / adapter)
      fit              (eliminate: predesignated to fit VRAM)
      servable         (eliminate: has a concrete loadable artifact)
      adjusted-usage   (rank: n_calls - tie_driven_picks)
      serving          (rank: prefer a model a worker holds right now)
      starred          (strongest rank: an operator-starred model wins)

* :data:`ORACLE_PIPELINE` / :data:`oracle_registry` — a stage-for-stage mirror
  of :func:`selection.select` steps 1-9, used only behind the parity gate.

The stage bodies delegate to existing seams; every seam read is fail-open so a
metrics DB round-trip or a missing worker store never raises into resolution.
"""
# Implementation moved DOWN to hugpy_engine.name_match (2026-09-22).
from hugpy_engine.name_match import (  # noqa: F401
    NAME_PIPELINE,
    Candidate,
    NameCall,
    StageRegistry,
    _distance,
    _tokens,
    name_registry,
    resolve_name,
    run_pipeline,
)
