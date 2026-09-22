"""Name matching: fuzzy model-name resolution and the eliminate-then-rank pipeline.

Moved DOWN from ``hugpy_oracle.pipeline`` / ``hugpy_oracle.pipelines``
(2026-09-22): ``assure_model_key`` needs these pure helpers and the engine must
not import the oracle. The oracle modules re-export every name from here.

A *pipeline* is an ordered tuple of named *stages*.  Each stage is one of:

  - ``eliminate`` — a keep-predicate ``fn(candidate, call) -> bool``.  Candidates
    that fail are dropped and recorded, so the decision log says *where* each
    loser fell out.
  - ``rank`` — a sort-key ``fn(candidate, call) -> comparable``.  Applied as a
    STABLE sort, so **the LAST ranker in the pipeline dominates** and earlier
    rankers only break ties beneath it.

The NAME pipeline (:data:`NAME_PIPELINE` / :data:`name_registry`) resolves a
fuzzy model NAME to a canonical registry key once ``assure_model_key``'s
exact / slug / hub / folder / bare-tail cascade has failed.  Precedence is
spelled out by ordering (last ranker dominates):

      closest-string   (weakest rank: token distance — pure tiebreaker)
      exact            (eliminate: alloc.exact -> keep only the requested ns)
      blocked          (eliminate: operator-blocked / unclassified / adapter)
      fit              (eliminate: predesignated to fit VRAM)
      servable         (eliminate: has a concrete loadable artifact)
      adjusted-usage   (rank: n_calls - tie_driven_picks)
      serving          (rank: prefer a model a worker holds right now)
      starred          (strongest rank: an operator-starred model wins)

Stdlib-only; every fact a stage needs arrives on the ``call`` object or the
:class:`Candidate` record (a plain frozen dataclass), both injectable for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional

__all__ = [
    "Candidate",
    "NAME_PIPELINE",
    "NameCall",
    "Stage",
    "StageRegistry",
    "Unresolvable",
    "name_registry",
    "resolve_name",
    "run_pipeline",
    "_distance",
    "_tokens",
]

class Unresolvable(Exception):
    """Every candidate was eliminated (or there were none to begin with).

    Carries the ordered ``steps`` log and the ``rejected`` verdicts so the
    caller can render "here is what matched and why each fell out" — the same
    information ``selection.select`` returns as a ``gap`` decision.
    """

    def __init__(self, call: Any, *, steps: tuple[str, ...] = (),
                 rejected: tuple[Any, ...] = ()) -> None:
        super().__init__(f"unresolvable: {call!r}")
        self.call = call
        self.steps = steps
        self.rejected = rejected


@dataclass(frozen=True, slots=True)
class Candidate:
    """A frozen per-candidate record for the NAME-resolution pipeline.

    ``namespace`` is the collision-qualified key ("unsloth~Qwen3-Coder-Next-GGUF")
    and ``name`` its bare tail — carrying both dissolves the requested-alias vs
    served-key ambiguity at the record level.  The remaining fields are the
    facts the name-pipeline stages read; each is derived once, at construction,
    from the existing registry/serving seams (see ``pipelines.py``).
    """
    namespace: str
    name: str
    blocked: bool = False
    starred: bool = False
    servable: bool = False        # concrete artifact on disk / loadable
    serving: bool = False         # a live worker holds it right now
    fits: bool = True             # predesignated to fit VRAM (w/ or w/o evict)
    success_rank: int = 0         # 0 = most-recent success; large = stale
    adjusted_use: int = 0         # n_calls - tie_driven_picks
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """The key a resolved candidate becomes — the collision-qualified one."""
        return self.namespace


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    kind: str                     # "eliminate" | "rank"
    fn: Callable[[Any, Any], Any]

    def __post_init__(self) -> None:
        if self.kind not in ("eliminate", "rank"):
            raise ValueError(f"stage {self.name!r}: kind must be "
                             f"'eliminate' or 'rank', got {self.kind!r}")


class StageRegistry:
    """A named collection of stages plus the decorator that fills it.

    Each pipeline owns its own registry so stage names never collide across
    pipelines and the PIPELINE tuple reads as a local table of precedence.
    """

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}

    def stage(self, name: str, kind: str) -> Callable[[Callable], Callable]:
        def _register(fn: Callable[[Any, Any], Any]) -> Callable:
            self._stages[name] = Stage(name, kind, fn)
            return fn
        return _register

    def get(self, name: str) -> Stage:
        try:
            return self._stages[name]
        except KeyError:
            raise KeyError(f"no stage registered under {name!r}; "
                           f"known: {sorted(self._stages)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._stages


def run_pipeline(
    call: Any,
    candidates: Iterable[Any],
    pipeline: tuple[str, ...],
    registry: StageRegistry,
    *,
    make_reject: Optional[Callable[[Any, str], Any]] = None,
) -> tuple[list[Any], list[Any], list[str]]:
    """Apply ``pipeline`` to ``candidates`` and return (kept, rejected, steps).

    ``kept`` is best-first (head is the winner).  ``rejected`` holds one record
    per eliminated candidate — built by ``make_reject(candidate, stage_name)``
    when given, else the raw candidate.  ``steps`` is the ordered human log.

    Ranking is a *stable* sort re-applied per ranker in pipeline order, so the
    final order reflects "last ranker dominates, earlier rankers break ties".
    Eliminate stages that empty the queue stop the pipeline early (there is
    nothing left to rank) and the emptying stage is named in the log.
    """
    queue: list[Any] = list(candidates)
    rejected: list[Any] = []
    steps: list[str] = []

    for name in pipeline:
        st = registry.get(name)
        if st.kind == "eliminate":
            kept: list[Any] = []
            dropped = 0
            for c in queue:
                if st.fn(c, call):
                    kept.append(c)
                else:
                    dropped += 1
                    rejected.append(make_reject(c, name) if make_reject else c)
            queue = kept
            steps.append(f"{name}: {len(queue)} kept"
                         + (f", {dropped} eliminated" if dropped else ""))
            if not queue:
                steps.append(f"-> unresolvable at {name}: no candidate left")
                return [], rejected, steps
        else:  # rank — stable sort, last ranker wins
            queue.sort(key=lambda c: st.fn(c, call))
            steps.append(f"{name}: ranked {[getattr(c, 'model_id', c) for c in queue]}")

    return queue, rejected, steps


# --------------------------------------------------------------------------- #
# name-resolution pipeline (Pipeline A)
# --------------------------------------------------------------------------- #

name_registry = StageRegistry()


def _tokens(value: str) -> list[str]:
    """Lower-case word tokens, folding ``_`` ``-`` ``~`` ``/`` and whitespace."""
    import re
    return [t for t in re.split(r"[\s_~/-]+", str(value).lower()) if t]


def _distance(name: str, want: str) -> int:
    """How far a candidate name is from the request: extra tokens it carries
    beyond what was asked (0 = exact token set).  A cheap, stable proxy for
    "closest" that does NOT prefer merely-shorter names — a candidate with the
    requested tokens plus one extra ranks ahead of one with three extra."""
    want_t = set(_tokens(want))
    cand_t = _tokens(name)
    return sum(1 for t in cand_t if t not in want_t)


@name_registry.stage("closest-string", "rank")
def _closest(c: Candidate, call: "NameCall") -> int:
    return _distance(c.name, call.requested)


@name_registry.stage("exact", "eliminate")
def _exact(c: Candidate, call: "NameCall") -> bool:
    # alloc.exact: keep only candidates matching the requested namespace/name.
    if not call.exact:
        return True
    req = call.requested.strip().lower()
    return c.namespace.lower() == req or c.name.lower() == req


@name_registry.stage("blocked", "eliminate")
def _not_blocked(c: Candidate, call: "NameCall") -> bool:
    return not c.blocked


@name_registry.stage("fit", "eliminate")
def _fits(c: Candidate, call: "NameCall") -> bool:
    return c.fits


@name_registry.stage("servable", "eliminate")
def _servable(c: Candidate, call: "NameCall") -> bool:
    # Only enforced when at least one candidate is servable — otherwise a
    # cold/never-installed registry would eliminate everything.  The registry
    # facts still let a truly-unservable lone candidate through to a later
    # error rather than pretending it is loadable.
    return c.servable if call.any_servable else True


@name_registry.stage("adjusted-usage", "rank")
def _usage(c: Candidate, call: "NameCall") -> int:
    return -c.adjusted_use          # higher use sorts first


@name_registry.stage("serving", "rank")
def _serving(c: Candidate, call: "NameCall") -> bool:
    return not c.serving            # False(0) sorts first -> loaded wins


@name_registry.stage("starred", "rank")
def _starred(c: Candidate, call: "NameCall") -> bool:
    return not c.starred            # starred sorts first (dominant, last)


NAME_PIPELINE: tuple[str, ...] = (
    "closest-string",   # weakest ranker (pure tiebreaker)
    "exact",            # eliminate: alloc.exact pin
    "blocked",          # eliminate
    "fit",              # eliminate
    "servable",         # eliminate
    "adjusted-usage",   # rank
    "serving",          # rank
    "starred",          # strongest ranker (last wins)
)


class NameCall:
    """The request side of a name resolution."""

    __slots__ = ("requested", "exact", "any_servable")

    def __init__(self, requested: str, *, exact: bool = False,
                 any_servable: bool = False) -> None:
        self.requested = str(requested)
        self.exact = bool(exact)
        self.any_servable = bool(any_servable)


def resolve_name(requested: str, candidates: list[Candidate], *,
                 exact: bool = False) -> tuple[list[Candidate], list[str]]:
    """Run the name pipeline; return (ranked candidates best-first, steps log).

    Empty ranked list means "eliminated to nothing" (caller fails with the
    candidate list).  A tie the rankers cannot break leaves >1 at distance 0 —
    the caller treats a non-unique head as ambiguous.
    """
    call = NameCall(requested, exact=exact,
                    any_servable=any(c.servable for c in candidates))
    kept, _rej, steps = run_pipeline(call, candidates, NAME_PIPELINE, name_registry)
    return kept, steps
