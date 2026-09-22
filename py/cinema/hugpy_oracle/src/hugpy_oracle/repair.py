"""Oracle repair controller (k90c): ONE bounded retry decision per failing card.

The movie precedent (``runners/movie.py`` attempt loop) retries a weak take
with a bumped seed, at most ``max_attempts_per_segment`` times. The oracle
generalizes that to a POLICY over the Scorecard's repair code — a decision
object first (``attempt_repair``: pure, no side effects), then at most ONE
execution (``execute_repair``), after which the route re-scores and answers
honestly whatever the second card says. Never a loop.

Policy (deliverable 2):

  WORKER_UNAVAILABLE / TIMEOUT   -> retry_next_model (next eligible model from
                                    the catalog's set on the route; none left
                                    -> action "none", honestly)
  EMPTY_OUTPUT / DECODE_FAILED   -> retry_same, once
  INTENT_MISMATCH on image.generate -> reseed (deterministic bumped seed —
                                    movie's seed-bump idiom, derived from the
                                    prompt because /oracle/route carries no
                                    base seed today)
  anything else                  -> none (no bounded repair defined; the card's
                                    diagnosis stands)

Both attempts' receipts are kept by the route (a ``receipts`` list on the
response); the repaired receipt is annotated via ``warnings`` so a repaired
run names itself on the existing contract shape — no contract change needed.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from typing import Any

from hugpy_oracle.contracts import ExecutionReceipt, GoalSpec, RepairCode, Scorecard
from hugpy_oracle.router import RouteDecision

# The closed action vocabulary — plain strings on the wire (k90d reads these).
ACTIONS = ("none", "retry_same", "retry_next_model", "reseed")

_RETRY_NEXT = (RepairCode.WORKER_UNAVAILABLE, RepairCode.TIMEOUT)
_RETRY_SAME = (RepairCode.EMPTY_OUTPUT, RepairCode.DECODE_FAILED)


@dataclass(frozen=True, slots=True)
class RepairDecision:
    """What the ONE bounded repair attempt should do, and why. ``model_id`` is
    set only for retry_next_model; ``seed`` only for reseed."""
    action: str
    rationale: str
    model_id: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"RepairDecision.action must be one of {ACTIONS}, "
                             f"got {self.action!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "rationale": self.rationale,
                "model_id": self.model_id, "seed": self.seed}


def next_eligible_model(route: RouteDecision) -> str | None:
    """The next model from the capability's eligible set (the k90a catalog's
    ordering, as recorded on the route) that is not the one that just failed."""
    for model_id in route.model_ids:
        if model_id != route.model_id:
            return model_id
    return None


def bumped_seed(goal: GoalSpec) -> int:
    """Deterministic reseed. /oracle/route carries no base seed today, so the
    bump is derived from the prompt (stable across retries of the same goal,
    different from the dispatch default of "no seed")."""
    return (zlib.crc32(goal.raw_prompt.encode("utf-8")) + 1) % (2 ** 31)


def attempt_repair(goal: GoalSpec, route: RouteDecision,
                   scorecard: Scorecard) -> RepairDecision:
    """The pure policy: failing card + repair code -> one bounded decision.
    Never executes anything."""
    if scorecard.hard_pass:
        return RepairDecision("none", "scorecard passed — nothing to repair")
    code = scorecard.repair_code
    if code is None:
        return RepairDecision(
            "none", "failing card carries no repair code (diagnosis only)")

    if code in _RETRY_NEXT:
        alt = next_eligible_model(route)
        if alt is None:
            return RepairDecision(
                "none",
                f"{code.value}: no eligible alternative model for "
                f"{route.capability} (eligible set: {list(route.model_ids)})")
        return RepairDecision(
            "retry_next_model",
            f"{code.value}: retry once on the next eligible model {alt!r}",
            model_id=alt)

    if code in _RETRY_SAME:
        return RepairDecision(
            "retry_same", f"{code.value}: retry the same route once")

    if code is RepairCode.INTENT_MISMATCH and route.capability == "image.generate":
        seed = bumped_seed(goal)
        return RepairDecision(
            "reseed",
            f"intent_mismatch on image.generate: regenerate once with bumped "
            f"seed {seed} (movie judge idiom)",
            seed=seed)

    return RepairDecision(
        "none", f"no bounded repair defined for {code.value} on {route.capability}")


def execute_repair(goal: GoalSpec, route: RouteDecision,
                   decision: RepairDecision,
                   ) -> tuple[list[dict[str, Any]], ExecutionReceipt, RouteDecision]:
    """Run the ONE repair attempt through the normal runtime and return
    (artifacts, receipt, route-actually-used). The receipt is annotated so a
    repaired run names itself on the existing contract shape."""
    from hugpy_oracle import runtime

    if decision.action == "none":
        raise ValueError("execute_repair called on a 'none' decision")

    repair_route = route
    overrides: dict[str, Any] | None = None
    if decision.action == "retry_next_model":
        # Placement was resolved for the failed model; "auto" hands the new
        # model to the DelegatingRunner's default placement.
        repair_route = replace(route, model_id=decision.model_id,
                               model_rationale="repair:next-eligible",
                               placement="auto")
    elif decision.action == "reseed":
        overrides = {"seed": decision.seed}

    artifacts, receipt = runtime.execute_route(goal, repair_route,
                                               overrides=overrides)
    receipt = replace(receipt, warnings=receipt.warnings + (
        f"repair attempt ({decision.action}): {decision.rationale}",))
    return artifacts, receipt, repair_route


def annotate_repaired(scorecard: Scorecard, decision: RepairDecision) -> Scorecard:
    """Note on the SECOND card that it is the post-repair verdict — pass or
    fail, the reader must see one bounded repair already happened."""
    note = f"after one bounded repair ({decision.action}: {decision.rationale})"
    diagnosis = f"{scorecard.diagnosis}; {note}" if scorecard.diagnosis else note
    return replace(scorecard, diagnosis=diagnosis)


__all__ = ["ACTIONS", "RepairDecision", "attempt_repair", "execute_repair",
           "annotate_repaired", "next_eligible_model", "bumped_seed"]
