"""k109 — the per-operation ROUTING MATRIX derived from benchmark rows.

The doc's instruction is blunt: *"Do not assume that one model will be best for
the entire pipeline. Produce a routing matrix identifying the best primary
model and fallback model for each artifact or operation."* This module is that
matrix — a pure function from result rows to a primary + fallback per
operation, with the evidence attached to every choice.

PURE ON PURPOSE. Nothing here dispatches, routes, reads a GPU or imports the
catalog. It takes the rows ``benchmark.py`` writes (plain dicts, exactly as
they land in ``attempts.jsonl``) and produces a serializable matrix. That means
the matrix can be re-derived from an old run dir, on a laptop, with the fleet
switched off, and it will come out identical.

THE ORDERING RULE, written down once so a leaderboard cannot mean something
different from the JSON::

    rank by  (1) success rate DESC   — a model that does not produce a valid
                                       artifact cannot be a route, however
                                       beautiful its prose
             (2) quality      DESC   — mean(deterministic, judge) when the
                                       judge was available, else deterministic
             (3) latency      ASC    — the tie-break, never a promoter

    primary  = rank 1, and ONLY if its success rate is > 0
    fallback = rank 2 with success rate > 0, and never the primary

A model with zero successes is still listed in ``candidates`` (with its
failures) and is never selected. An operation where NOBODY succeeded gets
``primary=None`` and a reason — an empty answer is a finding, and a matrix that
invented a route out of six failures would be worse than no matrix.

QUALITY AND PERFORMANCE STAY SEPARATE UNTIL THE LAST LINE. Every candidate
carries both sets of numbers unmixed. The composite (:data:`FORMULA_NOTE`) is
computed ONLY for the leaderboard's last column and is printed with its formula
directly above it, because a single number whose formula is not on the page is
the fastest way to lose an argument about model choice six weeks later.

CONSUMPTION. :func:`best_route` is the read API a router or the catalog CAN
consume later::

    from hugpy_oracle.routing_matrix import best_route
    choice = best_route("screenplay.complete")      # or (..., path=...)
    if choice:  choice.primary, choice.fallback, choice.evidence

Wiring that into the live router is deliberately NOT done here (k109 does not
edit ``catalog.py`` or ``router.py``): the export shape below is the contract a
follow-up task binds to.

No pathlib anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The composite formula, printed next to every number it produces.
COMPOSITE_QUALITY_WEIGHT: float = 0.70
COMPOSITE_SPEED_WEIGHT: float = 0.30
#: The HALF-MARK latency: a model that averages this scores 50 on the speed
#: axis. The curve is ``ref / (ref + latency)`` rather than a capped ratio so
#: it discriminates at every scale — with a cap, a cohort that all answers in
#: under the reference scores 100 across the board and the axis says nothing.
REFERENCE_LATENCY_S: float = 30.0

FORMULA_NOTE: str = (
    "quality = mean(deterministic_score, judge_score) when a judge was "
    "available, else deterministic_score. "
    f"speed = 100 * {REFERENCE_LATENCY_S:g}s / ({REFERENCE_LATENCY_S:g}s + "
    f"mean_latency_s) — {REFERENCE_LATENCY_S:g}s scores 50. "
    f"composite = {COMPOSITE_QUALITY_WEIGHT:g}*quality + "
    f"{COMPOSITE_SPEED_WEIGHT:g}*speed. "
    "The composite is DERIVED AFTER the fact and is never used for ranking — "
    "ranking is (success rate, quality, latency).")

#: The env var a consumer may point at a serialized matrix.
MATRIX_PATH_ENV: str = "ORACLE_ROUTING_MATRIX"

SCHEMA_VERSION: str = "oracle-routing-matrix/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.fmean(clean), 3) if clean else None


def _stdev(values: Sequence[float]) -> float | None:
    """Population-free sample stdev, or None when one sample cannot have one.

    Reported for every axis because ``--repeats N`` exists precisely to make it
    computable: a model whose score swings 30 points between two identical runs
    has not earned a routing slot on its best run."""
    clean = [float(v) for v in values if v is not None]
    return round(statistics.stdev(clean), 3) if len(clean) > 1 else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (operation, model) statistics over attempt rows.

    One dict per pair, sorted by operation then model, so two runs of the same
    sweep produce byte-identical summaries when the numbers agree."""
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("operation") or ""), str(row.get("model") or ""))
        buckets.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (operation, model), items in sorted(buckets.items()):
        det = [(r.get("deterministic") or {}).get("score") for r in items]
        judge_scores = [(r.get("judge") or {}).get("score") for r in items
                        if (r.get("judge") or {}).get("available")]
        latency = [(r.get("perf") or {}).get("latency_s") for r in items]
        toks = [(r.get("perf") or {}).get("tokens_per_s") for r in items]
        vram = [(r.get("perf") or {}).get("vram_used_delta_bytes")
                for r in items]
        oks = [bool(r.get("ok")) for r in items]
        failures = [str(r.get("failure")) for r in items if r.get("failure")]
        timeouts = sum(1 for f in failures if f == "timeout")
        preservation = [(r.get("deterministic") or {}).get("preservation")
                        for r in items]
        contradiction = [(r.get("deterministic") or {}).get("contradiction_rate")
                         for r in items]
        completeness = [(r.get("deterministic") or {}).get("completeness")
                        for r in items]
        constraints = [(r.get("deterministic") or {}).get("constraint_adherence")
                       for r in items]
        accuracy = [(r.get("deterministic") or {}).get("accuracy")
                    for r in items]
        out.append({
            "operation": operation, "model": model, "attempts": len(items),
            "ok": sum(1 for v in oks if v),
            "ok_rate": round(sum(1 for v in oks if v) / len(items), 4),
            "failure_rate": round(len(failures) / len(items), 4),
            "timeouts": timeouts,
            "failures": sorted(set(failures))[:4],
            "deterministic_mean": _mean(det), "deterministic_stdev": _stdev(det),
            "judge_mean": _mean(judge_scores),
            "judge_stdev": _stdev(judge_scores),
            "judged_attempts": len(judge_scores),
            "judge_models": sorted({(r.get("judge") or {}).get("judge_model")
                                    for r in items
                                    if (r.get("judge") or {}).get("judge_model")}),
            "latency_mean_s": _mean(latency), "latency_stdev_s": _stdev(latency),
            "tokens_per_s_mean": _mean(toks),
            "vram_used_delta_bytes_max": max(
                (int(v) for v in vram if isinstance(v, (int, float))),
                default=None),
            "preservation_mean": _mean(preservation),
            "contradiction_rate_mean": _mean(contradiction),
            "completeness_mean": _mean(completeness),
            "constraint_adherence_mean": _mean(constraints),
            "accuracy_mean": _mean(accuracy),
            "cases": sorted({str(r.get("case_id")) for r in items}),
            "modes": sorted({str(r.get("mode")) for r in items}),
        })
    return out


def quality_of(stat: Mapping[str, Any]) -> float:
    """The quality number the RANKING uses: the two layers averaged when both
    exist, the deterministic one alone when the judge was unavailable. An
    unjudged model is never punished for the fleet's judge being down."""
    det = stat.get("deterministic_mean")
    judged = stat.get("judge_mean")
    if det is None and judged is None:
        return 0.0
    if judged is None:
        return float(det)
    if det is None:
        return float(judged)
    return round((float(det) + float(judged)) / 2.0, 3)


def speed_of(stat: Mapping[str, Any]) -> float | None:
    """The speed axis, 0-100, on the half-mark curve (see
    :data:`REFERENCE_LATENCY_S`). None when nothing was timed."""
    latency = stat.get("latency_mean_s")
    if latency is None or float(latency) < 0:
        return None
    return round(100.0 * REFERENCE_LATENCY_S
                 / (REFERENCE_LATENCY_S + float(latency)), 3)


def composite_of(stat: Mapping[str, Any]) -> float | None:
    """The composite — derived AFTER quality and performance were reported
    separately, per the doc. None when there is no latency to speak of."""
    speed = speed_of(stat)
    if speed is None:
        return None
    return round(COMPOSITE_QUALITY_WEIGHT * quality_of(stat)
                 + COMPOSITE_SPEED_WEIGHT * speed, 3)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model's evidence for one operation."""
    model: str
    ok_rate: float = 0.0
    quality: float = 0.0
    deterministic: float | None = None
    judge: float | None = None
    latency_s: float | None = None
    attempts: int = 0
    failure_rate: float = 0.0
    timeouts: int = 0
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def routable(self) -> bool:
        return self.ok_rate > 0.0

    @property
    def rank_key(self) -> tuple[float, float, float, str]:
        """The documented ordering, as a sort key (negated for DESC axes)."""
        latency = self.latency_s if self.latency_s is not None else float("inf")
        return (-self.ok_rate, -self.quality, latency, self.model)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "ok_rate": self.ok_rate,
                "quality": self.quality, "deterministic": self.deterministic,
                "judge": self.judge, "latency_s": self.latency_s,
                "attempts": self.attempts, "failure_rate": self.failure_rate,
                "timeouts": self.timeouts, "routable": self.routable,
                "composite": composite_of(self.stats) if self.stats else None,
                "stats": dict(self.stats)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Candidate":
        return cls(model=d["model"], ok_rate=float(d.get("ok_rate", 0.0)),
                   quality=float(d.get("quality", 0.0)),
                   deterministic=d.get("deterministic"), judge=d.get("judge"),
                   latency_s=d.get("latency_s"),
                   attempts=int(d.get("attempts", 0)),
                   failure_rate=float(d.get("failure_rate", 0.0)),
                   timeouts=int(d.get("timeouts", 0)),
                   stats=dict(d.get("stats") or {}))


@dataclass(frozen=True, slots=True)
class RouteEntry:
    """One operation's route: primary, fallback, and why."""
    operation: str
    primary: str | None = None
    fallback: str | None = None
    candidates: tuple[Candidate, ...] = ()
    note: str = ""

    def evidence_for(self, model: str | None) -> Candidate | None:
        return next((c for c in self.candidates if c.model == model), None)

    @property
    def evidence(self) -> dict[str, Any]:
        return {"primary": (self.evidence_for(self.primary).to_dict()
                            if self.evidence_for(self.primary) else None),
                "fallback": (self.evidence_for(self.fallback).to_dict()
                             if self.evidence_for(self.fallback) else None)}

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "primary": self.primary,
                "fallback": self.fallback, "note": self.note,
                "evidence": self.evidence,
                "candidates": [c.to_dict() for c in self.candidates]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RouteEntry":
        return cls(operation=d["operation"], primary=d.get("primary"),
                   fallback=d.get("fallback"), note=d.get("note", ""),
                   candidates=tuple(Candidate.from_dict(c)
                                    for c in d.get("candidates", ())))


@dataclass(frozen=True, slots=True)
class RoutingMatrix:
    """The whole matrix, serializable and re-derivable."""
    entries: tuple[RouteEntry, ...] = ()
    registry_version: str | None = None
    mode: str = ""
    run_id: str = ""
    run_dir: str = ""
    generated_at: str = ""
    schema: str = SCHEMA_VERSION
    #: k109b. The stationary brief every row was measured under, and its
    #: digest. Empty on every k109 matrix, which is why they carry defaults:
    #: ``from_dict`` must keep reading a k109 file, and ``best_route`` must
    #: keep returning the same RouteChoice for one. A matrix with a
    #: scenario_version says "these routes were measured by asking every model
    #: the SAME question"; one without says nothing about comparability, and
    #: the difference belongs in the file rather than in a memory.
    scenario_version: str = ""
    scenario_digest: str = ""

    def entry(self, operation: str) -> RouteEntry | None:
        return next((e for e in self.entries if e.operation == operation), None)

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(e.operation for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "registry_version": self.registry_version,
                "mode": self.mode, "run_id": self.run_id,
                "run_dir": self.run_dir,
                "generated_at": self.generated_at or _utc_now(),
                "formula": FORMULA_NOTE,
                "scenario_version": self.scenario_version,
                "scenario_digest": self.scenario_digest,
                "entries": {e.operation: e.to_dict() for e in self.entries}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RoutingMatrix":
        raw = d.get("entries") or {}
        items = raw.values() if isinstance(raw, Mapping) else raw
        return cls(entries=tuple(RouteEntry.from_dict(e) for e in items),
                   registry_version=d.get("registry_version"),
                   mode=d.get("mode", ""), run_id=d.get("run_id", ""),
                   run_dir=d.get("run_dir", ""),
                   generated_at=d.get("generated_at", ""),
                   schema=d.get("schema", SCHEMA_VERSION),
                   scenario_version=d.get("scenario_version", ""),
                   scenario_digest=d.get("scenario_digest", ""))


def derive_matrix(rows: Iterable[Mapping[str, Any]], *,
                  registry_version: str | None = None,
                  mode: str = "", run_id: str = "", run_dir: str = "",
                  scenario_version: str = "", scenario_digest: str = ""
                  ) -> RoutingMatrix:
    """Rows -> matrix, by the ordering rule in the module docstring.

    ``scenario_version`` / ``scenario_digest`` (k109b, additive) stamp the
    stationary brief the rows were measured under. Omitted, the matrix is
    byte-identical to a k109 one apart from two empty strings, and
    :meth:`RoutingMatrix.from_dict` reads a file written without them."""
    stats = summarize(rows)
    by_operation: dict[str, list[Mapping[str, Any]]] = {}
    for stat in stats:
        by_operation.setdefault(stat["operation"], []).append(stat)

    entries: list[RouteEntry] = []
    for operation in sorted(by_operation):
        candidates = sorted(
            (Candidate(model=s["model"], ok_rate=float(s["ok_rate"]),
                       quality=quality_of(s),
                       deterministic=s.get("deterministic_mean"),
                       judge=s.get("judge_mean"),
                       latency_s=s.get("latency_mean_s"),
                       attempts=int(s["attempts"]),
                       failure_rate=float(s["failure_rate"]),
                       timeouts=int(s["timeouts"]), stats=s)
             for s in by_operation[operation]),
            key=lambda c: c.rank_key)
        routable = [c for c in candidates if c.routable]
        primary = routable[0].model if routable else None
        fallback = routable[1].model if len(routable) > 1 else None
        if primary is None:
            note = (f"no candidate produced a valid artifact for {operation} "
                    f"({len(candidates)} model(s) tried) — this operation has "
                    f"NO route on this fleet")
        elif fallback is None:
            note = (f"only one model produced a valid artifact for "
                    f"{operation}: there is no fallback, and a single-model "
                    f"route is a single point of failure")
        else:
            note = (f"primary {primary} then fallback {fallback}, ranked by "
                    f"(success rate, quality, latency)")
        entries.append(RouteEntry(operation=operation, primary=primary,
                                  fallback=fallback,
                                  candidates=tuple(candidates), note=note))

    return RoutingMatrix(entries=tuple(entries),
                         registry_version=registry_version, mode=mode,
                         run_id=run_id, run_dir=run_dir,
                         generated_at=_utc_now(),
                         scenario_version=scenario_version,
                         scenario_digest=scenario_digest)


# ---------------------------------------------------------------------------
# Serialization + the read API a router can consume
# ---------------------------------------------------------------------------


def save_matrix(matrix: RoutingMatrix, path: str) -> bool:
    """Write the matrix as JSON. False (never a raise) when the disk says no."""
    import tempfile
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(matrix.to_dict(), stream, indent=1, sort_keys=True,
                      default=str)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("routing matrix: could not write %s (%s: %s)", path,
                       type(exc).__name__, exc)
        return False


def load_matrix(path: str | None = None) -> RoutingMatrix | None:
    """Read a serialized matrix from ``path``, or from ``$ORACLE_ROUTING_
    MATRIX``. None when there is no matrix to read — a consumer must treat
    "no benchmark has run here" as a normal state, not an error."""
    if not path:
        from hugpy_oracle.config import matrix_path
        path = matrix_path()
    target = path or ""
    if not target or not os.path.isfile(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as stream:
            return RoutingMatrix.from_dict(json.load(stream))
    except Exception as exc:  # noqa: BLE001
        logger.warning("routing matrix: could not read %s (%s: %s)", target,
                       type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# k114's follow-up: find + verify the newest matrix, honestly
# ---------------------------------------------------------------------------
#
# The one function in this module that is NOT pure. Everything above takes
# rows or a path and returns a value; this one walks the filesystem and, by
# default, reads the LIVE catalog's registry_version. The import stays lazy
# and INSIDE the function (``screenplay.bind_llm`` and
# ``benchmark._registry_version`` use the same discipline) so the module
# itself remains importable with the fleet switched off — only calling this
# one function touches anything live.

#: Duplicated from ``benchmark.DEFAULT_RUN_ROOT`` / ``RUN_ROOT_ENV`` rather
#: than imported: ``benchmark.py`` imports ``.contracts`` and ``.screenplay``
#: at module scope, and pulling that in here would reintroduce the exact
#: import weight this module's docstring says it avoids. One recorded
#: literal, kept in sync by hand, is cheaper than that coupling — the same
#: call k114 made for the ``config.ts`` URL literal.
DEFAULT_RUN_ROOT: str = "/home/ubuntu/station/model-battery"
RUN_ROOT_ENV: str = "ORACLE_BENCHMARK_ROOT"


def _live_registry_version() -> str | None:
    """``catalog.registry_version()``, read lazily and never raised. None
    when the catalog cannot be read here — the caller decides what that
    means (it means "cannot verify", so :func:`load_latest_matrix` refuses
    to hand back a matrix rather than guess)."""
    try:
        from hugpy_oracle import catalog
        return catalog.registry_version()
    except Exception as exc:  # noqa: BLE001
        logger.info("routing matrix: live registry_version unavailable "
                   "(%s: %s)", type(exc).__name__, exc)
        return None


def load_latest_matrix(root: str | None = None, *,
                       live_registry_version: Callable[[], str | None] | None = None,
                       ) -> tuple[RoutingMatrix | None, str]:
    """Find the newest ``oracle-*`` run dir under ``root`` (default the
    battery root, or ``$ORACLE_BENCHMARK_ROOT``) that carries a serialized
    ``routing_matrix.json``, load it, and verify its ``registry_version``
    against the LIVE catalog before handing it back.

    Returns ``(matrix, reason)``. ``matrix`` is ``None`` — NEVER a stale one —
    whenever: the run root does not exist or cannot be listed, no run dir
    carries a matrix file, the newest matrix file will not parse, the live
    registry_version cannot be read, OR the matrix's ``registry_version``
    does not match the live one. ``reason`` is always a human-readable
    sentence, on the success branch as well as every failure branch, because
    a caller that shows an operator a route (or a fallback) needs the WHY
    either way — this is the same "a value, not silence" shape as
    :class:`~hugpy_oracle.screenplay.AuthoringGap`.

    "Newest" is by the matrix FILE's own mtime, not the run dir's name, so a
    matrix re-derived offline into an older run dir (``--from-run``, per
    k109's operator commands) is still found.

    ``live_registry_version`` is the injection seam a test uses — a callable
    returning a fixed string, or one that returns ``None`` to exercise the
    "cannot verify" branch — so this is exercisable with no catalog, no
    fleet and no GPU, the same discipline every other live seam in this
    package uses."""
    if not root:
        from hugpy_oracle.config import benchmark_root
        root = benchmark_root()
    base = root
    try:
        names = sorted(entry for entry in os.listdir(base)
                       if entry.startswith("oracle-")
                       and os.path.isdir(os.path.join(base, entry)))
    except OSError as exc:
        return None, (f"could not list run root {base!r}: "
                      f"{type(exc).__name__}: {exc}")

    candidates: list[tuple[float, str, str]] = []
    for name in names:
        path = os.path.join(base, name, "routing_matrix.json")
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        candidates.append((mtime, name, path))
    if not candidates:
        return None, (f"no oracle-* run dir under {base!r} carries a "
                      f"routing_matrix.json — no benchmark has been derived "
                      f"here yet")
    candidates.sort()
    _, latest_name, latest_path = candidates[-1]

    matrix = load_matrix(latest_path)
    if matrix is None:
        return None, f"{latest_path} did not parse as a routing matrix"

    live_version = (live_registry_version or _live_registry_version)()
    if live_version is None:
        return None, (
            f"found {latest_name}'s matrix (registry_version "
            f"{matrix.registry_version!r}) but could not read the live "
            f"catalog's registry_version to verify it — never honouring a "
            f"route that cannot be checked")
    if matrix.registry_version != live_version:
        return None, (
            f"found {latest_name}'s matrix but its registry_version "
            f"{matrix.registry_version!r} does not match the live fleet's "
            f"{live_version!r} — a matrix measured against a different "
            f"catalog snapshot describes a fleet that no longer exists, so "
            f"it is not honoured")
    return matrix, (f"loaded {latest_name}'s routing matrix, "
                    f"registry_version {live_version!r} verified live")


@dataclass(frozen=True, slots=True)
class RouteChoice:
    """What :func:`best_route` hands a consumer.

    EXPORT SHAPE (the contract a router/catalog binds to later)::

        RouteChoice(
            operation="screenplay.complete",
            primary="qwen3-14b",          # model id, or None
            fallback="mistral-nemo",      # model id, or None
            evidence={"primary": {...}, "fallback": {...}},
            registry_version="…",         # the snapshot the run was measured on
            mode="normalized",            # which benchmark mode produced it
            run_id="oracle-20260821-0130",
            note="…"                      # why, in one sentence
        )

    ``registry_version`` is the load-bearing field for a consumer: a matrix
    measured against a DIFFERENT catalog snapshot describes a fleet that no
    longer exists, and the consumer — not this module — decides whether to
    honour it or fall back to the catalog's default pick."""
    operation: str
    primary: str | None
    fallback: str | None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    registry_version: str | None = None
    mode: str = ""
    run_id: str = ""
    note: str = ""
    #: k109b, additive and defaulted: a consumer that only knows k109's fields
    #: keeps working, and one that wants to know whether these routes were
    #: measured under a stationary brief can ask.
    scenario_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "primary": self.primary,
                "fallback": self.fallback, "evidence": dict(self.evidence),
                "registry_version": self.registry_version, "mode": self.mode,
                "run_id": self.run_id, "note": self.note,
                "scenario_version": self.scenario_version}


def best_route(operation: str, matrix: RoutingMatrix | None = None,
               path: str | None = None) -> RouteChoice | None:
    """The read API: primary + fallback + evidence for one operation.

    Returns None when no matrix is available or the operation was never
    benchmarked — the caller keeps whatever it was doing before, which is the
    only safe default for a routing hint."""
    resolved = matrix if matrix is not None else load_matrix(path)
    if resolved is None:
        return None
    entry = resolved.entry(operation)
    if entry is None:
        return None
    return RouteChoice(operation=entry.operation, primary=entry.primary,
                       fallback=entry.fallback, evidence=entry.evidence,
                       registry_version=resolved.registry_version,
                       mode=resolved.mode, run_id=resolved.run_id,
                       note=entry.note,
                       scenario_version=resolved.scenario_version)


# ---------------------------------------------------------------------------
# The human-readable leaderboard
# ---------------------------------------------------------------------------


def _cell(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) / (1024 ** 3):+.2f}"


def render_leaderboard(matrix: RoutingMatrix,
                       rows: Iterable[Mapping[str, Any]] | None = None) -> str:
    """The markdown report: the matrix first, then QUALITY and PERFORMANCE in
    separate tables, then — last, and only then — the composite with its
    formula printed directly above it."""
    stats = summarize(rows) if rows is not None else [
        dict(c.stats) for e in matrix.entries for c in e.candidates if c.stats]
    lines: list[str] = [
        f"# Oracle model evaluation — {matrix.run_id or 'unnamed run'}",
        "",
        f"* mode: `{matrix.mode or 'unknown'}`",
        f"* registry_version: `{matrix.registry_version or 'unrecorded'}`",
        f"* generated: {matrix.generated_at or _utc_now()}",
        f"* run dir: `{matrix.run_dir or 'n/a'}`",
        "",
        "## Routing matrix (primary + fallback per operation)",
        "",
        "| operation | primary | fallback | primary ok | primary quality | "
        "primary latency | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in matrix.entries:
        best = entry.evidence_for(entry.primary)
        lines.append(
            f"| `{entry.operation}` | "
            f"{('`' + entry.primary + '`') if entry.primary else '**none**'} | "
            f"{('`' + entry.fallback + '`') if entry.fallback else '—'} | "
            f"{_cell((best.ok_rate * 100) if best else None, 0, '%')} | "
            f"{_cell(best.quality if best else None)} | "
            f"{_cell(best.latency_s if best else None, 1, 's')} | "
            f"{entry.note} |")

    lines += ["", "## Quality (no performance in this table)", "",
              "_`±` is the spread (sample stdev) across this model's attempts "
              "for the operation — cases AND repeats, so a wide spread may "
              "mean an inconsistent model or simply a hard case in the mix._",
              "",
              "| operation | model | ok | det | ±det | judge | ±judge | "
              "preserved | contradiction | complete | constraints | accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for stat in stats:
        lines.append(
            f"| `{stat['operation']}` | `{stat['model']}` | "
            f"{stat['ok']}/{stat['attempts']} | "
            f"{_cell(stat['deterministic_mean'])} | "
            f"{_cell(stat['deterministic_stdev'])} | "
            f"{_cell(stat['judge_mean'])} | {_cell(stat['judge_stdev'])} | "
            f"{_cell(stat['preservation_mean'], 2)} | "
            f"{_cell(stat['contradiction_rate_mean'], 2)} | "
            f"{_cell(stat['completeness_mean'], 2)} | "
            f"{_cell(stat['constraint_adherence_mean'], 2)} | "
            f"{_cell(stat['accuracy_mean'], 2)} |")

    lines += ["", "## Performance (no quality in this table)", "",
              "| operation | model | attempts | latency | ±latency | tok/s | "
              "VRAM Δ GiB | timeouts | failure rate |",
              "|---|---|---|---|---|---|---|---|---|"]
    for stat in stats:
        lines.append(
            f"| `{stat['operation']}` | `{stat['model']}` | "
            f"{stat['attempts']} | {_cell(stat['latency_mean_s'], 1, 's')} | "
            f"{_cell(stat['latency_stdev_s'], 1, 's')} | "
            f"{_cell(stat['tokens_per_s_mean'])} | "
            f"{_gib(stat['vram_used_delta_bytes_max'])} | "
            f"{stat['timeouts']} | {_cell(stat['failure_rate'] * 100, 0, '%')} |")

    lines += ["", "## Composite (derived AFTER the two tables above)", "",
              f"> {FORMULA_NOTE}", "",
              "| operation | model | quality | speed | composite |",
              "|---|---|---|---|---|"]
    for stat in sorted(stats, key=lambda s: (s["operation"],
                                             -(composite_of(s) or 0.0))):
        lines.append(
            f"| `{stat['operation']}` | `{stat['model']}` | "
            f"{_cell(quality_of(stat))} | {_cell(speed_of(stat))} | "
            f"{_cell(composite_of(stat))} |")

    if not stats:
        lines += ["", "_No attempts were recorded — see `run.log` and "
                  "`environment.json` for why._"]
    lines.append("")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# k109b — the per-POINT view: a lifecycle grid, not an operation ranking
# ---------------------------------------------------------------------------
#
# Everything above ranks MODELS within an OPERATION, which is what a router
# needs. k109b asks the operator's question instead — "which model is capable
# of each point, so they can all contribute their strengths" — and that needs
# the transpose: for each of the sixteen lifecycle steps, who can do it, who
# half can, and where the fleet simply cannot.
#
# These functions are PURE (rows in, markdown/dicts out) exactly like the rest
# of this module, so a report can be re-rendered from an old ``cells.jsonl``
# with no fleet, no GPU and no benchmark import at all.

#: Verdicts, worst to best, for a stable grid sort.
VERDICT_ORDER: tuple[str, ...] = ("NO_CANDIDATES", "incapable", "refused",
                                  "partial", "capable")

#: The one-character grid cell per verdict. A grid of 129 models x 19 points
#: is unreadable as words and perfectly readable as marks.
VERDICT_MARK: dict[str, str] = {
    "capable": "**Y**", "partial": "~", "refused": "R", "incapable": ".",
    "NO_CANDIDATES": "—",
}


def summarize_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-lifecycle-point statistics: the verdict tally and the winner.

    One dict per point, sorted by lifecycle step then point id, so two runs of
    the same sweep produce byte-identical summaries when the numbers agree.
    ``NO_CANDIDATES`` rows are counted and kept — a point with nothing but a
    NO_CANDIDATES row is the most important row in the report and dropping it
    for having no model would delete the finding."""
    buckets: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row.get("step") or 0), str(row.get("point_id") or ""))
        buckets.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (step, point_id), items in sorted(buckets.items()):
        tally = {v: 0 for v in VERDICT_ORDER}
        for row in items:
            verdict = str(row.get("verdict") or "incapable")
            tally[verdict] = tally.get(verdict, 0) + 1
        measured = [r for r in items if r.get("verdict") != "NO_CANDIDATES"]
        capable = [r for r in measured if r.get("verdict") == "capable"]
        ranked = sorted(
            capable,
            key=lambda r: (-(quality_of(_row_stat(r))),
                           (r.get("perf") or {}).get("latency_s") or float("inf"),
                           str(r.get("model"))))
        out.append({
            "step": step, "point_id": point_id,
            "operations": sorted({str(r.get("operation")) for r in items}),
            "capability": next((str(r.get("capability")) for r in items
                                if r.get("capability")), ""),
            "stage": next((str(r.get("stage")) for r in items
                           if r.get("stage")), ""),
            "candidates": len(measured), "verdicts": tally,
            "capable": [str(r.get("model")) for r in ranked],
            "winner": str(ranked[0].get("model")) if ranked else None,
            "runner_up": str(ranked[1].get("model")) if len(ranked) > 1 else None,
            "no_candidates": tally.get("NO_CANDIDATES", 0) > 0,
            "notes": sorted({str(r.get("note")) for r in items
                             if r.get("verdict") == "NO_CANDIDATES"
                             and r.get("note")}),
        })
    return out


def _row_stat(row: Mapping[str, Any]) -> dict[str, Any]:
    """One row in the shape :func:`quality_of` reads. A single row is a
    one-sample summary, which is exactly what the per-point ranking needs — the
    stationary sweep asks each model each question ONCE, on purpose, so that
    every cell in the grid cost the fleet the same amount."""
    det = (row.get("deterministic") or {}).get("score")
    judge = row.get("judge") or {}
    return {"deterministic_mean": det,
            "judge_mean": judge.get("score") if judge.get("available") else None}


def render_capability_grid(rows: Iterable[Mapping[str, Any]],
                           points: Sequence[Mapping[str, Any]],
                           *, title: str = "capability grid",
                           scenario_version: str = "",
                           scenario_digest: str = "",
                           registry_version: str | None = None) -> str:
    """models x lifecycle points, one mark per cell — the deliverable grid.

    Models appear as ROWS and points as COLUMNS because there are far more
    models than points and a table is easier to read tall than wide. A model is
    listed only for the points its capability class was swept at; a point it
    was never a candidate for is blank, which is NOT the same mark as a point
    it failed, and the legend says so."""
    rows = list(rows)
    ordered = sorted(points, key=lambda p: (int(p.get("step") or 0),
                                            str(p.get("point_id"))))
    columns = [str(p.get("point_id")) for p in ordered]
    by_model: dict[str, dict[str, str]] = {}
    for row in rows:
        model = str(row.get("model") or "")
        if model in ("", "(none)"):
            continue
        by_model.setdefault(model, {})[str(row.get("point_id"))] = \
            str(row.get("verdict") or "incapable")

    def rank(model: str) -> tuple:
        marks = by_model[model]
        capable = sum(1 for v in marks.values() if v == "capable")
        partial = sum(1 for v in marks.values() if v == "partial")
        return (-capable, -partial, model)

    lines: list[str] = [
        f"# {title}", "",
        f"* scenario: `{scenario_version or 'unversioned'}` "
        f"`{scenario_digest or ''}`",
        f"* registry_version: `{registry_version or 'unrecorded'}`",
        f"* generated: {_utc_now()}", "",
        "Every model at every point was asked a question derived from ONE "
        "brief — that is what makes two cells in this grid comparable.", "",
        "**Legend** — `**Y**` capable (the artifact validated) · `~` partial "
        "(structured output, validator refused it) · `R` refused (the model "
        "declined) · `.` incapable (nothing usable came back) · `—` "
        "NO_CANDIDATES (no model on this fleet serves the point) · blank "
        "(this model was not a candidate for this point).", "",
        "## Points, in lifecycle order", "",
        "| # | point | step | kind | capability | candidates |",
        "|---|---|---|---|---|---|",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        pid = str(row.get("point_id"))
        if row.get("verdict") != "NO_CANDIDATES":
            counts[pid] = counts.get(pid, 0) + 1
    for index, point in enumerate(ordered, start=1):
        pid = str(point.get("point_id"))
        cap = point.get("capability") or "/".join(
            point.get("missing_capability") or ()) or "—"
        lines.append(
            f"| {index} | `{pid}` | {point.get('step')} | "
            f"{point.get('kind')} | `{cap}` | {counts.get(pid, 0)} |")

    lines += ["", "## The grid", "",
              "| model | " + " | ".join(str(i) for i in
                                        range(1, len(columns) + 1)) +
              " | capable | partial |",
              "|---" * (len(columns) + 3) + "|"]
    for model in sorted(by_model, key=rank):
        marks = by_model[model]
        cells = [VERDICT_MARK.get(marks.get(c, ""), "") for c in columns]
        capable = sum(1 for v in marks.values() if v == "capable")
        partial = sum(1 for v in marks.values() if v == "partial")
        lines.append(f"| `{model}` | " + " | ".join(cells) +
                     f" | {capable} | {partial} |")
    if not by_model:
        lines.append("| _no model produced a row_ | " +
                     " | ".join("" for _ in columns) + " |  |  |")

    gaps = [p for p in ordered if str(p.get("kind")) == "gap"]
    pipeline = [p for p in ordered if str(p.get("kind")) == "pipeline"]
    lines += ["", "## Points with NO candidate on this fleet", "",
              "_The gap IS the data. Each row names what is missing._", ""]
    if gaps:
        lines += ["| step | point | missing capability | why |",
                  "|---|---|---|---|"]
        for point in gaps:
            lines.append(
                f"| {point.get('step')} | `{point.get('point_id')}` | "
                f"`{', '.join(point.get('missing_capability') or ())}` | "
                f"{str(point.get('note') or '').replace('|', '/')} |")
    else:
        lines.append("_None: every lifecycle point has at least one "
                     "candidate._")

    lines += ["", "## Model-free pipeline steps", "",
              "_These carry a `NO_CANDIDATES` row too, but they are NOT gaps: "
              "no model is supposed to serve them. They are executed by this "
              "codebase._", ""]
    if pipeline:
        lines += ["| step | point | executed by |", "|---|---|---|"]
        for point in pipeline:
            lines.append(
                f"| {point.get('step')} | `{point.get('point_id')}` | "
                f"{str(point.get('note') or '').replace('|', '/')} |")
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def render_point_leaderboards(rows: Iterable[Mapping[str, Any]],
                              points: Sequence[Mapping[str, Any]],
                              *, title: str = "per-point leaderboards",
                              scenario_version: str = "",
                              limit: int | None = 12) -> str:
    """One leaderboard per lifecycle point, in lifecycle order.

    ``limit`` caps each table (the text roster is 88 models long and the
    twelfth-best plot constructor is not a decision anybody makes), and the
    cap is STATED in the heading rather than applied silently — the full
    ordering is always in ``cells.jsonl``."""
    rows = list(rows)
    stats = {s["point_id"]: s for s in summarize_points(rows)}
    ordered = sorted(points, key=lambda p: (int(p.get("step") or 0),
                                            str(p.get("point_id"))))
    lines: list[str] = [
        f"# {title}", "",
        f"* scenario: `{scenario_version or 'unversioned'}`",
        f"* generated: {_utc_now()}", "",
        "One brief, every model, one attempt each. `det` is the deterministic "
        "score (validators only), `judge` the independent rubric score where "
        "one was available.", ""]

    for point in ordered:
        pid = str(point.get("point_id"))
        stat = stats.get(pid) or {}
        cap = point.get("capability") or "/".join(
            point.get("missing_capability") or ()) or "—"
        lines += [f"## Step {point.get('step')} — {point.get('name')}", "",
                  f"* point: `{pid}` · kind: `{point.get('kind')}` · "
                  f"capability: `{cap}`",
                  f"* operations: "
                  f"{', '.join('`' + o + '`' for o in point.get('operations') or ()) or '—'}",
                  ""]
        mine = [r for r in rows if str(r.get("point_id")) == pid]
        measured = [r for r in mine if r.get("verdict") != "NO_CANDIDATES"]
        if not measured:
            for row in mine:
                lines += [f"> **NO_CANDIDATES.** {row.get('note') or ''}", ""]
            if not mine:
                lines += ["> _Not reached in this run — no cell was recorded "
                          "for this point._", ""]
            continue

        tally = stat.get("verdicts") or {}
        lines.append(
            "* verdicts: " + ", ".join(
                f"{tally.get(v, 0)} {v}" for v in reversed(VERDICT_ORDER)
                if tally.get(v)))
        lines.append("")
        ranked = sorted(
            measured,
            key=lambda r: (VERDICT_ORDER.index(str(r.get("verdict") or
                                                   "incapable")) * -1,
                           -(quality_of(_row_stat(r))),
                           (r.get("perf") or {}).get("latency_s") or float("inf"),
                           str(r.get("model"))))
        shown = ranked if limit is None else ranked[:limit]
        if limit is not None and len(ranked) > limit:
            lines.append(f"_Top {limit} of {len(ranked)} candidate(s); the "
                         f"full ordering is in `cells.jsonl`._")
            lines.append("")
        lines += ["| # | model | verdict | det | judge | latency | VRAM Δ GiB "
                  "| note |", "|---|---|---|---|---|---|---|---|"]
        for index, row in enumerate(shown, start=1):
            perf = row.get("perf") or {}
            det = (row.get("deterministic") or {}).get("score")
            judge = row.get("judge") or {}
            lines.append(
                f"| {index} | `{row.get('model')}` | "
                f"{VERDICT_MARK.get(str(row.get('verdict')), '')} "
                f"{row.get('verdict')} | {_cell(det)} | "
                f"{_cell(judge.get('score') if judge.get('available') else None)} | "
                f"{_cell(perf.get('latency_s'), 1, 's')} | "
                f"{_gib(perf.get('vram_used_delta_bytes'))} | "
                f"{str(row.get('note') or '')[:110].replace('|', '/')} |")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "COMPOSITE_QUALITY_WEIGHT", "COMPOSITE_SPEED_WEIGHT", "Candidate",
    "DEFAULT_RUN_ROOT", "FORMULA_NOTE", "MATRIX_PATH_ENV",
    "REFERENCE_LATENCY_S", "RUN_ROOT_ENV", "RouteChoice",
    "RouteEntry", "RoutingMatrix", "SCHEMA_VERSION", "best_route",
    "composite_of", "derive_matrix", "load_latest_matrix", "load_matrix",
    "quality_of", "render_leaderboard", "save_matrix", "speed_of",
    "summarize",
    # --- k109b: the per-lifecycle-point view ---
    "VERDICT_MARK", "VERDICT_ORDER", "render_capability_grid",
    "render_point_leaderboards", "summarize_points",
]
