"""k120 — the TRIAL: make the candidate do the same work as the incumbent.

Two models are only comparable if they were asked the same thing. That is
k109b's entire premise and it is already built: ``oracle/stationary_scenario``
defines ONE brief (SALT LINE) and ``oracle/benchmark`` runs and scores every
lifecycle point against it. So the discovery trial does not invent a battery —
it BORROWS that one, which means a candidate's number and the incumbent's
number in the routing matrix were produced by the same code against the same
brief, and subtracting them means something.

WHAT RUNS, BY MODALITY
    text / vision-language  ``benchmark.run_llm_cell`` — k109's ``run_case``
                            with the stationary preamble, k110's validators,
                            the rubric judge
    image                   ``benchmark.run_image_cell`` — the fixed keyframe
                            prompt, geometry and seed, ONE render, with the
                            blank-image guard
    video                   ``benchmark.run_video_cell`` — one short low-res
                            clip of the same shot
    tts                     ``benchmark.run_tts_cell`` — the one locked line,
                            with the silent-wav guard and round-trip ASR

    Every one of those returns a ``Cell`` whose ``to_dict`` is a superset of an
    attempt row, so ``routing_matrix.summarize`` scores a candidate EXACTLY the
    way it scored the incumbent. Nothing is re-derived here.

THE TWO BACKENDS
    ``dispatch``    the candidate is servable by this fleet (it is in the
                    catalog). This is the good path and the only one that gets
                    judge scores and VRAM telemetry.
    ``local-gguf``  the weights are on disk but the fleet does not serve them
                    yet. The stationary prompts are run through llama.cpp in a
                    subprocess (``review/smoke``'s child) and the answers are
                    scored with ``benchmark.score_case``. Deterministic layer
                    only, one load per call, and the dossier SAYS so.

WHEN NOTHING CAN RUN
    ``TrialEvidence.blocked`` is filled with the cause in the operator's own
    terms — "download blocked: …", "gated repo", "no quant fits 24 GiB",
    "trial_depth is screen-only" — and ``verdicts.py`` refuses to file an
    evidence-backed verdict on it. A blocked trial is a RESULT. A silently
    empty one is a lie.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Mapping, Sequence

from hugpy_curation.dossier.dossier import (
    IncumbentComparison,
    SampleOutput,
    SampleScore,
    TrialEvidence,
    utc_now,
)

logger = logging.getLogger(__name__)

#: Modality -> the stationary point kind whose cells measure it.
KIND_FOR_MODALITY: dict[str, str] = {
    "text": "llm",
    "vision-language": "vlm",
    "image": "image",
    "video": "video",
    "audio": "tts",
}

#: Default operations to sample for a text candidate, cheapest useful first.
#: Overridden by whatever the routing matrix actually has an incumbent for —
#: a sample nobody can be compared against is half an answer.
DEFAULT_TEXT_OPERATIONS: tuple[str, ...] = (
    "plot.construct", "screenplay.complete", "continuity.bible",
    "screenplay.breakdown", "shots.design", "segment.compile-prompt")

#: How much of a text answer is kept inline on the dossier for the UI.
SNIPPET_CHARS: int = 900

#: Where sample artifacts land when the caller names no run dir.
DEFAULT_RUN_ROOT_ENV: str = "DOSSIER_TRIAL_ROOT"


def blocked(depth: str, cause: str) -> TrialEvidence:
    """The honest empty trial. Used everywhere a trial cannot start."""
    return TrialEvidence(depth=depth, backend="none", blocked=cause,
                         ran_at=utc_now())


def _scenario() -> tuple[str, str]:
    try:
        from hugpy_oracle.stationary_scenario import SCENARIO_VERSION, scenario_digest
        return SCENARIO_VERSION, scenario_digest()
    except Exception:                               # noqa: BLE001
        return "", ""


def trial_run_dir(root: str | None = None, label: str = "dossier") -> str:
    """A directory for this trial's artifacts. Under the benchmark run root by
    default so a dossier's keyframe sits beside the sweep's."""
    base = root
    if not base:
        # Curation's own tree (``DOSSIER_TRIAL_ROOT`` else <review_root>/trials);
        # never the oracle's benchmark run root — that is the oracle's state.
        from hugpy_curation.config import trials_root
        base = trials_root()
    path = os.path.join(base, f"dossier-{time.strftime('%Y%m%d-%H%M%S')}-{label}")
    try:
        os.makedirs(os.path.join(path, "keyframes"), exist_ok=True)
        os.makedirs(os.path.join(path, "raw"), exist_ok=True)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------------
# Is this candidate servable at all?
# ---------------------------------------------------------------------------


def catalog_model_for(hub_id: str) -> tuple[str | None, str]:
    """The catalog model_key that serves ``hub_id``, or None + why.

    Discovery's candidates are HUB IDS; the fleet dispatches on MODEL KEYS.
    The catalog is the only place that mapping lives, and asking it is how a
    trial finds out whether the model it is holding can be run at all."""
    try:
        from hugpy_oracle import catalog
    except Exception as exc:                        # noqa: BLE001
        return None, f"catalog unreadable ({type(exc).__name__}: {exc})"
    needle = hub_id.strip().lower()
    tail = needle.split("/")[-1]
    for capability in ("text.chat", "vision.describe", "image.generate",
                       "video.generate", "audio.tts"):
        try:
            view = catalog.get_capability(capability)
        except Exception:                           # noqa: BLE001
            continue
        if view is None:
            continue
        for model_id in (getattr(view, "model_ids", None) or ()):
            low = str(model_id).lower()
            if low == needle or low.replace("~", "/") == needle \
                    or low == tail or low.split("/")[-1] == tail:
                return str(model_id), (f"the catalog serves this repo as "
                                       f"{model_id!r} for {capability}")
    return None, (f"{hub_id} is not in this fleet's catalog — it has not been "
                  f"registered as a servable model")


# ---------------------------------------------------------------------------
# Backend A — through the fleet (the good path)
# ---------------------------------------------------------------------------


def _cell_to_sample(cell: Any, raw: str) -> SampleOutput:
    kind = {"image": "image", "video": "video", "tts": "audio"}.get(
        getattr(cell, "stage", "") or "", "text")
    artifact = getattr(cell, "artifact_ref", "") or None
    text = (raw or "").strip()
    return SampleOutput(
        operation=cell.operation, case_id=getattr(cell, "point_id", ""),
        kind=kind,
        snippet=(text[:SNIPPET_CHARS] if text and kind == "text" else None),
        artifact_ref=artifact, chars=len(text) or None,
        seconds=getattr(getattr(cell, "perf", None), "latency_s", None),
        ok=bool(getattr(cell, "ok", False)),
        failure=getattr(cell, "failure", None))


def _cell_to_score(cell: Any) -> SampleScore:
    det = getattr(cell, "deterministic", None)
    judge = getattr(cell, "judge", None)
    det_score = det.score if det is not None else None
    judge_score = (judge.score if judge is not None
                   and getattr(judge, "available", False) else None)
    quality = None
    if det_score is not None and judge_score is not None:
        quality = round((float(det_score) + float(judge_score)) / 2.0, 3)
    elif det_score is not None:
        quality = float(det_score)
    elif judge_score is not None:
        quality = float(judge_score)
    return SampleScore(
        operation=cell.operation, case_id=getattr(cell, "point_id", ""),
        ok=bool(getattr(cell, "ok", False)), deterministic=det_score,
        judge=judge_score, quality=quality,
        latency_s=getattr(getattr(cell, "perf", None), "latency_s", None),
        failure=getattr(cell, "failure", None),
        detail=getattr(cell, "note", "") or getattr(cell, "verdict", ""))


def _points_for(kind: str, operations: Sequence[str]) -> list[tuple[Any, str]]:
    """``[(point, operation), …]`` for a kind, filtered and ordered by
    ``operations`` when one is given."""
    from hugpy_oracle.stationary_scenario import points_for_kind
    pairs: list[tuple[Any, str]] = []
    for point in points_for_kind(kind):
        for operation in (point.operations or (point.point_id,)):
            pairs.append((point, operation))
    if not operations:
        return pairs
    order = {op: i for i, op in enumerate(operations)}
    wanted = [p for p in pairs if p[1] in order]
    wanted.sort(key=lambda p: order[p[1]])
    return wanted or pairs


def run_dispatch_trial(model_key: str, kind: str, *, sample_count: int = 2,
                       operations: Sequence[str] = (),
                       run_dir: str | None = None,
                       judge: bool = True,
                       deadline_s: float | None = None) -> TrialEvidence:
    """The stationary battery through the fleet's own dispatch path."""
    from hugpy_oracle import benchmark as ob

    version, digest = _scenario()
    directory = run_dir or trial_run_dir(label=kind)
    kwargs: dict[str, Any] = {"judge": judge, "label": "dossier"}
    if deadline_s:
        kwargs["deadline_s"] = float(deadline_s)
    config = ob.StationaryConfig(**kwargs)
    registry_version = ob._registry_version()

    if kind == "llm":
        ok, detail, _elapsed = ob.probe_text_model(model_key, config)
        if not ok:
            return blocked("full-samples",
                           f"{model_key} failed the admission probe: {detail}")

    samples: list[SampleOutput] = []
    scores: list[SampleScore] = []
    cells: list[Any] = []
    for point, operation in _points_for(kind, operations)[:max(1, sample_count)]:
        try:
            if kind == "llm":
                cell, raw = ob.run_llm_cell(
                    point, operation, model_key, config=config,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
            elif kind == "image":
                cell, raw = ob.run_image_cell(
                    point, model_key, config=config, run_dir=directory,
                    judge_model=None, registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
            elif kind == "video":
                cell, raw = ob.run_video_cell(
                    point, model_key, config=config, run_dir=directory,
                    judge_model=None, registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
            elif kind == "tts":
                cell, raw = ob.run_tts_cell(
                    point, model_key, config=config, run_dir=directory,
                    registry_version=registry_version,
                    scenario_version=version, scenario_digest=digest)
            else:
                return blocked("full-samples",
                               f"no stationary battery exists for a "
                               f"{kind!r} candidate")
        except Exception as exc:                    # noqa: BLE001 — a cell that
            # throws is a measurement that failed, and that IS the row.
            logger.info("dossier trial: %s cell failed for %s (%s: %s)",
                        operation, model_key, type(exc).__name__, exc)
            scores.append(SampleScore(
                operation=operation, case_id=getattr(point, "point_id", ""),
                ok=False, failure=f"{type(exc).__name__}: {exc}"[:300],
                detail="the cell raised before it could be scored"))
            continue
        cells.append(cell)
        samples.append(_cell_to_sample(cell, raw))
        scores.append(_cell_to_score(cell))

    return TrialEvidence(
        depth="full-samples", backend="dispatch", scenario_version=version,
        scenario_digest=digest, samples=tuple(samples), scores=tuple(scores),
        ran_at=utc_now(), run_dir=directory,
        blocked=None if scores else "no stationary point matched this "
                                    "candidate's modality",
        load={"cells": [c.to_dict() for c in cells]} if cells else None)


# ---------------------------------------------------------------------------
# Backend B — a GGUF on disk the fleet does not serve yet
# ---------------------------------------------------------------------------


def _smoke_llm(model_path: str, n_ctx: int, max_tokens: int
               ) -> Callable[[str], str]:
    """An ``llm(prompt) -> text`` backed by ``review/smoke``'s subprocess.

    ONE LOAD PER CALL, deliberately: a bad GGUF can hard-abort from native
    code, and a reviewer that dies mid-battery is worse than a slow one. The
    cost is recorded in the trial note so nobody reads the latency numbers as
    inference speed."""
    from hugpy_curation.review.smoke import smoke_test

    def llm(prompt: str) -> str:
        result = smoke_test(model_path, n_ctx=n_ctx, max_tokens=max_tokens,
                            probes=[prompt])
        if not result.ok:
            raise RuntimeError(result.error or "the local load produced nothing")
        probes = result.probes or []
        text = (probes[0].get("output") if probes else "") or ""
        if not text.strip():
            raise RuntimeError("the model returned no text")
        return text
    return llm


def run_local_gguf_trial(model_path: str, *, sample_count: int = 2,
                         operations: Sequence[str] = (),
                         n_ctx: int = 8192, max_tokens: int = 1200,
                         llm: Callable[[str], str] | None = None
                         ) -> TrialEvidence:
    """The stationary battery against a GGUF the catalog does not know.

    Deterministic layer only — there is no fleet route, so there is no judge
    and no VRAM telemetry — and the trial says exactly that in its note."""
    from hugpy_oracle import benchmark as ob
    from hugpy_oracle.benchmark_cases import STATIONARY_CASES
    from hugpy_oracle.stationary_scenario import SCENARIO_SCREENPLAY, stationary_preamble

    version, digest = _scenario()
    ask = llm or _smoke_llm(model_path, n_ctx, max_tokens)
    order = {op: i for i, op in enumerate(operations or DEFAULT_TEXT_OPERATIONS)}
    cases = sorted((c for c in STATIONARY_CASES if c.operation in order),
                   key=lambda c: order[c.operation])
    cases = (cases or list(STATIONARY_CASES))[:max(1, sample_count)]

    samples: list[SampleOutput] = []
    scores: list[SampleScore] = []
    for case in cases:
        started = time.monotonic()
        preamble = stationary_preamble(case.operation)
        try:
            result = ob.produce(case, ask, SCENARIO_SCREENPLAY, preamble)
        except Exception as exc:                    # noqa: BLE001
            scores.append(SampleScore(
                operation=case.operation, case_id=case.case_id, ok=False,
                failure=f"{type(exc).__name__}: {exc}"[:300],
                detail="the local load could not produce an answer"))
            samples.append(SampleOutput(
                operation=case.operation, case_id=case.case_id, ok=False,
                failure=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        latency = round(time.monotonic() - started, 3)
        raw = getattr(result, "raw", "") or ""
        if not raw and not hasattr(result, "code"):
            raw = str(result)
        det = ob.score_case(case, result, raw, SCENARIO_SCREENPLAY)
        samples.append(SampleOutput(
            operation=case.operation, case_id=case.case_id, kind="text",
            snippet=raw[:SNIPPET_CHARS] or None, chars=len(raw) or None,
            seconds=latency, ok=bool(det.valid)))
        scores.append(SampleScore(
            operation=case.operation, case_id=case.case_id, ok=bool(det.valid),
            deterministic=det.score, judge=None, quality=det.score,
            latency_s=latency,
            detail="deterministic layer only — this model is not served by the "
                   "fleet, so no rubric judge and no VRAM telemetry"))

    return TrialEvidence(
        depth="full-samples", backend="local-gguf", scenario_version=version,
        scenario_digest=digest, samples=tuple(samples), scores=tuple(scores),
        ran_at=utc_now(),
        load={"note": "one llama.cpp load per prompt — the latency numbers "
                      "here include model load and are NOT inference speed",
              "model_path": model_path})


# ---------------------------------------------------------------------------
# Comparison against the incumbent
# ---------------------------------------------------------------------------


def candidate_quality(operation: str, scores: Sequence[SampleScore],
                      cells: Sequence[Mapping[str, Any]] = ()
                      ) -> float | None:
    """The candidate's quality for one operation, on the MATRIX's own scale.

    When the trial produced full cell rows they go through
    ``routing_matrix.summarize`` + ``quality_of`` — literally the function that
    computed the incumbent's number. Only when there are no cell rows (the
    local-gguf backend) does this fall back to the mean of the deterministic
    scores, and the caller records that in the comparison's ``basis``."""
    if cells:
        try:
            from hugpy_oracle.routing_matrix import quality_of, summarize
            stats = [s for s in summarize(cells)
                     if s.get("operation") == operation]
            if stats:
                return round(float(quality_of(stats[0])), 3)
        except Exception as exc:                    # noqa: BLE001
            logger.info("dossier: matrix-scale quality unavailable (%s: %s)",
                        type(exc).__name__, exc)
    vals = [s.quality for s in scores
            if s.operation == operation and s.quality is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def compare_to_incumbent(evidence: TrialEvidence, *,
                         operations: Sequence[str] = (),
                         matrix: Any = None) -> tuple[IncumbentComparison, ...]:
    """Candidate vs the routing matrix's current primary, per operation.

    The matrix is loaded through ``load_latest_matrix``, which REFUSES a matrix
    whose ``registry_version`` does not match the live catalog — so a
    comparison is either against the fleet that exists or it is honestly
    ``untested``. That refusal is the whole reason this does not just read the
    newest json off disk."""
    from hugpy_oracle.routing_matrix import best_route, load_latest_matrix

    reason = ""
    if matrix is None:
        matrix, reason = load_latest_matrix()
    cells = ((evidence.load or {}).get("cells") or []) if evidence.load else []
    wanted = list(operations) or sorted({s.operation for s in evidence.scores})

    out: list[IncumbentComparison] = []
    for operation in wanted:
        mine = candidate_quality(operation, evidence.scores, cells)
        basis = ("routing-matrix scale (summarize + quality_of)" if cells
                 else "mean deterministic score — no fleet route, so no "
                      "judge layer")
        if matrix is None:
            out.append(IncumbentComparison(
                operation=operation, candidate_quality=mine,
                beats_incumbent="untested",
                basis=f"no usable routing matrix: {reason}"))
            continue
        choice = best_route(operation, matrix)
        if choice is None or not choice.primary:
            out.append(IncumbentComparison(
                operation=operation, candidate_quality=mine,
                beats_incumbent="untested",
                basis=f"the routing matrix has no measured primary for "
                      f"{operation}"))
            continue
        evidence_row = (choice.evidence or {}).get("primary") or {}
        theirs = evidence_row.get("quality")
        if mine is None or theirs is None:
            out.append(IncumbentComparison(
                operation=operation, incumbent=choice.primary,
                incumbent_quality=theirs, candidate_quality=mine,
                beats_incumbent="untested",
                basis="one side has no quality number to compare"))
            continue
        margin = round(float(mine) - float(theirs), 3)
        out.append(IncumbentComparison(
            operation=operation, incumbent=choice.primary,
            incumbent_quality=round(float(theirs), 3), candidate_quality=mine,
            margin=margin, beats_incumbent="yes" if margin > 0 else "no",
            basis=f"{basis}; matrix run {choice.run_id or '?'}, "
                  f"registry_version {choice.registry_version or '?'}"))
    return tuple(out)


# ---------------------------------------------------------------------------
# The entry point the pipeline calls
# ---------------------------------------------------------------------------


def run_trial(hub_id: str, *, modality: str | None, depth: str,
              sample_count: int = 2, operations: Sequence[str] = (),
              local_path: str | None = None, gated: Any = None,
              load: Mapping[str, Any] | None = None,
              compare_against: Sequence[str] = (),
              run_dir: str | None = None) -> TrialEvidence:
    """Everything above, chosen by depth and modality. Never raises.

    ``depth`` is the card's ``trial_depth``. ``screen-only`` and ``load-test``
    both return early WITH the load result they were given and a ``blocked``
    string naming the depth, because "we did not run samples" and "we ran
    samples and they failed" must never look the same on a dossier."""
    if depth == "screen-only":
        return TrialEvidence(depth=depth, backend="none", ran_at=utc_now(),
                             blocked="trial_depth is screen-only — this card "
                                     "asks for no download and no run")
    if depth == "load-test":
        evidence = TrialEvidence(
            depth=depth, backend="local-gguf" if local_path else "none",
            load=dict(load) if load else None, ran_at=utc_now(),
            blocked=None if (load or {}).get("ok") else
            "the load test did not succeed" if load else
            "trial_depth is load-test — no sample battery was asked for")
        if (load or {}).get("ok"):
            evidence.blocked = ("trial_depth is load-test — the model loaded "
                                "but no sample battery was asked for")
        return evidence

    if gated:
        return blocked(depth, f"gated repo — {hub_id} needs an accepted "
                              f"licence on this box's HF account before "
                              f"anything can be trialled")

    kind = KIND_FOR_MODALITY.get(modality or "", "llm")
    model_key, why = catalog_model_for(hub_id)
    if model_key:
        try:
            evidence = run_dispatch_trial(
                model_key, kind, sample_count=sample_count,
                operations=operations, run_dir=run_dir)
        except Exception as exc:                    # noqa: BLE001
            return blocked(depth, f"the stationary battery could not start: "
                                  f"{type(exc).__name__}: {exc}")
        evidence.comparisons = compare_to_incumbent(
            evidence, operations=compare_against or operations)
        return evidence

    if local_path and os.path.isfile(local_path):
        try:
            evidence = run_local_gguf_trial(
                local_path, sample_count=sample_count, operations=operations)
        except Exception as exc:                    # noqa: BLE001
            return blocked(depth, f"the local GGUF battery could not start: "
                                  f"{type(exc).__name__}: {exc}")
        evidence.comparisons = compare_to_incumbent(
            evidence, operations=compare_against or operations)
        return evidence

    return blocked(depth, f"nothing to run: {why}, and no local weights are "
                          f"on disk for it")


__all__ = ["DEFAULT_TEXT_OPERATIONS", "KIND_FOR_MODALITY", "blocked",
           "candidate_quality", "catalog_model_for", "compare_to_incumbent",
           "run_dispatch_trial", "run_local_gguf_trial", "run_trial",
           "trial_run_dir"]
