"""k120 — a verdict has to cite something.

The nightly judge used to answer ``{"verdict": "trial", "summary": "looks
promising"}``. That is a horoscope. It was produced from four probe prompts and
a VRAM estimate, it named nothing it had measured, and it went into the console
looking exactly like a conclusion.

So the rule, enforced here and nowhere else:

    NO EVIDENCE, NO VERDICT.
    A dossier whose trial produced nothing measurable can only carry
    ``screened-only``, and it must NAME the cause: "screened only — trial
    blocked: <why>". The three real verdicts (adopt / trial / reject) require
    :attr:`TrialEvidence.has_evidence`, and every reason must arrive with an
    ``evidence_refs`` entry pointing INTO the dossier.

    A judge that answers "adopt" with no refs gets its refs DERIVED from the
    facts it was handed, and if not even that is possible its verdict is
    downgraded to ``screened-only`` with the reason recorded. Downgrading is
    the safe direction: the cost of a missed adoption is a model that gets
    reviewed again tomorrow night; the cost of an unearned adoption is a fleet
    routing production work to something nobody measured.

The judge is optional. When no agent answers, :func:`rule_verdict` files one
from the numbers alone — beats-the-incumbent, licence, VRAM fit — with the
same reasons-and-refs discipline and ``judged_by=None``. An unreachable judge
must never cost the operator a decision they could have made from the data.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Sequence

from hugpy_curation.dossier import llm
from hugpy_curation.dossier.dossier import ModelDossier, TrialEvidence, VERDICTS, Verdict

logger = logging.getLogger(__name__)

#: Licences that need a human before anything routes production work to them.
RESTRICTIVE_LICENCE_MARKERS: tuple[str, ...] = (
    "cc-by-nc", "noncommercial", "non-commercial", "research-only",
    "research only", "openrail", "other", "proprietary")

JUDGE_SYSTEM = (
    "You review candidate models for a small self-hosted GPU fleet. You are "
    "given a DOSSIER of measured facts about one candidate: what it is "
    "specialized for, its size and quantization, the VRAM it needs, its "
    "licence, what the community says about it, and — the part that decides "
    "this — the scores it earned running the SAME fixed brief the fleet's "
    "current model was scored on. Judge only from those facts. Never invent a "
    "benchmark number. Be blunt about weaknesses.")

JUDGE_INSTRUCTION = """Return ONLY a JSON object, no prose around it:
  "verdict": "adopt" | "trial" | "reject"
  "reasons": array of 2-5 short strings. EVERY reason must refer to a fact \
above; a reason that could be written about any model is not a reason.
  "evidence": array of the same length as "reasons", each the id in square \
brackets of the fact that reason rests on (for example "trial.scores[0]" or \
"trust.license")
  "capability": integer 1-10
  "fit": integer 1-10
  "summary": one or two sentences

adopt  = measurably better than the incumbent at something the fleet does, and \
it fits.
trial  = promising but the evidence is thin, or it wins on one axis and loses \
on another.
reject = it does not fit, it is not better, its licence forbids the use, or it \
failed the battery.
"""


# ---------------------------------------------------------------------------
# The facts a judge is allowed to see
# ---------------------------------------------------------------------------


def evidence_facts(dossier: ModelDossier) -> dict[str, Any]:
    """The dossier, flattened to the facts a verdict may rest on — each under
    the REF that names it. The judge is asked to cite these ids, so what it
    can cite and what it can see are the same set by construction."""
    facts: dict[str, Any] = {}
    ident, spec = dossier.identity, dossier.specialization
    if ident:
        facts["identity.hub_id"] = ident.hub_id
        if ident.base_model:
            facts["identity.base_model"] = ident.base_model
        if ident.lineage:
            facts["identity.lineage"] = list(ident.lineage)
    if spec:
        facts["specialization.headline"] = spec.headline
        if spec.domains:
            facts["specialization.domains"] = list(spec.domains)
        if spec.finetune_focus:
            facts["specialization.finetune_focus"] = list(spec.finetune_focus)

    weights = dossier.weights
    if weights:
        if weights.params:
            facts["weights.params"] = weights.params
        if weights.context_length:
            facts["weights.context_length"] = weights.context_length
        for i, quant in enumerate(weights.quants):
            facts[f"weights.quants[{i}]"] = {
                "quant": quant.quant,
                "est_vram_gib": (round(quant.est_vram_bytes / 1024 ** 3, 2)
                                 if quant.est_vram_bytes else None),
                "fits_vram": quant.fits_vram}
        if weights.vram_budget_bytes:
            facts["weights.vram_budget_gib"] = round(
                weights.vram_budget_bytes / 1024 ** 3, 2)

    trust = dossier.trust
    if trust:
        facts["trust.license"] = trust.license
        facts["trust.gated"] = trust.gated
        facts["trust.downloads"] = trust.downloads
        facts["trust.trust_tier"] = trust.trust_tier

    research = dossier.research
    if research and research.card and research.card.benchmark_claims:
        facts["research.card.benchmark_claims"] = [
            f"{c.benchmark}={c.value} (CLAIMED BY THE CARD, not measured)"
            for c in research.card.benchmark_claims[:10]]
    if research and research.card and research.card.limitations:
        facts["research.card.limitations"] = research.card.limitations[:500]

    community = dossier.community
    if community and community.claims:
        facts["community.heat"] = community.heat
        facts["community.claims"] = [
            f"{c.kind}: {c.text}" for c in community.claims[:8]]

    trial = dossier.trial
    if trial:
        for i, score in enumerate(trial.scores):
            facts[f"trial.scores[{i}]"] = {
                "operation": score.operation, "ok": score.ok,
                "deterministic": score.deterministic, "judge": score.judge,
                "quality": score.quality, "failure": score.failure}
        for i, comparison in enumerate(trial.comparisons):
            facts[f"trial.comparisons[{i}]"] = {
                "operation": comparison.operation,
                "incumbent": comparison.incumbent,
                "incumbent_quality": comparison.incumbent_quality,
                "candidate_quality": comparison.candidate_quality,
                "margin": comparison.margin,
                "beats_incumbent": comparison.beats_incumbent}
        if trial.blocked:
            facts["trial.blocked"] = trial.blocked
    return facts


def doctrine_note() -> str:
    """One line about the environment doctrine (k118), when one is installed.

    Read through :mod:`hugpy_curation.providers` (``DoctrineSource``), never
    by importing the fleet: the fleet package sits above curation and may
    not be installed on the box running the timer. Its absence is a note,
    never a failure."""
    from hugpy_curation.providers import get_doctrine_source
    try:
        doctrine = get_doctrine_source().latest()
    except Exception as exc:                        # noqa: BLE001
        return (f"fleet doctrine unavailable ({type(exc).__name__}) — VRAM and "
                f"dependency fit are judged from the dossier's own numbers")
    if doctrine is None:
        return ("no fleet doctrine snapshot exists yet — VRAM and dependency "
                "fit are judged from the dossier's own numbers")
    version = getattr(doctrine, "version", None)
    if version is None and isinstance(doctrine, dict):
        version = doctrine.get("version")
    return (f"fleet doctrine {version if version is not None else '?'} is "
            f"available; a candidate needing a dependency the reference box "
            f"lacks would be a blocker")


# ---------------------------------------------------------------------------
# The rule-based verdict (no judge needed)
# ---------------------------------------------------------------------------


def _licence_problem(license_id: str | None) -> str | None:
    if not license_id:
        return "the repo declares no licence"
    low = str(license_id).lower()
    for marker in RESTRICTIVE_LICENCE_MARKERS:
        if marker in low:
            return f"licence {license_id!r} restricts use and needs a human"
    return None


def rule_verdict(dossier: ModelDossier) -> Verdict:
    """A verdict from the numbers alone, with reasons and refs.

    Used when no judge answers, and used as the FLOOR when one does: a judge
    saying "adopt" over a candidate that lost every comparison is overruled
    below by :func:`judge_verdict`'s consistency check."""
    trial = dossier.trial or TrialEvidence()
    if not trial.has_evidence:
        cause = trial.blocked or "no trial was run"
        return Verdict(verdict="screened-only",
                       reasons=(f"screened only — trial blocked: {cause}",),
                       evidence_refs=("trial.blocked",),
                       confidence="screened-only", blocked=cause)

    reasons: list[str] = []
    refs: list[str] = []
    wins = [c for c in trial.comparisons if c.beats_incumbent == "yes"]
    losses = [c for c in trial.comparisons if c.beats_incumbent == "no"]
    untested = [c for c in trial.comparisons if c.beats_incumbent == "untested"]

    for i, comparison in enumerate(trial.comparisons):
        if comparison.beats_incumbent == "untested":
            continue
        reasons.append(
            f"{comparison.operation}: scored "
            f"{comparison.candidate_quality} against incumbent "
            f"{comparison.incumbent} at {comparison.incumbent_quality} "
            f"(margin {comparison.margin:+})")
        refs.append(f"trial.comparisons[{i}]")

    failed = [s for s in trial.scores if not s.ok]
    if failed:
        reasons.append(f"{len(failed)} of {len(trial.scores)} sample(s) did "
                       f"not produce a valid artifact")
        refs.append(f"trial.scores[{trial.scores.index(failed[0])}]")

    licence_issue = _licence_problem(
        dossier.trust.license if dossier.trust else None)
    if licence_issue:
        reasons.append(licence_issue)
        refs.append("trust.license")

    weights = dossier.weights
    fitting = [q for q in (weights.quants if weights else ()) if q.fits_vram]
    if weights and weights.quants and not fitting:
        reasons.append(f"no published quant fits the "
                       f"{(weights.vram_budget_bytes or 0)/1024**3:.1f} GiB "
                       f"budget at {weights.target_context} context")
        refs.append("weights.quants[0]")
    elif fitting:
        best = fitting[-1]
        reasons.append(f"{best.quant} fits the budget at "
                       f"~{(best.est_vram_bytes or 0)/1024**3:.1f} GiB")
        refs.append(f"weights.quants[{list(weights.quants).index(best)}]")

    if not reasons:
        reasons.append(f"the battery ran but produced no comparable number "
                       f"({len(untested)} operation(s) have no measured "
                       f"incumbent)")
        refs.append("trial.scores[0]" if trial.scores else "trial.blocked")

    if failed and len(failed) == len(trial.scores):
        verdict = "reject"
    elif licence_issue and not wins:
        verdict = "trial"
    elif wins and not losses:
        verdict = "adopt"
    elif wins:
        verdict = "trial"
    elif losses:
        verdict = "reject"
    else:
        verdict = "trial"

    return Verdict(verdict=verdict, reasons=tuple(reasons),
                   evidence_refs=tuple(refs), confidence="evidence-backed",
                   judged_by=None)


# ---------------------------------------------------------------------------
# The judged verdict
# ---------------------------------------------------------------------------


def _clean_refs(raw: Any, facts: Mapping[str, Any], count: int) -> list[str]:
    """Keep only refs that name a fact the judge was actually shown.

    A ref like ``trial.scores[0]`` ENDS in a bracket, so the obvious
    ``strip("[]")`` silently turns it into ``trial.scores[0`` and the verdict
    is thrown away for citing nothing. Unwrap only when the whole string is
    wrapped, and check membership before and after."""
    rows = raw if isinstance(raw, list) else []
    kept: list[str] = []
    for row in rows:
        if not isinstance(row, str):
            continue
        ref = row.strip().strip('"\'')
        if ref not in facts and ref.startswith("[") and ref.endswith("]"):
            ref = ref[1:-1].strip()
        if ref in facts and ref not in kept:
            kept.append(ref)
    return kept[:count]


def judge_verdict(dossier: ModelDossier, *,
                  dispatch: Callable[[str, str, int], str] | None = None
                  ) -> Verdict:
    """Ask a catalog-resolved model for the verdict, then HOLD IT TO THE RULE.

    Four things can go wrong and all four are handled by falling back to
    :func:`rule_verdict`, never by filing the judge's answer anyway: no model
    answers, the answer will not parse, the verdict is not one of the three, or
    the answer cites nothing the judge was shown."""
    trial = dossier.trial or TrialEvidence()
    if not trial.has_evidence:
        return rule_verdict(dossier)                # the rule, not the judge

    facts = evidence_facts(dossier)
    baseline = rule_verdict(dossier)
    prompt = (f"DOSSIER FACTS (cite these ids):\n"
              f"{json.dumps(facts, indent=2, default=str)}\n\n"
              f"FLEET NOTE: {doctrine_note()}\n\n"
              f"{JUDGE_INSTRUCTION}")
    text, model, detail = llm.ask(f"{JUDGE_SYSTEM}\n\n{prompt}",
                                  max_tokens=800, dispatch=dispatch)
    if not text:
        baseline.reasons = baseline.reasons + (
            f"filed from the measured numbers — no judge answered ({detail})",)
        return baseline

    parsed = llm.extract_json(text)
    if not isinstance(parsed, Mapping):
        baseline.reasons = baseline.reasons + (
            "filed from the measured numbers — the judge's reply did not "
            "parse as JSON",)
        baseline.judged_by = model
        baseline.raw = text[:2000]
        return baseline

    verdict = str(parsed.get("verdict") or "").strip().lower()
    reasons = [str(r).strip() for r in (parsed.get("reasons") or [])
               if isinstance(r, str) and str(r).strip()]
    refs = _clean_refs(parsed.get("evidence"), facts, len(reasons) or 5)

    if verdict not in ("adopt", "trial", "reject") or not reasons:
        baseline.reasons = baseline.reasons + (
            f"filed from the measured numbers — the judge returned "
            f"{verdict!r} with {len(reasons)} usable reason(s)",)
        baseline.judged_by = model
        baseline.raw = text[:2000]
        return baseline

    if not refs:
        # It had an opinion and cited nothing. Keep the opinion as a REASON on
        # the rule verdict rather than letting it stand as the verdict.
        baseline.reasons = baseline.reasons + (
            f"the judge ({model}) said {verdict!r} but cited no dossier fact, "
            f"so the verdict is the one the numbers support",)
        baseline.judged_by = model
        baseline.raw = text[:2000]
        return baseline

    # Consistency floor: adopt requires at least one measured win.
    wins = [c for c in trial.comparisons if c.beats_incumbent == "yes"]
    if verdict == "adopt" and not wins:
        verdict = "trial"
        reasons.append("downgraded to trial: no operation was measured as "
                       "beating the incumbent")
        refs.append(next((r for r in facts if r.startswith("trial.comparisons")),
                         "trial.scores[0]"))

    summary = str(parsed.get("summary") or "").strip()
    if summary:
        reasons.append(f"judge summary: {summary[:300]}")
        refs.append("trial.scores[0]" if trial.scores else "trial.blocked")

    return Verdict(
        verdict=verdict, reasons=tuple(reasons[:8]),
        evidence_refs=tuple(dict.fromkeys(refs))[:8],
        confidence="evidence-backed", judged_by=model,
        capability=_as_int(parsed.get("capability")),
        fit=_as_int(parsed.get("fit")), raw=None)


def _as_int(value: Any) -> int | None:
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return None


def decide(dossier: ModelDossier, *, judge: bool = True,
           dispatch: Callable[[str, str, int], str] | None = None) -> Verdict:
    """The one entry point. ``judge=False`` files the rule verdict directly."""
    if not judge:
        return rule_verdict(dossier)
    try:
        return judge_verdict(dossier, dispatch=dispatch)
    except Exception as exc:                        # noqa: BLE001 — a judge
        # that explodes must not cost the dossier its verdict.
        logger.info("dossier verdict: judge path failed (%s: %s)",
                    type(exc).__name__, exc)
        verdict = rule_verdict(dossier)
        verdict.reasons = verdict.reasons + (
            f"filed from the measured numbers — the judge path failed "
            f"({type(exc).__name__})",)
        return verdict


__all__ = ["JUDGE_INSTRUCTION", "JUDGE_SYSTEM", "RESTRICTIVE_LICENCE_MARKERS",
           "decide", "doctrine_note", "evidence_facts", "judge_verdict",
           "rule_verdict"]
