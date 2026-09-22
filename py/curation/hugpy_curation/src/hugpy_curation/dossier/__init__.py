"""k120 — comprehensive model-discovery dossiers.

The nightly reviewer (``review/``) has run since July: screen HF on metadata,
download the best few, load-test them on the GPU, ask an agent for a verdict.
It works, and what it hands the operator is a MODEL NAME with a score. The
operator's read of that, verbatim:

    "just model names… this needs to be comprehensive in nature and allow for
    that complexity to be dictated. specializations and overall weights,
    research outside of the download sources themselves, trial download and
    sample data to compare various models. this needs to be genuinely useful."

    "scraping social media, AI forums, reddits, youtube transcriptions — all of
    these things to keep the edge on a gem that exists."

This package is the answer, in eight sections and one rule.

    dossier.py      the types. Every field SOURCED or None; model-generated
                    text labelled as such, everywhere it appears.
    cards.py        the README read like a reviewer — benchmark tables as
                    CLAIMS, limitations, the author's own purpose sentences,
                    and a 0..1 weight per specialization with its evidence.
    weights.py      params, every quant, and a VRAM number for EACH of them at
                    the same target context. The KV maths is ``review.screen``'s,
                    imported — two estimators that disagree is a 3am bug.
    fetch.py        the only way out to the network: never raises, polite
                    (real UA, one request per host per 2s), disk-cached for 20h.
    research.py     model card + linked papers + a written summary from a
                    catalog-resolved model, with its sources cited.
    community.py    Reddit / HN / HF discussions as typed Mentions, claims
                    extracted with the QUOTE that supports them, and a
                    recency-weighted heat. YouTube is wired and honest about
                    needing a dependency it will never hard-require (k121).
    radar.py        GEM RADAR: a second pass over the SAME cached pulls looking
                    for models no card is asking about yet.
    trial.py        k109b's stationary battery, borrowed whole. The candidate
                    does the SAME fixed work the incumbent in the routing
                    matrix was scored on, so subtracting the two means something.
    verdicts.py     THE RULE: no evidence, no verdict. A blocked trial can only
                    produce "screened only — trial blocked: <cause>".
    build.py        the orchestrator; a section that cannot be built leaves a
                    note and the other seven still arrive.
    store.py        the full dossier is a file; the review row carries a
                    compact summary and the path.

Nothing here is on the request path and nothing here is required: every entry
point degrades to an honest ``unavailable`` record rather than failing the
review that called it.
"""
from __future__ import annotations

from hugpy_curation.dossier.build import build_dossier
from hugpy_curation.dossier.dossier import (
    BenchmarkClaim,
    CardDigest,
    Claim,
    Community,
    EmphasisWeight,
    ExternalResearch,
    Identity,
    IncumbentComparison,
    Mention,
    ModelDossier,
    PaperRef,
    QuantFact,
    SCHEMA_VERSION,
    SampleOutput,
    SampleScore,
    Source,
    Specialization,
    TRIAL_DEPTHS,
    TrialEvidence,
    TrustSignals,
    VERDICTS,
    Verdict,
    WeightsFacts,
)
from hugpy_curation.dossier.radar import RadarHit, scan as radar_scan
from hugpy_curation.dossier.verdicts import decide, rule_verdict

__all__ = [
    "BenchmarkClaim", "CardDigest", "Claim", "Community", "EmphasisWeight",
    "ExternalResearch", "Identity", "IncumbentComparison", "Mention",
    "ModelDossier", "PaperRef", "QuantFact", "RadarHit", "SCHEMA_VERSION",
    "SampleOutput", "SampleScore", "Source", "Specialization", "TRIAL_DEPTHS",
    "TrialEvidence", "TrustSignals", "VERDICTS", "Verdict", "WeightsFacts",
    "build_dossier", "decide", "radar_scan", "rule_verdict",
]
