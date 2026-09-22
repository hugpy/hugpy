"""k120 — the DOSSIER: everything known about one discovered model, typed.

The nightly reviewer (``review/``) answered one question well — "does this repo
fit the card and does the GGUF load?" — and then handed the operator a MODEL
NAME with a score beside it. A name is not a decision. The operator asked for
"comprehensive… specializations and overall weights, research outside of the
download sources themselves, trial download and sample data to compare various
models", and this module is the shape that answer takes.

A :class:`ModelDossier` is eight sections, each of which is allowed to be
EMPTY and never allowed to be INVENTED:

  identity        repo id, org, and the lineage chain read off config + card
  specialization  what it is FOR — declared tasks, domains, languages, the
                  fine-tune focus, instruct/chat/base, modality, and an
                  ``emphasis`` weight per domain (the operator's "overall
                  weights", per specialization, 0..1 with the evidence behind
                  each one recorded)
  weights         the physical facts: params, file bytes, every quant with its
                  own VRAM estimate, context, architecture family, tokenizer
  trust           license, gating, org verification, downloads, likes, age,
                  open discussions
  research        the model card digested (benchmark tables, training data,
                  limitations), linked papers with best-effort abstracts, and
                  a ``research_notes`` summary written BY A MODEL and labelled
                  as such, with the sources it was given
  community       Reddit / HN / HF-discussion mentions, extracted claims, and
                  a recency-weighted ``heat`` (see ``community.py``)
  trial           what actually happened when it was run: load test, the
                  STATIONARY sample battery's outputs and scores, and the
                  comparison against the incumbent from the routing matrix
  verdict         adopt/trial/reject WITH reasons and the evidence refs those
                  reasons cite — or an honest "screened only", naming the
                  cause

THE TWO RULES THIS FILE EXISTS TO ENFORCE
    1. Every field is either SOURCED or ``None``. A number with no
       :class:`Source` behind it is a guess wearing a number's clothes. The
       ``sources`` list carries a URL and a ``fetched_at`` for every fetch,
       INCLUDING the ones that failed — an unreachable arXiv is a recorded
       ``unavailable`` row, never a silently missing paper.
    2. Model-generated text is LABELLED. ``research_notes`` and the community
       claims come from a local model reading the sources; both carry the
       model id that wrote them and both are surfaced as model-generated in
       the UI. Nothing here ever presents a model's summary as a measurement.

No pathlib, os.path only — the oracle/fleet_doctrine house style, and this is
imported by a systemd timer that must not drag in anything heavy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

#: Bump when a section's shape changes. Recorded on every dossier, because a
#: dossier read by a UI that expects another shape is worse than no dossier.
SCHEMA_VERSION: str = "dossier/1"

#: The three trial depths a card can ask for, cheapest first. Each is a
#: SUPERSET of the one before it.
TRIAL_DEPTHS: tuple[str, ...] = ("screen-only", "load-test", "full-samples")

#: The verdicts a judge may return, plus the one this module writes itself when
#: there is no trial evidence to judge.
VERDICTS: tuple[str, ...] = ("adopt", "trial", "reject", "screened-only")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Source:
    """One thing that was fetched (or could not be).

    ``ok=False`` is a FIRST-CLASS row, not an omission: "arXiv timed out" and
    "this model has no paper" are different facts and an operator deciding on
    a model deserves to know which one they are looking at."""
    kind: str                       # hf-api | hf-readme | arxiv | reddit | …
    url: str | None = None
    fetched_at: str | None = None
    ok: bool = True
    detail: str = ""                # honest cause when ok is False

    @classmethod
    def unavailable(cls, kind: str, detail: str,
                    url: str | None = None) -> "Source":
        return cls(kind=kind, url=url, fetched_at=utc_now(), ok=False,
                   detail=detail)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Section 1 — identity and lineage
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Identity:
    """Who this repo is and what it was made from.

    ``lineage`` is the chain NEAREST FIRST — a GGUF quant of a fine-tune of a
    base model reads ``("someone/foo-finetune", "Qwen/Qwen3-8B")``. It stops
    where the evidence stops; a chain that could not be walked further is
    short, never padded."""
    hub_id: str
    org: str | None = None
    name: str | None = None
    base_model: str | None = None
    lineage: tuple[str, ...] = ()
    relation: str | None = None      # quantization | finetune | merge | adapter
    is_derivative: bool | None = None
    created_from_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Section 2 — specialization ("what is it FOR")
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmphasisWeight:
    """One specialization and how strongly the evidence supports it.

    ``weight`` is 0..1 and ``evidence`` names WHERE it came from (a tag, a card
    phrase, the repo name). A weight with no evidence string is a bug — the
    constructor of this dataclass is the only place a number is allowed to be
    born, and ``build_emphasis`` never emits one without a reason."""
    domain: str
    weight: float
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Specialization:
    declared_tasks: tuple[str, ...] = ()
    pipeline_tag: str | None = None
    modality: str | None = None            # text | vision-language | image | …
    variant: str | None = None             # instruct | chat | base | reasoning
    instruct: bool | None = None
    languages: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    emphasis: tuple[EmphasisWeight, ...] = ()
    finetune_focus: tuple[str, ...] = ()   # verbatim card phrases
    notes: tuple[str, ...] = ()

    @property
    def headline(self) -> str:
        """One line for a UI row. Empty string when nothing is known — an
        honest blank beats a confident 'general purpose'."""
        bits = [b for b in (self.modality, self.variant) if b]
        if self.domains:
            bits.append("/".join(self.domains[:3]))
        if self.languages:
            bits.append(f"{len(self.languages)} languages"
                        if len(self.languages) > 3
                        else ", ".join(self.languages))
        return " · ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["headline"] = self.headline
        return d


# ---------------------------------------------------------------------------
# Section 3 — the weights themselves
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuantFact:
    """One installable variant, with its OWN VRAM estimate.

    The reviewer's screen already picked a best quant; a dossier keeps them
    ALL, because "the Q4 fits and the Q6 doesn't" is exactly the trade the
    operator wants to see rather than have made for them."""
    quant: str
    bytes: int | None = None
    files: tuple[str, ...] = ()
    bits_per_weight: float | None = None
    est_weights_bytes: int | None = None
    est_kv_bytes: int | None = None
    est_vram_bytes: int | None = None
    fits_vram: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WeightsFacts:
    params: int | None = None
    params_source: str | None = None       # safetensors | config | gguf-name | card
    total_bytes: int | None = None
    context_length: int | None = None
    context_source: str | None = None
    architecture: str | None = None
    architecture_family: str | None = None
    tokenizer: str | None = None
    torch_dtype: str | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    quants: tuple[QuantFact, ...] = ()
    best_quant: str | None = None
    vram_budget_bytes: int | None = None
    target_context: int | None = None
    notes: tuple[str, ...] = ()

    def quant(self, name: str | None) -> QuantFact | None:
        return next((q for q in self.quants if q.quant == name), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["geometry"] = dict(self.geometry)
        return d


# ---------------------------------------------------------------------------
# Section 4 — trust
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrustSignals:
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None
    age_days: int | None = None
    trust_tier: int | None = None
    org_verified: bool | None = None
    license: str | None = None
    license_url: str | None = None
    gated: Any = None
    private: bool | None = None
    discussions_open: int | None = None
    discussions_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Section 5 — external research
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkClaim:
    """A number the CARD claims. Never a number this fleet measured — the two
    live in different sections on purpose, and the UI labels this one
    'claimed'."""
    benchmark: str
    value: str | None = None
    metric: str | None = None
    comparator: str | None = None          # the row/column it was compared to
    source: str = "model card"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CardDigest:
    """The model card README, structured. Empty sections stay empty."""
    chars: int = 0
    headings: tuple[str, ...] = ()
    summary: str = ""
    benchmark_claims: tuple[BenchmarkClaim, ...] = ()
    benchmark_tables: int = 0
    training_data: str = ""
    limitations: str = ""
    intended_use: str = ""
    prompt_format: str = ""
    links: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperRef:
    arxiv_id: str | None = None
    title: str | None = None
    url: str | None = None
    abstract: str | None = None
    unavailable: str = ""              # why the abstract is missing, if it is

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExternalResearch:
    card: CardDigest | None = None
    papers: tuple[PaperRef, ...] = ()
    research_notes: str | None = None
    research_notes_model: str | None = None
    model_generated: bool = False
    cited: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"card": self.card.to_dict() if self.card else None,
                "papers": [p.to_dict() for p in self.papers],
                "research_notes": self.research_notes,
                "research_notes_model": self.research_notes_model,
                "model_generated": self.model_generated,
                "cited": list(self.cited),
                "unavailable": list(self.unavailable)}


# ---------------------------------------------------------------------------
# Section 6 — community intelligence (see community.py)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Mention:
    """One public post that names the model."""
    source: str                        # reddit:LocalLLaMA | hackernews | hf-discussions | youtube
    url: str | None = None
    title: str | None = None
    snippet: str | None = None
    author: str | None = None
    ts: float | None = None            # epoch seconds
    score: int | None = None           # upvotes/points where the source has them

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Claim:
    """One thing the community SAYS, with the quote and the link behind it.

    Model-generated (an LLM read the mentions and extracted these); the
    ``quote`` is what makes it checkable and the ``url`` is what makes it
    accountable. A claim with neither is dropped by ``community.py`` rather
    than shown."""
    kind: str                          # praise | criticism | benchmark | quirk
    text: str
    quote: str = ""
    url: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Community:
    heat: float = 0.0
    mentions: tuple[Mention, ...] = ()
    claims: tuple[Claim, ...] = ()
    sources: tuple[Source, ...] = ()
    generated_by: str | None = None     # the model that extracted the claims
    model_generated: bool = False

    @property
    def unavailable(self) -> tuple[str, ...]:
        return tuple(f"{s.kind}: {s.detail}" for s in self.sources if not s.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"heat": self.heat,
                "mentions": [m.to_dict() for m in self.mentions],
                "claims": [c.to_dict() for c in self.claims],
                "sources": [s.to_dict() for s in self.sources],
                "unavailable": list(self.unavailable),
                "generated_by": self.generated_by,
                "model_generated": self.model_generated}


# ---------------------------------------------------------------------------
# Section 7 — trial evidence
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampleOutput:
    """One piece of work the candidate actually produced.

    ``artifact_ref`` is a PATH on the shared store for anything that is not
    text (a keyframe png, a clip mp4, a wav); ``snippet`` carries the first
    part of a text answer so the UI has something to show without opening a
    file."""
    operation: str
    case_id: str = ""
    kind: str = "text"                 # text | image | video | audio
    snippet: str | None = None
    artifact_ref: str | None = None
    chars: int | None = None
    seconds: float | None = None
    ok: bool = True
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SampleScore:
    """k109b's scoring of one sample. ``deterministic`` and ``judge`` are the
    benchmark module's own numbers — imported, never recomputed here."""
    operation: str
    case_id: str = ""
    ok: bool = False
    deterministic: float | None = None
    judge: float | None = None
    quality: float | None = None
    latency_s: float | None = None
    failure: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IncumbentComparison:
    """Candidate vs the model the routing matrix currently routes to.

    ``beats_incumbent`` is deliberately a THREE-valued string. "untested" is
    the common case on a fresh fleet and is not a failure — it means the
    matrix has no measured row for this operation, and a verdict that claimed
    otherwise would be inventing a comparison."""
    operation: str
    incumbent: str | None = None
    incumbent_quality: float | None = None
    candidate_quality: float | None = None
    margin: float | None = None
    beats_incumbent: str = "untested"      # yes | no | untested
    basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrialEvidence:
    depth: str = "screen-only"
    backend: str | None = None             # dispatch | local-gguf | none
    scenario_version: str | None = None
    scenario_digest: str | None = None
    load: Mapping[str, Any] | None = None  # the smoke/load-test result
    samples: tuple[SampleOutput, ...] = ()
    scores: tuple[SampleScore, ...] = ()
    comparisons: tuple[IncumbentComparison, ...] = ()
    blocked: str | None = None             # honest cause when nothing ran
    ran_at: str | None = None
    run_dir: str | None = None

    @property
    def has_evidence(self) -> bool:
        """True only when something was actually MEASURED. A load test counts;
        a screen does not. This property is the gate ``verdicts.py`` uses, so
        the "a verdict must cite evidence" rule has exactly one definition."""
        return bool(self.scores or self.samples or (self.load or {}).get("ok"))

    @property
    def mean_quality(self) -> float | None:
        vals = [s.quality for s in self.scores if s.quality is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def to_dict(self) -> dict[str, Any]:
        return {"depth": self.depth, "backend": self.backend,
                "scenario_version": self.scenario_version,
                "scenario_digest": self.scenario_digest,
                "load": dict(self.load) if self.load else None,
                "samples": [s.to_dict() for s in self.samples],
                "scores": [s.to_dict() for s in self.scores],
                "comparisons": [c.to_dict() for c in self.comparisons],
                "blocked": self.blocked, "ran_at": self.ran_at,
                "run_dir": self.run_dir,
                "has_evidence": self.has_evidence,
                "mean_quality": self.mean_quality}


# ---------------------------------------------------------------------------
# Section 8 — the verdict
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Verdict:
    """adopt / trial / reject — and WHY, in refs a reader can follow.

    ``evidence_refs`` are dotted paths INTO this dossier
    (``trial.scores[0]``, ``weights.quants[2]``, ``trust.license``). They are
    what makes the verdict auditable: a reason with no ref is an opinion, and
    ``verdicts.py`` refuses to file one."""
    verdict: str = "screened-only"
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    confidence: str = "screened-only"      # evidence-backed | screened-only
    judged_by: str | None = None
    blocked: str | None = None
    capability: int | None = None
    fit: int | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The dossier
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelDossier:
    hub_id: str
    criteria: str = ""
    schema: str = SCHEMA_VERSION
    generated_at: str = field(default_factory=utc_now)
    identity: Identity | None = None
    specialization: Specialization | None = None
    weights: WeightsFacts | None = None
    trust: TrustSignals | None = None
    research: ExternalResearch | None = None
    community: Community | None = None
    trial: TrialEvidence | None = None
    verdict: Verdict | None = None
    screen: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[Source, ...] = ()
    notes: tuple[str, ...] = ()

    def add_source(self, source: Source) -> None:
        self.sources = self.sources + (source,)

    def add_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes = self.notes + (note,)

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Every fetch that did not happen, said out loud. The UI shows this
        verbatim; an empty tuple means every source answered."""
        rows = [f"{s.kind}: {s.detail}" for s in self.sources if not s.ok]
        if self.community:
            rows.extend(self.community.unavailable)
        if self.research:
            rows.extend(self.research.unavailable)
        seen, out = set(), []
        for row in rows:
            if row not in seen:
                seen.add(row)
                out.append(row)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "hub_id": self.hub_id,
            "criteria": self.criteria, "generated_at": self.generated_at,
            "identity": self.identity.to_dict() if self.identity else None,
            "specialization": (self.specialization.to_dict()
                               if self.specialization else None),
            "weights": self.weights.to_dict() if self.weights else None,
            "trust": self.trust.to_dict() if self.trust else None,
            "research": self.research.to_dict() if self.research else None,
            "community": self.community.to_dict() if self.community else None,
            "trial": self.trial.to_dict() if self.trial else None,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "screen": dict(self.screen),
            "sources": [s.to_dict() for s in self.sources],
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str,
                          sort_keys=False)

    # -- reading back ------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelDossier":
        """Rehydrate. TOLERANT by design: a dossier written by an older
        release is read for what it has, never rejected — the alternative is a
        console that goes blank after a deploy."""
        d = dict(d or {})
        return cls(
            hub_id=str(d.get("hub_id") or ""),
            criteria=str(d.get("criteria") or ""),
            schema=str(d.get("schema") or SCHEMA_VERSION),
            generated_at=str(d.get("generated_at") or utc_now()),
            identity=_load(Identity, d.get("identity")),
            specialization=_load_specialization(d.get("specialization")),
            weights=_load_weights(d.get("weights")),
            trust=_load(TrustSignals, d.get("trust")),
            research=_load_research(d.get("research")),
            community=_load_community(d.get("community")),
            trial=_load_trial(d.get("trial")),
            verdict=_load(Verdict, d.get("verdict")),
            screen=dict(d.get("screen") or {}),
            sources=tuple(_load(Source, s) for s in (d.get("sources") or [])
                          if isinstance(s, Mapping)),
            notes=_tuple(d.get("notes")),
        )


def _fields(cls) -> set[str]:
    return set(getattr(cls, "__dataclass_fields__", {}))


def _load(cls, payload: Any):
    """Build ``cls`` from a mapping, dropping unknown keys and coercing the
    declared tuple fields back to tuples. Returns None for a missing section
    (never a half-built object)."""
    if not isinstance(payload, Mapping):
        return None
    known = _fields(cls)
    vals: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            continue
        decl = str(cls.__dataclass_fields__[key].type)
        vals[key] = tuple(value) if (decl.startswith("tuple")
                                     and isinstance(value, list)) else value
    return cls(**vals)


def _load_specialization(payload: Any) -> Specialization | None:
    spec = _load(Specialization, payload)
    if spec is None:
        return None
    rows = (payload or {}).get("emphasis") or ()
    spec.emphasis = tuple(e for e in (_load(EmphasisWeight, r) for r in rows
                                      if isinstance(r, Mapping))
                          if e is not None)
    return spec


def _load_weights(payload: Any) -> WeightsFacts | None:
    facts = _load(WeightsFacts, payload)
    if facts is None:
        return None
    rows = (payload or {}).get("quants") or ()
    facts.quants = tuple(q for q in (_load(QuantFact, r) for r in rows
                                     if isinstance(r, Mapping))
                         if q is not None)
    return facts


def _load_research(payload: Any) -> ExternalResearch | None:
    if not isinstance(payload, Mapping):
        return None
    card = _load(CardDigest, payload.get("card"))
    if card is not None:
        card.benchmark_claims = tuple(
            c for c in (_load(BenchmarkClaim, r)
                        for r in (payload.get("card") or {}).get(
                            "benchmark_claims") or ()
                        if isinstance(r, Mapping)) if c is not None)
    return ExternalResearch(
        card=card,
        papers=tuple(p for p in (_load(PaperRef, r)
                                 for r in payload.get("papers") or ()
                                 if isinstance(r, Mapping)) if p is not None),
        research_notes=payload.get("research_notes"),
        research_notes_model=payload.get("research_notes_model"),
        model_generated=bool(payload.get("model_generated")),
        cited=_tuple(payload.get("cited")),
        unavailable=_tuple(payload.get("unavailable")))


def _load_community(payload: Any) -> Community | None:
    if not isinstance(payload, Mapping):
        return None
    return Community(
        heat=float(payload.get("heat") or 0.0),
        mentions=tuple(m for m in (_load(Mention, r)
                                   for r in payload.get("mentions") or ()
                                   if isinstance(r, Mapping)) if m is not None),
        claims=tuple(c for c in (_load(Claim, r)
                                 for r in payload.get("claims") or ()
                                 if isinstance(r, Mapping)) if c is not None),
        sources=tuple(s for s in (_load(Source, r)
                                  for r in payload.get("sources") or ()
                                  if isinstance(r, Mapping)) if s is not None),
        generated_by=payload.get("generated_by"),
        model_generated=bool(payload.get("model_generated")))


def _load_trial(payload: Any) -> TrialEvidence | None:
    if not isinstance(payload, Mapping):
        return None
    return TrialEvidence(
        depth=str(payload.get("depth") or "screen-only"),
        backend=payload.get("backend"),
        scenario_version=payload.get("scenario_version"),
        scenario_digest=payload.get("scenario_digest"),
        load=payload.get("load"),
        samples=tuple(s for s in (_load(SampleOutput, r)
                                  for r in payload.get("samples") or ()
                                  if isinstance(r, Mapping)) if s is not None),
        scores=tuple(s for s in (_load(SampleScore, r)
                                 for r in payload.get("scores") or ()
                                 if isinstance(r, Mapping)) if s is not None),
        comparisons=tuple(c for c in (_load(IncumbentComparison, r)
                                      for r in payload.get("comparisons") or ()
                                      if isinstance(r, Mapping))
                          if c is not None),
        blocked=payload.get("blocked"), ran_at=payload.get("ran_at"),
        run_dir=payload.get("run_dir"))


__all__ = [
    "BenchmarkClaim", "CardDigest", "Claim", "Community", "EmphasisWeight",
    "ExternalResearch", "Identity", "IncumbentComparison", "Mention",
    "ModelDossier", "PaperRef", "QuantFact", "SCHEMA_VERSION", "SampleOutput",
    "SampleScore", "Source", "Specialization", "TRIAL_DEPTHS", "TrialEvidence",
    "TrustSignals", "VERDICTS", "Verdict", "WeightsFacts", "utc_now",
]
