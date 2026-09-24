"""
Keyword extraction pipeline.

Two back-ends:
    "keybert"  — transformer-based (KeyBERT + sentence-BERT)
    "spacy"    — rule-based (POS + NER via spaCy)

Both are lazy-loaded singletons.  The combined pipeline merges results
from both, deduplicates, and computes density in one call.
"""

from __future__ import annotations

import re,dataclasses
from collections import Counter
from dataclasses import dataclass, field,asdict
from typing import Dict, List, Optional, Tuple

from abstract_essentials import SingletonMeta
from hugpy_platform.module_imports import get_sentence_transformers, is_available, require


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class ToDictMixin:
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class KeywordRequest(ToDictMixin):
    """Immutable parameter bag for any extraction call."""

    text: str
    top_n: int = 10
    diversity: float = 0.7
    use_mmr: bool = True
    stop_words: str = "english"
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)



@dataclass
class KeywordResult(ToDictMixin):
    """
    Structured output — callers inspect named fields instead of
    guessing dict keys or positional returns.
    """

    keywords_spacy: List[str] = field(default_factory=list)
    keywords_keybert: List[Tuple[str, float]] = field(default_factory=list)
    combined: List[str] = field(default_factory=list)
    density: Dict[str, float] = field(default_factory=dict)
    backends_used: List[str] = field(default_factory=list)
    backend_errors: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Presets — named parameter bundles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeywordPreset(ToDictMixin):
    """
    Frozen bag of defaults a preset name resolves to.
    Only non-None values override the caller's explicit kwargs.
    """

    top_n: Optional[int] = None
    diversity: Optional[float] = None
    use_mmr: Optional[bool] = None
    stop_words: Optional[str] = None
    keyphrase_ngram_range: Optional[Tuple[int, int]] = None
    min_density: Optional[float] = None
    max_density: Optional[float] = None
    min_score: Optional[float] = None
    max_words_per_phrase: Optional[int] = None
    dedupe_stems: Optional[bool] = None
    
_PRESETS: Dict[str, KeywordPreset] = {}


def register_preset(key: str, preset: KeywordPreset) -> None:
    if key in _PRESETS:
        raise KeyError(f"Keyword preset {key!r} already registered")
    _PRESETS[key] = preset


def available_presets() -> List[str]:
    return sorted(_PRESETS)


def get_preset(key: str) -> KeywordPreset:
    if key not in _PRESETS:
        raise KeyError(
            f"Unknown keyword preset {key!r}. Available: {available_presets()}"
        )
    return _PRESETS[key]


# ---- built-in presets ----------------------------------------------------

register_preset("default", KeywordPreset())

register_preset(
    "seo",
    KeywordPreset(
        top_n=15,                       # cast a wide net
        diversity=0.5,                  # moderate — want related clusters
        keyphrase_ngram_range=(1, 3),   # "machine learning pipeline" not just "machine"
        min_density=0.3,                # if it doesn't appear at all, it's not a real keyword
        max_density=4.0,                # above 4% starts looking like stuffing
        min_score=0.2,                  # drop low-confidence keybert noise
    ),
)

register_preset(
    "metadata",
    KeywordPreset(
        top_n=8,                        # tags should be tight
        diversity=0.7,                  # high — each tag covers a different facet
        keyphrase_ngram_range=(1, 2),   # short: "python", "web scraping"
        max_words_per_phrase=2,         # no long tails in meta tags
        min_score=0.3,                  # only confident picks
    ),
)

register_preset(
    "social",
    KeywordPreset(
        top_n=10,
        diversity=0.8,                  # every hashtag should feel distinct
        keyphrase_ngram_range=(1, 1),   # single words only — #machinelearning not #machine #learning
        max_words_per_phrase=1,
        min_score=0.15,
    ),
)

register_preset(
    "long_tail",
    KeywordPreset(
        top_n=12,
        diversity=0.3,                  # low — we *want* related phrases
        keyphrase_ngram_range=(2, 4),   # "best python web framework 2026"
        min_density=0.1,                # long-tail phrases are naturally sparse
        min_score=0.15,
    ),
)

register_preset(
    "article",
    KeywordPreset(
        top_n=12,
        diversity=0.6,                  # spread across the article's themes
        keyphrase_ngram_range=(1, 3),   # topics come in 1-3 word phrases
        min_density=0.2,                # must actually recur in the piece
        min_score=0.2,
    ),
)


# ---------------------------------------------------------------------------
# Refined result (extends KeywordResult with SEO/metadata analysis)
# ---------------------------------------------------------------------------

@dataclass
class RefinedResult(ToDictMixin):
    """
    Output of refine_keywords — the raw KeywordResult plus
    post-processed fields useful for SEO, metadata, and content strategy.
    """

    raw: KeywordResult = field(default_factory=KeywordResult)
    preset_used: str = "default"

    # Post-processed keyword lists
    primary: List[str] = field(default_factory=list)        # top picks after filtering
    secondary: List[str] = field(default_factory=list)      # didn't make primary but still relevant
    dropped: List[str] = field(default_factory=list)        # removed and why

    # Density analysis
    density: Dict[str, float] = field(default_factory=dict)
    density_flags: Dict[str, str] = field(default_factory=dict)  # "thin" / "ok" / "stuffed"

    # Convenience exports
    meta_keywords: str = ""         # comma-separated for <meta name="keywords">
    hashtags: List[str] = field(default_factory=list)       # #formatted
    slug_candidates: List[str] = field(default_factory=list)  # url-safe


# ---------------------------------------------------------------------------
# Sentence-BERT loader (used by KeyBERTManager)
# ---------------------------------------------------------------------------

def _default_keybert_path() -> str:
    """The registry's all-MiniLM-L6-v2 dir. Lazy on purpose: the engine's
    model registry walks the store on first touch, so importing this module
    must not pay for it."""
    from hugpy_engine.config.main import DEFAULT_PATHS
    return DEFAULT_PATHS["all-minilm-l6-v2"]


def _build_sentence_bert(model_path: Optional[str] = None):
    """
    Assemble a SentenceTransformer from components.
    Isolated here so the manager doesn't mix construction with lifecycle.
    """
    st = require("sentence_transformers", reason="needed by keybert backend")
    # sentence-transformers 5.x does NOT auto-import its `models` submodule on
    # `import sentence_transformers`, so `st.models` AttributeErrors — import it
    # explicitly. (require() already guaranteed the package is installed.)
    import sentence_transformers.models as st_models
    path = model_path or _default_keybert_path()

    word_model = st_models.Transformer(
        model_name_or_path=path,
        max_seq_length=256,
        do_lower_case=False,
    )
    pooling = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode="mean",
    )
    normalize = st_models.Normalize()

    return st.SentenceTransformer(
        modules=[word_model, pooling, normalize],
    )


# ---------------------------------------------------------------------------
# Manager: spaCy NLP
# ---------------------------------------------------------------------------

class SpacyManager(metaclass=SingletonMeta):

    def __init__(self):
        if not hasattr(self, "_ready"):
            spacy = require("spacy", reason="needed by spacy keyword backend")
            self.nlp = spacy.load("en_core_web_sm")
            self._ready = True


def get_nlp():
    return SpacyManager().nlp


# ---------------------------------------------------------------------------
# Manager: KeyBERT + sentence-BERT
# ---------------------------------------------------------------------------

class KeyBERTManager(metaclass=SingletonMeta):

    def __init__(self):
        if not hasattr(self, "_ready"):
            self._sbert = _build_sentence_bert()
            KeyBERT = require("keybert", reason="needed by keybert backend").KeyBERT
            self._keybert = KeyBERT(self._sbert)
            self._ready = True

    @property
    def sbert(self):
        return self._sbert

    @property
    def keybert(self):
        return self._keybert


def get_sbert():
    return KeyBERTManager().sbert


def get_keybert_instance():
    """Return the initialised KeyBERT model.  NOT named `get_keybert` — that
    import accessor lives in .imports and we never shadow it."""
    return KeyBERTManager().keybert


# ---------------------------------------------------------------------------
# Encoding / similarity (thin wrappers, no state)
# ---------------------------------------------------------------------------

def encode_sentences(
    sentences: List[str],
    *,
    model=None,
    model_path: Optional[str] = None,
):
    m = model or get_sbert()
    return m.encode(
        sentences,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def cosine_similarity(embeddings):
    cos_sim = get_sentence_transformers("cos_sim")
    return cos_sim(embeddings, embeddings)


# ---------------------------------------------------------------------------
# Back-end: KeyBERT extraction
# ---------------------------------------------------------------------------

def extract_keybert(req: KeywordRequest, *, model=None) -> List[Tuple[str, float]]:
    """Transformer-based keyword extraction.

    ``model`` is an already-built KeyBERT instance; None falls back to the
    module singleton (the default sentence-BERT). Dispatch passes one in so
    an explicit model_key is honored.
    """
    if not req.text:
        raise ValueError("No content provided for keyword extraction.")

    kw = model or get_keybert_instance()
    docs = req.text if isinstance(req.text, list) else [req.text]

    results = kw.extract_keywords(
        docs,
        keyphrase_ngram_range=req.keyphrase_ngram_range,
        stop_words=req.stop_words,
        top_n=req.top_n,
        use_mmr=req.use_mmr,
        diversity=req.diversity,
    )

    # extract_keywords returns List[Tuple] for single doc,
    # List[List[Tuple]] for multiple — normalise to flat list.
    if results and isinstance(results[0], list):
        flat: List[Tuple[str, float]] = []
        for batch in results:
            flat.extend(batch)
        return flat
    return results


# ---------------------------------------------------------------------------
# Back-end: spaCy rule-based extraction
# ---------------------------------------------------------------------------

_SPACY_POS = {"NOUN", "PROPN"}
_SPACY_ENT_LABELS = {"PERSON", "ORG", "GPE", "EVENT"}


def extract_spacy(req: KeywordRequest) -> List[str]:
    """Rule-based keyword extraction via POS tags + NER."""
    if not isinstance(req.text, str):
        raise ValueError(
            f"extract_spacy expects a string, got {type(req.text)}"
        )

    nlp = get_nlp()
    doc = nlp(req.text)

    word_counts = Counter(
        tok.text.lower()
        for tok in doc
        if tok.pos_ in _SPACY_POS and not tok.is_stop and len(tok.text) > 3
    )

    entity_counts = Counter(
        ent.text.lower()
        for ent in doc.ents
        if len(ent.text.split()) >= 2 and ent.label_ in _SPACY_ENT_LABELS
    )

    combined = entity_counts + word_counts
    return [kw for kw, _ in combined.most_common(req.top_n)]


# ---------------------------------------------------------------------------
# Keyword density
# ---------------------------------------------------------------------------

def keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    if not text:
        return {kw: 0.0 for kw in keywords}

    words = [
        w.strip(".,!?;:()\"'").lower()
        for w in re.split(r"\s+", text)
        if w.strip()
    ]
    total = len(words)
    if total == 0:
        return {kw: 0.0 for kw in keywords}

    return {kw: (words.count(kw.lower()) / total) * 100 for kw in keywords}


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def spacy_available() -> bool:
    return is_available("spacy")


def keybert_available() -> bool:
    return is_available("sentence_transformers") and is_available("keybert")


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

from hugpy_platform.choices import first_not_none as _resolve


def extract_keywords(
    text: str,
    *,
    preset: Optional[str] = None,
    top_n: Optional[int] = None,
    diversity: Optional[float] = None,
    use_mmr: Optional[bool] = None,
    keyphrase_ngram_range: Optional[Tuple[int, int]] = None,
    stop_words: Optional[str] = None,
    model=None,
) -> KeywordResult:
    """
    Run available back-ends, merge, deduplicate, compute density.

    If one backend is missing, the other still runs and the result's
    `backend_errors` dict tells you exactly what wasn't available.
    If *both* are missing, raises ImportError — there's nothing to do.
    """
    p = get_preset(preset) if preset else KeywordPreset()
    _d = {f.name: f.default for f in KeywordRequest.__dataclass_fields__.values()}

    req = KeywordRequest(
        text=text,
        top_n=_resolve(top_n, p.top_n, _d["top_n"]),
        diversity=_resolve(diversity, p.diversity, _d["diversity"]),
        use_mmr=_resolve(use_mmr, p.use_mmr, _d["use_mmr"]),
        stop_words=_resolve(stop_words, p.stop_words, _d["stop_words"]),
        keyphrase_ngram_range=_resolve(
            keyphrase_ngram_range, p.keyphrase_ngram_range, _d["keyphrase_ngram_range"]
        ),
    )

    result = KeywordResult()

    # -- spacy backend -----------------------------------------------------
    try:
        result.keywords_spacy = extract_spacy(req)
        result.backends_used.append("spacy")
    except ImportError as exc:
        result.backend_errors["spacy"] = str(exc)

    # -- keybert backend ---------------------------------------------------
    try:
        result.keywords_keybert = extract_keybert(req, model=model)
        result.backends_used.append("keybert")
    except ImportError as exc:
        result.backend_errors["keybert"] = str(exc)

    # -- nothing worked? ---------------------------------------------------
    if not result.backends_used:
        missing = "; ".join(
            f"{k}: {v}" for k, v in result.backend_errors.items()
        )
        raise ImportError(
            f"No keyword backends available. Install at least one:\n{missing}"
        )

    # -- merge whatever we got ---------------------------------------------
    spacy_lower = [k.lower() for k in result.keywords_spacy]
    keybert_lower = [p_kw.lower() for p_kw, _ in result.keywords_keybert]

    result.combined = list(dict.fromkeys(
        spacy_lower + keybert_lower
    ))[:req.top_n]

    result.density = keyword_density(text, result.combined)

    return result


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _to_slug(phrase: str) -> str:
    """'machine learning pipeline' → 'machine-learning-pipeline'"""
    return re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")


def _to_hashtag(phrase: str) -> str:
    """'machine learning' → '#machinelearning'"""
    return "#" + re.sub(r"[^a-z0-9]", "", phrase.lower())


def _classify_density(
    pct: float,
    min_d: Optional[float],
    max_d: Optional[float],
) -> str:
    if min_d is not None and pct < min_d:
        return "thin"
    if max_d is not None and pct > max_d:
        return "stuffed"
    return "ok"


# ---------------------------------------------------------------------------
# Refine pipeline — preset-driven post-processing for SEO / metadata
# ---------------------------------------------------------------------------

def refine_keywords(
    text: str,
    *,
    preset: str = "seo",
    top_n: Optional[int] = None,
    diversity: Optional[float] = None,
    use_mmr: Optional[bool] = None,
    keyphrase_ngram_range: Optional[Tuple[int, int]] = None,
    stop_words: Optional[str] = None,
    min_density: Optional[float] = None,
    max_density: Optional[float] = None,
    min_score: Optional[float] = None,
    max_words_per_phrase: Optional[int] = None,
    model=None,
) -> RefinedResult:
    """
    Extract → filter → classify → format.

    Wraps extract_keywords with preset-aware post-processing:
      - Drops keywords below min_score (keybert confidence)
      - Drops keywords below min_density (not actually in the text)
      - Flags keywords above max_density (potential stuffing)
      - Caps phrase length via max_words_per_phrase
      - Splits into primary / secondary / dropped
      - Generates meta_keywords, hashtags, and slug_candidates

    Explicit kwargs override preset values, same resolution as summarizers.
    """
    p = get_preset(preset) if preset else KeywordPreset()

    # -- resolve post-processing params ------------------------------------
    min_d = min_density if min_density is not None else p.min_density
    max_d = max_density if max_density is not None else p.max_density
    min_s = min_score if min_score is not None else p.min_score
    max_w = max_words_per_phrase if max_words_per_phrase is not None else p.max_words_per_phrase

    # -- extract -----------------------------------------------------------
    raw = extract_keywords(
        text,
        preset=preset,
        top_n=top_n,
        diversity=diversity,
        use_mmr=use_mmr,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        model=model,
    )

    # -- build a score lookup from keybert results -------------------------
    score_map: Dict[str, float] = {
        kw.lower(): score for kw, score in raw.keywords_keybert
    }

    # -- filter and classify -----------------------------------------------
    result = RefinedResult(raw=raw, preset_used=preset)
    result.density = dict(raw.density)

    for kw in raw.combined:
        score = score_map.get(kw, 1.0)  # spacy-only keywords get a pass
        d = raw.density.get(kw, 0.0)
        reasons: List[str] = []

        # score gate
        if min_s is not None and score < min_s and kw in score_map:
            reasons.append(f"score {score:.2f} < {min_s}")

        # phrase length gate
        if max_w is not None and len(kw.split()) > max_w:
            reasons.append(f"{len(kw.split())} words > max {max_w}")

        # density gate
        flag = _classify_density(d, min_d, max_d)
        result.density_flags[kw] = flag

        if flag == "thin" and min_d is not None:
            reasons.append(f"density {d:.2f}% < {min_d}%")

        if flag == "stuffed" and max_d is not None:
            reasons.append(f"density {d:.2f}% > {max_d}% (stuffing)")

        # route
        if reasons:
            result.dropped.append(kw)
        elif score >= score_map.get(kw, 0.5) or kw not in score_map:
            result.primary.append(kw)
        else:
            result.secondary.append(kw)

    # -- convenience exports -----------------------------------------------
    all_kept = result.primary + result.secondary

    result.meta_keywords = ", ".join(all_kept)
    result.hashtags = [_to_hashtag(kw) for kw in all_kept]
    result.slug_candidates = [_to_slug(kw) for kw in result.primary[:5]]

    return result
