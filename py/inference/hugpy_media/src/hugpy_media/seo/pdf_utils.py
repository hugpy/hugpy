"""
PDF SEO pipeline.

Two scopes:
    - full document  (all pages joined)
    - single page    (by index)

Each scope produces a summary + refined keywords bundled into a
PDFSeoResult so callers get one object, not two loose values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from hugpy_media.keywords.keybert_model import (
    KeywordPreset,
    RefinedResult,
    available_presets as kw_presets,
    refine_keywords,
    register_preset as register_keyword_preset,
)
from hugpy_media.schemas.summarizer_schemas import SummaryPreset
from hugpy_media.summarizers.summarizers import (
    register_preset as register_summary_preset,
    summarize_t5,
)


# ---------------------------------------------------------------------------
# Page-scoped keyword preset
# ---------------------------------------------------------------------------
# "seo" is tuned for full documents (min_density=0.3 etc).
# A single page has ~200-500 words — density thresholds need to be
# much looser or keywords get dropped for being "thin" when they're
# actually perfectly relevant at page scale.

try:
    register_keyword_preset(
        "page_seo",
        KeywordPreset(
            top_n=10,
            diversity=0.6,
            keyphrase_ngram_range=(1, 2),
            min_density=0.0,        # don't penalise sparse pages
            max_density=8.0,        # short text = naturally higher density
            min_score=0.15,
            max_words_per_phrase=2,
        ),
    )
except KeyError:
    pass  # already registered from a previous import


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class PDFSeoResult:
    scope: str = ""
    text: str = ""
    summary: str = ""
    keywords: Optional[RefinedResult] = None

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "text": self.text,
            "summary": self.summary,
            "keywords": self.keywords.to_dict() if self.keywords is not None else None,
        }


@dataclass
class PDFSeoReport:
    full: Optional[PDFSeoResult] = None
    pages: List[PDFSeoResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "full": self.full.to_dict() if self.full is not None else None,
            "pages": [p.to_dict() for p in self.pages],
        }


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _list_page_text_paths(texts_dir: str) -> List[str]:
    """
    Return sorted text file paths, excluding 'left'/'right' variants.
    """
    if not os.path.isdir(texts_dir):
        return []
    return sorted(
        os.path.join(texts_dir, f)
        for f in os.listdir(texts_dir)
        if f.endswith(".txt")
        and "left" not in f
        and "right" not in f
    )


def get_texts_dir(pdf_dir: str) -> str:
    """Convention: text files live in <pdf_dir>/texts/"""
    return os.path.join(pdf_dir, "texts")


def get_page_num_str(i: int) -> str:
    return str(i).zfill(3)


def load_all_texts(pdf_dir: str) -> List[str]:
    """Load every page's text content as a list of strings."""
    paths = _list_page_text_paths(get_texts_dir(pdf_dir))
    return [_read_file(p) for p in paths]


def load_full_text(pdf_dir: str) -> str:
    """All pages joined into one string."""
    return "\n".join(load_all_texts(pdf_dir))


def load_page_text(pdf_dir: str, page_index: int) -> Optional[str]:
    """
    Load a single page's text by index.
    Returns None if the page file doesn't exist.
    """
    texts_dir = get_texts_dir(pdf_dir)
    page_str = get_page_num_str(page_index)

    candidates = [
        os.path.join(texts_dir, f)
        for f in os.listdir(texts_dir)
        if f.endswith(f"{page_str}.txt")
        and "left" not in f
        and "right" not in f
    ]
    if not candidates:
        return None
    return _read_file(candidates[0])


# ---------------------------------------------------------------------------
# Single-scope analysis
# ---------------------------------------------------------------------------

def _analyze(
    text: str,
    scope: str,
    *,
    summary_preset: str = "article",
    keyword_preset: str = "seo",
    input_policy:str="allow"
) -> PDFSeoResult:
    """Run summary + keywords on a single block of text."""
    result = PDFSeoResult(scope=scope, text=text)
    result.summary = summarize_t5(text, preset=summary_preset)
    result.keywords = refine_keywords(text, preset=keyword_preset)
    return result


# ---------------------------------------------------------------------------
# Public API — document level
# ---------------------------------------------------------------------------

def analyze_full(pdf_dir: str) -> PDFSeoResult:
    """Summary + SEO keywords for the entire document."""
    text = load_full_text(pdf_dir)
    return _analyze(text, scope="full", summary_preset="article", keyword_preset="seo")


# ---------------------------------------------------------------------------
# Public API — page level
# ---------------------------------------------------------------------------

def analyze_page(pdf_dir: str, page_index: int) -> PDFSeoResult:
    """Summary + SEO keywords for a single page."""
    text = load_page_text(pdf_dir, page_index)
    if text is None:
        raise FileNotFoundError(
            f"No text file found for page {page_index} in {get_texts_dir(pdf_dir)}"
        )
    return _analyze(
        text,
        scope=f"page:{page_index}",
        summary_preset="brief",         # one page ≠ article-length
        keyword_preset="seo",      # density thresholds scaled for page length
    )


# ---------------------------------------------------------------------------
# Public API — full report
# ---------------------------------------------------------------------------

def analyze_pdf(pdf_dir: str) -> PDFSeoReport:
    """
    Full document analysis: document-level + every page.

    Returns a PDFSeoReport with .full and .pages[] so callers can
    inspect any scope without re-running inference.
    """
    report = PDFSeoReport()
    report.full = analyze_full(pdf_dir)

    page_texts = load_all_texts(pdf_dir)
    for i, text in enumerate(page_texts):
        report.pages.append(
            _analyze(
                text,
                scope=f"page:{i}",
                summary_preset="brief",
                keyword_preset="page_seo",
            )
        )

    return report
analyze_media_text = _analyze
