"""k120 — research OUTSIDE the download source.

The operator's phrase was "research outside of the download sources
themselves". A repo's own metadata is a sales page: the publisher chose the
tags, the pipeline tag, and which benchmark numbers to print. So a dossier goes
looking for three more things, in descending order of trustworthiness:

  THE CARD, STRUCTURED (``cards.parse_card``). Still the publisher's words, but
  the parts a publisher does NOT put in the tags — the limitations section, the
  training-data notes, the benchmark tables — and structured so a judge can be
  handed them as claims rather than as prose.

  THE PAPER. arXiv ids linked from the card, resolved to a title and an
  abstract through arXiv's public export API. Best-effort with a short timeout:
  a paper that will not resolve is a :class:`PaperRef` with ``unavailable``
  filled in and no invented title.

  A WRITTEN SUMMARY (``research_notes``). One local model reads exactly the
  sources above — nothing else, and it is told so — and writes a few sentences
  about what this model is for and where it is weak. It is stamped with the
  model that wrote it and ``model_generated=True``, and the ``cited`` list
  names every source it was given. It is the only prose in a dossier and it is
  never allowed to look like a measurement.

WHAT THIS MODULE WILL NOT DO
    It will not fetch a page it was not offered as an API. It will not follow
    arbitrary links out of a README. And it will never let a slow or dead
    external service delay discovery: the whole external half of a dossier is
    wrapped so that the worst case is a handful of ``unavailable`` strings.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Mapping, Sequence

from hugpy_curation.dossier import cards, llm
from hugpy_curation.dossier.dossier import ExternalResearch, PaperRef, Source, utc_now
from hugpy_curation.dossier.fetch import fetch

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
ARXIV_ABS = "https://arxiv.org/abs/{id}"
HF_README = "https://huggingface.co/{hub_id}/raw/main/README.md"
HF_REPO = "https://huggingface.co/{hub_id}"

#: How many linked papers are worth resolving. A card that links twelve is
#: linking a reading list, not its own paper.
MAX_PAPERS: int = 3

_ATOM = "{http://www.w3.org/2005/Atom}"

RESEARCH_PROMPT = """\
You are writing a short research note about ONE candidate model for a small \
self-hosted GPU fleet's model reviewer.

You are given ONLY the sources below. Use nothing else. If a source does not \
say something, do not say it either. Never invent a benchmark number, a \
parameter count or a release date.

SOURCES
{sources}

Write 3-5 sentences of plain prose covering, only where the sources support it:
  * what this model is specialized FOR, and what it was derived from
  * what its author claims for it, marked as a claim ("the card claims ...")
  * any stated limitation, licence restriction or known quirk
  * whether it looks like a meaningful step over a general-purpose model of \
the same size, and say plainly if the sources do not let you tell

No headings, no bullet points, no preamble, no closing offer of help. Prose only.
"""


def fetch_readme(hub_id: str, api: Any = None) -> tuple[str | None, Source]:
    """The raw README. Tries the Hub client first (it honours a token and the
    local HF cache), then the public raw URL. Returns ``(text, source)`` and
    the source is filled in on BOTH branches."""
    url = HF_README.format(hub_id=hub_id)
    if api is not None:
        try:
            path = api.hf_hub_download(hub_id, "README.md")
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(), Source(kind="hf-readme", url=url,
                                         fetched_at=utc_now(), ok=True,
                                         detail="via huggingface_hub")
        except Exception as exc:                    # noqa: BLE001
            logger.info("dossier: hub README fetch failed for %s (%s)",
                        hub_id, type(exc).__name__)
    result = fetch(url)
    if result.ok and result.text.strip():
        return result.text, Source(kind="hf-readme", url=url,
                                   fetched_at=utc_now(), ok=True,
                                   detail="cached" if result.from_cache else "")
    return None, Source.unavailable(
        "hf-readme", result.detail or "no README.md in this repo", url)


def _parse_arxiv_feed(xml_text: str) -> dict[str, tuple[str, str]]:
    """``{arxiv_id: (title, abstract)}`` from an arXiv Atom feed."""
    out: dict[str, tuple[str, str]] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        m = re.search(r"abs/(\d{4}\.\d{4,5})", raw_id)
        if not m:
            continue
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        out[m.group(1)] = (title, summary)
    return out


def fetch_papers(ids: Sequence[str], limit: int = MAX_PAPERS
                 ) -> tuple[tuple[PaperRef, ...], tuple[Source, ...]]:
    """arXiv ids -> refs with abstracts, best-effort.

    One request for all of them (arXiv's ``id_list`` takes a comma list), so a
    card linking three papers costs one polite call, not three."""
    wanted = [i for i in dict.fromkeys(ids) if i][:limit]
    if not wanted:
        return (), ()
    url = ARXIV_API.format(ids=",".join(wanted), n=len(wanted))
    result = fetch(url)
    if not result.ok:
        refs = tuple(PaperRef(arxiv_id=i, url=ARXIV_ABS.format(id=i),
                              unavailable=result.detail or "arXiv unreachable")
                     for i in wanted)
        return refs, (Source.unavailable("arxiv", result.detail
                                         or "arXiv unreachable", url),)
    found = _parse_arxiv_feed(result.text)
    refs = []
    for arxiv_id in wanted:
        title, abstract = found.get(arxiv_id, (None, None))
        refs.append(PaperRef(
            arxiv_id=arxiv_id, title=title, url=ARXIV_ABS.format(id=arxiv_id),
            abstract=(abstract[:2000] if abstract else None),
            unavailable="" if title else "arXiv returned no entry for this id"))
    return tuple(refs), (Source(kind="arxiv", url=url, fetched_at=utc_now(),
                                ok=True,
                                detail="cached" if result.from_cache else ""),)


def _sources_block(hub_id: str, card, papers: Sequence[PaperRef],
                   payload: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The exact text the summarizer is given, plus the citation list. Built
    here (not in the prompt template) so ``cited`` and what the model actually
    read can never drift apart."""
    cited: list[str] = []
    blocks: list[str] = []

    meta = {k: payload.get(k) for k in
            ("pipeline_tag", "license", "downloads", "likes", "gated",
             "base_model", "last_modified") if payload.get(k) is not None}
    blocks.append(f"[1] Hugging Face repo metadata for {hub_id}: {meta}")
    cited.append(HF_REPO.format(hub_id=hub_id))

    if card is not None:
        n = len(blocks) + 1
        parts = [f"summary: {card.summary}" if card.summary else ""]
        if card.intended_use:
            parts.append(f"intended use: {card.intended_use[:600]}")
        if card.training_data:
            parts.append(f"training data: {card.training_data[:600]}")
        if card.limitations:
            parts.append(f"limitations: {card.limitations[:600]}")
        if card.benchmark_claims:
            claimed = "; ".join(
                f"{c.benchmark}={c.value}" for c in card.benchmark_claims[:12])
            parts.append(f"benchmark numbers CLAIMED by the card: {claimed}")
        blocks.append(f"[{n}] The model card README: "
                      + " | ".join(p for p in parts if p))
        cited.append(HF_README.format(hub_id=hub_id))

    for paper in papers:
        if not paper.abstract:
            continue
        n = len(blocks) + 1
        blocks.append(f"[{n}] Linked paper arXiv:{paper.arxiv_id} "
                      f"\"{paper.title}\": {paper.abstract[:1200]}")
        if paper.url:
            cited.append(paper.url)
    return "\n\n".join(blocks), cited


def write_research_notes(hub_id: str, card, papers: Sequence[PaperRef],
                         payload: Mapping[str, Any],
                         dispatch: Callable[[str, str, int], str] | None = None,
                         ) -> tuple[str | None, str | None, tuple[str, ...], str]:
    """``(notes, model_id, cited, detail)``. Never raises.

    ``notes`` is None whenever no model answered — and ``detail`` then says
    which of the honest reasons applies. It is NOT an error; a dossier with
    every measured field and no prose is still a good dossier."""
    block, cited = _sources_block(hub_id, card, papers, payload)
    prompt = RESEARCH_PROMPT.format(sources=block)
    text, model, detail = llm.ask(prompt, max_tokens=500, dispatch=dispatch)
    if not text:
        return None, model, tuple(cited), detail
    # Strip a leading restatement of the task some small models emit.
    cleaned = re.sub(r"^\s*(here (is|are)|sure[,!]?)[^\n.]*[.:]\s*", "", text,
                     flags=re.I)
    return cleaned.strip()[:2500], model, tuple(cited), detail


def build_research(hub_id: str, payload: Mapping[str, Any], *,
                   api: Any = None, enabled: bool = True,
                   want_notes: bool = True,
                   dispatch: Callable[[str, str, int], str] | None = None,
                   ) -> tuple[ExternalResearch, tuple[Source, ...], str | None]:
    """The whole external half. Returns ``(research, sources, readme_text)``.

    ``enabled=False`` (the card knob ``external_research``) short-circuits the
    NETWORK work and says so in ``unavailable`` — the operator turned it off,
    which is a different fact from "it failed", and the dossier records which.
    """
    if not enabled:
        return (ExternalResearch(unavailable=(
            "external research is disabled for this card "
            "(external_research: false)",)), (), None)

    sources: list[Source] = []
    unavailable: list[str] = []

    readme, readme_source = fetch_readme(hub_id, api=api)
    sources.append(readme_source)
    if not readme_source.ok:
        unavailable.append(f"model card: {readme_source.detail}")
    card = cards.parse_card(readme)

    ids = cards.arxiv_ids(readme or "")
    for tag in (payload.get("tags") or ()):
        if isinstance(tag, str) and tag.lower().startswith("arxiv:"):
            ids = ids + (tag.split(":", 1)[1],)
    papers, paper_sources = fetch_papers(ids)
    sources.extend(paper_sources)
    for paper in papers:
        if paper.unavailable:
            unavailable.append(f"arXiv:{paper.arxiv_id}: {paper.unavailable}")
    if not ids:
        unavailable.append("no paper is linked from this repo's card or tags")

    notes = model = None
    cited: tuple[str, ...] = ()
    if want_notes:
        notes, model, cited, detail = write_research_notes(
            hub_id, card, papers, payload, dispatch=dispatch)
        if notes is None:
            unavailable.append(f"research notes: {detail}")

    return (ExternalResearch(
        card=card, papers=papers, research_notes=notes,
        research_notes_model=model, model_generated=bool(notes),
        cited=cited, unavailable=tuple(unavailable)),
        tuple(sources), readme)


__all__ = ["ARXIV_ABS", "ARXIV_API", "HF_README", "MAX_PAPERS",
           "build_research", "fetch_papers", "fetch_readme",
           "write_research_notes"]
