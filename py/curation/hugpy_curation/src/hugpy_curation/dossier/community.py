"""k120 — community intelligence: what people who ran it are actually saying.

The operator's brief: "scraping social media, AI forums, reddits, youtube
transcriptions — all of these things to keep the edge on a gem that exists."
The reason is real and specific. A repo's download count tells you what was
POPULAR three months ago. A thread on r/LocalLLaMA from Tuesday tells you that
the new quant is broken above 8k context, that it needs a setuptools pin, or
that a 9B nobody has heard of is beating models three times its size. That is
the edge, and it is nowhere in the metadata.

THE SOURCE INTERFACE
    Every source is one function with the same signature::

        source(query: MentionQuery) -> SourceReading

    returning typed :class:`Mention` rows and a :class:`Source` provenance row
    that is filled in whether or not anything was found. Adding YouTube or a
    forum later means writing one function and adding it to :data:`SOURCES` —
    nothing else in the package changes. That is why the interface exists
    before all the sources do.

WHAT IS LIVE HERE (k120)
    * ``reddit``          public ``search.json`` on r/LocalLLaMA,
                          r/StableDiffusion, r/MachineLearning
    * ``hackernews``      the Algolia search API
    * ``hf-discussions``  the Hub's own discussion list for the repo
    * ``youtube``         INTERFACE ONLY. Implemented and wired, but it needs
                          ``youtube-transcript-api``; without it the source
                          answers a recorded ``unavailable`` and the nightly
                          is never delayed by it. (k121.)

POLITENESS IS NOT OPTIONAL. Public JSON endpoints only, a real User-Agent, one
request per host per two seconds, a 20-hour disk cache (the timer is nightly),
and nothing that requires a login. See ``fetch.py`` — every rule lives there.

HEAT is recency-weighted on purpose: twelve mentions from last March is a model
that HAD a moment; three from this week is a model having one now. The half-life
is 30 days and the formula is in :func:`heat_of` where an operator can read it.
"""
from __future__ import annotations

import logging
import math
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from hugpy_curation.dossier import llm
from hugpy_curation.dossier.dossier import Claim, Community, Mention, Source, utc_now
from hugpy_curation.dossier.fetch import fetch, read_cache

logger = logging.getLogger(__name__)

#: Where self-hosters actually talk. Ordered by signal for this fleet's
#: questions; a card can override the list.
DEFAULT_SUBREDDITS: tuple[str, ...] = (
    "LocalLLaMA", "StableDiffusion", "MachineLearning")

REDDIT_SEARCH = ("https://www.reddit.com/r/{sub}/search.json?"
                 "q={q}&restrict_sr=1&sort=new&limit={limit}&t=year")
HN_SEARCH = ("https://hn.algolia.com/api/v1/search?query={q}"
             "&tags=(story,comment)&hitsPerPage={limit}")
HF_DISCUSSIONS = "https://huggingface.co/{hub_id}/discussions"

#: Mentions past this count are noise for one candidate, and every extra row
#: is prompt budget the claim extractor does not have.
MAX_MENTIONS: int = 24

#: A mention older than this is history, not intelligence.
MAX_AGE_DAYS: float = 400.0

#: Heat half-life. 30 days: a month-old thread counts half as much as one from
#: today, a quarter-old one an eighth.
HALF_LIFE_DAYS: float = 30.0

CLAIM_KINDS: tuple[str, ...] = ("praise", "criticism", "benchmark", "quirk")

CLAIMS_PROMPT = """\
Below are public posts that mention the model "{model}". Extract what they \
actually CLAIM about it.

Return ONLY a JSON object: {{"claims": [{{"kind": "...", "text": "...", \
"quote": "...", "url": "..."}}]}}

  kind      one of praise, criticism, benchmark, quirk
            (quirk = an operational gotcha: a version pin, a broken quant, a \
template requirement, a VRAM surprise)
  text      one short sentence in your own words
  quote     the EXACT words from the post that support it, verbatim
  url       the url of the post the quote came from, copied from the list

Rules: at most 8 claims. Every claim needs a real quote copied from a post \
below and that post's url. If the posts say nothing substantive about the \
model, return {{"claims": []}} — an empty list is a correct answer. Never \
invent a benchmark number.

POSTS
{posts}
"""


# ---------------------------------------------------------------------------
# The source interface
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MentionQuery:
    """What a source is asked for. ``aliases`` is what people actually TYPE:
    nobody writes "bartowski/Qwen3-8B-GGUF" in a reddit title."""
    hub_id: str
    aliases: tuple[str, ...] = ()
    limit: int = MAX_MENTIONS
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS
    api: Any = None                       # an HfApi, when the caller has one

    @property
    def primary(self) -> str:
        return self.aliases[0] if self.aliases else self.hub_id


@dataclass(slots=True)
class SourceReading:
    """One source's answer. ``sources`` is never empty — a source that found
    nothing still says it looked, and one that could not look says why."""
    mentions: tuple[Mention, ...] = ()
    sources: tuple[Source, ...] = ()
    raw_urls: tuple[str, ...] = ()        # what the radar re-reads


def aliases_for(hub_id: str) -> tuple[str, ...]:
    """Search strings for one repo, most specific first.

    A GGUF repo's tail is ``Qwen3.6-35B-A3B-Uncensored-Genesis-GGUF``; the
    conversation about it says "Genesis 35B" or "Qwen3.6-35B-A3B". So: the
    tail, the tail with the packaging suffixes stripped, and the tail with
    separators spaced. De-duplicated, short junk dropped."""
    tail = hub_id.split("/")[-1]
    out = [tail]
    stripped = re.sub(
        r"[-_.](?:GGUF|gguf|AWQ|GPTQ|EXL2|MLX|NVFP4|i1|imatrix|"
        r"[Qq]\d(?:_[A-Za-z0-9]+)*)$", "", tail)
    while stripped != tail and stripped:
        tail, stripped = stripped, re.sub(
            r"[-_.](?:GGUF|gguf|AWQ|GPTQ|EXL2|MLX|NVFP4|i1|imatrix|"
            r"[Qq]\d(?:_[A-Za-z0-9]+)*)$", "", stripped)
    if stripped and stripped not in out:
        out.append(stripped)
    spaced = re.sub(r"[-_]+", " ", out[-1]).strip()
    if spaced and spaced not in out:
        out.append(spaced)
    return tuple(a for a in dict.fromkeys(out) if len(a) >= 4)[:3]


def _age_days(ts: float | None) -> float | None:
    if not ts:
        return None
    return max(0.0, (time.time() - float(ts)) / 86400.0)


def _snippet(text: str | None, cap: int = 400) -> str | None:
    if not text:
        return None
    flat = " ".join(str(text).split())
    return flat[:cap] if flat else None


# ---------------------------------------------------------------------------
# Source: Reddit (public JSON)
# ---------------------------------------------------------------------------


def reddit_source(query: MentionQuery) -> SourceReading:
    """r/<sub>/search.json for each configured subreddit.

    Public, unauthenticated, one request per subreddit per candidate — and
    cached for the night, so the same candidate re-reviewed in the morning
    costs nothing. Reddit answers 403 to anonymous callers it does not like;
    that is a recorded ``unavailable``, not an exception."""
    mentions: list[Mention] = []
    sources: list[Source] = []
    urls: list[str] = []
    term = query.primary
    for sub in query.subreddits:
        url = REDDIT_SEARCH.format(sub=sub,
                                   q=urllib.parse.quote(f'"{term}"'),
                                   limit=min(25, query.limit))
        urls.append(url)
        result = fetch(url)
        if not result.ok:
            sources.append(Source.unavailable(
                f"reddit:{sub}", result.detail or "no answer", url))
            continue
        payload = result.json() or {}
        children = ((payload.get("data") or {}).get("children") or [])
        found = 0
        for child in children:
            data = (child or {}).get("data") or {}
            permalink = data.get("permalink")
            mentions.append(Mention(
                source=f"reddit:{sub}",
                url=(f"https://www.reddit.com{permalink}" if permalink
                     else data.get("url")),
                title=_snippet(data.get("title"), 240),
                snippet=_snippet(data.get("selftext"), 600),
                author=data.get("author"),
                ts=data.get("created_utc"),
                score=data.get("score")))
            found += 1
        sources.append(Source(
            kind=f"reddit:{sub}", url=url, fetched_at=utc_now(), ok=True,
            detail=(f"{found} posts" if found else "no posts mention this "
                                                  "model in the last year")))
    return SourceReading(tuple(mentions), tuple(sources), tuple(urls))


# ---------------------------------------------------------------------------
# Source: Hacker News (Algolia)
# ---------------------------------------------------------------------------


def hackernews_source(query: MentionQuery) -> SourceReading:
    """HN's Algolia index over stories AND comments — the comments are where
    the operational detail lives."""
    url = HN_SEARCH.format(q=urllib.parse.quote(query.primary),
                           limit=min(20, query.limit))
    result = fetch(url)
    if not result.ok:
        return SourceReading((), (Source.unavailable(
            "hackernews", result.detail or "no answer", url),), (url,))
    hits = (result.json() or {}).get("hits") or []
    mentions = []
    for hit in hits:
        object_id = hit.get("objectID")
        text = hit.get("comment_text") or hit.get("story_text")
        mentions.append(Mention(
            source="hackernews",
            url=(f"https://news.ycombinator.com/item?id={object_id}"
                 if object_id else hit.get("url")),
            title=_snippet(hit.get("title") or hit.get("story_title"), 240),
            snippet=_snippet(re.sub(r"<[^>]+>", " ", text or ""), 600),
            author=hit.get("author"),
            ts=hit.get("created_at_i"),
            score=hit.get("points")))
    return SourceReading(
        tuple(mentions),
        (Source(kind="hackernews", url=url, fetched_at=utc_now(), ok=True,
                detail=f"{len(mentions)} hits" if mentions
                else "no Hacker News item mentions this model"),),
        (url,))


# ---------------------------------------------------------------------------
# Source: the Hub's own discussions
# ---------------------------------------------------------------------------


def hf_discussions_source(query: MentionQuery) -> SourceReading:
    """The repo's discussion tab. Not "outside" research, but it is where a
    broken quant gets reported first and the author answers, so it belongs
    beside the forum chatter rather than in the metadata section."""
    url = HF_DISCUSSIONS.format(hub_id=query.hub_id)
    api = query.api
    if api is None:
        try:
            from hugpy_curation.review.screen import _hf_api
            api = _hf_api()
        except Exception as exc:                    # noqa: BLE001
            return SourceReading((), (Source.unavailable(
                "hf-discussions", f"no hub client ({type(exc).__name__})",
                url),), ())
    try:
        rows = list(api.get_repo_discussions(query.hub_id))[:query.limit]
    except Exception as exc:                        # noqa: BLE001
        return SourceReading((), (Source.unavailable(
            "hf-discussions", f"{type(exc).__name__}: {exc}", url),), ())
    mentions = []
    for row in rows:
        num = getattr(row, "num", None)
        created = getattr(row, "created_at", None)
        ts = None
        if created is not None:
            try:
                ts = created.timestamp()
            except Exception:                       # noqa: BLE001
                ts = None
        mentions.append(Mention(
            source="hf-discussions",
            url=f"{url}/{num}" if num else url,
            title=_snippet(getattr(row, "title", None), 240),
            snippet=_snippet(getattr(row, "status", None), 60),
            author=getattr(getattr(row, "author", None), "name", None)
            or (getattr(row, "author", None)
                if isinstance(getattr(row, "author", None), str) else None),
            ts=ts, score=None))
    return SourceReading(
        tuple(mentions),
        (Source(kind="hf-discussions", url=url, fetched_at=utc_now(), ok=True,
                detail=f"{len(mentions)} discussions" if mentions
                else "no discussions opened on this repo"),),
        ())


# ---------------------------------------------------------------------------
# Source: YouTube (interface complete, transcript dependency optional)
# ---------------------------------------------------------------------------


def youtube_source(query: MentionQuery) -> SourceReading:
    """Review videos and their transcripts.

    WIRED BUT NOT LIVE (k120). Pulling transcripts needs
    ``youtube-transcript-api``, which is not a dependency this package will
    ever hard-require — a nightly reviewer that cannot import must not stop
    reviewing. Without it, this returns one honest ``unavailable`` row and the
    dossier says so where an operator can see it. The signature and the return
    shape are final, so k121 fills in the body and nothing else moves."""
    try:
        import youtube_transcript_api  # noqa: F401
    except Exception:                               # noqa: BLE001
        return SourceReading((), (Source.unavailable(
            "youtube",
            "youtube-transcript-api is not installed in this venv — video "
            "review transcripts are not being read (k121)",
            "https://www.youtube.com/results?search_query="
            + urllib.parse.quote(query.primary)),), ())
    return SourceReading((), (Source.unavailable(
        "youtube",
        "youtube-transcript-api is present but the k120 build does not yet "
        "issue video searches — transcript ingest lands in k121", None),), ())


#: The registry. A card's ``community_sources`` names keys from here; an
#: unknown key is recorded as unavailable rather than silently ignored.
SOURCES: dict[str, Callable[[MentionQuery], SourceReading]] = {
    "reddit": reddit_source,
    "hackernews": hackernews_source,
    "hf-discussions": hf_discussions_source,
    "youtube": youtube_source,
}

#: What runs when a card does not say. YouTube is in the list deliberately —
#: it costs one import attempt and leaves an honest row explaining itself.
DEFAULT_SOURCES: tuple[str, ...] = ("reddit", "hackernews", "hf-discussions",
                                    "youtube")


# ---------------------------------------------------------------------------
# Scoring and claim extraction
# ---------------------------------------------------------------------------


def relevant(mention: Mention, aliases: Sequence[str]) -> bool:
    """Does this row actually name the model?

    Search engines are generous. A post that matched on "Qwen" alone is not a
    mention of THIS repo, and letting it through would inflate heat and feed
    the claim extractor quotes about a different model entirely."""
    hay = " ".join(filter(None, (mention.title, mention.snippet))).lower()
    return any(a.lower() in hay for a in aliases if a)


def heat_of(mentions: Iterable[Mention]) -> float:
    """Recency-weighted mention score.

        heat = Σ  0.5^(age_days / 30)  ×  (1 + log10(1 + upvotes))

    So one enthusiastic thread from this week outweighs five stale ones, and a
    post with no score still counts as 1. Rounded to 3 places; a model nobody
    has mentioned scores exactly 0.0, which is a fact and not a penalty."""
    total = 0.0
    for m in mentions:
        age = _age_days(m.ts)
        if age is None:
            weight = 0.25                    # undated: counted, discounted
        elif age > MAX_AGE_DAYS:
            continue
        else:
            weight = 0.5 ** (age / HALF_LIFE_DAYS)
        score = max(0, int(m.score or 0))
        total += weight * (1.0 + math.log10(1 + score))
    return round(total, 3)


def _posts_block(mentions: Sequence[Mention], cap: int = 12) -> str:
    lines = []
    for i, m in enumerate(mentions[:cap], start=1):
        body = " ".join(filter(None, (m.title, m.snippet)))[:500]
        lines.append(f"[{i}] source={m.source} url={m.url}\n{body}")
    return "\n\n".join(lines)


def extract_claims(model_name: str, mentions: Sequence[Mention],
                   dispatch: Callable[[str, str, int], str] | None = None,
                   ) -> tuple[tuple[Claim, ...], str | None, str]:
    """``(claims, model_id, detail)``. Never raises.

    A claim whose quote does not appear in ANY fetched post is DROPPED. That
    check is the whole reason the quote field exists: it is a cheap, local
    guard against the summarizer inventing community consensus, and it costs
    one substring scan per claim."""
    if not mentions:
        return (), None, "no mentions to read"
    prompt = CLAIMS_PROMPT.format(model=model_name,
                                  posts=_posts_block(mentions))
    text, model, detail = llm.ask(prompt, max_tokens=800, dispatch=dispatch)
    if not text:
        return (), model, detail
    parsed = llm.extract_json(text)
    rows = (parsed or {}).get("claims") if isinstance(parsed, Mapping) else None
    if not isinstance(rows, list):
        return (), model, "the extractor did not return a claims list"
    corpus = " ".join(
        " ".join(filter(None, (m.title, m.snippet))) for m in mentions).lower()
    by_url = {m.url: m.source for m in mentions if m.url}
    out: list[Claim] = []
    for row in rows[:8]:
        if not isinstance(row, Mapping):
            continue
        quote = str(row.get("quote") or "").strip()
        body = str(row.get("text") or "").strip()
        if not body or not quote:
            continue
        needle = " ".join(quote.split()).lower()[:60]
        if needle and needle not in corpus:
            logger.info("dossier community: dropped an unsupported claim for "
                        "%s (quote not present in any fetched post)", model_name)
            continue
        kind = str(row.get("kind") or "").strip().lower()
        url = row.get("url") if isinstance(row.get("url"), str) else None
        out.append(Claim(
            kind=kind if kind in CLAIM_KINDS else "praise",
            text=body[:400], quote=quote[:400], url=url,
            source=by_url.get(url)))
    return tuple(out), model, detail


# ---------------------------------------------------------------------------
# The whole community section
# ---------------------------------------------------------------------------


def gather(hub_id: str, *, sources: Sequence[str] = DEFAULT_SOURCES,
           subreddits: Sequence[str] = DEFAULT_SUBREDDITS,
           api: Any = None, limit: int = MAX_MENTIONS, want_claims: bool = True,
           dispatch: Callable[[str, str, int], str] | None = None,
           ) -> tuple[Community, tuple[str, ...]]:
    """Every enabled source, folded into one :class:`Community`.

    Returns ``(community, raw_urls)`` — the urls are handed to ``radar.py``,
    which re-reads their CACHED bodies looking for models nobody has a card
    for yet. One set of requests, two answers."""
    query = MentionQuery(hub_id=hub_id, aliases=aliases_for(hub_id),
                         limit=limit, subreddits=tuple(subreddits), api=api)
    mentions: list[Mention] = []
    provenance: list[Source] = []
    urls: list[str] = []

    for name in sources:
        fn = SOURCES.get(name)
        if fn is None:
            provenance.append(Source.unavailable(
                name, f"unknown community source {name!r} — the card names a "
                      f"source this build does not have"))
            continue
        try:
            reading = fn(query)
        except Exception as exc:                    # noqa: BLE001 — a source
            # that throws is a source that is down, never a failed review.
            provenance.append(Source.unavailable(
                name, f"{type(exc).__name__}: {exc}"))
            continue
        mentions.extend(reading.mentions)
        provenance.extend(reading.sources)
        urls.extend(reading.raw_urls)

    kept = [m for m in mentions if relevant(m, query.aliases)]
    dropped = len(mentions) - len(kept)
    if dropped:
        provenance.append(Source(
            kind="filter", fetched_at=utc_now(), ok=True,
            detail=f"{dropped} search hits did not actually name "
                   f"{query.primary!r} and were dropped"))
    kept.sort(key=lambda m: (m.ts or 0), reverse=True)
    kept = kept[:limit]

    claims: tuple[Claim, ...] = ()
    generated_by = None
    if want_claims and kept:
        claims, generated_by, detail = extract_claims(
            query.primary, kept, dispatch=dispatch)
        if not claims and detail:
            provenance.append(Source(
                kind="claims", fetched_at=utc_now(), ok=bool(generated_by),
                detail=f"no claims extracted: {detail}"))

    return (Community(heat=heat_of(kept), mentions=tuple(kept), claims=claims,
                      sources=tuple(provenance), generated_by=generated_by,
                      model_generated=bool(claims)),
            tuple(dict.fromkeys(urls)))


__all__ = ["CLAIM_KINDS", "DEFAULT_SOURCES", "DEFAULT_SUBREDDITS",
           "HALF_LIFE_DAYS", "MAX_MENTIONS", "MentionQuery", "SOURCES",
           "SourceReading", "aliases_for", "extract_claims", "gather",
           "hackernews_source", "heat_of", "hf_discussions_source",
           "reddit_source", "relevant", "youtube_source"]
