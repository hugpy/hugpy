"""k120 — GEM RADAR: the model nobody's card is looking for yet.

Every saved-search card asks the Hub a question the operator already knew to
ask. That is the limit of it: a card for "qwen3" finds Qwen derivatives, and a
9B from a lab nobody has heard of that three people on r/LocalLLaMA are quietly
raving about is invisible to all four cards at once. The radar is the pass that
catches it.

IT COSTS NOTHING EXTRA. The community scan (``community.py``) already pulled
those subreddit listings and HN hits and left them in the disk cache. The radar
RE-READS THE SAME CACHED BODIES — no second round of requests, no extra
politeness budget spent — and looks for something different in them: model
NAMES, whoever they belong to.

    1. harvest    every ``huggingface.co/<org>/<repo>`` link and every bare
                  ``org/repo`` or model-shaped token in the cached text
    2. filter     drop anything already known to a card (by repo tail, so a
                  quant of an incumbent is not "discovered"), drop obvious
                  non-models (paths, filenames, handles)
    3. resolve    fuzzy names go through one HF search each to become a real
                  repo id, or stay unresolved and say so
    4. rank       by the same recency-weighted ``heat`` a candidate gets, so
                  radar rows and candidate rows are on one scale

The output is a ``radar`` list on the discovery run. A card with ``radar: true``
publishes it; adopting a hit as a candidate next run is the operator's call,
because the radar is a TIP, not a screen — it has done no VRAM maths, checked no
licence and loaded nothing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from hugpy_curation.dossier.community import heat_of
from hugpy_curation.dossier.dossier import Mention, Source, utc_now
from hugpy_curation.dossier.fetch import cached_urls, read_cache

logger = logging.getLogger(__name__)

#: A hugging face link is the strongest possible signal: somebody posted the
#: actual repo.
_HF_LINK = re.compile(
    r"huggingface\.co/([A-Za-z0-9][\w.\-]{0,38})/([\w.\-]{2,80})", re.I)

#: A bare ``org/repo`` in prose. Only accepted when the right half looks like a
#: model (see ``_model_shaped``) — otherwise every file path in every comment
#: becomes a candidate.
_BARE_REPO = re.compile(
    r"(?<![\w/.])([A-Za-z0-9][\w.\-]{1,38})/([A-Za-z0-9][\w.\-]{2,80})"
    r"(?![\w/])")

#: A model-shaped bare token: a family-ish name carrying a parameter size.
_BARE_NAME = re.compile(
    r"\b([A-Za-z][\w.]{1,20}(?:[-_][A-Za-z0-9.]{1,20}){0,4}"
    r"[-_]\d{1,3}(?:\.\d)?[bB])\b")

#: Right-hand sides that mark a repo id as a model rather than a path.
_MODEL_TOKENS = re.compile(
    r"\d{1,3}(?:\.\d)?[bB]\b|gguf|instruct|chat|awq|gptq|exl2|mlx|"
    r"qwen|llama|mistral|gemma|phi-?\d|deepseek|glm|yi-|command-r|"
    r"stable-?diffusion|flux|wan\d|sdxl|whisper|distil|abliterat|uncensor",
    re.I)

#: Common false positives from prose: github paths, docs, handles.
_NOT_A_REPO = re.compile(
    r"^(?:https?|www|github|gitlab|docs?|blog|www2|api|src|lib|usr|etc|var|"
    r"home|tmp|opt|c|r|u|comments?|wiki|images?|files?)$", re.I)

#: A radar hit needs at least this much heat to be worth an operator's minute.
MIN_HEAT: float = 0.6

#: Cap what one run publishes. The radar is a shortlist, not a firehose.
MAX_HITS: int = 12


@dataclass(slots=True)
class RadarHit:
    """A model the cards are not asking about, and the posts that named it."""
    name: str                       # what was found in the text
    hub_id: str | None = None       # resolved repo id, when one was found
    resolved: bool = False
    heat: float = 0.0
    mentions: tuple[Mention, ...] = ()
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mentions"] = [m.to_dict() for m in self.mentions]
        return d


# ---------------------------------------------------------------------------
# Re-reading the night's cached pulls
# ---------------------------------------------------------------------------


def _rows_from_reddit(payload: Mapping[str, Any], url: str) -> list[Mention]:
    sub = ""
    m = re.search(r"/r/([^/]+)/", url)
    if m:
        sub = m.group(1)
    out = []
    for child in ((payload.get("data") or {}).get("children") or []):
        data = (child or {}).get("data") or {}
        permalink = data.get("permalink")
        out.append(Mention(
            source=f"reddit:{sub}" if sub else "reddit",
            url=(f"https://www.reddit.com{permalink}" if permalink
                 else data.get("url")),
            title=data.get("title"),
            snippet=(data.get("selftext") or "")[:1500],
            author=data.get("author"), ts=data.get("created_utc"),
            score=data.get("score")))
    return out


def _rows_from_hn(payload: Mapping[str, Any]) -> list[Mention]:
    out = []
    for hit in (payload.get("hits") or []):
        object_id = hit.get("objectID")
        text = hit.get("comment_text") or hit.get("story_text") or ""
        out.append(Mention(
            source="hackernews",
            url=(f"https://news.ycombinator.com/item?id={object_id}"
                 if object_id else hit.get("url")),
            title=hit.get("title") or hit.get("story_title"),
            snippet=re.sub(r"<[^>]+>", " ", text)[:1500],
            author=hit.get("author"), ts=hit.get("created_at_i"),
            score=hit.get("points")))
    return out


def read_cached_mentions(urls: Sequence[str] | None = None) -> list[Mention]:
    """Every post already on disk from this night's community pulls.

    Reads the CACHE, never the network. ``urls=None`` means "everything the
    cache holds for the sources we know how to parse"."""
    targets = list(urls) if urls is not None else (
        cached_urls("https://www.reddit.com/")
        + cached_urls("https://hn.algolia.com/"))
    rows: list[Mention] = []
    for url in targets:
        hit = read_cache(url)
        if hit is None or not hit.ok:
            continue
        try:
            payload = json.loads(hit.text)
        except ValueError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if "hn.algolia.com" in url:
            rows.extend(_rows_from_hn(payload))
        elif "reddit.com" in url:
            rows.extend(_rows_from_reddit(payload, url))
    return rows


# ---------------------------------------------------------------------------
# Harvesting names
# ---------------------------------------------------------------------------


def _model_shaped(repo: str) -> bool:
    return bool(_MODEL_TOKENS.search(repo))


def harvest_names(mention: Mention) -> set[str]:
    """Model names in one post. Repo ids keep their ``org/repo`` form; bare
    names are returned as typed."""
    text = " ".join(filter(None, (mention.title, mention.snippet)))
    found: set[str] = set()
    for m in _HF_LINK.finditer(text):
        found.add(f"{m.group(1)}/{m.group(2)}".rstrip("/.,"))
    for m in _BARE_REPO.finditer(text):
        org, repo = m.group(1), m.group(2)
        if _NOT_A_REPO.match(org) or _NOT_A_REPO.match(repo):
            continue
        if not _model_shaped(repo):
            continue
        found.add(f"{org}/{repo}")
    for m in _BARE_NAME.finditer(text):
        name = m.group(1)
        if len(name) >= 6 and not _NOT_A_REPO.match(name):
            found.add(name)
    return found


def _tail(name: str) -> str:
    return name.split("/")[-1].lower()


def _known_tails(known: Iterable[str]) -> set[str]:
    """Repo tails already covered by a card, plus their de-packaged forms, so
    ``bartowski/Foo-GGUF`` does not radar-report ``Foo``."""
    from hugpy_curation.dossier.community import aliases_for
    out: set[str] = set()
    for item in known or ():
        if not item:
            continue
        out.add(_tail(item))
        for alias in aliases_for(str(item)):
            out.add(alias.lower())
    return out


def is_known(name: str, skip: set[str]) -> bool:
    """Is this harvested name just an incumbent wearing different packaging?

    Three ways it can be, and all three showed up in the first ten minutes of
    real posts:

      * the exact repo tail (``bartowski/Foo-GGUF`` -> ``foo-gguf``);
      * the SAME model repackaged by someone else
        (``someoneelse/Foo-GGUF`` when the card already tracks
        ``bartowski/Foo``) — so the candidate's own packaging suffixes are
        stripped before comparing;
      * a PREFIX of a tracked name (``Foo-8B`` from a post about
        ``Foo-8B-Coder``) — the bare-name harvester finds these constantly, and
        without this rule every tracked model also radars as its own stem.

    Deliberately one-directional-plus: a strict prefix match on either side at a
    separator boundary counts, an unanchored substring does not, because
    ``Kestrel-9B`` and ``Kestrel-9B-Coder`` being the same family is useful and
    ``9B`` matching everything is not.
    """
    from hugpy_curation.dossier.community import aliases_for
    candidates = {_tail(name).lower()}
    candidates.update(a.lower() for a in aliases_for(name))
    for cand in candidates:
        if not cand:
            continue
        if cand in skip:
            return True
        for tracked in skip:
            if not tracked:
                continue
            longer, shorter = ((tracked, cand) if len(tracked) > len(cand)
                               else (cand, tracked))
            if len(shorter) >= 6 and longer.startswith(shorter) and \
                    longer[len(shorter)] in "-_. ":
                return True
    return False


def resolve_name(name: str, api: Any = None) -> tuple[str | None, str]:
    """Fuzzy name -> a real repo id, via ONE hub search. ``(hub_id, why)``.

    An ``org/repo`` that already looks like an id is returned as-is without a
    request. A name that resolves to nothing stays unresolved and says so — the
    radar publishes it anyway, because "three people are talking about
    something called Genesis-9B and we cannot find it" is still useful."""
    if "/" in name:
        return name, "posted as a repo id"
    if api is None:
        try:
            from hugpy_curation.review.screen import _hf_api
            api = _hf_api()
        except Exception as exc:                    # noqa: BLE001
            return None, f"no hub client ({type(exc).__name__})"
    try:
        hits = list(api.list_models(search=name, sort="downloads", limit=3,
                                    full=False))
    except Exception as exc:                        # noqa: BLE001
        return None, f"hub search failed ({type(exc).__name__}: {exc})"
    for hit in hits:
        model_id = getattr(hit, "modelId", None) or getattr(hit, "id", None)
        if model_id and name.lower().replace("-", "") in \
                _tail(model_id).replace("-", ""):
            return model_id, f"hub search matched {name!r}"
    return None, f"no hub repo matches {name!r}"


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def scan(known: Iterable[str] = (), *, urls: Sequence[str] | None = None,
         mentions: Sequence[Mention] | None = None, api: Any = None,
         resolve: bool = True, min_heat: float = MIN_HEAT,
         limit: int = MAX_HITS) -> tuple[tuple[RadarHit, ...], Source]:
    """The gem radar. ``(hits, provenance)``.

    ``known`` is every hub id any card has already screened — pass the store's
    distinct hub ids. ``mentions`` is the injection seam for tests; in
    production it is left None and the cached pulls are read."""
    rows = list(mentions) if mentions is not None else read_cached_mentions(urls)
    if not rows:
        return (), Source(kind="radar", fetched_at=utc_now(), ok=True,
                          detail="no cached community pulls to scan — the "
                                 "radar runs on what the mention scan already "
                                 "fetched")
    skip = _known_tails(known)
    by_name: dict[str, list[Mention]] = {}
    for mention in rows:
        for name in harvest_names(mention):
            if is_known(name, skip):
                continue
            by_name.setdefault(name, []).append(mention)

    # Collapse a repo id and its bare name onto the id (people post both).
    ids = {n for n in by_name if "/" in n}
    for bare in [n for n in by_name if "/" not in n]:
        match = next((i for i in ids if _tail(i).lower() == bare.lower()), None)
        if match:
            by_name[match].extend(by_name.pop(bare))

    hits: list[RadarHit] = []
    for name, found in by_name.items():
        unique = list({(m.url or id(m)): m for m in found}.values())
        heat = heat_of(unique)
        if heat < min_heat:
            continue
        hits.append(RadarHit(name=name, heat=heat,
                             mentions=tuple(unique[:6]),
                             why=f"named in {len(unique)} cached post(s) that "
                                 f"no card is tracking"))
    hits.sort(key=lambda h: (-h.heat, h.name))
    hits = hits[:limit]

    if resolve:
        for hit in hits:
            hub_id, why = resolve_name(hit.name, api=api)
            hit.hub_id, hit.resolved = hub_id, bool(hub_id)
            hit.why = f"{hit.why}; {why}"

    return tuple(hits), Source(
        kind="radar", fetched_at=utc_now(), ok=True,
        detail=f"scanned {len(rows)} cached posts, {len(by_name)} untracked "
               f"names, {len(hits)} above heat {min_heat}")


__all__ = ["MAX_HITS", "MIN_HEAT", "RadarHit", "harvest_names", "is_known",
           "read_cached_mentions", "resolve_name", "scan"]
