"""k120 — reading a model card like a reviewer, not like a search index.

The Hub search gives a repo a ``pipeline_tag`` and a bag of tags. That is how
a model gets FOUND; it is not how a model gets CHOSEN. What actually decides
whether a candidate is worth the disk is in the README: what it was fine-tuned
FOR, which languages it claims, whether it wants a chat template, the table of
benchmark numbers its author is proud of, and — the part nobody reads and
everybody regrets — the limitations section.

So this module turns a README into a :class:`CardDigest` and a
:class:`Specialization`, with three rules:

  QUOTE, DON'T PARAPHRASE. ``finetune_focus`` holds the author's own sentences.
  Paraphrasing here would put words in the publisher's mouth and then feed them
  to a judge as if they were facts.

  A TABLE IS A CLAIM. ``benchmark_claims`` are labelled as coming from the
  card, and they never mix with the fleet's own measured scores. A card that
  says 92.1 on HumanEval is evidence about the AUTHOR, not about the model.

  NOTHING IS INFERRED FROM SILENCE. No languages in the card means
  ``languages=()``, not ``("en",)``. An empty tuple reads honestly in the UI;
  a guessed default is a lie that survives three hops.

``emphasis`` is the operator's "overall weights": one 0..1 number per
specialization with the strings that earned it. Evidence is additive and
capped, so a repo whose NAME, TAGS and CARD all say "coder" outranks one that
merely has the tag — which is the ranking a human does by eye anyway.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from hugpy_curation.dossier.dossier import (
    BenchmarkClaim,
    CardDigest,
    EmphasisWeight,
    Specialization,
)

#: Headings we care about, matched case-insensitively against the heading text.
_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "training_data": re.compile(
        r"train(ing)?\s*(data|set|corpus|mixture|details)|dataset|data\s*mix",
        re.I),
    "limitations": re.compile(
        r"limitation|bias|risk|caveat|known\s*issue|out[- ]of[- ]scope|"
        r"ethical", re.I),
    "intended_use": re.compile(
        r"intended\s*use|use\s*case|usage|what\s+it'?s?\s+for|applications?",
        re.I),
    "prompt_format": re.compile(
        r"prompt\s*(format|template)|chat\s*template|system\s*prompt|"
        r"instruction\s*format", re.I),
}

#: Headings under which a markdown table is read as a benchmark table.
_BENCH_HEADING = re.compile(
    r"benchmark|eval(uation)?s?\b|result|score|performance|leaderboard|"
    r"comparison|metric", re.I)

#: Column names that mark a table as carrying scores even without the heading.
_BENCH_COLUMN = re.compile(
    r"mmlu|gsm8k|humaneval|hellaswag|arc|winogrande|truthfulqa|gpqa|math|"
    r"bbh|ifeval|mt-?bench|score|acc(uracy)?|pass@|bleu|rouge|wer|cer|"
    r"perplexity|ppl|elo|f1", re.I)

#: A first column that NAMES the benchmark. Its presence settles the table's
#: orientation: `| Benchmark | Score |` is row-oriented even though "Score"
#: matches _BENCH_COLUMN, and reading it the other way would file every number
#: under the benchmark "Score".
_BENCH_ROW_HEADER = re.compile(
    r"^\**\s*(benchmark|task|dataset|eval(uation)?|metric|test|subset)",
    re.I)

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_BARE_RE = re.compile(r"\barxiv[:\s]+(\d{4}\.\d{4,5})\b", re.I)
_LINK_RE = re.compile(r"https?://[^\s)\]\"'>]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*#*\s*$", re.M)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW_RE = re.compile(r"^\s*\|[\s:\-\|]+\|\s*$")

#: Card phrases that state a purpose. The MATCHED SENTENCE is kept verbatim.
_FOCUS_RE = re.compile(
    r"[^.\n]*\b(fine[- ]?tuned?|finetun\w*|trained|distilled|merged|"
    r"specialis\w+|specializ\w+|optimi[sz]ed|designed|built|tuned)\b"
    r"\s+(?:on|for|to|from|with)\b[^.\n]*\.", re.I)

#: Domain -> (name/tag needles, card needles). Both halves are evidence; a hit
#: in the repo NAME is the strongest single signal a publisher gives us.
DOMAIN_SIGNALS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "code": (("coder", "code", "codellama", "starcoder", "deepseek-coder",
              "devstral"),
             ("code generation", "programming", "software engineering",
              "humaneval", "swe-bench", "fill-in-the-middle")),
    "math": (("math", "mathstral", "numina"),
             ("mathematical reasoning", "gsm8k", "olympiad", "theorem")),
    "reasoning": (("reason", "thinking", "r1", "cot", "distill", "o1"),
                  ("chain of thought", "step by step reasoning",
                   "reasoning traces", "thinking mode")),
    "vision": (("vl", "vision", "llava", "vlm", "-v-", "internvl", "minicpm-v"),
               ("image understanding", "visual question answering",
                "multimodal", "ocr", "document understanding")),
    "roleplay": (("rp", "roleplay", "story", "novel", "erp", "character",
                  "mythos", "gutenberg"),
                 ("role-play", "roleplay", "creative writing",
                  "storytelling", "narrative")),
    "medical": (("med", "bio", "clinical", "pubmed"),
                ("medical", "clinical", "biomedical", "healthcare")),
    "legal": (("legal", "law"), ("legal", "case law", "contract")),
    "translation": (("translat", "nllb", "opus-mt", "madlad"),
                    ("machine translation", "translate between")),
    "agentic": (("agent", "tool", "function", "react"),
                ("function calling", "tool use", "tool calling", "agentic",
                 "json mode")),
    "uncensored": (("uncensor", "abliterat", "heretic", "unfiltered", "dan",
                    "dolphin"),
                   ("uncensored", "abliterated", "refusal removed",
                    "unaligned")),
    "long-context": (("128k", "256k", "1m", "longcontext", "long-context"),
                     ("long context", "context window of", "needle in a")),
    "embedding": (("embed", "gte", "bge", "e5", "minilm"),
                  ("sentence embeddings", "retrieval", "semantic search")),
    "speech": (("whisper", "asr", "tts", "speech", "voice", "chatterbox"),
               ("speech recognition", "text to speech", "voice cloning")),
}

#: pipeline_tag -> coarse modality, so a UI row can say what KIND of thing this
#: is before it says anything else.
MODALITY_BY_TASK: dict[str, str] = {
    "text-generation": "text",
    "text2text-generation": "text",
    "text-summarization": "text",
    "image-text-to-text": "vision-language",
    "visual-question-answering": "vision-language",
    "image-to-text": "vision-language",
    "text-to-image": "image",
    "image-to-image": "image",
    "text-to-video": "video",
    "image-to-video": "video",
    "text-to-speech": "audio",
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
}

#: Architecture class name -> family. Purely a display grouping; an unknown
#: architecture keeps its own name rather than being forced into a bucket.
ARCH_FAMILIES: tuple[tuple[str, str], ...] = (
    ("qwen3", "qwen3"), ("qwen2", "qwen2"), ("qwen", "qwen"),
    ("llama", "llama"), ("mistral", "mistral"), ("mixtral", "mixtral"),
    ("gemma", "gemma"), ("phi", "phi"), ("glm", "glm"), ("deepseek", "deepseek"),
    ("falcon", "falcon"), ("gpt_neox", "neox"), ("gptneox", "neox"),
    ("starcoder", "starcoder"), ("cohere", "command-r"), ("olmo", "olmo"),
    ("stablelm", "stablelm"), ("internlm", "internlm"), ("yi", "yi"),
    ("minicpm", "minicpm"), ("lfm", "lfm"), ("whisper", "whisper"),
    ("t5", "t5"), ("bert", "bert"), ("clip", "clip"), ("unet", "diffusion"),
    ("stablediffusion", "diffusion"), ("flux", "flux"), ("wan", "wan"),
)


# ---------------------------------------------------------------------------
# Markdown structure
# ---------------------------------------------------------------------------


def strip_front_matter(text: str) -> tuple[str, str]:
    """Split a card into (yaml front matter, body). A card with no front
    matter returns ``("", text)`` — the common case for GGUF quant repos."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end < 0:
        return "", text
    return text[3:end].strip(), text[end + 4:].lstrip("\n")


def sections(body: str) -> list[tuple[str, str]]:
    """``[(heading, text), …]`` in document order. Text before the first
    heading is returned under the empty heading, because that preamble is
    usually the only description a quant repo has."""
    marks = list(_HEADING_RE.finditer(body))
    out: list[tuple[str, str]] = []
    if not marks:
        return [("", body.strip())] if body.strip() else []
    if marks[0].start() > 0:
        lead = body[: marks[0].start()].strip()
        if lead:
            out.append(("", lead))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append((m.group(2).strip(), body[m.end():end].strip()))
    return out


def _tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in ``text`` as a list of rows of cells."""
    tables, current = [], []
    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line):
            if _SEP_ROW_RE.match(line):
                continue                       # the |---|---| divider
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            current.append(cells)
            continue
        if current:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


def _looks_numeric(cell: str) -> bool:
    return bool(re.match(r"^\**\s*[-+]?\d+(\.\d+)?\s*%?\s*\**$", cell or ""))


def benchmark_claims(body: str, limit: int = 40
                     ) -> tuple[tuple[BenchmarkClaim, ...], int]:
    """Benchmark tables -> claims. Returns ``(claims, table_count)``.

    Two shapes exist in the wild and both are handled:
      * benchmarks down the ROWS (``| MMLU | 74.2 |``) — the first cell names
        the benchmark, the numeric cells are the values;
      * benchmarks across the COLUMNS (``| model | MMLU | GSM8K |``) — the
        header names the benchmarks and the first data row is this model.

    A table with no numeric cell at all is not a benchmark table, whatever its
    heading says, and is skipped rather than emitted as a claim with a null
    value."""
    claims: list[BenchmarkClaim] = []
    count = 0
    for heading, text in sections(body):
        for table in _tables(text):
            header = table[0]
            rows = table[1:]
            scored_columns = sum(1 for c in header[1:] if _BENCH_COLUMN.search(c))
            # Orientation, decided in this order: a first column that names the
            # benchmark wins outright; otherwise TWO OR MORE scored columns
            # means the benchmarks are across the header.
            row_header = bool(header and _BENCH_ROW_HEADER.match(header[0]))
            header_is_bench = (not row_header) and scored_columns >= 2
            interesting = (bool(_BENCH_HEADING.search(heading)) or row_header
                           or scored_columns >= 1)
            if not interesting:
                continue
            if not any(_looks_numeric(c) for r in rows for c in r[1:]):
                continue
            count += 1
            if header_is_bench:
                # benchmarks across the columns; first data row is the subject
                for row in rows[:3]:
                    subject = row[0] if row else ""
                    for i, cell in enumerate(row[1:], start=1):
                        if i >= len(header) or not _looks_numeric(cell):
                            continue
                        claims.append(BenchmarkClaim(
                            benchmark=header[i].strip("* "),
                            value=cell.strip("* "), metric=None,
                            comparator=subject.strip("* ") or None))
            else:
                for row in rows:
                    if not row or not any(_looks_numeric(c) for c in row[1:]):
                        continue
                    value = next((c for c in row[1:] if _looks_numeric(c)), None)
                    claims.append(BenchmarkClaim(
                        benchmark=row[0].strip("* "),
                        value=(value or "").strip("* "),
                        metric=(header[1].strip("* ")
                                if len(header) > 1 else None)))
            if len(claims) >= limit:
                return tuple(claims[:limit]), count
    return tuple(claims[:limit]), count


def _first_section(body: str, pattern: re.Pattern, cap: int = 1200) -> str:
    for heading, text in sections(body):
        if heading and pattern.search(heading):
            return text[:cap].strip()
    return ""


def arxiv_ids(text: str) -> tuple[str, ...]:
    ids = [m.group(1) for m in _ARXIV_RE.finditer(text or "")]
    ids += [m.group(1) for m in _ARXIV_BARE_RE.finditer(text or "")]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return tuple(out)


def parse_card(readme: str | None) -> CardDigest | None:
    """README text -> :class:`CardDigest`. ``None`` in, ``None`` out — a repo
    with no card is a FACT, and an empty digest would hide it."""
    if not readme or not readme.strip():
        return None
    _front, body = strip_front_matter(readme)
    heads = tuple(h for h, _t in sections(body) if h)[:30]
    claims, tables = benchmark_claims(body)
    lead = next((t for h, t in sections(body) if not h), "")
    if not lead:
        lead = next((t for _h, t in sections(body) if t), "")
    summary = " ".join(
        line.strip() for line in lead.splitlines()
        if line.strip() and not line.strip().startswith(("|", "<", "!", "```")))
    links = []
    for m in _LINK_RE.finditer(body):
        url = m.group(0).rstrip(".,;")
        if url not in links:
            links.append(url)
    return CardDigest(
        chars=len(readme), headings=heads, summary=summary[:900],
        benchmark_claims=claims, benchmark_tables=tables,
        training_data=_first_section(body, _SECTION_PATTERNS["training_data"]),
        limitations=_first_section(body, _SECTION_PATTERNS["limitations"]),
        intended_use=_first_section(body, _SECTION_PATTERNS["intended_use"]),
        prompt_format=_first_section(body, _SECTION_PATTERNS["prompt_format"],
                                     cap=600),
        links=tuple(links[:25]))


# ---------------------------------------------------------------------------
# Specialization
# ---------------------------------------------------------------------------


def _languages(card_data: Mapping[str, Any] | None,
               tags: Sequence[str]) -> tuple[str, ...]:
    langs: list[str] = []
    raw = (card_data or {}).get("language")
    if isinstance(raw, str):
        langs = [raw]
    elif isinstance(raw, (list, tuple)):
        langs = [str(x) for x in raw]
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("language:"):
            langs.append(tag.split(":", 1)[1])
    seen, out = set(), []
    for lang in langs:
        low = str(lang).strip().lower()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return tuple(out[:40])


def _variant(hub_id: str, tags: Sequence[str], card_text: str) -> tuple[
        str | None, bool | None]:
    """(variant, instruct). ``None`` when the card and the name are silent —
    a repo that never says is not asserted to be a base model."""
    hay = f"{hub_id} {' '.join(str(t) for t in tags)}".lower()
    low = (card_text or "").lower()
    if "instruct" in hay or "-it" in hay or "instruction-tuned" in low:
        return "instruct", True
    if "chat" in hay or "chat template" in low or "chatml" in low:
        return "chat", True
    if any(k in hay for k in ("thinking", "reasoning", "-r1", "cot")):
        return "reasoning", True
    if re.search(r"\bbase\b", hay) or "base model" in low[:400]:
        return "base", False
    return None, None


def build_emphasis(hub_id: str, tags: Sequence[str], card_text: str,
                   pipeline_tag: str | None = None,
                   ) -> tuple[EmphasisWeight, ...]:
    """Per-domain 0..1 weights, each carrying the strings that earned it.

    Weighting: a hit in the repo NAME is 0.5 (a publisher names a repo after
    what it is for), a TAG is 0.3, a CARD phrase is 0.25, and the pipeline tag
    contributes 0.3 to the modality domain it implies. Capped at 1.0, sorted
    strongest first, and a domain with no evidence is simply absent."""
    name = hub_id.lower()
    tagset = [str(t).lower() for t in (tags or [])]
    low = (card_text or "").lower()
    rows: list[EmphasisWeight] = []
    for domain, (needles, phrases) in DOMAIN_SIGNALS.items():
        weight, evidence = 0.0, []
        for needle in needles:
            if needle in name.split("/")[-1]:
                weight += 0.5
                evidence.append(f"repo name contains {needle!r}")
                break
        for needle in needles:
            if any(needle in t for t in tagset):
                weight += 0.3
                evidence.append(f"tag matches {needle!r}")
                break
        hits = [p for p in phrases if p in low]
        if hits:
            weight += min(0.5, 0.25 * len(hits))
            evidence.append("card says " + ", ".join(repr(h) for h in hits[:3]))
        if domain == "vision" and pipeline_tag in (
                "image-text-to-text", "visual-question-answering",
                "image-to-text"):
            weight += 0.3
            evidence.append(f"pipeline_tag {pipeline_tag}")
        if domain == "speech" and pipeline_tag in (
                "text-to-speech", "automatic-speech-recognition"):
            weight += 0.3
            evidence.append(f"pipeline_tag {pipeline_tag}")
        if domain == "embedding" and pipeline_tag in (
                "feature-extraction", "sentence-similarity"):
            weight += 0.3
            evidence.append(f"pipeline_tag {pipeline_tag}")
        if weight > 0:
            rows.append(EmphasisWeight(domain=domain,
                                       weight=round(min(1.0, weight), 3),
                                       evidence=tuple(evidence)))
    rows.sort(key=lambda r: (-r.weight, r.domain))
    return tuple(rows)


def focus_sentences(card_text: str, limit: int = 4) -> tuple[str, ...]:
    """The author's own purpose sentences, verbatim and de-duplicated."""
    out: list[str] = []
    for m in _FOCUS_RE.finditer(card_text or ""):
        sentence = " ".join(m.group(0).split())
        if 25 <= len(sentence) <= 400 and sentence not in out:
            out.append(sentence)
        if len(out) >= limit:
            break
    return tuple(out)


def architecture_family(architecture: str | None,
                        hub_id: str = "") -> str | None:
    """Coarse family for a display grouping, from the architecture class name
    and the repo name. ``None`` when neither says anything — the UI shows the
    raw architecture instead of a made-up family."""
    hay = f"{architecture or ''} {hub_id}".lower()
    for needle, family in ARCH_FAMILIES:
        if needle in hay:
            return family
    return None


def build_specialization(hub_id: str, payload: Mapping[str, Any],
                         card: CardDigest | None,
                         card_text: str = "") -> Specialization:
    """Everything the card and the metadata say about what this model is FOR.

    ``payload`` is the reviewer's repo-info dict (screen.py's ``_repo_info``
    shape); ``card_text`` is the raw README when it was fetched."""
    tags = [str(t) for t in (payload.get("tags") or [])]
    pipeline_tag = payload.get("pipeline_tag")
    text = card_text or ""
    if card:
        text = text or " ".join(
            filter(None, (card.summary, card.intended_use, card.training_data)))
    variant, instruct = _variant(hub_id, tags, text)
    emphasis = build_emphasis(hub_id, tags, text, pipeline_tag)
    declared = [pipeline_tag] if pipeline_tag else []
    for tag in tags:
        if tag in MODALITY_BY_TASK and tag not in declared:
            declared.append(tag)
    notes: list[str] = []
    if not text:
        notes.append("no model card text was available — specialization is "
                     "read from tags and the repo name only")
    return Specialization(
        declared_tasks=tuple(declared),
        pipeline_tag=pipeline_tag,
        modality=MODALITY_BY_TASK.get(pipeline_tag or ""),
        variant=variant, instruct=instruct,
        languages=_languages(payload.get("card_data"), tags),
        domains=tuple(e.domain for e in emphasis if e.weight >= 0.3),
        emphasis=emphasis,
        finetune_focus=focus_sentences(text),
        notes=tuple(notes))


__all__ = ["ARCH_FAMILIES", "DOMAIN_SIGNALS", "MODALITY_BY_TASK",
           "architecture_family", "arxiv_ids", "benchmark_claims",
           "build_emphasis", "build_specialization", "focus_sentences",
           "parse_card", "sections", "strip_front_matter"]
