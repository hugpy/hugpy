"""Text utilities an agent actually calls: approximate token counting,
token/line chunking, and unified diffs. Stdlib only.

The token estimator is a dependency-free BPE approximation ported from
abstract_utilities.parse_utils.token_utils: no ``tiktoken`` import, no model
download. It is deterministic and good enough for budgeting (over-estimates
slightly on dense punctuation, which is the safe direction for a cap).
"""
from __future__ import annotations

import difflib
import math
import re

# Pre-tokeniser: contiguous runs of letters, digits, whitespace, or "other".
_PRETOKEN = re.compile(r"[A-Za-z]+|\d+|\s+|[^A-Za-z\d\s]+")

_W_LETTERS = 4.0   # ~4 chars/token for prose
_W_DIGITS = 3.0
_W_PUNCT = 1.5     # punctuation/symbols are dense
_W_SPACE = 8.0     # whitespace mostly merges into adjacent BPE tokens


def count_tokens(text: str) -> int:
    """Approximate the BPE token count of ``text`` (stdlib only, no model)."""
    text = str(text)
    if not text:
        return 0
    total = 0.0
    for m in _PRETOKEN.finditer(text):
        s = m.group()
        if s.isspace():
            total += len(s) / _W_SPACE
        elif s.isalpha():
            total += max(1.0, len(s) / _W_LETTERS)
        elif s.isdigit():
            total += math.ceil(len(s) / _W_DIGITS)
        else:
            total += math.ceil(len(s) / _W_PUNCT)
    return max(1, math.ceil(total))


def chunk_by_lines(text: str, max_lines: int, overlap: int = 0) -> list[str]:
    """Split ``text`` into chunks of at most ``max_lines`` lines, with an
    optional line ``overlap`` between consecutive chunks (for context bleed)."""
    max_lines = max(1, int(max_lines))
    overlap = max(0, min(int(overlap), max_lines - 1))
    lines = text.splitlines()
    if not lines:
        return []
    step = max_lines - overlap
    chunks = []
    for start in range(0, len(lines), step):
        chunks.append("\n".join(lines[start:start + max_lines]))
        if start + max_lines >= len(lines):
            break
    return chunks


def chunk_by_tokens(text: str, max_tokens: int, delimiter: str = "\n\n") -> list[str]:
    """Split ``text`` into chunks each within the approximate ``max_tokens``
    budget, breaking only on ``delimiter`` boundaries so paragraphs stay whole.

    A single segment larger than the budget is emitted on its own (never
    silently dropped); callers that need hard splitting should pre-split.
    """
    max_tokens = max(1, int(max_tokens))
    segments = [s for s in text.split(delimiter)]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for seg in segments:
        seg_tokens = count_tokens(seg)
        if cur and cur_tokens + seg_tokens > max_tokens:
            chunks.append(delimiter.join(cur))
            cur, cur_tokens = [], 0
        cur.append(seg)
        cur_tokens += seg_tokens
    if cur:
        chunks.append(delimiter.join(cur))
    return [c for c in chunks if c != "" or len(chunks) == 1]


def unified_diff(before: str, after: str, before_name: str = "before",
                 after_name: str = "after", context: int = 3) -> str:
    """Return a unified diff between two texts. Empty string when identical."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=before_name, tofile=after_name, n=max(0, int(context)))
    return "".join(diff)
