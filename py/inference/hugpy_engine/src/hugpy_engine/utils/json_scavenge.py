"""JSON SCAVENGER — recover a well-formed JSON value from a model reply.

WHY THIS EXISTS AS A SHARED SEAM
--------------------------------
Two independent copies of this logic already existed in the package —
``review/judge.py:_extract_json`` (object, fence + brace-balance walk) and
``comms/todo_keeper.py:_extract_json_array`` (array, whole-string / fence /
outermost-span) — written months apart for the same reason: **small local models
wrap JSON in prose or a ```json fence however firmly you tell them not to.** The
studio spread path (``video_intel/prompt_spread.py``) needed the same thing, and
a THIRD copy is where the behaviours quietly diverge. So the two originals now
delegate here and this module is the single definition.

WHAT IT IS NOT
--------------
This is recovery of a WELL-FORMED value that arrived with GARNISH — never a
tolerance for malformed content, and never a repair. Both entry points return
``None`` on failure so the caller can degrade HONESTLY (an error naming the
model and carrying the raw text), which is the whole point: a scavenger that
guesses would fabricate content the model never produced.

STRIP ``<think>`` FIRST. A reasoning block is this failure at its worst — it can
consume the entire token budget before the JSON starts, and it frequently
contains braces of its own that the brace walk would happily lock onto. Callers
run ``utils.no_think.strip_think`` before handing text here; that ordering is
load-bearing, not decorative.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

__all__ = ["extract_json_object", "extract_json_array", "extract_json_value"]

#: A fenced block, with or without the ``json`` info string. Non-greedy so the
#: FIRST fence wins when a chatty model emits several.
_OBJ_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_ANY_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json_object(text: str) -> Optional[dict]:
    """Pull the first parseable JSON OBJECT out of ``text``; ``None`` if none.

    Tries a fenced ``{...}`` block, then walks every ``{`` in the string doing a
    brace-depth scan to find its balanced partner. The walk (rather than
    ``find('{')`` + ``rfind('}')``) is what survives a preamble that itself
    contains braces — it simply moves on to the next candidate opener.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    fence = _OBJ_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def extract_json_array(text: str, accept_lone_object: bool = True) -> Optional[list]:
    """Pull a JSON ARRAY out of ``text``; ``None`` if none.

    Candidates in order: the whole string, a fenced block, the outermost
    ``[...]`` span. ``accept_lone_object`` keeps ``todo_keeper``'s long-standing
    near-miss tolerance — a model that returned ONE object where a list of one
    was asked for is wrapped rather than refused. Set it False when a bare object
    should be a failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [text.strip()]
    m = _ANY_FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
        if accept_lone_object and isinstance(parsed, dict):
            return [parsed]
    return None


def extract_json_value(text: str) -> Any:
    """Object first, else array. For callers that accept either shape."""
    obj = extract_json_object(text)
    if obj is not None:
        return obj
    return extract_json_array(text, accept_lone_object=False)
