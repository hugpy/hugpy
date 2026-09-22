"""extract_json — the pure reply-parsing half of the aptitude bench's api.py.

PORTED VERBATIM from ``evaluations/studio-aptitude/api.py``. Only the parsing
comes over: the bench's HTTP client, its sweep driver and its LLM judge have no
place on a worker, whose self-test must never make a network call or ask a model
for an opinion.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> tuple[dict | None, str]:
    """Best-effort object extraction. Returns (obj_or_None, how).

    `how` records HOW clean the model was, which feeds the schema score:
      exact  — the whole reply was the object
      fenced — wrapped in a markdown code fence
      embedded — an object was found inside surrounding prose
    """
    s = (text or "").strip()
    if not s:
        return None, "empty"
    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return o, "exact"
    except Exception:
        pass
    m = _FENCE.search(s)
    if m:
        try:
            o = json.loads(m.group(1).strip())
            if isinstance(o, dict):
                return o, "fenced"
        except Exception:
            pass
    # brace scan for the first balanced object
    start = s.find("{")
    while start != -1:
        depth = 0
        instr = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                instr = not instr
                continue
            if instr:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(s[start:i + 1])
                        if isinstance(o, dict):
                            return o, "embedded"
                    except Exception:
                        pass
                    break
        start = s.find("{", start + 1)
    return None, "none"
