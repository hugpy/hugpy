"""Parse a tolerant ``VERDICT/SCORE/WHY`` model reply."""

from __future__ import annotations

import re


def parse_verdict(text: str) -> dict:
    """Return a verdict, clamped score, and one-line explanation."""
    t = text or ""
    verdict = None
    m = re.search(r"VERDICT\s*[=:]\s*(YES|NO)", t, re.I)
    if m:
        verdict = m.group(1).upper()

    score = None
    m = re.search(r"SCORE\s*[=:]\s*(\d{1,3})", t, re.I)
    if m:
        score = max(0, min(100, int(m.group(1))))

    why = ""
    m = re.search(r"WHY\s*[=:]\s*(.+)", t, re.I | re.S)
    if m:
        why = m.group(1).strip().splitlines()[0].strip().rstrip(".").strip()

    if verdict is None:
        if re.search(r"\bYES\b", t, re.I):
            verdict = "YES"
        elif re.search(r"\bNO\b", t, re.I):
            verdict = "NO"

    return {"verdict": verdict, "score": score, "why": why}
