"""Small result-object adapters shared by routes and inference callers."""

from __future__ import annotations


def result_text(result) -> str:
    """Extract text from a mapping or result object, defaulting to empty text."""
    if isinstance(result, dict):
        return result.get("text") or ""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d.get("text") or ""
            except TypeError:
                continue
    return getattr(result, "text", "") or ""


def vision_result_text(result) -> str:
    """Extract a vision reply, retaining its representation as a last resort."""
    txt = getattr(result, "text", None)
    if txt:
        return txt
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                d = fn()
            except TypeError:
                continue
            if isinstance(d, dict) and d.get("text"):
                return d["text"]
    return str(result)


def result_payload(result) -> dict:
    """Convert a result to its mapping or a text fallback."""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    return {"text": str(result)}
