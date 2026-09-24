"""Pure text helpers shared by engine and HTTP clients."""

from __future__ import annotations

import re

NO_THINK_DIRECTIVE = "/no_think"
THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str]:
    """Return the prose and reasoning, including an unclosed think block."""
    if not text:
        return "", ""
    reasoning = "\n".join(m.group(1).strip() for m in THINK_BLOCK_RE.finditer(text))
    return THINK_BLOCK_RE.sub("", text).strip(), reasoning.strip()


def with_no_think(user_text: str) -> str:
    """Append the no-think directive once."""
    if not user_text:
        return NO_THINK_DIRECTIVE
    if NO_THINK_DIRECTIVE in user_text:
        return user_text
    return f"{user_text}\n\n{NO_THINK_DIRECTIVE}"
