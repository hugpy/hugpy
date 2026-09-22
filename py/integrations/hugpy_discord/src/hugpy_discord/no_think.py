"""NO-THINK helpers for the bot (client-side copy).

The bot is an HTTP client of a Hugpy central and must not depend on
``hugpy_engine``. These are the two pure-string halves of the engine's
``hugpy_engine.utils.no_think`` seam that a client needs: append the ``/no_think``
directive to a prompt before sending it, and strip any ``<think>…</think>`` the
model returns anyway (model adherence is never trusted). Keep them in sync with
the engine's definitions; they must produce identical results.
"""
from __future__ import annotations

import re

__all__ = ["NO_THINK_DIRECTIVE", "THINK_BLOCK_RE", "strip_think", "with_no_think"]

NO_THINK_DIRECTIVE = "/no_think"

#: Matches a think block, closed OR unclosed (``\Z`` alternative): when the token
#: budget runs out mid-thought there is no closing tag, and a truncated ramble
#: must never be served as prose.
THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str]:
    """Split a model reply into ``(prose, reasoning)``.

    Removes every ``<think>...</think>`` block and returns the surviving prose
    plus the concatenated reasoning. An unclosed ``<think>`` is treated as
    reasoning to the end of the string. A reply that was nothing but thinking
    yields ``("", reasoning)``; the caller turns that into an honest error.
    """
    if not text:
        return "", ""
    reasoning = "\n".join(m.group(1).strip() for m in THINK_BLOCK_RE.finditer(text))
    return THINK_BLOCK_RE.sub("", text).strip(), reasoning.strip()


def with_no_think(user_text: str) -> str:
    """Append the no-think directive to *user_text*; idempotent."""
    if not user_text:
        return NO_THINK_DIRECTIVE
    if NO_THINK_DIRECTIVE in user_text:
        return user_text
    return f"{user_text}\n\n{NO_THINK_DIRECTIVE}"
