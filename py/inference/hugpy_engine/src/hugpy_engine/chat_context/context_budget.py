from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from hugpy_platform.constants import DEFAULT_MAX_TOKENS

# A token counter maps text -> token count. When the caller has the model's
# real tokenizer (e.g. the in-process GGUF runner) it passes one in and the
# budgeting below becomes exact instead of a chars/token estimate.
TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int
    reserved_output_tokens: int
    # Do not reserve a full DEFAULT_MAX_TOKENS for system text.
    # That can starve the actual user prompt.
    reserved_system_tokens: int = 512
    chars_per_token: float = 4.0

    @property
    def input_token_budget(self) -> int:
        return max(
            512,
            self.max_context_tokens
            - self.reserved_output_tokens
            - self.reserved_system_tokens,
        )


def estimate_tokens(
    text: str,
    *,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> int:
    if not text:
        return 0

    if token_counter is not None:
        try:
            return max(1, int(token_counter(text)))
        except Exception:
            # A flaky tokenizer must never break budgeting; fall back to chars.
            pass

    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(
    message: dict,
    *,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> int:
    role = str(message.get("role", "user"))
    content = str(message.get("content", ""))

    return (
        estimate_tokens(role, chars_per_token=chars_per_token, token_counter=token_counter)
        + estimate_tokens(content, chars_per_token=chars_per_token, token_counter=token_counter)
        + 8
    )


def _assemble_head_tail(content: str, max_chars: int) -> str:
    marker = "\n\n...[middle omitted to fit model context]...\n\n"
    marker_len = len(marker)

    if max_chars <= marker_len + 256:
        return content[-max_chars:]

    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars - marker_len

    return content[:head_chars] + marker + content[-tail_chars:]


def trim_content_to_token_budget(
    content: str,
    token_budget: int,
    *,
    chars_per_token: float = 4.0,
    token_counter: Optional[TokenCounter] = None,
) -> str:
    """Shrink ``content`` (keeping head + tail) until it fits ``token_budget``.

    With a real ``token_counter`` this is exact: we estimate a character
    budget, build a head/tail excerpt, then shrink in a short loop until the
    counter confirms the result is under budget. Without one we fall back to
    the chars/token heuristic (single pass).
    """
    if estimate_tokens(content, chars_per_token=chars_per_token,
                        token_counter=token_counter) <= token_budget:
        return content

    max_chars = max(256, int(token_budget * chars_per_token))

    if token_counter is None:
        if len(content) <= max_chars:
            return content
        return _assemble_head_tail(content, max_chars)

    # Token-accurate: derive a chars/token ratio from this very content so the
    # first guess is close, then shrink until it actually fits.
    measured = max(1, int(token_counter(content)))
    ratio = max(1.0, len(content) / measured)
    max_chars = max(256, int(token_budget * ratio * 0.95))

    candidate = content
    for _ in range(8):
        if len(candidate) > max_chars:
            candidate = _assemble_head_tail(content, max_chars)
        if estimate_tokens(candidate, token_counter=token_counter) <= token_budget:
            return candidate
        max_chars = int(max_chars * 0.8)
        if max_chars < 256:
            break
    return _assemble_head_tail(content, max(256, max_chars))


def compact_messages_to_budget(
    messages: Iterable[dict],
    budget: ContextBudget,
    *,
    token_counter: Optional[TokenCounter] = None,
) -> list[dict]:
    """
    Keep system messages and the newest dialogue turns that fit.

    Critical rule:
    Never return only system messages when a user message exists.
    If the newest user message is too large, trim it instead of dropping it.

    Pass ``token_counter`` (the model's real tokenizer) to make every cost and
    trim exact rather than a chars/token estimate.
    """
    normalized = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
        if str(message.get("content", "")).strip()
    ]

    system_messages = [m for m in normalized if m["role"] == "system"]
    dialogue_messages = [m for m in normalized if m["role"] != "system"]

    if not dialogue_messages:
        return system_messages

    system_cost = sum(
        estimate_message_tokens(
            m, chars_per_token=budget.chars_per_token, token_counter=token_counter
        )
        for m in system_messages
    )

    remaining = max(256, budget.input_token_budget - system_cost)

    kept_reversed: list[dict] = []
    used = 0

    newest_message = dialogue_messages[-1]

    for message in reversed(dialogue_messages):
        cost = estimate_message_tokens(
            message,
            chars_per_token=budget.chars_per_token,
            token_counter=token_counter,
        )

        if used + cost > remaining:
            if not kept_reversed and message is newest_message:
                role = str(message.get("role", "user"))
                content = str(message.get("content", ""))

                # Reserve a small amount for role/template overhead.
                content_budget = max(128, remaining - 16)

                kept_reversed.append(
                    {
                        "role": role,
                        "content": trim_content_to_token_budget(
                            content,
                            content_budget,
                            chars_per_token=budget.chars_per_token,
                            token_counter=token_counter,
                        ),
                    }
                )

            break

        kept_reversed.append(message)
        used += cost

    compacted = system_messages + list(reversed(kept_reversed))

    # Hard safety guard: never silently send system-only when the request had user input.
    if not any(m["role"] == "user" for m in compacted):
        newest_user = next(
            (m for m in reversed(dialogue_messages) if m["role"] == "user"),
            None,
        )

        if newest_user is not None:
            compacted.append(
                {
                    "role": "user",
                    "content": trim_content_to_token_budget(
                        newest_user["content"],
                        max(128, remaining - 16),
                        chars_per_token=budget.chars_per_token,
                        token_counter=token_counter,
                    ),
                }
            )

    return compacted
