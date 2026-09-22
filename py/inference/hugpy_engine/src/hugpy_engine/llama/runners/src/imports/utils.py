from typing import Optional
from hugpy_engine.llama.runners.src.imports.constants import FINISH_REASON_MAP
from hugpy_engine.schemas.chat_schemas import ChatRequest
from hugpy_platform.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from hugpy_platform.utils import messages_to_dicts
# ---------------------------------------------------------------------------
# Prompt formatting — only used as a fallback when the chat template can't
# be applied (raw completion path on the HTTP runner).
# ---------------------------------------------------------------------------

def messages_to_prompt_from_dicts(messages: list[dict]) -> str:
    """Hand-rolled User:/Assistant: scaffolding for raw completion endpoints.

    Prefer the model's embedded chat template over this when possible —
    GGUFs from Qwen/Llama/etc ship with proper templates that match what
    they were trained on. This fallback exists for legacy /completion calls.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(req: ChatRequest) -> str:
    """ChatRequest variant of the above. One definition, not two."""
    return messages_to_prompt_from_dicts(messages_to_dicts(req.messages))

# ---------------------------------------------------------------------------
# Helpers — finish reason mapping, defaulted resolvers
# ---------------------------------------------------------------------------


def map_finish_reason(raw: Optional[str]) -> str:
    return FINISH_REASON_MAP.get(raw, "stop")


def resolve_max_tokens(requested: Optional[int]) -> int:
    if not requested or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return requested


def resolve_temperature(requested: Optional[float], do_sample: bool) -> float:
    if not do_sample:
        return 0.0
    if requested is None or requested < 0:
        return DEFAULT_TEMPERATURE
    return min(requested, 2.0)


def resolve_top_p(requested: Optional[float]) -> float:
    if requested is None or requested <= 0 or requested > 1:
        return DEFAULT_TOP_P
    return requested
