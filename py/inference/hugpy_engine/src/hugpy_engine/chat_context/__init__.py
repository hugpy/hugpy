"""Chat-context budgeting: fit a conversation into a model's context window."""
from hugpy_engine.chat_context.chat_context import (
    compact_chat_request,
    ctx_fit_chat_request,
    ctx_fit_keep_indices,
)
from hugpy_engine.chat_context.context_budget import (
    ContextBudget,
    TokenCounter,
    compact_messages_to_budget,
    estimate_message_tokens,
    estimate_tokens,
    trim_content_to_token_budget,
)
from hugpy_engine.chat_context.unbounded import (
    GenerateOnce,
    GenerationOutcome,
    map_finish_reason,
    run_unbounded,
)

__all__ = [
    "ContextBudget", "GenerateOnce", "GenerationOutcome", "TokenCounter",
    "compact_chat_request", "compact_messages_to_budget", "ctx_fit_chat_request",
    "ctx_fit_keep_indices", "estimate_message_tokens", "estimate_tokens",
    "map_finish_reason", "run_unbounded", "trim_content_to_token_budget",
]
