from __future__ import annotations

import logging

from hugpy_engine.config.models.models_default import default_context_tokens_for_model
from hugpy_platform.constants import DEFAULT_MAX_TOKENS
from hugpy_platform.utils import message_to_dict
from hugpy_engine.chat_context.context_budget import (
    ContextBudget,
    compact_messages_to_budget,
    estimate_message_tokens,
)

logger = logging.getLogger(__name__)


def compact_chat_request(req):
    max_context_tokens = default_context_tokens_for_model(req.model_key)
    requested_output_tokens = req.max_new_tokens or DEFAULT_MAX_TOKENS

    reserved_output_tokens = min(
        requested_output_tokens,
        max(4096, max_context_tokens // 3),
    )

    budget = ContextBudget(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    raw_messages = [message_to_dict(message) for message in req.messages]

    logger.info(
        "compact_chat_request before: model=%s count=%s roles=%s chars=%s",
        req.model_key,
        len(raw_messages),
        [m.get("role") for m in raw_messages],
        [len(str(m.get("content", ""))) for m in raw_messages],
    )

    compacted_dicts = compact_messages_to_budget(raw_messages, budget)

    logger.info(
        "compact_chat_request after: model=%s count=%s roles=%s chars=%s",
        req.model_key,
        len(compacted_dicts),
        [m.get("role") for m in compacted_dicts],
        [len(str(m.get("content", ""))) for m in compacted_dicts],
    )

    if req.messages:
        message_type = type(req.messages[0])
        compacted_messages = [message_type(**message) for message in compacted_dicts]
    else:
        compacted_messages = []

    return req.model_copy(update={"messages": compacted_messages})


# ---------------------------------------------------------------------------
# Ctx-fit guard — the serving-path seam.
#
# A chat session dies once its history outgrows the model's window (~10 turns
# on a 32k model): the engine refuses the over-ctx prompt and the UI posts an
# ever-longer history, so it can never recover. The fix is DROP-ONLY: shed the
# oldest non-system turns until the request fits, always keeping system
# message(s), the newest user turn, and as much recent tail as fits. Unlike
# compact_chat_request above it never rewrites message content, so when even
# the minimal set (system + newest user turn) cannot fit, the request passes
# through UNTOUCHED and today's honest refusal stands.
# ---------------------------------------------------------------------------

# The ADVERTISED-window resolver (2026-09-25). Central registers the same
# function /v1/models uses for ``context_length`` (live served ctx from the
# worker heartbeats, else the GGUF's trained ctx), so the guard trims against
# the window the model is really served at. Before this the guard read the
# manifest's model_max_length (32768 for coder-next) while the slot ran at
# 262144, and dropped most of an agent's history for nothing. Reverse-injected
# so the engine never imports the server layer.
_CTX_MAX_RESOLVER = None


def set_ctx_max_resolver(fn) -> None:
    """Register ``fn(model_key) -> int|None`` as the authoritative window."""
    global _CTX_MAX_RESOLVER
    _CTX_MAX_RESOLVER = fn


def _ctx_max_for_model(model_key: str) -> int:
    """The model's context window: the registered resolver's figure (the SAME
    one /v1/models reports as ``context_length``) when central registered one,
    else model meta ``model_max_length`` — live registry first (a
    runtime-registered model isn't in the import-time snapshot), snapshot dict
    second. 0 = unknown, and the guard SKIPS an unknown window rather than
    guessing one: guessing small would truncate a legitimate long-context
    request."""
    if not model_key:
        return 0
    if _CTX_MAX_RESOLVER is not None:
        try:
            v = _CTX_MAX_RESOLVER(model_key)
            if v:
                return int(v)
        except Exception:  # noqa: BLE001 — fall back to the manifest figure
            pass
    try:
        from hugpy_engine.config.models.models_config import get_models_dict
        cfg = get_models_dict().get(model_key)
        ctx_max = getattr(cfg, "model_max_length", None)
        if ctx_max:
            return int(ctx_max)
    except Exception:  # noqa: BLE001 — sourcing must never break the guard
        pass
    try:
        from hugpy_engine.config.models.models_default import DEFAULT_CONTEXT_TOKENS_BY_MODEL
        ctx_max = DEFAULT_CONTEXT_TOKENS_BY_MODEL.get(model_key)
        if ctx_max:
            return int(ctx_max)
    except Exception:  # noqa: BLE001
        pass
    return 0


def ctx_fit_keep_indices(message_dicts, *, ctx_max, requested_output_tokens=None):
    """Which messages survive the drop-oldest ctx-fit -> (keep_indices, dropped).

    ``(None, 0)`` means "leave the request untouched": it already fits, the
    window is unknown, or dropping alone cannot make it fit (the honest-refusal
    path — there is no user turn, or system + the newest user turn alone
    overflow the window).

    Costs use estimate_message_tokens (the existing chars/4 heuristic). The
    output reservation mirrors compact_chat_request above: the schema default
    max_new_tokens equals a whole 32k window, so the raw request value is
    clamped to max(4096, ctx_max // 3) before it is held against the prompt.
    """
    ctx_max = int(ctx_max or 0)
    if ctx_max <= 0:
        return None, 0

    requested = int(requested_output_tokens or DEFAULT_MAX_TOKENS)
    reserved_output = min(requested, max(4096, ctx_max // 3))
    # Small fixed margin for the chat template's own wrapping tokens.
    input_budget = ctx_max - reserved_output - 512
    if input_budget <= 0:
        return None, 0

    costs = [estimate_message_tokens(m) for m in message_dicts]
    roles = [str(m.get("role", "user")) for m in message_dicts]

    system_cost = sum(c for c, r in zip(costs, roles) if r == "system")
    dialogue = [i for i, r in enumerate(roles) if r != "system"]
    if not dialogue:
        return None, 0

    if system_cost + sum(costs[i] for i in dialogue) <= input_budget:
        return None, 0

    user_turns = [i for i in dialogue if roles[i] == "user"]
    if not user_turns:
        return None, 0
    newest_user = user_turns[-1]

    # Minimal set: system message(s) + the newest user turn (and anything the
    # continuation loop appended after it). If even that overflows, dropping
    # cannot fix this request — pass it through for today's honest refusal.
    mandatory = [i for i in dialogue if i >= newest_user]
    used = system_cost + sum(costs[i] for i in mandatory)
    if used > input_budget:
        return None, 0

    # Grow the kept tail backwards from the newest user turn while it fits —
    # the first older turn that doesn't fit ends the tail (contiguous suffix,
    # so "dropped" is exactly the oldest N dialogue turns).
    kept = list(mandatory)
    for i in reversed([i for i in dialogue if i < newest_user]):
        if used + costs[i] > input_budget:
            break
        kept.append(i)
        used += costs[i]
    kept.sort()

    # Strict chat templates demand the first non-system turn be a user turn;
    # never keep an assistant reply whose user turn was just dropped.
    while kept and roles[kept[0]] != "user":
        kept.pop(0)

    dropped = len(dialogue) - len(kept)
    if dropped <= 0:
        return None, 0

    keep = sorted([i for i, r in enumerate(roles) if r == "system"] + kept)
    return keep, dropped


def ctx_fit_chat_request(req):
    """Fit a built ChatRequest into its model's window by dropping oldest turns.

    Returns ``req`` itself whenever nothing should (or can) change, else a
    ``model_copy`` with the shortened history. FAIL-OPEN by contract: any error
    inside the guard returns the request untouched.
    """
    try:
        message_dicts = [message_to_dict(m) for m in req.messages]
        ctx_max = _ctx_max_for_model(req.model_key)
        keep, dropped = ctx_fit_keep_indices(
            message_dicts,
            ctx_max=ctx_max,
            requested_output_tokens=req.max_new_tokens,
        )
        if not keep or not dropped:
            return req
        kept_messages = [req.messages[i] for i in keep]
        logger.info(
            "ctx_fit: dropped %s oldest turn(s) for model=%s "
            "(kept %s of %s messages, ctx_max=%s)",
            dropped, req.model_key, len(kept_messages), len(req.messages), ctx_max,
        )
        return req.model_copy(update={"messages": kept_messages})
    except Exception:  # noqa: BLE001 — the guard must never break serving
        logger.warning("ctx_fit guard errored; request passed through unchanged",
                       exc_info=True)
        return req
