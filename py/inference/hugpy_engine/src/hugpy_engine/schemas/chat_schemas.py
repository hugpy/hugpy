from typing import Any, Mapping, Optional, Union
from abstract_essentials import read_from_file
from pydantic import BaseModel, ConfigDict, Field, field_validator
from hugpy_engine.schemas.task_schemas import TaskResult
from hugpy_platform.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FINISH_REASONS,
    ROLES,
)
from hugpy_platform.utils import get_message, get_messages, get_request_id
from pydantic import model_validator, model_serializer
ChatInput = Union["ChatRequest", Mapping, str]  # request | dict-ish | bare prompt

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ROLES = "user"
    # content is optional on the OpenAI tool-calling shapes: an assistant turn
    # that ONLY calls a tool carries `tool_calls` and a null `content`. Kept
    # str|None (default "") so plain chat is byte-for-byte unchanged.
    content: Optional[str] = ""
    # OpenAI tool-calling fields (the /v1 loop, step 2+). Added explicitly so
    # the schema stays extra="forbid": an assistant echo-back carries its
    # `tool_calls` array, and a `{"role":"tool"}` result carries `tool_call_id`.
    # The runner never sees these — v1_helpers renders them into `content` text
    # (the Qwen <tool_call>/<tool_response> convention) before relay, so a
    # released worker on the frozen role+content wire keeps working.
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_null_tool_fields(self, handler):
        # Workers on the released package re-validate the relayed request with
        # a ChatMessage that is still role+content only and extra="forbid" —
        # dumping tool_calls/tool_call_id as None keys breaks EVERY offloaded
        # chat. Omit them unless actually set so the relay wire stays
        # byte-identical to the frozen schema for all non-tool traffic.
        data = handler(self)
        for key in ("tool_calls", "tool_call_id"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

# ---------------------------------------------------------------------------
# Role-alternation normalisation — the ONE shim for chat templates that demand
# strict user/assistant/user/assistant order (Mistral/Nemo family, and every
# GLM/Gutenberg merge built on them). Their jinja raises
#
#   TemplateError: After the optional system message, conversation roles must
#   alternate user/assistant/user/assistant/...
#
# ...which is a POISON LOOP in any chat UI that posts the whole history: a turn
# that fails appends NO assistant reply, so the next attempt posts [user, user],
# then [user, user, user] — every retry is more malformed than the last and the
# conversation can never recover (operator report 2026-07-27:
# DavidAU~MN-GRAND-23.5B-…-GLM4.7-Thinking on ae, never once answered).
#
# Merging adjacent same-role turns is the standard, CONTENT-PRESERVING shim:
# nothing is reordered, dropped or invented — two consecutive user turns become
# one user turn holding both texts. It also fixes legitimate double-sends.
#
# It lives on ChatRequest because that is the single funnel every chat path
# passes through (console route, /v1, the agent client, utils.text, and the
# worker re-validating a relayed payload), and because it runs at CENTRAL
# *before* _worker_payload() dumps the request onto the relay wire — so already
# released workers get well-formed histories with no wire change and no new
# field.
# ---------------------------------------------------------------------------
_TOOL_SHAPE_KEYS = ("tool_calls", "tool_call_id")


def _msg_role(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("role", "user")
    return getattr(message, "role", None)


def _msg_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


def _is_tool_shaped(message: Any) -> bool:
    """A tool-loop turn (role 'tool', or an assistant echo carrying tool_calls).

    These are NEVER merged and are never merged ACROSS — the working /v1 tool
    loop (2026-07-17) renders them into a precise <tool_call>/<tool_response>
    dialect and folding two of them together would corrupt it.
    """
    if _msg_role(message) == "tool":
        return True
    if isinstance(message, Mapping):
        return any(message.get(k) for k in _TOOL_SHAPE_KEYS)
    return any(getattr(message, k, None) for k in _TOOL_SHAPE_KEYS)


def _is_mergeable(message: Any) -> bool:
    if _is_tool_shaped(message):
        return False
    if not isinstance(_msg_role(message), str):
        return False
    # content must already be flat text; a list (multimodal parts) is flattened
    # by _normalize_multimodal upstream, and anything else is left alone.
    return isinstance(_msg_content(message), (str, type(None)))


def merge_consecutive_messages(messages: Any) -> Any:
    """Merge adjacent same-role turns; return the input unchanged if none merge.

    Content-preserving: same order, same text, joined by a blank line. Empty
    contents contribute nothing (no leading/trailing blank lines). Tool-shaped
    turns act as barriers. An already-alternating history is returned as the
    SAME object — byte-identical, zero effect on every model that never needed
    this.
    """
    if not isinstance(messages, (list, tuple)) or len(messages) < 2:
        return messages
    out: list = []
    merged_any = False
    for message in messages:
        if (out
                and _is_mergeable(message)
                and _is_mergeable(out[-1])
                and _msg_role(message) == _msg_role(out[-1])):
            prev = out[-1]
            parts = [p for p in (_msg_content(prev), _msg_content(message)) if p]
            base = dict(prev) if isinstance(prev, Mapping) else {
                "role": _msg_role(prev)}
            base["content"] = "\n\n".join(parts)
            out[-1] = base
            merged_any = True
            continue
        out.append(message)
    return out if merged_any else messages


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(default_factory=lambda: get_request_id())
    model_key: str = None
    messages: list[ChatMessage]
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = False
    unbounded: bool = False
    max_chunks: Optional[int] = None
    file: Optional[str] = None
    # Inline images for the current turn as base64 (raw or full data: URI). The
    # no-upload vision path: the runner folds these into the latest user turn as
    # image_url parts (see LlamaCppBaseRunner._attach_image). None for text chat.
    images: Optional[list[str]] = None
    # Dedicated worker pool to route this request to ("" / None = general pool).
    # Resolved at the route from the API key's bound pool + an optional override.
    pool: Optional[str] = None
    # Per-REQUEST allocation triggers (operator ask 2026-07-29): a dict of spill
    # keys (alloc_mode, bnb_4bit, n_cpu_moe, n_gpu_layers, gpu_mem_gib,
    # cpu_mem_gib, threads) overlaid on the worker's per-assignment spill for
    # THIS call, so a client can exercise one model under a chosen config
    # (4-bit on/off, MoE split, RAM-only, ...) without editing designations.
    # ⚠ WIRE LANDMINE: the relay MUST pop this key before the payload ships —
    # released workers run extra="forbid" and an unknown key rejects ALL chat
    # (the 2026-07-17 None-key incident). _relay_payload owns that pop + the
    # whitelist; this field never crosses the wire itself.
    # Takes effect at LOAD time: an already-resident model keeps its current
    # placement until evicted/reseated (same rule as designation edits).
    alloc: Optional[dict] = None
    # t74 HARD NO-THINK: per-request engine chat keys, forwarded verbatim onto
    # the slot child's llama-server /v1/chat/completions body.
    #   chat_template_kwargs — reaches the chat TEMPLATE (the Qwen3
    #     ``{"enable_thinking": false}`` idiom emits a pre-closed <think> block,
    #     so suppression is enforced by rendering, not by model obedience — the
    #     fix for models that ignore the /no_think soft switch);
    #   logit_bias — OpenAI-style token bias (the <think>-token-ban fallback).
    # ⚠ WIRE LANDMINE, same class as ``alloc``: released workers re-validate
    # relays with extra="forbid" schemas and builders that don't forward these
    # keys. They are omitted from model_dump() when None (see the serializer
    # below), and when SET the relay version-gates them per worker
    # (alloc_modes.gate_chat_extras_for_worker) — strip + LOUD log for a worker
    # predating CHAT_EXTRAS_MIN_PKG_VERSION, never a silent no-op.
    chat_template_kwargs: Optional[dict] = None
    logit_bias: Optional[dict] = None
    # k96 NO-EVICT GUARANTEE for agent-brain calls (operator ruling
    # 2026-08-06). When true, serving this request must never cost the fleet a
    # resident model: a WARM model serves exactly as today, but a cold load may
    # spend only genuinely free headroom — dispatch runs the whole admission
    # POLITELY (the k56 no_evict path: in-process contention yield skipped,
    # cross-tier make-room refuses instead of evicting, the slot pool neither
    # ceiling-evicts nor bumps a seat) and fails FAST with a capacity-class
    # refusal ("won't fit … refusing without evicting") that the agent's brain
    # ladder walks down. hugpy_agent sends it on every brain chat.
    # ⚠ WIRE LANDMINE, same class as ``alloc``: released workers re-validate
    # relays with extra="forbid". The relay ALWAYS pops this key and instead
    # rides the guarantee on the spill's ``no_evict`` (version-gated per
    # worker: an old worker that would evict gets the load REFUSED at central
    # rather than a silently broken promise). Omitted from model_dump() when
    # None (serializer below).
    no_makeroom: Optional[bool] = None
    # HARNESS ATTRIBUTION (2026-09-24): who made this call — a harness/client
    # identity ("hermes", "aider", "codex", an API-key name, ...) the /v1 route
    # derives (header, OpenAI ``user`` field, or the bearer key's name) so the
    # per-call metrics row (resolvers.remote.per_call_row -> compute_actions
    # detail.caller) attributes the call to it instead of the bare default
    # "api". Central-only: it never crosses the worker wire — the serializer
    # below drops it from model_dump() UNCONDITIONALLY (the relay ships
    # model_dump() and released workers run extra="forbid"). None = the honest
    # default, exactly today's behaviour.
    caller: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_null_engine_extras(self, handler):
        # Same discipline as ChatMessage._omit_null_tool_fields: dumping the
        # t74 keys as None would break EVERY offloaded chat against a released
        # worker (extra="forbid" on the re-validated wire). Omit unless set so
        # the relay wire stays byte-identical for all non-no-think traffic.
        data = handler(self)
        for key in ("chat_template_kwargs", "logit_bias", "no_makeroom"):
            if data.get(key) is None:
                data.pop(key, None)
        # ``caller`` is a CENTRAL-ONLY attribution field (metrics read it off the
        # object as ``req.caller``); it is never a worker input, so it is dropped
        # from EVERY model_dump() — the relay wire (remote._relay_payload /
        # peer relay both ship model_dump()) stays byte-identical and an
        # extra="forbid" worker never sees an unknown key.
        data.pop("caller", None)
        return data

    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return get_messages(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_multimodal(cls, data: Any) -> Any:
        """Funnel every message shape down to one well-formed history.

        Two normalisations, in order, because the second needs the first's
        output to be flat text:

        1. Whatever a client sends — OpenAI-style ``content`` arrays of
           ``{"type":"text"}`` / ``{"type":"image_url"}`` parts, a bare image
           part, ``input_image`` — the image(s) are hoisted into ``images``
           (data-URI / base64 / url strings) and the message ``content`` is
           flattened to its text. The runner then turns ``images`` into the
           bytes the model sees, so there is a single image path regardless of
           how the request was phrased.
        2. Adjacent same-role turns are merged (see
           ``merge_consecutive_messages``) so a history is always alternating
           by the time any chat template renders it. A leading system message
           has no same-role neighbour and is therefore passed through as-is,
           and an already-alternating history is returned unchanged.
        """
        if not isinstance(data, Mapping):
            return data
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return data
        collected: list = list(data.get("images") or [])

        def _img_url(part: Mapping) -> Optional[str]:
            val = part.get("image_url", part.get("url", part.get("image")))
            if isinstance(val, Mapping):
                val = val.get("url")
            return val if isinstance(val, str) and val else None

        out_msgs = []
        for m in msgs:
            if isinstance(m, Mapping) and isinstance(m.get("content"), list):
                m = dict(m)
                texts: list[str] = []
                for part in m["content"]:
                    if not isinstance(part, Mapping):
                        if isinstance(part, str):
                            texts.append(part)
                        continue
                    ptype = part.get("type")
                    if ptype in ("image_url", "input_image", "image"):
                        url = _img_url(part)
                        if url:
                            collected.append(url)
                    elif ptype == "text":
                        texts.append(part.get("text") or "")
                    else:  # unknown part with a usable text/url, be lenient
                        if isinstance(part.get("text"), str):
                            texts.append(part["text"])
                m["content"] = "\n".join(t for t in texts if t)
            out_msgs.append(m)

        # Alternation shim — runs LAST so it sees flattened text content.
        # Returns `out_msgs` itself when nothing needed merging.
        normalized = merge_consecutive_messages(out_msgs)

        data = dict(data)
        data["messages"] = normalized
        if collected:
            data["images"] = collected
        return data

    @classmethod
    def coerce(cls, value: ChatInput, *, model_key: Optional[str] = None) -> "ChatRequest":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model_key=model_key, messages=value)  # validator handles it
        if isinstance(value, Mapping):
            data = dict(value)
            
            if "messages" not in data and "prompt" in data:
                prompt = data.pop("prompt")
                file = data.pop("file", None)
                if file:
                    content = read_from_file(file)
                    prompt = f"{prompt}\n------{file}------\n{content}"
                system = data.pop("system", None)
                msgs = []
                if system:
                    msg = get_message(content=prompt,role="system")
                    msgs.append(msg)
                msg = get_message(content=prompt,role="user")
                msgs.append(msg)
                data["messages"] = msgs
            if "model_key" not in data and model_key:
                data["model_key"] = model_key
            return cls.model_validate(data)
        raise TypeError(f"cannot coerce {type(value).__name__} to ChatRequest")

class ChatResult(TaskResult):
    text: str
    finish_reason: FINISH_REASONS
    usage: Optional[dict] = None
    output_chunks: int = 0
