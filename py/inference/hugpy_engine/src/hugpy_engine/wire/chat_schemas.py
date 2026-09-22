from typing import List, Optional
from pydantic import BaseModel, field_validator, model_validator

class ChatBody(BaseModel):
    model_key: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    file: Optional[str] = None          # server path from /api/uploads
    images: Optional[List[str]] = None  # base64, if you also do inline images
    # None = "as many as the model allows" — resolved to the model's context at
    # request time. The worker also auto-continues past this per-call cap, so a
    # response is never truncated by the token budget.
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    # False forces a bounded single pass even when max_new_tokens is omitted.
    unbounded: Optional[bool] = None
    # Explicit dispatch task key (e.g. "image-text-to-text"); wins over the
    # model's primary_task and over the text-only auto-routing below.
    task: Optional[str] = None
    # Client-supplied id so a chat can be cancelled mid-stream
    # (POST /api/llm/chat/cancel/<request_id>).
    request_id: Optional[str] = None
    # Dedicated worker pool override. The route resolves the effective pool from
    # the API key's bound pool (default) + this field (override, if the key
    # allows it). "" / None = the general pool.
    pool: Optional[str] = None
    # Per-request allocation triggers (alloc_mode / bnb_4bit / n_cpu_moe /
    # n_gpu_layers / gpu_mem_gib / cpu_mem_gib / threads) — overlaid on the
    # designation spill for this call only. Whitelisted + wire-scrubbed in
    # managers.resolvers.remote._relay_payload; see ChatRequest.alloc.
    alloc: Optional[dict] = None
    # Attribution for the F5 job record (unified jobs view): which transport
    # originated this (web | discord | cli | v1) and its conversational
    # context id (discord channel, web session). Never used for routing.
    transport: Optional[str] = None
    channel: Optional[str] = None
    # Resolved server-side in chat_stream() from the request's credential and
    # ALWAYS overwritten there — never trusted from the client body.
    principal: Optional[str] = None

    @field_validator("channel", "request_id", "transport", mode="before")
    @classmethod
    def _stringify_scalar_ids(cls, v):
        """Tolerant reader for id-ish fields: the Discord bot sends `channel`
        as a raw snowflake INT (1522465972311818300) and pydantic v2 does not
        coerce int→str — every bot chat 500'd on validation. Attribution ids
        are opaque strings to us; accept any scalar and stringify it."""
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            return str(int(v))
        return str(v)

    @model_validator(mode="after")
    def _require_one_input(self):
        if not self.prompt and not self.messages:
            raise ValueError("ChatBody needs either 'prompt' or 'messages'")
        return self
    
class Message(BaseModel):
    role: str
    content: str
    images: List[str] | None = None
    file: str | None = None     # server path from /api/uploads
