from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict
from hugpy_platform.constants import FINISH_REASONS
class TokenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["token"] = "token"
    request_id: str
    text: str

class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["done"] = "done"
    request_id: str
    input_tokens: int
    output_chunks: int
    finish_reason: FINISH_REASONS
    # Token accounting for the finished stream, when the producer knows it:
    # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    # (engine-reported or tokenizer-counted). Optional + additive — every
    # existing constructor call is unchanged and consumers must treat None as
    # "unavailable". /v1 threads this into the OpenAI `usage` object.
    usage: Optional[dict] = None
    # THE ENGINE'S OWN DECODE RATE (operator, 2026-07-25). llama-server reports
    # a `timings` block — `predicted_per_second` is authoritative tok/s measured
    # by the engine, not wall-clock arithmetic of ours (which would include
    # queueing/network/prompt-eval and would not be decode rate at all).
    # Verified in llama.cpp tools/server/server-task.cpp: the FINAL streaming
    # chunk carries it unconditionally, so streaming and one-shot both have it.
    #
    # Optional + additive, exactly like `usage` above: every existing
    # constructor call is unchanged, and a producer that omits it leaves None.
    # Consumers must treat None as "unavailable" and degrade, never guess.
    #
    # ⚠ RECORDING ONLY — nothing ranks on this; eviction.sort_key is untouched.
    timings: Optional[dict] = None

class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    request_id: str
    message: str

class StatusEvent(BaseModel):
    """Out-of-band passthrough event — provisioning progress, continuation
    segment markers, and anything a remote worker emits that isn't a
    token/done/error. ``extra="allow"`` so it can carry stage/message/progress/
    done_bytes/etc. without a rigid schema; ``type`` defaults to "status" but is
    overwritten when reconstructed from a worker line (e.g. type="request").
    The route serializer just model_dump()s these straight to the SSE wire."""
    model_config = ConfigDict(extra="allow")
    type: str = "status"
    request_id: str = ""
