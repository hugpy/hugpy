from __future__ import annotations
"""Chat-family request / result types.

Used by both DeepCoderChatRunner (transformers) and LlamaCppChatRunner
(llama_cpp). One schema, two backends. The schema doesn't know or care
which backend serves it.

Why a separate ChatMessage class instead of {role: str, content: str}:
    Pydantic validation at the boundary catches typos like 'rolle' or
    missing content before they hit the model loader. Cheap defense.
"""
"""Shared types for the runner protocol.

Every task family (chat, summarize, transcribe, vision, keyword, ...) defines:
    - a TaskRequest subclass describing its inputs
    - a TaskResult subclass describing its output
    - a Runner class implementing .run() and optionally .stream()

The route layer doesn't import any concrete runner — it goes through
runner_for(model_key) and operates on the Runner protocol only.

Naming:
    TaskRequest / TaskResult are deliberately not called BaseRequest /
    BaseResult so they don't collide with the dozen other "Base*" things
    that already exist in this codebase.

Streaming events:
    Reuses the StreamEvent / TokenEvent / DoneEvent / ErrorEvent types
    from the existing schema. Don't redefine them here — that's how you
    end up with two parallel event hierarchies.
"""
from typing import (
    Literal,
    Optional,
    Union,
    Mapping,
    Any,
    AsyncIterator,
    Protocol,
    runtime_checkable
    )

# in your schemas module
from pydantic import  (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
    )
from ..constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    FINISH_REASONS,
    ROLES,
    MIN_INPUT_WORDS_DEFAULT,
    DEFAULT_LOCAL_FILES_ONLY
    )
from hugpy_platform.utils import get_request_id, get_messages, get_message

from ..init_imports import *
