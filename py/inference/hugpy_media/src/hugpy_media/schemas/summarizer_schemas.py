# imports/src/schemas/summarize_schemas.py
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional
from pydantic import ConfigDict, Field, field_validator
from hugpy_platform.utils import get_request_id
from hugpy_engine.schemas.task_schemas import TaskRequest, TaskResult

class InputPolicy(Enum):
    """
    Controls what happens when input text is too short to summarize
    meaningfully.

    STRICT  — raise ValueError.  The safe default.
    WARN    — return a prefixed warning + whatever the model produces.
    ALLOW   — pass it straight through.  You asked for fiction, you get fiction.
    """

    STRICT = "strict"
    WARN = "warn"
    ALLOW = "allow"


MIN_INPUT_WORDS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Internal request — the contract every back-end actually consumes.
#
# SummarizeRequest (below) is the wire/runner envelope: all knobs Optional,
# validated by pydantic, carries request_id/model_key. SummaryRequest is the
# resolved, concrete-defaults dataclass that summarize() builds and hands to a
# backend. Keeping them separate means the backend never has to cope with None
# knobs, and the public summarize() owns default resolution in exactly one place.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SummaryRequest:
    """Immutable bag of parameters every back-end understands."""

    text: str
    max_chunk_tokens: int = 450
    min_length: int = 100
    max_length: int = 512
    do_sample: bool = False
    summary_mode: Literal["short", "medium", "long", "auto"] = "medium"
    input_policy: InputPolicy = InputPolicy.STRICT
    min_input_words: int = MIN_INPUT_WORDS_DEFAULT

    # Consolidation pass (seq2seq backend merges chunk summaries into one).
    consolidation_min_length: int = 80
    consolidation_max_length: int = 160
    max_output_words: int = 150

    def check_input(self) -> Optional[str]:
        """Return a human-readable problem string if the input is suspect,
        or None if everything looks fine."""
        word_count = len(self.text.split())
        if word_count < self.min_input_words:
            return (
                f"Input has {word_count} word(s); need at least "
                f"{self.min_input_words} for a meaningful summary."
            )
        return None

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class SummarizeRequest(TaskRequest):
    """One summarization unit of work.

    Mirrors SummaryRequest fields from summarizers.py but flattens them into
    the TaskRequest envelope the runner protocol expects. Everything beyond
    `text` is optional — if both `preset` and the explicit knobs are None,
    the backend uses its built-in defaults.
    """
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=get_request_id)
    text: str = Field(min_length=1)

    preset: Optional[str] = None
    summary_mode: Optional[Literal["short", "medium", "long", "auto"]] = None
    input_policy: Optional[Literal["strict", "warn", "allow"]] = None

    # Generation knobs
    max_chunk_tokens: Optional[int] = Field(default=None, gt=0)
    min_length: Optional[int] = Field(default=None, gt=0)
    max_length: Optional[int] = Field(default=None, gt=0)
    do_sample: Optional[bool] = None
    min_input_words: Optional[int] = Field(default=None, ge=0)

    # Consolidation pass (T5/seq2seq backends)
    consolidation_min_length: Optional[int] = Field(default=None, gt=0)
    consolidation_max_length: Optional[int] = Field(default=None, gt=0)
    max_output_words: Optional[int] = Field(default=None, gt=0)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

FinishReason = Literal[
    "ok",                # summary produced normally
    "input_too_short",   # InputPolicy fired
    "truncated",         # max_output_words clipped the tail
    "error",             # backend raised; see TaskResult.error
]


class SummarizeResult(TaskResult):
    """What summarize() returns, made inspectable.

    `text` is the actual summary string and the only field most callers care
    about. The rest is provenance: which backend ran, what preset was applied,
    how the input was carved up, whether anything was clipped. Lets you tell
    'short summary because the source was short' from 'short summary because
    we hit max_output_words' without re-running.
    """
    model_config = ConfigDict(extra="forbid")

    # The thing the caller actually wanted
    text: str = ""

    # Provenance — answers "where did this come from"
    backend: str                              # "seq2seq_chunked" | "pipeline_chunked" | ...
    preset_used: Optional[str] = None         # None if no preset was applied

    # Shape of work
    chunks_processed: int = Field(default=0, ge=0)
    consolidation_passes: int = Field(default=0, ge=0)

    # Input/output sizing — handy for spot-checking summary ratios
    input_word_count: int = Field(default=0, ge=0)
    output_word_count: int = Field(default=0, ge=0)
    truncated: bool = False                   # True if max_output_words clipped

    # Why the run ended
    finish_reason: FinishReason = "ok"

    # If InputPolicy.WARN or .ALLOW let a problem-input run through,
    # the human-readable explanation lands here. None on clean runs.
    input_warning: Optional[str] = None

    @field_validator("output_word_count", mode="before")
    @classmethod
    def _derive_output_count(cls, v, info):
        # Convenience: if the caller didn't set output_word_count,
        # derive it from `text` so callers don't have to remember.
        if v == 0 and info.data.get("text"):
            return len(info.data["text"].split())
        return v

# ---------------------------------------------------------------------------
# Presets — named parameter bundles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SummaryPreset:
    """
    A frozen bag of defaults that a preset name resolves to.
    Every field here maps 1:1 to a SummaryRequest field.
    Only non-None values override the caller's explicit kwargs.
    """

    max_chunk_tokens: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    do_sample: Optional[bool] = None
    summary_mode: Optional[Literal["short", "medium", "long", "auto"]] = None
    input_policy: Optional[InputPolicy] = None
    min_input_words: Optional[int] = None
    consolidation_min_length: Optional[int] = None
    consolidation_max_length: Optional[int] = None
    max_output_words: Optional[int] = None
