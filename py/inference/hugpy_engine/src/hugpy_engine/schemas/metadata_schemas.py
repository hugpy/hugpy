"""Model metadata enrichment. Disk first, Hub as gap-filler.

Each resolver returns a partial dict. The merger runs them in order and the
first non-None value for a field wins. `_sources` records which resolver
filled each field so you can audit later without rerunning.
"""
from dataclasses import asdict, dataclass
from typing import Any, List, Optional
# ---------------------------------------------------------------------------
# MetaData — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMetadata:
    """One row of everything we know. All Optional — partial fills are valid."""
    hub_id:                     Optional[str]       = None

    # declared identity — from the hugpy.json marker (authoritative) or the
    # storage-path layout (models/<family>/<task>/…). Carrying these here is
    # what stops discovery minting rows with DEFAULT task/framework for dirs
    # whose truth is already on disk (the 2026-07-05 sd-turbo phantoms).
    name:                       Optional[str]       = None
    framework:                  Optional[str]       = None
    tasks:                      Optional[List[str]] = None
    primary_task:               Optional[str]       = None

    # task / framework
    pipeline_tag:               Optional[str]       = None
    library_name:               Optional[str]       = None
    auto_model_class:           Optional[str]       = None
    # peft / adapter — set only for LoRA-style adapters
    peft_type:                  Optional[str]       = None   # "LORA", etc.
    base_model:                 Optional[str]       = None   # base_model_name_or_path
    # True when the dir holds ONLY adapter weights (k61): a delta, not a servable
    # model. Carried here so the verdict survives into the discovery row instead
    # of being re-guessed — or, as it was, read as a null task and defaulted to
    # text-generation.
    adapter:                    Optional[bool]      = None

    # architecture
    architectures:              Optional[List[str]] = None
    model_type:                 Optional[str]       = None
    torch_dtype:                Optional[str]       = None
    vocab_size:                 Optional[int]       = None

    # context window — keep these separate; they answer different questions
    max_position_embeddings:    Optional[int] = None   # architectural ceiling
    tokenizer_model_max_length: Optional[int] = None   # tokenizer's declared cap
    sliding_window:             Optional[int] = None
    rope_scaling:               Optional[dict] = None

    # size
    parameter_count:            Optional[int] = None

    # governance
    license:                    Optional[str] = None
    gated:                      Optional[bool] = None
    languages:                  Optional[List[str]] = None
    tags:                       Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
