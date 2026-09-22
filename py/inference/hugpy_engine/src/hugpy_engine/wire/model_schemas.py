from enum import Enum
from pydantic import BaseModel, Field
class Runtime(str, Enum):
    transformers = "transformers"
    gguf = "gguf"                 # HF Hub's library tag for GGUF repos
    dataset = "dataset"
    unknown = "unknown"


class FileSpec(BaseModel):
    path: str
    size: int | None = None

class ModelSpec(BaseModel):
    hub_id: str
    license: str | None = None
    gated: bool | str | None = None          # False | "auto" | "manual"
    last_modified: str | None = None
    total_bytes: int | None = None
    num_params: int | None = None             # from safetensors metadata
    context_length: int | None = None
    gguf_quants: list[str] = Field(default_factory=list)
    files: list[FileSpec] = Field(default_factory=list)




class ModelSearchResult(BaseModel):
    hub_id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = Field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    private: bool | None = None
    total_bytes: int | None = None
    last_modified: str | None = None
    created_at: str | None = None
    # Uploader trust tier for the ranking/badge: "first-party" (canonical org),
    # "community" (reputable repackager/quantizer), or None (unvetted). Curated
    # allowlist, not an official HF signal — see search_routes._TRUST_TIER1/2.
    trust: str | None = None
