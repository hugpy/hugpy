"""Wire DTOs for download requests/jobs. Storage-owned (the download request
is a storage contract); pure pydantic, no Flask, no engine."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class Runtime(str, Enum):
    """The storage runtime family a download lands under (mirrors
    ``model_paths.RUNTIME_FAMILIES`` plus the hub's dataset/unknown tags)."""
    transformers = "transformers"
    gguf = "gguf"                 # HF Hub's library tag for GGUF repos
    misc = "misc"
    dataset = "dataset"
    unknown = "unknown"


class DownloadStatus(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"




class DownloadRequest(BaseModel):
    hub_id: str = Field(..., examples=["Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"])
    framework: Runtime | str = Field(default="transformers")
    task: str = Field(default="text-generation")
    filename: str | None = None
    include: str | list[str] | None = None
    repo_type: Literal["model", "dataset"] = "model"


class DownloadJob(BaseModel):
    job_id: str
    hub_id: str
    framework: str
    task: str
    destination: str
    status: DownloadStatus
    error: str | None = None
