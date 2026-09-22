from pydantic import BaseModel, Field
# ──────────────────────────────────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────────────────────────────────
class HFRepoDownloadRequest(BaseModel):
    hub_id: str = Field(..., examples=["Qwen/Qwen2.5-VL-7B-Instruct"])
    framework: str = Field(default="transformers")
    task: str = Field(default="text-generation")
    filename: str | None = None
    include: str | list[str] | None = None
    name: str | None = None
    register: bool = True
    total_bytes: int | None = None   # chosen variant size, for the progress bar
