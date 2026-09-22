from pydantic import BaseModel, Field

class InstallOption(BaseModel):
    id: str                                   # "gguf:Q4_K_M" | "transformers"
    label: str
    framework: str                            # transformers | llama_cpp
    filename: str | None = None
    include: list[str] | None = None
    total_bytes: int | None = None
    fits_disk: bool | None = None


class InstallOptions(BaseModel):
    hub_id: str
    task: str
    options: list[InstallOption] = Field(default_factory=list)
    recommended: str | None = None
