from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Optional
from pydantic import model_validator
from hugpy_engine.schemas.metadata_schemas import ModelMetadata
from hugpy_platform.constants import DEFAULT_LOCAL_FILES_ONLY, DEFAULT_MAX_TOKENS
from hugpy_platform.utils import safe_dtype_name

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hub_id: str
    folder: str
    model_key: str
    framework: str
    tasks: list
    primary_task: str
    base_model: Optional[str] = None   # set => this row is a PEFT adapter on base_model
    model_max_length: int = DEFAULT_MAX_TOKENS
    filename: Optional[str] = None
    include: Optional[str] = None
    port: Optional[int] = None
    host: Optional[str] = None
    timeout_s: Optional[int] = 3600
    meta: Optional[ModelMetadata] = None
    extra: dict = field(default_factory=dict)   # everything it wasn't expecting

    def __init__(self, **kwargs):
        for f in fields(type(self)):
            if f.name == "extra":
                continue
            if f.name in kwargs:
                value = kwargs.pop(f.name)
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                raise TypeError(
                    f"ModelConfig missing required field {f.name!r}"
                )
            object.__setattr__(self, f.name, value)
        object.__setattr__(self, "extra", kwargs)   # leftovers, kept not dropped
        self.__post_init__()
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        if self.primary_task not in self.tasks:
            raise ValueError(
                f"{self.model_key}: primary_task={self.primary_task!r} "
                f"not in tasks={sorted(self.tasks)!r}"
            )



    @model_validator(mode="after")
    def _check_primary_in_tasks(self):
        if self.primary_task not in self.tasks:
            raise ValueError(
                f"{self.model_key}: primary_task={self.primary_task!r} "
                f"not in tasks={sorted(self.tasks)!r}"
            )
        return self

@dataclass(frozen=True)
class DeepCoderRuntime:
    model_dir: str
    device: str
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY
    max_new_tokens_cap: int = DEFAULT_MAX_TOKENS
    max_concurrent_generations: int = 1
    adapter_dir: Optional[str] = None     # the LoRA dir; None => plain base model

    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.max_new_tokens_cap,
            self.max_concurrent_generations,
            self.adapter_dir,            # distinct cache slot per adapter
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "torch_dtype": safe_dtype_name(self.torch_dtype),
            "use_quantization": self.use_quantization,
            "use_flash_attention": self.use_flash_attention,
            "local_files_only": self.local_files_only,
            "max_new_tokens_cap": self.max_new_tokens_cap,
            "max_concurrent_generations": self.max_concurrent_generations,
            "adapter_dir": self.adapter_dir
        }
