import os.path as osp
import base64
import binascii
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataclasses import asdict, dataclass

from hugpy_platform.constants import (
    DEFAULT_LOCAL_FILES_ONLY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    VISION_HOST,
)
QWEN_PATCH = 28
QWEN_PIXELS_PER_TOKEN = QWEN_PATCH * QWEN_PATCH
BAD_PATH_STRINGS = frozenset({
    "",
    "[object object]",
    "undefined",
    "null",
    "none",
})


class VisionBackendConfig(BaseModel):
    """Where vision work goes. Built once at startup, reused for every request."""
    model_config = ConfigDict(frozen=True)

    model_key: str = Field(min_length=1)
    port: Optional[int] = Field(default=None, gt=0, le=65535)
    host: str = VISION_HOST
    timeout_s: float = Field(default=DEFAULT_TIMEOUT, gt=0)


class VisionRequest(BaseModel):
    """One unit of vision work. Built per call."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    pool: Optional[str] = None   # dedicated worker pool (routing); None = general
    prompt: str = "Analyze this image."
    max_new_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0, le=32768)
    max_tokens: Optional[int] = Field(default=None, gt=0)

    image_path: Optional[str] = None
    image_b64: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_image_source(self) -> "VisionRequest":
        sources = [s for s in (self.image_path, self.image_b64) if s]
        if len(sources) != 1:
            raise ValueError(
                "VisionRequest needs exactly one of image_path or image_b64; "
                f"got image_path={self.image_path!r}, "
                f"image_b64={'<bytes>' if self.image_b64 else None}"
            )
        if self.image_path is not None:
            cleaned = self.image_path.strip()
            if cleaned.lower() in BAD_PATH_STRINGS:
                raise ValueError(
                    f"image_path looks like a serialization artifact: {self.image_path!r}"
                )
            if not osp.exists(cleaned):
                raise FileNotFoundError(f"Image not found on server: {cleaned}")
        if self.image_b64 is not None:
            try:
                base64.b64decode(self.image_b64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise ValueError(f"image_b64 is not valid base64: {e}") from e
        return self


class VisionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    text: str
    error: Optional[str] = None


@dataclass(frozen=True)
class VisionCoderConfig:
    model_key: str
    model_dir: str
    device: str
    torch_dtype: object
    min_tokens: int = 16
    max_tokens: int = 128
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY

    # New options
    device_map: Optional[str] = "auto"
    gpu_max_memory: str = "5GiB"
    cpu_max_memory: str = "24GiB"
    use_cache: bool = False

    @property
    def min_pixels(self) -> int:
        return self.min_tokens * QWEN_PIXELS_PER_TOKEN

    @property
    def max_pixels(self) -> int:
        return self.max_tokens * QWEN_PIXELS_PER_TOKEN
