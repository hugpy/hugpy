from pydantic import BaseModel
from hugpy_platform.constants import (
    DEFAULT_ROOT,
    HF_HOME,
    HF_HUB_CACHE,
    MODELS_DICT_PATH,
    PIP_CACHE_DIR,
    TORCH_HOME,
)
class Settings(BaseModel):
    storage_root: str = DEFAULT_ROOT
    manifest_path: str = MODELS_DICT_PATH

    @property
    def hf_home(self) -> str:
        return HF_HOME

    @property
    def hf_hub_cache(self) -> str:
        return HF_HUB_CACHE

    @property
    def torch_home(self) -> str:
        return TORCH_HOME

    @property
    def pip_cache_dir(self) -> str:
        return PIP_CACHE_DIR


settings = Settings()
