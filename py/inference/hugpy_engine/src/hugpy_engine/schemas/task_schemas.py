from typing import Any, Callable, Dict, Optional, Tuple, Type
from pydantic import BaseModel, ConfigDict
class TaskRequest(BaseModel):
    """Marker base for all task-family request schemas.

    request_id and model_key are universal — every request we route needs
    to identify itself and say which model to send to. Everything else
    (messages, audio_path, prompt, etc.) lives on the subclass.
    """
    model_config = ConfigDict(extra="forbid")
    request_id: str
    model_key: str
    # Dedicated worker pool for routing (None/"" = general). Universal so any
    # task can be reserved to an app's pool — see DelegatingRunner._select.
    pool: Optional[str] = None


class TaskResult(BaseModel):
    """Marker base for all task-family result schemas.

    `ok` and `error` are part of the base so the route layer can return
    a consistent envelope regardless of which runner produced the result.
    Successful runs leave `error=None`; failures set ok=False and put the
    message in error.
    """
    model_config = ConfigDict(extra="allow")
    request_id: str
    model_key: str
    ok: bool = True
    error: Optional[str] = None
# ---------------------------------------------------------------------------
# Resolution — the contract between resolution and execution.
# ---------------------------------------------------------------------------

class Resolution(BaseModel):
    """Frozen decision object. Everything downstream reads from here."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model_key: str
    framework: str
    task: str                 # effective task, NOT cfg.primary_task
    cfg: Any                  # ModelConfig
    builder: Callable[[Dict[str, Any], str], BaseModel]
    runner_cls: Type          # Runner subclass
    cache_key: Tuple[str, str]   # (model_key, task)
