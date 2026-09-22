from typing import AsyncIterator, Protocol, Union, runtime_checkable
from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, StatusEvent, TokenEvent
from hugpy_engine.schemas.task_schemas import TaskRequest, TaskResult
# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Runner(Protocol):
    """The contract every runner implements.

    request_type / result_type are class attributes (not instance attrs)
    so the dispatch layer can ask 'what shape does this runner expect?'
    without instantiating anything.

    .run() is required. .stream() is optional — runners that don't support
    streaming should raise NotImplementedError so the route layer can
    return a clean 4xx instead of a 500.
    """

    request_type: type[TaskRequest]
    result_type: type[TaskResult]
    model_key: str

    async def run(self, req: TaskRequest) -> TaskResult: ...

    async def stream(
        self,
        req: TaskRequest,
        cancel_event,
    ) -> AsyncIterator: ...
    
StreamEvent = Union[TokenEvent, DoneEvent, ErrorEvent, StatusEvent]
