# vision_runner.py
# vision_runner.py
from hugpy_engine.schemas.event_schemas import DoneEvent, TokenEvent
from hugpy_media.vision.schemas import VisionRequest, VisionResult, VisionBackendConfig
from hugpy_media.vision.vision_backends import build_backend


class VisionRunner:
    request_type = VisionRequest
    result_type = VisionResult

    def __init__(self, cfg: VisionBackendConfig):
        self.cfg = cfg
        self.backend = build_backend(self.cfg)

    async def run(self, req: VisionRequest) -> VisionResult:
        if req.model_key != self.cfg.model_key:
            raise ValueError(
                f"VisionRunner bound to {self.cfg.model_key!r}, "
                f"got request for {req.model_key!r}"
            )
        return await self.backend.run(req)
    async def stream(self, req: VisionRequest, cancel_event=None):
        result = await self.run(req)
        yield TokenEvent(request_id=req.request_id, text=getattr(result, "text", "") or "")
        yield DoneEvent(request_id=req.request_id, input_tokens=0,
                        output_chunks=1, finish_reason="stop")
