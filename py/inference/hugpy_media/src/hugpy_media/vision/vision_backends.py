import asyncio
from typing import Protocol, Callable
from hugpy_media.vision.schemas import VisionRequest, VisionResult, VisionBackendConfig
from hugpy_media.vision.vision_coder import get_vision_coder
from hugpy_platform.constants import VISION_HOST

class VisionBackend(Protocol):
    async def run(self, req: VisionRequest) -> VisionResult: ...


class InProcessBackend:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.vision = get_vision_coder(model_key=model_key)

    async def run(self, req: VisionRequest) -> VisionResult:
        text = await asyncio.to_thread(
            self.vision.analyze_image,
            image_path=req.image_path,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            max_tokens=req.max_tokens,
        )
        return VisionResult(
            request_id=req.request_id,
            model_key=req.model_key,
            text=text,
        )


class HttpBackend:
    def __init__(self, model_key: str, host: str, port: int, timeout_s: float):
        import aiohttp  # local import so inprocess users don't need aiohttp
        self._aiohttp = aiohttp
        self.model_key = model_key
        self.host = host
        self.url = f"http://{self.host}:{port}/analyze"

        self.timeout_s = timeout_s

    async def run(self, req: VisionRequest) -> VisionResult:
        timeout = self._aiohttp.ClientTimeout(total=self.timeout_s)
        async with self._aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(self.url, json=req.model_dump()) as r:
                body = await r.text()
                if r.status >= 400:
                    raise RuntimeError(
                        f"vision server returned {r.status} from {self.url}: {body}"
                    )
                payload = await r.json() if body else {}
        return VisionResult.model_validate(payload)


# ---- registry -------------------------------------------------------------

_BACKENDS: dict[str, Callable[[VisionBackendConfig], VisionBackend]] = {}


def register_backend(name: str):
    def deco(fn):
        if name in _BACKENDS:
            raise KeyError(f"backend {name!r} already registered")
        _BACKENDS[name] = fn
        return fn
    return deco


@register_backend("inprocess")
def _build_inprocess(cfg: VisionBackendConfig) -> VisionBackend:
    return InProcessBackend(model_key=cfg.model_key)


@register_backend("http")
def _build_http(cfg: VisionBackendConfig) -> VisionBackend:
    if cfg.port is None:
        raise ValueError("http backend requires cfg.port")
    return HttpBackend(
        model_key=cfg.model_key,
        host=cfg.host,
        port=cfg.port,
        timeout_s=cfg.timeout_s,
    )


def build_backend(cfg: VisionBackendConfig) -> VisionBackend:
    import os

    # A worker box has no separate vision server, so a manifest port would send
    # it to http://none:7002. HUGPY_VISION_INPROCESS=1 forces the in-process
    # backend (load + run the VL model on this GPU directly), regardless of any
    # configured port. The worker agent sets this; central leaves it unset and
    # keeps using its http vision server.
    force_inprocess = (os.environ.get("HUGPY_VISION_INPROCESS", "").strip().lower()
                       in ("1", "true", "yes", "on"))
    name = "inprocess" if (force_inprocess or cfg.port is None) else "http"
    if name not in _BACKENDS:
        raise KeyError(
            f"Unknown vision backend {name!r}. Registered: {list(_BACKENDS)}"
        )
    return _BACKENDS[name](cfg)
