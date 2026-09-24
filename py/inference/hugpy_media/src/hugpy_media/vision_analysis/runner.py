"""Vision-analysis runners — ONE generic transformers-pipeline runner.

Serves the image-analysis task family:

    ("transformers", "depth-estimation")      DepthEstimationRunner
    ("transformers", "object-detection")      ObjectDetectionRunner
    ("transformers", "image-classification")  ImageClassificationRunner
    ("transformers", "image-segmentation")    ImageSegmentationRunner

All four are the same runner with a different ``pipeline_task`` class attr:
transformers.pipeline(task, model=dir) covers the whole family, so adding the
next HF task is a two-line subclass + registry rows — not a new runner. The
pipeline cache / lazy import / thread-offload pattern mirrors ImageGenRunner.

Image OUTPUTS (the depth map, segmentation masks) are saved under
UPLOADS_HOME/generated and returned as imagegen GeneratedImage rows, so the
console renders them exactly like generated images.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
from typing import Any, Dict, List

from hugpy_engine.schemas.event_schemas import DoneEvent, ErrorEvent, TokenEvent
from hugpy_platform.constants import UPLOADS_HOME
from hugpy_storage.download_models import ensure_model

from hugpy_media.imagegen.schemas import GeneratedImage
from hugpy_media.vision_analysis.schemas import VisionAnalysisRequest, VisionAnalysisResult

logger = logging.getLogger(__name__)


from hugpy_platform.runner_config import RunnerConfig


class VisionAnalysisRunner(RunnerConfig):
    """Generic transformers-pipeline runner for single-image analysis tasks."""

    request_type = VisionAnalysisRequest
    result_type = VisionAnalysisResult
    pipeline_task: str = ""          # set by the per-task subclass

    _PIPELINES: Dict[tuple, Any] = {}
    _LOCK = threading.Lock()


    # --- pipeline loading (lazy, singleton) ---------------------------------

    @property
    def pipeline(self):
        key = (self.model_key, self.pipeline_task)
        cached = self._PIPELINES.get(key)
        if cached is not None:
            return cached

        with self._LOCK:
            cached = self._PIPELINES.get(key)
            if cached is not None:
                return cached

            try:
                import torch
                from transformers import pipeline as hf_pipeline
            except ImportError as exc:
                raise RuntimeError(
                    f"transformers + torch are required for {self.pipeline_task} "
                    "tasks but are not installed. `pip install "
                    "'abstract_hugpy_dev[transformers]'`."
                ) from exc

            model_dir = ensure_model(self.model_key)
            cuda = torch.cuda.is_available()
            pipe = hf_pipeline(
                self.pipeline_task,
                model=model_dir,
                device=0 if cuda else -1,
            )
            logger.info(
                "VisionAnalysisRunner: loaded task=%s model=%s dir=%s device=%s",
                self.pipeline_task, self.model_key, model_dir,
                "cuda" if cuda else "cpu",
            )
            self._PIPELINES[key] = pipe
            return pipe

    # --- input / output helpers ---------------------------------------------

    def _load_image(self, req: VisionAnalysisRequest):
        from PIL import Image
        if req.image_path:
            return Image.open(req.image_path).convert("RGB")
        return Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")

    def _save_image(self, image, req: VisionAnalysisRequest, suffix: str) -> GeneratedImage:
        out_dir = os.path.join(UPLOADS_HOME, "generated")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{req.request_id}_{suffix}.png")
        image.save(path, format="PNG")
        b64 = None
        if req.return_b64:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return GeneratedImage(path=path, b64=b64,
                              width=image.width, height=image.height)

    # --- per-task result shaping ---------------------------------------------

    def _analyze(self, req: VisionAnalysisRequest) -> VisionAnalysisResult:
        """Blocking analysis. Called from a worker thread by .run()."""
        image = self._load_image(req)
        call_kwargs: Dict[str, Any] = {}
        if req.top_k is not None and self.pipeline_task == "image-classification":
            call_kwargs["top_k"] = req.top_k
        if req.threshold is not None and self.pipeline_task == "object-detection":
            call_kwargs["threshold"] = req.threshold
        if req.candidate_labels is not None:
            call_kwargs["candidate_labels"] = req.candidate_labels

        output = self.pipeline(image, **call_kwargs)

        items: List[Dict[str, Any]] = []
        images: List[GeneratedImage] = []
        task = self.pipeline_task

        if task == "depth-estimation":
            # {"predicted_depth": tensor, "depth": PIL.Image}
            depth_img = output["depth"] if isinstance(output, dict) else output
            images.append(self._save_image(depth_img, req, "depth"))
            try:
                tensor = output.get("predicted_depth")
                items.append({
                    "min_depth": float(tensor.min()),
                    "max_depth": float(tensor.max()),
                    "mean_depth": float(tensor.mean()),
                })
            except Exception:
                pass
            text = f"depth map computed ({depth_img.width}x{depth_img.height}): {images[0].path}"

        elif task == "object-detection":
            for det in output:
                items.append({
                    "label": det.get("label"),
                    "score": round(float(det.get("score", 0.0)), 4),
                    "box": det.get("box"),
                })
            labels = ", ".join(f"{i['label']} ({i['score']:.2f})" for i in items[:10])
            text = f"{len(items)} object(s) detected" + (f": {labels}" if labels else "")

        elif task == "image-classification":
            for row in output:
                items.append({
                    "label": row.get("label"),
                    "score": round(float(row.get("score", 0.0)), 4),
                })
            labels = ", ".join(f"{i['label']} ({i['score']:.2f})" for i in items[:5])
            text = f"classified: {labels}" if labels else "no classification produced"

        elif task == "image-segmentation":
            for index, seg in enumerate(output):
                row: Dict[str, Any] = {
                    "label": seg.get("label"),
                    "score": (round(float(seg["score"]), 4)
                              if seg.get("score") is not None else None),
                }
                mask = seg.get("mask")
                if mask is not None:
                    art = self._save_image(mask, req, f"mask{index}")
                    row["mask_path"] = art.path
                    images.append(art)
                items.append(row)
            labels = ", ".join(str(i["label"]) for i in items[:10])
            text = f"{len(items)} segment(s)" + (f": {labels}" if labels else "")

        else:   # future task wired to this runner without a shaper
            items.append({"raw": repr(output)[:2000]})
            text = f"{task} produced {type(output).__name__}"

        return VisionAnalysisResult(
            request_id=req.request_id, model_key=req.model_key,
            ok=True, task=task, items=items, images=images, text=text,
        )

    # --- public API -----------------------------------------------------------

    async def run(self, req: VisionAnalysisRequest) -> VisionAnalysisResult:
        try:
            return await asyncio.to_thread(self._analyze, req)
        except Exception as exc:
            logger.exception(
                "VisionAnalysisRunner.run failed: task=%s model=%s req=%s",
                self.pipeline_task, self.model_key, req.request_id,
            )
            return VisionAnalysisResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, task=self.pipeline_task,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, req: VisionAnalysisRequest, cancel_event=None):
        """One-shot wrapped as a stream, mirroring ImageGenRunner."""
        result = await self.run(req)
        if result.ok:
            yield TokenEvent(request_id=req.request_id, text=result.text)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                            output_chunks=1, finish_reason="stop")
        else:
            yield ErrorEvent(request_id=req.request_id,
                             message=result.error or f"{self.pipeline_task} failed")


class DepthEstimationRunner(VisionAnalysisRunner):
    pipeline_task = "depth-estimation"


class ObjectDetectionRunner(VisionAnalysisRunner):
    pipeline_task = "object-detection"


class ImageClassificationRunner(VisionAnalysisRunner):
    pipeline_task = "image-classification"


class ImageSegmentationRunner(VisionAnalysisRunner):
    pipeline_task = "image-segmentation"
