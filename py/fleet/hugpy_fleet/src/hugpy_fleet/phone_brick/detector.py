"""ONNX YOLO detector — the single inference core for the phone workers.

Every field worker (`worker.py`, `worker-yolo-6class-v0.0.1.py`,
`worker-multiclass-v0.0.2.py`, the 2-class helmet variant, …) carried its own
copy of the same three things: the model load, non-max suppression, and the
letterbox-infer-rescale routine. This module is that core, parameterised by
:class:`DetectorConfig` so one class covers the 2-class helmet model, the
6-class PPE model, and any sidecar-defined multiclass model.

Heavy deps (`onnxruntime`, `numpy`, `Pillow`) are imported lazily so importing
this module — e.g. on a control box that only orchestrates — does not require
the inference stack a phone needs.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

from hugpy_fleet.phone_brick.schemas import Detection, DetectorConfig


# ---------------------------------------------------------------------------
# Class-list resolution (explicit > sidecar JSON > fallback)
# ---------------------------------------------------------------------------
def find_sidecar(model_path: str) -> str | None:
    """Locate a class-list JSON next to the ONNX model, if any.

    Search order:
        1. ``<model>.json``              (``model.onnx`` -> ``model.json``)
        2. ``<model_path>.json``         (``model.onnx.json``)
        3. ``<dir>/classes.json``        (a shared sidecar)
    """
    base, _ = os.path.splitext(model_path)
    candidates = [
        base + ".json",
        model_path + ".json",
        os.path.join(os.path.dirname(model_path), "classes.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_class_list(sidecar_path: str) -> list[str]:
    """Read class names from a sidecar JSON.

    Accepts either a top-level list or ``{"classes": [...]}``.
    """
    with open(sidecar_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("classes"), list):
        return [str(x) for x in data["classes"]]
    raise ValueError(f"sidecar {sidecar_path} has no usable 'classes' field")


def resolve_classes(config: DetectorConfig) -> tuple[list[str], str]:
    """Resolve the class list for a config. Returns ``(classes, source)``.

    ``source`` is a short human-readable description of where the list came
    from, surfaced by the worker's ``/status`` endpoint for debuggability.
    """
    if config.classes:
        return list(config.classes), "explicit"

    sidecar = find_sidecar(config.model_path)
    if sidecar:
        try:
            return load_class_list(sidecar), f"sidecar:{sidecar}"
        except Exception as exc:  # noqa: BLE001 — fall back, but report why
            return (list(config.fallback_classes),
                    f"fallback (sidecar {sidecar} failed: {exc})")
    return list(config.fallback_classes), "fallback (no sidecar found)"


# ---------------------------------------------------------------------------
# Non-max suppression
# ---------------------------------------------------------------------------
def non_max_suppression(boxes_xyxy, scores, iou_threshold: float = 0.5) -> list[int]:
    """Greedy NMS. Returns indices of the boxes to keep, highest score first."""
    import numpy as np

    if len(boxes_xyxy) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = ((boxes_xyxy[i, 2] - boxes_xyxy[i, 0])
                  * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1]))
        area_rest = ((boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0])
                     * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1]))
        overlap = inter / np.maximum(area_i + area_rest - inter, 1e-6)
        order = rest[overlap < iou_threshold]
    return keep


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class OnnxYoloDetector:
    """Lazily-loaded ONNX YOLOv8 detector that returns :class:`Detection` lists.

    The ONNX session is built on first use (``load``) and reused thereafter, so
    constructing a detector is cheap and the model only pays its load cost when
    a request actually arrives.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._session = None
        self._input_name: str | None = None
        self._classes: list[str] = []
        self._classes_source: str = "unloaded"

    # -- properties --------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def classes_source(self) -> str:
        return self._classes_source

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Build the ONNX session and resolve the class list (idempotent)."""
        if self._session is not None:
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.intra_op_num_threads
        self._session = ort.InferenceSession(
            self.config.model_path, opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._classes, self._classes_source = resolve_classes(self.config)

    # -- inference ---------------------------------------------------------
    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id]
        return f"cls{class_id}"

    def infer_bytes(self, image_bytes: bytes,
                    conf_threshold: float | None = None) -> list[Detection]:
        """Run detection on raw image bytes. Returns rescaled detections."""
        import numpy as np
        from PIL import Image

        self.load()
        conf = self.config.conf_threshold if conf_threshold is None else conf_threshold
        imgsz = self.config.imgsz

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        arr = np.array(img.resize((imgsz, imgsz)), dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        out = self._session.run(None, {self._input_name: arr})
        pred = out[0][0].T               # (num_boxes, 4 + num_classes)
        boxes = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = np.max(class_scores, axis=1)

        mask = confs >= conf
        boxes, class_ids, confs = boxes[mask], class_ids[mask], confs[mask]
        if len(class_ids) == 0:
            return []

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = non_max_suppression(xyxy, confs, self.config.iou_threshold)
        scale_x = orig_w / imgsz
        scale_y = orig_h / imgsz

        detections: list[Detection] = []
        for i in keep:
            detections.append(Detection(
                cls=self._class_name(int(class_ids[i])),
                conf=float(confs[i]),
                bbox=(int(x1[i] * scale_x), int(y1[i] * scale_y),
                      int(x2[i] * scale_x), int(y2[i] * scale_y)),
            ))
        return detections

    def infer_url(self, url: str, conf_threshold: float | None = None,
                  timeout: float = 20.0) -> tuple[list[Detection], int, int]:
        """Fetch an image URL and run detection.

        Returns ``(detections, width, height)`` so the caller can format a
        protocol header with the original dimensions.
        """
        from PIL import Image

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            image_bytes = resp.read()
        width, height = Image.open(io.BytesIO(image_bytes)).size
        detections = self.infer_bytes(image_bytes, conf_threshold)
        return detections, width, height
