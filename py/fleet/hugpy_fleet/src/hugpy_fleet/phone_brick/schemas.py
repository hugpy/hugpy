"""Typed data structures for the phone-brick PPE-detection mechanic.

These model the three layers of the system:

* :class:`Detection` — one object a worker found in an image.
* :class:`DetectorConfig` / :class:`WorkerConfig` — how a phone worker loads its
  ONNX model and serves it.
* :class:`PhoneSpec` / :class:`ChainConfig` / :class:`PhaseResult` /
  :class:`ChainResult` — how the orchestrator fans one image across a chain of
  phones and collates their verdicts.

Confidence is stored canonically as a float in ``[0, 1]`` everywhere; the
``conf_pct`` helper exposes the integer-percent form used in rendered labels
and in the orchestrator's result filenames.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

#: Default ONNX YOLO vision model a phone worker loads when none is configured.
#: This is the 6-class PPE detector shipped to field phones; ``config_from_env``
#: and direct :class:`DetectorConfig` construction both fall back to it.
DEFAULT_MODEL_PATH = os.path.expanduser("~/phone-brick/ppe-tanishjain-6class.onnx")

try:
    from pydantic import BaseModel, ConfigDict, Field
    _HAS_PYDANTIC = True
except ImportError:
    # pydantic-core has no wheels for some worker platforms (e.g. Termux on
    # Android), so phone workers fall back to plain frozen dataclasses. Nothing
    # in this package uses pydantic beyond construction + attribute access.
    _HAS_PYDANTIC = False


class RunCancelled(Exception):
    """Raised inside a chain run when a caller asks to cancel it mid-flight.

    The orchestrator checks an optional ``cancel_check`` callable between phones
    and while waiting on a phone; when it returns truthy, this propagates out so
    the caller can record the run as cancelled rather than failed.
    """


# ---------------------------------------------------------------------------
# Detections (the unit of inference output)
# ---------------------------------------------------------------------------
if _HAS_PYDANTIC:
    class Detection(BaseModel):
        """One detected object: a class label, a confidence, and a pixel box.

        ``bbox`` is ``(x1, y1, x2, y2)`` in the *original* image's pixel space
        (the detector rescales from the model's letterboxed input back to source
        coordinates before constructing this).
        """

        model_config = ConfigDict(frozen=True)

        cls: str = Field(min_length=1)
        conf: float = Field(ge=0.0, le=1.0)
        bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        @property
        def conf_pct(self) -> int:
            """Confidence as an integer percent, e.g. ``0.873 -> 87``."""
            return int(round(self.conf * 100))
else:
    @dataclass(frozen=True)
    class Detection:
        """Dataclass fallback for :class:`Detection` (see the pydantic branch)."""

        cls: str
        conf: float
        bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        @property
        def conf_pct(self) -> int:
            """Confidence as an integer percent, e.g. ``0.873 -> 87``."""
            return int(round(self.conf * 100))


# ---------------------------------------------------------------------------
# Worker-side configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectorConfig:
    """Everything the ONNX YOLO detector needs to load and run a model.

    ``classes`` is resolved at load time: an explicit list wins, otherwise a
    sidecar JSON next to the model is consulted, otherwise ``fallback_classes``
    is used. See :func:`abstract_hugpy_dev.phone_brick.detector.resolve_classes`.

    ``model_path`` defaults to :data:`DEFAULT_MODEL_PATH` so a worker constructed
    without an explicit path still has a vision model to load.
    """

    model_path: str = DEFAULT_MODEL_PATH
    classes: Optional[list[str]] = None
    fallback_classes: tuple[str, ...] = (
        "Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe",
    )
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    intra_op_num_threads: int = 2


@dataclass(frozen=True)
class WorkerConfig:
    """How a phone worker exposes its detector over HTTP."""

    detector: DetectorConfig
    host: str = "0.0.0.0"
    port: int = 5002
    # Arbitrary shell execution (the legacy ``sh`` verb) is a deliberate
    # footgun and stays opt-in. Enable only on a trusted LAN you control.
    enable_shell: bool = False


# ---------------------------------------------------------------------------
# Orchestrator-side configuration and results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhoneSpec:
    """One node in an inference chain: a name, where to reach it, how to draw it."""

    name: str
    host: str
    port: int = 5002
    color_hex: str = "#58a6ff"


@dataclass(frozen=True)
class ChainConfig:
    """Parameters for one fan-out over a chain of phones."""

    phones: list[PhoneSpec]
    file_server: str
    push_timeout_s: float = 5.0
    drain_timeout_s: float = 60.0
    drain_poll_s: float = 2.0


if _HAS_PYDANTIC:
    class PhaseResult(BaseModel):
        """What one phone reported for one image, plus the consensus flag for it."""

        model_config = ConfigDict(frozen=True)

        phone: str
        top_cls: str
        top_conf: float = Field(ge=0.0, le=1.0)
        detections: list[Detection] = Field(default_factory=list)
        consensus: str = "NOD"  # AGR | DIS | NOD
        timestamp: int = 0

        @property
        def top_conf_pct(self) -> int:
            return int(round(self.top_conf * 100))

    class ChainResult(BaseModel):
        """The collated verdict of a whole chain over one image."""

        model_config = ConfigDict(frozen=True)

        image: str
        phases: list[PhaseResult] = Field(default_factory=list)
        output_path: Optional[str] = None
else:
    @dataclass(frozen=True)
    class PhaseResult:
        """Dataclass fallback for :class:`PhaseResult` (see the pydantic branch)."""

        phone: str
        top_cls: str
        top_conf: float
        detections: list[Detection] = field(default_factory=list)
        consensus: str = "NOD"  # AGR | DIS | NOD
        timestamp: int = 0

        @property
        def top_conf_pct(self) -> int:
            return int(round(self.top_conf * 100))

    @dataclass(frozen=True)
    class ChainResult:
        """Dataclass fallback for :class:`ChainResult` (see the pydantic branch)."""

        image: str
        phases: list[PhaseResult] = field(default_factory=list)
        output_path: Optional[str] = None
