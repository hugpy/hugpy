"""The wire protocol shared by the phone worker and the orchestrator.

The worker emits detections as plain text and the orchestrator parses that text
back into :class:`Detection` objects. Historically the formatting (worker) and
the parsing regex (orchestrator/"coord") lived in separate files and had to be
kept in lockstep by hand. Centralising both here makes the line shape a single
contract: change :data:`DETECTION_LINE_FMT` and :func:`format_detections` and
:func:`parse_detections` move together.

Wire shape::

    yolo <url> (WxH)
      <class>      conf=0.NNN bbox=(x1,y1)-(x2,y2)
      <class>      conf=0.NNN bbox=(x1,y1)-(x2,y2)

The header line carries the source URL and original dimensions; each indented
line is one detection.
"""
from __future__ import annotations

import re

from hugpy_fleet.phone_brick.schemas import Detection

# Verb names accepted by the worker. ``yolo`` is the inference verb; the rest
# are small text/utility helpers carried over from the field workers.
VERB_YOLO = "yolo"
VERB_FETCH = "fetch"
VERB_SHELL = "sh"
VERB_COUNT = "count"
VERB_UPPER = "upper"
VERB_LOWER = "lower"
VERB_REVERSE = "reverse"

# Detection line: two leading spaces, left-justified class, conf, bbox.
DETECTION_LINE_FMT = "  {cls:12s} conf={conf:.3f} bbox=({x1},{y1})-({x2},{y2})"

# Primary parser: class + conf + bbox. Matches the line emitted above.
_DETECTION_RE = re.compile(
    r"^\s+(\S+)\s+conf=([0-9.]+)\s+bbox=\((-?\d+),(-?\d+)\)-\((-?\d+),(-?\d+)\)",
    re.MULTILINE,
)

# Fallback parser: class + conf only (no bbox -> not drawable). Lets the
# orchestrator still record a verdict from a bbox-less worker response.
_DETECTION_NOBBOX_RE = re.compile(r"^\s+(\S+)\s+conf=([0-9.]+)\b", re.MULTILINE)


def format_header(url: str, width: int, height: int) -> str:
    return f"{VERB_YOLO} {url} ({width}x{height})"


def format_no_detections(url: str, conf_threshold: float) -> str:
    return f"{VERB_YOLO} {url}\nNo detections >= {conf_threshold}"


def format_detections(url: str, width: int, height: int,
                      detections: list[Detection]) -> str:
    """Render a header plus one line per detection, in worker wire format."""
    lines = [format_header(url, width, height)]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        lines.append(DETECTION_LINE_FMT.format(
            cls=det.cls, conf=det.conf, x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return "\n".join(lines)


def parse_detections(text: str) -> list[Detection]:
    """Parse worker output text back into :class:`Detection` objects.

    Tries the bbox form first; if nothing matches, falls back to the
    class+conf-only form (those detections carry a zero bbox and so won't be
    drawn, but still count toward consensus).
    """
    if not text:
        return []

    detections: list[Detection] = []
    for m in _DETECTION_RE.finditer(text):
        detections.append(Detection(
            cls=m.group(1),
            conf=float(m.group(2)),
            bbox=(int(m.group(3)), int(m.group(4)),
                  int(m.group(5)), int(m.group(6))),
        ))
    if detections:
        return detections

    for m in _DETECTION_NOBBOX_RE.finditer(text):
        detections.append(Detection(
            cls=m.group(1),
            conf=float(m.group(2)),
            bbox=(0, 0, 0, 0),
        ))
    return detections
