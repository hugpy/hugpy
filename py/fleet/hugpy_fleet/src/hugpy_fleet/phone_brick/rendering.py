"""Draw detections onto an image, in a phone's assigned colour.

OpenCV (`cv2`) is imported lazily and treated as optional: orchestration still
works without it (you just don't get the annotated output image).
"""
from __future__ import annotations

from hugpy_fleet.phone_brick.schemas import Detection


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """``#58a6ff`` -> ``(255, 166, 88)`` in OpenCV's BGR order."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_detections(image_path: str, color_hex: str,
                    detections: list[Detection]) -> bool:
    """Draw boxes + labels onto ``image_path`` in place. Returns success.

    Detections with a zero bbox (e.g. bbox-less worker responses) are skipped.
    Returns ``False`` if cv2 is unavailable or the image can't be read.
    """
    try:
        import cv2
    except ImportError:
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    color = hex_to_bgr(color_hex)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if (x1, y1, x2, y2) == (0, 0, 0, 0):
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det.cls}_{det.conf_pct}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(image_path, img)
    return True
