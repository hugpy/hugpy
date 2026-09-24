"""hugpy-imagegen-v1: text-to-image grading ("generate an image of X").

Requests go through central's ``POST /ml/imagine`` amenity (task
text-to-image -> ``execute_prompt``; the body becomes
``hugpy_media.imagegen.schemas.ImageGenRequest``).  ``ImageGenRequest`` has no
``alloc`` field, so a worker pin would be silently dropped: placement stays
hugpy's and the benchmark grades an imagegen model ONCE (worker
"hugpy-placed"), not per worker lane.  The response is the dumped
``ImageGenResult``: ``{"ok", "images": [{"b64", "path", "width", "height",
"seed"}], "error"}``.

Scoring is deterministic PIL + numpy over the returned pixels: validity,
requested size, non-degeneracy, dominant colours per region, luminance
contrast and silhouette fill ratio.  Seed / steps / size are fixed so runs are
comparable.  6 tasks x 3 tiers = 18 points.

An OPTIONAL vision-model judge (a model already hot on the fleet, asked over
hugpy's own ``/ml/vision``) can be recorded per call under ``judge_<task>``; it never
counts toward the deterministic score unless the run asks for it.
"""
from __future__ import annotations

import base64
import io
import time

from .common import choice, retrying_request

NAME = "hugpy-imagegen-v1"
SEED = 1234
STEPS = 20
WIDTH = HEIGHT = 512
NEGATIVE = "text, watermark, signature, border, frame"


# -------------------------------------------------------------- analysis ----
def decode(b64):
    """base64 PNG/JPEG -> RGB PIL image, or None when it does not decode."""
    from PIL import Image
    try:
        raw = base64.b64decode(b64 or "", validate=False)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception:
        return None


def _arrays(image):
    import numpy as np
    hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
    lum = np.asarray(image.convert("L"), dtype=np.float64)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2], lum


def color_masks(image):
    """Named colour masks on PIL's 0-255 HSV scale."""
    h, s, v, _lum = _arrays(image)
    chroma = (s >= 90) & (v >= 80)
    return {
        "white": (v >= 200) & (s <= 40),
        "black": v <= 70,
        "red": chroma & ((h <= 12) | (h >= 240)),
        "orange": chroma & (h > 12) & (h <= 30),
        "yellow": chroma & (h > 30) & (h <= 50),
        "green": chroma & (h > 50) & (h <= 120),
        "blue": chroma & (h > 120) & (h <= 190),
        "purple": chroma & (h > 190) & (h < 240),
    }


def _box(shape, x0, y0, x1, y1):
    """Fractional box -> numpy slices."""
    rows, cols = shape
    return slice(int(y0 * rows), max(int(y0 * rows) + 1, int(y1 * rows))), \
        slice(int(x0 * cols), max(int(x0 * cols) + 1, int(x1 * cols)))


REGIONS = {
    "center": (0.35, 0.35, 0.65, 0.65),
    "inner": (0.25, 0.25, 0.75, 0.75),
    "left": (0.05, 0.1, 0.45, 0.9), "right": (0.55, 0.1, 0.95, 0.9),
    "top": (0.1, 0.03, 0.9, 0.28), "middle": (0.1, 0.38, 0.9, 0.62), "bottom": (0.1, 0.72, 0.9, 0.97),
}


def region_fraction(image, color, region):
    masks = color_masks(image)
    mask = masks[color]
    if region == "border":
        import numpy as np
        edge = np.zeros(mask.shape, dtype=bool)
        k = max(1, int(0.08 * min(mask.shape)))
        edge[:k, :] = edge[-k:, :] = edge[:, :k] = edge[:, -k:] = True
        return float(mask[edge].mean())
    return float(mask[_box(mask.shape, *REGIONS[region])].mean())


def dominant(image, region):
    masks = color_masks(image)
    fractions = {name: float(m[_box(m.shape, *REGIONS[region])].mean()) for name, m in masks.items()}
    best = max(fractions, key=fractions.get)
    return best, fractions[best]


def region_luminance(image, region):
    import numpy as np
    lum = _arrays(image)[3]
    if region == "border":
        k = max(1, int(0.08 * min(lum.shape)))
        return float(np.median(np.concatenate([lum[:k].ravel(), lum[-k:].ravel(), lum[:, :k].ravel(), lum[:, -k:].ravel()])))
    return float(np.median(lum[_box(lum.shape, *REGIONS[region])]))


def stats(image):
    """Luminance std-dev, histogram entropy (bits) and a noise ratio.

    noise ratio = mean |adjacent-pixel difference| / std.  i.i.d. noise sits
    near 2/sqrt(pi) ~= 1.13; any real picture is far below 0.5.
    """
    import numpy as np
    lum = _arrays(image)[3]
    std = float(lum.std())
    hist = np.bincount(lum.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    diff = (np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()) / 2
    return {"std": std, "entropy": entropy, "noise_ratio": float(diff / std) if std > 1e-6 else 0.0}


def silhouette(image, dark=100):
    """Fill ratio + aspect of the dark mask's robust bounding box."""
    import numpy as np
    mask = _arrays(image)[3] < dark
    coverage = float(mask.mean())
    if coverage < 0.01:
        return {"coverage": coverage, "fill": 0.0, "aspect": 0.0}
    ys, xs = np.nonzero(mask)
    y0, y1 = np.percentile(ys, [0.5, 99.5]); x0, x1 = np.percentile(xs, [0.5, 99.5])
    h, w = max(1.0, y1 - y0 + 1), max(1.0, x1 - x0 + 1)
    inside = mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
    return {"coverage": coverage, "fill": float(inside.mean()), "aspect": float(w / h)}


# -------------------------------------------------------------- checkers ----
def _rule(text):
    def mark(fn):
        fn.rule = text
        return fn
    return mark


def decodes(image):
    return image is not None and min(image.size) >= 64


def not_degenerate(image, min_std=8.0):
    return decodes(image) and stats(image)["std"] >= min_std


def not_noise(image, max_ratio=0.5):
    return not_degenerate(image) and stats(image)["noise_ratio"] < max_ratio


def check_render(level):
    @_rule({1: "decodes, lum std>=8", 2: "+ noise_ratio<0.5", 3: "+ entropy>=5 bits"}[level])
    def check(image):
        if level == 1:
            return not_degenerate(image)
        if level == 2:
            return not_noise(image)
        return not_noise(image) and stats(image)["entropy"] >= 5.0
    return check


def check_size(width, height):
    @_rule(f"size == {width}x{height} and lum std>=8")
    def check(image):
        return decodes(image) and image.size == (width, height) and not_degenerate(image)
    return check


def check_figure(figure, background, figure_region="center", min_figure=0.4, min_background=0.5):
    @_rule(f"{figure_region} {figure}>={min_figure}, border {background}>={min_background}")
    def check(image):
        return (decodes(image) and region_fraction(image, figure, figure_region) >= min_figure
                and region_fraction(image, background, "border") >= min_background)
    return check


def check_small_figure(figure, background):
    @_rule(f"border {background}>=0.5, inner {figure}>=0.03")
    def check(image):
        return (decodes(image) and region_fraction(image, background, "border") >= 0.5
                and region_fraction(image, figure, "inner") >= 0.03)
    return check


def check_regions(layout, min_share=0.3):
    @_rule("dominant colour per region: " + ", ".join(f"{r}={c}" for r, c in layout.items()))
    def check(image):
        if not decodes(image):
            return False
        for region, color in layout.items():
            best, share = dominant(image, region)
            if best != color or share < min_share:
                return False
        return True
    return check


def check_lum_split(darker, lighter, margin):
    @_rule(f"median lum {darker} <= {lighter} - {margin}")
    def check(image):
        return decodes(image) and region_luminance(image, darker) <= region_luminance(image, lighter) - margin
    return check


def check_gray_contrast(margin=30):
    @_rule(f"center <= border - {margin} lum, border not saturated")
    def check(image):
        if not decodes(image):
            return False
        import numpy as np
        _h, s, _v, _l = _arrays(image)
        k = max(1, int(0.08 * min(s.shape)))
        border_sat = float(np.median(np.concatenate([s[:k].ravel(), s[-k:].ravel()])))
        return border_sat < 60 and region_luminance(image, "center") <= region_luminance(image, "border") - margin
    return check


def check_silhouette(lo, hi):
    @_rule(f"dark-mask fill in [{lo},{hi}], aspect 0.75-1.33, coverage 3-80%")
    def check(image):
        if not decodes(image):
            return False
        shape = silhouette(image)
        return 0.03 <= shape["coverage"] <= 0.8 and lo <= shape["fill"] <= hi and 0.75 <= shape["aspect"] <= 1.33
    return check


def _spec(prompt, subject, width=WIDTH, height=HEIGHT):
    return {"prompt": prompt, "subject": subject, "width": width, "height": height}


TASKS = {
    "render": (
        ("easy", _spec("a photograph of a mountain landscape at sunset", "a mountain landscape"), check_render(1)),
        ("medium", _spec("a photograph of a bowl of fruit on a wooden table", "a bowl of fruit"), check_render(2)),
        ("hard", _spec("a busy city street at night with neon signs, highly detailed photograph", "a city street at night"), check_render(3)),
    ),
    "size": (
        ("easy", _spec("a red apple on a table", "an apple", 512, 512), check_size(512, 512)),
        ("medium", _spec("a wide panoramic view of a sandy beach", "a beach", 768, 512), check_size(768, 512)),
        ("hard", _spec("a tall lighthouse on a cliff, portrait orientation", "a lighthouse", 512, 768), check_size(512, 768)),
    ),
    "color": (
        ("easy", _spec("a plain red square on a white background, flat color, minimal", "a red square on a white background"),
         check_figure("red", "white")),
        ("medium", _spec("a plain blue circle on a white background, flat color, minimal", "a blue circle on a white background"),
         check_figure("blue", "white")),
        ("hard", _spec("a small yellow star in the middle of a solid green background, flat colors, minimal",
                       "a yellow star on a green background"), check_small_figure("yellow", "green")),
    ),
    "layout": (
        ("easy", _spec("left half solid black, right half solid white, flat two-tone abstract image",
                       "a black left half and a white right half"), check_lum_split("left", "right", 60)),
        ("medium", _spec("left half solid blue, right half solid green, flat two-color abstract image",
                         "a blue left half and a green right half"), check_regions({"left": "blue", "right": "green"})),
        ("hard", _spec("three horizontal stripes, red on top, white in the middle, blue on the bottom, flat flag design",
                       "red, white and blue horizontal stripes"),
         check_regions({"top": "red", "middle": "white", "bottom": "blue"})),
    ),
    "contrast": (
        ("easy", _spec("a black circle on a white background, flat, minimal", "a black circle on white"),
         check_lum_split("center", "border", 80)),
        ("medium", _spec("a white circle on a black background, flat, minimal", "a white circle on black"),
         check_lum_split("border", "center", 80)),
        ("hard", _spec("a dark gray square on a light gray background, flat, minimal", "a dark gray square on light gray"),
         check_gray_contrast(30)),
    ),
    "shape": (
        ("easy", _spec("a solid black square centered on a white background, flat vector icon", "a black square"),
         check_silhouette(0.85, 1.0)),
        ("medium", _spec("a solid black circle centered on a white background, flat vector icon", "a black circle"),
         check_silhouette(0.68, 0.88)),
        ("hard", _spec("a solid black triangle centered on a white background, flat vector icon", "a black triangle"),
         check_silhouette(0.38, 0.62)),
    ),
}


# ------------------------------------------------------------- transport ----
IMAGINE_PATH = "/ml/imagine"


def body(lane, variation, spec):
    """/ml/imagine body (ImageGenRequest kwargs). Fixed seed/steps/size; no pin."""
    return {"model": lane["model"], "prompt": spec["prompt"],
            "negative_prompt": NEGATIVE, "width": spec.get("width", WIDTH), "height": spec.get("height", HEIGHT),
            "num_inference_steps": STEPS, "seed": SEED, "num_images": 1, "return_b64": True}


def call(client, lane, variation, spec, tokens=None):
    from ..fleet_grading import FleetError
    if isinstance(spec, str):
        spec = _spec(spec, spec)
    started = time.time()
    response, error = retrying_request(client, IMAGINE_PATH, body(lane, variation, spec), FleetError)
    elapsed = time.time() - started
    image, raw_bytes = None, None
    if error is None:
        if not isinstance(response, dict) or response.get("ok") is False or response.get("error"):
            error = "imagegen error: " + str((response or {}).get("error") if isinstance(response, dict) else response)[:500]
        else:
            images = response.get("images") or []
            b64 = images[0].get("b64") if images and isinstance(images[0], dict) else None
            if not b64:
                error = "imagegen returned no base64 image"
            else:
                raw_bytes = len(base64.b64decode(b64, validate=False))
                image = decode(b64)
    output = (f"{image.size[0]}x{image.size[1]} {raw_bytes} bytes" if image is not None
              else "undecodable image" if raw_bytes else "")
    return {"answer": image, "output": output, "error": error, "elapsed_s": round(elapsed, 4),
            "tok_s": None, "ctx_in": None, "ctx_out": None, "image_bytes": raw_bytes,
            "s_per_image": round(elapsed, 4) if image is not None else None,
            "images_per_s": round(1.0 / elapsed, 6) if image is not None and elapsed > 0 else None}


def score(response, checker):
    image = response.get("answer") if isinstance(response, dict) else response
    try:
        return bool(checker(image))
    except Exception:
        return False


# ----------------------------------------------------------------- judge ----
JUDGE_LABEL = "vision-model judge (non-deterministic, advisory)"


def judge(client, judge_target, response, spec, tokens=16):
    """Ask a HOT vision model whether the image shows the subject (yes/no).

    ``judge_target`` = {"model", "worker"} of a model already resident; the
    call pins that worker so it can never trigger a cold load elsewhere.
    """
    from ..fleet_grading import FleetError
    from .vision import VISION_PATH, answer_of, data_url
    image = response.get("answer") if isinstance(response, dict) else None
    if image is None:
        return {"model": judge_target["model"], "label": JUDGE_LABEL, "answer": "", "pass": False, "error": "no image"}
    question = f"Does this image show {spec['subject']}? Answer yes or no."
    body = {"model": judge_target["model"], "max_new_tokens": tokens, "max_chunks": 1, "temperature": 0,
            "messages": [{"role": "user", "content": question + " /no_think"}],
            "images": [data_url(image)], "alloc": {"worker": judge_target["worker"]}}
    reply, error = retrying_request(client, VISION_PATH, body, FleetError)
    answer = answer_of(reply) if reply is not None else ""
    return {"model": judge_target["model"], "label": JUDGE_LABEL, "question": question, "answer": answer,
            "pass": not error and choice("yes", ("yes", "no"))(answer), "error": error}


SEAT = _spec("a plain white square", "a white square")
SPEED = None  # throughput comes from the graded generations themselves (s/image)
