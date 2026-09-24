"""hugpy-vision-v1: image-text-to-text grading ("look at this image and answer").

Every test image is drawn deterministically with PIL at run time (no fixture
downloads, no randomness), sent as a base64 PNG data URI in the ``images`` list
of central's ``POST /ml/vision`` amenity (task image-text-to-text) with the
same per-request worker pin (``alloc.worker``) the text suite uses, and the
reply is judged by the deterministic string checkers in :mod:`.common`.
6 tasks x 3 tiers = 18 points.

Why not ``/v1/chat/completions`` with ``image_url`` content parts: the /v1
funnel (``v1_helpers._render_tool_messages``) downcasts every message content
to ``str`` and forwards no ``images`` key, so the image never reaches the
worker there.  ``/ml/vision`` hands ``images`` to ``execute_prompt`` where
``ChatRequest.images`` carries it to the runner's multimodal path.
"""
from __future__ import annotations

import base64
import io
import math
import time

from .common import choice, contains_normalized, number_is, retrying_request

NAME = "hugpy-vision-v1"
SIZE = 384
_WHITE = (255, 255, 255)
RGB = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 160, 60),
       "yellow": (240, 210, 20), "orange": (245, 130, 20), "purple": (130, 40, 170),
       "black": (0, 0, 0)}
COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown", "gray", "grey")
SIDES = ("left", "right")
SHAPES = ("circle", "square", "triangle", "hexagon", "rectangle", "star", "pentagon", "oval")


# --------------------------------------------------------------- drawing ----
def _canvas():
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (SIZE, SIZE), _WHITE)
    return image, ImageDraw.Draw(image)


def _shape(draw, kind, cx, cy, r, color):
    fill = RGB.get(color, color)
    if kind == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    elif kind == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill)
    elif kind == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    elif kind == "hexagon":
        draw.polygon([(cx + r * math.cos(math.pi / 3 * i), cy + r * math.sin(math.pi / 3 * i))
                      for i in range(6)], fill=fill)
    else:
        raise ValueError(kind)


def _scene(*shapes):
    """``shapes`` = (kind, cx, cy, r, color) tuples on a white canvas."""
    def render():
        image, draw = _canvas()
        for spec in shapes:
            _shape(draw, *spec)
        return image
    return render


def _text(words, size):
    def render():
        from PIL import ImageFont
        image, draw = _canvas()
        font = ImageFont.load_default(size=size)
        left, top, right, bottom = draw.textbbox((0, 0), words, font=font)
        x, y = (SIZE - (right - left)) / 2 - left, (SIZE - (bottom - top)) / 2 - top
        draw.text((x, y), words, fill=(0, 0, 0), font=font)
        return image
    return render


def _bars(left_h, right_h):
    def render():
        image, draw = _canvas()
        base = SIZE - 40
        draw.rectangle((90, base - left_h, 150, base), fill=RGB["blue"])
        draw.rectangle((234, base - right_h, 294, base), fill=RGB["blue"])
        return image
    return render


# Nine small circles on a fixed irregular layout (deterministic, non-grid).
_NINE = ((60, 70), (170, 50), (300, 80), (110, 170), (230, 160), (330, 210),
         (70, 290), (190, 300), (310, 320))
_ODD = ((80, 90), (192, 90), (304, 90), (80, 290), (192, 290), (304, 290))


def _spec(prompt, render):
    return {"text": prompt, "image": render}


TASKS = {
    "color": (
        ("easy", _spec("What color is the square in this image? Reply with one word.",
                       _scene(("square", 192, 192, 110, "red"))), choice("red", COLORS)),
        ("medium", _spec("What color is the shape on the right side of this image? Reply with one word.",
                         _scene(("circle", 100, 192, 70, "blue"), ("square", 284, 192, 70, "yellow"))),
         choice("yellow", COLORS)),
        ("hard", _spec("All circles in this image share one color except one. What color is the odd one out? Reply with one word.",
                       _scene(*[("circle", x, y, 42, "orange" if (x, y) == (304, 290) else "green") for x, y in _ODD])),
         choice("orange", COLORS, ignore=("green circles", "green ones"))),
    ),
    "count": (
        ("easy", _spec("How many circles are in this image? Reply with only the number.",
                       _scene(("circle", 110, 192, 60, "black"), ("circle", 274, 192, 60, "black"))), number_is(2)),
        ("medium", _spec("How many circles are in this image? Do not count other shapes. Reply with only the number.",
                         _scene(("circle", 70, 80, 40, "red"), ("square", 190, 80, 40, "blue"), ("circle", 310, 80, 40, "green"),
                                ("square", 110, 280, 40, "orange"), ("circle", 270, 280, 40, "purple"))), number_is(3)),
        ("hard", _spec("How many dots are in this image? Reply with only the number.",
                       _scene(*[("circle", x, y, 18, "black") for x, y in _NINE])), number_is(9)),
    ),
    "spatial": (
        ("easy", _spec("Is the red circle on the left or the right? Reply with one word.",
                       _scene(("circle", 90, 192, 60, "red"), ("square", 294, 192, 60, "blue"))), choice("left", SIDES)),
        ("medium", _spec("Is the triangle above or below the square? Reply with one word.",
                         _scene(("triangle", 192, 90, 60, "green"), ("square", 192, 290, 60, "red"))),
         choice("above", ("above", "below"))),
        ("hard", _spec("What shape is directly below the blue circle? Reply with one word.",
                       _scene(("circle", 100, 100, 55, "blue"), ("square", 284, 100, 55, "red"),
                              ("triangle", 100, 284, 55, "green"), ("circle", 284, 284, 55, "yellow"))),
         choice("triangle", SHAPES, ignore=("blue circle",))),
    ),
    "read": (
        ("easy", _spec("What word is written in this image? Reply with only the word.", _text("CAT", 120)),
         contains_normalized("cat")),
        ("medium", _spec("What number is written in this image? Reply with only the number.", _text("4827", 90)),
         contains_normalized("4827")),
        ("hard", _spec("What phrase is written in this image? Reply with only the phrase.", _text("OPEN AT NINE", 36)),
         contains_normalized("open at nine")),
    ),
    "shape": (
        ("easy", _spec("What shape is in this image? Reply with one word.", _scene(("circle", 192, 192, 110, "black"))),
         choice("circle", SHAPES)),
        ("medium", _spec("What shape is in this image? Reply with one word.", _scene(("triangle", 192, 200, 120, "black"))),
         choice("triangle", SHAPES)),
        ("hard", _spec("What shape is in this image? Reply with one word.", _scene(("hexagon", 192, 192, 120, "black"))),
         choice("hexagon", SHAPES)),
    ),
    "compare": (
        ("easy", _spec("Which circle is larger, the left one or the right one? Reply with one word: left or right.",
                       _scene(("circle", 100, 192, 30, "black"), ("circle", 270, 192, 90, "black"))), choice("right", SIDES)),
        ("medium", _spec("Which bar is taller, the left one or the right one? Reply with one word: left or right.",
                         _bars(250, 190)), choice("left", SIDES)),
        ("hard", _spec("Which square is bigger, the left one or the right one? Reply with one word: left or right.",
                       _scene(("square", 100, 192, 56, "black"), ("square", 284, 192, 64, "black"))), choice("right", SIDES)),
    ),
}


# ------------------------------------------------------------- transport ----
def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


VISION_PATH = "/ml/vision"


def body(lane, variation, spec, tokens):
    """/ml/vision body: plain-text user turn + the image as a data URI in
    ``images``; worker-pinned through ``alloc`` exactly like the text suite."""
    _precision, mode, extras = variation[:3]
    image = spec["image"]() if callable(spec["image"]) else spec["image"]
    return {"model": lane["model"],
            "messages": [{"role": "user", "content": spec["text"] + " /no_think"}],
            "images": [data_url(image)],
            "max_new_tokens": tokens, "max_chunks": 1, "temperature": 0,
            "alloc": {"worker": lane["worker"], "alloc_mode": mode, **extras}}


def answer_of(response):
    """TaskResult ``text`` (ChatResult); an OpenAI-shaped reply as a fallback."""
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("text"), str):
        return response["text"]
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def call(client, lane, variation, spec, tokens):
    from ..fleet_grading import FleetError
    if isinstance(spec, str):              # seat / plain text probes
        spec = {"text": spec, "image": _scene(("square", 192, 192, 60, "black"))}
    started = time.time()
    response, error = retrying_request(client, VISION_PATH,
                                       body(lane, variation, spec, tokens), FleetError)
    elapsed = time.time() - started
    if error is None and isinstance(response, dict) and (response.get("ok") is False or response.get("error")):
        error = "VisionError: " + str(response.get("error"))[:1000]
    answer = answer_of(response) if error is None else ""
    usage = response.get("usage") if isinstance(response, dict) else None
    generated = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(generated, (int, float)) or generated <= 0:
        generated = max(1, round(len(answer) / 4)) if answer else None
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        prompt_tokens = None
    # The reply's call-ledger fields (request_id, prompt_s, generation_s,
    # gen_tokens, served stamp — common.served_fields); tok_s is
    # gen_tokens / generation_s when present, else tokens / wall seconds.
    from .common import served_fields
    extra = served_fields(response)
    gt, gs = extra.get("gen_tokens"), extra.get("generation_s")
    if isinstance(gt, (int, float)) and gt > 0 and isinstance(gs, (int, float)) and gs > 0:
        speed = gt / gs
    else:
        speed = generated / elapsed if isinstance(generated, (int, float)) and generated > 0 and elapsed > 0 else None
    return {"answer": answer, "output": answer, "error": error, "elapsed_s": round(elapsed, 4),
            "tok_s": speed, "ctx_in": prompt_tokens, "ctx_out": generated, **extra}


def score(response, checker):
    return bool(checker(response.get("answer") if isinstance(response, dict) else response))


SEAT = {"text": "Reply only: ready", "image": _scene(("square", 192, 192, 60, "black"))}
SPEED = {"text": "Describe this image in about 100 words: the shapes, their colors and positions.",
         "image": TASKS["spatial"][2][1]["image"]}
