"""Tiered MEDIA grading: image-to-text (vision) and text-to-image (2026-09-23).

The text grader (``fleet_grading.TASKS_TIERED``) asks a model nine categories
of easy/medium/hard questions and scores passes. This module gives the two media
directions the same three-tier structure and the same result shape
(``{category: {"tier": passes, "max": 3, "history": [...]}}``), so the station
benchmark, the persisted results and the metrics grade stay one format.

Everything is MECHANICAL and deterministic — no model judges another model:

* image-to-text: the test images are SYNTHESIZED here (flat shapes, a 5x7
  bitmap font) as PNG bytes, so each question has one known answer and no
  fixture files ship. The answer is checked with the same kind of lenient
  string checks the text suite uses.
* text-to-image: the generated image is DECODED here (a stdlib PNG reader;
  Pillow only as an optional fallback for JPEG/WebP/interlaced output) and the
  prompt's claim is checked on the pixels — region colors, a dark shape in a
  light field, a count of dark blobs, requested size, seed reproducibility.

No network, no Flask, no runner import: the caller hands in an ``ask`` / ``gen``
function (fleet_grading wires them to /ml/vision and /ml/imagine).
"""
from __future__ import annotations

import base64
import colorsys
import math
import random
import re
import struct
import zlib

# --------------------------------------------------------------------------- #
# Raster canvas + PNG encode
# --------------------------------------------------------------------------- #

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RED, GREEN, BLUE = (220, 30, 30), (30, 170, 60), (30, 70, 220)
YELLOW, ORANGE, PURPLE = (245, 215, 30), (245, 140, 20), (130, 40, 170)


class Canvas:
    """A tiny RGB raster: fills, rects, circles, polygons, bitmap text."""

    def __init__(self, w: int = 256, h: int = 256, bg=WHITE):
        self.w, self.h = w, h
        self.px = [bytearray(bytes(bg) * w) for _ in range(h)]

    def set(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y][3 * x:3 * x + 3] = bytes(c)

    def get(self, x, y):
        r = self.px[y]
        return r[3 * x], r[3 * x + 1], r[3 * x + 2]

    def rect(self, x0, y0, x1, y1, c):
        for y in range(max(0, y0), min(self.h, y1)):
            for x in range(max(0, x0), min(self.w, x1)):
                self.set(x, y, c)
        return self

    def circle(self, cx, cy, r, c):
        for y in range(max(0, cy - r), min(self.h, cy + r + 1)):
            for x in range(max(0, cx - r), min(self.w, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.set(x, y, c)
        return self

    def polygon(self, pts, c):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        for y in range(max(0, min(ys)), min(self.h, max(ys) + 1)):
            for x in range(max(0, min(xs)), min(self.w, max(xs) + 1)):
                if _inside(x + .5, y + .5, pts):
                    self.set(x, y, c)
        return self

    def text(self, s, x, y, scale, c=BLACK):
        for ch in s:
            glyph = FONT_5X7.get(ch.upper())
            if glyph:
                for gy, row in enumerate(glyph):
                    for gx, bit in enumerate(row):
                        if bit == "#":
                            self.rect(x + gx * scale, y + gy * scale,
                                      x + (gx + 1) * scale, y + (gy + 1) * scale, c)
            x += 6 * scale
        return self

    def png(self) -> bytes:
        return encode_png(self.w, self.h, self.px)


def _inside(x, y, pts):
    n, inside = len(pts), False
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def encode_png(w, h, rows) -> bytes:
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


FONT_5X7 = {
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
}

# --------------------------------------------------------------------------- #
# PNG decode (stdlib) + pixel statistics
# --------------------------------------------------------------------------- #


class Image:
    """Decoded RGB image: ``w``, ``h`` and ``rows`` (bytearray RGB per row)."""

    def __init__(self, w, h, rows):
        self.w, self.h, self.rows = w, h, rows

    def get(self, x, y):
        r = self.rows[y]
        return r[3 * x], r[3 * x + 1], r[3 * x + 2]


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def decode_png(data: bytes) -> Image:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, palette, trns = 8, [], None, None
    w = h = depth = ctype = interlace = None
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            w, h, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if interlace or depth not in (8, 16) and ctype != 3 or (ctype == 3 and depth != 8):
        raise ValueError(f"unsupported PNG (depth={depth} type={ctype} interlace={interlace})")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp = channels * (depth // 8)
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    prev = bytearray(stride)
    rows, i = [], 0
    for _y in range(h):
        ft, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ft == 1:
                line[x] = (line[x] + a) & 255
            elif ft == 2:
                line[x] = (line[x] + b) & 255
            elif ft == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif ft == 4:
                line[x] = (line[x] + _paeth(a, b, c)) & 255
        prev = line
        step = depth // 8
        out = bytearray()
        for x in range(w):
            px = line[x * bpp:(x + 1) * bpp:step]      # high byte of 16-bit
            if ctype == 0 or ctype == 4:
                out += bytes((px[0],) * 3)
            elif ctype == 3:
                out += palette[3 * px[0]:3 * px[0] + 3]
            else:
                out += px[:3]
        rows.append(out)
    return Image(w, h, rows)


def decode_image(data: bytes) -> Image:
    """PNG via the stdlib reader; anything else via Pillow when installed."""
    try:
        return decode_png(data)
    except Exception as png_exc:
        try:
            import io
            from PIL import Image as _PIL        # optional, lazily
            im = _PIL.open(io.BytesIO(data)).convert("RGB")
            w, h = im.size
            flat = im.tobytes()
            return Image(w, h, [bytearray(flat[y * 3 * w:(y + 1) * 3 * w]) for y in range(h)])
        except ImportError:
            raise ValueError(f"undecodable image ({png_exc}); Pillow not installed") from png_exc


def _b64_to_bytes(s: str) -> bytes:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def lum(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def color_name(rgb) -> str:
    """Coarse perceptual name: black/white/gray or a hue family."""
    r, g, b = (v / 255 for v in rgb)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if ll < .18:
        return "black"
    if ll > .85 and ss < .5 or ll > .93:
        return "white"
    if ss < .22:
        return "gray"
    deg = hh * 360
    for limit, name in ((15, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
                        (255, "blue"), (320, "purple"), (361, "red")):
        if deg < limit:
            return name
    return "red"


def _samples(img: Image, fx0, fy0, fx1, fy1, n=16):
    """An n x n grid of samples over a fractional region (inset 10%)."""
    x0, x1 = fx0 + (fx1 - fx0) * .1, fx1 - (fx1 - fx0) * .1
    y0, y1 = fy0 + (fy1 - fy0) * .1, fy1 - (fy1 - fy0) * .1
    out = []
    for j in range(n):
        for i in range(n):
            x = int((x0 + (x1 - x0) * (i + .5) / n) * img.w)
            y = int((y0 + (y1 - y0) * (j + .5) / n) * img.h)
            out.append(img.get(min(x, img.w - 1), min(y, img.h - 1)))
    return out


def region_is(img, region, color, share=.6) -> bool:
    s = _samples(img, *region)
    return sum(color_name(p) == color for p in s) >= share * len(s)


def region_lum(img, region) -> float:
    s = _samples(img, *region)
    return sum(lum(p) for p in s) / len(s)


def dark_blobs(img: Image, grid=96, thresh=100, min_frac=.0015) -> int:
    """Connected dark components on a downsampled grid (small specks ignored)."""
    gw, gh = grid, max(1, int(grid * img.h / img.w))
    dark = [[lum(img.get(min(img.w - 1, int((x + .5) * img.w / gw)),
                         min(img.h - 1, int((y + .5) * img.h / gh)))) < thresh
             for x in range(gw)] for y in range(gh)]
    seen, count, min_cells = set(), 0, max(2, int(min_frac * gw * gh))
    for y in range(gh):
        for x in range(gw):
            if dark[y][x] and (x, y) not in seen:
                stack, size = [(x, y)], 0
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    size += 1
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < gw and 0 <= ny < gh and dark[ny][nx] and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                if size >= min_cells:
                    count += 1
    return count


def lum_stddev(img: Image) -> float:
    s = [lum(p) for p in _samples(img, 0, 0, 1, 1, n=24)]
    m = sum(s) / len(s)
    return math.sqrt(sum((v - m) ** 2 for v in s) / len(s))


# --------------------------------------------------------------------------- #
# Answer checks (same leniency as the text suite)
# --------------------------------------------------------------------------- #

def _ints(s):
    return [int(x) for x in re.findall(r"-?\d+", s or "")]


def _last(n):
    return lambda s: _ints(s)[-1:] == [n]


def _word(yes, *no):
    """``yes`` appears as a word and none of ``no`` does."""
    def check(s):
        low = (s or "").lower()
        return (re.search(rf"\b{yes}\b", low) is not None
                and not any(re.search(rf"\b{n}\b", low) for n in no))
    return check


# --------------------------------------------------------------------------- #
# image-to-text suite: (tier, prompt, image_builder -> PNG bytes, checker)
# --------------------------------------------------------------------------- #

def _circles(n, r, seed):
    """n non-overlapping black circles on a 4x4 cell grid (deterministic)."""
    rnd, cv = random.Random(seed), Canvas()
    cells = rnd.sample(range(16), n)
    for c in cells:
        cx, cy = 32 + (c % 4) * 64, 32 + (c // 4) * 64
        cv.circle(cx + rnd.randint(-6, 6), cy + rnd.randint(-6, 6), r, BLACK)
    return cv.png()


def _triangle(cv, cx, cy, r, c):
    return cv.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], c)


COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray")


def _one_color(want):
    return _word(want, *[c for c in COLORS if c != want])


VISION_TASKS_TIERED = {
    "color": (
        ("easy", "What is the main color of this image? Reply with one word.",
         lambda: Canvas(bg=RED).png(), _one_color("red")),
        ("medium", "What color is the LEFT half of this image? Reply with one word.",
         lambda: Canvas(bg=YELLOW).rect(0, 0, 128, 256, BLUE).png(), _one_color("blue")),
        ("hard", "What color is the small square in the top-right corner? Reply with one word.",
         lambda: Canvas().circle(90, 170, 60, PURPLE).rect(196, 12, 244, 60, ORANGE).png(),
         _one_color("orange")),
    ),
    "count": (
        ("easy", "How many black circles are in this image? Reply with only the number.",
         lambda: _circles(2, 24, 1), _last(2)),
        ("medium", "How many black circles are in this image? Reply with only the number.",
         lambda: _circles(5, 20, 2), _last(5)),
        ("hard", "How many black circles are in this image? Reply with only the number.",
         lambda: _circles(9, 14, 3), _last(9)),
    ),
    "shape": (
        ("easy", "What shape is shown in this image? Reply with one word.",
         lambda: Canvas().circle(128, 128, 90, BLACK).png(), _word("circle", "square", "triangle")),
        ("medium", "What shape is shown in this image? Reply with one word.",
         lambda: _triangle(Canvas(), 128, 128, 90, BLACK).png(), _word("triangle", "circle", "square")),
        ("hard", "There is a circle, a square and a triangle. What color is the triangle? "
                 "Reply with one word.",
         lambda: _triangle(Canvas().circle(50, 128, 36, RED).rect(96, 92, 160, 156, BLUE),
                           208, 128, 36, GREEN).png(), _one_color("green")),
    ),
    "spatial": (
        ("easy", "Is the red dot above or below the blue dot? Reply with one word.",
         lambda: Canvas().circle(128, 60, 30, RED).circle(128, 196, 30, BLUE).png(),
         _word("above", "below")),
        ("medium", "Is the red square on the left or the right? Reply with one word.",
         lambda: Canvas().rect(20, 88, 100, 168, GREEN).rect(156, 88, 236, 168, RED).png(),
         _word("right", "left")),
        ("hard", "Which circle is the largest: red, green, or blue? Reply with one word.",
         lambda: Canvas().circle(40, 128, 18, RED).circle(128, 128, 60, GREEN)
         .circle(214, 128, 34, BLUE).png(), _one_color("green")),
    ),
    "read_text": (
        ("easy", "What digit is shown in this image? Reply with only the digit.",
         lambda: Canvas().text("7", 98, 58, 20).png(), _last(7)),
        ("medium", "What number is written in this image? Reply with only the number.",
         lambda: Canvas().text("305", 38, 86, 12).png(), _last(305)),
        ("hard", "What word is written in this image? Reply with only the word.",
         lambda: Canvas(320, 128).text("HUGPY", 10, 36, 8).png(),
         lambda s: re.search(r"\bhugpy\b", (s or "").lower()) is not None),
    ),
}


# --------------------------------------------------------------------------- #
# text-to-image suite: (tier, prompt, request params, checker(images) -> bool)
# ``images`` is the list of decoded Images one task produced (two for the
# reproducibility tier). Checks are deliberately lenient: flat-color prompts
# are judged on the inner 80% of each region, >= 60% of samples.
# --------------------------------------------------------------------------- #

FLAT = " Flat solid colors, minimal vector style, no texture, no shading, no text."
T2I_SIZE = (512, 512)


def _repro(imgs):
    a, b = imgs
    if (a.w, a.h) != (b.w, b.h):
        return False
    diffs = [abs(lum(p) - lum(q)) for p, q in zip(_samples(a, 0, 0, 1, 1, 24),
                                                  _samples(b, 0, 0, 1, 1, 24))]
    return sum(diffs) / len(diffs) < 3.0


IMAGEGEN_TASKS_TIERED = {
    "delivery": (
        ("easy", "A red apple on a wooden table.", {},
         lambda imgs: lum_stddev(imgs[0]) > 6),                      # decodable, not blank
        ("medium", "A red apple on a wooden table.", {"width": 512, "height": 384},
         lambda imgs: (imgs[0].w, imgs[0].h) == (512, 384)),          # honors the size
        ("hard", "A red apple on a wooden table.", {"seed": 1234, "repeat": 2},
         _repro),                                                     # same seed, same image
    ),
    "color": (
        ("easy", "A plain solid red background filling the whole image, nothing else." + FLAT, {},
         lambda imgs: region_is(imgs[0], (.2, .2, .8, .8), "red")),
        ("medium", "The top half is solid blue and the bottom half is solid yellow." + FLAT, {},
         lambda imgs: region_is(imgs[0], (0, 0, 1, .45), "blue")
         and region_is(imgs[0], (0, .55, 1, 1), "yellow")),
        ("hard", "The flag of France: three equal vertical stripes, blue on the left, white "
                 "in the middle, red on the right, filling the whole image." + FLAT, {},
         lambda imgs: region_is(imgs[0], (0, .15, .3, .85), "blue")
         and region_is(imgs[0], (.36, .15, .64, .85), "white")
         and region_is(imgs[0], (.7, .15, 1, .85), "red")),
    ),
    "layout": (
        ("easy", "A single large black circle in the center of a plain white background." + FLAT, {},
         lambda imgs: region_lum(imgs[0], (.4, .4, .6, .6)) < 90
         and min(region_lum(imgs[0], r) for r in ((0, 0, .15, .15), (.85, 0, 1, .15),
                                                   (0, .85, .15, 1), (.85, .85, 1, 1))) > 170),
        ("medium", "A black square in the top-left corner of a plain white background, the "
                   "rest of the image empty white." + FLAT, {},
         lambda imgs: region_lum(imgs[0], (0, 0, .5, .5)) + 40
         < min(region_lum(imgs[0], r) for r in ((.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)))),
        ("hard", "Exactly three black dots in a horizontal row on a plain white background." + FLAT, {},
         lambda imgs: dark_blobs(imgs[0]) == 3),
    ),
}

SUITES = {"text": None, "vision": VISION_TASKS_TIERED, "imagegen": IMAGEGEN_TASKS_TIERED}

# Tasks with no grader yet — a model whose primary task is one of these is
# reported ``no_grader`` instead of being asked the text suite.
UNGRADED_TASKS = frozenset({
    "image-to-image", "image-to-video", "text-to-video", "text-to-speech",
    "automatic-speech-recognition", "feature-extraction", "sentence-similarity",
    "depth-estimation", "object-detection", "image-classification",
    "image-segmentation", "pipeline-component", "adapter",
})


def suite_for(model: dict) -> str:
    """Which suite grades this catalog row: text | vision | imagegen | none."""
    primary = (model.get("primary_task") or "").strip()
    tasks = set(model.get("tasks") or [])
    if primary == "text-to-image" or (not primary and "text-to-image" in tasks
                                       and "text-generation" not in tasks):
        return "imagegen"
    if primary == "image-text-to-text" or (not primary and "image-text-to-text" in tasks):
        return "vision"
    if primary in UNGRADED_TASKS or (not primary and tasks and tasks <= UNGRADED_TASKS):
        return "none"
    return "text"


# --------------------------------------------------------------------------- #
# Runners — the transport is the caller's
# --------------------------------------------------------------------------- #

def _summarize(detail, suite):
    depths = {k: v["tier"] for k, v in detail.items()}
    return {"suite": suite, "detail": detail,
            "answers": {k: v["history"] for k, v in detail.items()},
            "score": sum(depths.values()), "max": 3 * len(detail),
            "task_best": max(depths, key=depths.get) if depths else "N/A",
            "task_worst": min(depths, key=depths.get) if depths else "N/A"}


def grade_vision(ask, on_call=None, stop=None):
    """``ask(prompt, png_bytes) -> (answer, error, elapsed, speed, ctx_in,
    ctx_out)``. Returns the grade dict (+ ``calls``)."""
    detail, calls = {}, []
    for category, tiers in VISION_TASKS_TIERED.items():
        history = []
        for tier, prompt, build, check in tiers:
            answer, error, elapsed, speed, ctx_in, ctx_out = ask(prompt, build())
            passed = not error and bool(check(answer))
            history.append({"tier": tier, "pass": passed})
            call = {"task": f"{category} ({tier})", "suite": "vision", "prompt": prompt,
                    "elapsed_s": elapsed, "tok_s": speed or "N/A", "ctx_in": ctx_in,
                    "ctx_out": ctx_out, "output": answer, "passed": passed,
                    "grade": "ERROR" if error else "PASS" if passed else "FAIL",
                    "error": error or "N/A"}
            calls.append(call)
            if on_call:
                on_call(call)
            if stop is not None and stop.is_set():
                break
        detail[category] = {"tier": sum(x["pass"] for x in history), "max": 3, "history": history}
    return {**_summarize(detail, "vision"), "calls": calls}


def grade_imagegen(gen, on_call=None, stop=None):
    """``gen(prompt, params) -> (b64_or_bytes | None, error, elapsed)``. A task
    with ``repeat`` generates that many times with the same params. Returns the
    grade dict (+ ``calls`` and ``s_per_image``)."""
    detail, calls, times = {}, [], []
    for category, tiers in IMAGEGEN_TASKS_TIERED.items():
        history = []
        for tier, prompt, params, check in tiers:
            params = dict(params)
            repeat = int(params.pop("repeat", 1))
            params.setdefault("width", T2I_SIZE[0])
            params.setdefault("height", T2I_SIZE[1])
            imgs, error, elapsed = [], None, 0.0
            for _ in range(repeat):
                data, err, secs = gen(prompt, params)
                elapsed += secs or 0.0
                if err or not data:
                    error = err or "no image returned"
                    break
                try:
                    raw = _b64_to_bytes(data) if isinstance(data, str) else data
                    imgs.append(decode_image(raw))
                except Exception as exc:  # noqa: BLE001 — an undecodable image fails the task
                    error = f"{type(exc).__name__}: {exc}"
                    break
                times.append(secs or 0.0)
            passed = False
            if not error:
                try:
                    passed = bool(check(imgs))
                except Exception as exc:  # noqa: BLE001
                    error = f"check failed: {type(exc).__name__}: {exc}"
            history.append({"tier": tier, "pass": passed})
            call = {"task": f"{category} ({tier})", "suite": "imagegen", "prompt": prompt,
                    "params": params, "elapsed_s": round(elapsed, 4), "tok_s": "N/A",
                    "output": [f"{i.w}x{i.h}" for i in imgs], "passed": passed,
                    "grade": "ERROR" if error else "PASS" if passed else "FAIL",
                    "error": error or "N/A"}
            calls.append(call)
            if on_call:
                on_call(call)
            if stop is not None and stop.is_set():
                break
        detail[category] = {"tier": sum(x["pass"] for x in history), "max": 3, "history": history}
    spi = round(sum(times) / len(times), 3) if times else None
    return {**_summarize(detail, "imagegen"), "calls": calls, "s_per_image": spi}
