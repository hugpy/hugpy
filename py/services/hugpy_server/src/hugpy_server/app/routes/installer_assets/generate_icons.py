#!/usr/bin/env python3
"""Regenerate the installer launcher icons served by agent_routes.

Source of truth: the hugpy mark shipped in the console bundle
(``console_dist/assets/hugpy.*.png`` — the largest/cleanest variant, 293x314).
We DON'T serve that hashed bundle file directly (its name churns on every UI
build); instead we materialize two stable, committed assets beside this script:

  * hugpy-icon.png  — the mark as RGBA PNG (Linux .desktop Icon=)
  * hugpy-icon.ico  — a multi-size .ico (16/32/48/256), square-padded onto a
                      transparent canvas (Windows .lnk IconLocation needs a
                      real .ico; .desktop/.lnk want a square image)

This is a ONE-TIME conversion committed as static files — the route serves the
bytes, never runs PIL per request. Re-run only when the mark changes:

    cd abstract_hugpy_dev
    venv/bin/python src/abstract_hugpy_dev/flask_app/app/routes/installer_assets/generate_icons.py

Requires Pillow (already in the abstract_hugpy_dev venv).
"""
from __future__ import annotations

import glob
import os

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_mark() -> str:
    # Walk up to the package root, then into the console bundle assets. Prefer
    # the largest PNG (the full mark) over the small 135x135 header variant.
    d = _HERE
    for _ in range(12):
        cand_dir = os.path.join(d, "console_dist", "assets")
        if os.path.isdir(cand_dir):
            pngs = glob.glob(os.path.join(cand_dir, "hugpy*.png"))
            if pngs:
                return max(pngs, key=lambda p: Image.open(p).size[0]
                           * Image.open(p).size[1])
        d = os.path.dirname(d)
    raise SystemExit("could not locate the hugpy mark PNG in console_dist/assets")


def main() -> int:
    src = _find_mark()
    im = Image.open(src).convert("RGBA")
    png_out = os.path.join(_HERE, "hugpy-icon.png")
    im.save(png_out, format="PNG")
    print("wrote", png_out, im.size)

    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2), im)
    ico_out = os.path.join(_HERE, "hugpy-icon.ico")
    canvas.save(ico_out, format="ICO",
                sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    print("wrote", ico_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
