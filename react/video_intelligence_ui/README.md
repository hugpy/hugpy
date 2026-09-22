# @hugpy/video-intelligence-ui

Video-intelligence console — sibling arm of `media_intelligence_ui` (Vite +
Tailwind v4), served at `/video`. Four stations (image crop, audio crop,
frames/models, generate) over the hugpy job bus.

## Demo media is hosted, not bundled

The canned demo (`?demo=1`) replays recorded job envelopes whose media
(~36 MB of sample mp4/png) is **not** shipped in this package or its build.
It loads from a hosted base, default `https://hugpy.ai/demo-media`.

Resolution order (see `src/demo/mediaBase.ts`, resolved once at load):

1. `?mediaBase=` query param (http(s) or path-absolute values only)
2. `window.__HUGPY_MEDIA_BASE__` — a non-empty string; `index.html` ships a
   `window.__HUGPY_MEDIA_BASE__ = window.__HUGPY_MEDIA_BASE__ ?? null;`
   marker that servers/self-hosters may rewrite (replace `null` with a
   JSON string)
3. `VITE_HUGPY_DEMO_MEDIA_BASE` (build-time, read in `src/config.ts` — the
   one module permitted to hold URL literals)
4. the default `https://hugpy.ai/demo-media`

## Self-hosting the demo media (`hugpy serve`)

The Python server (`abstract_hugpy_dev` / `hugpy`) reads two env vars
(env or `.env`) and rewrites the marker in the served `index.html`:

- `HUGPY_DEMO_MEDIA_BASE=<url>` — point the demo at any hosted copy of the
  media tree.
- `HUGPY_DEMO_MEDIA_DIR=<path>` — serve a local copy of the tree yourself at
  `/demo-media/` (wins over `HUGPY_DEMO_MEDIA_BASE`). Unset → the route does
  not exist.

## Canonical media tree

The canonical copy lives at `/var/www/hugpy-media/` on the hugpy.ai VM,
served by an `location /demo-media/` block in the brochure nginx vhost —
deliberately OUTSIDE the brochure docroot (`/var/www/hugpy.ai`) so brochure
rebuilds (`stage.sh` rsync `--delete`) can never wipe it. Its layout mirrors
the `demoMedia("...")` specifiers: `generate/`, `generate2/`, top-level
`demo-*` files, and `sampels/` (that dir is intentionally spelled "sampels"
— do NOT rename it).
