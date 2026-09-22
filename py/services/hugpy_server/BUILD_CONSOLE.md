# Building the console bundles

`hugpy_server` serves the React applications as **package data**:
`src/hugpy_server/console_dist/` is mounted by `hugpy_server.wsgi_app.mount_console`
and resolved at runtime through `importlib.resources`, so a plain
`pip install hugpy-server && hugpy-serve` includes the UI. The Python package
never runs `npm`; the bundles are built from the workspace `react/` tree and
copied in before the wheel is built.

| React package (under `react/`) | Mount | Copy `dist/` to |
|---|---|---|
| `ui` (`@hugpy/ui`) | `/` | `console_dist/` |
| `agents_ui` (`@hugpy/agents-ui`) | `/fleet` | `console_dist/fleet/` |
| `media_intelligence_ui` | `/media` | `console_dist/media/` |
| `video_intelligence_ui` | `/video` | `console_dist/video/` |

`ui_shared` is a library consumed by the four builds; it has no mount.

## Steps

From the workspace root:

```bash
SERVER=py/services/hugpy_server/src/hugpy_server/console_dist

for pkg in ui agents_ui media_intelligence_ui video_intelligence_ui; do
  (cd react/$pkg && npm ci && npm run build)
done

rm -rf "$SERVER" && mkdir -p "$SERVER"
cp -r react/ui/dist/.                      "$SERVER/"
cp -r react/agents_ui/dist/.               "$SERVER/fleet/"
cp -r react/media_intelligence_ui/dist/.   "$SERVER/media/"
cp -r react/video_intelligence_ui/dist/.   "$SERVER/video/"
```

Then build the wheel (`python -m build --wheel` inside `py/services/hugpy_server`).
`pyproject.toml` lists `console_dist/**/*` as package data, so every file under
it ships.

## Runtime rules

* A mount only exists when its `index.html` exists. With no `console_dist/index.html`
  at all the server runs API-only and every non-API path is a plain 404.
* Deep links under an arm (`/video/foo`) fall back to that arm's own
  `index.html`; an arm that is not shipped falls back to the console root SPA.
* `HUGPY_UI_DIST=/path/to/dist` (env or `.env`) overrides the packaged bundle
  for development against a live webpack build.
* The video arm's `index.html` carries the `window.__HUGPY_MEDIA_BASE__` marker;
  the server rewrites it when `HUGPY_DEMO_MEDIA_DIR` / `HUGPY_DEMO_MEDIA_BASE`
  is configured (`wsgi_app._video_index_response`).
* `/studio` redirects (302) into the video arm.
