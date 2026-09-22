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

The table above lives in `console_manifest.json` (mount -> React package dir,
npm name, pinned published version, copy target under `console_dist/`).
`tools/build_console.py` (stdlib only) reads it. From this package directory:

```bash
python tools/build_console.py --from-source           # npm ci + npm run build in react/<pkg>, copy dist/
python tools/build_console.py --from-npm              # npm pack <npm>@<pinned version>, copy its dist/
python tools/build_console.py --from-source --only /fleet   # one mount; repeat --only for more
python tools/build_console.py --check                 # which mounts are present, index.html per mount
```

* Only the selected mounts are replaced. Rebuilding `/` clears the root files
  but never `fleet/`, `media/` or `video/`, and it does not copy the arm copies
  that ui's `postbuild` puts in `ui/dist/<arm>/` (each arm has its own mount).
* `--from-source` installs with `npm ci --workspaces=false` (`npm install` when a
  package has no lockfile) because `react/` is an npm workspace root without a
  root lockfile. `--skip-install` reuses the existing `node_modules`.
* `--from-npm` needs a release whose tarball includes `dist/`. `"dist"` was added
  to `files` after the currently pinned versions were published (ui 0.3.0,
  agents-ui 0.3.0, media-intelligence-ui 0.3.0, video-intelligence-ui 0.3.2),
  so those versions stop with a clear error. Bump the pins in the manifest
  after the next tag-published release (`react/README.md`).
* `--check` exits 1 when a selected mount has no `index.html`. `--console-dist`
  and `--react-root` override the manifest paths (tests use a temp dir).

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
