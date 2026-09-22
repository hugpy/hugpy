# @hugpy/agents-ui

Agent fleet console for Hugpy: portable runtime, node roster and dispatch for
`hugpy-agent` nodes. A standalone Vite + Tailwind v4 arm, served at `/fleet` by
`hugpy-server` and talking only to the same-origin Hugpy API (`/api`). It mounts
the shared Help widget and navbar links from `@hugpy/ui-shared` (`react/ui_shared`).

## Install

```bash
npm i @hugpy/agents-ui
```

The package ships the source and, from the first release published by the
tag workflow, the built bundle in `dist/` (built with `base: "/fleet/"`).

## Usage

Build and run it from the monorepo (the source imports `../ui_shared`, so it
builds inside `react/`, not from an isolated install):

```bash
cd react/agents_ui
npm ci --workspaces=false
npm run dev      # vite dev server; /api is proxied to HUGPY_API_URL (default http://127.0.0.1:7002)
npm run build    # dist/, served at /fleet
```

Ship it inside `hugpy-server`:

```bash
cd py/services/hugpy_server
python tools/build_console.py --from-source --only /fleet
```

## Architecture

How the React packages and the Python distributions fit together:
[PARTITION.md](https://github.com/hugpy/hugpy/blob/main/PARTITION.md); build and
mount rules: `py/services/hugpy_server/BUILD_CONSOLE.md`.

## License

hugpy Source-Available License; see [LICENSE](LICENSE).
