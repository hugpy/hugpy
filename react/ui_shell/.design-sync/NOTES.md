# design-sync notes — @hugpy/ui

Repo-specific gotchas for future syncs. Append as you learn things.

## Re-sync (one command)
From the package root `dev/ui/`, after `npm run build:lib` (only if src changed):
```
BASE=/tmp/.../design-sync   # re-copy staged scripts: cp -r $BASE/{package-*.mjs,resync.mjs,lib,storybook} .ds-sync/
(cd .ds-sync && npm i esbuild ts-morph @types/react playwright)   # PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
# fetch the project anchor first:
#   DesignSync(get_file, path:"_ds_sync.json") -> .design-sync/.cache/remote-sync.json
DS_CHROMIUM_PATH=/usr/bin/google-chrome-stable node .ds-sync/resync.mjs \
  --config .design-sync/config.json --node-modules ./node_modules \
  --entry ./dist-lib/hugpy-ui.mjs --out ./ds-bundle \
  --remote .design-sync/.cache/remote-sync.json
```
Project: `https://claude.ai/design/p/e9b198fd-7111-4619-8b4d-b4ec0a32af4b` (pinned in config.json).
Durable inputs (no git here — all live on disk under `.design-sync/`): `config.json`,
`NOTES.md`, `conventions.md`, `previews/` (16 authored), `preview-runtime.jsx`, `fonts-src/`.

## Setup / build
- **Package root** is `dev/ui/` (the skill was invoked from `dev/ui/src`). Run the
  converter from `dev/ui/`. There is **no git repo** anywhere in this tree, so
  nothing is committed — durable state lives only on disk under `.design-sync/`.
- **Shape: package** (no Storybook). Component list = PascalCase `.d.ts` exports
  from the library build `dist-lib/`.
- **Build the library first**: `npm run build:lib` (vite, `vite.lib.config.js`) →
  `dist-lib/hugpy-ui.mjs` (ESM, the bundle `--entry`), `dist-lib/index.d.ts`
  (+ tree, the DTS source), `dist-lib/style.css` (tokens + every component's CSS,
  the `cssEntry`). The webpack `dist/` is the standalone app — NOT the library;
  do not point the converter at it.
- **Dual lockfiles**: both `yarn.lock` and `package-lock.json` are present and
  `node_modules/` was already installed (recent). I did NOT reinstall — the build
  works as-is. If a clean install is ever needed, pick ONE manager deliberately
  (no `packageManager` field pins it). `@hugpy/console` is a `file:../../console`
  symlink dep, lazy-loaded by the Console tab only.
- Node: v20.20.2 in this env (no `.nvmrc` / `engines`).

## Render check (playwright)
- No playwright cache and no `playwright` in the repo, BUT system
  `/usr/bin/google-chrome-stable` (Chrome 146) exists. Installed just the
  `playwright` npm package into `.ds-sync/` (PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1,
  no 200MB chromium download) and run validate with
  `DS_CHROMIUM_PATH=/usr/bin/google-chrome-stable`. Re-syncs must set that env var.

## Fonts
- `--font-mono` stack is `'Fira Code', 'Cascadia Code', 'JetBrains Mono',
  ui-monospace, monospace`. The repo ships NO `@font-face` for these (default
  theme relies on system mono). User chose to **ship the brand mono fonts**:
  self-hosted woff2 for Fira Code (400/500/700) + JetBrains Mono (400/700) under
  `.design-sync/fonts-src/`, wired via `cfg.extraFonts` → `brand-mono.css`.
  Fetched from jsdelivr fontsource CDN.
- **Known render warn (triaged):** `[FONT_MISSING] "Cascadia Code"` — benign.
  Cascadia is the *second* fallback behind Fira Code (which now ships), so it
  never actually renders; it's Windows-only and not webfont-distributable. Leave
  it; do not chase on re-sync.

## Previews: the demo-fetch provider (critical)
- Panels are **data-driven**: they call `hugpyFetch`/`fetchJson` against a backend
  on mount. The repo's `src/showroom/` is a backend-less demo harness with canned
  fixtures (`fixtures.js`) + a fetch shim (`demoFetch.js`, exports `demoFetch` and
  `installDemoMode`) — the same thing that powers the public `/console` demo.
- `cfg.provider` = **`DemoProvider`**, defined in `.design-sync/preview-runtime.jsx`
  and bundled via `cfg.extraEntries` so it shares the SAME runtime-config singleton
  and react-router instance as the panels. It: (1) calls `installDemoMode()` for
  the global, config-independent patches (Phone-Brick `EventSource` replay,
  alert/confirm→toast); (2) wraps children in the bundle's own `HugpyProvider`
  with `fetch={demoFetch}` — `HugpyProvider` calls `configureHugpy` during render,
  so every panel's fetch is answered from fixtures; (3) wraps in `MemoryRouter`
  (Landing/Navbar/BrandMark/Auth forms use react-router).
- WHY a prop, not the global singleton: the bundle `--entry` is the COMPILED
  `dist-lib`, which has its own baked-in config module. Importing `installDemoMode`
  from `src/` configures a DIFFERENT config instance — useless for the panels. So
  the fetch is delivered via `HugpyProvider`'s `fetch` prop (same dist-lib
  instance the panels read). `installDemoMode`'s global side effects still apply.
- Prop-driven components (`ModelTable`, `ChatPanel`, `HFSearch`) take their data
  as PROPS — author previews pass fixtures directly (no shim needed for the data).
  Container panels (`WorkersPanel`, `Landing`, `PeersBar`, etc.) self-fetch — the
  DemoProvider shim populates them.

## Component scope
- 19 discovered. User scoped **16 visual** for authored previews. The 3 pure
  wrappers — `AuthProvider`, `HugpyProvider`, `PrivateRoute` — stay on floor cards
  (they're context/route guards, no sensible visual card) and are documented in
  the conventions header instead.

## Preview authoring — per-component (folded from fan-out learnings)
- `DemoProvider` chain (in `preview-runtime.jsx`): `MemoryRouter` >
  `HugpyProvider(fetch=demoFetch)` > `AuthProvider(mode="open")` > card. The
  AuthProvider was ADDED after the fan-out: HugpyConsole + PrivateRoute + every
  auth form read `useAuth()` and throw without it. `mode="open"` short-circuits
  config resolution (no `/api/auth/config` fetch; state = authed-local). The auth
  form previews ALSO wrap themselves in `<AuthProvider mode="open">` (redundant
  now that it's global, but harmless — inner wins).
- **`embedded` = the full expanded panel** for ApiAccess / BridgePanel /
  DiscordPanel / WorkersPanel; non-embedded collapses to a click-to-expand summary
  bar that can't render rich statically. Always use `embedded`.
- **HugpyConsole**: root is `.layout{height:100%}` (`src/App/App.css`), which
  collapses to 0 in an auto-height card. Fixed by a `100vh` wrapper in the preview
  `.tsx` PLUS `cfg.overrides.HugpyConsole {cardMode:single, viewport:"1280x900"}`.
  Pass `fetch={demoFetch}` explicitly (its own HugpyProvider would otherwise reset
  to real fetch).
- **ChatPanel / WorkersPanel**: `cfg.overrides.<>: {cardMode:"column"}` — they
  render wider than a grid cell (GRID_OVERFLOW). Presentation-only.
- **ChatPanel** is prop-driven: message shape `{role, content, model?, attachment?}`;
  image attachment `{name, isImage:true, dataUrl}` (inline data-URI SVG works).
- **Logout** renders `null` by design (fire-and-navigate). Its card mounts it in a
  small labeled dashed frame — a side-effect-only control, not a defect.
- **Login / Register / ChangePassword** are MUI light-theme Card forms (render on
  white, distinct from the dark DS panels — faithful to the real components).
  LoginForm is the dependency-light dark branded form.

## Known render warns (triaged — re-syncs check against this list)
- `[FONT_MISSING] "Cascadia Code"` — benign (see Fonts above; second fallback
  behind Fira Code which ships).

## KNOWN FIDELITY GAP — Discord/Bridge panels show the empty/create state
- **BridgePanel** and **DiscordPanel** render their styled, complete *create/empty*
  state ("No bridges yet" / "No models are bound yet"), NOT populated rows. This is
  faithful to the repo's own showroom demo, which shows the same — the cause is a
  fixture shape mismatch in `src/showroom/fixtures.js`: `DISCORD_BRIDGES/_CHANNELS/
  _MESSAGES/_BINDINGS/_USERS` are bare arrays, but the panels read WRAPPED objects
  (`d?.bridges`, `d?.bindings`, `d?.channels`, …), so every list falls back to `[]`.
  The fixtures' own header even calls these "thin shapes … not a showroom focus".
- These cards are graded `good` (the create state is a legitimate, useful card). To
  show POPULATED bridges/bindings on a future sync, the REPO would need its showroom
  fixtures reshaped to `{bridges:[…]}` / `{bindings:[…]}` / `{channels:[…]}` etc.
  (field shapes were documented in the deleted batch-a learnings; reconstruct from
  the panel source if pursuing). NOT done here — it's a repo change, and inventing
  data the real demo doesn't produce would be lower-fidelity, not higher.

## Re-sync risks
- BridgePanel/DiscordPanel populated state depends on a repo fixture fix (above) —
  re-syncs will keep showing the create state until then. Don't treat the empty
  state as a regression.
- The global `AuthProvider(mode="open")` masks any guest/unauthenticated posture —
  if a future cell needs to show the logged-out state, it must override locally.
- `preview-runtime.jsx` imports from `src/showroom/demoFetch.js` and
  `dist-lib/hugpy-ui.mjs` by relative path. If the showroom's `demoFetch` export
  or the runtime-config API (`configureHugpy`/`HugpyProvider`) changes, the
  provider may silently stop populating panels — watch for floor cards reappearing
  on container panels after a re-sync.
- Brand mono woff2 are vendored under `.design-sync/fonts-src/` (network-fetched
  once). They're durable (committed-equivalent on disk); no re-fetch needed.
- Fixtures (`src/showroom/fixtures.js`) are the single source of preview data;
  if the API shapes drift from fixtures, panels may render empty/error states.
