<!-- Written by a per-subsystem code-review pass (2026-09-14). Follows _TEMPLATE.md. -->

# Frontends — mechanics

**Scope (paths this doc covers):** `react/` — `ui/` (the console SPA), `agents_ui/`,
`media_intelligence_ui/`, `video_intelligence_ui/` (the "arms"), `ui_shared/` (shared
plain-DOM chrome) — and `station-app/` (Electron shell + its aiohttp backend). Paths
below are relative to `/srv/hugpy/src/` unless given in full: this subsystem sits
outside the `abstract_hugpy_dev` package, so the usual relative root doesn't apply.
**One-liner:** Four independently-built React surfaces (one webpack SPA + three Vite
"arms") that central assembles into one origin at request time, plus a separate
Electron+aiohttp desktop/headless console (`station-app`) for fleet operators.
**Owner process(es):** No frontend here runs its own prod process. Central
(`7002_hugpy_api`, Flask) serves the built SPA and arms as static files
(`abstract_hugpy_dev/flask_app/wsgi_app.py` `_mount_ui`, `_ui_dist_dir` — the one piece
of the package this doc reaches into). Dev-only: webpack devServer (`react/ui/`, :7001)
and each arm's own `vite dev`. `station-app/resources/backend/server.py` (aiohttp) runs
either as an Electron-spawned child (desktop app) or as `hugpy-station-web@<instance>`
(headless — e.g. `hugpy-station-web@hugpy-demo`, `SERVICES.md:36`).

## 1. Purpose & responsibilities
- `react/ui/`: the operator console product — model registry, worker fleet, chat,
  metrics, API keys, Discord bindings, docs — one webpack build, served same-origin by
  central behind `/api`.
- Three "arms" (`agents_ui`→`/fleet`, `media_intelligence_ui`→`/media`,
  `video_intelligence_ui`→`/video`): independently-built Vite+Tailwind SPAs, "loosely
  coupled, NOT bundled into the SPA" (`react/ui/webpack.config.js:91-93`) — each owns its
  own React/Vite toolchain and node_modules, mounted at a path prefix rather than
  compiled into the main bundle.
- `ui_shared/`: the one piece of code literally shared by all four surfaces — a
  dependency-free plain-DOM Help widget and a data-only navbar-links registry — so the
  four separate React installs never have to agree on a version.
- `station-app/`: a separate product (fleet/keeper operations, not the model console)
  — an Electron desktop shell over a self-contained aiohttp backend that serves a
  hand-written, buildless single-file React(+htm) page plus VM/terminal/keeper/fleet-mail
  APIs.
- Does NOT do: none of the four `react/` builds talk to the toolserver (7004) directly —
  that's documented for operators (`react/ui/src/pages/Docs/ToolserverPage.jsx`) and
  driven live only from `station-app`'s backend.

## 2. Key modules
| file | responsibility |
|---|---|
| `react/ui/webpack.config.js` | dev bundler: `/api`,`/endpoints`,`/v1` proxy to central; static-mounts the 3 arms' `dist/`; `/studio`→`/video` redirect |
| `react/ui/scripts/bundle-media-arm.mjs` | `postbuild`: builds + copies the 3 arms into `ui/dist/<arm>/`, per-arm non-fatal on failure |
| `react/ui/src/App/App.jsx` | routes (`/`,`/login`,`/docs`,`/console/*`) + the `Console` shell (tabs, models/workers/chat wiring) |
| `react/ui/src/api.ts` | `fetchJson`/`uploadFile` — error-surfacing fetch wrapper every panel uses |
| `react/ui/src/runtime/config.ts` | `hugpyFetch`/`resolveApiUrl` — the one config seam the `@hugpy/ui` package is built around |
| `react/ui/src/runtime/feeds.js` | the one live-feed subscription (`GET /api/llm/events`, `useFeed(name)`), snapshot-poll fallback |
| `react/ui/src/components/ChatPanel/chatStore.js` | module-level chat + SSE-stream store — outlives component unmount/route change |
| `react/ui/src/components/ChatPanel/useChats.js` | `useSyncExternalStore` binding over `chatStore` |
| `react/ui/src/components/ChatPanel/ChatPanel.jsx` | chat UI; calls `chatStore.sendMessage`/`stopMessage` |
| `react/ui/src/components/ModelTable/ModelTable.jsx` | model registry table (+ `ServingControl`/`QuantControl`/`PlacementControl`) |
| `react/ui/src/components/WorkersPanel/WorkersPanel.jsx` | worker-fleet pool UI (register/assign/load/evict/restart/…) |
| `react/ui/src/components/MetricsPanel/MetricsPanel.jsx` | tok/s charts from `/api/llm/model-metrics2` |
| `react/ui/src/pages/Docs/Docs.jsx` | hash-routed docs shell (no router dependency) |
| `react/ui/src/Auth/AuthProvider.tsx` | single source of auth state + networking (`open` vs `external` mode) |
| `ui_shared/help/helpWidget.js` | the one Help widget (plain DOM), mounted identically by all 4 surfaces |
| `ui_shared/navbar/links.js` | shared nav-link data, rendered per-arm (React-Router `<Link>` vs `<a>`) |
| `agents_ui/entry.tsx`, `vite.config.ts` | fleet-arm bootstrap, `base:"/fleet/"` |
| `media_intelligence_ui/entry.tsx`, `vite.config.ts` | media-arm bootstrap, `base:"/media/"` |
| `video_intelligence_ui/entry.tsx`, `vite.config.ts` | video-arm bootstrap, `base:"/video/"`, demo-media base injection |
| `abstract_hugpy_dev/flask_app/wsgi_app.py` | central's static serving: `_mount_ui`, `_ui_dist_dir`, `ApiPrefixMiddleware` |
| `station-app/main.js` | Electron shell: spawns `server.py`, waits for it, opens the `BrowserWindow` |
| `station-app/resources/backend/server.py` | aiohttp backend: terminals, VM control, keeper/fleet-mail, `/api/frontier/*`, serves `static/index.html` |
| `station-app/resources/backend/static/index.html` | the buildless single-file React(+htm) page (~9.4k lines, hand-maintained) |
| `station-app/resources/systemd/hugpy-station-web@.service` | headless unit template for the packaged backend |

## 3. Entry points
- Console SPA: browser → `/`, `/login`, `/docs`, `/console/*` (`react-router` `Routes`,
  `App.jsx:621-640`); `/console/*` is gated by `PrivateRoute`+`AuthProvider`.
- Arms: browser → `/fleet`, `/media`, `/video` (central's `_mount_ui`, or dev's webpack
  `static`/`historyApiFallback`) — each arm's `entry.tsx` does
  `createRoot(...).render(<BrowserRouter basename=…>)`.
- Build: `npm run build` in `react/ui/` (`package.json` scripts: `build`→webpack,
  `postbuild`→`bundle-media-arm.mjs`); each arm has its own `npm run build`→`vite build`.
- `station-app` desktop: OS launches Electron's `main` (`package.json:"main":"main.js"`)
  → `app.whenReady()` (`main.js:198`) spawns `server.py`, opens a `BrowserWindow` at
  `http://127.0.0.1:8899`.
- `station-app` headless: `systemctl start hugpy-station-web@<instance>` →
  `python3 server.py` directly (`resources/systemd/hugpy-station-web@.service:27`),
  loopback `:8898` by default, fronted by nginx/a reverse proxy.
- `station-app` backend HTTP surface: ~130 `/api/*` routes (`server.py`, registration
  block ~9500-9648) plus `GET /` (`index`, serves buildless `static/index.html`) and
  static mounts `/static/`, `/docs/`.

## 4. Data flow (the spine)

**A. Build & deploy (console + arms → central)**
1. Edit `react/ui/src/...` (or an arm's `src/`).
2. `cd react/ui && npm run build` — webpack (`webpack.config.js:8-16`) needs Node run
   under nvm, not the bare system interpreter (`signals/README.md:73`, "`nvm node`").
3. `postbuild` runs `scripts/bundle-media-arm.mjs`: for each of `agents_ui`(→fleet),
   `media_intelligence_ui`(→media), `video_intelligence_ui`(→video) it `execSync`s
   `npm run build` in that arm's own dir, then `cpSync`s the arm's `dist/` into
   `ui/dist/<arm>/` (`bundle-media-arm.mjs:38-67`). Each arm's failure is caught and only
   logged — the console build "succeeds" even if an arm silently didn't bundle
   (`bundle-media-arm.mjs:11-15,51-52`).
4. The automated `deploy-ui.trigger` (host `hugpy-jobs.timer`, `User=solcatcher`,
   `/srv/hugpy/README.md:29,32`) runs the same build and republishes the result.
5. Central does **not** read that output directly (unless `HUGPY_UI_DIST` is set) — it
   serves `abstract_hugpy_dev/console_dist/`, a checked-in copy inside the src package
   (`_ui_dist_dir()`, `wsgi_app.py:37-63`). Refreshing it is a manual step ("step 0c",
   `scripts/deploy/DEPLOY.md:46-60`): rsync `ui/dist/{assets,fleet,media}` + `index.html`
   into `abstract_hugpy_dev/src/abstract_hugpy_dev/console_dist/` and commit — "NO JOB
   DOES THIS" (`DEPLOY.md:47`).
6. Central re-reads `console_dist/index.html` from disk per request
   (`wsgi_app.py:120-164`) — so once `console_dist/` is refreshed, no restart of
   `7002_hugpy_api` is needed for the console; a restart *is* needed for Python route
   changes.

**B. Runtime request routing (prod, one origin)**
1. Browser hits `dev.hugpy.ai/console` → nginx → central `:7002` (`SERVICES.md:14`).
2. `_mount_ui`'s catch-all (`wsgi_app.py:136-164`) serves real files from
   `console_dist/` by path; anything else falls back to `console_dist/index.html` (SPA
   deep links) **except** under `/media/*` or `/video/*`, which fall back to that arm's
   *own* `index.html` instead (`wsgi_app.py:148-164`) — mirroring dev's webpack
   `historyApiFallback` rewrites (`webpack.config.js:106-112`).
3. `/studio/*` 302-redirects to `/video/*` in both dev (`webpack.config.js:113-135`) and
   prod (`wsgi_app.py:125-134`) — a redirect, not a rewrite, because the video arm's
   `BrowserRouter basename` is baked from its Vite `base` at build time and only matches
   when the URL is actually under `/video/` (`video_intelligence_ui/entry.tsx:56-59`).
4. The SPA's own calls go `api.ts` `fetchJson` → `runtime/config.ts` `hugpyFetch`, which
   resolves relative `/api/...` against an empty-by-default `baseUrl` (same-origin, no
   CORS in the bundled console — `runtime/config.ts:40-43,112-114`). Central strips the
   `/api` prefix itself (`ApiPrefixMiddleware`, `wsgi_app.py:20-34`, wired at
   `wsgi_app.py:408`), so the SPA and a direct `curl :7002/health` both work.
5. Live tables (workers, models, jobs, serving, downloads, …) ride ONE `EventSource`
   against `GET /api/llm/events` (`runtime/feeds.js:5-13`); panels call `useFeed(name)`
   instead of polling their own endpoint. A dead stream falls back to
   `GET /api/llm/snapshot` every 20s until it reconnects.

**C. Chat send/stream (module-level store survives navigation)**
1. `ChatPanel.jsx:149` calls `chatStore.sendMessage(modelKey, {...})` — a plain async
   function, not a hook (`chatStore.js:116`).
2. `sendMessage` POSTs `/api/chat/stream`, reads the SSE body itself, and pushes
   `token`/`status`/`error` events into module-scope `chats`/`streaming`/`allocation`
   objects, calling `commit()` to notify subscribers (`chatStore.js:150-208`).
3. `useChats()` (`useChats.js:13-19`) binds a component via `useSyncExternalStore`, so a
   route change that unmounts `Console` (`App.jsx:636`) or a model switch that remounts
   `ChatPanel` via `key={activeChat}` (`App.jsx:451`) does **not** drop in-flight tokens —
   nothing about the request is owned by a React fiber (`chatStore.js:1-25`, the "t142"
   fix).
4. Stop is explicit only — `chatStore.stopMessage()` POSTs
   `/api/llm/chat/cancel/<request_id>` and aborts the controller (`chatStore.js:105-111`);
   unmounting never aborts a request.
5. `ui_shared/help/helpWidget.js`'s Ask tab reuses the identical contract (POST
   `{apiBase}/chat/stream`, same SSE shape — `helpWidget.js:18-21`), mounted the same way
   in all four surfaces.

## 5. State, persistence & invariants
- localStorage: `hugpy.chats.v1` (`chatStore.js:29`, sanitized — no base64 attachment
  `dataUrl`s, no transient `status` field, `chatStore.js:39-52`), `hugpy.activeChat`,
  `hugpy.activeTab` (`App.jsx:32-48`).
- sessionStorage (`useSessionState`): `hugpy.sess.wp.open`, `hugpy.sess.wp.tab`
  (`WorkersPanel.jsx:18,41`) — per-tab UI layout, not synced across tabs.
- In-memory only, by design: `chatStore`'s `controllers` (AbortControllers) and
  `requestIds` (`chatStore.js:58-59`, comment "in-memory only, on purpose") — neither can
  survive a reload regardless.
- Two independent client-side "worker roster" copies exist: `App.jsx`'s `workers` state
  fed by the shared `useFeed('workers')` (used by `StatusBar`/tab badges/`ModelTable`'s
  assign submenu, `App.jsx:110-111`), and `WorkersPanel.jsx`'s own
  `fetchJson('/api/llm/workers')` poll (`WorkersPanel.jsx:16,123`) — `App.jsx:482` passes
  `WorkersPanel` only `models`, not `workers`.
- `station-app`: PTY sessions live in an in-process `SESSIONS` dict with a reaper task
  (`_start_reaper`/`_stop_reaper`); a signed session cookie + pbkdf2 password hash live on
  disk next to `server.py` (`.secret`, `.auth` — module docstring, `server.py:10-19`);
  keeper mail is a JSON-rows file (`_keeper_mail_write`, around `server.py:6962`); the
  doctrine docs served at `/docs/` are plain `.md` files under `static/docs/`.
- Invariant: `console_dist/`'s existence gates whether central serves *any* UI —
  `_ui_dist_dir()` returns `None` (API-only mode) when no candidate directory has an
  `index.html` (`wsgi_app.py:60-63`).

## 6. Cross-subsystem edges
- → central (`abstract_hugpy_dev/flask_app/`, ~35 blueprints): essentially all panel
  traffic — `/api/models`, `/api/llm/*` (see `[[worker-fleet]]`, `[[serving-core]]`),
  `/api/jobs/*`, `/api/keys/*`, `/api/discord/*`, `/api/agent/*`, `/api/chat/stream`,
  `/api/auth/config` — plus the OpenAI-compatible `/v1/*` (bare, no `/api` prefix,
  proxied directly in dev — `webpack.config.js:80-89`). See `[[api-routes]]`.
- → central's static layer specifically: `wsgi_app.py` `_mount_ui`/`_ui_dist_dir` (§4.A/B)
  — the one piece of `abstract_hugpy_dev` this doc covers, since it's the other half of
  "how the console is served."
- `station-app`'s backend → toolserver (7004) via `_ts_call()` (`server.py:5544-5555`,
  `comms/ping`, `loci/pointers`, …) — the *only* place in this whole subsystem that talks
  to the toolserver live; the console SPA never does (only documents it, see §1).
- `station-app`'s backend → LXD (`lxc exec`) for per-station terminals and fleet-mail
  delivery into a station's inbox file (`fleet_message_post`, `server.py:6946-6999`).
- ← operators/browsers: `dev.hugpy.ai` (console+API), `demo.hugpy.ai` (station demo via
  `hugpy-station-web@hugpy-demo`), and the desktop Electron app (loopback only).
- `ui_shared` is the only literal code-sharing edge between the four `react/` surfaces;
  everything else (auth pattern, API-base resolution) is convention, not shared code.
- The video arm hosts the "Studio" experience referenced by `/studio` — see
  `[[video-oracle]]` for the planning/generation side it calls into.

## 7. Key contracts / types
- SSE events on `/api/chat/stream`: `{type:'request', request_id}`,
  `{type:'status', served_by?, worker_id?, worker_name?, progress?, message?, stage?}`,
  `{type:'token', text}`, `{type:'error', message}` (`chatStore.js:172-205`) — consumed
  identically by `ChatPanel` and `helpWidget`.
- `HugpyRuntimeConfig` (`runtime/config.ts:15-38`): `{baseUrl, fetch, headers?,
  credentials?}` — the one config surface the whole `@hugpy/ui` package is built around.
- `ui_shared/package.json:13-18` `exports` map: only the literal, extension-qualified
  subpaths `./help/helpWidget.js`, `./help/helpWidget.css`, `./navbar/links.js`,
  `./navbar/navbar.css` resolve — no extensionless alias.
- Vite `base` per arm is the wire contract with `_mount_ui`'s mount path: `/fleet/`
  (`agents_ui/vite.config.ts:15`), `/media/` (`media_intelligence_ui/vite.config.ts:14`),
  `/video/` (`video_intelligence_ui/vite.config.ts:14`) — must agree with
  `wsgi_app.py`'s arm handling and `bundle-media-arm.mjs`'s `ARMS` list
  (`bundle-media-arm.mjs:26-32`), three separate files, no shared source of truth.
- `POST /api/fleet/message` body `{to, text, from?}` (`server.py:6946-6959`) — `to` is
  `"keeper"`, an LXD station name, or an ssh/toolserver locus name; the handler picks the
  delivery path (local mail file / `lxc exec` / toolserver `comms/ping`) accordingly.

## 8. Gotchas, tech-debt & review findings
- ⚠ `wsgi_app.py:155` — `_mount_ui`'s deep-link SPA fallback special-cases only
  `for arm in ("media", "video")`; `"fleet"` (agents_ui) is absent. `agents_ui` already
  mounts a `BrowserRouter`/`<Routes>`, and its own bootstrap says more routed sub-pages
  are coming ("S3+ adds routed sub-pages (/fleet/nodes, /fleet/start)",
  `agents_ui/entry.tsx:4`). Dev's webpack devServer already rewrites `/fleet/*` deep
  links (`webpack.config.js:106-112`); prod does not. Harmless today (fleet is a single
  `/` route) but will blank-screen a direct prod load of a future `/fleet/<sub>` URL.
- △ `scripts/deploy/DEPLOY.md:46-60,112-116` — the wheel's console (`console_dist/`) is a
  manual rsync copy of `ui/dist`, wired into no automated job ("NO JOB DOES THIS",
  `DEPLOY.md:47`; "`deploy-ui` ≠ the wheel… step 0c is the bridge", `DEPLOY.md:112-113`).
  Confirmed live: `console_dist/` was hand-refreshed 2026-09-14 (mtime matches `ui/dist`),
  but its `media/`/`video/` subdirs are still dated 2026-09-12 — the refresh only ever
  copies whatever `ui/dist` currently has (see next finding). A stale console is this
  system's documented normal failure mode, not an edge case.
- △ Observed 2026-09-14: `react/ui/dist/` contains `fleet/` (built same-day) but no
  `media/` or `video/` at all; `media_intelligence_ui/dist/` and
  `video_intelligence_ui/dist/` are dated 2026-09-12. Consistent with the documented
  failure mode (§4.A step 3 — `bundle-media-arm.mjs` catches and only logs each arm's
  build/copy error). One concrete, reproducible permission hazard that would produce
  exactly this: `media_intelligence_ui/dist/` and `video_intelligence_ui/dist/` carry a
  facl granting `solcatcher` (the account `hugpy-jobs.timer`/`deploy-ui.trigger` runs as,
  `/srv/hugpy/README.md:29`) `r-x` only, no write. `agents_ui/dist` carries the identical
  ACL yet *was* refreshed today, so the ACL alone doesn't fully explain this specific
  gap — treat it as a real, reproducible hazard for any solcatcher-run build touching
  these two dirs, not a confirmed single root cause for every instance.
- ℹ `agents_ui/ui_shared/` is an empty directory (verified: zero entries) — not a
  symlink or copy of the real `react/ui_shared/`. `agents_ui/package.json`'s `files`
  array lists `"ui_shared"` for npm publishing, but `entry.tsx` imports the working copy
  via the relative sibling path `../ui_shared/help/helpWidget`, which resolves to
  `react/ui_shared/` only inside this monorepo checkout. A standalone
  `npm install @hugpy/agents-ui` elsewhere would ship the empty local dir with no sibling
  to fall back to — latent, not a risk for the current in-tree build.
- ℹ `ui_shared/package.json:13-18`'s `exports` map has no extensionless alias (see §7) —
  anyone importing the published `@hugpy/ui-shared` package (unlike every current
  consumer, which uses in-tree relative paths) must spell the `.js`/`.css` extension in
  full or resolution fails under Node/bundler `exports` semantics.
- ℹ Doc drift on `deploy-ui.trigger`'s build target: `/srv/hugpy/README.md:32`
  (2026-09-02 reorg, newest) says `src/react/ui` → `www/console`; `signals/README.md:25,
  49,73-74` and `scripts/deploy/DEPLOY.md:53,68,89-90` both still say
  `/srv/hugpy/ui/dist`/`/srv/hugpy/react/ui/dist` and `llm.hugpy.ai/app` — a
  pre-2026-09-02-retirement, VM-relative layout. Per this repo's own doc-precedence rule
  (newer + higher-listed wins), trust the top-level README; verify the live path before
  relying on either of the other two.
- ℹ `App.jsx:482` passes `<WorkersPanel models={models} />` without `workers` —
  `WorkersPanel` keeps its own `/api/llm/workers` poll rather than the shared
  `useFeed('workers')` roster `App.jsx` already holds (§5). Two independently-refreshed
  client copies of the same server list; not observed to desync, but worth knowing before
  "just pass the feed data down" looks like a free simplification.

## 9. Deploy/run boundary
- Console/arms: edit `react/{ui,agents_ui,media_intelligence_ui,video_intelligence_ui}/src`
  → `npm run build` in `ui/` (webpack + arm bundling; needs nvm-managed Node, §4.A step 2)
  → `ui/dist/`. Not live anywhere yet.
- Prod (central, `:7002`): needs the extra manual "step 0c" rsync into
  `abstract_hugpy_dev/src/abstract_hugpy_dev/console_dist/` (§4.A step 5) — no restart of
  `7002_hugpy_api` required afterward (static files are re-read per request); a stale
  console after a "successful" deploy almost always means step 0c was skipped.
- `www/console`: a second, separate static docroot that `deploy-ui.trigger` also
  produces — not what central serves; don't debug a central console issue by checking
  this path.
- Dev (webpack devServer, `ui/`, `:7001`, and each arm's own `vite dev`): fully
  live-reload, proxies `/api` to `HUGPY_API_URL` (default `127.0.0.1:7002`) — no
  build/copy step at all in this mode.
- `station-app`: `main.js`/`server.py` are not built — the Electron shell spawns
  `server.py` straight from `resources/backend/` (dev) or `process.resourcesPath/backend`
  (packaged, `main.js:21-26`); editing `server.py` or `static/index.html` is live on the
  next app restart / browser reload (it's the buildless htm.js page, §1 — no compile
  step). Packaging into the `.deb`/AppImage is `build-release.sh` → `dist/`
  (`/srv/hugpy/README.md`'s Station-releases line); the headless unit
  (`hugpy-station-web@.service`) runs whatever is installed at
  `/opt/hugpy-station/resources/backend/server.py` — a separately-deployed copy from this
  src tree until re-packaged and reinstalled.
