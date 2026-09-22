# Media-Intelligence Console — Change Writeup (2026-06-25)

Full-session writeup of the work to bring the media-intelligence console from
"shows features that don't function" to fully wired, plus three follow-on fixes.
Audience: the hugpy VM keeper / anyone deploying the dev API.

**Canonical trees touched (edit ONLY these — stale twins exist and must not be edited):**
- UI: `dev/media_intelligence_ui/`
- Dev backend: `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/flask_app/`
- Prod backend: `prod/api/api/app/flask_app/`

---

## TL;DR for the keeper

Most of this is **already live** on `https://dev.hugpy.ai/media/` (frontend is
served from `dist/` by the webpack devServer; no HMR, build → reload). The
**backend half is source-complete but DORMANT** — it activates with a single:

```
# (one-time) install the bridge extra into the dev venv:
pip install 'media_intelligence[bridge]'      # for POST /api/media/analyze
sudo systemctl restart hugpy-api-dev           # picks up media_extract.py + ml_routes.py changes
```

That restart activates the **generic-text read fallback** immediately. The other
two require their deps in the SERVICE venv:
- **Media bridge route** needs `media_intelligence[bridge]` installed.
- **Webpage assessment** needs `abstract_webtools >= 0.1.6.429` **in the service
  venv** (see the landmine below) — the restart alone is NOT enough.

No prod runtime was touched; no deploy/signal tokens were fired. Prod source files
were edited to keep the trees in parity, but prod is not deployed by this work.

> ⚠ **LANDMINE (caught by the keeper review).** The service runs from
> `dev/abstract_hugpy_dev/venv` (**python3.12**), whose `abstract_webtools` is an
> older build that (a) lacks/!-imports the `managers/assessManager` subpackage and
> (b) errors on a missing transitive dep (`pytesseract`). So in the LIVE service,
> `assess_url`'s `from abstract_webtools.managers.assessManager import …` RAISES,
> is caught, and silently degrades to `fetch_url_text`. The assessment feature is
> **inert-but-safe** until the SERVICE venv's `abstract_webtools` is bumped to
> >=0.1.6.429 (with its deps). My original `example.com` smoke test passed only
> because it ran in `/home/op/miniconda/bin/python` (3.13, newer build) — NOT the
> service interpreter. The inert window is the right time to harden SSRF (below).

---

## 1. Console full enablement (LIVE)

Every visible feature in the console + chat was given real behavior:

- **Chains** — `chainsBuiltin.ts` registers `doc.brief` (extract→summarize),
  `web.brief` (fetch→summarize), `audio.notes` (transcribe→summarize). Chains
  project into the page registry as `chain:<key>` PageSpecs and run step-by-step.
- **Dictation** — Web Speech API wired into `Composer.tsx` (mic toggle).
- **Chat history / conversations** — `chatHistory.ts` (new): localStorage-backed
  conversations with `lighten()` to strip heavy tool results before persisting
  (quota guard). Sidebar gained Recents, search, Library scroll, Profile menu
  ("Clear all conversations"). Turn-level regenerate / edit / delete.
- **Output renderers** — new `ImageGenOutput`, `EmbeddingOutput`,
  `DocumentTextOutput`; registered in `outputRegistry.tsx`. `pageSpec.ts` /
  `schemas.ts` / `pagesBuiltin.ts` gained the `imagegen` operation.
- **Transport** — `vite.config.ts` proxies `/api` → `HUGPY_API_URL`
  (default `http://127.0.0.1:7002`), stripping `/api`.

Backend: all 9 `/ml/*` amenities (transcribe / summarize / keywords / embed /
similarity / vision / imagine / extract / fetch) confirmed present and wired.
**Caveat:** heavy inference (vision / imagine / text-generation) has no serving
worker, so it runs in-process LOCAL → OOM risk on this swapless box. Lightweight
amenities (extract / fetch / summarize / keywords / embed) are safe.

## 2. AttachmentBar CSS fix (LIVE)

Root cause: `--bg-elevated-secondary` was undefined in `.hugpy-chat-scope`, so it
fell back to a light `#ececec` → light-on-light, unreadable filename + suggestion
text. Fix: component now uses `--bg-elevated-primary` (dark in-scope) and the dark
secondary token was added to the scope in `chat.css`.

## 3. Action-chip alignment (LIVE)

The blue solid action chips were restyled to match the outlined tool-tray pills:
transparent fill, `1px solid var(--border, rgba(127,127,127,.35))`, inherited text
color, hover tint via `hover:bg-token-surface-hover`.

## 4. File-read-by-default (frontend LIVE; backend dormant)

Goal (user's words): *"it should default on trying to simply read the file if it
cannot determine the type."* Modeled on `abstract_utilities … reader_utils`'s
`read_any_file`, kept stdlib-only (no pandas/ocr — the user wanted hugpy imports
lighter than reader_utils).

- **Frontend (live):** `EXTRACT_BY_KIND.text = { specKey: "ml/extract", stage:
  "extract" }` in `mediaIntelligence.ts`; `extractForSearch` / `attachFiles`
  extended to the `text` category in `main.tsx`. Undetermined files now route to
  `/ml/extract` instead of dead-ending.
- **Backend (dormant):** `_extract_generic_text(path)` added to both
  `media_extract.py` files. `extract_document`'s `else` branch (unknown extension)
  now attempts a guarded UTF-8 read instead of returning "unsupported". Guard: a
  cheap binary sniff (NUL byte, or >10% replacement/control chars) declines binary
  files so images/archives/spreadsheets don't return replacement-char garbage.
  `_MAX_TEXT_BYTES = 16 MiB` cap.

## 5. Webpage assessment integration (frontend LIVE; backend dormant) — NEW

Goal (user's words): integrate `abstract_webtools.managers.assessManager` so the
URL amenity returns a *structured, LLM-ready assessment* instead of just text.

- **`assess_url(url)` added to both `media_extract.py` files.** Delegates to
  `abstract_webtools.managers.assessManager.assess_webpage`, which fetches cheaply
  (requests) and only escalates to a full browser render when a page comes back
  near-empty (<200 chars; JS-walled / bot-blocked). Adapts the result into the
  existing `{ok, kind, text, …}` contract, now carrying `title`, `description`
  (og/twitter/meta), `metadata`, `jsonld`, same-domain `links`, `render` mode, and
  `truncated`.
- **Route:** `_run_fetch` (`/ml/fetch`, both trees) now calls `assess_url` instead
  of `fetch_url_text`.
- **Frontend (live):** `DocumentTextOutput.tsx` surfaces the new fields when
  present — meta-description line, `🖥 rendered` badge (selenium-forced),
  `✂ truncated` note, collapsible "N links on this page" list. Plain `/ml/extract`
  output is unchanged (fields only render when present), so there is **no
  UI/backend mismatch** — the UI shows nothing extra until the restarted backend
  starts sending the richer payload.

### Design guarantees / landmines for the assessment path

- **SSRF (read carefully):** `assess_url` validates the *initial* URL at the front
  door via the pre-existing `_assert_public_url` (http/https only, public addresses
  only). **BUT** `assessManager` follows redirects internally, so individual
  redirect HOPS are NOT re-validated — unlike the old `fetch_url_text` which
  re-checks every hop. The front-door check is the only guarantee on this path.
  This is a deliberate, disclosed tradeoff to use the user's tool as-is.
- **Cost / OOM:** `abstract_webtools` (pulls Selenium + matplotlib) is imported
  LAZILY on the first `/ml/fetch` only — module load stays cheap. A browser render
  is attempted only as an auto-fallback for a near-empty page, never by default.
- **Graceful degradation:** if `abstract_webtools` is absent / errors / yields no
  text, `assess_url` falls back to the lightweight `fetch_url_text`. So `/ml/fetch`
  is never *worse* than before.
- **Selenium currently broken on this box:** the smoke test showed the Chrome /
  chromedriver auto-render path crashes (stack dump). `abstract_webtools` catches
  it internally and returns the cheap-fetch result, so normal pages work fine; only
  JS-walled pages won't get a browser render until Chrome/chromedriver is fixed.
- Confirmed: `abstract_webtools` + `selenium 4.41.0` import cleanly in the dev
  backend interpreter (`/home/op/miniconda/bin/python`, py3.13).

---

## Files changed

**UI (`dev/media_intelligence_ui/src/`):**
- `chat/src/ui/{Composer,ChatTurnActions,MessageTurn,Thread,Sidebar,ToolRunTurn,AttachmentBar}.tsx`
- `chat/src/main.tsx`, `chat/src/imports/types.ts`
- `chat/src/utilities/{chatHistory.ts (new),mediaIntelligence.ts}`
- `chat/src/styles/chat.css`
- `console/src/imports/page/UtilityPage/output/{ImageGenOutput,EmbeddingOutput,DocumentTextOutput}.tsx`
  + `output/outputRegistry.tsx`
- `console/src/imports/pages/{pageSpec.ts,pagesBuiltin.ts}`, `console/src/imports/schemas.ts`
- `console/src/imports/chain/chainsBuiltin.ts`
- `vite.config.ts`

**Backend (dev + prod in parity):**
- `…/flask_app/app/functions/media_extract.py` — `_extract_generic_text`,
  `assess_url`, `_MAX_TEXT_BYTES`, `_MAX_ASSESS_CHARS`, `_MAX_ASSESS_LINKS`;
  `extract_document` unknown-ext fallback.
- `…/flask_app/app/routes/ml_routes.py` — `_run_fetch` → `assess_url`.

## Verification done

- esbuild `vite build` is the real gate (no `@types/react`, so `tsc --noEmit` is
  pure config noise — ~79 spurious errors, not a usable gate). Build green; live
  bundle `index-RRSo0SvP.js` confirmed served at `dev.hugpy.ai/media`.
- `py_compile` clean on all 4 backend files.
- `assess_webpage` live smoke test vs `example.com`: returned the full key set
  (`title="Example Domain"`, body text, links/metadata/jsonld/render/truncated).
  ⚠ This ran in `/home/op/miniconda/bin/python` (3.13), NOT the service venv (3.12,
  which currently can't import `abstract_webtools` — see the landmine above). It
  validates the adapter↔`assess_webpage` contract, not live-service activation.
- `_extract_generic_text` unit-checked: text file → returned; binary → declined.

## Standing activation checklist (keeper / user)

> ⚠ Use the SERVICE venv `dev/abstract_hugpy_dev/venv/bin/python` (3.12) for all
> dep changes below — NOT miniconda. That is the interpreter the service runs.

1. **Generic-text read fallback** — already source-complete; activates on the
   restart in step 4, no dep needed (stdlib-only).
2. **Media bridge** — `dev/abstract_hugpy_dev/venv/bin/pip install
   'media_intelligence[bridge]'` (enables `POST /api/media/analyze`, the optional
   server bridge the UI prefers). Bridge import is already guarded, so its absence
   never crashes startup.
3. **Webpage assessment** — bump `abstract_webtools` to **>=0.1.6.429 in the
   service venv** and ensure its deps resolve (`pytesseract` etc.). Until then
   `assess_url` degrades to `fetch_url_text` (safe). **Recommended: harden
   per-redirect SSRF revalidation on the assess path BEFORE this bump**, so the
   weaker-SSRF path never goes live unhardened.
4. `sudo systemctl restart hugpy-api-dev` — picks up the `media_extract.py` +
   `ml_routes.py` changes (generic-text fallback live immediately; bridge +
   assessment live only if steps 2/3 were done).
5. (Optional) fix Chrome/chromedriver on the box so the Selenium auto-render path
   works for JS-walled pages.
6. Prod: source is in parity but NOT deployed. Promote via the normal
   signal-token pipeline when ready — not done here by design.

## Known accepted items (not bugs, do not "fix")

- `/uploads` anonymous access — accepted by the user, intentionally not changed.
- Heavy inference runs in-process LOCAL (no GPU worker) → OOM risk; lightweight
  amenities are the safe surface.
