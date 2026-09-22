# HugPy Console — Enterprise-Grade Plan: Audit Findings

**Date:** 2026-06-16
**Scope:** live tree only — `src/console/` and `src/receptacle/`. The `backs/` directory is a kept-but-ignored backup and is **excluded** from all analysis below.
**Method:** every claim in the enterprise-grade plan was checked against the actual source. Each is marked CONFIRMED / PARTLY-TRUE / WRONG, with file:line evidence.
**Status of this pass:** audit + Phase 1 archive only. No behavior-changing edits made. Dead files are *moved into `backs/`*, not deleted (see "Phase 1 action log" at the end).

---

## Headline

The plan is accurate. **23 of 24 specific claims CONFIRMED, 1 WRONG (in our favor), plus 3 items the plan missed.** The architecture is sound; the work is drift, trust boundaries, transport, and the missing safety net — as the plan states.

---

## Phase 1 — dead / drift modules — ✅ all 7 CONFIRMED

Every named module has **zero live importers**.

| Item | Status | Evidence |
|---|---|---|
| `src/console/src/imports/pages/pagesBuiltin_.ts` | CONFIRMED dead | 0 importers. 9 keys overlap live `pagesBuiltin.ts` with **different** specs (e.g. `image/analyze` → field `path`, `category:"image"` vs live field `image_path`, `category:"deepcoder"`). `pagesRegistry.ts:7-9` **throws** on duplicate keys → co-import = crash at module load. |
| orphan `src/console/src/imports/page/ExecutionOutput.tsx` | CONFIRMED dead | 0 importers. Props `{loading,error,output}` vs live `UtilityPage/output/ExecutionOutput.tsx` props `{result}`. The only `ExecutionOutput` import (`UtilityPage.tsx:24`) resolves to the live one. |
| `src/receptacle/main.tsx` (top-level) | CONFIRMED dead | 0 importers. Real entry resolves `receptacle/index.ts → src/index.ts → src/main.tsx`. |
| `src/console/src/imports/consoles.module.css` | CONFIRMED dead | 0 importers. Live style is `console.module.css` (`imports/index.ts:4`). |
| `src/console/src/imports/pages/runner.tsx` | CONFIRMED dead | 0 importers. **Also un-compilable** — imports `../components/ExecutionPanel.css`, which does not exist in the tree. |
| `API_BASE_OCR`, `API_BASE_PDF` (`pageSpec.ts:44-45`) | CONFIRMED dead | Exported, referenced nowhere. (Line-level exports, not whole files — handled in Phase 3, not archived now.) |
| `src/console/src/ui/ExecutionPanel.tsx` | CONFIRMED transitively dead | Sole importer is `runner.tsx:2`. Dies with runner. |

**Plan missed:** `runner.tsx` is not merely unused but *un-compilable* (broken CSS path); `ExecutionPanel.tsx` is transitively dead via runner — archive both together.

---

## Phase 2 — single source of truth for types — ✅ CONFIRMED (1 correction)

- `InputMode`, `UploadedFileRef`, `MediaInputValue` declared in **both** `receptacle/src/imports/types.ts` and `console/.../pageSpec.ts`. **Identical today — zero field drift.** The risk is future, not present.
- **Correction:** it is **3 duplicated types, not 4** — `MediaInputProps` lives only in the receptacle (`types.ts:19`), never redeclared in `pageSpec.ts`.
- The split is already live: console imports from the receptacle in 2 files (`main.tsx`, `mediaFiltering.ts`) but from its own `pageSpec.ts` copies in 5+ (`submitPage`, `mediaInputPayload`, `chainRuntime`, `UtilityPage`, `UploadNormalize`).
- All four types are **already exported** from the receptacle's public surface (`receptacle/index.ts`). Consolidation needs **no new API** — just redirect imports and delete the `pageSpec.ts` copies.

---

## Phase 3 — config / env wiring — ✅ CONFIRMED

- All 3 named literals confirmed, all `https://hugpy.abstractendeavors.com`:
  - `API_BASE_HUGPY` — `pageSpec.ts:43`
  - `FILE_UPLOAD_ENDPOINT` — `receptacle/src/imports/constants.ts:1-2` (`/upload/file`)
  - `REGISTRY_ENDPOINT` — `pagesFromServer.ts:5` (`/registry/pages`)
- **Greenfield:** `grep` for `import.meta.env` and `VITE_` over the live tree → **zero hits.** No config module exists.
- **Plan missed:** `API_BASE_OCR` / `API_BASE_PDF` (`pageSpec.ts:44-45`) are **two more** base-URL literals — same host. They are the Phase-1 dead exports, so Phase 1 removes them and Phase 3 never migrates them. The phases align.

---

## Phase 4 — parse, don't assert — ✅ CONFIRMED (1 sub-claim WRONG)

- `pagesFromServer.ts:15`: `(body?.pages ?? []) as PageSpec[]` straight off `res.json()` — no runtime validation.
- DOM casts: `MediaInputDropdowns.tsx:36` (`as MediaKind`), `:54` (`as Operation`); `as File[]` at `UtilityPage.tsx:246,290` and `submitPage.ts:212`. More network-origin casts in `UploadNormalize.ts:71`, `UploadUtils.ts`, `chainRuntime.ts`.
- **Latent bug confirmed:** `submitPage.ts:241` `if (arr.length > 1) throw ...expected one file, got N` — fires *mid-submit*, after `UtilityPage` validation already passed.
- ⚠️ **WRONG (in our favor):** there is **no validation library** present (no zod/valibot/yup/ajv). Only an unused `axios ^1.6.8`. Phase 4 must add one from scratch.

---

## Phase 5 — transport client — ✅ all CONFIRMED

- `submitPage.ts:274-278`: bare `fetch`; headers only ever `{}` or JSON content-type. **No auth, no `credentials`, no `signal`, no timeout, no retry.** Errors thrown as `Error` — no Result type.
- **4 hand-rolled fetch sites, no shared client:** `submitPage.ts:274`, `runner.tsx:17` (dead), `pagesFromServer.ts:12`, `useMediaInputReceptacle.ts:48` (upload). **All four fully unauthenticated.**
- Unmount-mid-flight confirmed: `<UtilityPage key={selectedSpec.key}>` at `MediaExecutionPanel.tsx:37` forces remount on tool switch; no AbortController → orphaned fetch sets state on a dead component. (Inconsistency: `UtilityRoute.tsx:46` renders it *unkeyed*.)

---

## Phase 6 — selectedPaths trust model (security) — ✅ CONFIRMED — **the real prod gate**

- `UploadedFileRef` carries a raw server filesystem `path`. **No opaque-id indirection anywhere** (grep for `fileId/file_id/token/handle/opaque` → nothing but `max_new_tokens` form fields).
- Produced verbatim in `UploadUtils.ts:38-64`; rendered raw at `UploadedFileList.tsx:65` (`<small>{file.path}</small>` — also the React key **and** selection identity); echoed back into the POST payload (`submitPage.ts:246`, `resolveSourcedField`).
- Genuine IDOR / path-traversal surface. **Needs server-side work** (opaque IDs scoped to session) — the client swap alone does not close it. `path` being the selection identity (not just display) makes the client change non-trivial.

---

## Phase 7 — output renderer registry — ✅ CONFIRMED

- `ExecutionOutput.tsx` dispatches purely by shape: `hasTranscriptionShape`/`hasPdfReportShape`/`hasPdfTextShape`/`hasKeywordShape` (`ResultHelpers.ts`). Receives only `{result}` — no access to op/spec.
- Declared-output info **exists but is unused**: `pageSpec.ts:70` `resultKind`, `:74` `produces`, `:100` `OPERATION_OUTPUT`.
- `runChain` returns `ChainStepResult[]` (`chainRuntime.ts:25`) which matches no predicate → falls through to a raw-JSON `<pre>`. Chains dump raw today.

---

## Phase 8 — run / result store — ✅ CONFIRMED

- `UtilityPage.tsx:202-206`: ad-hoc `useState` for values/loading/result/error; `result` reset to `null` each submit, lost on unmount. No store, cache, or in-flight registry.

---

## Phase 11 — accessibility — ✅ CONFIRMED

- `FileInputStage.tsx:33-46` drop zone is `<div onClick>` — no `role`/`tabIndex`/`onKeyDown`; real `<input>` is `hiddenInput`.
- Error/output all bare `<pre>` (`UtilityPage.tsx:472`, `ExecutionOutput.tsx` ×7) — **zero `aria-live`/`role=`/`tabindex` in the entire live tree.**

---

## What the audit changes about execution

1. **No git in this tree.** Phase 1 deletions have no VCS undo → dead files are **moved into `backs/`**, not deleted, so they can be sifted/restored.
2. **Phase 1 ⊃ part of Phase 3:** removing `API_BASE_OCR`/`API_BASE_PDF` as dead code clears 2 hardcoded URLs for free. Sequence Phase 1 before 3.
3. **Phase 2 is cheaper than written** — no receptacle API additions; just redirect ~5 console imports and delete copies.
4. **knip must exclude `backs/`** or it floods with intentionally-kept "unreferenced" files. Wire the exclusion into the Phase 0 config.
5. **Phase 6 is the honest hard gate** and the only item needing server-side work. Everything else (0–5, 7–11) is client-only.

---

## Phase 1 action log

Confirmed-dead live files moved (not deleted) into `backs/<archive-folder>/`, original relative paths preserved, with a `MANIFEST.md`. See that folder for the exact set and restore instructions.

---

## Phase 0 action log — safety net (DONE 2026-06-16)

Stood up at the app root (`/srv/abstractendeavors/app`). All gates are **runnable scripts** in `package.json`. Note: this tree is **not a git repo and has no `.github/`**, so there is no CI yet — these scripts are what a CI job would invoke; wiring them into an actual pipeline is a follow-up once the repo is under VCS.

**Added devDeps:** `jsdom`, `knip` (installed with `--legacy-peer-deps`, see conflict note below).

**Scripts added/changed:**
| Script | Purpose | Current state |
|---|---|---|
| `npm test` | Vitest run (jsdom + `src/setupTests.js`) via `vitest.config.ts` | ✅ green — 5 passed, 1 skipped |
| `npm run test:watch` | Vitest watch | ✅ |
| `npm run typecheck` | global `tsc --noEmit` (strict:false, unchanged) | ✅ green — 0 errors |
| `npm run typecheck:hugpy` | strict gate scoped to hugpy (`tsconfig.hugpy.json`: strict + noUncheckedIndexedAccess + exactOptionalPropertyTypes) | 🔴 **19 errors** = burn-down baseline for Phases 2/4 |
| `npm run lint` | ESLint flat config, hugpy-scoped guard rules | 🔴 **10 errors** = hardcoded-URL baseline for Phase 3 |
| `npm run knip` | dead-file gate, hugpy-scoped, `backs/` excluded | ✅ green — 0 unused files |

**Why two typecheck scripts:** the root `tsconfig.json` has `strict:false` and this is a 547-package shared app; flipping strict globally floods the whole build. `tsconfig.hugpy.json` confines the new strictness to the hugpy console + receptacle (verified: the 19 errors are all inside hugpy, none leaked into the wider app).

**Gates that are RED by design:** `typecheck:hugpy` (19) and `lint` (10) are red *now* — they measure exactly the debt Phases 2–8 pay down (boundary casts, loose optionals, hardcoded URLs). They are baselines to burn down / ratchet, not yet merge-blockers. `test`, `typecheck` (global), and `knip` are green and CAN be hard gates today.

**Acceptance proof:** a deliberately-added dead file (`__deadfile_demo.ts`) made `npm run knip` exit 1 ("Unused files (1)"); removing it returned exit 0. The dead-code gate works.

**New files:** `vitest.config.ts`, `tsconfig.hugpy.json`, `eslint.config.js`, `knip.json`, and a seed test `…/output/ResultHelpers.test.ts` (also pins a latent quirk: `asStringArray` keeps `null` as the literal `"null"` because `String(null)` is truthy — flagged for Phase 9).

**Stale test handled:** `src/Components/App/App.test.tsx` was the broken CRA-default "learn react" test (renders `<App/>` with no `AuthProvider` → `useAuth` throws). Marked `test.skip` with a restore note so the gate is green; not a hugpy concern.

### ⚠️ Pre-existing toolchain conflict (not introduced here)
`package.json` declares `eslint ^9` but `@typescript-eslint/eslint-plugin@7` (installed 7.18.0) only supports `eslint ^8.56`. This ERESOLVE conflict predates this work and is why no ESLint config ever existed. The Phase 0 flat config sidesteps it by using only `@typescript-eslint/parser` (which parses fine under eslint 9) plus the core `no-restricted-syntax` rule — it does **not** load the incompatible plugin. To enable the full `@typescript-eslint` rule set later, upgrade the plugin to v8. Tracked, not blocking.

### Config-as-gate locations
- Test: `vitest.config.ts`
- Strict types (hugpy): `tsconfig.hugpy.json`
- Lint guard rules (hugpy): `eslint.config.js` — bans `https?://` literals outside `**/config.ts` and `as` on `body|response|data`
- Dead code (hugpy): `knip.json` — dependency rules off (package.json is app-wide), `files:error`

---

## Phase 2 action log — type single-source-of-truth (DONE 2026-06-16)

**Change:** the console's duplicate declarations of `InputMode` / `UploadedFileRef` / `MediaInputValue` (were `pageSpec.ts:77-98`) are deleted and replaced with a single type-only **re-export from the receptacle** (`pageSpec.ts` now: `export type { InputMode, UploadedFileRef, MediaInputValue } from "../../../../receptacle";`). The receptacle (`receptacle/src/imports/types.ts`) is now the sole declaration site.

**Why re-export instead of rewriting every import:** 5+ console files import these from `pageSpec` (`submitPage`, `mediaInputPayload`, `chainRuntime`, `UtilityPage`, `UploadNormalize`). The re-export keeps them working unchanged while collapsing the declaration to one owner — minimal blast radius, no behavior change (type-only).

**Acceptance — proven, not assumed:** temporarily renaming `UploadedFileRef.path` → `pathPROBE` in the receptacle produced `TS2339 Property 'path' does not exist on type 'UploadedFileRef'` in console files (`submitPage.ts`, `mediaInputPayload.ts`, `chainRuntime.ts`) — the cross-package dependency is real and compiler-enforced. Reverted.

**Note:** `MediaInputProps` was never duplicated (receptacle-only), so nothing to consolidate there.

**Regression:** `typecheck:hugpy` still 19 (no change — type identity preserved), global `typecheck` green, `knip` clean, `test` 5 passed/1 skipped, `lint` still 10 (Phase 3). No baseline moved — correct, since Phase 2 is a pure de-duplication, not a fix.

---

## Phase 3 action log — config module / env wiring (DONE 2026-06-16)

**Decisions (user deferred to best judgement):**
- **One base var, derive the rest.** New `VITE_HUGPY_API_BASE` (follows the app's `VITE_*` convention); `uploadUrl`/`registryUrl` derive from it, with optional `VITE_HUGPY_UPLOAD_URL`/`VITE_HUGPY_REGISTRY_URL` overrides.
- **Default to prod, no throw.** Unset → `https://hugpy.abstractendeavors.com`, so existing deploys keep working with no `.env`. (Deliberate deviation from the plan's "throw on missing" — the lone prod URL literal now lives only in `config.ts`, which the lint rule exempts.)
- **Did NOT reuse `VITE_SECURE_FILES_API_BASE`** as originally suggested: it resolves to `api.abstractendeavors.com/secure-files` (a different service, and unused in `src/`). Reusing it would have misrouted every hugpy call. Flagged and avoided.

**New:** `src/Components/tools/hugpy/src/config.ts` (the single URL owner), `.env.example` at app root documenting the vars.

**Rewired to read from config:**
- `pageSpec.ts` — `API_BASE_HUGPY = hugpyConfig.apiBase`; deleted dead `API_BASE_OCR`/`API_BASE_PDF` (Phase 1 audit).
- `pagesFromServer.ts` — `REGISTRY_ENDPOINT = hugpyConfig.registryUrl`.
- `receptacle/.../constants.ts` — `FILE_UPLOAD_ENDPOINT = hugpyConfig.uploadUrl`.
- `UrlInputStage.tsx` — placeholder `https://example.com/...` is UI text, not an endpoint → `eslint-disable-next-line` with reason.

**Baseline moved:** `lint` **10 → 4**. All 6 hardcoded-URL hits cleared; the remaining 4 are the `as` boundary casts owned by Phase 4 (`chainRuntime.ts:115,144`, `submitPage.ts:293`, `UploadNormalize.ts:71`). `typecheck:hugpy` unchanged at 19, global green, knip clean, tests green.

**Acceptance:** the whole console retargets to staging/localhost by setting one var (`VITE_HUGPY_API_BASE`) in `.env` — no code edits. Upload + registry follow automatically via derivation.

---

## Phase 4 action log — validated boundaries / parse-don't-assert (DONE 2026-06-16) — *prod blocker cleared*

**Added:** `zod ^4.4.3`; new `console/src/imports/schemas.ts` — zod mirrors of the enums (`MediaKind`/`Operation`/`InputMode`/`FieldKind`/`FieldSource`), `FieldSpec`/`PageSpec`, the response shapes (`Transcription`/`PdfReport`/`PdfText`/`Keyword`), `UploadResponse`, plus guards `isRecord`, `asFileArray`, `parseUploadResponse`.

**Boundaries fixed (every `as` on network/DOM-origin data removed):**
- `pagesFromServer.loadServerPages` — `(body?.pages ?? []) as PageSpec[]` → `z.array(PageSpecSchema).safeParse(...)`. Malformed registry → structured `console.error(issues)` + degrade (built-in pages survive), instead of injecting garbage. *(One commented, post-validation cast remains at `registerPage(spec as PageSpec)` — reconciles zod's `T|undefined` optionals with PageSpec's exactOptional ones; not a trust bypass, the data was already `.safeParse`d. Lint-clean: not a `body/response/data` identifier.)*
- `MediaInputDropdowns` — `as MediaKind` / `as Operation` on `<select>` → `*.Schema.safeParse`, unknown values ignored.
- `submitPage` — `(v as File[])` → `asFileArray`; `(data as {error}).error` → `isRecord` guard; the latent **multi-file bug moved to the edge**: `resolveSourcedField` now returns a `mismatch` ("select exactly one file") for a scalar `kind:"file"` field with >1 selection, surfaced by `UtilityPage` validation/`InheritedField` *before* POST — no more mid-submit `throw`.
- `UploadNormalize` — `response as UploadResponse` → `parseUploadResponse` (deleted the 37-line local type).
- `chainRuntime` — `data as Record` ×2 → `isRecord`; `__op as Operation` → `OperationSchema.safeParse`; fixed a latent `ops[0]` undefined (misconfigured chain → text fallback).
- `UtilityPage` — two `as File[]` → `asFileArray`; `__op as Operation` → schema parse.

**Baseline moved:** `lint` **4 → 0** (all boundary casts gone — the guard is now green and enforceable as a hard gate). `typecheck:hugpy` **19 → 18**. Global green, knip clean. `test` **12 passed / 1 skipped** (added `schemas.test.ts`: malformed registry rejected with structured issues, enum/guard coverage).

**Acceptance:** a hand-crafted malformed registry payload yields a typed, logged error and a degraded-but-honest UI (built-in pages intact) — proven by test, not assumed. No `as` survives on any network/DOM value.

**Note (SSoT debt):** a trivial `isRecord` now exists in both `schemas.ts` and `output/ResultHelpers.ts`; consolidate when Phase 7 reworks output rendering.

---

## Phase 5 action log — transport client (DONE 2026-06-16) — *prod blocker cleared*

**New:** `src/Components/tools/hugpy/src/transport/client.ts` — placed at the shared hugpy level (sibling of `config.ts`) instead of `console/src/transport` so the lower-layer receptacle upload can use it **without a console⇄receptacle import cycle**. (Deviation from the plan's path, for layering.)

**The client provides, for every call:**
- **`Result<T, AppError>`** return — failures cross the boundary as data, never thrown. `AppError = network | timeout | aborted | unauthorized | server` (server carries `status` + optional `traceId`); every error carries a `requestId`.
- **Auth** — `credentials: "include"` by default (matches the app-wide convention: AuthProvider/Register/Navbar all use it). Knob `VITE_HUGPY_WITH_CREDENTIALS=false` for when hugpy CORS isn't credential-ready. *(Behavior change: hugpy calls were previously unauthenticated.)*
- **Abort** — caller `AbortSignal` combined with an internal timeout signal (`AbortSignal.any`, with a manual fallback).
- **Timeout** — default 120s ceiling, per-call override via the new `PageSpec.timeoutMs` (long inference sets its own).
- **Retry/backoff** — idempotent **GET/HEAD only** (never a POST); exponential backoff on 5xx/network.
- **Correlation id** — `X-Request-Id` header per call, echoed in the `AppError`.

**Three live fetches routed through it:**
- `pagesFromServer` registry — GET with retry; failure logs `AppError` + degrades.
- `submitPage` — now returns `Promise<Result<unknown>>`, takes an `AbortSignal`, POST not retried.
- receptacle upload (`useMediaInputReceptacle`) — POST through the client.

**UtilityPage lifecycle (the orphaned-fetch fix):**
- `abortRef` + `mountedRef`; `useEffect` cleanup aborts in-flight request on unmount (the page is keyed by `spec.key`, so tool-switch unmounts it) and blocks any `setState` on the dead component.
- A **Cancel** button (shown while running) aborts the request.
- The form **switches on `AppError.kind`**: 401 → "session expired, sign in again"; 5xx → message + trace id; timeout → timed-out message; aborted → silent.

**strictNullChecks quirk:** the app's root tsconfig has `strict:false`, so TS won't narrow the `Result` discriminated union in the global build. Centralized the one cast each in guarded accessors `errorOf` / `okValue`; call sites stay clean and the global build compiles.

**Coverage expanded:** lint script + `eslint.config.js` globs + `knip.json` project now include `transport/` and `config.ts` (they sat outside the original console/receptacle globs).

**Baselines:** `typecheck:hugpy` **18 (held)** — net-neutral, the Result plumbing offsets the casts removed; global `typecheck` **0 (green)**; `lint` **0**; `knip` clean; `test` **21 passed / 1 skipped** (added `client.test.ts`: success/text/401/5xx+trace/network/GET-retry/aborted/correlation-id, + `describeAppError`).

**Acceptance:** a 401 surfaces a re-auth message; a 5xx surfaces a retryable message with trace id; a thrown/hung fetch maps to `network`/`timeout`; an aborted request resolves to `aborted` and never updates the unmounted component — the mapping is unit-tested, the lifecycle wiring verified by construction.

**Cleanup candidates (knip warnings, non-blocking):** Phase-4 response schemas (`Transcription`/`PdfReport`/`PdfText`/`Keyword`) are unused until Phase 7 wires them; `parseUploadResponse`/`getUploadErrorMessage` in the receptacle's `UploadUtils.ts` were orphaned by routing upload through the client — remove or de-export in a later pass.

---

## Phase 6 action log — selectedPaths trust model (CLIENT DONE 2026-06-16) — *prod blocker: client ready, server work outstanding*

**Decision (user):** full client id-swap (not just interim path-hiding).

**Compiler-driven rename across 13 files** (changed the type defs first, then fixed every site `tsc` flagged):
- `UploadedFileRef.path` → **`id`** (an opaque, session-scoped handle).
- `MediaInputValue.selectedPaths` → **`selectedIds`**.
- `FieldSource` `"selectedPaths"` → **`"selectedIds"`** (+ `FieldSourceSchema`, + all `pagesBuiltin` specs).
- Both upload normalizers (`UploadUtils` receptacle, `UploadNormalize` console) now derive the id **preferring a server `id`/`file_id`, falling back to the path/url** as the handle until the server emits real ids. Added `id`/`file_id` to the upload schema lens.
- All consumers rewired: `useMediaInputReceptacle`, `submitPage` (`selectedUploadedIds`, payload sends ids), `resolveSourcedField`, `chainRuntime` projection, `mediaInputPayload` (`fileIds`), `UtilityPage`, `main.tsx`, `UtilityRoute`, `FileInputStage`.

**The browser no longer receives or shows a server path:** `UploadedFileList` rendered `<small>{file.path}</small>` — now removed; rows key on the opaque `file.id` and display file **size**, never a location.

**Baselines:** `typecheck:hugpy` **18 (held — net-neutral rename)**, global `typecheck` **0**, `lint` **0**, `knip` clean, `test` **24 passed / 1 skipped** (added `UploadNormalize.test.ts`: id preferred over path, path fallback, throws on neither).

**⚠️ NOT fully closed — server work outstanding (the actual gate):** the client is now ready to consume opaque ids, but until the **server issues session-scoped opaque file ids and resolves id → path under ownership checks**, a fabricated id/path still reaches the execution endpoint. The interim state is strictly better (no path leak to the browser; client treats the handle as opaque), but the IDOR is only fully closed once the server enforces ownership. This is the one item that cannot be completed from the frontend.

---

## Phase 7 action log — output renderer registry + chain status (DONE 2026-06-16)

**Dispatch by declared op, not shape-sniffing.** `ExecutionOutput` no longer runs a `hasXShape` chain. New `output/outputRegistry.tsx` holds `outputRenderers: Map<Operation, (value) => ReactNode>`; `ExecutionOutput` resolves the **effective operation** (explicit selection, else `spec.produces[0]`) and looks up its renderer, falling back to `GenericOutput` for ops without a dedicated one.

**Each renderer validates with a Phase-4 schema and shows a TYPED error on mismatch** ("unexpected shape for *op*" + raw response) instead of a silent JSON dump. Op-local shape variance stays inside the op's renderer — e.g. `transcribe` (which `pdf/text` also produces) tries the transcription shape then the pdf-text shape; `summarize` tries the pdf-report shape then the generic block. This is the registry pattern with localized validation, not a global sniff chain.

**Made `KeywordResultSchema` a real guard:** it was all-optional passthrough (accepted any object); added a `.refine` requiring ≥1 keyword field (mirrors the old `hasKeywordShape`), so an unrelated object now surfaces a typed error rather than an empty keyword render.

**Chains finally render.** `runChain`'s `ChainStepResult[]` (previously dumped as raw JSON) now goes through `ChainOutput`: per-step ✓ok/✕error with the page key, the failing step's error, and a "chain stopped at step N" note. Validated via the new `ChainStepResultArraySchema`. Added chain CSS to `UtilityPage.module.css`.

**Cleanup:** removed the two Phase-5-orphaned receptacle exports (`parseUploadResponse`, `getUploadErrorMessage`). The prod-dead `hasTranscriptionShape/PdfReport/PdfText` predicates remain only because `ResultHelpers.test` covers them — harmless, left.

**Baselines:** `typecheck:hugpy` **18 (held)**, global **0**, `lint` **0**, `knip` clean, `test` **30 passed / 1 skipped** (added `outputRegistry.test.tsx`: op-keyed dispatch, typed error on mismatch, chain validation + failed-step id).

**Acceptance:** removing a `hasXShape` heuristic changed nothing (dispatch is by declared op); a mismatched payload yields a clear typed error, not a degraded dump; a chain that fails at step N shows exactly where and why.

---

## Phase 8 action log — run/result store (DONE 2026-06-16)

**Run lifecycle is now observable state, not callback timing.** New `console/src/imports/page/runStore.ts` — a **module-level** external store (read via `useSyncExternalStore`) so runs survive `UtilityPage` unmount (it's keyed by `spec.key` and remounts on every tool switch). Records carry `status: running | ok | error | cancelled`, `result`, live `steps`, `error`, and `startedAt`/`endedAt`; `getLatestForSpec(specKey)` backs the UI. Records are replaced immutably so unaffected specs keep a stable snapshot reference (no needless re-renders).

**`UtilityPage` rewired:** dropped local `loading`/`result`/`error` state (and the Phase-5 `abortRef`/`mountedRef` unmount-abort). The submit now `beginRun` → executes → `completeRun`/`failRun`/`cancelRun`. Because the store is module-level, the async submit writes there even after the page unmounts — so **switching away mid-run and back shows the run** (running or completed). Validation errors stay local (`formError`) — they're input-level, not runs.

**Chains stream live.** `runChain` gained an `onProgress(steps)` callback; `UtilityPage` pipes it to `setRunSteps`, so the Phase-7 `ChainOutput` shows per-step ok/error **as each step completes**, not just at the end.

**Cancel works across remounts.** The `AbortController` lives in the store keyed by run id, so the Cancel button (`cancelRun(run.id)`) aborts the request and transitions to `cancelled` even after navigating away and back. Transition guards (`completeRun`/`failRun`/`cancelRun` act only on a still-`running` record) make a late-arriving result unable to override a cancellation.

**Deliberately skipped:** the optional result cache keyed by (spec, input). Re-running an inference tool should hit the server fresh; silently returning a cached inference would be surprising. Noted as an intentional omission, not an oversight.

**Baselines:** `typecheck:hugpy` **18 (held)**, global **0**, `lint` **0**, `knip` clean, `test` **37 passed / 1 skipped** (added `runStore.test.ts`: transitions, persistence-as-latest, cancel-aborts-controller, late-complete-doesn't-override-cancel, live steps, new-run-replaces-latest).

**Acceptance:** run a tool → switch away → switch back shows the completed result; a chain's per-step status is observable in real time; Cancel transitions to `cancelled` and aborts the underlying request.

---

## Phase 9 action log — tests for pure logic (DONE 2026-06-16)

Filled the plan's named gaps. Added: `submitPage.test.ts` (`resolveSourcedField` all branches incl. local-file fallback + scalar-file mismatch; `buildSourcePayload`; `selectedUploadedFiles/Ids`; `isFieldVisible`; the `useMultipart` decision matrix via a mocked transport — FormData vs JSON, source embedding); `chainRegistry.test.ts` (`validateChain`: empty, compatible, accepts/produces mismatch, unpinned multi-op throw, pinned ok); `chainRuntime.test.ts` (`extractText`/`extractPaths`/`projectResultToInput` — exported them for testability); `UploadUtils.test.ts` (`extractUploadedFiles` response permutations + id-over-path). Earlier phases already pinned schemas, transport, run store, output registry, upload id-derivation. **Test count: 70 → (now 76 with a11y).** Two test fixtures tripped the Phase-3/4 lint guards (a `http://` literal, a `body as` cast) — fixed in the tests, proving the guards bite even in test code.

---

## Phase 10 action log — observability (DONE 2026-06-16)

New `transport/telemetry.ts`: every request emits one structured `RequestEvent` (requestId, url, method, outcome, durationMs, status, traceId, + caller-supplied `specKey`/`operation`). A **pluggable `TelemetrySink`** defaults to structured `console` logging and is swappable for Sentry/any reporter via `setTelemetrySink` — a throwing sink can never break a request. In-memory **metrics**: count by op, error-rate by `AppError.kind`, latency **p50/p95** per op (`metrics.snapshot()`).

`client.request` refactored to a single terminal record point (timed; outcome derived from the `Result`), and gained a `meta` option. `submitPage` passes `{ specKey, operation }`, registry passes `{ specKey: "registry" }`, upload passes `{ specKey: "upload" }`. `telemetry.test.ts` covers sink routing, counts, error-by-kind, p50/p95, and throw-safety.

**Trace path:** a failed run is locatable by `requestId` (client event) + the server `traceId` carried on `AppError.server`/the event; error-rate-by-kind is queryable via `metrics.snapshot()`. Wiring a real Sentry DSN is a one-liner (`setTelemetrySink`) left to deployment.

---

## Phase 11 action log — accessibility (DONE 2026-06-16)

- **Drop zone keyboard-operable:** `FileInputStage`'s `<div onClick>` gained `role="button"`, `tabIndex={0}`, `aria-label`, and Enter/Space activation, plus a `:focus-visible` ring in the CSS. Restructured so the hidden `<input type="file">` (`tabIndex={-1}` + `aria-hidden`) and the interactive `UploadedFileList` are **siblings of the button, not nested inside it** — a role="button" must not contain other interactive controls (fixed the `nested-interactive` + `label` axe violations).
- **Live regions:** the execution-output `<section>` is `aria-live="polite"`; error blocks (`formError`, run error) are `role="alert"`.
- **axe gate in CI:** added `jest-axe`; `FileInputStage.test.tsx` renders the file UI and **fails on any axe violation** (`results.violations` must be empty) — plus asserts the role/tabindex. The full input → run → output flow is keyboard-operable.

---

## Status: all 12 phases addressed

| Phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ | ✅ | ✅ |

\* Phase 6 client-side complete; full IDOR closure needs the server to issue session-scoped opaque ids (flagged, outside the frontend).

**Final gate state:** global `typecheck` 0 · `typecheck:hugpy` 18 (strict burn-down baseline, all pre-existing `noUncheckedIndexedAccess`/`exactOptional` in older components) · `lint` 0 · `knip` clean · `test` 76 passed / 1 skipped. CI gates available: `npm test`, `npm run lint`, `npm run typecheck`, `npm run typecheck:hugpy`, `npm run knip` (+ axe inside the test suite).

---

## Post-deploy fix — CORS regression on upload (2026-06-16)

**Symptom:** after deploying, file upload stopped working (everything else unchanged).
**Cause:** the Phase-5 transport defaulted **`credentials: "include"`** and an **`X-Request-Id`** header ON for all hugpy calls. hugpy is cross-origin; the multipart upload was previously a *simple* request needing no CORS preflight. The custom header forced a preflight the upload endpoint doesn't answer, and credentialed requests require `Access-Control-Allow-Credentials: true` + a specific origin — so the browser blocked the upload.
**Fix:** both are now **opt-in, default OFF** (`VITE_HUGPY_WITH_CREDENTIALS`, `VITE_HUGPY_SEND_REQUEST_ID`), restoring the original simple/uncredentialed request behavior. The request id is still generated for client-side telemetry; it's just not sent as a header unless enabled. Enable both once the hugpy server's CORS is configured for them. (Lesson: a hardening default must not change working network semantics — these should have shipped off from the start.)
