# hugpy · Video Intelligence UI — Roadmap

Sibling of `dev/media_intelligence_ui/` — same serving posture (Vite/React, built
to `dist/`, served by the dev webserver; optional `hf-space/` mirror), same
transport/session plumbing, new stations. Architecture, schemas, and job wiring
live in the companion doc: **[hugpy_video_intelligence_map.md](hugpy_video_intelligence_map.md)**.
This file is the *sequence*; the map is the *shape*. When they disagree, fix the
map first.

---

## North star — the sample image

![Generate-video sample UI](assets/sample-generate-video-ui.png)

`assets/sample-generate-video-ui.png` (canonical copy of
`assets/Screenshot from 2026-07-02 21-20-54.png`) is the visual target for the
end state. What it shows, mapped to our stations:

| Region of the sample | Our equivalent |
|---|---|
| Left rail — model, resolution, aspect ratio, frames/sec, duration | **Frame / Model Config station** — `ModelConfig` dropdown + `FrameExtractSpec` knobs (`fps`, `quality`, `fmt`, `window`, `max_frames`) |
| Frame thumbnail strip under the settings | Frame grid output of `frame_extract` (virtualized/paginated) |
| Large center preview | Job result viewer, subscribed by `job_id` (SSE/poll) |
| Prompt bar along the bottom | **Generate station** — `MediaInputReceptacle` collecting ordered text/image/video parts → `GenerateImageSpec` |
| Top tabs (Gallery / Generate / Edit) | PageSpec-registered station switcher |

The sample is a *layout and affordance* reference, not a spec — our knobs are the
explicit ones from the map (§4.3), no silent defaults.

> ⚠ `assets/Pasted image.png` is a truncated/corrupt PNG (32 KiB, fails to
> decode). Left in place per no-deletion policy; do not reference it.

---

## Phases

Ordering follows the map's build order (§11): the backbone
(schema → job → worker → subscribe) is proven on the cheapest capability first,
then every later phase is just another spec + runner + station on the same rails.

### Phase 0 — Scaffold ✅ (done 2026-07-02)
- Clone the `media_intelligence_ui` Vite/React skeleton (`entry.tsx`, `index.html`,
  `vite.config.ts`, `src/config.ts`, `src/session.ts`, `src/transport/`) into this
  tree; strip media-console-specific stations.
- Register an empty Video Intelligence PageSpec so the shell serves from `dist/`.
- **Done when:** the empty shell builds and loads (verified via `vite preview`;
  serving at the dev URL is the Phase 8 docroot wiring).
- *As built:* station registry + shell in `src/stations/` (four placeholder
  stations declaring their phase and planned affordances), transport cloned
  verbatim, config with the demo layer stripped, Vite base `/video/`.

### Phase 1 — Media substrate ✅ (done 2026-07-02)
- `MediaRef` schema + MediaStore ingest; metadata resolved once at ingest.
- **Done when:** upload of image/audio/video yields an immutable `MediaRef` with
  correct dims/duration/fps/sample-rate. ✔ verified.
- *As built:* new additive backend package `abstract_hugpy_dev/video_intel/`.
  `media_schema.py` (`MediaRef` + `make_media_ref`), `media_store.py`
  (`ingest(path)` resolves metadata ONCE via a single
  `ffprobe -show_format -show_streams` call — the faithful media analog of the
  map's "layered resolver"; authoritative `kind` from the stream set; realpath
  jailed under `UPLOADS_HOME`/`DEFAULT_ROOT`). Verified in the VM venv:
  image→640×480 kind=image; audio→3.0 s / 16000 Hz / 1 ch; video→320×240 /
  2.0 s / 25 fps. (See map §12.)

### Phase 2 — Crop, headless ✅ (done 2026-07-02)
- `CropSpec` (one spec, spatial/temporal as orthogonal optional axes — map §4.2),
  `(ffmpeg, crop)` pure runner, `crop` entry in `JOB_REGISTRY`.
- Errors as `JobError` data; single-writer job state (map §6).
- **Done when:** enqueue → claim → run → `JobResult` round-trips from a test
  script with no UI. ✔ verified.
- *As built:* `crop_schema.py`, `result_schema.py`, `job_schema.py`
  (`JOB_REGISTRY` — only `crop` registered now; frame/audio/generate entries are
  appended as Phase 4+ lands), `runners/ffmpeg_crop.py` (pure
  `run_crop(spec, job_id) -> JobResult`; spatial `-vf crop`, temporal `-ss/-t`;
  expected failures return `JobError` data: `region_out_of_bounds`,
  `ffmpeg_failed`, `missing_output`), and `media_bus.py` — a **new durable job
  bus** (stdlib `sqlite3`, WAL, at `$DEFAULT_ROOT/video_intel/media_jobs.db`).
  It was built fresh because the pre-existing `comms/jobs.py` job_store is a
  lifecycle tracker with no enqueue/claim/dispatch and no result storage.
  Path: `enqueue → claim (atomic BEGIN IMMEDIATE) → run_claimed → JobResult
  written once`; single-writer enforced by `claim_token`; `work_once()` drives it
  headlessly. Verified in the VM venv: spatial crop round-trips to a 200×150
  artifact on disk; factory raises locally on bad axis combos; an out-of-bounds
  bbox returns a `JobError` (data, `status=failed`), never a raise. (See map §12.)

### Phase 3 — Image Crop station (backbone proof) ✅ (done 2026-07-02)
- `RegionEditor` contract → `SpatialRegionEditor` (bbox drag/resize, aspect lock,
  numeric x/y/w/h, presets, multi-crop queue).
- Image Crop station end-to-end: `MediaRef(image)` → editor → `CropSpec` →
  enqueue → subscribe → result in the store.
- **Done when:** a user crops an image entirely through the bus. This proves the
  whole enqueue → job → subscribe loop; everything after rides these rails.
  ✔ verified end-to-end at the public URL.
- *As built:* Backend HTTP surface `flask_app/app/routes/video_routes.py`,
  dual-mounted bare + `/api` like `worker_bp`: `POST /video/ingest`,
  `POST /video/jobs/crop`, `GET /video/jobs/<job_id>`,
  `GET /video/media?handle=<uri>` (bytes, jailed). The worker daemon is started
  per gunicorn process in `get_hugpy_flask()` (guarded + try/excepted so it can
  never break app creation); with 3 gunicorn workers the bus's atomic claim makes
  exactly one process run each job, and any process can answer a poll because
  state/results live in the shared SQLite store. Frontend: `src/regions/types.ts`
  (RegionEditor contract), `src/regions/SpatialRegionEditor.tsx` (bbox kept in
  NATIVE pixels via display→native scale conversion; aspect-lock, presets, numeric
  x/y/w/h, multi-crop queue), `src/video/contract.ts`, `src/stations/useCropJobs.ts`
  (800 ms poll loop, 60 s cap), `src/stations/ImageCropStation.tsx`; the
  `image-crop` station is flipped to `active` in the registry and rendered by the
  shell (other three remain placeholders). Verified: live public round-trip
  (upload 1640×1309 → ingest → enqueue → **daemon**-processed → 120×90 crop →
  served by handle) ALL PASS; `dev.hugpy.ai/video/image-crop` serves the new
  bundle whose code carries the station. All calls go through the transport
  `request<T>()`/`Result<T>` layer with URLs sourced only from `config.ts`.
  (See map §12 for the as-built job-bus + contract.)

### Phase 4 — Frames + model config ✅ (done 2026-07-03)
- `FrameExtractSpec` + `(ffmpeg, frame_extract)` runner (fan-out per `wmtrim`'s
  parallel-ffmpeg pattern).
- Config station: fps / quality-per-fmt / fmt / window / `max_frames` (loud cap),
  model dropdown from the `ModelConfig` registry filtered by task.
- Virtualized frame grid; frames land as `MediaRef`s, never inlined.
- **Done when:** the left-rail + frame-strip portion of the sample image is
  functionally real.
- *As built:* `frame_schema.py` + `runners/ffmpeg_frames.py` (single fps-filter
  ffmpeg pass; `frame_cap_exceeded` refused LOUDLY up front; `BoundedSemaphore(1)`
  so a long extract can't starve gunicorn); `POST /video/jobs/frame_extract`;
  UI `FrameExtractStation` + `useFrameJobs` + `src/video/mediaLibrary.ts`
  (sessionStorage FieldSpec-analog propagating produced MediaRefs to Generate);
  model dropdown via `useModels` on `GET /v1/models` filtered to text-to-image.
  *Verified live (public URL, 2026-07-03):* extract → 6 frames, each servable by
  handle; capped job → `failed` + `JobError.code=frame_cap_exceeded`.

### Phase 5 — Audio Crop station ✅ (done 2026-07-03)
- `audio_extract` JobSpec/runner (video → audio-track `MediaRef`).
- `TemporalRegionEditor` (waveform, region handles, scrub/loop, numeric
  start/end, ~~spectrogram toggle~~ (deferred — future work), multi-region).
- Station accepts `MediaRef(audio)` directly or `MediaRef(video)` via the
  `video→audio` chain.
- **Done when:** temporal crop of an uploaded video's audio round-trips.
  ✔ verified end-to-end at the public URL.
- *As built:* Backend (additive): `audio_schema.py`
  (`AudioExtractSpec(source, fmt="wav")` + `make_audio_extract`; `fmt ∈
  {wav,mp3,m4a}`, codec map wav→pcm_s16le / mp3→libmp3lame / m4a→aac) and
  `runners/ffmpeg_audio.py` (pure `run_audio_extract`: `ffmpeg -vn -acodec …`,
  output under `DEFAULT_ROOT/video_intel/audio/`, re-ingested → `MediaRef(audio)`;
  EXPECTED failures as JobError DATA: `missing_input`, `not_a_video`,
  `no_audio_track`, `ffmpeg_failed`, `missing_output`). Registered in
  `JOB_REGISTRY` + runner `DISPATCH` + `media_bus.SPEC_DESERIALIZERS`; route
  `POST /video/jobs/audio_extract` (dual bare+`/api`). **Temporal crop of the
  extracted audio reuses the EXISTING `crop` job** (`ffmpeg_crop.py` temporal
  branch) — no duplicate crop path. Gate tool `video_intel/_selftest_audio.py`.
  Frontend: `regions/TemporalRegionEditor.tsx` (canvas waveform via Web Audio
  `decodeAudioData` over `mediaBytesUrl` bytes — transport gained an
  `expect:"arraybuffer"` branch for binary; draggable region band + numeric
  start/end + region-loop `<audio>` playback + multi-region queue),
  `stations/AudioCropStation.tsx` (upload audio→editor, or video→`audio_extract`
  →editor; cropped clips play in-station and push to `mediaLibrary`),
  `useAudioExtractJob.ts` + `useAudioCropJobs.ts`; `audio-crop` flipped `active`.
  *Verified live (public URL, 2026-07-03):* upload video-with-audio → ingest
  (kind=video, sample_rate=44100) → `audio_extract` → servable 3.02 s WAV →
  temporal crop `[0.5,1.5)` → servable 1.0 s audio clip. `no_audio_track` path
  refuses a silent video with clean JobError data (headless). The served public
  bundle carries `video/jobs/audio_extract` + `decodeAudioData` + the station.
  (See map §12.3.)

### Phase 6 — Generate station (text + image) ✅ (done 2026-07-03)
- `GenerateImageSpec` + ~~`(diffusers, generate_image)`~~ a THIN
  `(hugpy, generate_image)` runner that REUSES the existing inference plane
  (`managers.dispatch.execute_prompt(task="text-to-image", …)`) — no parallel
  diffusers stack; generation rides the same worker control plane as everything
  else (map capability #1).
- Ordered prompt composer collects text + library image picks; `mediaLibrary.ts`
  propagation closes the crop/frames → generate loop.
- **Done when:** text + cropped-frame conditioning produces an image through the
  bus, with the prompt bar from the sample image as the input surface.
- *As built:* `gen_schema.py` + `runners/imagegen.py` (lazy-imports the managers
  tree so media runners stay importable independently; pool left UN-set);
  `POST /video/jobs/generate_image`; UI `GenerateStation` + `useGenerateJob`,
  result image lands back in the media library. *Verified live (public URL,
  2026-07-03):* text prompt → real 512×512 PNG (590 KB) through an actual fleet
  model, servable by handle.

### Phase 7 — Video as a generation input ✅ (done 2026-07-03)
- Register the `video → frame_extract → frame-pick → image-conditioning` chain in
  the chain registry (map §4.4); the diffusion runner never sees video unless a
  `ModelConfig` explicitly declares native video conditioning.
- Frame selection: manual pick from the existing grid, uniform-N fallback
  (map §10.4 default).
- **Done when:** dropping a video part into Generate resolves through the chain
  with no runner special-casing.
- *As built:* `chains.py` `resolve_video_parts` (uniform-N) runs in the ENQUEUE
  path (`video_routes.py`), so the generation runner only ever sees text/image
  parts. *Verified live (public URL, 2026-07-03):* generate with a video part →
  chain resolved to image parts → real generated image, `status=done`.

### Phase 8 — Parity polish + deploy ✅ (done 2026-07-03, hf-space still open)
- ~~Layout pass against the sample image~~ **done (2026-07-03):** Frames +
  Generate recomposed to the sample's rail / center-preview / thumbnail-strip /
  prompt-bar layout in the `app.css` token system (shared `.vi-studio*` grid
  primitives; final bundle `index-C-DqCgPV.js`). Layout is CSS-verified — eyeball
  the stations at wide and <62rem widths for the grid collapse.
- **Poll-cap fix (2026-07-03):** job-poll hooks no longer declare failure at the
  cap — they surface "still running, check again" with a re-poll affordance;
  generation gets a longer cap (worker cold-loads take minutes).
- ~~Wire into the dev docroot~~ **done early (2026-07-02):**
  `https://dev.hugpy.ai/video/` is live. Two docroots serve the arm, mirroring
  `/media` — update BOTH on every deploy (`npm run build`, then):
  1. webpack `:7001` (public dev.hugpy.ai pages via host nginx) statically
     mounts `../video_intelligence_ui/dist` at `/video` — `dev/ui/webpack.config.js`.
  2. gunicorn `:7002` (`/api` + the VM-internal self-contained surface) serves
     `console_dist/video/` — rsync `dist/` → `dev/abstract_hugpy_dev/src/abstract_hugpy_dev/console_dist/video/`;
     the SPA arm fallback in `flask_app/wsgi_app.py` covers `media` + `video`.
     Stale bundle hashes go to `console_dist/video/assets_archive/` (never delete).
- Decide whether an `hf-space/` mirror is wanted (OPEN — user decision).
- **Done when:** dev URL serves the full four-station console. ✔ All five
  stations' flows verified live 2026-07-03: full e2e (gates 4a/4b/5/6/7) ALL PASS.

### Workbench restructure ✅ (2026-07-03)
Frontend-only UX pass, no backend change (bundle `index-CCvxOdfx.js` /
`index-nlfi_lix.css`, live at `https://dev.hugpy.ai/video/`):
- The three crop/frames stations are now **sub-tabs of ONE `WorkbenchStation`**
  (`src/stations/WorkbenchStation.tsx`) laid out as `[ session-library sidebar |
  tabbed main ]`. Top nav collapses to **two tabs — Studio + Generate** (driven
  by `NAV_SECTIONS` derived from a new `group` field on `StationSpec`; Generate
  keeps its own top tab + route).
- **Deep links preserved:** `/video/image-crop`, `/video/audio-crop`,
  `/video/frames` each render the workbench with that sub-tab active (studio
  routed on a dynamic `/:workbenchTab` segment, ranked below the static
  `/generate`); the Studio top tab is active for any of the three.
- **Sidebar = the session media-library queue** (`WorkbenchSidebar.tsx`): a live
  view over `mediaLibrary.ts` (newest-first, kind badge, image thumbnail / inline
  `<audio>` / video placeholder, dims-duration-mime meta), static across sub-tab
  switches (one workbench instance stays mounted). Per-item "Use in Generate"
  (link to the Generate tab, where the shared library is already pickable) +
  "Remove" (forgets the LOCAL entry only via new `removeFromLibrary` — never
  deletes server media). Collapses behind a toggle at <62rem.
- Users must **hard-reload** to pick up the new bundle (SPA cache).

#### Follow-up ✅ (2026-07-03, bundle `index-h3yot71G.js`)
- **Sidebar hoisted to the shell.** The session-library sidebar + its `<62rem`
  toggle now live in `StationShell.tsx` with state ABOVE the `<Routes>`, so the
  `[ sidebar | station area ]` grid wraps EVERY route — the sidebar is now static
  and present on **Generate** too, not just Studio. `WorkbenchStation.tsx`
  de-nested to render only its sub-tabs + body into the shell's
  `.vi-workbench-main` column (exactly one sidebar). No CSS change — reuses the
  existing `.vi-workbench*` classes + the 62rem collapse rule.
- **Auto-library for every produced artifact.** Image crops (`useCropJobs.ts`,
  origin `image-crop`) and every extracted frame (`useFrameJobs.ts`, origin
  `frames`) now land in `mediaLibrary` the moment their job completes — joining
  audio clips + generations, which already did. De-dup by `ref.uri` keeps the
  Frames "Add N to Generate" button a harmless no-op. The intermediate
  video→audio track (`useAudioExtractJob.ts`) is intentionally still not pushed.
- **Add-to-prompt from the sidebar.** On the Generate route a sidebar image/video
  item shows **"Add to prompt"**, injecting it as an ordered composer part via a
  tiny typed seam (`src/video/composerBridge.ts`: single-handler register +
  `requestAddPart`; GenerateStation registers while mounted). Off Generate the
  item keeps its "Use in Generate" link; audio items get no prompt action.

#### Follow-up ✅ (2026-07-03, bundle `index-AyWXgElg.js` / `index-dJJ-NL00.css`)
- **Generate folded into the workbench tabs.** Giving the `generate` registry entry
  `group:"studio"` collapses the old Studio/Generate top-level split — there is now
  ONE tab strip [Image Crop | Audio Crop | Frames & Models | Generate] inside the
  single workbench, all sharing the persistent shell sidebar. The static `/generate`
  route retires (`ungrouped` is empty); `/video/generate` now flows through
  `/:workbenchTab` → `WorkbenchStation` → `GenerateStation` as the 4th sub-tab.
  Route contract unchanged (deep links to all four ids work; unknown → `/image-crop`).
  The `"generate"`-segment route-gate for "Add to prompt", `composerBridge`, and all
  job hooks are untouched. The now-dead `NAV_SECTIONS`/`GROUP_TITLES` top-nav model
  was retired from `registry.ts`; `group` stays an optional (now single-valued) field
  in `types.ts`.
- **Sitewide navbar adopted.** The `.vi-brand`/`.vi-tabs` header row is replaced by a
  faithful LEAN port of the shared hugpy `Navbar` from the media arm
  (`media_intelligence_ui/src/chat/src/ui/{Navbar,BrandMark}.{tsx,css}` +
  `assets/hugpy-mark.png`) copied into `src/nav/` + `src/assets/` — no cross-arm
  import, no new deps. Renders at the top above the workbench; links Docs/Console/Media
  + a current **Video** entry, brand → site home via `hugpyConfig.siteUrl`. CSS made
  self-contained (nav color custom-props moved onto `.hugpy-navbar`); the `demo/mode`
  embed helper and the two chat-scope override rules were dropped.

#### Follow-up ✅ (2026-07-03, bundle `index-DmrNkE-o.js` / `index-CNiekhe8.css`)
- **Stations always fully expanded — upload no longer gates the layout.** Every
  station now renders its FULL layout (settings rail, preview, strip, action bar) on
  tab entry; the center item-view is a **browse + drag-and-drop receptacle** while no
  item is loaded. New shared `src/video/DropReceptacle.tsx` (pure intake UI, NO upload
  logic) — click/Enter/Space opens the file picker, drop feeds `dataTransfer.files[0]`;
  it just surfaces a `File` and calls the station's EXISTING `onPick`/`onPickVideo`
  handler (same FormData→`uploadUrl`→`videoIngestUrl` path, unchanged). All four
  stations use the ONE component (no copies). Per station: ImageCrop/AudioCrop swap the
  `SpatialRegionEditor`/`TemporalRegionEditor` for the receptacle when `!source`
  (`!audioSource`, extract-progress UI still owns the stage mid-extract) and disable Run
  until a source exists; FrameExtract drops the outer `{source && …}` gate so the whole
  `.vi-studio` grid + rail knobs render immediately with the receptacle inside
  `.vi-studio-preview` and Extract disabled until a source loads; Generate's main
  workspace was ALREADY ungated (text-driven) so only its video-part intake gained the
  receptacle (`onPickVideo`, accept `video/*`). Empty-state knobs render visible but
  inert. Marker: `Drop a file or click to browse` + class `vi-dropzone`.
- **Uniform viewport-height workbench.** Replaced the sidebar's `sticky` +
  `max-height:calc(100dvh-3rem)` with a flex/`min-height:0` chain so the sidebar AND all
  four station panels fill one viewport-derived box below the navbar+tabs and scroll
  internally: `.vi-shell{height:100dvh}` → `.vi-main{flex:1;min-height:0;overflow:hidden}`
  → `.vi-workbench{height:100%;min-height:0;align-items:stretch}` →
  `.vi-workbench-main{display:flex;flex-direction:column;min-height:0;height:100%}` with
  `.vi-subtabs{flex-shrink:0}` (fixed top row) + `.vi-workbench-body{flex:1;min-height:0;
  overflow-y:auto}` (the scroll container) and `.vi-workbench-sidebar{height:100%;
  min-height:0;overflow:auto}`. No magic numbers. Frame strip stays horizontal-scroll.
- **Mobile hamburger drawer — mirrored from the media arm.** The media arm
  (`media_intelligence_ui`) puts a 3-line "menu" glyph in `chat/src/ui/ThreadHeader.tsx`
  toggling a `sidebarCollapsed` boolean at the 768px `md:` breakpoint; the sidebar is an
  off-canvas drawer (`max-md:fixed inset-y-0 left-0 z-40 -translate-x-full` → `translate-x-0`)
  with a `fixed inset-0 z-30 bg-black/40 md:hidden` scrim (`main.tsx`) that closes on tap.
  Mirrored here in hand-written CSS (this arm is not Tailwind) at the arm's OWN 62rem
  pivot for consistency: the in-flow `.vi-workbench-toggle` button is retired; a mobile-only
  `.vi-hamburger` glyph (aria/title "Session library") is the leading child of the tab
  strip (directly under the navbar) and toggles the same `sidebarOpen` state; `<62rem` the
  `.vi-workbench-sidebar` becomes `position:fixed;top:0;left:0;height:100dvh;width:min(20rem,
  85vw);z-index:40;transform:translateX(-100%)` → `.is-open` `translateX(0)` with a
  `.vi-scrim` (`z-index:35;background:rgba(0,0,0,.45)`); closes on scrim click, Escape, and
  tab navigation. Desktop keeps the always-visible inline sidebar.
- **Deploy:** rsync additive; superseded `index-AyWXgElg.js` + `index-dJJ-NL00.css`
  archived by exact name to `console_dist/video/assets_archive/` (`.claude/` untouched).
  All five public routes 200; live HTML references the new hash; served bundle carries the
  new `Drop a file or click to browse` / `vi-dropzone` markers alongside `Add to prompt`,
  `Session library`, `hugpy-navbar`, the four tab ids, and the `Still running on the
  server — check again.` invariant. SPA users hard-reload.

#### Follow-up ✅ (2026-07-03, bundle `index-DpLofb60.js` / `index-uvz1wgvu.css`)
- **Session-wide job tracker — processes survive sub-tab switches AND page reloads.**
  Fixes the reported amnesia: switching sub-tabs unmounted a station → its hook's
  private polling interval + JobCard state died → the user returned to an apparently
  idle tab while the durable server bus kept working, and outputs never reached the
  library (the unmounted hook's done-branch never fired). New module-level store
  `src/video/jobTracker.ts` mirrors `mediaLibrary.ts` (module `cache`, `listeners`
  Set, `emit`, one-time storage bind, `getSessionId()` sessionStorage write-through
  under key `vi.jobTracker.v1`, `useSyncExternalStore`). ONE `window.setInterval`
  (800ms) polls ALL active jobs via `jobStatusUrl` + `request<T>()` regardless of
  which tab is mounted; arms while ≥1 job is non-terminal/non-capped, idles otherwise.
  Per-job cap reuses `pollCaps.ts` (`JOB_POLL_CAP_MS` 180s; `generate_image` gets
  `GENERATE_POLL_CAP_MS` 300s) — on cap it pauses THAT job (not failed), surfaces the
  verbatim `Still running on the server — check again.` copy + a "Check again" resume
  that re-anchors the poll clock. Persists active + terminal records; on module init
  rehydrates and RESUMES polling anything non-terminal (fixes reload amnesia). A poll
  returning `{status:null}` (id gone server-side) marks the record expired and stops
  polling it (no crash) — verified against the live server.
- **The five hooks are now thin selectors over the tracker.** Each keeps its enqueue
  request-body construction + POST, then calls `trackJob(...)`; its returned status /
  outputs / `capped` / `running` / `resume` / `reset` are derived (useMemo) from
  `useJobTracker()` filtered to its station — public API to each station body is
  unchanged (`useCropJobs`/`useAudioCropJobs` still return `{jobs,runCrops,resume,
  reset}`; the three single-job hooks still return `{jobId,status,output|frames,error,
  capped,running,run,resume,reset}`). Stations re-entering a tab show their in-flight
  AND recent jobs again.
- **Done-push centralized.** The single library-push site is `pushOutputs()` in
  `jobTracker.ts`, run once per record on `done` (frames→`frame N` per output;
  image/audio crop→carried `crop W×H`/`clip a–bs` label; generate→`generated`;
  audio_extract→none, still intermediate). All five hook-level `addToLibrary` pushes
  were removed; `mediaLibrary` uri de-dup remains a safety net → no double-adds. This
  is what actually fixes "outputs vanish when I leave the tab" — the push no longer
  depends on the station being mounted.
- **Sidebar "Processes" section** (`WorkbenchSidebar.tsx`, above the library list,
  visible from ANY tab): heading `Active processes`, per row a spinner glyph while
  running, station name, `.vi-status-*`-colored status, live mm:ss elapsed, label,
  failed/expired/capped states; capped rows show the poll-cap copy + resume; each row
  is a `NavLink` to its station tab. New `.vi-proc*` CSS classes (+ `@keyframes
  vi-proc-spin`) reuse the `--vi-*` tokens.
- **Deploy:** frontend-only, no backend change (server bus was already durable). rsync
  additive; superseded `index-DmrNkE-o.js` + `index-CNiekhe8.css` archived by exact
  name to `console_dist/video/assets_archive/` (`.claude/` untouched). All five public
  routes 200; live HTML references the new hash; served bundle carries new markers
  `vi.jobTracker.v1` / `Active processes` / `vi-proc` alongside `Still running on the
  server — check again.`, `Session library`, `Add to prompt`, `hugpy-navbar`, the four
  tab ids. A live `frame_extract` e2e ran queued→done on the `/api/video/jobs/<id>`
  seam. SPA users hard-reload.

### Scene generation ✅ (2026-07-03, bundle `index-mtHG1DJ6.js` / `index-DM-ij-jO.css`)
- **One Generate query → N consecutive frames + an assembled playable clip.** New
  backend job `generate_scene` on the existing rails: `scene_schema.py`
  (`GenerateSceneSpec` frozen dataclass — ordered `GenPromptPart`s, `model_id`,
  `width/height/steps/guidance`, explicit `n_frames`/`fps`/`assemble`, `seed`/`motion`/
  `negative`; `make_generate_scene` validates locally and refuses over-cap up front with
  `frame_cap_exceeded`, `FRAME_CAP=24`), `runners/scene.py` (`run_generate_scene`), and
  a new `POST /video/jobs/generate_scene` route. The GPU-worker guard from `imagegen.py`
  was factored into a shared `runners/_gpu_guard.py` (`guard_gpu_worker`) reused by both
  runners — behavior byte-identical.
- **Coherence mode shipped: seed + prompt-schedule v1 (NOT img2img).** Investigated the
  managers plane first: the only registered image pair is `("transformers","text-to-image")`
  — no img2img `(framework,task)` runner/builder, no init-image seam on `ImageGenRequest`,
  and no fleet model advertises an img2img task (`sd-turbo` → `["text-to-image"]`). So the
  runner loops N sequential `text-to-image` generations through the SAME inference plane
  imagegen uses (`managers.dispatch.execute_prompt`, remote-GPU-bound), with a
  deterministic per-frame seed schedule (`seed+i` when a base seed is set) + a per-frame
  prompt progression (base + `", frame {i+1} of {n}"` + `motion` template with safe
  `{i}`/`{n}` substitution). Runner is structured so an img2img chain drops in later
  (see map §12.5 for the exact upgrade path).
- **img2img chaining v2 shipped as code + UI, STAGED dark pending a fleet release
  (bundle `index-BwCqe9va.js` / `index-BFC06I-3.css`).** All three v1 blockers are now
  resolved additively in the managers plane: a new `Img2ImgRunner`
  (`AutoPipelineForImage2Image`) + `_build_img2img_request` registered for
  `("transformers","image-to-image")`, `image_path`+`strength` optional fields on
  `ImageGenRequest`, and `RUNNER_PAIRS` updated (`HF_TASK_TO_TASKS` intentionally NOT — keeps
  discovery from auto-advertising unvetted flux/Qwen img2img; sd-turbo is the only supported
  img2img model, flux is not shipped). `scene.py` now does true chaining when a start-frame
  image part is present (`chain=true` sequential, `chain=false` all-off-the-start-frame; no
  image part → v1), `imagegen.py` uses a single image part as the img2img init (else v1 with a
  loud log), and the UI gains a `strength` knob (default 0.45) + `chain` toggle (default true)
  with a chain-aware poll cap. **The advertisement flip (`sd-turbo → tasks += image-to-image`)
  is HELD** because the GPU worker runs a pinned PyPI wheel (`abstract_hugpy_dev==0.1.92`) with
  no img2img runner — going live is a fleet-wide release (the mission's declared stop
  boundary). So on live central the runner's availability probe returns unavailable and a
  chained start-frame scene fails with the honest retryable
  `image_to_image_unavailable` ("image-to-image not available on the fleet") — proven by e2e
  GATE 9; the real generation was proven locally (VM CPU sd-turbo img2img, 256×256). Go-live =
  publish the engine to PyPI → converge op via `/ops/update` → flip the one-line
  `sd-turbo.tasks` (map §12.5 follow-up has the full runbook + rollback snapshot).
- **Assembly:** when `assemble`, frames are written as `frame_%05d.png` under
  `DEFAULT_ROOT/video_intel/scenes/<job_id>/` and ffmpeg builds a browser-playable mp4
  (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`, even-dim scale pad), guarded by
  a module-level `BoundedSemaphore(1)` so it can't starve gunicorn; the video MediaRef is
  appended LAST in `outputs`.
- **Frontend:** Generate station gains an Image | Scene mode switch (reuses the composer,
  model dropdown, and shared knobs; adds `n_frames`/`motion`/`fps`/`assemble`). New tracker
  kind `generate_scene` on `jobTracker.ts` with an n_frames-scaled poll cap (`pollCaps.ts`:
  300s base + 60s/frame) — the verbatim `Still running on the server — check again.` copy
  preserved. Result view: the N frames as a `.vi-frames-grid .vi-studio-strip` + the
  assembled clip in the arm's first `<video>` player. Outputs auto-land in the session
  library via the tracker's single done-push (frames `scene frame N`, clip `scene clip`).
- **Verification:** three VM selftests pass (base + gen regression + new `_selftest_scene.py`,
  whose assembly unit produces a real mp4 GPU-independently); ONE `hugpy-api-dev` restart
  (active before/after); `/api/version`, `/media/`, `/video/`, `/` all 200. Live e2e GATE 8
  (`/tmp/vi_e2e.py`): scene job `done` with 4 outputs `[image,image,image,video]`, each
  servable 200 via `/video/media?handle=`; full e2e ALL PASS, no regressions. Remote-GPU
  proof: guard refuses local + `HUGPY_VIDEOGEN_LOCAL` unset → the daemon journal shows the
  frames dispatched to op (`192.168.1.113:9100/infer`). Deploy additive; superseded
  `index-DpLofb60.js` + `index-uvz1wgvu.css` archived by exact name (`.claude/` untouched);
  served JS carries `scene frame`/`scene clip` markers. SPA users hard-reload.

### Backlog (post-roadmap)
- hf-space mirror decision (above).
- Spectrogram toggle in TemporalRegionEditor (stretch, skipped in Phase 5).
- HTTP Range support on `GET /video/media` if audio scrubbing needs it.
- Register the `(transformers, text-to-speech)` runner + builder so the
  discovered `chatterbox` model activates (sibling workstream).

---

## Standing constraints (from the map — do not relitigate per phase)

1. One `CropSpec`; spatial and temporal never diverge into separate crop types.
2. `MediaRef` immutable; metadata resolved once at ingest.
3. Everything through the job bus; runners pure `spec → JobResult`; UI read-only
   on state.
4. Registries over globals: `JOB_REGISTRY`, runner dispatch, `ModelConfig`,
   chain registry, PageSpec.
5. Explicit knobs, loud caps — no silent fps or frame-count defaults.

## Open questions

Tracked in the map §10 with defaults (audio-crop consumer, native-video models,
`quality` UX, frame-pick strategy). Defaults apply until overridden; record any
override in both docs.
