// Single source of truth for hugpy service URLs (video-intelligence arm).
//
// Cloned from media_intelligence_ui/src/config.ts. Demo layer: CANNED only
// (?demo=1 — see demo/mode.ts; deliberately no live flavor, since video jobs
// drive real worker GPUs). Same conventions: one VITE_* base var defaulting to
// the SAME-ORIGIN current-hugpy API (/api), derived URLs with per-URL
// overrides, and this being the ONE module permitted to hold URL literals.

import { isCanned } from "./demo/mode";
import { withShareParam } from "./share";

const DEFAULT_API_BASE = "/api";

// Read import.meta.env without depending on vite/client ambient types (the scoped
// strict tsconfig doesn't include them).
const env: Record<string, string | undefined> =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env ??
  {};

function read(key: string): string | undefined {
  const v = env[key];
  return v && v.trim() ? v.trim() : undefined;
}

function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

function readBool(key: string, dflt: boolean): boolean {
  const v = read(key);
  if (v == null) return dflt;
  return !/^(0|false|no|off)$/i.test(v);
}

const apiBase = stripTrailingSlash(read("VITE_HUGPY_API_BASE") ?? DEFAULT_API_BASE);

export const hugpyConfig = {
  /** Base host for all hugpy API calls. */
  apiBase,
  /** Public hugpy site — CTA target. Override with VITE_HUGPY_SITE_URL. */
  siteUrl: stripTrailingSlash(read("VITE_HUGPY_SITE_URL") ?? "https://hugpy.ai"),
  /**
   * Hosted demo media tree (canned-demo samples live OUT of the bundle).
   * Build-time override: VITE_HUGPY_DEMO_MEDIA_BASE. Runtime overrides
   * (?mediaBase= / window.__HUGPY_MEDIA_BASE__) are layered in demo/mediaBase.ts.
   */
  demoMediaBase: stripTrailingSlash(
    read("VITE_HUGPY_DEMO_MEDIA_BASE") ?? "https://hugpy.ai/demo-media",
  ),
  /** Multipart file upload endpoint. */
  uploadUrl: read("VITE_HUGPY_UPLOAD_URL") ?? `${apiBase}/uploads`,
  /**
   * Resolve an uploaded server path into a typed MediaRef (native dims, mime,
   * kind). Phase 3 — the first hop after upload for every media station.
   */
  videoIngestUrl: read("VITE_HUGPY_VIDEO_INGEST_URL") ?? `${apiBase}/video/ingest`,
  /**
   * char360 S4 — VIDEO → CHARACTER VIEW-SET extraction (Identities station,
   * "Extract from video"): POST { source:<MediaRef, kind:"video">,
   * target:"create"|"<slug>", char360_params?:{stride?,yolo_model?,min_h_frac?,
   * cluster_dist?,min_faces?} } → 200 { job_id, target }. Validation errors are a
   * flat 400 {error}; an unknown add-target slug is 404 {error}. The job itself
   * is polled on the generic `videoJobUrl`/`jobStatusUrl` (GET /video/jobs/<id>),
   * NOT a bespoke status route. Declared here so the extract control is a
   * pure-additive flip that never re-touches this URL-literal module.
   */
  videoExtractUrl:
    read("VITE_HUGPY_VIDEO_EXTRACT_URL") ?? `${apiBase}/video/identity-profiles/video-extract`,
  /**
   * k94 — ONE PATH identity creation (Identities tab, "Create identity"):
   *   from-video:  POST { source:<MediaRef, kind:"video">, name, mesh_params? }
   *                → 202 { job_id, name, slug, kind:"identity_from_video" }
   *                ONE chained char360 + Hunyuan3D job; on done ONE profile per
   *                detected character (slug, slug-2, …) with crops + canonical + GLB.
   *   from-images: POST { sources:[<MediaRef, kind:"image">…], name, mesh_params? }
   *                → 202 { job_id, recon_id, slug, profile, kind:"identity_mesh_build" }
   *                create profile + the one-click full build chained server-side.
   * Both jobs are polled on `videoJobUrl` (GET /video/jobs/<id>): `progress`
   * carries stage / progress / log_tail; `result.identities` (from-video) lists
   * what was minted. Declared here (the one URL-literal module).
   */
  identityFromVideoUrl:
    read("VITE_HUGPY_IDENTITY_FROM_VIDEO_URL") ?? `${apiBase}/video/identity-profiles/from-video`,
  identityFromImagesUrl:
    read("VITE_HUGPY_IDENTITY_FROM_IMAGES_URL") ?? `${apiBase}/video/identity-profiles/from-images`,
  /** Enqueue a crop job (spatial and/or temporal axis) on the job bus. */
  cropEnqueueUrl: read("VITE_HUGPY_CROP_ENQUEUE_URL") ?? `${apiBase}/video/jobs/crop`,
  /** Enqueue a frame-extract job (Frames & Models station) — one job → many frames. */
  frameExtractEnqueueUrl:
    read("VITE_HUGPY_FRAME_EXTRACT_URL") ?? `${apiBase}/video/jobs/frame_extract`,
  /**
   * Enqueue an audio-extract job (Audio Crop station, Phase 5) — pull a video's
   * audio track into ONE standalone audio MediaRef the temporal editor edits over.
   * Mirrors frameExtractEnqueueUrl: declared here so the station build is a
   * pure-additive registry flip that never re-touches this URL-literal module.
   */
  audioExtractEnqueueUrl:
    read("VITE_HUGPY_AUDIO_EXTRACT_URL") ?? `${apiBase}/video/jobs/audio_extract`,
  /**
   * Enqueue a text/image → image generate job (Generate station, later phase).
   * Declared here now so the Generate-station build is a pure-additive registry
   * flip that never re-touches this URL-literal module.
   */
  generateEnqueueUrl:
    read("VITE_HUGPY_GENERATE_URL") ?? `${apiBase}/video/jobs/generate_image`,
  /**
   * Enqueue a text/image → scene generate job (Generate station, Scene mode) — one
   * ordered multimodal prompt → N consecutive image frames + (optionally) one
   * assembled mp4 clip. Same job-bus + status seam as generateEnqueueUrl; declared
   * here so Scene mode is a pure-additive flip that never re-touches this
   * URL-literal module.
   */
  generateSceneEnqueueUrl:
    read("VITE_HUGPY_GENERATE_SCENE_URL") ?? `${apiBase}/video/jobs/generate_scene`,
  /**
   * Enqueue a GOAL-TIMELINE → movie generate job (Generate station, Movie mode) — an
   * ordered, contiguous list of goals tiling [0,total) → an N-segment movie (segment
   * frames + one assembled mp4 LAST), with an opt-in vision director loop. Same
   * job-bus + poll/cancel seam as generateSceneEnqueueUrl (the poll returns a NESTED
   * movie progress); declared here so Movie mode is a pure-additive flip that never
   * re-touches this URL-literal module.
   */
  generateMovieEnqueueUrl:
    read("VITE_HUGPY_GENERATE_MOVIE_URL") ?? `${apiBase}/video/jobs/generate_movie`,
  /**
   * Curated "ideal default loads" for the Generate station — GET → { presets: [
   * { id, name, description, mode, model_key, defaults{...}, recommended } ] }. Each
   * preset prefills the generation knobs + model; the per-id apply POST
   * (presetApplyUrl) pre-warms that model on a GPU worker. Declared here so the
   * dropdown is a pure-additive flip that never re-touches this URL-literal module.
   */
  presetsUrl: read("VITE_HUGPY_PRESETS_URL") ?? `${apiBase}/video/presets`,
  /**
   * Curated MOVIE templates for the Generate-station Movie tab — GET → [{ id, name,
   * description, model_key, width, height, steps, guidance, fps, chain,
   * goals:[{start_frame,end_frame,prompt}], vision_enabled, score_threshold }]. Unlike
   * `presetsUrl` (a {presets:[…]} envelope of knob-only "loads"), a movie preset is a
   * BARE ARRAY and each item carries the WHOLE shot list — picking one drops a ready
   * goal timeline + settings into the Movie editor. The per-id apply POST
   * (moviePresetApplyUrl) echoes the same object. Declared here so the Movie-template
   * dropdown is a pure-additive flip that never re-touches this URL-literal module.
   */
  moviePresetsUrl: read("VITE_HUGPY_MOVIE_PRESETS_URL") ?? `${apiBase}/movie/presets`,
  /**
   * Enqueue a STUDIO image-to-video clip (Studio Clips viewer, slice #3) — POST a
   * {resolution:{width,height,fps}, seed, ...} body → {job_id}. The job runs through
   * the cinema-studio spine (router → manifest → runner → content-addressed clip)
   * and its mp4 is cataloged in the media store, playable via `studioClipUrl`.
   */
  studioI2VEnqueueUrl:
    read("VITE_HUGPY_STUDIO_I2V_URL") ?? `${apiBase}/video/studio/i2v`,
  /**
   * Enqueue a STUDIO MOVIE (Studio Movie composer, slice B2/B3) — POST an ordered
   * goal timeline {resolution:{width,height,fps}, seed, vram_budget_gb, negative_prompt,
   * goals:[{prompt, seed?, branch_frame?}]} → {job_id}. Each goal renders one studio
   * clip conjoined at splice points into one NLE row (segment clips + an assembled
   * movie.mp4 LAST); polled on the generic GET /video/jobs/<id> (jobStatusUrl). Same
   * idiom as studioI2VEnqueueUrl — declared here (the one URL-literal module) so the
   * composer imports it instead of deriving the path inline.
   */
  studioMovieUrl:
    read("VITE_HUGPY_STUDIO_MOVIE_URL") ?? `${apiBase}/video/studio/movie`,
  /**
   * STUDIO TESTER — POST {category, prompt, models?} to sweep one prompt across
   * EVERY servable model of the category's type, recording a model-battery
   * run-dir (one row per model). Operator-gated; runs as a background job and
   * returns {job_id, battery_run_dir}. Per-model results stream to the studio-
   * assist log. Same URL-literal idiom as studioI2VEnqueueUrl.
   */
  studioTesterUrl:
    read("VITE_HUGPY_STUDIO_TESTER_URL") ?? `${apiBase}/video/studio/tester`,
  /**
   * DURABLE recent studio clips for the Studio Clips list — GET → { clips: [{ job_id,
   * status, playable, created, updated, output:{asset_id,width,height,duration_s} }] }.
   * Sourced from the media CATALOG (not the comms /llm/jobs view, which drops terminal
   * rows after ~600s), so a clip produced an hour ago still lists. Each `playable` row's
   * `job_id` feeds `studioClipUrl(job_id)` to stream the clip. Declared here so the
   * Studio Clips station is a pure-additive registry flip that never re-touches this
   * module.
   */
  studioClipsUrl:
    read("VITE_HUGPY_STUDIO_CLIPS_URL") ?? `${apiBase}/video/studio/clips`,
  /**
   * CINEMA SESSIONS — every movie the studio-movies root holds, newest first: GET →
   * { movies: [{ movie_id, job_id, title, project, status:"partial"|"paused"|"done"|
   * "failed"|"running", segments_completed, segments_total, resumable, width, height,
   * fps, id_lock, updated, movie, segments:[{ index, segment_id, prompt, status,
   * clip_available, media, duration_s, frames, resumed, error:{code,message}|null }] }] }.
   *
   * A FILESYSTEM listing, not a bus query (k91): a movie is ONE job that can run for
   * hours, and when that job ends — cancelled, reaped, or lost with the tab — its
   * rendered segments stay on disk. This is the handle that outlives the job, which is
   * why the id is the movie DIR leaf (stable across a resume, which mints a new job id)
   * and never the job id. Each done segment carries a ready `/video/media?handle=` url,
   * so a session's clips are watchable AS THEY LAND. Paired with
   * `studioMoviePauseUrl` / `studioMovieResumeUrl` for the two actions on a session.
   */
  studioMoviesUrl:
    read("VITE_HUGPY_STUDIO_MOVIES_URL") ?? `${apiBase}/video/studio/movies`,
  /**
   * Bus-wide MEDIA JOBS listing for the console-wide "Active Processes" view — GET
   * (?all=1 appends recent terminal rows; ?limit=N, default 50) → { jobs: [{ job_id,
   * name, status, created, updated, principal, progress:{…|phase:"awaiting_capacity",
   * reason, held_since, overtaken}|null, placement?:{source:"reservation"|"template",
   * host, worker_id, gpu, process, reserved_bytes} }] }. Unlike `llmJobsUrl` (the
   * unified cross-transport feed, terminal rows dropped after ~600s) this is the
   * media CATALOG projection and carries PLACEMENT — WHERE each in-flight render
   * physically executes (ae · cuda:0 · P-studio). Declared here so the Active
   * Processes surface never derives the path inline.
   */
  mediaJobsUrl: read("VITE_HUGPY_MEDIA_JOBS_URL") ?? `${apiBase}/video/jobs`,
  /**
   * LIVE tok/s telemetry for the generate workbench's Toks panel. Two endpoints,
   * both same-origin `/api`, polled together every 2s by `useToks`:
   *   • `toksRecentUrl`  GET → { entries:[{ ts, worker_id, worker_name, model_key,
   *     tok_s, ttft_s, completion_tokens, config_key, ok }] } newest-first — the
   *     live tail + the per-worker strip's source (avg computed from this window).
   *   • `toksReportUrl`  GET → { groups:[{ worker_name, model_key, config_key, n,
   *     mean_tok_s, p50_tok_s, p95_tok_s }] } sorted best mean first — the
   *     best-config leaderboard. Declared here so the panel never derives the path.
   */
  toksRecentUrl: read("VITE_HUGPY_TOKS_RECENT_URL") ?? `${apiBase}/llm/toks/recent`,
  toksReportUrl: read("VITE_HUGPY_TOKS_REPORT_URL") ?? `${apiBase}/llm/toks/report`,
  /**
   * Curated STUDIO clip presets for the Studio Clips station — GET → { presets: [{
   * id, name, description, capability:"i2v"|"t2v", width, height, fps, vram_budget_gb,
   * seed, prompt, negative, recommended }] } (a {presets:[…]} envelope, like
   * `presetsUrl`/`moviePresetsUrl`). Unlike a video preset (which pins a model_key),
   * a studio preset pins a CAPABILITY + geometry + a routing `vram_budget_gb` and lets
   * the studio router pick the model — picking one prefills the generate affordance
   * (and drives the enqueue's capability/budget/prompt). The per-id apply POST
   * (studioPresetApplyUrl) echoes the same object as a POSTable /video/studio/i2v body.
   * Declared here so the Studio-preset dropdown is a pure-additive flip that never
   * re-touches this URL-literal module.
   */
  studioPresetsUrl:
    read("VITE_HUGPY_STUDIO_PRESETS_URL") ?? `${apiBase}/video/studio/presets`,
  /**
   * RENDER PRESETS — what this fleet can ACTUALLY render today. GET → { presets: [{
   * id, title, capability, capabilities[], model, precision, geometry, width, height,
   * fps, default_frames, max_frames, inputs[], proven, evidence, composes[], joints[],
   * vram_envelope_gb, vram_need_gib, fits_render_box }], unavailable: [{ capability,
   * reason, refusal }], menu, frame_cadence, render_box, render_box_vram_gib }.
   *
   * The FOURTH and only MEASURED preset surface: the other three (`presetsUrl`,
   * `moviePresetsUrl`, `studioPresetsUrl`) publish what we CURATED; this one publishes
   * the eight ratified rows measured on the live fleet, plus — the half that makes a
   * compatibility-aware picker possible — every capability NO preset covers, with the
   * measured blocker. Read by useRenderPresets.ts, which every studio surface consults
   * to decide what to OFFER. It is a courtesy layer: the server's own capability gate
   * on POST /video/studio/i2v stays authoritative.
   */
  renderPresetsUrl:
    read("VITE_HUGPY_RENDER_PRESETS_URL") ?? `${apiBase}/video/render/presets`,
  /**
   * Distinct known PROJECT names — GET → { projects: [name, …] } (read-only; derived
   * from the media-bus job store). Feeds the Settings "Project" combobox (choose an
   * existing name or free-type a new one) that threads `project` into the enqueue.
   * Empty = auto-named (the default): the field is optional scaffolding.
   */
  projectsUrl: read("VITE_HUGPY_PROJECTS_URL") ?? `${apiBase}/video/projects`,
  /**
   * IDENTITY PROFILES (studio stage (a)) — GET → { profiles: [{ slug, name,
   * reference_images[], created_at, notes }] }; POST { name, reference_images[1..4],
   * notes? } → { profile } (409 on a duplicate name). An identity profile is the
   * DURABLE form of "the reference set IS the identity": a named, curated reference
   * set saved ONCE and associated anywhere (single clips, movies, stills) via the
   * enqueue body's `identity_profile:<slug>`. Declared here so the identity picker /
   * save affordance is a pure-additive flip that never re-touches this URL-literal module.
   */
  identityProfilesUrl:
    read("VITE_HUGPY_IDENTITY_PROFILES_URL") ?? `${apiBase}/video/identity-profiles`,
  /**
   * CHARACTER-GROUPS COMMIT (dev/CHARACTER-GROUPS-PLAN.md slice S3) — POST the
   * user-CURATED output of the char360 "review" extraction:
   *   { groups: [ { name?: string, reference_images: string[] } ] }
   * where each `reference_images` entry is a media-handle (jailed absolute path,
   * a `view.url` from the S1 review manifest) the user KEPT in that group after
   * remove/move/merge edits, and only the groups the user marked "proceed" are
   * sent. Response 200 { results: [ { name, ok, slug?, error? } ] } — one result
   * per submitted group, positionally aligned to the request order, each minting
   * (or failing to mint) one identity profile. Declared here so the
   * character-groups panel is a pure-additive flip that never re-touches this
   * URL-literal module. (S3 builds this route to THIS exact body shape.)
   */
  identityProfilesFromGroupsUrl:
    read("VITE_HUGPY_IDENTITY_FROM_GROUPS_URL") ??
    `${apiBase}/video/identity-profiles/from-groups`,
  /**
   * LLM PROMPT ASSIST for the Generate-station composer — POST { mode:"detail"|
   * "generate", draft?, model?, context?:{kind?,hint?} } → 200 { prompt, model,
   * kind }. `detail` enriches the caller's `draft` (required, non-empty); `generate`
   * writes a full prompt from scratch (draft optional, used as a loose theme). Send
   * NO `model` (the backend picks the best resolvable chat model) and pass
   * `context.kind` = the active sub-mode (image|scene|movie) so video modes get
   * motion/camera phrasing. Errors: 400 (validation/unknown model) / 502 (no worker /
   * generation failed) carry a flat { error:"<string>" }. Declared here so the
   * composer's assist buttons are a pure-additive flip that never re-touch this
   * URL-literal module.
   */
  promptAssistUrl:
    read("VITE_HUGPY_PROMPT_ASSIST_URL") ?? `${apiBase}/video/prompt/assist`,
  /**
   * CINEMA PRODUCER (k120 slice 1): premise → full structured plan → per-segment
   * {prompt, negative, seconds, joint}. The composer's 🎬 Produce action POSTs
   * here and populates its segment rows from the reply.
   */
  producerPlanUrl:
    read("VITE_HUGPY_PRODUCER_PLAN_URL") ?? `${apiBase}/video/producer/plan`,
  /**
   * STUDIO-ASSIST LIVE LOG (operator directive, 2026-07-31) — a stored record per
   * prompt-generate attempt: the UNTRUNCATED model reply, what was stripped
   * (text/reasoning), whether the answer came FROM the reasoning, and the outcome
   * (served | empty | parse_error | worker_error | resolve_error). Lets the
   * operator self-diagnose "the assistant returned only reasoning" / "did not
   * return the spread JSON" without asking the keeper.
   *   GET  {logUrl}?limit&since&after_id → { events:[…], count, cursor } (page-load
   *        backfill; `raw` bounded for the load only).
   *   GET  {logStreamUrl}               → SSE, replay-then-live, full `raw`.
   * Same auth as every other /video call (the video gate) — no new credential.
   */
  promptAssistLogUrl:
    read("VITE_HUGPY_PROMPT_ASSIST_LOG_URL")
      ?? `${apiBase}/video/prompt/assist/log`,
  promptAssistLogStreamUrl:
    read("VITE_HUGPY_PROMPT_ASSIST_LOG_STREAM_URL")
      ?? `${apiBase}/video/prompt/assist/log/stream`,
  /**
   * Which TEXT GENERATORS the fleet can actually run — GET →
   * { models: [{ model, serving, framework, default }], default, excluded, ... }.
   * Feeds the Enhance/Generate picker. Deliberately NOT the full model registry:
   * the backend offers only rows a live worker holds, and drops catalog entries
   * that are mis-tagged text-generation (a Flux image LoRA is in there), so the
   * picker cannot offer a choice that 404s or renders nothing.
   */
  promptAssistModelsUrl:
    read("VITE_HUGPY_PROMPT_ASSIST_MODELS_URL")
      ?? `${apiBase}/video/prompt/assist/models`,
  /**
   * INTENT ROUTER (STUDIO-SPREAD-SPEC §1d) — POST { text, scope:"segment"|"movie" }
   * → 200 { intent:"empty"|"direction"|"scene_prompt"|"ambiguous", operation,
   * confidence, cached, degraded }. Decides whether what the user typed is a SCENE
   * (→ enhance) or a DIRECTION (→ generate from it), so the composer can pre-arm the
   * right button instead of making that an unassisted guess.
   *
   * ALWAYS 200 — that is the contract, not an accident: blank text short-circuits with
   * no model call, and ANY router failure degrades to `ambiguous` + `degraded:true` so
   * the UI shows BOTH actions. A classifier outage must never make a text box unusable,
   * and a classification must never run a generation on its own. Call it on field BLUR
   * or just before an action, never per keystroke (the backend caches by input hash,
   * which is a latency nicety, not a licence to spam it).
   */
  promptIntentUrl:
    read("VITE_HUGPY_PROMPT_INTENT_URL") ?? `${apiBase}/video/prompt/intent`,
  /**
   * Model registry — GET → { data: [{ id, task, tasks, hub_id, ... }] }. The Frames
   * and Generate stations filter this to the GENERATION image tasks
   * (text-to-image + image-to-image) off the FULL `tasks` capability list for the
   * model dropdown (see useModels.ts); the registry is revalidated in-session so
   * newly-adopted models appear without a reload.
   */
  modelsUrl: read("VITE_HUGPY_MODELS_URL") ?? `${apiBase}/v1/models`,
  /** Task defaults — GET → { defaults: { "text-to-image": <id>, ... }, tasks:[...] }. */
  promptTasksUrl: read("VITE_HUGPY_PROMPT_TASKS_URL") ?? `${apiBase}/prompt/tasks`,
  /**
   * The AUTHORITATIVE unified feed of EVERY in-flight worker call across ALL
   * transports (web / v1 / discord / cli / media) — GET → { jobs: [{ id, kind,
   * status, transport, model, model_name, worker, principal, elapsed, tokens,
   * error, ... }] }. Unlike the per-session jobTracker (which only knows jobs THIS
   * browser enqueued) this shows fleet-wide work — identity reconstructions
   * (kind:"identity_reconstruction", transport:"media"), studio renders, and any
   * call from another station/transport. The Active Processes panel polls it,
   * keeps the in-flight rows (pending/processing/streaming, minus `download`
   * provisioning), and merges them with the tracker rows. Declared here so the
   * panel never derives the path inline.
   */
  llmJobsUrl: read("VITE_HUGPY_LLM_JOBS_URL") ?? `${apiBase}/llm/jobs`,
  /**
   * k9 VIDEO-SHARE links — GET → { keys:[{id,label,prefix,created_at,expires_at,
   * expired,revoked,last_used}] } (OPERATOR-ONLY; a share/anon session gets 401,
   * which the Share button uses as its auth probe). POST { label?, ttl_days? } →
   * { key, url, id, ... } (the full key + link, shown ONCE). DELETE the per-id
   * revoke via `keysVideoShareRevokeUrl`. Operator-gated + off the /video surface
   * server-side, so a share principal can never reach it.
   */
  keysVideoShareUrl:
    read("VITE_HUGPY_KEYS_VIDEO_SHARE_URL") ?? `${apiBase}/keys/video-share`,
  /**
   * Default endpoint for the ComfyUI viewer station — the URL its <iframe> embeds
   * until the user overrides it (the override is kept per-tab in sessionStorage).
   * MUST be an HTTPS endpoint: the video UI is served over HTTPS, so an http:// or
   * localhost URL is silently blocked as mixed content. Set VITE_HUGPY_COMFY_URL to
   * a documented placeholder (e.g. https://comfy.hugpy.ai); NEVER hardcode a secret.
   * Empty by default — the station then shows its "enter an endpoint" empty state.
   */
  comfyUrl: read("VITE_HUGPY_COMFY_URL") ?? "",
  /**
   * Dedicated worker pool for this arm's jobs. Same semantics as the media arm:
   * empty string = general resolution (un-pooled); set VITE_HUGPY_POOL to route
   * to reserved workers once a pooled worker exists.
   */
  pool: env["VITE_HUGPY_POOL"] != null ? env["VITE_HUGPY_POOL"].trim() : "",
  /**
   * Send cookies on hugpy requests for session auth. DEFAULT OFF — see the
   * media arm's config for the CORS rationale. Opt in with
   * VITE_HUGPY_WITH_CREDENTIALS=true once hugpy's CORS is credential-ready.
   */
  withCredentials: readBool("VITE_HUGPY_WITH_CREDENTIALS", false),
  /**
   * Attach an X-Request-Id header for server-side correlation. DEFAULT OFF —
   * a custom header forces a CORS preflight the server may not answer. Opt in
   * with VITE_HUGPY_SEND_REQUEST_ID=true. The request id is still generated and
   * used for client-side telemetry regardless.
   */
  sendRequestId: readBool("VITE_HUGPY_SEND_REQUEST_ID", false),
} as const;

// Per-id / per-handle endpoints. Kept as helpers (not static strings) because the
// job id and media handle are runtime values — but they still derive from apiBase
// here, in the ONE module permitted to hold URL literals. Both callers must reach
// these through config, never by concatenating paths themselves.

/** Poll a single crop/extract job by id: GET → {job_id, status, result}. */
export function jobStatusUrl(id: string): string {
  return `${apiBase}/video/jobs/${encodeURIComponent(id)}`;
}

/** Revoke one video-share link by id: DELETE → {ok} (404 unknown id). Per-id like
 *  jobStatusUrl — derived from apiBase here, the one module permitted to hold URL
 *  literals. (List + create use `keysVideoShareUrl`.) */
export function keysVideoShareRevokeUrl(id: string): string {
  return `${apiBase}/keys/video-share/${encodeURIComponent(id)}`;
}

/**
 * The general VIDEO-JOBS status route, by job id: GET → { job_id, progress,
 * result }. Byte-identical to `jobStatusUrl` (same route) but named generically
 * for callers — like the char360 S4 "Extract from video" poll — that aren't
 * crop-specific. `result` is null/absent while running (`progress` 0..1 with a
 * stage/message); terminal is `result.ok===true` (success — no slug list, the
 * created/updated profile is the durable record) or `result.ok===false` with
 * `result.error.{code,message,retryable}` (honest error-as-data). Per-id like
 * `jobStatusUrl` — derived from apiBase here, the one module permitted to hold
 * URL literals.
 */
export function videoJobUrl(jobId: string): string {
  return `${apiBase}/video/jobs/${encodeURIComponent(jobId)}`;
}

/** Cooperative cancel: POST → {job_id, status, cancelled}. Queued jobs die
 * outright; a running scene stops between frames. */
export function jobCancelUrl(id: string): string {
  return `${apiBase}/video/jobs/${encodeURIComponent(id)}/cancel`;
}

/**
 * Cancel a NON-media-transport worker call by id: POST → cooperative stop. The
 * unified `/llm/jobs` feed carries calls from every transport; a `transport:"media"`
 * job cancels through `jobCancelUrl` (the video job bus), but everything else
 * (web / v1 / discord / cli chat + generation) cancels here. Per-id like
 * `jobCancelUrl` — derived from apiBase, the one module permitted to hold URL literals.
 */
export function llmChatCancelUrl(id: string): string {
  return `${apiBase}/llm/chat/cancel/${encodeURIComponent(id)}`;
}

/**
 * HARD-CANCEL a job by id on the unified feed: POST → {cancelled, status}. Unlike
 * the transport-specific routes above, this is the ONE cancel the fleet-wide
 * `/llm/jobs` Active-Processes rows hit regardless of which transport launched the
 * job — the backend fans it out to the right bus. Per-id like `jobCancelUrl` —
 * derived from apiBase, the one module permitted to hold URL literals.
 */
export function llmJobCancelUrl(id: string): string {
  return `${apiBase}/llm/jobs/${encodeURIComponent(id)}/cancel`;
}

/**
 * Apply (pre-warm) a curated Generate-station preset by id: POST → on success
 * {ok:true, worker:{name,id}, model_key, mode, defaults{...}, warming}; on failure
 * a {ok:false, error:{code,message}} with 404 (unknown preset/model) or 409 (no GPU
 * worker / won't fit). Per-id like jobStatusUrl — derived from apiBase here.
 */
export function presetApplyUrl(id: string): string {
  return `${apiBase}/video/presets/${encodeURIComponent(id)}/apply`;
}

/**
 * Apply a curated Movie template by id: POST → the same full preset object the list
 * endpoint returns (id/name/model_key/dims/knobs + the whole `goals` shot list +
 * director defaults). Per-id like presetApplyUrl — derived from apiBase here. The
 * Movie-template dropdown populates the editor from the ALREADY-fetched list object,
 * so this seam exists for parity / a future server-side pre-warm.
 */
export function moviePresetApplyUrl(id: string): string {
  return `${apiBase}/movie/presets/${encodeURIComponent(id)}/apply`;
}

/**
 * Raw media bytes for a MediaRef `uri` handle — usable directly as an <img> src.
 * The handle is opaque and may contain reserved characters, so it is always
 * URL-encoded into the query.
 *
 * Canned demo (?demo=1): fixture MediaRefs carry BUNDLED asset URLs as their
 * uri — <img>/<video> element loads bypass window.fetch, so the demo shim can
 * never answer them; the asset URL itself must be the src. Pass those through.
 */
export function mediaBytesUrl(handle: string): string {
  if (isCanned() && /^(\/|https?:|data:|blob:)/.test(handle) && !handle.startsWith("/mnt/")) {
    return handle;
  }
  // k9: element-src loads (this is used as <img>/<video> src) can't carry the
  // X-Video-Share header, so a share session rides the credential in the query.
  return withShareParam(`${apiBase}/video/media?handle=${encodeURIComponent(handle)}`);
}

/**
 * Stream a produced STUDIO clip by media-bus job id — usable directly as an HTML5
 * `<video>` src (Studio Clips viewer, slice #3). The route resolves the job's
 * content-addressed clip.mp4 server-side and streams it Range-aware (seek-able), so
 * the browser never handles a filesystem path. Per-id like `jobStatusUrl` — derived
 * from apiBase here, the one module permitted to hold URL literals.
 */
export function studioClipUrl(jobId: string): string {
  // k9: used directly as a <video> src (element load, no header) — a share
  // session rides its credential in the query, exactly like mediaBytesUrl.
  return withShareParam(`${apiBase}/video/studio/clip/${encodeURIComponent(jobId)}`);
}

/**
 * The exact CREATION PARAMETERS of a studio render, for a list-row expander: GET →
 * {job_id, status, spec, manifest|null, error|null}. `manifest` is the content-addressed
 * render manifest for a done clip (the TRUE params); `error` is {code,message,retryable}
 * for a failed/cancelled job. LAZY (fetched on expand, not on the list poll). Per-id like
 * `studioClipUrl` — derived from apiBase here, the one module permitted to hold URL
 * literals.
 */
export function studioClipDetailUrl(jobId: string): string {
  return `${apiBase}/video/studio/clip/${encodeURIComponent(jobId)}/detail`;
}

/**
 * ARCHIVE a produced studio clip by job id: POST → {ok, job_id, archived:true,
 * already, archived_at}. Fixes "removed clips just reappear" — GET
 * `studioClipsUrl` is DB-driven (not a filesystem walk), so this is the ONE way
 * to make a clip actually stop being listed; the clip's bytes are never touched
 * (never-delete doctrine — see media_bus.archive's docstring). 404 for an unknown
 * id; idempotent (a repeat archive answers `already:true`, never an error) — see
 * `unarchiveUrl` for the reversible half. Per-id like `studioClipUrl` — derived
 * from apiBase here, the one module permitted to hold URL literals.
 */
export function studioClipArchiveUrl(jobId: string): string {
  return `${apiBase}/video/studio/clip/${encodeURIComponent(jobId)}/archive`;
}

/**
 * The honest counterpart to `studioClipArchiveUrl`: POST → {ok, job_id,
 * archived:false, already}. Clears the archive mark so the clip rejoins the
 * list; the bytes were never moved, so there is nothing to restore on disk.
 * Per-id like `studioClipUrl` — derived from apiBase here, the one module
 * permitted to hold URL literals.
 */
export function studioClipUnarchiveUrl(jobId: string): string {
  return `${apiBase}/video/studio/clip/${encodeURIComponent(jobId)}/unarchive`;
}

/**
 * PAUSE a Cinema session by movie id (k91): POST → {ok, movie_id, job_id, status:
 * "paused", cancelled, job_status, segments_completed, segments_total}. Pause is the
 * cooperative cancel PLUS the manifest write that says "parked, not abandoned", so it
 * is IDEMPOTENT: a session whose job already ended still parks (`cancelled:false`
 * says there was nothing live to stop). 404 for an id that is not a session dir.
 * Per-id like `studioClipUrl` — the movie id is a DIR LEAF, so it is URL-encoded here
 * exactly like a job id, in the one module permitted to hold URL literals.
 */
export function studioMoviePauseUrl(movieId: string): string {
  return `${apiBase}/video/studio/movie/${encodeURIComponent(movieId)}/pause`;
}

/**
 * RESUME a Cinema session by movie id (k91): POST → {ok, movie_id, job_id, …} once
 * the persisted spec is back on the bus. There is no checkpoint format — the resumed
 * run re-renders the SAME spec and content addressing walks the already-rendered
 * prefix in seconds, so THE CLIPS ARE THE CHECKPOINT.
 *
 * Two honest refusals the caller must SHOW rather than swallow, both 409 {error}:
 * a movie that predates spec.json (nothing to re-enqueue — its segments stay listed
 * and playable), and one whose job is already in flight ("pause it first"). The
 * transport surfaces either body's `error` verbatim through `describeAppError`.
 * Per-id like `studioMoviePauseUrl`.
 */
export function studioMovieResumeUrl(movieId: string): string {
  return `${apiBase}/video/studio/movie/${encodeURIComponent(movieId)}/resume`;
}

/**
 * Studio "Send to Editor" (k12): POST → {ok:true, filename, path, mode}. Hands a
 * produced clip, in a Filmora-native MP4, into a stable inbox the operator's LAN
 * Windows workstation (Filmora desktop) picks up. CONSOLE-OPERATOR ONLY — a
 * video-share guest is refused 403 by the route even though it passes the /video
 * gate (the button is hidden for non-operators; see StudioViewer's probe). No
 * request body (the job id rides the URL, like archive/unarchive). `mode` is
 * "copy" (lossless remux) or "transcode". Per-id like `studioClipUrl` — derived
 * from apiBase here, the one module permitted to hold URL literals.
 */
export function studioClipToEditorUrl(jobId: string): string {
  return `${apiBase}/video/studio/clip/${encodeURIComponent(jobId)}/to-editor`;
}

/**
 * Apply a curated Studio preset by id: POST → the pure-prefill envelope
 * {ok:true, id, name, capability, request:{…a POSTable /video/studio/i2v body}} (a
 * 404 {ok:false,error} for an unknown id). Per-id like `studioClipUrl` — derived
 * from apiBase here, the one module permitted to hold URL literals. The Studio-preset
 * dropdown prefills from the ALREADY-fetched list object, so this seam exists for
 * parity / a future server-side pre-warm (mirrors moviePresetApplyUrl).
 */
export function studioPresetApplyUrl(id: string): string {
  return `${apiBase}/video/studio/presets/${encodeURIComponent(id)}/apply`;
}

/**
 * One identity profile by slug — GET → { profile } (404 unknown); DELETE → { ok,
 * archived, slug } (the store ARCHIVES under `_deleted`, never erases). Per-slug like
 * `studioClipUrl` — derived from apiBase here, the one module permitted to hold URL
 * literals. (List + create use `identityProfilesUrl`.)
 */
export function identityProfileUrl(slug: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}`;
}

/**
 * Enqueue a TURNAROUND RECONSTRUCTION ("character sheet") for one identity profile
 * — POST { prompt?, views?:string[] (default ["front","three_quarter","profile",
 * "back"]), seed? } → { job_id | job_ids, recon_id } (the enqueue may fan out to ONE
 * job or several — one per requested view; the caller flattens whichever it gets).
 * Each returned job is polled on the existing GET /video/jobs/<id> (`jobStatusUrl`);
 * on completion the profile's `reconstructions[]` carries the generated view paths
 * (served via `mediaBytesUrl`). Per-slug like `identityProfileUrl` — derived from
 * apiBase here, the one module permitted to hold URL literals.
 */
export function identityReconstructionUrl(slug: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/reconstruction`;
}

/**
 * PROMOTE reconstruction views to a profile's CANONICAL set — POST { recon_id,
 * views:number[] (indices into that reconstruction's `views`) } → { profile } (the
 * updated profile, now carrying the enlarged `canonical[]`). The durable "these are
 * the approved reference angles" step after a turnaround is generated. Per-slug like
 * `identityReconstructionUrl` — derived from apiBase here, the one module permitted
 * to hold URL literals.
 */
export function identityCanonicalUrl(slug: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/canonical`;
}

/**
 * ONE-CLICK FULL IDENTITY GENERATION — POST { views?, texture?, turntable?,
 * auto_promote? } (a bare {} is the intended happy-path call) → { job_id, recon_id }.
 * Turns a saved identity PROFILE into a complete 3D identity in a single action: a
 * Hunyuan3D mesh, a 360° turntable, and (by default) auto-promoted canonical angles —
 * no prior reconstruction needed. Poll the minted recon on `identityMeshStatusUrl`.
 * Per-slug like `identityReconstructionUrl` — derived from apiBase here, the one module
 * permitted to hold URL literals.
 */
export function identityGenerateUrl(slug: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/generate`;
}

/**
 * Mesh-build STATUS for one reconstruction — GET → the mesh state block
 * ({ status: "none"|"queued"|"running"|"done"|"error"|"cancelled", error?, glb_path?,
 * video_path?, auto_promoted? }). Polled after `identityGenerateUrl` (or the 5g mesh
 * build) until terminal. Per-id like `jobStatusUrl` — derived from apiBase here, the one
 * module permitted to hold URL literals.
 */
export function identityMeshStatusUrl(slug: string, reconId: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/reconstruction/${encodeURIComponent(reconId)}/mesh`;
}

/**
 * IDENTITY VERSIONS slice (design: dev/IDENTITY-VERSIONS-SLICE.md) — per-identity
 * generation settings: PATCH → a PARTIAL `gen_settings` body ({texture?, pose?,
 * frame_count?, fps?, width?, height?, auto_promote?, front_ref?, remove_background?,
 * vision_model?, cleanup_prompt?, negative_prompt?}) → { profile } (an omitted key is
 * left untouched server-side, same partial-PATCH idiom as `identityProfileUrl`).
 * Persists what the Settings panel edits and what a bare `/generate` click honors.
 * cleanup_prompt/negative_prompt are the CLEANUP-PROMPT slice's Advanced-panel "Avoid /
 * cleanup" + "Negative prompt" fields (C4). Per-slug like `identityProfileUrl` —
 * derived from apiBase here, the one module permitted to hold URL literals.
 */
export function identityProfileSettingsUrl(slug: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/settings`;
}

/**
 * One VERSION of an identity (the clay base + N named render-sets it accrues) by
 * id — PATCH { name?, notes? } (rename/annotate) and DELETE (archive; the base
 * version and the currently-ACTIVE version are refused with a 400 — never-delete
 * applies at the version layer too) both hang off this path. The response
 * envelope for these two isn't fixed by the wire contract (unlike the settings
 * PATCH above), so callers should treat it tolerantly. Per-slug+id like
 * `identityMeshStatusUrl` — derived from apiBase here, the one module permitted
 * to hold URL literals.
 */
export function identityVersionUrl(slug: string, versionId: string): string {
  return `${apiBase}/video/identity-profiles/${encodeURIComponent(slug)}/versions/${encodeURIComponent(versionId)}`;
}

/**
 * Make one version the identity's ACTIVE version: POST (no body) → the resolver
 * (`_reference_images_from_body`) then takes THIS version's canonical as the
 * id_lock DNA for future generations that don't name a specific
 * `identity_version`. Per-slug+id like `identityVersionUrl` — derived from
 * apiBase here, the one module permitted to hold URL literals.
 */
export function identityVersionActivateUrl(slug: string, versionId: string): string {
  return `${identityVersionUrl(slug, versionId)}/activate`;
}
