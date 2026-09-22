// The canned-demo transport: ONE window.fetch interceptor answering every hugpy
// call from fixtures.ts. The real station shell + transport run unchanged; only
// the wire is faked. Mirrors media_intelligence_ui/src/demo/demoFetch.ts, minus
// the live pass-through flavor (the video arm has no live demo — see mode.ts).
//
// Patching window.fetch (not a config hook) is deliberate: the arm's typed
// transport, entry.tsx's contract probe, and any raw fetch all call the global
// fetch, so a single wrapper covers every path. Fully restorable on uninstall.
// <img>/<video> src loads do NOT pass through fetch — fixtures point those at
// bundled asset URLs instead (see fixtures.ts + config.ts mediaBytesUrl).

import { hugpyConfig } from "../config";
import * as fx from "./fixtures";
import type { JobKind } from "./fixtures";
import { samplePromptFor, enhancePrompt } from "./promptAssist";
import { demoIdentityLuigi } from "./identityFixture";

// ── request introspection (same helpers as the media arm) ───────────────────
function pathOf(input: RequestInfo | URL): string {
  let url =
    typeof input === "string"
      ? input
      : input instanceof Request
        ? input.url
        : String((input as URL)?.toString?.() ?? input ?? "");
  if (/^https?:\/\//i.test(url)) {
    try {
      const u = new URL(url);
      url = u.pathname + u.search;
    } catch {
      /* keep raw */
    }
  }
  return url;
}

function methodOf(input: RequestInfo | URL, init?: RequestInit): string {
  return String(
    init?.method ??
      (input instanceof Request ? input.method : undefined) ??
      "GET",
  ).toUpperCase();
}

function apiPrefix(): string {
  const b = hugpyConfig.apiBase || "/api";
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, "");
    } catch {
      return "/api";
    }
  }
  return b.replace(/\/+$/, "");
}

function normalize(path: string): string {
  const pre = apiPrefix();
  if (pre && path.startsWith(pre)) return path.slice(pre.length) || "/";
  return path;
}

async function readBody(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<string> {
  if (init && typeof init.body === "string") return init.body;
  if (input instanceof Request) {
    try {
      return await input.clone().text();
    } catch {
      return "";
    }
  }
  return "";
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ── the router ───────────────────────────────────────────────────────────────
const JOB_ROUTES: Record<string, JobKind> = {
  "/video/jobs/crop": "crop",
  "/video/jobs/frame_extract": "frame_extract",
  "/video/jobs/audio_extract": "audio_extract",
  "/video/jobs/generate_image": "generate_image",
  "/video/jobs/generate_scene": "generate_scene",
  "/video/jobs/generate_movie": "generate_movie",
};

async function route(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response | null> {
  const full = pathOf(input);
  const [pathname, query = ""] = full.split("?");
  const path = normalize(pathname);
  const method = methodOf(input, init);

  // Contract probe (entry.tsx) — always the expected version.
  if (path === "/version") return jsonResponse({ api: 1 });

  // Session metering — pretend to meter, touch nothing.
  if (path.startsWith("/session/")) return jsonResponse({ ok: true });

  // Upload: accept anything, answer with a plausible server file handle. The
  // subsequent /video/ingest resolves it to the canned clip regardless.
  if (path === "/uploads" && method === "POST") {
    let name = "demo-clip.mp4";
    const body = init?.body;
    if (body instanceof FormData) {
      const f = body.get("file");
      if (f instanceof File && f.name) name = f.name;
    }
    return jsonResponse({ path: `/demo/uploads/${name}`, name, size: 55757 });
  }

  // Ingest: whatever was "uploaded", the demo's source is the canned clip.
  if (path === "/video/ingest" && method === "POST") {
    return jsonResponse(fx.demoClip);
  }

  // Job enqueues — remember the source kind so crop can answer image vs video.
  const kind = JOB_ROUTES[path];
  if (kind && method === "POST") {
    let sourceKind: string | null = null;
    let submitted = "";
    try {
      const body = JSON.parse((await readBody(input, init)) || "{}") as {
        source?: { kind?: string };
        parts?: Array<{ kind?: string; text?: string | null }>;
        goals?: Array<{ prompt?: string | null }>;
        prompt?: string;
      };
      sourceKind = body?.source?.kind ?? null;
      submitted = [
        ...(body?.parts ?? []).map((p) => (p?.kind === "text" ? p?.text ?? "" : "")),
        ...(body?.goals ?? []).map((g) => g?.prompt ?? ""),
        body?.prompt ?? "",
      ].join(" ");
    } catch {
      /* body optional */
    }
    return jsonResponse({ job_id: fx.enqueueJob(kind, sourceKind, submitted) });
  }

  // Job cancel (before poll — its path is a superset).
  const cancelM = path.match(/^\/video\/jobs\/([^/]+)\/cancel$/);
  if (cancelM && method === "POST") {
    return jsonResponse(fx.cancelJob(decodeURIComponent(cancelM[1])));
  }

  // Job poll.
  const poll = path.match(/^\/video\/jobs\/([^/]+)$/);
  if (poll && method === "GET") {
    return jsonResponse(fx.jobStatus(decodeURIComponent(poll[1])));
  }

  // Media bytes via fetch (waveform decode etc.): serve the bundled asset the
  // handle already IS — through the REAL fetch, never back into the shim.
  // Element src loads never reach here.
  if (path === "/video/media" && method === "GET") {
    const handle = new URLSearchParams(query).get("handle");
    if (handle && _realFetch) return _realFetch(handle);
    return jsonResponse({ error: "not found" }, 404);
  }

  // LLM PROMPT-ASSIST (Enhance / Generate) — the composers' ✨ buttons POST here. In
  // canned mode we answer CLIENT-SIDE from the fragment banks in promptAssist.ts (zero
  // backend contact): "generate" composes a fresh cinematic prompt (never the same one
  // twice in a row), "detail" enriches the submitted draft with appended descriptors.
  // The body shape { prompt, model, kind } matches what the stations parse
  // (readAssistPrompt reads `.prompt`); `kind` echoes the request's context.kind.
  if (path === "/video/prompt/assist" && method === "POST") {
    let mode = "generate";
    let draft = "";
    let kind = "image";
    try {
      const body = JSON.parse((await readBody(input, init)) || "{}") as {
        mode?: string;
        draft?: string;
        context?: { kind?: string };
      };
      if (body?.mode) mode = body.mode;
      if (typeof body?.draft === "string") draft = body.draft;
      if (body?.context?.kind) kind = body.context.kind;
    } catch {
      /* body optional — defaults stand */
    }
    const prompt = mode === "detail" ? enhancePrompt(draft) : samplePromptFor(kind);
    return jsonResponse({ prompt, model: "demo-showroom-llm", kind });
  }

  // Model registry + task defaults for the station dropdowns.
  if (path === "/v1/models") return jsonResponse(fx.V1_MODELS);
  if (path === "/prompt/tasks") return jsonResponse(fx.PROMPT_TASKS);

  // Unified in-flight worker feed (Active Processes poller, useWorkerCalls.ts,
  // ~2s cadence). The brochure runs nothing, so the honest envelope is an empty
  // in-flight list — { jobs: [] } is exactly the shape parseJobs() projects, and
  // an empty list keeps the panel truthful (no fabricated running render). Routed
  // (not gated) so the poll simply never reaches the wire in canned mode.
  if (path === "/llm/jobs" && method === "GET") {
    return jsonResponse({ jobs: [] });
  }

  // Bus-wide MEDIA JOBS listing (Active Processes media poller, useMediaJobs.ts,
  // ~2s cadence, GET bare /video/jobs — no id, so the job-poll regex below never
  // matches it). Same honest empty in-flight list as /llm/jobs. Must sit ABOVE
  // the /video/jobs/<id> poll route so the bare path resolves here first.
  if (path === "/video/jobs" && method === "GET") {
    return jsonResponse({ jobs: [] });
  }

  // Known PROJECT names (Settings project combobox, useProjects.ts, on mount) —
  // GET → { projects: [name,…] }. The brochure has produced nothing durable, so
  // an empty list is honest and passes projectsResponseSchema.
  if (path === "/video/projects" && method === "GET") {
    return jsonResponse({ projects: [] });
  }

  // Curated STUDIO clip presets (Studio Clips station, useStudioPresets.ts, on
  // mount) — GET → { presets: [...] }. Empty envelope passes
  // studioPresetsResponseSchema; the showroom's Studio curation is the movie/
  // video presets already routed above.
  if (path === "/video/studio/presets" && method === "GET") {
    return jsonResponse({ presets: [] });
  }

  // Durable recent STUDIO CLIPS (Studio Clips list) — GET → { clips: [...] }.
  // The brochure has no persisted renders; empty is honest.
  if (path === "/video/studio/clips" && method === "GET") {
    return jsonResponse({ clips: [] });
  }

  // CINEMA SESSIONS (the Cinema composer's sessions panel) — GET → { movies: [...] }.
  // Same honesty as the clips list above: the brochure has no movie dirs on disk, so
  // there is nothing to resume; an empty envelope passes the panel's schema and it
  // says so in words rather than erroring.
  if (path === "/video/studio/movies" && method === "GET") {
    return jsonResponse({ movies: [] });
  }

  // IDENTITY PROFILES (Studio identity picker, useIdentityProfiles.ts, on mount)
  // — GET → { profiles: [...] }. One REAL canned identity ("luigi": 12 refs →
  // 72-view turntable → textured GLB, assets on the hosted demo-media tree)
  // showcases the full pipeline; mutations still 401/501 honestly below.
  if (path === "/video/identity-profiles" && method === "GET") {
    return jsonResponse({ profiles: [demoIdentityLuigi] });
  }

  // k9 VIDEO-SHARE links (SharePanel probe on mount + mint/revoke). The public
  // brochure principal is NOT an operator, so the honest answer to every verb is
  // 401 — exactly what the real operator-gated route returns to a non-operator.
  // The transport folds a shim 401 into a clean `unauthorized` Result (no throw,
  // no browser console error, no wire hit), and SharePanel then hides the Share
  // affordance (authed=false), matching production behaviour for a share/anon
  // session. Covers the list GET, the mint POST, and the per-id revoke DELETE.
  if (path === "/keys/video-share" || path.startsWith("/keys/video-share/")) {
    return jsonResponse({ error: "operator authentication required" }, 401);
  }

  // STUDIO CLIP enqueue (Clip tab, POST /video/studio/i2v → {job_id}) and
  // CINEMA enqueue (StudioMovieComposer, POST /video/studio/movie → {job_id}).
  // Both poll the generic GET /video/jobs/<id> above, so the whole render
  // lifecycle is canned: queued → running (staged progress) → done with
  // bundled clips (cinema: segments + assembled movie LAST, the contract the
  // composer plays back).
  if ((path === "/video/studio/i2v" || path === "/video/studio/movie") && method === "POST") {
    let submitted = "";
    try {
      const body = JSON.parse((await readBody(input, init)) || "{}") as {
        prompt?: string;
        goals?: Array<{ prompt?: string | null }>;
      };
      submitted = [body?.prompt ?? "", ...(body?.goals ?? []).map((g) => g?.prompt ?? "")].join(" ");
    } catch { /* body optional */ }
    const k = path.endsWith("i2v") ? "studio_i2v" : "generate_studio_movie";
    return jsonResponse({ job_id: fx.enqueueJob(k as JobKind, null, submitted) });
  }
  // Cinema session pause/resume + clip archive/unarchive — acknowledge; the
  // canned session store is empty so there is nothing durable to mutate.
  const studioAct = path.match(
    /^\/video\/studio\/(?:movie\/[^/]+\/(pause|resume)|clip\/[^/]+\/(archive|unarchive))$/,
  );
  if (studioAct && method === "POST") {
    return jsonResponse({ ok: true });
  }
  // Studio tester sweep — operator-gated on the real backend; the brochure
  // principal is not an operator, so the honest answer is 401 (same shape as
  // the video-share keys above).
  if (path === "/video/studio/tester" && method === "POST") {
    return jsonResponse({ error: "operator authentication required" }, 401);
  }

  // Curated presets for the Generate-station dropdown + the per-id pre-warm.
  if (path === "/video/presets" && method === "GET") {
    return jsonResponse(fx.PRESETS);
  }
  const presetApply = path.match(/^\/video\/presets\/([^/]+)\/apply$/);
  if (presetApply && method === "POST") {
    const result = fx.applyPreset(decodeURIComponent(presetApply[1]));
    // Mirror the contract's status codes: 404 for an unknown preset, else 200.
    return jsonResponse(result, result.ok ? 200 : 404);
  }

  // Curated MOVIE TEMPLATES for the Movie-tab dropdown + the per-id apply. The list
  // returns a {presets:[…]} envelope (matches /video/presets + the real backend);
  // apply echoes the same full object (404 unknown).
  if (path === "/movie/presets" && method === "GET") {
    return jsonResponse({ presets: fx.MOVIE_PRESETS });
  }
  const moviePresetApply = path.match(/^\/movie\/presets\/([^/]+)\/apply$/);
  if (moviePresetApply && method === "POST") {
    const found = fx.applyMoviePreset(decodeURIComponent(moviePresetApply[1]));
    return jsonResponse(found ?? { error: { code: "unknown_preset" } }, found ? 200 : 404);
  }

  // Assist LOG surfaces (audit 2026-08-13b): models dropdown + log fetch.
  // The SSE stream itself is handled by the FakeEventSource below (EventSource
  // BYPASSES window.fetch — unshimmed it hit the demo host's nginx 404 and
  // auto-reconnected forever: the reported "gets stuck" loop).
  if (path === "/video/prompt/assist/models" && method === "GET") {
    return jsonResponse({ models: [{ id: "demo-showroom-llm", label: "demo-showroom-llm" }] });
  }
  if (path === "/video/prompt/assist/log" && method === "GET") {
    return jsonResponse({ lines: [] });
  }
  if (path === "/video/prompt/intent" && method === "POST") {
    return jsonResponse({ ok: false }, 200);
  }
  if (path === "/video/render/presets" && method === "GET") {
    return jsonResponse({ presets: [] });
  }
  const chatCancel = path.match(/^\/llm\/chat\/cancel\/[^/]+$/);
  if (chatCancel && method === "POST") {
    return jsonResponse({ ok: true });
  }
  // Identity flows are operator/member-gated on the real backend — honest 401
  // (the SharePanel/keys pattern) keeps those affordances hidden in the demo.
  if (path.startsWith("/video/identity-profiles/") && method === "POST") {
    return jsonResponse({ error: "operator authentication required" }, 401);
  }
  // Single cinema-session probe (resume path) — the showroom has no durable
  // movie dirs, so an unknown id is the honest answer.
  const movieOne = path.match(/^\/video\/studio\/movie\/[^/]+$/);
  if (movieOne && method === "GET") {
    return jsonResponse({ ok: false, error: "unknown movie" }, 404);
  }

  // BACKSTOP (2026-08-13): anything addressed to the API base that no route
  // above canned must FAIL FAST AND HONESTLY — never fall through to the real
  // fetch. On the demo host nginx hard-404s /api, so a fall-through shows a
  // spinner that dies slowly ("renders a while, then fails" — the exact
  // reported symptom class). A crisp canned 501 lets every surface fail-soft
  // immediately instead. Bundled assets / HMR never carry the API prefix, so
  // the pass-through below still serves them.
  if (pathname !== path || path.startsWith("/video/") || path.startsWith("/v1/")) {
    return jsonResponse(
      { error: "showroom: this surface is not canned in the demo" },
      501,
    );
  }

  return null; // not a hugpy call — pass through (bundled assets, HMR, …)
}

// ── install / uninstall ──────────────────────────────────────────────────────
let _realFetch: typeof window.fetch | null = null;
let _realES: typeof window.EventSource | null = null;

// EventSource does NOT route through window.fetch, so the canned transport
// must shim it separately: a silent, open stream for the assist log (no
// events, no reconnect churn), pass-through for anything else.
class FakeEventSource {
  url: string;
  readyState = 0;
  onopen: ((e: Event) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  constructor(url: string | URL) {
    this.url = String(url);
    setTimeout(() => {
      this.readyState = 1;
      this.onopen?.(new Event("open"));
    }, 50);
  }
  addEventListener(type: string, fn: (e: Event) => void): void {
    if (type === "open") this.onopen = fn as (e: Event) => void;
  }
  removeEventListener(): void {}
  close(): void { this.readyState = 2; }
}

export function installDemoTransport(): void {
  if (_realFetch) return; // already installed
  _realFetch = window.fetch.bind(window);
  _realES = window.EventSource;
  window.EventSource = function (url: string | URL, init?: EventSourceInit) {
    if (String(url).includes("/prompt/assist/log/stream")) {
      return new FakeEventSource(url) as unknown as EventSource;
    }
    return new _realES!(url, init);
  } as unknown as typeof window.EventSource;
  window.fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    try {
      const canned = await route(input, init);
      if (canned) return canned;
    } catch {
      /* fall through to the real fetch on any shim error */
    }
    return _realFetch!(input, init);
  };
}

export function uninstallDemoTransport(): void {
  if (!_realFetch) return;
  window.fetch = _realFetch;
  _realFetch = null;
  if (_realES) {
    window.EventSource = _realES;
    _realES = null;
  }
}
