// Single source of truth for hugpy service URLs (Phase 3).
//
// Env wiring: one hugpy-specific base var following the app's existing VITE_*
// convention, DEFAULTING to the production host so existing deploys keep working with
// no .env entry. Upload + registry URLs derive from the base; each has an optional
// override for the rare case a path diverges from the base host.
//
// This is the ONE module permitted to hold URL literals — eslint's no-restricted-syntax
// rule exempts **/config.ts. Every other module must read URLs from here.
//
// NOTE: intentionally NOT reusing VITE_SECURE_FILES_API_BASE. That var resolves to
// api.abstractendeavors.com/secure-files — a different service. Reusing it would
// misroute every hugpy request to the wrong host.

// Built as a standalone "arm" served at /media by the hugpy dev UI's origin:
// default to the SAME-ORIGIN current-hugpy API (`/api` → Flask backend), NOT the
// legacy pre-hugpy.ai host. Override with VITE_HUGPY_API_BASE at build time.
import { getDemoConfig } from "./demo/mode";

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

// In the LIVE demo (?demo=live) the app runs on the public brochure (hugpy.ai),
// which serves NO /api — a same-origin "/api" call 405s and the response is the
// HTML SPA shell (hence "JSON.parse: unexpected character" on model/chat fetches).
// So default live mode to the real hugpy backend at dev.hugpy.ai; an explicit
// ?demo=live&api=<base> still wins. CANNED mode (?demo=1) keeps /api — the
// demoFetch shim intercepts those calls, so no real backend is hit.
const DEFAULT_LIVE_API_BASE = "https://dev.hugpy.ai/api";
function liveApiBase(): string | undefined {
  const d = getDemoConfig();
  if (d.mode !== "live") return undefined;
  return d.apiBase || DEFAULT_LIVE_API_BASE;
}

const apiBase = stripTrailingSlash(
  read("VITE_HUGPY_API_BASE") ?? liveApiBase() ?? DEFAULT_API_BASE,
);

export const hugpyConfig = {
  /** Base host for all hugpy API calls. */
  apiBase,
  /**
   * Public hugpy site — the all-in-one demo / info / live platform. Used as the
   * welcome-screen CTA target so visitors can see the real thing and host their
   * own. DEFAULTS to PROD (independent of apiBase, which may point at dev for a
   * showcase deploy). Override with VITE_HUGPY_SITE_URL.
   */
  siteUrl: stripTrailingSlash(read("VITE_HUGPY_SITE_URL") ?? "https://hugpy.ai"),
  /** Multipart file upload endpoint. */
  uploadUrl: read("VITE_HUGPY_UPLOAD_URL") ?? `${apiBase}/uploads`,
  /** Server-driven task catalog (current hugpy: GET /prompt/tasks -> {tasks, defaults}). */
  registryUrl: read("VITE_HUGPY_REGISTRY_URL") ?? `${apiBase}/prompt/tasks`,
  /**
   * The media_intelligence HTTP bridge (POST {url} -> DocumentIntelligence): the
   * package's MediaPipeline served by hugpy at /api/media/analyze (wsgi_app mounts it).
   * The chat sends an attachment here to get one structured record instead of
   * orchestrating /ml/* itself. Defaults ON to the same-origin bridge; if it isn't
   * mounted the call fails fast and the chat falls back to client-side /ml/*
   * orchestration (graceful, no user-visible error). Point it elsewhere with
   * VITE_HUGPY_MEDIA_ANALYZE_URL (e.g. a separate bridge host).
   */
  mediaAnalyzeUrl: read("VITE_HUGPY_MEDIA_ANALYZE_URL") ?? `${apiBase}/media/analyze`,
  /**
   * Dedicated worker pool for the media arm. ALL of the arm's inference — the
   * /ml task models AND the chat narrator/router (the coder-gguf) — is tagged
   * with this pool so it routes to reserved workers and never competes with
   * general traffic. Must match the server's HUGPY_ML_POOL (default "ml") so the
   * whole arm shares ONE pool. On a single-box (no pool worker) it falls back to
   * LOCAL — harmless — and becomes truly dedicated once a pooled worker exists.
   * Empty string opts out (general resolution). Override with VITE_HUGPY_POOL.
   *
   * NOTE: read() collapses "" to undefined, which made the documented empty
   * opt-out impossible — the "ml" default always won, so on fleets with NO
   * ml-pool worker EVERY media request silently fell back to central even
   * when the model was assigned+loaded on a general worker. Honor an explicit
   * empty value as a real opt-out.
   */
  // Default UN-pooled (opt-in reservation): the old "ml" default meant any
  // deployment without an ml-tagged worker had every media request silently
  // fall back to central. Set VITE_HUGPY_POOL=ml to opt back in.
  pool: env["VITE_HUGPY_POOL"] != null ? env["VITE_HUGPY_POOL"].trim() : "",
  /**
   * Send cookies on hugpy requests for session auth. DEFAULT OFF: hugpy is cross-origin,
   * and credentialed requests require the server to return
   * Access-Control-Allow-Credentials: true + a specific origin (not `*`). Turning this on
   * before the server supports it makes the browser block the response. Opt in with
   * VITE_HUGPY_WITH_CREDENTIALS=true once hugpy's CORS is credential-ready.
   */
  withCredentials: readBool("VITE_HUGPY_WITH_CREDENTIALS", false),
  /**
   * Attach an X-Request-Id header for server-side correlation. DEFAULT OFF: a custom
   * header turns an otherwise-"simple" cross-origin request (e.g. the multipart upload)
   * into one that needs a CORS preflight the server may not answer. Opt in with
   * VITE_HUGPY_SEND_REQUEST_ID=true once hugpy allows the header. The request id is still
   * generated and used for client-side telemetry regardless.
   */
  sendRequestId: readBool("VITE_HUGPY_SEND_REQUEST_ID", false),
} as const;
