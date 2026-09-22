// Demo-mode configuration for the media-intelligence arm.
//
// One URL-driven switch turns the SAME polished front end into either:
//
//   ?demo=1            → CANNED: a self-contained showroom (no backend). A
//                        window.fetch shim answers every hugpy call from canned
//                        fixtures and the thread is seeded with a worked example.
//                        Works anywhere — the public brochure, a static host.
//
//   ?demo=live&key=K   → LIVE: the same demo chrome, but powered by a REAL hugpy
//                        backend. No fixtures; requests pass through, with
//                        `Authorization: Bearer K` injected so a gated
//                        media-intelligence backend (see /api/ml/gate) accepts
//                        them. This is the "pitch demo" — share a link that opens
//                        the live front end against dev.hugpy.ai.
//
//   &embed=1           → embedded inside the dev UI showroom (an <iframe>). The
//                        host provides the demo banner, so the arm suppresses its
//                        own — but keeps the nav bar for parity with the console
//                        surface.
//
//   &api=<base>        → (live, optional) point at a cross-origin backend, e.g.
//                        https://dev.hugpy.ai/api. Default: same-origin /api.
//
// Host default: on the dedicated demo host (demo.hugpy.ai) a BARE hit — no
// `demo` param — defaults to CANNED, so any build served there is demo-first
// without needing host-side redirect rules. An explicit `?demo=...` still wins
// (including `?demo=live&key=K` and explicit off values).
//
// Resolved once at module load; query params persist across the arm's in-SPA
// navigation, so the mode sticks for the session.

export type DemoMode = "off" | "canned" | "live";

export interface DemoConfig {
  mode: DemoMode;
  embed: boolean;
  /** Live: API key injected as `Authorization: Bearer <key>` (null = none). */
  key: string | null;
  /** Live: cross-origin backend base override (null = same-origin /api). */
  apiBase: string | null;
}

function truthy(v: string | null): boolean {
  return v != null && /^(1|true|yes|on)$/i.test(v.trim());
}

// The dedicated demo host — a bare hit here defaults to CANNED (see below).
// Narrow on purpose: only the exact `demo.hugpy.ai` label, not any *.hugpy.ai.
function isDemoHost(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      /^demo\.hugpy\.ai$/i.test(window.location.hostname)
    );
  } catch {
    return false;
  }
}

function parse(): DemoConfig {
  let params: URLSearchParams;
  try {
    params = new URLSearchParams(
      typeof window !== "undefined" ? window.location.search : "",
    );
  } catch {
    params = new URLSearchParams();
  }

  const raw = (params.get("demo") || "").trim().toLowerCase();
  const hasParam = params.has("demo");
  let mode: DemoMode = "off";
  if (raw === "live") mode = "live";
  else if (/^(1|true|yes|on|canned|demo)$/.test(raw)) mode = "canned";
  // Host default: bare hit on demo.hugpy.ai (no explicit `demo` param) → canned.
  // An explicit param always wins — including an explicit off value.
  else if (!hasParam && isDemoHost()) mode = "canned";

  const key = (params.get("key") || "").trim();
  const api = (params.get("api") || "").trim();

  return {
    mode,
    embed: truthy(params.get("embed")),
    key: key || null,
    apiBase: api ? api.replace(/\/+$/, "") : null,
  };
}

let _config: DemoConfig | null = null;

export function getDemoConfig(): DemoConfig {
  if (!_config) _config = parse();
  return _config;
}

export const isDemo = (): boolean => getDemoConfig().mode !== "off";
export const isCanned = (): boolean => getDemoConfig().mode === "canned";
export const isLive = (): boolean => getDemoConfig().mode === "live";
export const isEmbedded = (): boolean => getDemoConfig().embed;

/** Build a /media demo URL for a given mode, preserving the current key/api. */
export function demoUrl(mode: DemoMode): string {
  const cfg = getDemoConfig();
  const base =
    typeof window !== "undefined"
      ? window.location.origin + window.location.pathname
      : "/media/";
  const p = new URLSearchParams();
  if (mode !== "off") p.set("demo", mode === "canned" ? "1" : "live");
  if (cfg.embed) p.set("embed", "1");
  if (mode === "live" && cfg.key) p.set("key", cfg.key);
  if (mode === "live" && cfg.apiBase) p.set("api", cfg.apiBase);
  const qs = p.toString();
  return qs ? `${base}?${qs}` : base;
}
