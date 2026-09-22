// Demo-mode configuration for the video-intelligence arm.
//
// Mirrors media_intelligence_ui/src/demo/mode.ts with ONE deliberate cut: there
// is NO live mode here. The video arm's jobs mutate real worker GPUs (ffmpeg,
// diffusion), so the only demo flavor is the self-contained showroom:
//
//   ?demo=1   → CANNED: a window.fetch shim answers every hugpy call from canned
//               fixtures, and the media library is seeded with a worked example.
//               Works anywhere — the public brochure, a static host. No backend
//               is ever contacted.
//
//   &embed=1  → embedded inside the dev UI showroom (an <iframe>). The host
//               provides the demo banner, so the arm suppresses its own.
//
// `?demo=live` is intentionally treated as OFF — a shared demo link must never
// drive real video jobs.
//
// Host default: on the dedicated demo host (demo.hugpy.ai) a BARE hit — no
// `demo` param — defaults to CANNED, so any build served there is demo-first
// without needing host-side redirect rules. An explicit `?demo=...` still wins
// (and the live-is-off rule above still applies).
//
// Resolved once at module load; query params persist across in-SPA navigation,
// so the mode sticks for the session.

export type DemoMode = "off" | "canned";

export interface DemoConfig {
  mode: DemoMode;
  embed: boolean;
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
  // NOTE: "live" is NOT accepted — canned or nothing (see module comment).
  let mode: DemoMode = /^(1|true|yes|on|canned|demo)$/.test(raw)
    ? "canned"
    : "off";
  // Host default: bare hit on demo.hugpy.ai (no explicit `demo` param) → canned.
  // An explicit param always wins — including "live", which stays off here.
  if (mode === "off" && !params.has("demo") && isDemoHost()) mode = "canned";

  return { mode, embed: truthy(params.get("embed")) };
}

let _config: DemoConfig | null = null;

export function getDemoConfig(): DemoConfig {
  if (!_config) _config = parse();
  return _config;
}

export const isDemo = (): boolean => getDemoConfig().mode !== "off";
export const isCanned = (): boolean => getDemoConfig().mode === "canned";
export const isEmbedded = (): boolean => getDemoConfig().embed;
