// Demo media base — where the canned-demo sample tree is served from.
//
// The demo media (~36M of mp4/png samples) is NOT bundled; it lives at a
// hosted base (default https://hugpy.ai/demo-media, see config.ts — the ONE
// module permitted to hold URL literals). Resolved ONCE at module load, same
// as demo/mode.ts. Precedence:
//   1. ?mediaBase= query param (http(s) or path-absolute only)
//   2. window.__HUGPY_MEDIA_BASE__ (non-empty string — servers/self-hosters
//      set this by rewriting the index.html marker; see index.html)
//   3. VITE_HUGPY_DEMO_MEDIA_BASE (build-time, read in config.ts)
//   4. the config.ts default.

import { hugpyConfig } from "../config";

declare global {
  interface Window {
    __HUGPY_MEDIA_BASE__?: string | null;
  }
}

function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Accept only http(s) URLs or path-absolute ("/...") bases. */
function acceptable(raw: string): boolean {
  return /^(https?:\/\/|\/)/i.test(raw);
}

function resolve(): string {
  // 1. ?mediaBase= — same query-param precedent as demo/mode.ts (?demo=).
  try {
    const params = new URLSearchParams(
      typeof window !== "undefined" ? window.location.search : "",
    );
    const raw = (params.get("mediaBase") || "").trim();
    if (raw && acceptable(raw)) return stripTrailingSlash(raw);
  } catch {
    /* fall through */
  }

  // 2. window.__HUGPY_MEDIA_BASE__ — server-injected (index.html marker).
  try {
    const injected =
      typeof window !== "undefined" ? window.__HUGPY_MEDIA_BASE__ : null;
    if (typeof injected === "string" && injected.trim()) {
      return stripTrailingSlash(injected.trim());
    }
  } catch {
    /* fall through */
  }

  // 3./4. Build-time env override or the config default (both in config.ts).
  return hugpyConfig.demoMediaBase;
}

const base = resolve();

/** Absolute URL for a demo media file, e.g. demoMedia("generate2/s0.png"). */
export function demoMedia(relPath: string): string {
  return `${base}/${relPath}`;
}
