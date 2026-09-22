// Per-tab session identity + connection metering for per-session server uploads.
//
// The browser owns a random session id in sessionStorage (so it dies with the
// tab/session, matching the per-session client storage). Uploads carry the sid
// as a form field; while the page is open we heartbeat /session/ping so the
// server keeps that session's uploads alive; on tab close we beacon /session/end
// to wipe them immediately. A server-side 1h sweep is the safety net for missed
// beacons. Disabled entirely in canned demo mode (no backend is contacted).
import { hugpyConfig } from "./config";
import { isCanned } from "./demo/mode";

const SID_KEY = "hugpy.media.sid";
const HEARTBEAT_MS = 60_000;

function makeSid(): string {
  try {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    if (typeof crypto !== "undefined" && crypto.getRandomValues) {
      const a = new Uint8Array(16);
      crypto.getRandomValues(a);
      return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
    }
  } catch {
    /* fall through to the non-crypto fallback */
  }
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`;
}

/** Stable per-tab session id (sessionStorage). Created on first use. */
export function getSessionId(): string {
  try {
    let sid = sessionStorage.getItem(SID_KEY);
    if (!sid) {
      sid = makeSid();
      sessionStorage.setItem(SID_KEY, sid);
    }
    return sid;
  } catch {
    // storage unavailable — a stable-enough per-load id still tags the upload.
    return makeSid();
  }
}

let started = false;

/** Start heartbeat + tab-close beacon so the server can wipe uploads per session. */
export function startSessionMetering(): void {
  if (started || isCanned() || typeof window === "undefined") return;
  started = true;

  const sid = getSessionId();
  const base = hugpyConfig.apiBase.replace(/\/$/, "");
  const q = `sid=${encodeURIComponent(sid)}`;

  // Heartbeat — a "simple" POST (query param, no custom header / JSON) so it never
  // triggers a CORS preflight in cross-origin live mode. Failures are ignored.
  const ping = () => {
    try {
      void fetch(`${base}/session/ping?${q}`, {
        method: "POST",
        credentials: hugpyConfig.withCredentials ? "include" : "same-origin",
        keepalive: true,
      }).catch(() => {});
    } catch {
      /* ignore */
    }
  };

  ping(); // announce on load
  window.setInterval(ping, HEARTBEAT_MS);

  // Tab close → wipe now (best-effort). Skip bfcache navigations (persisted),
  // which may be restored; the heartbeat resumes if so, and the 1h sweep is the
  // backstop either way.
  window.addEventListener("pagehide", (e) => {
    if ((e as PageTransitionEvent).persisted) return;
    try {
      navigator.sendBeacon?.(`${base}/session/end?${q}`);
    } catch {
      /* ignore */
    }
  });
}
