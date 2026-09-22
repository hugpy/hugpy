// Standalone bootstrap for the video-intelligence arm. The dev UI links to
// /video; this mounts the station shell there. A BrowserRouter (basename from
// the Vite base) is provided because stations are routed tabs.
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { StationShell } from "./src/stations/StationShell";
import { hugpyConfig } from "./src/config";
import { startSessionMetering } from "./src/session";
import { initShare } from "./src/share";
import { installVideoDemo } from "./src/demo";
// Sitewide floating Help button. ONE shared framework-free DOM module mounted
// identically by all four surfaces (main SPA + media/video/fleet arms) — see
// ../ui_shared/help/helpWidget.js for why it isn't a copied React component.
import { mountHelpWidget } from "../ui_shared/help/helpWidget";
import "./src/style/app.css";

// k9: lift a `?share=<key>` share-link credential out of the URL (stash + strip)
// BEFORE any API call, so the very first fetch already carries it. No-op for a
// normal console session (no key present).
initShare();

// Canned demo (?demo=1): install the fetch shim + seed the library BEFORE
// anything below can touch the network. No-op outside demo mode.
installVideoDemo();

// Per-session upload lifecycle: heartbeat the server so it keeps this session's
// uploads, and beacon a wipe on tab close. (No-op in canned demo mode.)
startSessionMetering();

// Contract guard: this arm depends ENTIRELY on the current hugpy API, whose
// /version reports an integer `api` contract version. Fire-and-forget on boot —
// never block or delay rendering, and swallow any fetch/parse error.
const EXPECTED_API_CONTRACT = 1;
void (async () => {
  try {
    const res = await fetch(`${hugpyConfig.apiBase}/version`, {
      credentials: hugpyConfig.withCredentials ? "include" : "same-origin",
    });
    if (!res.ok) return;
    const body = (await res.json()) as { api?: unknown };
    const api = body?.api;
    if (api !== EXPECTED_API_CONTRACT) {
      console.warn(
        `hugpy API contract mismatch (got ${String(api)}, expected ${EXPECTED_API_CONTRACT})`,
      );
    }
  } catch {
    // swallow: never let the contract check disrupt boot.
  }
})();

const el = document.getElementById("root");
if (!el) throw new Error("video-intelligence: #root not found");

// Router basename must match where the bundle is SERVED, which equals the Vite
// build base: "/video/" for the in-hugpy build, "/" for a standalone deploy.
// A mismatched basename makes React-Router render nothing → blank screen.
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

// Mounted outside the React tree (its own fixed-position node on <body>), so it
// is unaffected by station routing and by the arm's Tailwind layout.
mountHelpWidget({ apiBase: hugpyConfig.apiBase, surface: "studio" });

createRoot(el).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      {/* The DemoBanner now lives INSIDE StationShell, directly beneath the nav
          (operator directive 2026-07-21: nav + demo tag = one fixed header
          stack), rather than floating above the whole shell. */}
      <StationShell />
    </BrowserRouter>
  </React.StrictMode>,
);
