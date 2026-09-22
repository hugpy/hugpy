// Standalone bootstrap for the media-intelligence arm. The dev UI links to
// /media; this mounts the console there. A BrowserRouter (basename "/media") is
// provided because the console's sub-tools use react-router (Link / routes).
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
// Chat-first surface: the media-intelligence console IS the chat shell now
// (the page/form console is retired by routing; its registry + execution are
// reused under the hood). Import HugpyChat + the shared Navbar from the barrel.
import { HugpyChat, Navbar } from "./src/chat";
import { hugpyConfig } from "./src/config";
// Demo layer: ?demo=1 (canned showroom) / ?demo=live&key=… (live pitch). The
// transport shim MUST install before the contract guard below fires, so even
// /version is answered from fixtures in canned mode.
import { installDemoTransport, DemoBanner } from "./src/demo";
import { startSessionMetering } from "./src/session";
// Sitewide floating Help button. ONE shared framework-free DOM module mounted
// identically by all four surfaces (main SPA + media/video/fleet arms) — see
// ../ui_shared/help/helpWidget.js for why it isn't a copied React component.
import { mountHelpWidget } from "../ui_shared/help/helpWidget";

installDemoTransport();

// Per-session upload lifecycle: heartbeat the server so it keeps this session's
// uploads, and beacon a wipe on tab close. No-op in canned demo mode (no backend).
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
if (!el) throw new Error("media-intelligence: #root not found");

// Router basename must match where the bundle is SERVED, which equals the Vite
// build base: "/media/" for the in-hugpy build (served at /media), "/" for a
// standalone deploy (e.g. an HF Space served at root). A basename that doesn't
// match the URL makes React-Router render nothing → blank screen. Strip the
// trailing slash; empty means root.
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

// Mounted outside the React tree (its own fixed-position node on <body>), so it
// is unaffected by route changes and by the arm's Tailwind layout.
mountHelpWidget({ apiBase: hugpyConfig.apiBase, surface: "media" });

createRoot(el).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      {/* The arm shell: the shared nav, the demo banner (demo modes only), then
          the chat. .hugpy-arm carries a dark token context + column layout so
          the chrome above the chat scope reads consistently and the chat fills
          the remaining height. */}
      <div className="hugpy-arm">
        {/* Nav + demo banner = ONE sticky header stack, pinned to the top of the
            viewport as a single unit (operator directive 2026-07-21). Wrapping
            both in .hugpy-header-stack (the sticky element) keeps the banner
            directly beneath the nav — neither slides under the other on scroll. */}
        <header className="hugpy-header-stack">
          <Navbar />
          <DemoBanner />
        </header>
        <HugpyChat />
      </div>
    </BrowserRouter>
  </React.StrictMode>,
);
