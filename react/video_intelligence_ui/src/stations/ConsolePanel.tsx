// ConsolePanel — the hugpy console (dev/ui's `/console`) embedded in the shared
// expand-into-the-screen drawer, so the operator never has to keep a second
// browser tab open just to watch it. Reachable from every station (the drawer is
// mounted once by WorkbenchSidebar, the shared sidebar), without duplicating the
// affordance per-station.
//
// Since 2026-08-06 this file no longer renders a drawer of its own: it
// contributes ONE TAB to the single unified drawer (drawerShell.tsx →
// WorkbenchDrawer.tsx), alongside ComfyUI and Active. The shell owns the window
// (portal-to-<body>, half/full sizing, Esc-to-close, persisted open/size/tab);
// this file owns only the console-specific part, i.e. the iframe below and its
// security posture.
//
// ── SAME-ORIGIN, NOT SANDBOXED (read before touching the iframe) ────────────
// The console SPA (dev/ui App.jsx) is a SIBLING route of THIS one, served from
// the SAME origin on dev.hugpy.ai (and on a local `hugpy serve`, per the
// product's self-hosted-first posture) — never a foreign site. Reusing
// nav/Navbar's own `siteBase`/`onHugpyOrigin` rule (the SAME computation the
// Navbar's plain `<a href="/console">` link already uses) means the href is
// "/console" (origin-relative → guaranteed same-origin) on every real hugpy
// deployment; it only falls through to the full `siteUrl` off a hugpy hostname
// (e.g. a bare local Vite preview), exactly like the Navbar link already does.
// Same-origin means the browser sends the operator's existing console session
// cookie — logged in outside, logged in inside the panel; logged out outside,
// the panel shows the console's own front door/login, same as opening the tab
// would. No new auth surface, nothing to weaken.
//
// Deliberately UNSANDBOXED: the console (ApiAccess "Revoke key", ModelTable
// "Delete"/"Prune", …) relies on window.confirm()/alert() for destructive
// actions and navigator.clipboard for copy buttons. A sandboxed iframe without
// `allow-modals` silently swallows confirm()/alert() (the dialog never shows;
// confirm() just returns false) — Delete/Revoke would look broken. Since this
// is first-party same-origin content (not a foreign/untrusted embed like
// ComfyStation's user-supplied URL), there is no isolation benefit to
// sandboxing it, only breakage — so `sandbox` is omitted entirely. Sharing a
// window with the SANDBOXED ComfyUI tab changes nothing here: they are separate
// <iframe> elements with independent sandbox attributes, never one reused frame.
//
// ── FRAME-ANCESTORS (pre-checked, see task notes) ────────────────────────────
// Neither the Flask app (abstract_hugpy_dev/flask_app) nor the dev.hugpy.ai
// nginx vhost sends X-Frame-Options or a frame-ancestors CSP, so the browser
// does not block this frame. Re-verify after any nginx/Flask header change.
import { siteBase } from "../nav/Navbar";
import type { DrawerTab } from "./drawerShell";

// Same origin-relative rule the Navbar's own Console link uses.
const consoleHref = `${siteBase}/console`;

/** The console's tab in the unified drawer. */
export function consoleDrawerTab(): DrawerTab<"console"> {
  return {
    id: "console",
    label: "Console",
    hint: "Open the hugpy console alongside this station — no tab switch",
    toggleClassName: "vi-console-toggle",
    toggleLabel: "▤ Console",
    ariaLabel: "hugpy console",
    // Stays in the DOM while another tab shows: an iframe that is merely
    // `display:none` keeps its document, its session and its scroll position, so
    // switching to Active and back does not throw the operator out to a reload.
    keepMounted: true,
    headerActions: (
      <a
        className="vi-btn vi-btn-sm vi-btn-ghost"
        href={consoleHref}
        target="_blank"
        rel="noreferrer"
        title="Open the console in its own tab"
      >
        Open in new tab ↗
      </a>
    ),
    body: (
      <iframe
        className="vi-console-iframe"
        src={consoleHref}
        title="hugpy console"
        allow="clipboard-read; clipboard-write; fullscreen"
      />
    ),
  };
}
