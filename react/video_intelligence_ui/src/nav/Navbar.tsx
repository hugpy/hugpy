/*
 * Navbar — the shared hugpy top nav, ported into the video-intelligence arm.
 *
 * A faithful copy of the media arm's shared Navbar (COPIED, never imported
 * cross-arm): a three-zone bar — brand pinned left, the primary links
 * (Docs / Console / Media / Video) optically centered, and a page-specific
 * controls slot pinned right — so this arm reads as part of the same product.
 * Sticky, blurred, and bottom-bordered, matching the sitewide nav.
 *
 * Cross-SPA links: this arm is a SEPARATE SPA mounted under /video (its
 * react-router basename is "/video"). The other destinations (/docs, /console,
 * /media, the site root) are routes of OTHER hugpy SPAs, NOT of this one — so
 * they are plain full-navigation <a> links, never react-router <Link>s (a Link
 * here would resolve under /video and dead-end). On a hugpy origin the site
 * links are origin-relative (dev.hugpy.ai/video → dev.hugpy.ai/docs); on any
 * OTHER origin they point at hugpyConfig.siteUrl (hugpy.ai) instead.
 *
 * Ported from media_intelligence_ui/src/chat/src/ui/Navbar.tsx, dropping the
 * demo/mode (`isEmbedded`) import — this arm has no iframe showroom, so the
 * `target="_top"` escape hatch is unnecessary and every link is a normal nav.
 */
import type { ReactNode } from "react";
import BrandMark from "./BrandMark";
import { hugpyConfig } from "../config";
// The link SET + order come from the ONE shared manifest so this arm can never
// drift from the rest of the site. Rendering stays local (runtime siteBase,
// current-marking). Plain .js, imported via allowJs with no .d.ts.
import { buildNavItems } from "../../../ui_shared/navbar/links";
import "../../../ui_shared/navbar/navbar.css";
import "./Navbar.css";

// Origin-relative on a hugpy origin; the public hugpy site (siteUrl) otherwise.
// Strip a trailing slash so `${base}/x` never doubles up. Decided at RUNTIME
// from the live hostname (the same build can be served at more than one origin).
// Exported so any other in-arm component that needs a same-origin link to a
// sibling hugpy SPA (e.g. ConsolePanel's iframe target) can reuse the exact
// same rule instead of re-deriving it — this is the one place it's computed.
export const onHugpyOrigin =
  typeof window !== "undefined" &&
  /(^|\.)hugpy\.ai$/i.test(window.location.hostname);
export const siteBase = onHugpyOrigin ? "" : hugpyConfig.siteUrl.replace(/\/$/, "");

export default function Navbar({ children }: { children?: ReactNode }) {
  return (
    <nav className="hugpy-navbar">
      <span className="hugpy-navbar-brand">
        <BrandMark />
      </span>
      <span className="hugpy-navbar-links">
        {/* Link SET + order come from the ONE shared manifest so they stay
            identical sitewide. Rendering is local: every destination is a plain
            full-navigation <a> to a sibling hugpy SPA, prefixed with the runtime
            siteBase. Video is the current surface. */}
        {buildNavItems({ currentKey: "video", siteBase }).map((item) => (
          <a
            key={item.key}
            href={item.href}
            aria-current={item.current ? "page" : undefined}
            className={item.current ? "is-current" : undefined}
          >
            {item.label}
          </a>
        ))}
      </span>
      <span className="hugpy-navbar-side">{children}</span>
    </nav>
  );
}
