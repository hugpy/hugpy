/*
 * Navbar — the shared hugpy top nav, ported into the media-intelligence arm.
 *
 * Mirrors the dev UI's one shared Navbar EXACTLY: a three-zone bar — brand
 * pinned left, the primary links (Docs / Console / Media / Video) optically
 * centered, and page-specific controls pinned right — so this arm reads as
 * part of the same product rather than an island. Sticky, blurred, and
 * bottom-bordered, matching the welcome page's nav.
 *
 * Cross-SPA links: this arm is a SEPARATE SPA mounted under /media (its
 * react-router basename is "/media"). The other destinations (/docs, /console,
 * the welcome anchors) are routes of the MAIN hugpy app, NOT of this SPA — so
 * they are plain full-navigation <a> links, never react-router <Link>s (a Link
 * here would resolve under /media and dead-end). On a hugpy origin the site
 * links are origin-relative, so dev.hugpy.ai/media → dev.hugpy.ai/docs (same
 * origin, like the shared nav). On any OTHER origin — notably the hugpy-chat HF
 * Space, which reverse-proxies this exact build but only serves /media + /api —
 * an origin-relative /docs would 404, so the links point at hugpyConfig.siteUrl
 * (hugpy.ai) instead. This is decided at runtime from the live hostname because
 * the same build serves both origins (see below).
 */
import BrandMark from "./BrandMark";
import { hugpyConfig } from "../../../config";
import { isDemo, isEmbedded } from "../../../demo/mode";
// The link SET + order come from the ONE shared manifest so this arm can never
// drift from the rest of the site. Rendering stays local (runtime siteBase,
// iframe target, current-marking, the demo `?demo=1` retarget). Plain .js, so
// TS imports it via allowJs with no .d.ts.
import { buildNavItems } from "../../../../../ui_shared/navbar/links";
import "../../../../../ui_shared/navbar/navbar.css";
import "./Navbar.css";

// Origin-relative on a hugpy origin; the public hugpy site (siteUrl) otherwise.
// Strip a trailing slash so `${base}/x` never doubles up.
//
// Decided at RUNTIME, not from the build-time base: ONE build is served at two
// origins — directly at dev.hugpy.ai/media (where /docs and /console exist), and
// verbatim via reverse proxy at the hugpy-chat HF Space (where they don't, so an
// origin-relative /docs 404s). BASE_URL is identical in both, so only the live
// hostname can tell them apart.
const onHugpyOrigin =
  typeof window !== "undefined" &&
  /(^|\.)hugpy\.ai$/i.test(window.location.hostname);
const siteBase = onHugpyOrigin ? "" : hugpyConfig.siteUrl.replace(/\/$/, "");

export default function Navbar({
  children,
}: {
  children?: React.ReactNode;
}): JSX.Element {
  // Embedded in the showroom <iframe>: site links escape the frame and load at
  // top level, so clicking Docs/Console doesn't strand you inside a tiny frame.
  const target = isEmbedded() ? "_top" : undefined;
  return (
    <nav className="hugpy-navbar">
      <span className="hugpy-navbar-brand">
        <BrandMark />
      </span>
      <span className="hugpy-navbar-links">
        {/* Link SET + order come from the ONE shared manifest so they stay
            identical sitewide. Rendering is local: every destination is a plain
            full-navigation <a> (they're OTHER SPAs, not routes of this one),
            prefixed with the runtime siteBase and given target="_top" when
            iframed. Media is the current surface. In ANY demo flavor the Video
            link routes into the video arm's CANNED demo (it has no live mode),
            so a demo visitor stays in demo land — hence the `?demo=1` override. */}
        {buildNavItems({
          currentKey: "media",
          siteBase,
          hrefByKey: {
            video: `${siteBase}/video/${isDemo() ? "?demo=1" : ""}`,
          },
        }).map((item) => (
          <a
            key={item.key}
            href={item.href}
            target={target}
            aria-current={item.current ? "page" : undefined}
            className={item.current ? "is-current" : undefined}
          >
            {item.label}
          </a>
        ))}
      </span>
      <span className="hugpy-navbar-side">
        {children}
      </span>
    </nav>
  );
}
