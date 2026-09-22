/*
 * ui_shared/navbar/links.js — the ONE canonical top-nav link set for the whole
 * hugpy site.
 *
 * This file is the single source of truth for WHICH links the top nav shows and
 * in WHAT ORDER, imported by every SPA arm so the set can never drift again:
 *   - ui/                     (main app, webpack, react-router) — welcome/docs/console
 *   - media_intelligence_ui/  (vite)  — /media
 *   - video_intelligence_ui/  (vite)  — /video
 *   - agents_ui/              (vite)  — /fleet
 *
 * It is deliberately a plain, dependency-free .js data+helper module (NO JSX, no
 * React import): the main app is JS/JSX on webpack; the three arms are TS/TSX on
 * Vite. A pure .js module with allowJs (already on in every tsconfig) imports
 * cleanly into all four bundlers from OUTSIDE their src roots, with no alias, no
 * .d.ts, and no JSX-vs-TSX pragma mismatch. Each arm keeps its own thin
 * rendering (react-router <Link> vs plain <a>, runtime siteBase, iframe target,
 * demo query, current-marking) — only the link SET is shared here.
 *
 * `path` is the site-absolute path of each destination, WITHOUT an origin. Arms
 * that must retarget cross-origin (the media/video arms, whose build is also
 * reverse-proxied by an HF Space) prefix it with their runtime `siteBase`; the
 * main app maps `console` through its auth mode. See buildNavItems() below.
 */

/**
 * @typedef {Object} NavLink
 * @property {string} key   Stable identifier for the surface ("docs","console",
 *                          "media","video","fleet"). Used to mark the current
 *                          surface and to key per-arm href overrides.
 * @property {string} label Visible text.
 * @property {string} path  Site-absolute path (leading slash, no origin).
 */

/** The canonical ordered link set. Order here IS the on-screen order sitewide. */
export const NAV_LINKS = /** @type {NavLink[]} */ ([
  { key: "docs", label: "Docs", path: "/docs" },
  { key: "console", label: "Console", path: "/console" },
  { key: "media", label: "Media", path: "/media/" },
  { key: "video", label: "Video", path: "/video/" },
  { key: "fleet", label: "Fleet", path: "/fleet/" },
]);

/**
 * @typedef {Object} NavItem  A ready-to-render descriptor for one link.
 * @property {string} key
 * @property {string} label
 * @property {string} href     Final href (siteBase + path + any per-key override).
 * @property {boolean} current Whether this link is the surface currently shown.
 */

/**
 * Build the per-arm list of render descriptors from the canonical set.
 *
 * Every arm calls this and then renders each item its own way (a <Link>, an <a>,
 * with/without target, etc.) — so the SET/ORDER stay shared while the rendering
 * mechanics stay per-arm.
 *
 * @param {Object} [opts]
 * @param {string} [opts.currentKey]  key of the surface being shown → item.current
 * @param {string} [opts.siteBase]    origin prefix for cross-origin arms (""=same
 *                                    origin). A trailing slash is stripped so
 *                                    `${siteBase}${path}` never doubles up.
 * @param {Object.<string,string>} [opts.hrefByKey]  full-href overrides by key
 *                                    (e.g. main app's auth-resolved console href,
 *                                    or the video arm's `?demo=1` variant). When
 *                                    present the override is used verbatim and
 *                                    siteBase is NOT applied to it.
 * @returns {NavItem[]}
 */
export function buildNavItems(opts = {}) {
  const { currentKey, siteBase = "", hrefByKey = {} } = opts;
  const base = siteBase ? siteBase.replace(/\/$/, "") : "";
  return NAV_LINKS.map((l) => ({
    key: l.key,
    label: l.label,
    href: hrefByKey[l.key] != null ? hrefByKey[l.key] : `${base}${l.path}`,
    current: currentKey === l.key,
  }));
}
