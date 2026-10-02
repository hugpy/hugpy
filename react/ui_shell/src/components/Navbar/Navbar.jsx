// Navbar — the one shared top nav, modeled on the welcome page's nav.
//
// Sticky, blurred, bottom-bordered bar whose right side is a horizontally
// scrollable link row (no hamburger — same mobile behavior the welcome page
// uses, which the operator signed off on). One source of truth so the brand +
// links read identically on every page (welcome, login, console, docs, the
// no-backend connect shell).
//
//   brand     — show the BrandMark on the left (default). The welcome page
//               passes brand={false} because its hero lockup already carries
//               the brand.
//   left      — content pinned to the far left, before the brand (e.g. the docs
//               sidebar toggle).
//   children  — extra page-specific controls appended to the right of the link
//               row (e.g. the console's "Sign out" button).
import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getAuthConfig } from '../../Auth/authConfig'
import BrandMark from '../BrandMark/BrandMark'
import { buildNavItems } from '../../../../ui_shared/navbar/links'
import { isDemoHost } from '../../runtime/localCentral'
import '../../../../ui_shared/navbar/navbar.css'
import './Navbar.css'

// Which link keys are react-router routes of THIS app (rendered as <Link>) vs
// separate SPA arms reached by a full navigation (<a>). The arms live at
// /media, /video, /fleet and are NOT react-router routes here.
const ROUTER_KEYS = new Set(['docs', 'console'])

export default function Navbar({ brand = true, left, children, banner, className = '' }) {
  // "Console" never dead-ends: open-mode → /console, external-auth → /login.
  // Resolved without useAuth() so the nav stays usable outside AuthProvider.
  const [consoleTo, setConsoleTo] = useState('/console')
  // Media, video, and fleet are all public demo arms now — each ships a
  // self-contained canned showroom (the prod brochure defaults a bare hit to
  // ?demo=1), so their nav links are plain, ungated full-navigations. (Video
  // was previously auth-gated like the console; it graduated to a public
  // showroom on 2026-07-15.) Only the console still routes through /login in
  // external-auth mode.
  useEffect(() => {
    let alive = true
    getAuthConfig().then(c => {
      if (!alive) return
      setConsoleTo(c.mode === 'external' ? '/login' : '/console')
    })
    return () => { alive = false }
  }, [])

  // The nav + its optional demo banner are ONE sticky header stack, pinned to
  // the top of the viewport as a single unit (operator directive 2026-07-21:
  // "the navbar and its demo tag should be a static component fixed at the top,
  // site and demo wide"). Wrapping both in a single `position: sticky` element
  // (.hugpy-navbar-stack) means the banner sits DIRECTLY beneath the nav and neither
  // slides under the other on scroll — the whole stack stays put. When there's
  // no banner the stack is just the bare nav.
  return (
    <header className="hugpy-navbar-stack">
      <nav className={`hugpy-navbar ${className}`}>
        {/* Three zones: brand pinned left, the primary nav centered, page-specific
            controls (e.g. the console "Sign out") pinned right. The 1fr side
            columns keep the centered links optically centered regardless of how
            wide the brand or the right-side controls are. */}
        <span className="hugpy-navbar-brand">
          {left}
          {brand && <BrandMark />}
        </span>
        <span className="hugpy-navbar-links">
          {/* Link SET + order come from the ONE shared manifest
              (ui_shared/navbar/links.js) so they can't drift from the arms.
              Rendering stays per-arm: docs/console are react-router routes of
              THIS app (<Link>); the media/video/fleet arms are separate SPAs
              reached by a plain full navigation (<a>). Console resolves through
              auth mode (open → /console, external → /login) via hrefByKey. All
              three arms are public demo arms — ungated, no /login hop. */}
          {buildNavItems({ hrefByKey: { console: consoleTo } }).map((item) =>
            ROUTER_KEYS.has(item.key) ? (
              <Link key={item.key} to={item.href}>{item.label}</Link>
            ) : (
              <a key={item.key} href={item.href}>{item.label}</a>
            )
          )}
          {/* Demo-host extras (demo.hugpy.ai only): the Station showroom and
              its hugpy-agent seat playback — canned fictional fleet, served
              same-origin at /station (webpack → in-VM console-showroom, always
              the current release's UI). Same links the HF cards carry. */}
          {isDemoHost() && (
            <>
              <a href="/station/" target="_blank" rel="noreferrer">Station</a>
              <a href="/station/?open=tern:keeper" target="_blank" rel="noreferrer">Agent</a>
            </>
          )}
        </span>
        <span className="hugpy-navbar-side">
          {children}
        </span>
      </nav>
      {/* Demo banner (when a demo flavor supplies one) — rides in the same
          sticky stack, directly under the nav. */}
      {banner}
    </header>
  )
}
