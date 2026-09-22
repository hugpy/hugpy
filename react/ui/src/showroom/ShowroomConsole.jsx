// ShowroomConsole — the explorable demo wrapper for the backend-less /console.
//
// On the public front door (hugpy.ai/console for a non-install onlooker) the
// backend is unreachable, so ConsoleGate renders this instead of the bare
// CentralConnect shell. It installs the demo-fetch shim (so the REAL <Console/>,
// passed as children, renders fully populated from canned fixtures with zero
// backend exposure), overlays a demo banner that folds in the pip-install /
// "bring your own inference" go-live path, surfaces demo-mode toasts, and — since
// the canned demo doubles as a style sandbox — a live THEME switcher.
//
// The shim is installed during this component's first render (a lazy useState
// initializer) — before <Console/> mounts and fires its data-loading effects —
// and torn down on unmount, so the demo is fully scoped to this view.
import { useState, useEffect, cloneElement, isValidElement } from 'react'
import { Helmet } from 'react-helmet-async'
import { CentralConnect } from '../components'
import { installDemoMode, uninstallDemoMode, onDemoToast } from './demoFetch'
import { isDemoHost } from '../runtime/localCentral'
import { THEMES, DEFAULT_THEME, getTheme, themeStyle } from './themes.js'
import './showroom.css'

const INSTALL_CMD = 'pip install hugpy && hugpy serve --port 7002'
const THEME_KEY = 'hugpy.showroom.theme'

// ── theme application (inject fonts + a single scoped <style>) ───────────────
function loadFonts(urls) {
  for (const href of urls || []) {
    if (document.querySelector(`link[data-sr-font="${href}"]`)) continue
    const l = document.createElement('link')
    l.rel = 'stylesheet'
    l.href = href
    l.setAttribute('data-sr-font', href)
    document.head.appendChild(l)
  }
}
function setThemeStyle(css) {
  let el = document.getElementById('sr-theme-style')
  if (!el) {
    el = document.createElement('style')
    el.id = 'sr-theme-style'
    document.head.appendChild(el)
  }
  el.textContent = css || ''
}

function DemoBanner({ onConnect, theme, setTheme, surface, setSurface }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    try {
      navigator.clipboard.writeText(INSTALL_CMD)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard blocked */ }
  }
  return (
    <div className="sr-banner" role="note">
      <span className="sr-badge">DEMO</span>
      {/* Two faces of hugpy, both explorable in the demo: the operator console
          (manage models / compute / API) and the clean end-user Media chat. */}
      <span className="sr-surface" role="group" aria-label="Demo surface">
        <button
          type="button"
          className={surface === 'console' ? 'is-current' : ''}
          aria-pressed={surface === 'console'}
          onClick={() => setSurface('console')}
        >
          Operator console
        </button>
        <button
          type="button"
          className={surface === 'media' ? 'is-current' : ''}
          aria-pressed={surface === 'media'}
          onClick={() => setSurface('media')}
        >
          Media chat
        </button>
      </span>
      <span className="sr-text">
        You're exploring hugpy with <strong>sample data</strong> — nothing here runs models or
        touches a real backend. Run it for real on hardware <em>you</em> own:
      </span>
      <button type="button" className="sr-cmd" onClick={copy} title="Copy install command">
        <code>{INSTALL_CMD}</code>
        <span className="sr-cmd-ico">{copied ? '✓ copied' : '⧉'}</span>
      </button>
      <label className="sr-theme-pick" title="Try a different style on the demo">
        <span aria-hidden="true">🎨</span>
        <select value={theme} onChange={(e) => setTheme(e.target.value)} aria-label="Demo theme">
          {THEMES.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
        </select>
      </label>
      <button type="button" className="sr-connect" onClick={onConnect}>
        Connect your inference →
      </button>
    </div>
  )
}

function DemoToasts() {
  const [items, setItems] = useState([])
  useEffect(() =>
    onDemoToast((msg) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
      setItems((xs) => [...xs.slice(-3), { id, msg }])
      setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), 4200)
    }), [])
  if (!items.length) return null
  return (
    <div className="sr-toasts" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className="sr-toast">{t.msg}</div>
      ))}
    </div>
  )
}

export default function ShowroomConsole({ children }) {
  // Install the demo shim synchronously on first render (lazy init runs once),
  // before the wrapped <Console/> mounts and starts fetching.
  useState(() => { installDemoMode(); return true })
  const [view, setView] = useState('demo') // 'demo' | 'connect'
  const [surface, setSurface] = useState('console') // 'console' | 'media'
  const [theme, setThemeState] = useState(() => {
    try {
      // ?theme=<id> wins (shareable demo-style links), then the saved choice.
      const q = new URLSearchParams(window.location.search).get('theme')
      if (q && THEMES.some((t) => t.id === q)) return q
      return localStorage.getItem(THEME_KEY) || DEFAULT_THEME
    } catch { return DEFAULT_THEME }
  })
  const setTheme = (id) => {
    setThemeState(id)
    try { localStorage.setItem(THEME_KEY, id) } catch { /* private mode */ }
  }

  // Apply the active theme: load its fonts + inject its scoped token/override CSS.
  useEffect(() => {
    const t = getTheme(theme)
    loadFonts(t.fonts)
    setThemeStyle(themeStyle(t))
  }, [theme])

  // On the brochure (hugpy.ai ?demo=1) the shim is scoped to this view: tear it
  // down on unmount so leaving the demo restores real fetch. On the dedicated
  // demo host the shim is installed process-wide at boot (see showroom/earlyInstall)
  // and must SURVIVE navigating away from /console — otherwise Landing's readiness
  // probe would 404 — so we keep it installed there.
  useEffect(() => () => { if (!isDemoHost()) uninstallDemoMode(); setThemeStyle('') }, [])

  // The "Connect your inference" path: the real CentralConnect shell, which
  // probes the visitor's own localhost:7002 with a raw fetch (unaffected by the
  // shim) — the genuine go-live bridge.
  if (view === 'connect') return <CentralConnect />

  // The DEMO banner is built once and placed INSIDE the header stack of whichever
  // surface is active, so it always sits directly beneath that surface's nav
  // (operator directive 2026-07-21: nav + demo tag = one fixed header stack).
  //   • operator console: hand the banner to <Console banner=…/> → it renders
  //     inside the console Navbar's own sticky .navbar-stack, under the nav.
  //   • media chat: the arm lives in an iframe with its OWN nav; embed=1 drops
  //     the arm's banner (host supplies chrome), so we render this banner above
  //     the frame here — it is the frame's header.
  const bannerEl = (
    <DemoBanner
      onConnect={() => setView('connect')}
      theme={theme}
      setTheme={setTheme}
      surface={surface}
      setSurface={setSurface}
    />
  )

  return (
    <div className={`showroom sr-theme-${theme}`}>
      <Helmet>
        <title>Hugpy Console — Live Demo · Self-Hosted LLM Server</title>
        <meta name="description" content="Explore the Hugpy console in your browser: pull Hugging Face models, chat with streaming auto-continuation, mint API keys, and watch GPU worker allocation — a live, no-install demo of the self-hosted LLM platform." />
        <link rel="canonical" href="https://hugpy.ai/console" />
        <meta property="og:title" content="Hugpy Console — Live Demo" />
        <meta property="og:url" content="https://hugpy.ai/console" />
        <meta property="og:description" content="A live, no-install browser demo of the Hugpy self-hosted LLM console." />
      </Helmet>
      <div className="showroom-body">
        {/* The end-user side (the media-intelligence arm) is a separate build
            served at /media; embed it in its own canned demo mode. embed=1 tells
            the arm to drop its own banner (this showroom provides the chrome) but
            keep its nav bar, mirroring the operator console's own nav. The banner
            is the frame's header here (the iframe's own nav lives inside it). */}
        {surface === 'media' ? (
          <>
            {bannerEl}
            <iframe
              className="sr-media-frame"
              src="/media/?demo=1&embed=1"
              title="hugpy · media intelligence (demo)"
            />
          </>
        ) : (
          // Inject the banner INTO the console so it renders inside the console
          // Navbar's sticky stack, directly under the nav.
          isValidElement(children) ? cloneElement(children, { banner: bannerEl }) : children
        )}
      </div>
      <DemoToasts />
    </div>
  )
}
