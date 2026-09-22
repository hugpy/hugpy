// Shared helpers for the visitor's OWN local hugpy central — the "bring your own
// central" target a backendless front door (the public hugpy.ai brochure) hands
// off to. Single source of truth, used by:
//   - the /console gate (App.jsx) to decide: real local console vs. showroom
//   - CentralConnect (the manual connect shell)
//
// Why a `no-cors` probe to /readiness:
//   The brochure is served over https (hugpy.ai), but the visitor's own central
//   is plain http on a loopback address (http://localhost:7002). Two things make
//   this work and nothing else does:
//     1. Mixed content — browsers EXEMPT loopback (localhost / 127.0.0.1) from
//        https mixed-content blocking, so an https page may fetch it.
//     2. CORS — the central sends no CORS headers to a cross-origin page, so a
//        normal fetch would be blocked reading the response. `mode:'no-cors'`
//        sidesteps that: the request still goes out and the promise RESOLVES if a
//        server answered (opaque response) and REJECTS on connection-refused —
//        which is exactly the connected/disconnected signal we need.
//   A real local central is SAME-origin when opened directly at its port, so the
//   full console there has no CORS problem — we just navigate the browser to it.

export const DEFAULT_CENTRAL = 'http://localhost:7002'
export const CENTRAL_LS_KEY = 'hugpy.centralUrl'

// The dedicated demo host (demo.hugpy.ai). The console gate uses this to ALWAYS
// serve the showroom there — even when a same-origin backend is reachable and
// without probing the visitor's localhost central — so a demo visitor never
// lands in a live console. Narrow on purpose: only the exact `demo.hugpy.ai`
// label, not any *.hugpy.ai.
export function isDemoHost() {
  try {
    return /^demo\.hugpy\.ai$/i.test(window.location.hostname)
  } catch {
    return false
  }
}

export function normalizeCentral(u) {
  let s = (u || '').trim().replace(/\/+$/, '')
  if (s && !/^https?:\/\//i.test(s)) s = 'http://' + s
  return s
}

/** The central the visitor last pointed at, else the localhost default. */
export function readSavedCentral() {
  try {
    return normalizeCentral(localStorage.getItem(CENTRAL_LS_KEY) || '') || DEFAULT_CENTRAL
  } catch {
    return DEFAULT_CENTRAL
  }
}

/**
 * Probe whether a hugpy central is listening at `base`. Resolves `true` if a
 * server answers (even without CORS headers), `false` on connection-refused,
 * abort/timeout, or a blank URL. See the module header for why this is `no-cors`.
 */
export async function probeLocalCentral(raw, { timeoutMs = 2500 } = {}) {
  const base = normalizeCentral(raw)
  if (!base) return false
  try {
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), timeoutMs)
    try {
      await fetch(base + '/readiness', { mode: 'no-cors', cache: 'no-store', signal: ctrl.signal })
    } finally {
      clearTimeout(t)
    }
    return true
  } catch {
    return false
  }
}
