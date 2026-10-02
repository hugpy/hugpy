// Auth wiring is decided by the API instance, not baked into the bundle.
// GET /api/auth/config → { mode: "external", base: "https://…" }
//                      | { mode: "open",     base: null }
// "external": classic login against a separate auth service.
// "open":     single-operator instance — no login wall at all.
//
// If the API can't be reached at all (network error, or a 5xx from a dead /api
// upstream — e.g. the public hugpy.ai front door, whose backend isn't hosted),
// we fall back to OPEN, and flag `reachable: false`. Rationale: an unreachable
// auth service cannot authenticate anyone, so falling back to *external* is a
// dead end — the bundle would show a login wall against a 502, or hang forever
// in "Checking session…" (a 502 from /me is neither ok nor 401/403). hugpy is
// self-hosted-first; "no central reachable" means "single-operator open", not
// "log in". `reachable` lets the router show the welcome/Landing front door for
// a backendless deployment instead of a console that can't talk to anything.
// `base` is kept same-origin (/api/auth-svc) so getAuthBase() never returns
// null, even though open mode never hits it.

import { hugpyFetch } from '../runtime/config'

export type AuthConfig = {
  mode: 'external' | 'open'
  base: string | null
  /** Did GET /api/auth/config actually answer? false ⇒ backend unreachable. */
  reachable: boolean
}

const FALLBACK: AuthConfig = {
  mode: 'open',
  base: '/api/auth-svc',
  reachable: false,
}

let cached: Promise<AuthConfig> | null = null

// One probe attempt. Returns a resolved AuthConfig, or the sentinel 'retry' when
// the failure looks TRANSIENT (network error / abort-timeout / 5xx upstream) so
// the caller can try again. A 200 that isn't our JSON (the public hugpy.ai static
// front door returns the SPA's index.html) is DEFINITIVE "no API here" → we don't
// retry that, so prod drops to the demo instantly with no delay.
async function probeAuthConfigOnce(): Promise<AuthConfig | 'retry'> {
  try {
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), 4000)
    let r: Response
    try {
      r = await hugpyFetch('/api/auth/config', {
        headers: { Accept: 'application/json' },
        signal: ctrl.signal,
      })
    } finally {
      clearTimeout(t)
    }
    if (r.status >= 500) return 'retry'           // transient upstream (502/503/504/…)
    if (!r.ok) return { ...FALLBACK }             // 4xx → no usable API here (definitive)
    const text = await r.text()
    try {
      const d: any = JSON.parse(text)
      return d && (d.mode === 'open' || d.mode === 'external')
        ? { mode: d.mode, base: d.base ?? FALLBACK.base, reachable: true }
        : { ...FALLBACK }                         // 200 but wrong shape → no API (definitive)
    } catch {
      return { ...FALLBACK }                       // 200 HTML (prod static) → no API (definitive)
    }
  } catch {
    return 'retry'                                 // network error / timeout → transient
  }
}

// Resolve reachability robustly: a single transient blip from a LIVE instance
// (e.g. the dev API momentarily starved) must NOT flip us to the backendless
// (showroom/demo) posture — that caused the "showroom flash" on dev. We only
// conclude `reachable:false` after transient failures persist across a few quick
// retries; a definitive "no API" (prod) returns on the first attempt.
async function resolveAuthConfig(): Promise<AuthConfig> {
  const ATTEMPTS = 3
  for (let i = 0; i < ATTEMPTS; i++) {
    const res = await probeAuthConfigOnce()
    if (res !== 'retry') return res
    if (i < ATTEMPTS - 1) await new Promise(res2 => setTimeout(res2, 600 * (i + 1)))
  }
  return { ...FALLBACK }                            // transient failures exhausted → unreachable
}

export function getAuthConfig(): Promise<AuthConfig> {
  if (!cached) cached = resolveAuthConfig()
  return cached
}

export async function getAuthBase(): Promise<string> {
  const cfg = await getAuthConfig()
  return cfg.base ?? FALLBACK.base!
}
