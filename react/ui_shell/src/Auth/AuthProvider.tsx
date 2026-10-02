// AuthProvider — the single source of truth for auth state *and* auth
// networking. Components never call the auth API directly; they read state and
// call the actions exposed here (signIn/signOut/signUp/changePassword/refresh).
// Centralizing both concerns in one place is deliberate: auth + networking are
// the easy things to scatter and get subtly wrong.
//
// Configuration is layered, most-specific wins:
//   1. props on <AuthProvider> (base, mode, endpoints, fetch, credentials)
//   2. the API instance's own answer at GET /api/auth/config
//      → { mode: 'open' } (no login wall) | { mode: 'external', base } (login
//        against a separate auth service)
//   3. a hard fallback (see ./authConfig)
//
// "open" mode short-circuits everything to a local single-operator session.

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { getAuthConfig } from './authConfig'

export type AuthUser = { username?: string; local?: boolean; [key: string]: unknown }

export type AuthState =
  | { status: 'checking' }
  | { status: 'guest' }
  | { status: 'authed'; user: AuthUser }

/** Result of a write action that may carry a server-supplied error message. */
export interface AuthResult {
  ok: boolean
  error?: string
}

/** Paths on the auth service, relative to its base. Override individually. */
export interface AuthEndpoints {
  me: string
  login: string
  logout: string
  register: string
  changePassword: string
}

/** The product this console signs into, sent as `site` on POST /login. The
 *  central auth service uses it to enforce site membership for non-admins (the
 *  server side of the same rule operator_auth.MEMBER_SITES applies here). */
const AUTH_SITE = 'hugpy'

const DEFAULT_ENDPOINTS: AuthEndpoints = {
  me: '/me',
  login: '/login',
  logout: '/logout',
  register: '/register',
  changePassword: '/change-password',
}

export interface AuthContextValue {
  state: AuthState
  /** Resolved auth mode ('open' = no login wall, 'external' = login service). */
  mode: 'open' | 'external'
  /** false ⇒ the API instance is unreachable (backendless front door). */
  reachable: boolean
  /** Re-check the current session (GET /me). */
  refresh: () => Promise<void>
  /** Returns true on success. */
  signIn: (username: string, password: string) => Promise<boolean>
  signOut: () => Promise<void>
  signUp: (input: { username: string; email?: string; password: string }) => Promise<AuthResult>
  changePassword: (input: { currentPassword: string; newPassword: string }) => Promise<AuthResult>
}

export interface AuthProviderProps {
  children?: React.ReactNode
  /** Auth service origin. If omitted, resolved from GET /api/auth/config. */
  base?: string
  /** Force a mode. If omitted, resolved from config. */
  mode?: 'open' | 'external'
  /** Override one or more endpoint paths. */
  endpoints?: Partial<AuthEndpoints>
  /** Custom fetch (e.g. to inject headers). Defaults to global fetch. */
  fetch?: typeof fetch
  /** Credentials mode for auth requests. Defaults to 'include' (cookie auth). */
  credentials?: RequestCredentials
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({
  children,
  base,
  mode,
  endpoints,
  fetch: fetchProp,
  credentials = 'include',
}: AuthProviderProps) {
  const [state, setState] = useState<AuthState>({ status: 'checking' })
  // Resolved once auth config is known. Defaults assume a healthy local
  // instance; flipped to the backendless posture if the API can't be reached.
  const [resolvedMode, setResolvedMode] = useState<'open' | 'external'>('open')
  const [reachable, setReachable] = useState(true)

  // Stabilize the fetch reference. Without useMemo this is a new function on
  // every render, which cascades through callAuth → refresh and makes the
  // `useEffect(refresh, [refresh])` below fire on every render — an infinite
  // setState/re-render loop that freezes the page (most visibly on login/logout).
  const doFetch = useMemo(
    () => fetchProp ?? ((...args: Parameters<typeof fetch>) => globalThis.fetch(...args)),
    [fetchProp],
  )
  const eps = useMemo<AuthEndpoints>(() => ({ ...DEFAULT_ENDPOINTS, ...endpoints }), [endpoints])

  // Resolve { mode, base, reachable } once: props win, else ask the API
  // instance. Forced props imply a known config (reachable: true).
  const resolved = useRef<
    Promise<{ mode: 'open' | 'external'; base: string; reachable: boolean }> | null
  >(null)
  const getCfg = useCallback(async () => {
    if (!resolved.current) {
      resolved.current = (async () => {
        if (mode === 'open') return { mode: 'open' as const, base: base ?? '', reachable: true }
        if (mode === 'external' && base)
          return { mode: 'external' as const, base, reachable: true }
        const cfg = await getAuthConfig()
        return { mode: cfg.mode, base: base ?? cfg.base ?? '', reachable: cfg.reachable }
      })()
      // Publish the resolved posture so the router/Landing can react to a
      // backendless front door (open + unreachable) vs a live local instance.
      resolved.current.then(c => {
        setResolvedMode(c.mode)
        setReachable(c.reachable)
      })
    }
    return resolved.current
  }, [mode, base])

  const callAuth = useCallback(
    async (path: string, init?: RequestInit) => {
      const cfg = await getCfg()
      return doFetch(`${cfg.base}${path}`, { credentials, ...init })
    },
    [getCfg, doFetch, credentials],
  )

  const refresh = useCallback(async () => {
    const cfg = await getCfg()
    if (cfg.mode === 'open') {
      setState({ status: 'authed', user: { username: 'local', local: true } })
      return
    }
    try {
      const r = await callAuth(eps.me, { method: 'GET', headers: { Accept: 'application/json' } })
      if (r.ok) {
        setState({ status: 'authed', user: await r.json() })
        return
      }
      if (r.status === 401 || r.status === 403) setState({ status: 'guest' })
    } catch {
      setState((cur) => (cur.status === 'checking' ? { status: 'guest' } : cur))
    }
  }, [getCfg, callAuth, eps.me])

  const signIn = useCallback(
    async (username: string, password: string) => {
      const cfg = await getCfg()
      if (cfg.mode === 'open') {
        await refresh()
        return true
      }
      const r = await callAuth(eps.login, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        // `site` scopes the sign-in to THIS product: the central auth service
        // enforces that a non-admin holds the named site grant before it issues
        // a session, so an approved account with no hugpy grant is turned away
        // at login instead of half-signing-in and then failing the member gate
        // on every /media, /ml, /uploads, /session and /chat call.
        body: JSON.stringify({ username, password, site: AUTH_SITE }),
      })
      if (!r.ok) {
        setState({ status: 'guest' })
        return false
      }
      await refresh()
      return true
    },
    [getCfg, callAuth, eps.login, refresh],
  )

  const signOut = useCallback(async () => {
    const cfg = await getCfg()
    if (cfg.mode === 'open') {
      // Open mode has no server session to clear, but we still drop to guest so
      // the operator can leave the console and reach /login (e.g. to exercise
      // the external sign-in flow). A reload re-establishes the open session.
      setState({ status: 'guest' })
      return
    }
    try {
      await callAuth(eps.logout, { method: 'POST', headers: { Accept: 'application/json' } })
    } finally {
      setState({ status: 'guest' })
    }
  }, [getCfg, callAuth, eps.logout])

  const readError = useCallback(async (r: Response, fallback: string) => {
    try {
      const body = await r.json()
      if (body && typeof body.error === 'string' && body.error.trim()) return body.error
    } catch {
      /* keep fallback */
    }
    return fallback
  }, [])

  const signUp = useCallback<AuthContextValue['signUp']>(
    async ({ username, email, password }) => {
      const r = await callAuth(eps.register, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ username, email, password }),
      })
      if (!r.ok) {
        return { ok: false, error: await readError(r, 'There was an issue registering your account.') }
      }
      return { ok: true }
    },
    [callAuth, eps.register, readError],
  )

  const changePassword = useCallback<AuthContextValue['changePassword']>(
    async ({ currentPassword, newPassword }) => {
      const r = await callAuth(eps.changePassword, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      })
      if (!r.ok) {
        return { ok: false, error: await readError(r, 'There was an issue changing your password.') }
      }
      await refresh()
      return { ok: true }
    },
    [callAuth, eps.changePassword, readError, refresh],
  )

  useEffect(() => {
    refresh()
  }, [refresh])

  const value = useMemo<AuthContextValue>(
    () => ({ state, mode: resolvedMode, reachable, refresh, signIn, signOut, signUp, changePassword }),
    [state, resolvedMode, reachable, refresh, signIn, signOut, signUp, changePassword],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
