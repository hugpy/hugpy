/**
 * FLEET DISTRIBUTION MODE — the Workers panel's read + operator-write hook
 * (operator ruling 2026-09-24: "an obvious catchall that balances the friction
 * of the extreme micro that hugpy demands in its default state").
 *
 * READS   GET  /api/llm/fleet/distribution   open; {mode, source, env_override,
 *                                             stored}.
 * WRITES  POST /api/llm/fleet/distribution {"mode": "feasible"|"designated"}
 *                                             OPERATOR-GATED, server-side.
 *
 * OPTIMISTIC-THEN-REVERT, NOT DISABLED-UNTIL-PROVEN — identical rationale to
 * useModelGroups: the console has no "am I the operator" flag (the gate is
 * entirely server-side, operator_auth._SENSITIVE), and in HUGPY_AUTH_MODE=open
 * everyone IS the operator, so a client-side lock would be a lie in the common
 * dev case. So flip immediately, send, and on 401/403 snap back and show the
 * lock hint. The server stays the only authority; the UI reports what it said.
 *
 * ENV OVERRIDE: when HUGPY_DISTRIBUTION pins the effective mode, a write is still
 * stored (and reported) but does NOT change the effective mode — the hook keeps
 * ``mode`` unchanged and surfaces the env notice so the operator is never misled
 * into thinking the toggle took effect.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchJson } from '../../api.ts'
import { hugpyFetch } from '../../runtime/config.ts'

export const MODES = ['feasible', 'designated']

export const MODE_HELP = {
  feasible: 'Any online worker where the model feasibly fits is a routing '
    + 'candidate; designations become an ordered preference with a feasible-set '
    + 'fallback. The default.',
  designated: 'The legacy sealed scope: only designated / resident / granted '
    + 'homes and per-worker wildcard opt-ins serve, and an unmet preference '
    + 'refuses.',
}

const LOCK_HINT = 'Operator credential required — the server refused this write.'

export function useFleetDistribution () {
  const [state, setState] = useState({
    mode: 'feasible', source: 'default', envOverride: false, stored: null,
    loading: true, error: null,
  })
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const alive = useRef(true)

  useEffect(() => () => { alive.current = false }, [])

  const refresh = useCallback(async () => {
    try {
      const d = await fetchJson('/api/llm/fleet/distribution')
      if (!alive.current) return
      setState({
        mode: d?.mode || 'feasible',
        source: d?.source || 'default',
        envOverride: !!d?.env_override,
        stored: d?.stored ?? null,
        loading: false,
        error: null,
      })
    } catch (e) {
      if (!alive.current) return
      // A distribution endpoint that isn't there yet must not break the panel —
      // it renders the default and nothing else changes.
      setState(s => ({ ...s, loading: false, error: e?.message || String(e) }))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  /**
   * Set the fleet mode. Optimistic; reverts and explains on a server refusal.
   * Returns true when the server accepted the write.
   */
  const setMode = useCallback(async (mode) => {
    if (!MODES.includes(mode)) return false
    const before = state
    setNotice(null)
    setBusy(true)
    // Optimistic: reflect what a write does — the effective mode only moves when
    // env is NOT pinning it; the stored value always moves.
    setState(s => ({ ...s, mode: s.envOverride ? s.mode : mode, stored: mode }))
    try {
      const res = await hugpyFetch('/api/llm/fleet/distribution', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      })
      const body = await res.text()
      if (res.status === 401 || res.status === 403) throw new Error(LOCK_HINT)
      if (!res.ok) throw new Error(body.trim() || `${res.status} ${res.statusText}`)
      let d = {}
      try { d = JSON.parse(body) } catch { d = {} }
      if (!alive.current) return true
      setState({
        mode: d?.mode || mode,
        source: d?.source || 'store',
        envOverride: !!d?.env_override,
        stored: d?.stored ?? mode,
        loading: false,
        error: null,
      })
      if (d?.env_override) {
        setNotice('Stored — but HUGPY_DISTRIBUTION pins the effective mode until '
          + 'the env var is cleared.')
      }
      return true
    } catch (e) {
      if (!alive.current) return false
      setState(before)
      setNotice(e?.message || String(e))
      return false
    } finally {
      if (alive.current) setBusy(false)
    }
  }, [state])

  return {
    ...state,
    notice,
    busy,
    setMode,
    refresh,
    clearNotice: () => setNotice(null),
  }
}
