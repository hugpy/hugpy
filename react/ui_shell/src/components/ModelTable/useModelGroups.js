/**
 * MODEL GROUPS — the Models tab's read + tick-write hook.
 *
 * Spec: dev/MODEL-GROUPS-SPEC.md. Vocabulary: dev/GLOSSARY.md (model group,
 * tick, ladder walk).
 *
 * READS   GET /api/llm/groups        open; works whether or not the feature is
 *                                    enabled (the payload says which).
 * WRITES  POST /api/settings/model_groups/<group_key> {"merge": {...}}
 *                                    OPERATOR-GATED, server-side.
 *
 * WHY THE WRITE IS OPTIMISTIC-THEN-REVERT, NOT DISABLED-UNTIL-PROVEN
 *   The console has no notion of "am I the operator" — there is no is_operator
 *   flag on /me or the auth config, and the gate is entirely server-side
 *   (operator_auth._SENSITIVE). Inventing a client-side permission model to
 *   grey the switch out would be a SECOND source of truth about authority, and
 *   the wrong one: in HUGPY_AUTH_MODE=open with no token set, everyone IS the
 *   operator, so a hard-coded lock would be a lie in the common dev case.
 *
 *   So: flip immediately, send, and on 401/403 snap back and show the lock
 *   hint. The server stays the only authority; the UI just reports what it
 *   said. Request-blocked writes are BY DESIGN — a bounced toggle is the gate
 *   working, not a bug.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchJson } from '../../api.ts'
import { hugpyFetch } from '../../runtime/config.ts'

export const TICKS = ['quality', 'speed', 'priority']

/** One-line explanation per tick — the hover text on each switch. */
export const TICK_HELP = {
  quality: 'Rule out degraded variants: the 4-bit class and below. Sets the '
    + 'floor of any ladder walk.',
  speed: 'Rule out ram-only placement and spill: the chosen member must be '
    + 'fully GPU-resident.',
  priority: 'May EVICT other residents to meet the ticked standards. Without '
    + 'it, quality and speed soften to preferences. (Confers no protection on '
    + "the group's own residents.)",
}

const LOCK_HINT = 'Operator credential required — the server refused this write.'
const OFF_HINT = 'Model groups are OFF. An operator enables them before ticks '
  + 'take effect; verdicts below are advisory.'

export function useModelGroups () {
  const [state, setState] = useState({
    enabled: false, source: 'default', groups: [], loading: true, error: null,
  })
  const [notice, setNotice] = useState(null)
  const alive = useRef(true)

  useEffect(() => () => { alive.current = false }, [])

  const refresh = useCallback(async () => {
    try {
      const d = await fetchJson('/api/llm/groups')
      if (!alive.current) return
      setState({
        enabled: !!d?.enabled,
        source: d?.source || 'default',
        groups: Array.isArray(d?.groups) ? d.groups : [],
        loading: false,
        error: d?.error || null,
      })
    } catch (e) {
      if (!alive.current) return
      // A groups endpoint that isn't there yet must not break the Models tab —
      // the table renders ungrouped and nothing else changes.
      setState(s => ({ ...s, loading: false, error: e?.message || String(e) }))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  /**
   * Flip one tick. Optimistic; reverts and explains on a server refusal.
   * Returns true when the server accepted it.
   */
  const setTick = useCallback(async (groupKey, tick, next) => {
    if (!TICKS.includes(tick)) return false
    const before = state.groups
    setNotice(null)
    setState(s => ({
      ...s,
      groups: s.groups.map(g => (g.group_key === groupKey
        ? { ...g, ticks: { ...(g.ticks || {}), [tick]: next } }
        : g)),
    }))
    try {
      const res = await hugpyFetch(
        `/api/settings/model_groups/${encodeURIComponent(groupKey)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ merge: { ticks: { [tick]: next } } }),
        })
      if (res.status === 401 || res.status === 403) throw new Error(LOCK_HINT)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      // Re-read rather than trust the local flip: the verdicts change with the
      // tick, and they are the whole reason the operator flipped it.
      await refresh()
      return true
    } catch (e) {
      if (!alive.current) return false
      setState(s => ({ ...s, groups: before }))
      setNotice(e?.message || String(e))
      return false
    }
  }, [state.groups, refresh])

  /** model_key -> { group, member, verdictByWorker } for the member rows. */
  const byModel = useMemo(() => {
    const out = new Map()
    for (const g of state.groups || []) {
      for (const m of g.members || []) {
        out.set(m.model_key, { group: g, member: m })
      }
    }
    return out
  }, [state.groups])

  /** Only groups with something to choose between get a header in the table. */
  const multiMember = useMemo(
    () => (state.groups || []).filter(g => (g.members || []).length > 1),
    [state.groups])

  return {
    ...state,
    byModel,
    multiMember,
    setTick,
    notice,
    clearNotice: () => setNotice(null),
    offHint: state.enabled ? null : OFF_HINT,
    refresh,
  }
}

/**
 * The per-worker verdict chips for one member: what it does on each box, or
 * why it lost there. Pure — the backend already decided; this only formats.
 */
export function memberVerdicts (group, modelKey) {
  const out = []
  for (const [worker, v] of Object.entries(group?.verdicts || {})) {
    if (v?.preferred === modelKey) {
      out.push({
        worker,
        tone: 'ok',
        text: v.as ? `serves on ${worker} as ${v.as}` : `serves on ${worker}`,
        title: v.why || '',
      })
      continue
    }
    const ex = (v?.excluded || []).find(e => e?.model_key === modelKey)
    if (ex) {
      out.push({
        worker,
        tone: 'muted',
        text: `${worker}: ${ex.reason}`,
        title: ex.reason || '',
      })
    }
  }
  return out
}
