import { useCallback, useState } from 'react'

// useColumnLayout — persistent, user-adjustable column ORDER + WIDTHS for a data
// table, stored in localStorage (survives reloads AND fresh visits — the
// deliberate difference from useSessionState, whose sessionStorage resets on a
// new visit). A table wires it up by handing over its default column KEYS (in
// their default left-to-right order); the hook returns the live `order` array +
// a `widths` map, plus `moveColumn` (drag-reorder), `setWidth` (drag-resize),
// and `reset`.
//
// SCHEMA-DRIFT SAFE MERGE — the whole reason this isn't a plain JSON blob. A
// stored layout is reconciled against the CURRENT defaults every time it loads:
//   • keys the defaults no longer have are dropped (a removed column can't
//     resurrect from an old saved layout),
//   • keys the defaults gained since the layout was saved are appended at their
//     default position (a future column add is never left unrenderable),
//   • widths for dropped keys are pruned the same way.
// So adding/removing a column later needs no migration and no version bump —
// only an incompatible SHAPE change would (bump the vN suffix in storageKey).
//
// Storage failures (private mode, quota) degrade to in-memory state.
export default function useColumnLayout(storageKey, defaultKeys) {
  // Fold a (possibly stale/partial) stored layout onto the current defaults.
  const reconcile = useCallback((stored) => {
    const known = new Set(defaultKeys)
    const order = []
    const seen = new Set()
    // 1. keep the user's stored order, minus any keys that no longer exist
    for (const k of (stored?.order || [])) {
      if (known.has(k) && !seen.has(k)) { order.push(k); seen.add(k) }
    }
    // 2. append defaults the stored layout never carried, at their default slot
    for (const k of defaultKeys) {
      if (!seen.has(k)) { order.push(k); seen.add(k) }
    }
    // 3. keep only widths for still-known keys (sane, positive px)
    const widths = {}
    for (const [k, w] of Object.entries(stored?.widths || {})) {
      if (known.has(k) && Number.isFinite(w) && w > 0) widths[k] = w
    }
    return { order, widths }
  }, [defaultKeys])

  const [layout, setLayout] = useState(() => {
    try {
      const raw = localStorage.getItem(storageKey)
      if (raw != null) return reconcile(JSON.parse(raw))
    } catch { /* fall through to defaults */ }
    return reconcile(null)
  })

  // Every mutation writes through to localStorage in the same tick, so the
  // layout survives the panel's 10s refresh AND a reload without an effect.
  const commit = useCallback((next) => {
    try { localStorage.setItem(storageKey, JSON.stringify(next)) } catch { /* ignore */ }
    return next
  }, [storageKey])

  // Move `key` so it lands immediately before `beforeKey` (null → to the end).
  const moveColumn = useCallback((key, beforeKey) => {
    if (key === beforeKey) return
    setLayout(prev => {
      const order = prev.order.filter(k => k !== key)
      const at = beforeKey == null ? order.length : order.indexOf(beforeKey)
      order.splice(at < 0 ? order.length : at, 0, key)
      return commit({ ...prev, order })
    })
  }, [commit])

  const setWidth = useCallback((key, px) => {
    setLayout(prev => commit({ ...prev, widths: { ...prev.widths, [key]: Math.max(48, Math.round(px)) } }))
  }, [commit])

  // Reset = forget the stored layout entirely and fall back to the defaults.
  const reset = useCallback(() => {
    try { localStorage.removeItem(storageKey) } catch { /* ignore */ }
    setLayout(reconcile(null))
  }, [storageKey, reconcile])

  return { order: layout.order, widths: layout.widths, moveColumn, setWidth, reset }
}
