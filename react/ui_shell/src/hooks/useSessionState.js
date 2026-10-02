import { useCallback, useState } from 'react'

// useSessionState — useState whose value survives the SESSION (per-tab
// sessionStorage) but not a fresh visit.
//
// The console's defaults (HF search task, model-table filters/sort, panel
// expansions) are good FIRST answers, but re-imposing them on every remount
// and reload threw away the operator's in-session choices. This keeps the
// contract "defaults when fresh, your settings while you work": a new tab
// starts from the default; within the tab, the last-set value wins across
// tab-switches and reloads. localStorage is deliberately NOT used here —
// cross-session stickiness would make defaults unreachable ever again.
//
// Values round-trip through JSON; storage failures (private mode, quota)
// degrade to plain useState.
export default function useSessionState(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = sessionStorage.getItem(key)
      if (raw != null) return JSON.parse(raw)
    } catch { /* fall through to the default */ }
    return typeof initial === 'function' ? initial() : initial
  })

  const set = useCallback((next) => {
    setValue(prev => {
      const v = typeof next === 'function' ? next(prev) : next
      try { sessionStorage.setItem(key, JSON.stringify(v)) } catch { /* ignore */ }
      return v
    })
  }, [key])

  return [value, set]
}
