import { useEffect, useState } from 'react'
import { fetchJson } from '../api'

// EXPLICIT priority groups (/llm/model-groups) — shared, module-cached read so
// the N worker rows and the model table don't each refetch the same registry.
// Cache is refreshed on demand (reload) and shared via subscription; a failed
// read keeps the last good list rather than blanking every consumer.

// The spellings a model key may legitimately be named by — mirrors
// comms.priority_groups.key_forms (raw, lowercase, "/"-tail, "~"-tail).
export function pgKeyForms(key) {
  const s = String(key || '').trim()
  if (!s) return []
  const out = new Set([s, s.toLowerCase()])
  const tail = s.split('/').pop()
  out.add(tail); out.add(tail.toLowerCase())
  if (s.includes('~')) {
    const base = s.split('~').slice(1).join('~')
    if (base) { out.add(base); out.add(base.toLowerCase()) }
  }
  return [...out]
}

// form -> group index over the ENABLED groups (the routing truth).
export function pgIndexOf(groups) {
  const idx = new Map()
  for (const g of groups || []) {
    if (!g.enabled) continue
    for (const m of (g.members || [])) {
      for (const f of pgKeyForms(m)) if (!idx.has(f)) idx.set(f, g)
    }
  }
  return idx
}

let _cache = null
let _inflight = null
const _subs = new Set()

function load(force = false) {
  if (_inflight) return _inflight
  if (_cache && !force) return Promise.resolve(_cache)
  _inflight = fetchJson('/api/llm/model-groups')
    .then(d => {
      _cache = d.groups || []
      _subs.forEach(fn => fn(_cache))
      return _cache
    })
    .catch(() => _cache || [])
    .finally(() => { _inflight = null })
  return _inflight
}

export default function usePriorityGroups() {
  const [groups, setGroups] = useState(_cache || [])
  useEffect(() => {
    _subs.add(setGroups)
    load().then(setGroups)
    return () => { _subs.delete(setGroups) }
  }, [])
  return { groups, reload: () => load(true) }
}
