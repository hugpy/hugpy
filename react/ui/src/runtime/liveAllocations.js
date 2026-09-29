// Pure helpers for runtime/feeds.js (no React, no fetch) so `node --test` can pin them.
// LOAD-STATE-LATENCY (2026-09-29): the liveness feed now carries each
// allocation's LOAD STATE (healthy/busy/loading/materialized/vram_bytes/device)
// every beat. Patch those scalars onto the cached roster rows so a seat reads
// "loading" and then "serving" (with its measured VRAM) seconds after the worker
// says so, instead of after the next ≤30 s roster rebuild. Rules:
//   • a slot row is keyed by slot_id: if the seat now holds/loads a DIFFERENT
//     model, the stale occupant row is replaced by the live (slim) row;
//   • a ram row is keyed by model_key;
//   • a live row with no roster counterpart is APPENDED (slim — the roster's
//     size/tier fields arrive with the next rebuild; vramBytesFor tolerates it);
//   • a roster slot row whose seat is no longer reported by the live feed is
//     kept as-is (the feed may be from an older worker without allocations).
const LIVE_ALLOC_KEYS = ['healthy', 'busy', 'loading', 'loading_since', 'materialized',
                         'vram_bytes', 'device', 'serving']
export function mergeAllocations(roster, live) {
  if (!Array.isArray(live) || !live.length) return roster
  const rows = Array.isArray(roster) ? roster.slice() : []
  const used = new Set()
  for (const l of live) {
    if (!l || !l.model_key) continue
    let idx = -1
    if (l.kind === 'slot' && l.slot_id != null) {
      idx = rows.findIndex(a => a && a.kind === 'slot' && a.slot_id != null
        && String(a.slot_id) === String(l.slot_id))
    }
    if (idx < 0) idx = rows.findIndex(a => a && a.model_key === l.model_key && (a.kind ?? 'ram') === (l.kind ?? 'ram'))
    if (idx >= 0 && rows[idx].model_key !== l.model_key) {
      // the seat changed hands: the live row is the truth for this slot now
      rows[idx] = { ...l }
      used.add(idx)
      continue
    }
    if (idx >= 0) {
      const a = rows[idx]
      const patch = {}
      for (const k of LIVE_ALLOC_KEYS) {
        if (l[k] !== undefined) patch[k] = l[k]
        else if (k === 'loading' && a.loading) patch.loading = false   // load ended
      }
      rows[idx] = { ...a, ...patch }
      used.add(idx)
      continue
    }
    rows.push({ ...l })
    used.add(rows.length - 1)
  }
  // a roster row that says `loading` but the live feed no longer lists at all
  // is a finished-or-refused load: clear the flag so no badge lingers.
  return rows.map((a, i) => (a && a.loading && !used.has(i) ? { ...a, loading: false } : a))
}

export function anyLoadingIn(feeds) {
  const live = feeds?.liveness
  if (Array.isArray(live)) return live.some(l => Array.isArray(l?.loading) && l.loading.length > 0)
  const roster = feeds?.workers
  return Array.isArray(roster) && roster.some(w => Array.isArray(w?.loading) && w.loading.length > 0)
}

