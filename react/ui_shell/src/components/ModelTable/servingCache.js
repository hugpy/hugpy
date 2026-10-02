import { hugpyFetch } from '../../runtime/config.ts'

// ONE fetch per model for GET /api/llm/serving/<key>, shared by every control
// that needs it (ServingControl, QuantControl, PlacementControl, WorkerRow's
// alloc menu). Before this each control fetched on its own, and an expanded
// row hit central with a burst of identical serving GETs on every re-render
// (15 in a row was observed on 2026-09-10) — each one makes central compute
// placement feasibility and read a GGUF header. Concurrent callers share the
// in-flight promise; results live for TTL_MS unless a POST to the same key
// invalidates them.

const TTL_MS = 20_000
const cache = new Map()      // key -> { at, data }
const inflight = new Map()   // key -> Promise

export async function getServing(modelKey, { force = false } = {}) {
  const now = Date.now()
  const hit = cache.get(modelKey)
  if (!force && hit && now - hit.at < TTL_MS) return hit.data
  if (!force && inflight.has(modelKey)) return inflight.get(modelKey)
  const p = (async () => {
    const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`)
    if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`)
    const data = await r.json()
    cache.set(modelKey, { at: Date.now(), data })
    return data
  })()
  inflight.set(modelKey, p)
  try { return await p } finally { inflight.delete(modelKey) }
}

export function invalidateServing(modelKey) {
  if (modelKey == null) cache.clear()
  else cache.delete(modelKey)
}

export function primeServing(modelKey, data) {
  cache.set(modelKey, { at: Date.now(), data })
}

// Multi-GPU SHARD-eligibility flag (GET /settings/shard_models/<key>), same
// shared/deduped cache. PlacementControl used to fetch it in a mount effect,
// and the whole expanded row REMOUNTED on every table render (ModelDetail was a
// component defined inside ModelTable's body — a new type each render), so
// every status poll re-issued this GET per open row: the flicker. The flag only
// changes when the operator toggles it here, which primes the cache.
const shardCache = new Map()     // key -> { at, on }
const shardInflight = new Map()  // key -> Promise<boolean>

export async function getShardFlag(modelKey, { force = false } = {}) {
  const hit = shardCache.get(modelKey)
  if (!force && hit && Date.now() - hit.at < TTL_MS) return hit.on
  if (!force && shardInflight.has(modelKey)) return shardInflight.get(modelKey)
  const p = (async () => {
    const r = await hugpyFetch(`/settings/shard_models/${encodeURIComponent(modelKey)}`)
    if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`)
    const d = await r.json()
    const on = !!(d && d.value)
    shardCache.set(modelKey, { at: Date.now(), on })
    return on
  })()
  shardInflight.set(modelKey, p)
  try { return await p } finally { shardInflight.delete(modelKey) }
}

export function primeShardFlag(modelKey, on) {
  shardCache.set(modelKey, { at: Date.now(), on: !!on })
}
