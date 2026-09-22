import { useSyncExternalStore } from 'react'
import { resolveApiUrl, getHugpyConfig } from './config.ts'
import { fetchJson } from '../api.ts'

// ONE subscription for the console's live feeds (workers, slots, queue, jobs,
// downloads, phones, serving, peers). Central materializes them in Postgres and
// pushes only what changed over GET /api/llm/events; this store fans that out to
// every panel through useFeed(name). Before 2026-09-10 each panel ran its own
// timer against its own endpoint (six-plus requests every few seconds per tab).
//
// Rules: a feed is only replaced by a well-formed payload (a bad reply keeps the
// last good one); if the stream is down we fall back to GET /api/llm/snapshot
// every FALLBACK_MS until it comes back; the EventSource reconnects by itself.

export const EMPTY_ARRAY = Object.freeze([])
export const EMPTY_OBJECT = Object.freeze({})

const FALLBACK_MS = 20_000
const state = { feeds: {}, versions: {}, conn: 'idle', error: null, ts: 0 }
let snapshot = { ...state }
const listeners = new Set()
let es = null
let refs = 0
let fallbackTimer = null

function emit() {
  snapshot = { ...state, feeds: state.feeds, versions: state.versions }
  for (const fn of listeners) { try { fn() } catch { /* one bad subscriber must not stall the rest */ } }
}

// `liveness` is the small per-beat feed (status, last_seen, answering, VRAM free);
// the big `workers` roster is only re-pushed on structural change. Merge the
// live fields into the cached roster rows so every pill stays fresh from the
// small feed alone.
function mergeLiveness(live) {
  const roster = state.feeds.workers
  if (!Array.isArray(roster) || !Array.isArray(live)) return
  const byId = new Map(live.map(l => [l.id, l]))
  let changed = false
  const next = roster.map(w => {
    const l = byId.get(w.id)
    if (!l) return w
    const answering = new Set(l.answering || [])
    const gpus = Array.isArray(w.gpus) ? w.gpus.map((g, i) => {
      const lg = (l.gpus || [])[i]
      return lg && lg.memory_free != null ? { ...g, memory_free: lg.memory_free } : g
    }) : w.gpus
    const allocations = Array.isArray(w.allocations)
      ? w.allocations.map(a => (a && a.model_key ? { ...a, busy: answering.has(a.model_key) } : a))
      : w.allocations
    const same = w.status === l.status && w.last_seen === l.last_seen
      && JSON.stringify(w.loaded_models) === JSON.stringify(l.loaded_models)
      && JSON.stringify(w.gpus) === JSON.stringify(gpus)
      && JSON.stringify(w.allocations) === JSON.stringify(allocations)
    if (same) return w
    changed = true
    return { ...w, status: l.status, last_seen: l.last_seen, loaded_models: l.loaded_models,
             loading: l.loading, free_ram: l.free_ram ?? w.free_ram, gpus, allocations }
  })
  if (changed) state.feeds = { ...state.feeds, workers: next }
}

function accept(name, payload, version, updated) {
  if (payload === undefined || payload === null) return false
  state.feeds = { ...state.feeds, [name]: payload }
  state.versions = { ...state.versions, [name]: version }
  state.ts = updated || Date.now() / 1000
  if (name === 'liveness') mergeLiveness(payload)
  if (name === 'workers' && state.feeds.liveness) mergeLiveness(state.feeds.liveness)
  return true
}

function acceptSnapshot(d) {
  if (!d || typeof d !== 'object' || !d.feeds || typeof d.feeds !== 'object') return false
  let any = false
  for (const [name, payload] of Object.entries(d.feeds)) {
    if (accept(name, payload, d.versions?.[name], d.updated?.[name])) any = true
  }
  return any
}

function setConn(c, error = null) {
  if (state.conn === c && state.error === error) return
  state.conn = c; state.error = error
  emit()
}

async function pollSnapshot() {
  try {
    const d = await fetchJson('/api/llm/snapshot')
    if (acceptSnapshot(d)) { state.error = null; emit() }
  } catch (e) {
    setConn(state.conn, e.message)
  }
}

function startFallback() {
  if (fallbackTimer) return
  pollSnapshot()
  fallbackTimer = setInterval(pollSnapshot, FALLBACK_MS)
}

function stopFallback() {
  if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null }
}

function open() {
  if (es) return
  setConn('connecting')
  let source
  try {
    source = new EventSource(resolveApiUrl('/api/llm/events'),
      { withCredentials: getHugpyConfig().credentials === 'include' })
  } catch (e) {
    setConn('disconnected', e.message); startFallback(); return
  }
  es = source
  source.onopen = () => { stopFallback(); setConn('live') }
  source.addEventListener('snapshot', ev => {
    let d; try { d = JSON.parse(ev.data) } catch { return }
    if (acceptSnapshot(d)) emit()
  })
  source.addEventListener('feed', ev => {
    let d; try { d = JSON.parse(ev.data) } catch { return }
    if (d && d.feed && accept(d.feed, d.payload, d.version, d.updated)) emit()
  })
  source.onerror = () => {
    // EventSource retries on its own; meanwhile keep the panels fed via polling.
    setConn('reconnecting')
    startFallback()
  }
}

function close() {
  if (es) { es.close(); es = null }
  stopFallback()
  setConn('idle')
}

function subscribe(fn) {
  listeners.add(fn)
  refs++
  if (refs === 1) open()
  return () => {
    listeners.delete(fn)
    refs--
    if (refs === 0) close()
  }
}

function getSnapshot() { return snapshot }

/** The latest payload for a feed, or `fallback` (pass a STABLE constant). */
export function useFeed(name, fallback = null) {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  const v = s.feeds[name]
  return v === undefined ? fallback : v
}

/** Connection state for a status pill: idle | connecting | live | reconnecting | disconnected. */
export function useFeedConn() {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return { conn: s.conn, error: s.error, ts: s.ts }
}

/** Force a fresh read (after a verb that changes state), without waiting for the push. */
export function refreshFeeds() { return pollSnapshot() }
