// ONE eviction stream for the whole console.
//
// The feed is now mounted in two places at once — the full Evictions tab and
// the compact feed under the model list in the Models tab. Each mount opening
// its own EventSource would mean two SSE connections per operator, two backfill
// GETs, and two gunicorn threads pinned per viewer, for one identical stream.
// Browsers also cap concurrent connections per origin, so "one stream per view"
// is a bill that only grows.
//
// So the stream lives HERE, at module scope, and the views subscribe. The
// connection is reference-counted: it opens when the first view mounts and
// closes when the last unmounts, which keeps the "no viewer, no thread" property
// the single-panel version had.
//
// The store keeps RAW events grouped by run_id. Folding into display rows is
// the view's job (see evictionFormat.buildRun) because the two views fold to
// different densities.

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJson } from '../../api.ts'
import { resolveApiUrl, getHugpyConfig } from '../../runtime/config.ts'
import { eventKey } from './evictionFormat.js'

const MAX_RUNS = 200          // shared cap; oldest runs drop off the front
const BACKFILL_LIMIT = 300
const RETRY_MS = 5000

let runs = []                 // oldest-first
let conn = 'idle'             // idle | connecting | live | disconnected
let error = null
let version = 0               // bumped on every change; the hooks' snapshot key

const seen = new Set()        // de-dup across backfill + SSE replay
const listeners = new Set()

let es = null
let refs = 0
let retryTimer = null
let backfilled = false

function emit() {
  version++
  for (const fn of listeners) {
    try { fn() } catch { /* one bad subscriber must not stall the rest */ }
  }
}

// Fold a batch of raw events into the run list. Runs keep first-seen order;
// events with no run_id get a synthetic per-event id so they still card up.
function ingest(batch) {
  const fresh = []
  for (const ev of batch) {
    if (!ev || typeof ev !== 'object') continue
    // The server's "you are attached" marker is a transport signal, not an
    // eviction — it must never render as a run.
    if (ev.stage === 'stream.ready') continue
    const k = eventKey(ev)
    if (seen.has(k)) continue
    seen.add(k)
    fresh.push(ev)
  }
  if (!fresh.length) return

  const next = runs.slice()
  const index = new Map(next.map((r, i) => [r.id, i]))
  for (const ev of fresh) {
    const rid = ev.run_id || `solo:${eventKey(ev)}`
    const at = index.get(rid)
    if (at == null) {
      index.set(rid, next.length)
      next.push({
        id: rid,
        worker_id: ev.worker_id,
        startTs: Number(ev.ts) || Date.now() / 1000,
        events: [ev],
      })
    } else {
      const r = next[at]
      next[at] = { ...r, worker_id: r.worker_id || ev.worker_id, events: [...r.events, ev] }
    }
  }
  runs = next.length > MAX_RUNS ? next.slice(next.length - MAX_RUNS) : next
  emit()
}

function setConn(c) {
  if (conn === c) return
  conn = c
  emit()
}

function openStream() {
  if (es) return
  setConn('connecting')
  let source
  try {
    source = new EventSource(
      resolveApiUrl('/api/llm/evictions/stream'),
      { withCredentials: getHugpyConfig().credentials === 'include' },
    )
  } catch {
    setConn('disconnected')
    return
  }
  es = source
  source.onopen = () => setConn('live')
  source.onmessage = (e) => {
    let ev
    try { ev = JSON.parse(e.data) } catch { return }
    ingest([ev])
  }
  source.onerror = () => {
    source.close()
    if (es === source) es = null
    setConn('disconnected')
    // Reconnect only while someone is still watching. The server caps a stream
    // at an hour and the client is expected to re-attach; the replay window
    // plus the de-dup set means the operator sees no gap.
    if (refs > 0) {
      clearTimeout(retryTimer)
      retryTimer = setTimeout(() => { if (refs > 0) openStream() }, RETRY_MS)
    }
  }
}

function closeStream() {
  clearTimeout(retryTimer)
  retryTimer = null
  if (es) { es.close(); es = null }
  setConn('idle')
}

// Backfill runs ONCE per page load, before the stream is trusted to be
// complete. The stream's replay window overlaps it, and the de-dup set has to
// be primed first, so this is deliberately not re-run per mount.
function backfillOnce() {
  if (backfilled) return
  backfilled = true
  fetchJson(`/api/llm/evictions?limit=${BACKFILL_LIMIT}`)
    .then(d => {
      error = null
      ingest(Array.isArray(d?.events) ? d.events : [])
      emit()
    })
    .catch(e => { error = e?.message || String(e); emit() })
}

function acquire() {
  refs++
  if (refs === 1) { backfillOnce(); openStream() }
}

function release() {
  refs = Math.max(0, refs - 1)
  if (refs === 0) closeStream()
}

/** Force a reconnect (the panel's Reconnect button). */
export function reconnect() {
  closeStream()
  if (refs > 0) openStream()
}

/**
 * Subscribe to the shared eviction stream.
 *
 * `paused` freezes THIS view's snapshot without touching the stream or the
 * other view: the store keeps ingesting, and `held` counts what arrived while
 * frozen, so resuming lands everything in order. Pause is per-view on purpose —
 * pausing the Models-tab feed to read a row must not blind the Evictions tab.
 */
export function useEvictionRuns({ paused = false } = {}) {
  const [snap, setSnap] = useState(() => ({ runs, conn, error, version }))
  const [held, setHeld] = useState(0)
  const pausedRef = useRef(paused)
  pausedRef.current = paused

  useEffect(() => {
    acquire()
    const onChange = () => {
      if (pausedRef.current) {
        // Count the runs we are choosing not to show yet. Nothing is dropped —
        // the store still holds them.
        setHeld(h => h + 1)
      } else {
        setSnap({ runs, conn, error, version })
      }
    }
    onChange()
    listeners.add(onChange)
    return () => { listeners.delete(onChange); release() }
  }, [])

  // Resuming re-syncs to whatever the store holds now.
  useEffect(() => {
    if (!paused) {
      setHeld(0)
      setSnap({ runs, conn, error, version })
    }
  }, [paused])

  return {
    runs: snap.runs,
    conn: snap.conn,
    error: snap.error,
    held,
    reconnect: useCallback(() => reconnect(), []),
  }
}

/** Test/diagnostic reset — drops everything and detaches. */
export function _resetForTests() {
  closeStream()
  runs = []
  seen.clear()
  listeners.clear()
  refs = 0
  backfilled = false
  error = null
}
