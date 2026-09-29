// ONE shared poller for GET /api/llm/models/status — the Models table, the
// Metrics picker, the Workers panel rows and the Downloads queue all read the
// same snapshot, so a model/worker pair shows the same state everywhere.
//
// Poll cadence follows the payload: <= 3 s while any worker is downloading /
// loading / serving / answering, else 15 s. Feature-detected: an older central
// (404, or the SPA's index.html) marks the store `unsupported` and it re-checks
// once a minute; every consumer then falls back to its previous behaviour.
// A failed poll never destroys the last good snapshot.
import { useSyncExternalStore } from 'react'
import { hugpyFetch } from '../../runtime/config.ts'
import { VERIFY_IDLE, indexStatus, verifyReducer } from './modelStatus.js'

const STATUS_URL = '/api/llm/models/status'
const RETRY_UNSUPPORTED_MS = 60000

let snap = { index: indexStatus(null), loaded: false, unsupported: false, error: null, at: 0 }
const listeners = new Set()
let timer = null
let inflight = null
const missingConfirmations = new Map()

// A worker can briefly report a model absent while its inventory is being
// refreshed. Don't turn a previously observed cold/on-central model into
// "missing" on one contradictory snapshot; require the next complete status
// poll to confirm that transition. The status endpoint is already single-flight,
// so these are consecutive observations rather than overlapping requests.
function stabilizeMissing(payload) {
  if (!Array.isArray(payload?.models)) return payload
  const seen = new Set()
  const models = payload.models.map(row => {
    if (!row || !Array.isArray(row.workers)) return row
    const previous = snap.index.byKey.get(row.model_key)
    let changed = false
    const workers = row.workers.map(worker => {
      const name = worker?.worker || worker?.worker_id
      if (!name) return worker
      const id = `${row.model_key}\u0000${name}`
      seen.add(id)
      const base = worker.base || worker.state
      const prior = previous?.workers?.find(w =>
        (w.worker || w.worker_id) === name)
      const priorBase = prior?.base || prior?.state
      if (base !== 'missing') {
        missingConfirmations.delete(id)
        return worker
      }
      if (priorBase !== 'cold' && priorBase !== 'on central') {
        missingConfirmations.delete(id)
        return worker
      }
      const count = (missingConfirmations.get(id) || 0) + 1
      missingConfirmations.set(id, count)
      if (count >= 2) return worker
      changed = true
      return { ...worker, base: prior.base || priorBase,
        state: prior.state || priorBase,
        label: prior.label || priorBase,
        detail: `one inventory read reported missing; confirming on the next poll. Previous: ${prior.detail || priorBase}` }
    })
    return changed ? { ...row, workers } : row
  })
  for (const id of missingConfirmations.keys()) {
    if (!seen.has(id)) missingConfirmations.delete(id)
  }
  return { ...payload, models }
}

function emit(next) {
  snap = { ...snap, ...next }
  for (const fn of listeners) { try { fn() } catch { /* one bad subscriber must not stall the rest */ } }
}

async function getJson(url) {
  const r = await hugpyFetch(url)
  const text = await r.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = null }
  return { ok: r.ok, status: r.status, data }
}

function schedule(ms) {
  clearTimeout(timer)
  if (listeners.size) timer = setTimeout(poll, ms)
}

export function poll() {
  if (inflight) return inflight
  inflight = (async () => {
    let next = 15000
    try {
      const { ok, status, data } = await getJson(STATUS_URL)
      const index = indexStatus(stabilizeMissing(data))
      if (index.available) {
        emit({ index, loaded: true, unsupported: false, error: null, at: Date.now() })
        next = index.pollMs
      } else if (status === 404 || (ok && !index.available)) {
        emit({ loaded: true, unsupported: true, error: null,
               unsupportedWhy: status === 404 ? `GET ${STATUS_URL} → HTTP 404`
                 : `GET ${STATUS_URL} → HTTP ${status} with no models array in the body` })
        next = RETRY_UNSUPPORTED_MS
      } else {
        emit({ loaded: true, error: `HTTP ${status}${data?.error ? `: ${data.error}` : ''}` })
      }
    } catch (e) {
      emit({ loaded: true, error: String(e?.message || e) })
    } finally {
      inflight = null
      schedule(next)
    }
  })()
  return inflight
}

function subscribe(fn) {
  listeners.add(fn)
  if (listeners.size === 1) poll()
  return () => {
    listeners.delete(fn)
    if (!listeners.size) { clearTimeout(timer); timer = null }
  }
}

const getSnap = () => snap

/** {index, loaded, unsupported, error, at} — index per modelStatus.indexStatus. */
export function useModelStatus() {
  return useSyncExternalStore(subscribe, getSnap, getSnap)
}

/** Force a fresh read (after a verb), without waiting for the timer. */
export function refreshModelStatus() { return poll() }

/** One model with the detail block (verification log, grade rows, fit, override). */
export async function fetchModelStatusDetail(modelKey) {
  const { ok, status, data } = await getJson(`${STATUS_URL}?model=${encodeURIComponent(modelKey)}&detail=1`)
  if (!ok || !data || !Array.isArray(data.models)) {
    throw new Error(`GET model status detail for ${modelKey}: HTTP ${status}: ${data?.error || (data ? `no models array in ${JSON.stringify(data)}` : 'non-JSON or empty body')}`)
  }
  return data.models[0] || null
}

export async function fetchAdmission(modelKey) {
  const { ok, status, data } = await getJson(`/api/llm/admission/${encodeURIComponent(modelKey)}`)
  if (!ok || !data || typeof data !== 'object') {
    throw new Error(`GET admission for ${modelKey}: HTTP ${status}: ${data?.error || data?.description || (data ? JSON.stringify(data) : 'non-JSON or empty body')}`)
  }
  return data
}

export async function postAdmissionRerun(modelKey) {
  try {
    const r = await hugpyFetch(`/api/llm/admission/${encodeURIComponent(modelKey)}/rerun`, { method: 'POST' })
    const text = await r.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = null }
    return { ok: r.ok, status: r.status, job: data?.job || null,
      message: (data && (data.error || data.description || data.message)) || (!data ? text : '') }
  } catch (e) {
    return { ok: false, status: 0, job: null, message: String(e?.message || e) }
  }
}

// ---- "Verify + grade" per-model action state (module-level) ----------------
// The Models table's detail row remounts on every table render, so the action
// state cannot live in a component: one store keyed by model, one poll timer
// per active model (GET /llm/admission/<key> every 3 s until done/failed).
const vstates = new Map()
const vlisteners = new Set()
const vtimers = new Map()
let vsnap = new Map()

function vset(key, ev) {
  const prev = vstates.get(key) || VERIFY_IDLE
  const next = verifyReducer(prev, ev)
  if (next === prev) return next
  vstates.set(key, next)
  vsnap = new Map(vstates)
  for (const fn of vlisteners) { try { fn() } catch { /* keep going */ } }
  if (next.phase === 'queued' || next.phase === 'running') vschedule(key)
  if ((next.phase === 'done' || next.phase === 'failed') && prev.phase !== next.phase) poll()
  return next
}

function vschedule(key) {
  if (vtimers.has(key)) return
  vtimers.set(key, setTimeout(async () => {
    vtimers.delete(key)
    let data = null
    try { data = await fetchAdmission(key) } catch { data = null }
    if (data) vset(key, { type: 'poll', data })
    else vschedule(key)
  }, 3000))
}

function vsubscribe(fn) { vlisteners.add(fn); return () => vlisteners.delete(fn) }
const vgetSnap = () => vsnap

/** The Verify + grade state for one model (see modelStatus.verifyReducer). */
export function useVerifyState(modelKey) {
  const all = useSyncExternalStore(vsubscribe, vgetSnap, vgetSnap)
  return all.get(modelKey) || VERIFY_IDLE
}

export async function startVerify(modelKey) {
  const s = vset(modelKey, { type: 'click' })
  if (s.phase !== 'requesting') return s
  const res = await postAdmissionRerun(modelKey)
  return vset(modelKey, { type: 'posted', ...res })
}

export function resetVerify(modelKey) { return vset(modelKey, { type: 'reset' }) }
