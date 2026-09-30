// Per-worker TEST FIRE (operator ask 2026-09-29): "a test button where it
// starts to randomly call every model on that worker which does text gen".
//
// POST /api/llm/workers/<id>/test-fire starts a background sweep on central
// (test_fire_routes.py): every text-generation model on the worker, once per
// round in a fresh random order, tiny prompt, max_tokens small, pinned to the
// worker. We poll GET .../test-fire/<job_id> every POLL_MS while it runs and
// render a compact strip: round, done/planned, ok/failed and a per-model chip
// (green ok / red fail, hover = the error; grey = skipped, hover = why).
import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJson } from '../../api'
import { hugpyFetch } from '../../runtime/config'
import { TF_FILTERS, filterTestFireRows, loadTestFireFilter, saveTestFireFilter, testFireCounts, testFireRows } from './testFireFilter'

export const TEST_FIRE_POLL_MS = 1500
const RESULT_LIMIT = 60

function base(worker) {
  return `/api/llm/workers/${encodeURIComponent(worker.id)}/test-fire`
}

// Start: raw fetch so a 409 ("already running") hands us the live job_id to
// adopt instead of surfacing as an error.
async function startTestFire(worker, opts) {
  const r = await hugpyFetch(base(worker), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts || {}),
  })
  const text = await r.text()
  let body = null
  try { body = text ? JSON.parse(text) : null } catch { body = null }
  if (r.status === 202 || r.status === 200 || r.status === 409) {
    if (body && body.job_id) return { jobId: body.job_id, adopted: r.status === 409, body }
  }
  const msg = (body && (body.error || body.message)) || text || `HTTP ${r.status}`
  const err = new Error(msg)
  err.status = r.status
  err.body = body
  throw err
}

export function useTestFire(worker) {
  const [job, setJob] = useState(null)        // last status snapshot
  const [busy, setBusy] = useState(false)     // start/stop request in flight
  const [error, setError] = useState(null)
  const [open, setOpen] = useState(true)
  const [filter, setFilterState] = useState(() => loadTestFireFilter(worker?.id))
  const setFilter = useCallback((f) => { setFilterState(f); saveTestFireFilter(worker?.id, f) }, [worker?.id])
  const activeId = useRef(null)
  const alive = useRef(true)

  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])

  const poll = useCallback(async (jobId) => {
    if (!jobId || !worker?.id) return null
    try {
      const s = await fetchJson(`${base(worker)}/${encodeURIComponent(jobId)}?limit=${RESULT_LIMIT}`)
      if (alive.current && activeId.current === jobId) { setJob(s); setError(null) }
      return s
    } catch (e) {
      if (alive.current) setError(e.message)
      return null
    }
  }, [worker?.id])   // eslint-disable-line react-hooks/exhaustive-deps

  // Poll loop: one timer per active job while it reports running.
  useEffect(() => {
    if (!job || !job.running || job.job_id !== activeId.current) return undefined
    const t = setTimeout(() => { poll(job.job_id) }, TEST_FIRE_POLL_MS)
    return () => clearTimeout(t)
  }, [job, poll])

  const start = useCallback(async (opts) => {
    if (!worker?.id) return
    setBusy(true); setError(null)
    try {
      const { jobId } = await startTestFire(worker, opts || { rounds: 1, concurrency: 1, max_tokens: 32 })
      activeId.current = jobId
      setOpen(true)
      await poll(jobId)
    } catch (e) {
      setError(e.message)
    } finally {
      if (alive.current) setBusy(false)
    }
  }, [worker, poll])

  const stop = useCallback(async () => {
    const jobId = activeId.current
    if (!jobId || !worker?.id) return
    setBusy(true)
    try {
      await fetchJson(`${base(worker)}/${encodeURIComponent(jobId)}/stop`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      })
      await poll(jobId)
    } catch (e) {
      setError(e.message)
    } finally {
      if (alive.current) setBusy(false)
    }
  }, [worker, poll])

  const dismiss = useCallback(() => { activeId.current = null; setJob(null); setError(null) }, [])

  return { job, busy, error, open, setOpen, start, stop, dismiss, filter, setFilter }
}

export function TestFireButton({ tf, disabled = false }) {
  const running = !!(tf.job && tf.job.running)
  if (running) {
    return (
      <button className="wp-test-fire wp-test-fire-stop" title="Stop the test fire after the in-flight call finishes"
              onClick={tf.stop} disabled={tf.busy}>
        ■ Stop test
      </button>
    )
  }
  return (
    <button className="wp-test-fire"
            title="Test fire: randomly call every text-generation model on this worker with a tiny prompt (one round, one call at a time, max 32 tokens)"
            onClick={() => tf.start({ rounds: 1, concurrency: 1, max_tokens: 32 })}
            disabled={disabled || tf.busy}>
      {tf.busy ? '⟳ …' : '⟳ Test fire'}
    </button>
  )
}

function fmtErr(s) {
  if (!s) return ''
  const t = String(s)
  return t.length > 600 ? t.slice(0, 600) + '…' : t
}

function chipTitle(key, pm) {
  if (!pm) return key
  const lines = [key, `ok ${pm.ok} · failed ${pm.failed}`]
  if (pm.last_latency_s != null) lines.push(`last ${pm.last_latency_s}s${pm.last_tok_s != null ? ` · ${pm.last_tok_s} tok/s` : ''}`)
  if (pm.last_status === 'fail') lines.push(`${pm.last_error_kind || 'error'}: ${fmtErr(pm.last_error)}`)
  return lines.join('\n')
}

export function TestFireStrip({ tf }) {
  const { job, error } = tf
  if (!job && !error) return null
  const rows = testFireRows(job)
  const counts = testFireCounts(rows)
  const filter = tf.filter || 'all'
  const shown = filterTestFireRows(rows, filter)
  const skipped = (job && job.skipped) || []
  const planned = job && job.total_planned != null ? job.total_planned : '∞'
  const state = !job ? 'error' : job.running ? 'running' : (job.stop_reason === 'complete' ? 'done' : `stopped (${job.stop_reason || 'stopped'})`)
  const last = job && job.results && job.results.length ? job.results[job.results.length - 1] : null
  return (
    <div className={`wp-tf-strip${job && job.running ? ' wp-tf-running' : ''}`}>
      <div className="wp-tf-head">
        <span className="wp-tf-title">⟳ test fire</span>
        <span className={`wp-tf-state wp-tf-state-${job && job.running ? 'running' : 'idle'}`}>{state}</span>
        {job && (
          <>
            <span className="wp-tf-stat" title="round (rounds=0 runs until stopped)">round {job.round}{job.rounds ? `/${job.rounds}` : ''}</span>
            <span className="wp-tf-stat" title="calls done / planned">{job.done_calls}/{planned}</span>
            <span className="wp-tf-stat wp-tf-ok" title="calls that returned text">✓ {job.summary.ok}</span>
            <span className="wp-tf-stat wp-tf-fail" title="calls that errored (hover a red chip for the reason)">✗ {job.summary.failed}</span>
            {skipped.length > 0 && (
              <span className="wp-tf-stat wp-tf-skip"
                    title={skipped.map(s => `${s.model_key}: ${s.reason}`).join('\n')}>
                skipped {skipped.length}
              </span>
            )}
            {job.in_flight && job.in_flight.length > 0 && (
              <span className="wp-tf-stat wp-tf-inflight" title="in flight now">
                → {job.in_flight.map(f => f.model_key).join(', ')}
              </span>
            )}
          </>
        )}
        {error && <span className="wp-tf-err" title={error}>⚠ {fmtErr(error)}</span>}
        <span className="wp-tf-spacer" />
        {job && (
          <button className="wp-tf-toggle" onClick={() => tf.setOpen(!tf.open)} title={tf.open ? 'hide per-model results' : 'show per-model results'}>
            {tf.open ? '▾' : '▸'}
          </button>
        )}
        {job && !job.running && (
          <button className="wp-tf-toggle" onClick={tf.dismiss} title="dismiss">✕</button>
        )}
      </div>
      {job && tf.open && (
        <div className="wp-tf-models">
          <div className="wp-tf-filters" role="group" aria-label="filter test-fire results">
            {TF_FILTERS.map(f => (
              <button key={f.id} type="button"
                      className={`wp-tf-filter wp-tf-filter-${f.id}${filter === f.id ? ' wp-tf-filter-on' : ''}`}
                      aria-pressed={filter === f.id}
                      onClick={() => tf.setFilter && tf.setFilter(f.id)}>
                {f.label} <em>{counts[f.id]}</em>
              </button>
            ))}
          </div>
          {shown.map(({ key, pm, status, skip }) => {
            if (status === 'skipped') {
              return (
                <span key={`skip:${key}`} className="wp-tf-chip wp-tf-chip-skip" title={`${key}\nskipped: ${skip.reason}`}>
                  ⊘ {key} <em className="wp-tf-skip-why">{skip.reason}</em>
                </span>
              )
            }
            const cls = status === 'passed' ? 'wp-tf-chip-ok' : status === 'failed' ? 'wp-tf-chip-fail' : 'wp-tf-chip-pending'
            const inflight = job.in_flight && job.in_flight.some(f => f.model_key === key)
            return (
              <span key={key} className={`wp-tf-chip ${cls}${inflight ? ' wp-tf-chip-live' : ''}`} title={chipTitle(key, pm)}>
                {status === 'pending' ? '○' : '●'} {key}
                {pm && pm.last_tok_s != null && status === 'passed' && <em> {pm.last_tok_s} t/s</em>}
                {pm && pm.failed > 0 && pm.ok > 0 && <em> {pm.ok}/{pm.ok + pm.failed}</em>}
              </span>
            )
          })}
          {last && (
            <div className="wp-tf-last" title={last.error ? fmtErr(last.error) : (last.content80 || '')}>
              last: {last.model_key} · {last.ok ? 'ok' : (last.error_kind || 'fail')} · {last.latency_s}s
              {last.tok_s != null ? ` · ${last.tok_s} tok/s` : ''}
              {last.ok && last.content80 ? ` · “${last.content80}”` : ''}
              {!last.ok && last.error ? ` · ${fmtErr(last.error).slice(0, 160)}` : ''}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
