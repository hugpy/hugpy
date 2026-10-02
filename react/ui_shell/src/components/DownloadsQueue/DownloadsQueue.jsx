import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchJson } from '../../api'
import { EMPTY_ARRAY, useFeed } from '../../runtime/feeds'
import { WorkerPulls } from '../ModelTable/StatusCells'
import { workerProvisioning } from '../ModelTable/modelStatus'
import '../ModelTable/ModelStatus.css'
import './DownloadsQueue.css'

// Model DOWNLOADS — their OWN queue, kept strictly separate from the inference
// "Queue" tile (which counts models being CALLED). Polls GET /api/jobs (~2.5s),
// the download manager's own store (legacy job shape). Renders a StatusBar tile
// ("Downloads · N downloading") that expands into a per-job popover. Each job is
// its own row (keyed by job.id) so several quant variants of one repo pulling at
// once each show individually. The "N downloading" count is ACTIVE only
// (queued/running) — a finished/installed download NEVER shows as downloading.
//
// 2026-08-13: also renders as a PANE (`variant="panel"`) — the "⬇ Queue" tab
// inside Add models. Same server-truth poll and verb wiring; only the shell
// differs (always-visible block instead of a hover popover).
//
// k121 (operator ruling 2026-08-20): FAILURE RECORDS ARE PERMANENT AND IN
// PLACE. failed/expired rows persist server-side — reason, guidance and keeper
// diagnosis attached — until an admin explicitly discards them (✕ now calls
// POST /jobs/<id>/discard; the old client-side "clear" illusion is gone).
// `expired` (the stalled-queue outcome) is rendered, not silently dropped.
// Every failed/expired row without a diagnosis triggers ONE keeper inference
// (POST /jobs/<id>/diagnose) and pins the answer to the record.

function fmtBytes(n) {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}

const ACTIVE = new Set(['queued', 'running'])
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'expired'])
const FAILED = new Set(['failed', 'expired'])

export default function DownloadsQueue({ variant = 'tile' }) {
  const [jobs, setJobs] = useState([])
  const [open, setOpen] = useState(false)
  const alive = useRef(true)
  const diagnosing = useRef(new Set())   // job ids with a diagnose in flight / done this session

  // Re-usable loader so verbs can optimistically refetch right after their
  // POST, instead of waiting up to a full poll interval.
  const load = useCallback(async () => {
    try {
      const d = await fetchJson('/api/jobs')
      if (!alive.current) return
      setJobs(Array.isArray(d) ? d : [])
    } catch { /* transient — keep last good */ }
  }, [])

  // Fed by the one live subscription (runtime/feeds.js); `load()` stays for the
  // optimistic refetch right after a verb.
  const fDownloads = useFeed('downloads', null)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => { if (Array.isArray(fDownloads)) setJobs(fDownloads) }, [fDownloads])

  // THE INFERENCE TRIGGER: one keeper diagnosis per undiagnosed failure record,
  // at most one in flight (a diagnosis is real inference, not a decoration).
  useEffect(() => {
    const candidate = jobs.find(j =>
      FAILED.has(j.status) && !j.diagnosis && !diagnosing.current.has(j.id))
    if (!candidate) return
    diagnosing.current.add(candidate.id)
    fetchJson(`/api/jobs/${candidate.id}/diagnose`, { method: 'POST' })
      .then(load)
      .catch(() => { /* unauthorized/offline — the button remains */ })
  }, [jobs, load])

  // Cancel works for ANY non-terminal job (queued OR running).
  const cancel = useCallback((id) => {
    fetchJson(`/api/jobs/${id}/cancel`, { method: 'POST' }).then(load).catch(() => {})
  }, [load])

  // Retry resumes a failed/cancelled/expired job from its partial files.
  const retry = useCallback((id) => {
    fetchJson(`/api/jobs/${id}/retry`, { method: 'POST' }).then(load).catch(() => {})
  }, [load])

  // ADMIN DISCARD — the only way a failure record leaves the queue. Server-side
  // and permanent; a 401 means the caller isn't entitled to dismiss it.
  const discard = useCallback((id) => {
    fetchJson(`/api/jobs/${id}/discard`, { method: 'POST' })
      .then(load)
      .catch(e => alert(`Discard refused: ${e.message || e}`))
  }, [load])

  const diagnose = useCallback((id) => {
    diagnosing.current.add(id)
    fetchJson(`/api/jobs/${id}/diagnose`, { method: 'POST' })
      .then(load)
      .catch(e => alert(`Diagnose failed: ${e.message || e}`))
  }, [load])

  const active = jobs.filter(j => ACTIVE.has(j.status))
  const done = jobs.filter(j => TERMINAL.has(j.status))
  // Worker-side pulls from central (heartbeat provisioning/provision_progress):
  // a worker copying weights is a download too, and the operator must see it.
  const fWorkers = useFeed('workers', EMPTY_ARRAY)
  const pulls = workerProvisioning(fWorkers)

  const busy = active.length > 0 || pulls.length > 0
  const failedN = done.filter(j => FAILED.has(j.status)).length
  const hasPop = active.length > 0 || done.length > 0 || pulls.length > 0

  const rows = (
    <>
      {pulls.length > 0 && (
        <>
          <div className="dlq-head">Worker pulls from central · {pulls.length}</div>
          <div className="dlq-row"><WorkerPulls workers={fWorkers} /></div>
        </>
      )}
      {active.length > 0 && (
        <>
          <div className="dlq-head">Downloading · {active.length}</div>
          {active.map(j => <ActiveRow key={j.id} job={j} onCancel={cancel} />)}
        </>
      )}
      {done.length > 0 && (
        <>
          <div className="dlq-head dlq-head-muted">
            Recent{failedN > 0 ? ` · ${failedN} failed (kept until discarded)` : ''}
          </div>
          {done.map(j => (
            <DoneRow key={j.id} job={j} onRetry={retry} onDiscard={discard}
                     onDiagnose={diagnose} />
          ))}
        </>
      )}
    </>
  )

  if (variant === 'panel') {
    return (
      <div className="dlq-panel">
        {!hasPop && (
          <div className="dlq-panel-empty">
            No model downloads right now. Every download shows here — including
            ones started from other sessions or before a reload. Failed or
            expired downloads stay here, with the reason and a keeper
            diagnosis, until an admin discards them.
          </div>
        )}
        {rows}
      </div>
    )
  }

  return (
    <div
      className={`sb-seg dlq ${busy ? 'dlq-busy' : ''}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span className="sb-k">Downloads</span>
      <button
        type="button"
        className="dlq-toggle"
        onClick={() => setOpen(o => !o)}
        title={busy
          ? `${active.length} model download${active.length > 1 ? 's' : ''} in progress — separate from the inference queue`
          : 'Model downloads — separate from the inference queue'}
      >
        {busy
          ? <span className="sb-v dlq-count">{[active.length && `${active.length} downloading`, pulls.length && `${pulls.length} pulling`].filter(Boolean).join(' · ')}</span>
          : failedN > 0
            ? <span className="sb-v dlq-count">{failedN} failed</span>
            : <span className="sb-v sb-zero">idle</span>}
      </button>

      {open && hasPop && (
        <div className="dlq-pop" role="status">
          {rows}
        </div>
      )}
    </div>
  )
}

// One in-flight download — repo/model + a live progress bar + %/bytes/speed and
// a ✕ cancel (valid for queued OR running). Mirrors ModelTable's DownloadProgress
// / HFSearch's DownloadBar visual vocabulary. A queued row shows the server's
// `message` verbatim — that is where "the downloader service is not running"
// arrives, and hiding it behind a cheerful "queued…" was the old lie.
function ActiveRow({ job, onCancel }) {
  const pct = Math.round((job.progress ?? 0) * 100)
  const indeterminate = job.status === 'running' && !job.total_bytes
  const bps = job.bytes_per_second

  return (
    <div className="dlq-row dlq-row-active">
      <span className="dlq-name" title={job.model_key}>{job.model_key}</span>
      <div className={`dlq-bar ${indeterminate ? 'dlq-bar-indet' : ''} ${job.stalled ? 'dlq-bar-stalled' : ''} dlq-bar-${job.status}`}>
        <div className="dlq-bar-fill" style={{ width: indeterminate ? '40%' : `${pct}%` }} />
      </div>
      <span className="dlq-label" title={job.error || job.message || ''}>
        {job.status === 'queued' && (job.message || 'queued…')}
        {job.status === 'running' && (job.stalled
          ? '⚠ stalled — resuming…'
          : indeterminate
            ? `downloading… ${fmtBytes(job.downloaded_bytes)}`
            : `${pct}% · ${fmtBytes(job.downloaded_bytes)} / ${fmtBytes(job.total_bytes)}`)}
        {job.status === 'running' && job.attempt > 1 && ` · try ${job.attempt}/${job.max_attempts}`}
        {job.status === 'running' && !job.stalled && bps > 0 && ` · ${fmtBytes(bps)}/s`}
      </span>
      <button className="dlq-btn dlq-cancel" onClick={() => onCancel(job.id)} title="Cancel download">
        ✕
      </button>
    </div>
  )
}

// A finished/terminal download. completed → ✓ installed (✕ discards);
// failed/expired → THE FAILURE RECORD, in place: typed reason, full error,
// server guidance, keeper diagnosis, ↻ retry, 🩺 diagnose, ✕ admin discard.
function DoneRow({ job, onRetry, onDiscard, onDiagnose }) {
  const failed = FAILED.has(job.status)
  const retryable = failed || job.status === 'cancelled'

  return (
    <div className={`dlq-row dlq-row-done dlq-${job.status}`}>
      <div className="dlq-row-main">
        <span className="dlq-name" title={job.model_key}>{job.model_key}</span>
        <span className="dlq-label" title={job.error || job.message || ''}>
          {job.status === 'completed' && '✓ installed'}
          {job.status === 'failed' &&
            `✗ failed${job.error_reason ? ` [${job.error_reason}]` : ''}`}
          {job.status === 'expired' && '✗ expired — never ran'}
          {job.status === 'cancelled' && 'cancelled'}
        </span>
        {retryable && (
          <button className="dlq-btn dlq-retry" onClick={() => onRetry(job.id)}
                  title="Resume from where it stopped">
            ↻
          </button>
        )}
        {failed && !job.diagnosis && (
          <button className="dlq-btn dlq-diagnose" onClick={() => onDiagnose(job.id)}
                  title="Ask the hugpy keeper to diagnose this failure">
            🩺
          </button>
        )}
        <button className="dlq-btn dlq-clear" onClick={() => onDiscard(job.id)}
                title={failed
                  ? 'Discard this failure record (admin — removes it for everyone)'
                  : 'Discard this row'}>
          ✕
        </button>
      </div>
      {failed && (job.error || job.message) && (
        <div className="dlq-detail">
          {job.error && <div className="dlq-detail-error">{job.error}</div>}
          {job.message && job.message !== job.error && (
            <div className="dlq-detail-msg">{job.message}</div>
          )}
        </div>
      )}
      {failed && job.diagnosis && (
        <div className="dlq-detail dlq-diagnosis"
             title="Keeper diagnosis — pinned to this record">
          🩺 {job.diagnosis}
        </div>
      )}
    </div>
  )
}
