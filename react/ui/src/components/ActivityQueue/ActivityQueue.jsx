import { useEffect, useState, useRef } from 'react'
import { fetchJson } from '../../api'
import { EMPTY_ARRAY, useFeed } from '../../runtime/feeds'
import { WorkerPulls } from '../ModelTable/StatusCells'
import { workerProvisioning } from '../ModelTable/modelStatus'
import '../ModelTable/ModelStatus.css'
import './ActivityQueue.css'

// Live in-flight GENERATION across ALL transports (CON-01). Polls GET
// /api/llm/jobs (~1.5s) — the unified F5 job store (web / v1 / discord / cli) —
// and shows a compact chip in the topbar with per-transport counts;
// hover/click expands the per-job list (state, transport, model, worker,
// principal, elapsed, tokens). Model DOWNLOADS (kind:'download') are filtered
// out — they belong to the DownloadsQueue, not this activity chip.
const LIVE = new Set(['pending', 'processing', 'streaming'])

export default function ActivityQueue() {
  const [jobs, setJobs]   = useState([])
  const [open, setOpen]   = useState(false)
  const timer = useRef(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        // Union of the two live views: /llm/jobs (F5 store — downloads,
        // discord, cli, attributed transport/principal/worker) and the legacy
        // /llm/queue (which is where in-flight CHAT streams actually appear —
        // web chats don't register in the F5 store yet; see CON-01 note).
        const [jr, qr] = await Promise.all([
          fetchJson('/api/llm/jobs').catch(() => null),
          fetchJson('/api/llm/queue').catch(() => null),
        ])
        if (!alive) return
        const fromJobs = (Array.isArray(jr?.jobs) ? jr.jobs : [])
          .filter(j => j.kind !== 'download' && LIVE.has(j.status))
        const seen = new Set(fromJobs.map(j => j.id))
        const fromQueue = (Array.isArray(qr?.active) ? qr.active : [])
          .filter(e => !seen.has(e.request_id))
          .map(e => ({
            id: e.request_id,
            status: e.state === 'active' ? 'streaming' : 'pending',
            transport: null,
            kind: e.kind || 'chat',
            model: e.model || e.model_key,
            worker: null, principal: null,
            elapsed: e.elapsed, tokens: e.tokens,
          }))
        setJobs([...fromJobs, ...fromQueue])
      } catch { /* transient — keep last good */ }
    }
    load()
    timer.current = setInterval(load, 1500)
    return () => { alive = false; clearInterval(timer.current) }
  }, [])

  // Worker pulls from central are activity too (heartbeat provisioning).
  const fWorkers = useFeed('workers', EMPTY_ARRAY)
  const pulls = workerProvisioning(fWorkers)
  const busy = jobs.length > 0 || pulls.length > 0
  const byTransport = jobs.reduce((acc, j) => {
    const t = j.transport || j.kind || '?'
    acc[t] = (acc[t] || 0) + 1
    return acc
  }, {})
  const transportSummary = Object.entries(byTransport)
    .sort((a, b) => b[1] - a[1])
    .map(([t, n]) => `${t} ${n}`)
    .join(' · ')

  const stateIcon = s => (s === 'streaming' ? '▶' : s === 'processing' ? '⚙' : '⏳')

  // WHERE a media/video job physically executes — the placement object the media-bus
  // bridge stamps onto /llm/jobs rows ({source,host,worker_id,gpu,process,
  // reserved_bytes}). Rendered as a compact "host · gpu · process" badge ONLY when a
  // row carries it; absent (every non-media job) → nothing, so no layout changes for
  // chat/download rows. "external" = runs off the central box (ae worker / ComfyUI /
  // identity-render).
  const placementLine = p => {
    if (!p) return ''
    return [p.host, p.gpu || 'cpu', p.process].filter(Boolean).join(' · ')
  }
  const isExternal = p => !!p && !!p.host && p.host !== 'central'

  return (
    <div
      className={`activity-queue ${busy ? 'is-busy' : 'is-idle'}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        className="aq-chip"
        onClick={() => setOpen(o => !o)}
        title="Live generation across every transport (web, /v1, discord, cli)"
      >
        <span className={`dot ${busy ? 'dot-amber' : 'dot-dim'}`} />
        {busy ? [jobs.length && `${jobs.length} live · ${transportSummary}`, pulls.length && `${pulls.length} pulling`].filter(Boolean).join(' · ') : 'queue idle'}
      </button>

      {open && busy && (
        <div className="aq-pop" role="status">
          {pulls.length > 0 && <><div className="aq-head">Worker pulls from central · {pulls.length}</div>
            <div className="aq-row"><WorkerPulls workers={fWorkers} /></div></>}
          <div className="aq-head">In-flight jobs · {jobs.length}</div>
          {jobs.map(j => (
            <div key={j.id} className={`aq-row aq-${j.status}`}>
              <span className="aq-state" title={j.status}>
                {stateIcon(j.status)} {j.status}
              </span>
              <span className="aq-transport" title="transport">{j.transport || j.kind}</span>
              <span className="aq-model" title={j.model}>{j.model}</span>
              <span className="aq-meta">
                {j.worker ? `@${j.worker} · ` : ''}
                {j.principal ? `${j.principal} · ` : ''}
                {j.elapsed}s
                {j.tokens ? ` · ${j.tokens} tok` : ''}
              </span>
              {j.placement && placementLine(j.placement) ? (
                <span
                  className={`aq-placement${isExternal(j.placement) ? ' aq-placement-external' : ''}`}
                  title={isExternal(j.placement) ? 'Runs off the central box (external worker/service)' : 'Execution locus'}
                >
                  {isExternal(j.placement) ? '⇄ ' : ''}{placementLine(j.placement)}
                  {isExternal(j.placement) ? ' · external' : ''}
                </span>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
