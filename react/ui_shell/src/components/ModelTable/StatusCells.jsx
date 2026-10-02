// Cells + actions for the "worth my time" columns (data: GET /llm/models/status
// via useModelStatus; views: modelStatus.js). Every cell renders a labelled
// state — a missing fact is shown as missing, with why and what would fix it.
import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import { DiagnosticsView, LogBlock, RawLink } from '../Diagnostics/Diagnostics'
import {
  admissionView, failureView, fmtAgo, gradeView, provisionLine, servableView, throughputView, verificationView,
  verifyActive, verifyLabel, workerChips, workerProvisioning, worthView,
} from './modelStatus'
import { resetVerify, startVerify, useModelStatus, useVerifyState } from './useModelStatus'

export function Chip({ tone = 'muted', title, children, className = '' }) {
  return <span className={`ms-chip ms-${tone} ${className}`} title={title}>{children}</span>
}

export function WorthCell({ row }) {
  const v = worthView(row)
  return <div className="ms-cell">
    <Chip tone={v.tone} title={v.title} className="ms-worth">{v.text}</Chip>
    {v.why && <span className="ms-why" title={v.title}>{v.why}</span>}
  </div>
}

export function VerificationCell({ row }) {
  const v = verificationView(row)
  return <details className="ms-cell ms-expand">
    <summary><Chip tone={v.tone} title={v.title}>{v.text}</Chip></summary>
    <div className="ms-pop">{v.title}
      {v.log && <LogBlock text={v.log.text} source={v.log.source} logRef={v.log.logRef} bytes={v.log.bytes} />}</div>
  </details>
}

export function AdmissionCell({ row }) {
  const v = admissionView(row)
  return <div className="ms-cell">
    <Chip tone={v.tone} title={v.title}>{v.text}</Chip>
    <span className="ms-why" title={v.title}>{row?.admission?.reason || ''}</span>
  </div>
}

export function GradeCell({ row }) {
  const v = gradeView(row)
  return <div className="ms-cell" title={v.title}>
    <Chip tone={v.tone}>{v.text}</Chip>
    {v.sub && <span className="ms-sub">{v.sub}</span>}
  </div>
}

export function ThroughputCell({ row }) {
  const v = throughputView(row)
  return <span className={`ms-cell${v.none ? ' ms-none' : ''}`} title={v.title}>{v.text}</span>
}

export function FailureCell({ row }) {
  const v = failureView(row)
  if (v.none) return <span className="ms-cell ms-none" title={v.title}>{v.text}</span>
  return <details className="ms-cell ms-expand">
    <summary title={v.title}><span className="ms-fail-line">{v.text}</span></summary>
    <div className="ms-pop">{v.title}
      {v.log && <LogBlock text={v.log.text} source={v.log.source} logRef={v.log.logRef} bytes={v.log.bytes} />}</div>
  </details>
}

export function ServableCell({ row }) {
  const v = servableView(row)
  return <div className="ms-cell">
    <Chip tone={v.tone} title={v.title}>{v.text}</Chip>
    <span className="ms-why" title={v.title}>{row?.servable?.reason || ''}</span>
  </div>
}

// One chip per worker, the shared vocabulary (missing / cold / downloading from
// central / loading / hot / serving / answering / failed: <class>, + held).
export function WorkerStateChips({ row, only }) {
  if (!row) return <span className="ms-none">no status for this model</span>
  const chips = workerChips(row).filter(c => !only || c.worker === only)
  if (!chips.length) return <span className="ms-none">no workers registered</span>
  return <span className="ms-wchips">{chips.map(c =>
    <span key={c.worker} className={`ms-wchip ms-${c.tone}${c.online ? '' : ' ms-offline'}`} title={c.title}>
      {!only && <b>{c.worker}</b>} {c.icon} {c.label}
    </span>)}</span>
}

// "Verify + grade": POST /llm/admission/<key>/rerun (the admission runner does
// the audit + the benchmark for this one model), then poll
// /llm/admission/<key> every 3 s until the job finishes. State lives in the
// shared store (useModelStatus.startVerify) so a re-render never loses it.
export function VerifyGradeButton({ modelKey }) {
  const state = useVerifyState(modelKey)
  const title = state.message || 'Run the integrity audit and the aptitude benchmark for this model now (operator)'
  return <span className="ms-verify">
    <button type="button" className={`mt-act ms-verify-btn ms-v-${state.phase}`} disabled={verifyActive(state)}
      onClick={() => (state.phase === 'idle' ? startVerify(modelKey) : resetVerify(modelKey))} title={title}>
      {state.phase === 'idle' ? '🔬 Verify + grade' : verifyLabel(state)}
    </button>
    {state.phase !== 'idle' && state.message && <span className="ms-verify-msg" title={state.message}>{state.message}</span>}
  </span>
}

// Diagnostics for the last routing refusal (its stored structured record).
export function RefusalDiagnostics({ row }) {
  const [open, setOpen] = useState(false)
  const [diag, setDiag] = useState(null)
  const rid = row?.last_failure?.kind === 'routing refusal' ? row.last_failure.request_id : null
  useEffect(() => {
    if (!open || !rid || diag) return
    fetchJson(`/api/llm/diagnostics/${encodeURIComponent(rid)}`).then(setDiag).catch(e => setDiag({ error: String(e?.message || e) }))
  }, [open, rid, diag])
  if (!rid) {
    return <button type="button" className="mt-act" disabled title="no routing refusal on record for this model">🩺 Diagnostics</button>
  }
  return <>
    <button type="button" className={`mt-act${open ? ' mt-act-on' : ''}`} onClick={() => setOpen(o => !o)}
      title={`stored routing-refusal record ${rid}`}>🩺 Diagnostics</button>
    {open && <div className="ms-diag"><RawLink logRef={`request:${rid}`} />{diag ? <DiagnosticsView value={diag} /> : 'loading…'}</div>}
  </>
}

// Worker-side pulls from central ("aeb ← central: <model> 12.4/21.6 GB (57%)"),
// live off the workers feed (provisioning + provision_progress), with the
// elapsed time from the status snapshot's provision.start when known.
export function listWorkerPulls(workers, index) {
  const rows = workerProvisioning(workers)
  const seen = new Set(rows.map(r => `${r.worker}\0${r.model}`))
  if (index?.available) {
    for (const [mk, row] of index.byKey) {
      for (const ws of row.workers || []) {
        if (ws.base !== 'downloading from central') continue
        const k = `${ws.worker}\0${mk}`
        const hit = rows.find(r => `${r.worker}\0${r.model}` === k || (r.worker === ws.worker && mk.endsWith(r.model)))
        if (hit) {
          hit.since = ws.progress?.since || null
          // central's own transfer ledger is authoritative over the heartbeat
          if (ws.progress?.source === 'central-ledger') {
            hit.progress = ws.progress
            hit.line = provisionLine(ws.worker, hit.model, ws.progress)
          }
          continue
        }
        if (seen.has(k)) continue
        seen.add(k)
        rows.push({ worker: ws.worker, model: mk, progress: ws.progress || {}, since: ws.progress?.since || null,
          line: provisionLine(ws.worker, mk, ws.progress || {}) })
      }
    }
  }
  return rows
}

export function WorkerPulls({ workers }) {
  const { index } = useModelStatus()
  const rows = listWorkerPulls(workers, index)
  if (!rows.length) return null
  return <>{rows.map(r => {
    const frac = r.progress?.frac ?? (r.progress?.total_bytes ? (r.progress.done_bytes || 0) / r.progress.total_bytes : null)
    return <div key={`${r.worker}:${r.model}`} className="ms-prov" title={`${r.line}${r.progress?.source ? ` · source ${r.progress.source}` : ''}`}>
      <span className="ms-prov-bar"><i style={{ width: frac != null ? `${Math.round(frac * 100)}%` : '30%' }} /></span>
      <span>{r.line}</span>
      <span className="ms-sub">{r.progress?.mb_per_s != null ? `${r.progress.mb_per_s} MB/s · ` : ''}{r.since ? `${fmtAgo(r.since).replace(' ago', '')} elapsed` : 'elapsed unknown'}
        {r.progress?.status === 'stalled' ? ' · STALLED' : ''} · {r.progress?.source === 'central-ledger' ? 'central ledger' : 'worker heartbeat'}</span>
    </div>
  })}</>
}
