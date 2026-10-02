// Per-model failure log (Models tab "logs"): load failures (class,
// loader_stderr, worker, ts) and routing refusals (predicate, request_id),
// newest first, plus the integrity verdict and admission reason. Dev-grade:
// full text, never truncated, copyable.
//
// Sources, feature-detected so the panel never breaks on an older central:
//   GET /api/llm/models/<key>/failures?limit=500  (new; 404 -> "not available yet")
//   GET /api/llm/compute-actions?action=load&outcome=fail&model=<key>  (fallback)
//   GET /api/llm/model-metrics2?model=<key>&suite=integrity             (integrity verdict)
//   GET /api/llm/logs?ref=admission:<job>&format=json                    (admission job log, whole)
import { useCallback, useEffect, useState } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { CopyButton, DiagnosticsView, LogBlock, toJsonText } from '../Diagnostics/Diagnostics'
import { normalizeFailures, tsOf } from './failures'
import './ModelLogs.css'

// The detail row that hosts this remounts on every table render, so results
// are cached per model for a short while; ↻ forces a refetch.
const CACHE = new Map()   // key -> {at, data}
const TTL_MS = 30000

async function getJson(url) {
  try {
    const r = await hugpyFetch(url)
    const text = await r.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = text }
    return { ok: r.ok, status: r.status, data }
  } catch (e) { return { ok: false, status: 0, data: String(e?.message || e) } }
}

const tail = (k) => String(k || '').split('~').pop()

const LIMIT = 500

async function fetchLogs(modelKey, admJob) {
  const enc = encodeURIComponent(modelKey)
  const [fail, integ, admLog] = await Promise.all([
    getJson(`/api/llm/models/${enc}/failures?limit=${LIMIT}`),
    getJson(`/api/llm/model-metrics2?model=${enc}&suite=integrity&limit=20`),
    admJob ? getJson(`/api/llm/logs?ref=${encodeURIComponent(`admission:${admJob}`)}&format=json`) : Promise.resolve(null),
  ])
  let items = [], source = 'failures', note = '', extra = null
  // An unknown route on central answers 200 with the SPA's index.html, so
  // "available" means a JSON object/array came back, not just 2xx.
  const failJson = fail.ok && fail.data != null && typeof fail.data === 'object'
  if (failJson) {
    items = normalizeFailures(fail.data)
    if (fail.data && typeof fail.data === 'object' && !Array.isArray(fail.data)) extra = fail.data
  } else {
    note = (fail.status === 404 || (fail.ok && !failJson))
      ? `GET /llm/models/<key>/failures → ${fail.status === 404 ? 'HTTP 404' : `HTTP ${fail.status} with a non-JSON body`} — showing load failures from the compute-action log`
      : `/llm/models/<key>/failures failed (HTTP ${fail.status || 'network'}): ${typeof fail.data === 'string' ? fail.data : toJsonText(fail.data)}`
    source = 'compute-actions'
    const ca = await getJson(`/api/llm/compute-actions?action=load&outcome=fail&limit=${LIMIT}&model=${enc}`)
    if (ca.ok) items = normalizeFailures({ actions: ca.data?.actions || [] })
    else note += ` · compute-actions also failed (HTTP ${ca.status || 'network'}): ${typeof ca.data === 'string' ? ca.data : toJsonText(ca.data)}`
  }
  // Integrity: newest integrity row for this model (the filter is additive
  // server-side; an older central ignores it, so match the name here too).
  let integrity = null
  let integRows = integ.ok ? (integ.data?.rows || []) : []
  const integErr = !integ.ok ? `integrity read failed (HTTP ${integ.status || 'network'}): ${typeof integ.data === 'string' ? integ.data : toJsonText(integ.data)}`
    : (integ.data?.error ? `integrity read: ${integ.data.error}` : '')
  if (integErr) note = note ? `${note} · ${integErr}` : integErr
  if (integRows.some(r => tail(r.model_name) !== tail(modelKey))) {
    // central predates the ?model= filter (it returned the unfiltered sheet)
    const all = await getJson('/api/llm/model-metrics2?limit=5000')
    integRows = all.ok ? (all.data?.rows || []) : []
  }
  if (integRows.length) {
    const rows = integRows.filter(r => r.grade_suite === 'integrity' && tail(r.model_name) === tail(modelKey))
      .sort((a, b) => (Number(b.graded_at) || 0) - (Number(a.graded_at) || 0))
    if (rows[0]) {
      let d = rows[0].grade_detail
      try { d = typeof d === 'string' ? JSON.parse(d) : d } catch { /* keep text */ }
      integrity = { grade: Number(rows[0].grade), graded_at: tsOf(rows[0].graded_at), detail: d }
    }
  }
  const admissionLog = !admJob ? null
    : admLog?.ok && admLog.data && typeof admLog.data === 'object' ? admLog.data
      : { error: `GET /llm/logs?ref=admission:${admJob} → HTTP ${admLog?.status || 'network'}: ${typeof admLog?.data === 'string' ? admLog.data : toJsonText(admLog?.data)}` }
  return { items, source, note, integrity, extra, admissionLog, fetchedAt: Date.now() }
}

const fmtTs = (t) => (t ? new Date(t).toLocaleString() : '—')

export default function ModelLogs({ modelKey, model }) {
  const [state, setState] = useState(() => CACHE.get(modelKey)?.data || null)
  const [loading, setLoading] = useState(false)
  const madm = model?.admission
  const admJob = madm && typeof madm === 'object' ? (madm.job || '') : ''
  const load = useCallback((force = false) => {
    const hit = CACHE.get(modelKey)
    if (!force && hit && Date.now() - hit.at < TTL_MS) { setState(hit.data); return }
    setLoading(true)
    fetchLogs(modelKey, admJob).then(data => { CACHE.set(modelKey, { at: Date.now(), data }); setState(data) })
      .finally(() => setLoading(false))
  }, [modelKey, admJob])
  useEffect(() => { load(false) }, [load])

  const adm = model?.admission ?? state?.extra?.admission
  const admission = adm == null ? null : typeof adm === 'string' ? { status: adm, reason: model?.admission_reason || '' } : adm
  const alog = state?.admissionLog
  const integ = state?.integrity
  const idetail = integ?.detail && typeof integ.detail === 'object' ? integ.detail : null

  return <div className="mlogs">
    <div className="mlogs-head">
      <b>Failure log</b>
      <span className="mlogs-sub">{state ? `${state.items.length} entr${state.items.length === 1 ? 'y' : 'ies'} · source ${state.source} · fetched ${fmtTs(state.fetchedAt)}` : ''}</span>
      <button type="button" className="mlogs-btn" onClick={() => load(true)} disabled={loading}>{loading ? '…' : '↻'}</button>
      {state && <CopyButton text={() => toJsonText({ model: modelKey, admission, ...state })} label="copy all" />}
    </div>
    {state?.note && <div className="mlogs-note">{state.note}</div>}

    {admission && <div className="mlogs-block">
      <div className="mlogs-kind">admission: {admission.status}{admission.job ? ` · job ${admission.job}` : ''}</div>
      {/* the verdict is a heading; the job's own log is the log */}
      <div className="mlogs-heading">{admission.reason || `(admission record${admission.job ? ` job ${admission.job}` : ''}${admission.at ? ` at ${fmtTs(admission.at * (admission.at < 1e12 ? 1000 : 1))}` : ''} carries no reason text)`}</div>
      {admission.job && (alog?.error
        ? <div className="mlogs-note">{alog.error}</div>
        : alog && <LogBlock text={alog.text} source={alog.source} logRef={`admission:${admission.job}`} bytes={alog.bytes} />)}
      {!admission.job && <div className="mlogs-note">the admission record names no job id, so no admission job log can be read for it</div>}
    </div>}
    {integ && <div className={`mlogs-block ${integ.grade === 0 ? 'bad' : ''}`}>
      <div className="mlogs-kind">integrity: {idetail?.verdict || (integ.grade === 0 ? 'fail' : 'ok')} · grade {integ.grade} · {fmtTs(integ.graded_at)}</div>
      {idetail?.why && <div className="mlogs-heading">{idetail.why}</div>}
      {idetail ? <>
        <LogBlock text={Array.isArray(idetail.log) ? idetail.log.join('\n') : (idetail.log ?? '')}
          source={`integrity grade_detail.log (model-metrics2, ${Array.isArray(idetail.log) ? idetail.log.length : 0} lines)`} />
        <details><summary>raw record</summary>
          <DiagnosticsView value={Object.fromEntries(Object.entries(idetail).filter(([k]) => !['log', 'why'].includes(k)))} />
        </details>
      </> : <LogBlock text={String(integ.detail ?? '')} source="integrity grade_detail (model-metrics2, unparsed)" />}
    </div>}

    {state && !state.items.length && <div className="mlogs-empty">no load failures or routing refusals recorded for {modelKey} (read source {state.source}, up to {LIMIT} rows, 0 found, fetched {fmtTs(state.fetchedAt)})</div>}
    {state?.items.map((it, i) => <div key={i} className="mlogs-block">
      <div className="mlogs-kind">
        {it.kind} · {fmtTs(it.ts)}{it.worker ? ` · ${it.worker}` : ''}{it.cls ? ` · ${it.cls}` : ''}
        {it.predicate ? ` · predicate ${it.predicate}` : ''}{it.request_id ? ` · request ${it.request_id}` : ''}
      </div>
      <LogBlock text={it.text} source={it.source} logRef={it.logRef} />
      {it.fileLogRef && it.fileLogRef !== it.logRef && <div className="mlogs-note">per-load log file: {it.fileLogRef}</div>}
      {it.message && it.message !== it.text && <LogBlock text={it.message} source={`${it.id != null ? `compute_actions#${it.id} ` : ''}detail.message`} />}
      <details><summary>raw</summary><DiagnosticsView value={it.raw} /></details>
    </div>)}
  </div>
}
