// Model detail for the "worth my time" question: everything behind the row's
// cells — full verification log + findings + suggested fix, admission history,
// grade tiers per task, failures, per-worker live state and fit numbers.
// Every section states what is missing and how to produce it.
import { useCallback, useEffect, useState } from 'react'
import ModelLogs from '../ModelLogs/ModelLogs'
import { matrixFor, resolveSuite, throughputText } from '../MetricsPanel/suites'
import { GradedItem, itemsFromDetail } from '../MetricsPanel/GradedItem'
import { CopyButton, LogBlock, toJsonText } from '../Diagnostics/Diagnostics'
import { fmtAgo, gradeView, throughputView, verificationView, admissionView, worthView } from './modelStatus'
import { Chip, RefusalDiagnostics, VerifyGradeButton, WorkerStateChips } from './StatusCells'
import { fetchAdmission, fetchModelStatusDetail, useVerifyState } from './useModelStatus'

const gib = (b) => (b == null ? '?' : `${(Number(b) / 2 ** 30).toFixed(1)} GiB`)
const when = (t) => (t ? `${new Date(Number(t) * 1000).toLocaleString()} (${fmtAgo(t)})` : 'never')

// The detail row remounts on every table render: detail + admission history
// are cached per model for a short while (↻ forces a refetch).
const CACHE = new Map()   // key -> {at, detail, adm, err}
const TTL_MS = 10000

export default function ModelStatusDetail({ modelKey, row, model }) {
  const hit = CACHE.get(modelKey)
  const [detail, setDetail] = useState(hit?.detail || null)
  const [adm, setAdm] = useState(hit?.adm || null)
  const [err, setErr] = useState(hit?.err || '')
  const load = useCallback((force = false) => {
    const c = CACHE.get(modelKey)
    if (!force && c && Date.now() - c.at < TTL_MS) return
    const entry = { at: Date.now(), detail: c?.detail || null, adm: c?.adm || null, err: '' }
    CACHE.set(modelKey, entry)
    setErr('')
    fetchModelStatusDetail(modelKey)
      .then(d => { entry.detail = d; setDetail(d) })
      .catch(e => { entry.err = String(e?.message || e); setErr(entry.err) })
    fetchAdmission(modelKey)
      .then(a => { entry.adm = a; setAdm(a) })
      .catch(e => { entry.adm = { error: String(e?.message || e) }; setAdm(entry.adm) })
  }, [modelKey])
  useEffect(() => { load(false) }, [load])
  const verify = useVerifyState(modelKey)
  useEffect(() => { if (verify.phase === 'done' || verify.phase === 'failed') load(true) }, [verify.phase, load])

  const r = row || detail
  if (!r) {
    return <div className="ms-detail ms-none">No status for this model: this central does not serve
      /llm/models/status (older build){err ? ` — ${err}` : ''}.</div>
  }
  const d = detail?.detail || {}
  const w = worthView(r)
  const ver = verificationView(r)
  const g = gradeView(r)
  const a = admissionView(r)
  const vd = d.verification || {}
  const jobs = Array.isArray(adm?.jobs) ? adm.jobs : []

  return <div className="ms-detail">
    <div className="ms-detail-head">
      <Chip tone={w.tone} className="ms-worth">{w.text}</Chip>
      <span className="ms-detail-why">{w.why}</span>
      <span className="ms-sub">→ {w.action}</span>
      <span className="ms-spacer" />
      <VerifyGradeButton modelKey={modelKey} />
      <RefusalDiagnostics row={r} />
      <button type="button" className="mt-act" onClick={() => load(true)} title="re-read status, admission and detail">↻</button>
      <CopyButton text={() => toJsonText({ status: r, admission: adm })} label="copy all" />
    </div>
    {err && <div className="ms-note">detail read failed: {err} — showing the table row</div>}

    <section className="ms-sec">
      <h4>Per-worker state</h4>
      <WorkerStateChips row={r} />
      <ul className="ms-list">{(r.workers || []).map(ws => <li key={ws.worker}>
        <b>{ws.worker}</b> — {ws.label}: {ws.detail}
        {ws.failed && <LogBlock text={ws.failed.text ?? ''} bytes={ws.failed.bytes}
          source={ws.failed.text_source || `${ws.worker} failed-load record (this central predates the whole-text field)`}
          logRef={ws.failed.log_ref || ''} />}</li>)}</ul>
    </section>

    <section className="ms-sec">
      <h4>Verification <Chip tone={ver.tone}>{ver.text}</Chip>
        <span className="ms-sub">source {r.verification?.source || 'none'} · {when(r.verification?.at)}</span></h4>
      {r.verification?.source === 'none'
        ? <p className="ms-missing">NOT VERIFIED — no audit entry, no integrity grade, no admission record. Verify + grade runs the audit.</p>
        : <p>{r.verification?.why || '(no detail recorded)'}</p>}
      {r.suggested_fix && <p className="ms-fix"><b>Suggested fix:</b> {r.suggested_fix}</p>}
      {(vd.findings || []).length > 0 && <table className="ms-table"><thead><tr><th>check</th><th>verdict</th><th>detail</th><th>fix</th></tr></thead>
        <tbody>{vd.findings.map((f, i) => <tr key={i}><td>{f.check}</td><td>{f.verdict}</td><td>{f.detail}{f.file ? ` (${f.file})` : ''}{f.worker ? ` @${f.worker}` : ''}</td><td>{f.fix || '—'}</td></tr>)}</tbody></table>}
      {detail
        ? <LogBlock title={`audit log (${(vd.log || []).length} lines)`} text={(vd.log || []).join('\n')}
            source="GET /llm/models/status?model=<key>&detail=1 → detail.verification.log (model_audit.json)" />
        : <p className="ms-missing">audit log loads with the detail (GET /llm/models/status?detail=1 not answered yet{err ? `: ${err}` : ''})</p>}
    </section>

    <section className="ms-sec">
      <h4>Admission <Chip tone={a.tone}>{a.text}</Chip><span className="ms-sub">{when(r.admission?.at)}</span></h4>
      <p>{r.admission?.reason || '(no reason recorded)'}</p>
      {adm?.error && <p className="ms-note">admission history unavailable: {adm.error}</p>}
      {adm && !adm.error && !jobs.length && <p className="ms-missing">no admission jobs on record for this model — Verify + grade queues one</p>}
      {jobs.map((j, ji) => <details key={j.id} className="ms-job" open={ji === 0}>
        <summary><b>{j.status}</b> · {j.source || '—'} · created {when(j.created_at)}{j.finished_at ? ` · finished ${when(j.finished_at)}` : ''} <span className="ms-sub">{j.id}</span></summary>
        <LogBlock text={j.log || ''} source={`admission_jobs.log id=${j.id} (GET /llm/admission/<key>)`} logRef={`admission:${j.id}`} />
      </details>)}
    </section>

    <section className="ms-sec">
      <h4>Grade <Chip tone={g.tone}>{g.text}</Chip>{g.sub && <span className="ms-sub">{g.sub}</span>}</h4>
      {r.grade?.value == null
        ? <p className="ms-missing">{r.grade?.suite == null
          ? `${(r.grade?.reason || 'no grader').toUpperCase()} — no automated grading suite covers this task; its quality is unknown until judged by hand.`
          : `NOT GRADED — ${r.grade.suite} has never graded this model. Verify + grade runs the benchmark.`}</p>
        : <p>{r.grade.detail_summary} · graded {when(r.grade.at)}{r.grade.worker ? ` on ${r.grade.worker}` : ''}{r.grade.quant ? ` · ${r.grade.quant}` : ''}{r.grade.rows > 1 ? ` · best of ${r.grade.rows} rows ${Math.round(r.grade.best)}` : ''}</p>}
      {(r.grade?.stray || []).length > 0 && <p className="ms-note">ignored (suite does not match the task): {r.grade.stray.map(s => `${s.suite} ${s.value ?? '?'}${s.worker ? ` @${s.worker}` : ''}`).join(', ')}</p>}
      {(d.grade_rows || []).map((gr, i) => {
        const suite = resolveSuite({ gradeSuite: gr.grade_suite, detail: gr.grade_detail })
        const mx = matrixFor(gr.grade_detail, suite)
        return <div key={i} className="ms-graderow">
          <div className="ms-sub"><b>{mx.recorded ? mx.text : `${Math.round(gr.grade)}% (no per-item detail)`}</b> · {suite.name}{suite.legacy ? ' (legacy suite)' : ''} · {gr.worker || '—'} · {gr.quant || '—'} · {gr.alloc_mode || '—'} · {fmtAgo(gr.graded_at)}</div>
          <table className="ms-table"><thead><tr>{mx.columns.map(t => <th key={t}>{t}</th>)}</tr></thead>
            <tbody><tr>{mx.cells.map(c => <td key={c.task}>
              <span className="metrics-tier">{c.markers.map((m, j) => <i key={j} className={m.pass === true ? 'pass' : m.pass === false ? 'fail' : 'metrics-tier-untested'}
                title={`${c.task} ${m.tier}: ${m.pass === true ? 'PASS' : m.pass === false ? 'FAIL' : 'not run'}${m.expected ? `\nexpected: ${m.expected}` : ''}${m.actual != null ? `\nactual: ${m.actual}` : ''}${m.why ? `\nwhy: ${m.why}` : ''}`} />)}</span>
            </td>)}</tr></tbody></table>
          {itemsFromDetail(gr.grade_detail, suite).length > 0 && <details className="ms-graded-items"><summary>every item: prompt · expected response · model response</summary>
            <div className="graded-list">{itemsFromDetail(gr.grade_detail, suite).map((it, j) => <GradedItem key={j} item={it} heading={`${it.task} · ${it.tier}`} />)}</div>
          </details>}
        </div>
      })}
    </section>

    <section className="ms-sec">
      <h4>Throughput <span className="ms-sub">{throughputView(r).text}</span></h4>
      {!(r.throughput?.n_calls) ? <p className="ms-missing">no calls recorded — tok/s is the mean of recorded calls, and there are none yet.</p>
        : <p>{throughputView(r).title}</p>}
      {(d.throughput_cells || []).length > 0 && <table className="ms-table"><thead><tr><th>worker</th><th>quant</th><th>alloc</th><th>tok/s (avg of calls)</th></tr></thead>
        <tbody>{d.throughput_cells.map((c, i) => <tr key={i}><td>{c.worker}</td><td>{c.quant || 'runtime default'}</td><td>{c.alloc_mode || '—'}</td><td>{throughputText(c)}</td></tr>)}</tbody></table>}
    </section>

    <section className="ms-sec">
      <h4>Fit per worker <span className="ms-sub">needs ≈{gib((r.servable?.need_bytes || 0) * (d.fit_headroom || 1.15))} incl. overhead · {r.servable?.reason}</span></h4>
      {(d.fit || []).length ? <table className="ms-table"><thead><tr><th>worker</th><th>online</th><th>GPU</th><th>RAM</th><th>fit</th></tr></thead>
        <tbody>{d.fit.map(f => <tr key={f.worker}><td>{f.worker}</td><td>{f.online ? 'yes' : 'no'}</td><td>{gib(f.gpu_bytes)}</td><td>{gib(f.ram_bytes)}</td>
          <td className={`ms-fit-${f.fit}`}>{f.fit === 'gpu' ? 'fits in VRAM' : f.fit === 'offload' ? 'GPU + RAM offload' : f.fit === 'unknown' ? 'size unknown' : 'does not fit'}</td></tr>)}</tbody></table>
        : <p className="ms-missing">{detail ? 'no workers registered' : 'fit numbers load with the detail'}</p>}
      <p className="ms-sub">serve override: {d.override ? <code>{JSON.stringify(d.override)}</code> : 'none'}</p>
    </section>

    <section className="ms-sec">
      <h4>Failures</h4>
      <ModelLogs modelKey={modelKey} model={model} />
    </section>
  </div>
}
