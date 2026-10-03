// TEST-FIRE HISTORY (operator 2026-10-02): every past test fire on this worker,
// always available — persisted on central (test_fire_store), so it survives
// switching workers, leaving the screen and central restarts. A run opens in a
// popup with every call; "analyze" opens the per-model inference over the
// stored runs (baseline vs latest, verdict healthy/flaky/failing/regressed).
import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { fetchJson } from '../../api'
import { hugpyFetch } from '../../runtime/config'

const SHOWN = 8

function base(worker) {
  return `/api/llm/workers/${encodeURIComponent(worker.id)}/test-fire`
}

function when(ts) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function dur(a, b) {
  if (!a || !b) return null
  const s = Math.max(0, Math.round(b - a))
  return s >= 60 ? `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s` : `${s}s`
}

function Popup({ title, onClose, children }) {
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return createPortal(
    <div className="wp-tfh-overlay" onClick={onClose}>
      <div className="wp-tfh-popup" role="dialog" aria-label={title} onClick={e => e.stopPropagation()}>
        <div className="wp-tfh-popup-head">
          <span className="wp-tfh-popup-title">{title}</span>
          <span className="wp-tf-spacer" />
          <button className="wp-tf-toggle" onClick={onClose} title="close (Esc)">✕</button>
        </div>
        <div className="wp-tfh-popup-body">{children}</div>
      </div>
    </div>,
    document.body)
}

// Resumable = cut off (central restart) or stopped by the operator.
const resumable = run => !run.live && run.state !== 'complete' && run.state !== 'running'

function RunPopup({ worker, run, onClose, onResumed }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [resuming, setResuming] = useState(false)
  const resume = async () => {
    setResuming(true); setError(null)
    try {
      const r = await hugpyFetch(`${base(worker)}/history/${encodeURIComponent(run.job_id)}/resume`, { method: 'POST' })
      const b = await r.json().catch(() => ({}))
      if (!r.ok || !b.job_id) throw new Error(b.error || `HTTP ${r.status}`)
      onResumed && onResumed(b.job_id)
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setResuming(false)
    }
  }
  useEffect(() => {
    let off = false
    fetchJson(`${base(worker)}/history/${encodeURIComponent(run.job_id)}`)
      .then(d => { if (!off) setData(d) })
      .catch(e => { if (!off) setError(e.message) })
    return () => { off = true }
  }, [worker, run.job_id])
  const results = data?.results || []
  const perModel = {}
  for (const r of results) {
    const m = perModel[r.model_key] || (perModel[r.model_key] = { ok: 0, failed: 0, lat: [], tps: [] })
    if (r.ok) { m.ok++; if (r.latency_s != null) m.lat.push(r.latency_s); if (r.tok_s != null) m.tps.push(r.tok_s) } else m.failed++
  }
  const med = xs => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const skipped = data?.skipped || []
  return (
    <Popup title={`test fire ${run.job_id} · ${when(run.created)} · ${worker.name}`} onClose={onClose}>
      {error && <div className="wp-tf-err">⚠ {error}</div>}
      {!data && !error && <div className="wp-lt-muted">loading…</div>}
      {data && (
        <>
          <div className="wp-tfh-meta">
            <span>{run.state}</span>
            <span>✓ {data.ok} · ✗ {data.failed}</span>
            <span>{data.rounds ? `${data.rounds} round(s)` : 'until stopped'} · concurrency {data.concurrency} · max {data.max_tokens} tok</span>
            {dur(data.started, data.finished) && <span>took {dur(data.started, data.finished)}</span>}
            {skipped.length > 0 && <span title={skipped.map(s => `${s.model_key}: ${s.reason}`).join('\n')}>skipped {skipped.length}</span>}
            {data.resumed_from && <span>resumed from {data.resumed_from}</span>}
            {resumable(run) && (
              <button className="wp-test-fire" disabled={resuming} onClick={resume}
                      title={`start a new run that restarts ${data.current_model || 'the interrupted model'} and proceeds through the rest of the round${data.rounds > 1 ? ' and the remaining rounds' : ''}`}>
                {resuming ? '▶ …' : `▶ resume${data.current_model ? ` from ${data.current_model}` : ''}`}
              </button>
            )}
          </div>
          <table className="wp-tfh-table">
            <thead><tr><th>model</th><th>ok</th><th>failed</th><th>median latency</th><th>median tok/s</th></tr></thead>
            <tbody>
              {Object.entries(perModel).map(([k, m]) => (
                <tr key={k} className={m.failed && !m.ok ? 'wp-tfh-bad' : m.failed ? 'wp-tfh-warn' : ''}>
                  <td>{k}</td><td>{m.ok}</td><td>{m.failed}</td>
                  <td>{med(m.lat) != null ? `${med(m.lat)}s` : '—'}</td><td>{med(m.tps) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="wp-tfh-sub">every call ({results.length})</div>
          <table className="wp-tfh-table">
            <thead><tr><th>time</th><th>model</th><th></th><th>latency</th><th>tokens</th><th>tok/s</th><th>result</th></tr></thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} className={r.ok ? '' : 'wp-tfh-bad'}>
                  <td>{r.started ? new Date(r.started * 1000).toLocaleTimeString() : '—'}</td>
                  <td>{r.model_key}</td>
                  <td>{r.ok ? '✓' : '✗'}</td>
                  <td>{r.latency_s != null ? `${r.latency_s}s` : '—'}</td>
                  <td>{r.tokens ?? '—'}</td>
                  <td>{r.tok_s ?? '—'}</td>
                  <td className="wp-tfh-result" title={r.ok ? `${r.prompt || ''}\n→ ${r.content80 || ''}` : r.error || ''}>
                    {r.ok ? (r.content80 || '') : `${r.error_kind || 'error'}: ${String(r.error || '').slice(0, 140)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Popup>
  )
}

function AnalysisPopup({ worker, onClose }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let off = false
    fetchJson(`${base(worker)}/analysis?runs=20`)
      .then(d => { if (!off) setData(d) })
      .catch(e => { if (!off) setError(e.message) })
    return () => { off = true }
  }, [worker])
  const rows = Object.entries(data?.models || {})
  const order = { failing: 0, regressed: 1, flaky: 2, untested: 3, healthy: 4 }
  rows.sort((a, b) => (order[a[1].verdict] ?? 9) - (order[b[1].verdict] ?? 9) || a[0].localeCompare(b[0]))
  const th = data?.thresholds
  return (
    <Popup title={`test fire analysis · ${worker.name}`} onClose={onClose}>
      {error && <div className="wp-tf-err">⚠ {error}</div>}
      {!data && !error && <div className="wp-lt-muted">analyzing…</div>}
      {data && (
        <>
          <div className="wp-tfh-meta">
            <span>over the last {data.runs} run(s)</span>
            {Object.entries(data.verdicts || {}).map(([v, n]) => <span key={v} className={`wp-tfh-v wp-tfh-v-${v}`}>{v} {n}</span>)}
          </div>
          {th && (
            <div className="wp-lt-muted wp-tfh-rule">
              baseline = medians of each model's ok calls before the latest run (≥{th.min_baseline} needed) ·
              failing = last {th.failing_streak} calls failed · regressed = latest tok/s &lt; {Math.round(th.regress_tok_s * 100)}% or latency &gt; {Math.round(th.regress_latency * 100)}% of baseline ·
              flaky = under {Math.round(th.healthy_rate * 100)}% ok
            </div>
          )}
          <table className="wp-tfh-table">
            <thead><tr><th>model</th><th>verdict</th><th>ok / calls</th><th>baseline tok/s · latency</th><th>latest tok/s · latency</th><th>errors</th><th>why</th></tr></thead>
            <tbody>
              {rows.map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td><span className={`wp-tfh-v wp-tfh-v-${v.verdict}`}>{v.verdict}</span></td>
                  <td>{v.ok}/{v.calls}{v.success_rate != null ? ` (${Math.round(v.success_rate * 100)}%)` : ''}</td>
                  <td>{v.baseline ? `${v.baseline.tok_s ?? '—'} · ${v.baseline.latency_s ?? '—'}s (${v.baseline.samples})` : '—'}</td>
                  <td>{v.latest ? `${v.latest.tok_s ?? '—'} · ${v.latest.latency_s ?? '—'}s (${v.latest.samples})` : '—'}</td>
                  <td>{Object.entries(v.error_kinds || {}).map(([e, n]) => `${e}×${n}`).join(', ') || '—'}</td>
                  <td className="wp-tfh-result">{v.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && <div className="wp-lt-muted">no stored calls yet — run a test fire.</div>}
        </>
      )}
    </Popup>
  )
}

export default function TestFireHistory({ worker, tf }) {
  const [runs, setRuns] = useState(null)
  const [stored, setStored] = useState(true)
  const [error, setError] = useState(null)
  const [all, setAll] = useState(false)
  const [openRun, setOpenRun] = useState(null)
  const [analysis, setAnalysis] = useState(false)
  const liveKey = `${tf?.job?.job_id || ''}:${tf?.job?.running ? 1 : 0}:${tf?.job?.done_calls || 0}`

  const load = useCallback(() => {
    if (!worker?.id) return
    fetchJson(`${base(worker)}/history?limit=100`)
      .then(d => { setRuns(d.runs || []); setStored(d.stored !== false); setError(null) })
      .catch(e => setError(e.message))
  }, [worker?.id])   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { load() }, [load, liveKey])

  const list = runs || []
  const shown = all ? list : list.slice(0, SHOWN)
  return (
    <div className="wp-tfh">
      <div className="wp-tf-head">
        <span className="wp-tf-title">🗂 test fire history</span>
        <span className="wp-tf-stat">{runs == null ? '…' : `${list.length} run(s)`}</span>
        {!stored && <span className="wp-tf-err" title="central has no DB for test-fire history">not stored</span>}
        {error && <span className="wp-tf-err" title={error}>⚠ {error}</span>}
        <span className="wp-tf-spacer" />
        <button className="wp-tf-toggle" disabled={!list.length} onClick={() => setAnalysis(true)}
                title="per-model inference over the stored runs: baseline vs latest, healthy/flaky/failing/regressed">
          🔍 analyze
        </button>
      </div>
      {list.length > 0 && (
        <div className="wp-tfh-runs">
          {shown.map(r => (
            <button key={r.job_id} type="button" className={`wp-tfh-run wp-tfh-run-${r.state}`}
                    onClick={() => setOpenRun(r)}
                    title={`${r.job_id} · ${r.state}${r.stop_reason ? ` (${r.stop_reason})` : ''} · open for every call`}>
              <span>{when(r.created)}</span>
              <span className="wp-tf-ok">✓{r.ok}</span>
              <span className="wp-tf-fail">✗{r.failed}</span>
              {r.state !== 'complete' && <em>{r.state}{resumable(r) ? ' ▶' : ''}</em>}
            </button>
          ))}
          {list.length > SHOWN && (
            <button type="button" className="wp-tf-toggle" onClick={() => setAll(!all)}>
              {all ? 'fewer' : `+${list.length - SHOWN} more`}
            </button>
          )}
        </div>
      )}
      {openRun && <RunPopup worker={worker} run={openRun} onClose={() => setOpenRun(null)}
                            onResumed={jobId => { tf?.adopt && tf.adopt(jobId); load() }} />}
      {analysis && <AnalysisPopup worker={worker} onClose={() => setAnalysis(false)} />}
    </div>
  )
}
