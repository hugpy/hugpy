import { useEffect, useMemo, useState } from 'react'
import { fetchJson } from '../../api'
import './MetricsPanel.css'

// Worker categorical palette — validated colorblind-safe (dataviz skill), dark
// steps (the console is dark). ae/computron get their fixed brand colors; any
// other worker name (aeb, a-brain-Super-Server, ...) cycles the fallback ramp
// deterministically by first-seen order, so the fleet can grow workers without
// a code change here.
const WORKER_COLORS = { ae: '#3987e5', computron: '#d95926' }
const FALLBACK = ['#199e70', '#c98500', '#d55181', '#9085e9', '#4fb3bf', '#b0854a']
// Real decode tops out in the low hundreds of tok/s; a value above this is the
// known bad-measurement artifact (guarded in the recorder now, but old rows may
// linger) — render it as ∞* instead of blowing out the axis.
const SENTINEL = 5000

const colorFor = (w, i) => WORKER_COLORS[w] || FALLBACK[i % FALLBACK.length]
const isBad = (v) => v != null && v >= SENTINEL
const fmt = (n, dp = 1) =>
  n == null || isNaN(n) ? '—' : isBad(n) ? '∞*'
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: dp })
const fmtStr = (s) => (s == null || s === '' ? '—' : s)
const shortModel = (m) => String(m || '').replace(/-GGUF$/, '').replace(/^[^~]*~/, '')

// Deterministic per-panel-load worker color assignment — first-seen order
// over the LIVE rows, so aeb/a-brain-Super-Server/etc. get a stable color for
// the duration of a render even though they aren't in WORKER_COLORS.
function workerPalette(rows) {
  const seen = []
  for (const r of rows) if (r.worker && !seen.includes(r.worker)) seen.push(r.worker)
  const map = {}
  seen.forEach((w, i) => { map[w] = colorFor(w, i) })
  return map
}

export default function MetricsPanel() {
  const [data, setData] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [error, setError] = useState(null)
  const [benchmarkError, setBenchmarkError] = useState(null)
  const [catalog, setCatalog] = useState([])
  const [metric, setMetric] = useState('tok_per_s')  // tok_per_s | tok_per_s_avg

  const load = () => {
    // NOTE: the /api prefix — the SPA's hugpyFetch resolves /api/llm/... (same
    // as every other panel, e.g. WorkersPanel's /api/llm/workers). A bare
    // /llm/... does not resolve and the panel would render empty.
    //
    // model-metrics2 (t146) is the LIVE db-hugpy `model_metrics` sheet — one
    // row per (model x quant x alloc_mode x worker). The OLDER /llm/model-metrics
    // (db-toolserver EMA snapshot) is abandoned here: that store is on the
    // wrong database with a role that lacks CREATE, so it never had rows.
    fetchJson('/api/llm/model-metrics2')
      .then((d) => { setData(d); setError(null) })
      .catch((e) => setError(String(e && e.message || e)))
    fetchJson('/api/llm/benchmark/status')
      .then((d) => { setBenchmark(d); setBenchmarkError(null) })
      .catch((e) => setBenchmarkError(String(e && e.message || e)))
    fetchJson('/api/v1/models')
      .then(d => setCatalog((d?.data || []).map(m => m.id).filter(Boolean)))
      .catch(() => {})
  }
  useEffect(() => {
    load()
    const t = setInterval(load, benchmark?.status === 'running' ? 2500 : 15000)
    return () => clearInterval(t)
  }, [benchmark?.status])

  if (!data && !benchmark) return <div className="metrics-panel"><p className="metrics-muted">Loading metrics…</p></div>

  const rows = data?.rows || []
  const models = new Set(rows.map(r => r.model_name)).size
  const workers = new Set(rows.map(r => r.worker)).size
  const palette = workerPalette(rows)
  const dataError = data?.error

  return (
    <div className="metrics-panel">
      <div className="metrics-head">
        <div className="metrics-sub">
          {rows.length} rows · {models} models · {workers} workers
          {dataError ? <span className="metrics-error"> · {dataError}</span> : null}
          {data?.generated_at ? ` · generated ${new Date(data.generated_at * 1000).toLocaleTimeString()}` : ''}
          {error ? <span className="metrics-error"> · historical metrics: {error}</span> : null}
          {benchmarkError ? <span className="metrics-error"> · benchmark metrics: {benchmarkError}</span> : null}
        </div>
        <a className="metrics-link" href="#" onClick={(e) => { e.preventDefault(); load() }}>↻ refresh</a>
      </div>

      <BenchmarkLiveOutput benchmark={benchmark} onRefresh={load} />
      <BenchmarkWorkbook benchmark={benchmark} historical={rows} catalog={catalog} />

      <div className="metrics-grid">
        <section className="metrics-card">
          <h3>Most-active models <span className="metrics-sub">by sample count</span></h3>
          <SamplesChart rows={rows} />
        </section>
        <section className="metrics-card">
          <div className="metrics-card-head">
            <h3>Throughput <span className="metrics-sub">tok/s by worker</span></h3>
            <button className="metrics-toggle" onClick={() => setMetric(metric === 'tok_per_s' ? 'tok_per_s_avg' : 'tok_per_s')}>
              {metric === 'tok_per_s' ? 'last' : 'avg'} ▾
            </button>
          </div>
          <Legend palette={palette} />
          <WorkerChart rows={rows} metric={metric} palette={palette} />
        </section>
      </div>

      <section className="metrics-card">
        <h3>model_metrics <span className="metrics-sub">live sheet — db-hugpy, per model x quant x alloc_mode x worker (click a header to sort)</span></h3>
        <LiveTable rows={rows} palette={palette} />
      </section>
    </div>
  )
}

function BenchmarkLiveOutput({ benchmark, onRefresh }) {
  const [cancelling, setCancelling] = useState(false)
  if (!benchmark) return null
  const progress = benchmark.progress || {}
  const calls = benchmark.calls || []
  const events = benchmark.events || []
  const results = benchmark.results || []
  const current = [progress.worker || progress.worker_id, progress.model, progress.quant,
    progress.config || progress.alloc_mode].filter(Boolean).join(' · ')
  const cancel = () => {
    if (benchmark.status !== 'running' || cancelling) return
    if (!confirm('Cancel the active capacity test? The result rows already recorded will be retained.')) return
    setCancelling(true)
    fetchJson('/api/llm/benchmark/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: 'execution' }) })
      .then(onRefresh).finally(() => setCancelling(false))
  }
  if (benchmark.status === 'idle' && !events.length && !calls.length && !results.length) return null
  return <section className={`metrics-card metrics-live metrics-live-${benchmark.status || 'idle'}`}>
    <div className="metrics-live-head">
      <div><h3>Live test output</h3>
        <span className="metrics-sub">{benchmark.run_id ? `run ${benchmark.run_id} · ` : ''}{benchmark.status || 'idle'}</span></div>
      <div className="metrics-live-actions">
        <button onClick={onRefresh}>↻ refresh</button>
        {benchmark.status === 'running' && <button className="metrics-cancel" onClick={cancel} disabled={cancelling}>
          {cancelling ? 'cancelling…' : '■ Cancel test'}
        </button>}
      </div>
    </div>
    <div className="metrics-live-progress">
      <progress max="100" value={progress.percent || 0} />
      <b>{progress.completed || 0}/{progress.total || benchmark.plan?.runnable || 0}</b>
      <span>{progress.percent || 0}%</span>
    </div>
    {current && benchmark.status === 'running' && <div className="metrics-live-current"><span className="metrics-live-dot" />currently testing <code>{current}</code></div>}
    {benchmark.error && <p className="metrics-error">{benchmark.error}</p>}
    <div className="metrics-live-counts">
      <span><b>{results.length}</b> configurations</span><span><b>{calls.length}</b> graded calls</span>
      <span><b>{calls.filter(c => c.grade === 'PASS').length}</b> passed</span>
      <span><b>{calls.filter(c => c.grade === 'FAIL').length}</b> failed</span>
    </div>
    {!!events.length && <details className="metrics-live-log" open={benchmark.status === 'running'}>
      <summary>Activity log ({events.length})</summary>
      <pre>{events.slice().reverse().map(e => `${e.at ? new Date(e.at * 1000).toLocaleTimeString() : '—'}  ${e.kind || 'event'}  ${e.message || (e.value == null ? '' : JSON.stringify(e.value))}`).join('\n')}</pre>
    </details>}
    {!!calls.length && <details className="metrics-live-log">
      <summary>Completed calls ({calls.length})</summary>
      <pre>{calls.slice().reverse().map(c => `${c.grade || '—'}  ${c.worker || c.worker_id || '—'}  ${c.model || '—'} / ${c.quant || '—'}  ${c.task || '—'}  ${c.elapsed_s ?? '—'}s  ${c.tok_s ?? '—'} tok/s`).join('\n')}</pre>
    </details>}
  </section>
}

const BENCH_TASKS = ['math', 'wordprob', 'factual', 'format_primes', 'logic',
  'exact_instruction', 'coding', 'json', 'letters']
const benchKey = r => [r.worker_id || r.worker, r.model, r.quant, r.config, r.alloc_mode].join('\0')
const metricValue = v => v == null || v === 'N/A' || isNaN(Number(v)) ? 'N/A' : Number(v).toFixed(1)
const seconds = v => v == null || v === 'N/A' || isNaN(Number(v)) ? 'N/A' : `${Number(v).toFixed(1)} s`
const bytes = v => {
  if (v == null || isNaN(Number(v))) return 'N/A'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']; let n = Number(v), i = 0
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(i && n < 10 ? 1 : 0)} ${units[i]}`
}
const tierDepth = value => {
  if (value && typeof value === 'object') return Number(value.tier) || 0
  return Number(value) || 0
}
function TierBlocks({ value }) {
  if (value == null) return 'N/A'
  const depth = Math.max(0, Math.min(3, tierDepth(value)))
  const history = Array.isArray(value?.history) ? value.history : []
  return <span className="metrics-tier" title={`${depth} tier${depth === 1 ? '' : 's'} passed`} aria-label={`${depth} of 3 tiers passed`}>
    {[0, 1, 2].map(i => <i key={i} title={['Easy', 'Medium', 'Hard'][i]} className={history[i]?.pass === true ? 'pass' : history[i]?.pass === false ? 'fail' : 'metrics-tier-untested'} />)}
    <em>{depth}</em>
  </span>
}

function WorkerSummary({ rows, calls, workers, lifetime = [] }) {
  const stats = workers.map(w => {
    const mine = rows.filter(r => (r.worker_id || r.worker) === w.id)
    const speeds = mine.map(r => Number(r.tok_s_avg ?? r.tok_s)).filter(Number.isFinite)
    const lifetimeRow = lifetime.find(r => r.worker === w.id || r.worker === w.name)
    return { ...w, peak: speeds.length ? Math.max(...speeds) : 0,
      calls: lifetimeRow?.n_calls ?? calls.filter(c => (c.worker_id || c.worker) === w.id).length }
  })
  const maxSpeed = Math.max(1, ...stats.map(s => s.peak)); const maxCalls = Math.max(1, ...stats.map(s => s.calls))
  const chart = (field, max) => <div className="metrics-bars">{stats.map((s, i) => <div className="metrics-bar-row" key={s.id}>
    <b>{s.name}</b><span><i style={{ width: `${100 * s[field] / max}%`, background: colorFor(s.name, i) }} /></span>
    <em>{field === 'peak' ? metricValue(s[field]) : s[field]}</em>
  </div>)}</div>
  return <div className="metrics-model-summary">
    <div><h3>Throughput <small>Peak tok/s by worker</small></h3>{chart('peak', maxSpeed)}</div>
    <div><h3>Total Calls <small>Selected-model evaluations</small></h3>{chart('calls', maxCalls)}</div>
  </div>
}

function GradingMatrix({ rows, workers }) {
  return <div className="metrics-sheetwrap"><table className="metrics-sheet metrics-grade-matrix"><thead><tr>
    <th>Quant / Size</th><th>Cold</th><th>Grade Tier</th><th>Alloc</th><th>Hot</th><th>tok/s</th>
    {BENCH_TASKS.map(t => <th key={t}>{t}</th>)}<th>task_best</th><th>task_worst</th><th>score</th>
  </tr></thead><tbody>{workers.flatMap((worker, workerIndex) => {
    const workerRows = rows.filter(r => (r.worker_id || r.worker) === worker.id)
    const quantNames = [...new Set(workerRows.map(r => r.quant))]
    const band = <tr className="metrics-band" key={`band:${worker.id}`}><th><span className="metrics-pill" style={{ color: colorFor(worker.name, workerIndex) }}>{worker.name}</span></th><td colSpan={17}>Worker allocation matrix</td></tr>
    const matrixRows = quantNames.flatMap(quant => {
      const quantRows = workerRows.filter(r => r.quant === quant).sort((a, b) =>
        ((a.config === '4-bit') - (b.config === '4-bit')) || String(a.alloc_mode).localeCompare(String(b.alloc_mode)))
      const precisions = [...new Set(quantRows.map(r => r.config || 'standard'))]
      let quantOffset = 0
      return precisions.flatMap(precision => {
        const precisionRows = quantRows.filter(r => (r.config || 'standard') === precision)
        const gradeRow = precisionRows.find(r => r.detail && Object.keys(r.detail).length) || precisionRows[0]
        const rendered = precisionRows.map((r, allocationIndex) => {
          const constrained = r.status === 'hardware_constraint_failed'; const firstPrecision = allocationIndex === 0
          const firstQuant = quantOffset === 0 && firstPrecision
          const sourceTitle = constrained ? r.constraint_reason || r.error : r.metric_source ? `durable ${r.metric_source}` : 'normal central inference'
          return <tr key={`${worker.id}:${quant}:${precision}:${r.alloc_mode}`} className={`${constrained ? 'na ' : ''}${firstPrecision ? 'precision-start' : ''}`} title={sourceTitle}>
            {firstQuant && <td className="metrics-quant-cell" rowSpan={quantRows.length}><b>{quant}</b><small>{bytes(r.disk_bytes)}</small></td>}
            {firstQuant && <td rowSpan={quantRows.length}>{constrained ? 'N/A' : seconds(r.cold_s)}</td>}
            {firstPrecision && <td rowSpan={precisionRows.length}><span className={`metrics-tier-tag ${precision === '4-bit' ? 'bit4' : 'standard'}`}>{precision}</span></td>}
            <td>{constrained ? 'N/A' : r.alloc_mode}</td><td>{constrained ? 'N/A' : seconds(r.hot_load_s)}</td>
            <td>{constrained ? 'N/A' : metricValue(r.tok_s_avg ?? r.tok_s)}</td>
            {firstPrecision && BENCH_TASKS.map(t => <td rowSpan={precisionRows.length} key={t}>{constrained || gradeRow.legacy_grade ? 'N/A' : <TierBlocks value={gradeRow.detail?.[t]} />}</td>)}
            {firstPrecision && <><td rowSpan={precisionRows.length}>{constrained ? 'N/A' : gradeRow.task_best || 'N/A'}</td>
              <td rowSpan={precisionRows.length}>{constrained ? 'N/A' : gradeRow.task_worst || 'N/A'}</td>
              <td rowSpan={precisionRows.length} title={gradeRow.legacy_grade ? 'Legacy single-prompt score. Rerun this model for Easy/Medium/Hard grading.' : undefined}><b>{constrained ? 'N/A' : gradeRow.grade || 'N/A'}</b></td></>}
          </tr>
        })
        quantOffset += precisionRows.length
        return rendered
      })
    })
    return [band, ...matrixRows]
  })}</tbody></table></div>
}

function EvaluationLog({ calls, model }) {
  return <div className="metrics-call-section"><div><h3>Evaluation Call Log</h3><span className="metrics-sub">Historical executions retained for {model}</span></div>
    <div className="metrics-sheetwrap metrics-call-log"><table className="metrics-sheet"><thead><tr>
      <th>calls</th><th>model</th><th>quant</th><th>alloc</th><th>worker</th><th>tok/s</th><th>tok/s avg</th><th>ctx-in</th><th>ctx-out</th><th>task</th><th>date</th><th>time</th><th>caller</th>
    </tr></thead><tbody>{calls.map((c, i) => { const when = c.ts || c.timestamp || c.finished; const date = when ? new Date(Number(when) * 1000) : null
      const detail = c.error && c.error !== 'N/A' ? c.error : (c.output || '')
      return <tr key={`${c.request_id || c.task}:${i}`}><td>{i + 1}</td><td>{c.model || model}</td><td>{c.quant}</td><td>{c.alloc_mode || c.config || 'standard'}</td><td>{c.worker}</td>
        <td>{metricValue(c.tok_per_s ?? c.tok_s)}</td><td>{metricValue(c.tok_s_avg ?? c.tok_per_s)}</td><td>{c.prompt_tokens ?? c.ctx_in ?? 'N/A'}</td><td>{c.completion_tokens ?? c.ctx_out ?? 'N/A'}</td><td>{c.task}</td>
        <td>{date ? date.toLocaleDateString() : 'N/A'}</td><td>{date ? date.toLocaleTimeString() : 'N/A'}</td><td>{c.caller || 'orchestrator'}{detail && <details className="metrics-call-detail"><summary>details</summary><pre>{detail}</pre></details>}</td></tr> })}
    </tbody></table></div></div>
}

function BenchmarkWorkbook({ benchmark, historical = [], catalog = [] }) {
  const [model, setModel] = useState('')
  const [starting, setStarting] = useState(false)
  const [notice, setNotice] = useState('')
  const [hiddenWorkers, setHiddenWorkers] = useState([])
  const [modelHistory, setModelHistory] = useState({ calls: [], worker_averages: [] })
  const rows = useMemo(() => {
    const results = benchmark?.results || []
    const byKey = new Map(results.map(r => [benchKey(r), r]))
    const planned = (benchmark?.plan?.rows || []).map(p => ({ ...p, ...(byKey.get(benchKey(p)) || {}) }))
    const keys = new Set(planned.map(benchKey))
    const history = new Map(historical.map(h => [[h.worker, h.model_name, h.quant, h.alloc_mode].join('\0'), h]))
    const active = [...planned, ...results.filter(r => !keys.has(benchKey(r)))]
    const activeHistoryKeys = new Set(active.map(r => {
      const config = r.config || 'standard'
      const alloc = config === 'standard' || config === 'full' ? (r.alloc_mode || '') : `${config}:${r.alloc_mode || ''}`
      return [r.worker, r.model, r.quant, alloc].join('\0')
    }))
    const historicRows = historical.filter(h => (h.grade != null || h.grade_detail || h.alloc_mode) &&
      !/(^|:)(max_gpu|max_ram)$/.test(String(h.alloc_mode || '')) &&
      !activeHistoryKeys.has([h.worker, h.model_name, h.quant, h.alloc_mode].join('\0'))).map(h => {
      const parts = String(h.alloc_mode || '').split(':')
      if (parts[0] === 'undefined' || parts[0] === 'null') parts.shift()
      const prefixed = parts.length > 1 && ['4-bit', '4bit', 'moe', 'standard'].includes(parts[0])
      let detail = null
      try { detail = typeof h.grade_detail === 'string' ? JSON.parse(h.grade_detail) : h.grade_detail } catch {}
      const legacyGrade = !!detail && Object.values(detail).some(v => typeof v === 'number')
      return { model: h.model_name, worker: h.worker, worker_id: h.worker,
        quant: h.quant || 'runtime default', config: prefixed ? (parts[0] === '4bit' ? '4-bit' : parts[0]) : 'standard',
        alloc_mode: prefixed ? parts.slice(1).join(':') : h.alloc_mode || 'HugPy default',
        cold_s: h.cold_load_s, hot_load_s: h.hot_load_s, tok_s: h.tok_per_s,
        tok_s_avg: h.tok_per_s_avg, detail: legacyGrade ? null : detail, legacy_grade: legacyGrade,
        score: h.grade == null ? null : Math.round(Number(h.grade) * (legacyGrade ? 9 : 27) / 100),
        max: legacyGrade ? 9 : 27, grade: h.grade == null ? 'N/A' : legacyGrade
          ? `legacy ${Math.round(Number(h.grade) * 9 / 100)}/9`
          : `${Math.round(Number(h.grade) * 27 / 100)}/27`,
        n_samples: h.n_samples, finished: h.updated_at, metric_source: 'database', metric_updated_at: h.updated_at }
    })
    return [...active, ...historicRows].map(r => {
      const config = r.config || 'standard'
      const alloc = config === 'standard' || config === 'full' ? (r.alloc_mode || '') : `${config}:${r.alloc_mode || ''}`
      const h = history.get([r.worker, r.model, r.quant, alloc].join('\0'))
      if (!h) return r
      let historicDetail = null
      try { historicDetail = typeof h.grade_detail === 'string' ? JSON.parse(h.grade_detail) : h.grade_detail } catch {}
      const missing = v => v == null || v === '' || v === 'N/A'
      const failedRun = r.status === 'error' || r.status === 'failed' || !!(r.error && r.error !== 'N/A')
      if (failedRun) {
        return { ...r, cold_s: h.cold_load_s, hot_load_s: h.hot_load_s,
          tok_s: h.tok_per_s, tok_s_avg: h.tok_per_s_avg,
          detail: historicDetail, grade: h.grade == null ? 'N/A' : `${Math.round(h.grade * 27 / 100)}/27`,
          n_samples: h.n_samples, metric_source: 'database preserved after failed run', metric_updated_at: h.updated_at }
      }
      return { ...r, cold_s: missing(r.cold_s) ? h.cold_load_s : r.cold_s,
        hot_load_s: missing(r.hot_load_s) ? h.hot_load_s : r.hot_load_s,
        tok_s: missing(r.tok_s) ? h.tok_per_s : r.tok_s,
        tok_s_avg: missing(r.tok_s_avg) ? h.tok_per_s_avg : r.tok_s_avg,
        detail: r.legacy_grade ? null : r.detail || historicDetail, n_samples: r.n_samples ?? h.n_samples,
        grade: r.grade === 'N/A' && h.grade != null ? `${Math.round(h.grade * 27 / 100)}/27` : r.grade,
        metric_source: 'database fallback', metric_updated_at: h.updated_at }
    })
  }, [benchmark, historical])
  const models = useMemo(() => [...new Set([...catalog, ...rows.map(r => r.model)].filter(Boolean))].sort(), [catalog, rows])
  const coverage = useMemo(() => {
    const suites = new Map()
    for (const call of (benchmark?.calls || [])) {
      const key = benchKey(call)
      if (!suites.has(key)) suites.set(key, new Set())
      suites.get(key).add(String(call.task || '').replace(/ \([^)]*\)$/, ''))
    }
    const out = {}
    for (const m of models) {
      const planned = new Set(rows.filter(r => r.model === m).map(benchKey)).size
      const modelSuites = [...suites.entries()].filter(([key]) => key.split('\0')[1] === m)
      const complete = modelSuites.filter(([, tasks]) => BENCH_TASKS.every(t => tasks.has(t))).length
      const calls = modelSuites.reduce((n, [, tasks]) => n + tasks.size, 0)
      out[m] = { planned, complete, calls,
        status: complete ? (complete >= planned ? 'tested' : 'tested · matrix incomplete')
          : calls ? 'attempted · suite incomplete' : 'not tested' }
    }
    return out
  }, [benchmark?.calls, models, rows])
  const untestedModels = useMemo(() => models.filter(m => !coverage[m]?.complete), [models, coverage])
  useEffect(() => {
    if (models.length && !models.includes(model)) setModel(models.includes(benchmark?.progress?.model) ? benchmark.progress.model : models[0])
  }, [models, model, benchmark?.progress?.model])
  useEffect(() => {
    if (!model) return
    fetchJson(`/api/llm/models/${encodeURIComponent(model)}/metrics?limit=500`)
      .then(value => setModelHistory(value || { calls: [], worker_averages: [] }))
      .catch(() => setModelHistory({ calls: [], worker_averages: [] }))
  }, [model])
  const page = useMemo(() => rows.filter(r => r.model === model), [rows, model])
  const workers = useMemo(() => {
    const out = new Map(); page.forEach(r => out.set(r.worker_id || r.worker, { id: r.worker_id || r.worker, name: r.worker || r.worker_id }))
    return [...out.values()].sort((a, b) => String(a.name).localeCompare(String(b.name)))
  }, [page])
  const visibleWorkers = workers.filter(w => !hiddenWorkers.includes(w.id))
  const visiblePage = page.filter(r => visibleWorkers.some(w => w.id === (r.worker_id || r.worker)))
  const liveCalls = (benchmark?.calls || []).filter(c => c.model === model)
  const modelCalls = [...liveCalls, ...(modelHistory.calls || [])]
    .filter(c => visibleWorkers.some(w => w.id === (c.worker_id || c.worker)))
  const index = Math.max(0, models.indexOf(model))
  const runModel = () => {
    if (!model || benchmark?.status === 'running' || starting) return
    if (!confirm(`Test ${model} through HugPy's normal central inference path on each designated worker?\n\nThe test does not change placement, transfer, loading, or eviction behavior.`)) return
    setStarting(true); setNotice('starting model benchmark…')
    fetchJson('/api/llm/benchmark/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ executor: 'hugpy-central', tokens: 128, models: [model], workers: [] }) })
      .then(() => setNotice(`started HugPy-native test for ${model}`))
      .catch(e => setNotice(`could not start: ${e.message}`))
      .finally(() => setStarting(false))
  }
  const resumeUntested = () => {
    if (!untestedModels.length || benchmark?.status === 'running' || starting) return
    if (!confirm(`Test ${untestedModels.length} models with no completed nine-task suite through normal HugPy inference?\n\nAlready tested models will not be included.`)) return
    setStarting(true); setNotice(`starting ${untestedModels.length} untested models…`)
    fetchJson('/api/llm/benchmark/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ executor: 'hugpy-central', tokens: 128, models: untestedModels, workers: [] }) })
      .then(() => setNotice(`started resume run for ${untestedModels.length} untested models`))
      .catch(e => setNotice(`could not resume: ${e.message}`))
      .finally(() => setStarting(false))
  }
  if (!benchmark || !rows.length) return <section className="metrics-card metrics-workbook"><h3>Benchmark workbook</h3><p className="metrics-muted">No benchmark matrix has been retained yet.</p></section>
  return <section className="metrics-card metrics-workbook">
    <div className="metrics-workbook-head"><div><h3>Benchmark workbook</h3>
      <span className="metrics-sub">{benchmark.status} · {benchmark.progress?.completed || 0}/{benchmark.progress?.total || benchmark.plan?.runnable || 0} allocation variations</span></div>
      <div className="metrics-model-nav"><button disabled={!index} onClick={() => setModel(models[index - 1])}>←</button>
        <select value={model} onChange={e => setModel(e.target.value)}>{models.map(m => <option key={m} value={m}>{m} — {coverage[m]?.status}</option>)}</select>
        <span>{index + 1}/{models.length}</span><button disabled={index >= models.length - 1} onClick={() => setModel(models[index + 1])}>→</button>
        <button className="metrics-test-model" disabled={benchmark?.status === 'running' || starting} onClick={runModel}
          title={benchmark?.status === 'running' ? 'Wait for the active fleet benchmark to finish' : `Test ${model} only`}>
          {starting ? 'starting…' : 'Test this model'}</button></div>
    </div>
    <div className="metrics-worker-toggles"><span>Workers</span>{workers.map(w => <button key={w.id}
      className={hiddenWorkers.includes(w.id) ? '' : 'active'} onClick={() => setHiddenWorkers(old => old.includes(w.id) ? old.filter(id => id !== w.id) : [...old, w.id])}>{w.name}</button>)}</div>
    <div className="metrics-coverage">
      <span><b>{models.filter(m => coverage[m]?.complete).length}</b> tested</span>
      <span><b>{models.filter(m => !coverage[m]?.complete && coverage[m]?.calls).length}</b> incomplete suites</span>
      <span><b>{models.filter(m => !coverage[m]?.calls).length}</b> no graded calls</span>
      <span>selected: <b>{coverage[model]?.complete || 0}/{coverage[model]?.planned || 0}</b> complete suites · {coverage[model]?.calls || 0} distinct calls</span>
      <button disabled={benchmark?.status === 'running' || starting || !untestedModels.length} onClick={resumeUntested}
        title={benchmark?.status === 'running' ? 'Stop or finish the active run first' : 'Exclude every model with a completed nine-task suite'}>
        Resume untested models ({untestedModels.length})
      </button>
    </div>
    {notice && <p className="metrics-run-notice">{notice}</p>}
    <WorkerSummary rows={visiblePage} calls={modelCalls} workers={visibleWorkers} lifetime={modelHistory.worker_averages || []} />
    <div className="metrics-matrix-head"><div><h3>Dual Cognitive Grading Matrix</h3><span className="metrics-sub">9 categories · all difficulties tested independently (Easy | Medium | Hard)</span></div>
      <div className="metrics-tier-legend"><span><i className="pass" />Pass</span><span><i className="fail" />Fail</span><span><i className="metrics-tier-untested" />Untested</span></div></div>
    <GradingMatrix rows={visiblePage} workers={visibleWorkers} />
    <EvaluationLog calls={modelCalls} model={model} />
  </section>
}

function Legend({ palette }) {
  const workers = Object.keys(palette)
  if (!workers.length) return null
  return (
    <div className="metrics-legend">
      {workers.map((w) => <span key={w}><i style={{ background: palette[w] }} />{w}</span>)}
    </div>
  )
}

function SamplesChart({ rows }) {
  const byModel = {}
  for (const r of rows) byModel[r.model_name] = (byModel[r.model_name] || 0) + (r.n_samples || 0)
  const top = Object.entries(byModel).map(([model, n]) => ({ model, n }))
    .sort((a, b) => b.n - a.n).slice(0, 12)
  if (!top.length) return <p className="metrics-muted">No samples recorded yet.</p>
  const max = Math.max(...top.map(r => r.n), 1)
  const W = 500, rowH = 26, padL = 150, padR = 56, bw = W - padL - padR
  const H = 12 + top.length * rowH
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="metrics-svg" role="img" aria-label="samples per model">
      {top.map((r, i) => {
        const y = 6 + i * rowH, w = Math.max(2, (r.n / max) * bw)
        return (
          <g key={r.model}>
            <text className="m-label" x={4} y={y + rowH / 2 + 4}>{shortModel(r.model).slice(0, 24)}</text>
            <rect x={padL} y={y + 5} width={w} height={rowH - 12} rx={4} fill="#199e70" />
            <text className="m-val" x={padL + w + 6} y={y + rowH / 2 + 4}>{r.n}</text>
          </g>
        )
      })}
    </svg>
  )
}

function WorkerChart({ rows, metric, palette }) {
  const workers = Object.keys(palette)
  const grouped = useMemo(() => {
    const g = {}
    for (const r of rows) {
      const cur = (g[r.model_name] ||= {})[r.worker]
      const v = r[metric]
      // one model x worker can have several quant/alloc_mode rows — keep the
      // fastest measured for the chart; the full sheet below lists every row.
      if (v != null && (cur == null || v > cur)) (g[r.model_name])[r.worker] = v
    }
    return g
  }, [rows, metric])
  const models = Object.keys(grouped)
    .sort((a, b) => Math.max(...vals(grouped[b])) - Math.max(...vals(grouped[a]))).slice(0, 12)
  if (!models.length) return <p className="metrics-muted">No throughput recorded yet.</p>
  const allv = rows.map(r => r[metric]).filter(v => v != null && !isBad(v))
  const max = Math.max(...allv, 1)
  const rowH = Math.max(24, workers.length * 15 + 10)
  const W = 500, padL = 160, padR = 52, bw = W - padL - padR
  const H = 12 + models.length * rowH
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="metrics-svg" role="img" aria-label={`${metric} by worker`}>
      {models.map((m, i) => {
        const y = 6 + i * rowH
        return (
          <g key={m}>
            <text className="m-label" x={4} y={y + rowH / 2 + 4}>{shortModel(m).slice(0, 26)}</text>
            {workers.map((w, wi) => {
              const v = grouped[m][w]; if (v == null) return null
              const bad = isBad(v), ww = bad ? 6 : Math.max(2, (v / max) * bw)
              const by = y + 4 + wi * ((rowH - 10) / workers.length)
              const bh = (rowH - 10) / workers.length - 3
              return (
                <g key={w}>
                  <rect x={padL} y={by} width={ww} height={bh} rx={3} fill={palette[w]} opacity={bad ? 0.4 : 1} />
                  <text className="m-val m-small" x={padL + ww + 5} y={by + bh / 2 + 3}>{bad ? '∞*' : fmt(v)}</text>
                </g>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}
function vals(o) { return Object.values(o).filter(v => v != null && !isBad(v)).concat([0]) }

function LiveTable({ rows, palette }) {
  const [sortKey, setSortKey] = useState('updated_at')
  const [dir, setDir] = useState(-1)
  const cols = [
    ['model_name', 'Model'], ['quant', 'Quant'], ['alloc_mode', 'Alloc / distribution'],
    ['worker', 'Worker'], ['tok_per_s', 'tok/s', 1], ['tok_per_s_avg', 'tok/s avg', 1],
    ['n_samples', 'n', 1], ['task', 'Task'], ['updated_at', 'Updated', 1],
  ]
  const sorted = [...rows].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey]
    if (typeof x === 'string' || typeof y === 'string') return dir * String(x || '').localeCompare(String(y || ''))
    return dir * ((x ?? -1) - (y ?? -1))
  })
  const click = (k) => { if (k === sortKey) setDir(-dir); else { setSortKey(k); setDir(-1) } }
  if (!rows.length) return <p className="metrics-muted">No live metrics yet — this fills as the fleet serves models (db-hugpy `model_metrics`, HUGPY_REGISTRY_DB=pg).</p>
  return (
    <div className="metrics-tablewrap">
      <table className="metrics-table">
        <thead><tr>{cols.map(([k, lab, num]) =>
          <th key={k} className={num ? 'num' : ''} onClick={() => click(k)}>
            {lab}{sortKey === k ? (dir < 0 ? ' ▾' : ' ▴') : ''}</th>)}</tr></thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i}>
              <td className="m-model" title={r.model_name}>{shortModel(r.model_name)}</td>
              <td>{fmtStr(r.quant)}</td>
              <td>{fmtStr(r.alloc_mode)}</td>
              <td><span className="metrics-pill" style={{ background: `color-mix(in srgb, ${palette[r.worker] || '#888'} 24%, transparent)`, color: palette[r.worker] || '#888' }}>{r.worker}</span></td>
              <td className="num">{fmt(r.tok_per_s)}</td>
              <td className="num">{fmt(r.tok_per_s_avg)}</td>
              <td className="num">{r.n_samples ?? '—'}</td>
              <td>{fmtStr(r.task)}</td>
              <td className="num">{r.updated_at ? new Date(r.updated_at * 1000).toLocaleString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
