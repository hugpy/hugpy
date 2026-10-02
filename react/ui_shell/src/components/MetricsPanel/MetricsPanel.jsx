import { Component, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchJson } from '../../api'
import { hugpyFetch } from '../../runtime/config'
import {
  ALL_SUITE_TASKS, EMPTY_FILTERS, GRADE_BUCKETS, buildModelIndex, canonicalRows, filterModels,
  filterOptions, filterRows, firstLine, hasActiveFilters, isUntested, parseQuery,
  resolveSelection, statusChips, writeQuery,
} from './modelFilters'
import { useModelStatus } from '../ModelTable/useModelStatus'
import {
  combineThroughput, fromLedger, gradedCallRows, matrixFor, resolveSuite, scoreText, suiteIndex, throughputText, suiteForTask } from './suites'
import { statusFor, workerChips, worthView } from '../ModelTable/modelStatus'
import { GradedItem } from './GradedItem'
import RunError from './RunError'
import ModelLiveState from '../ModelLiveState/ModelLiveState'
import LoadModelPicker from '../ModelPicker/ModelPicker'
import { LogBlock } from '../Diagnostics/Diagnostics'
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
// A ledger row now relays EVERY key of the persisted call, including nested
// sub-objects of `state` (state.model = {backend, loaded_at_pick, model_key,
// size_bytes}, and state.alloc/vram/request/outcome). A table cell that assumed
// a scalar would render such an object as a raw React child and throw React #31
// ("Objects are not valid as a React child"), which unmounts the whole app.
// scalar() keeps a scalar as-is and renders any object/array whole as pretty
// JSON in a <pre> — data is shown, never dropped, never "[object Object]".
const scalar = (v) => (v != null && typeof v === 'object'
  ? <pre className="call-cell-json">{(() => { try { return JSON.stringify(v, null, 1) } catch { return String(v) } })()}</pre>
  : v)
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

const MODEL_STORE = 'hugpy.metrics.model'
const QUEUE_STORE = 'hugpy.metrics.queue'
const readStore = (store, k, fallback) => { try { const v = store.getItem(k); return v == null ? fallback : JSON.parse(v) } catch { return fallback } }
const writeStore = (store, k, v) => { try { if (v == null || v === '' || (Array.isArray(v) && !v.length)) store.removeItem(k); else store.setItem(k, JSON.stringify(v)) } catch { /* private mode / quota */ } }
const safeLocal = () => { try { return window.localStorage } catch { return null } }
const safeSession = () => { try { return window.sessionStorage } catch { return null } }

// POST that keeps the body on non-2xx: the benchmark route answers 409 WITH the
// active run's state, which the plain fetchJson helper throws away.
async function postJson(url, body) {
  try {
    const r = await hugpyFetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
    const text = await r.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = null }
    const message = (data && (data.error || data.detail || data.message)) || (!data && text.trim()) || `HTTP ${r.status}`
    return { ok: r.ok, status: r.status, data, message }
  } catch (e) {
    return { ok: false, status: 0, data: null, message: String(e && e.message || e) }
  }
}

// The grading-suite registry (GET /llm/benchmark/suites), fetched once per
// page; an older central (no route) keeps the offline FALLBACK_SUITES table.
let _suitesCache = null
let _suitesErr = null   // why the live suite registry is not in use, when it isn't
function useSuites() {
  const [idx, setIdx] = useState(() => _suitesCache || suiteIndex(null))
  useEffect(() => {
    if (_suitesCache) return
    fetchJson('/api/llm/benchmark/suites')
      .then(d => { if (d && Array.isArray(d.suites)) { _suitesCache = suiteIndex(d.suites); _suitesErr = null; setIdx(_suitesCache) }
        else _suitesErr = `GET /api/llm/benchmark/suites returned no suites array (built-in fallback suites in use)` })
      .catch(e => { _suitesErr = `GET /api/llm/benchmark/suites failed: ${e?.message || e} (built-in fallback suites in use)` })
  }, [])
  return idx
}

// Defense in depth: a single bad render inside the metrics tab must NOT unmount
// the whole console (React #31 did exactly that when a nested state.* object hit
// a JSX child). This boundary catches any render throw in the panel and shows
// the real error message + stack inline instead of blanking the app. The scalar
// guards above are the primary fix; this is the backstop for the next bad field.
class MetricsErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null, info: null } }
  static getDerivedStateFromError(error) { return { error } }
  componentDidCatch(error, info) { this.setState({ info }); try { console.error('MetricsPanel render error', error, info) } catch { /* noop */ } }
  render() {
    if (!this.state.error) return this.props.children
    const e = this.state.error
    const text = [String(e?.message || e), e?.stack || '', this.state.info?.componentStack || ''].filter(Boolean).join('\n\n')
    return <div className="metrics-panel">
      <section className="metrics-card metrics-error">
        <h3>The metrics tab hit a render error</h3>
        <p className="metrics-sub">The rest of the console is intact; this panel stopped rendering to avoid taking the app down. The exact error, whole:</p>
        <pre className="call-cell-json">{text}</pre>
      </section>
    </div>
  }
}

// `models` (optional) = the /api/models catalog rows App already holds; when
// the panel is embedded without it, it fetches the catalog itself.
export default function MetricsPanel(props = {}) {
  return <MetricsErrorBoundary><MetricsPanelBody {...props} view="metrics" /></MetricsErrorBoundary>
}

// The per-model tester (run picker, live output, filters, benchmark workbook)
// lives on the Grader tab; Metrics keeps the fleet-wide charts and live sheet.
// `benchmark` = the Grader's own latest benchmark state, applied immediately so
// a run it starts shows here without waiting for the next poll.
export function GraderWorkbook(props = {}) {
  return <MetricsErrorBoundary><MetricsPanelBody {...props} view="grader" /></MetricsErrorBoundary>
}

function MetricsPanelBody({ models: catalogProp, view = 'metrics', benchmark: liveBenchmark, benchmarkWorkers,
  gradeModels = [], onGradeModelsChange } = {}) {
  const grader = view === 'grader'
  const topSuites = useSuites()
  const [data, setData] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [error, setError] = useState(null)
  const [benchmarkError, setBenchmarkError] = useState(null)
  const [v1Models, setV1Models] = useState([])
  const [v1Ok, setV1Ok] = useState(false)
  const [fetchedCatalog, setFetchedCatalog] = useState([])

  // Selection + filters: URL query first (shareable), then localStorage.
  // `chosen` = the user picked it (click / URL / storage); only a chosen model
  // is sticky — an unchosen one may follow the running benchmark.
  const initial = useMemo(() => parseQuery(typeof window === 'undefined' ? '' : window.location.search), [])
  const [model, setModel] = useState(() => initial.model || readStore(safeLocal(), MODEL_STORE, '') || '')
  const [chosen, setChosen] = useState(() => !!(initial.model || readStore(safeLocal(), MODEL_STORE, '')))
  const [filters, setFilters] = useState(initial.filters)
  const [dropped, setDropped] = useState('')
  const selectModel = useCallback((m) => { setModel(m); setChosen(true); setDropped('') }, [])

  // A POST's benchmark state must not be overwritten by a status poll that was
  // already in flight when the POST was sent (that is the "blip": the button
  // flips back to idle until the next poll). Every applied POST bumps the
  // generation; poll replies from an older generation are dropped.
  const benchGen = useRef(0)
  const applyBenchmark = useCallback((d) => { if (d && typeof d === 'object') { benchGen.current++; setBenchmark(d); setBenchmarkError(null) } }, [])
  useEffect(() => { if (liveBenchmark?.status) applyBenchmark(liveBenchmark) }, [liveBenchmark, applyBenchmark])
  // Every run is archived server-side (GET /llm/benchmark/runs); a past run
  // can be reopened whole, so a new run never hides the previous run's calls.
  const [runs, setRuns] = useState([])
  const [runsError, setRunsError] = useState(null)
  const [runView, setRunView] = useState('live')
  const [pastRun, setPastRun] = useState(null)
  const loadRuns = useCallback(() => {
    fetchJson('/api/llm/benchmark/runs')
      .then(d => { if (Array.isArray(d?.runs)) { setRuns(d.runs); setRunsError(d.errors?.length ? `${d.errors.length} archived run file(s) unreadable: ${d.errors.map(e => `${e.file}: ${e.error}`).join('; ')}` : null) } else setRunsError(d?.reason || d?.error || `GET /llm/benchmark/runs returned no runs array (${JSON.stringify(d).slice(0, 200)})`) })
      .catch(e => setRunsError(`GET /llm/benchmark/runs failed: ${String(e?.message || e)}`))
  }, [])
  useEffect(() => { loadRuns() }, [loadRuns, benchmark?.run_id, benchmark?.status])
  const selectRun = useCallback((id) => {
    setRunView(id)
    if (id === 'live') { setPastRun(null); return }
    fetchJson(`/api/llm/benchmark/runs/${encodeURIComponent(id)}`)
      .then(d => setPastRun(d && d.run_id ? d : { run_id: id, status: 'unreadable', error: d?.error || `GET /llm/benchmark/runs/${id} returned no run`, results: [], calls: [], events: [] }))
      .catch(e => setPastRun({ run_id: id, status: 'unreadable', error: `GET /llm/benchmark/runs/${id} failed: ${String(e?.message || e)}`, results: [], calls: [], events: [] }))
  }, [])
  const shownRun = runView === 'live' ? benchmark : (pastRun || benchmark)

  const catalogRows = Array.isArray(catalogProp) ? catalogProp : fetchedCatalog
  const loadCatalog = useCallback(() => {
    if (Array.isArray(catalogProp)) return
    fetchJson('/api/models').then(d => { if (Array.isArray(d) && d.length) setFetchedCatalog(d) }).catch(() => {})
  }, [catalogProp])
  useEffect(() => { loadCatalog() }, [loadCatalog])

  const load = useCallback(() => {
    // NOTE: the /api prefix — the SPA's hugpyFetch resolves /api/llm/... (same
    // as every other panel, e.g. WorkersPanel's /api/llm/workers). A bare
    // /llm/... does not resolve and the panel would render empty.
    //
    // model-metrics2 (t146) is the LIVE db-hugpy `model_metrics` sheet — one
    // row per (model x quant x alloc_mode x worker). The OLDER /llm/model-metrics
    // (db-toolserver EMA snapshot) is abandoned here: that store is on the
    // wrong database with a role that lacks CREATE, so it never had rows.
    //
    // A POLL MUST NEVER DESTROY GOOD DATA (same rule as App's model list): a
    // degraded reply keeps the last good rows and only records the error, so
    // the model list (and the selection) cannot collapse on one bad poll.
    fetchJson('/api/llm/model-metrics2?limit=5000')   // whole sheet (~1k rows); the 500 default dropped models
      .then((d) => {
        setData(prev => (d?.error && !(d.rows || []).length && prev?.rows?.length) ? { ...prev, error: d.error } : d)
        setError(null)
      })
      .catch((e) => setError(String(e && e.message || e)))
    const gen = benchGen.current
    fetchJson('/api/llm/benchmark/status')
      .then((d) => { if (gen === benchGen.current) { setBenchmark(d); setBenchmarkError(null) } })
      .catch((e) => setBenchmarkError(String(e && e.message || e)))
    fetchJson('/api/v1/models')
      .then(d => { const list = (d?.data || []).filter(m => m && m.id); if (list.length) { setV1Models(list); setV1Ok(true) } })
      .catch(() => {})
  }, [])
  useEffect(() => {
    load()
    const t = setInterval(load, benchmark?.status === 'running' ? 2500 : 15000)
    return () => clearInterval(t)
  }, [benchmark?.status, load])

  const rows = useMemo(() => data?.rows || [], [data])
  const benchModels = useMemo(() => [...(benchmark?.results || []), ...(benchmark?.plan?.rows || [])].map(r => r.model),
    [benchmark?.results, benchmark?.plan?.rows])
  const { canon, infos } = useMemo(() => buildModelIndex({ metricsRows: rows, catalogRows, v1Models, extraModels: benchModels }),
    [rows, catalogRows, v1Models, benchModels])
  const navList = useMemo(() => filterModels(infos, { filters }), [infos, filters])
  const options = useMemo(() => filterOptions(infos), [infos])
  const tableRows = useMemo(() => filterRows(rows, infos, canon, filters), [rows, infos, canon, filters])
  const historical = useMemo(() => canonicalRows(rows, canon), [rows, canon])

  // Sticky selection: see resolveSelection (modelFilters.js). "ready" = the
  // catalog and the metrics sheet both loaded cleanly — the only state in which
  // a missing model is really gone rather than a transient/partial poll.
  const running = benchmark?.status === 'running' ? (benchmark?.progress?.model || (benchmark?.models || [])[0] || '') : ''
  // Grader: a newly started run takes the display to the model under test and
  // follows it model to model; picking a model during the run stops following
  // (the "follow live" button resumes it).
  const followedRun = useRef(null)
  useEffect(() => {
    if (!grader || benchmark?.status !== 'running' || !benchmark?.run_id) return
    if (followedRun.current !== benchmark.run_id) { followedRun.current = benchmark.run_id; setChosen(false) }
  }, [grader, benchmark?.status, benchmark?.run_id])
  const followLive = useCallback(() => { setChosen(false); setDropped('') }, [])
  const ready = (v1Ok || catalogRows.length > 0) && !!data && !data.error && !error
  useEffect(() => {
    const r = resolveSelection({ current: model, chosen, known: new Set(infos.keys()), ready, running: canon(running), fallback: navList[0] || '' })
    if (r.model !== model) setModel(r.model)
    if (r.chosen !== chosen) setChosen(r.chosen)
    if (r.dropped) setDropped(r.dropped)
  }, [model, chosen, infos, ready, running, navList, canon])

  // Persist: URL query (model only when user-chosen, plus every filter) and
  // localStorage for the model. replaceState — no history spam, no reload.
  useEffect(() => {
    writeStore(safeLocal(), MODEL_STORE, chosen ? model : '')
    try {
      const { pathname, search, hash } = window.location
      const next = new URLSearchParams(writeQuery(search, { model: chosen ? model : '', filters }))
      if (chosen || hasActiveFilters(filters)) next.set('tab', grader ? 'review' : 'metrics')
      const s = next.toString() ? `?${next}` : ''
      if (s !== search) window.history.replaceState(window.history.state, '', `${pathname}${s}${hash}`)
    } catch { /* non-browser host */ }
  }, [model, chosen, filters, grader])

  const refreshAll = () => { load(); loadCatalog() }

  if (!data && !benchmark) return <div className="metrics-panel"><p className="metrics-muted">Loading {grader ? 'benchmark workbook' : 'metrics'}…</p></div>

  const modelCount = new Set(rows.map(r => r.model_name)).size
  const workers = new Set(rows.map(r => r.worker)).size
  const palette = workerPalette(rows)
  const dataError = data?.error

  return (
    <div className="metrics-panel">
      <div className="metrics-head">
        <div className="metrics-sub">
          {rows.length} rows · {modelCount} models · {workers} workers
          {dataError ? <span className="metrics-error"> · {dataError}</span> : null}
          {data?.generated_at ? ` · generated ${new Date(data.generated_at * 1000).toLocaleTimeString()}` : ''}
          {error ? <span className="metrics-error"> · historical metrics: {error}</span> : null}
          {benchmarkError ? <span className="metrics-error"> · benchmark metrics: {benchmarkError}</span> : null}
        </div>
        <a className="metrics-link" href="#" onClick={(e) => { e.preventDefault(); refreshAll() }}>↻ refresh</a>
      </div>

      <FilterBar filters={filters} setFilters={setFilters} options={options} shown={navList.length} total={infos.size} suites={topSuites} />
      {grader ? <>
        <RunPicker runs={runs} runsError={runsError} value={runView} onChange={selectRun} live={benchmark} />
        <BenchmarkLiveOutput benchmark={shownRun} onRefresh={load} onBenchmark={applyBenchmark} />
        {dropped && <p className="metrics-run-notice metrics-error">{dropped} is no longer in the catalog — selection reset. <a className="metrics-link" href="#" onClick={e => { e.preventDefault(); setDropped('') }}>dismiss</a></p>}
        {running && runView === 'live' && <p className="metrics-run-notice">
          {!chosen && canon(running) === model ? <>● following the run: <b>{model}</b> is being tested now</>
            : <>● the run is testing <b>{canon(running)}</b> · you are viewing {model || 'no model'} <button className="metrics-toggle" onClick={followLive}>follow live</button></>}</p>}
        <BenchmarkWorkbook benchmark={shownRun} historical={historical} infos={infos} canon={canon}
          navList={navList} model={model} onSelect={selectModel} onBenchmark={applyBenchmark} onRefresh={load}
          benchmarkWorkers={benchmarkWorkers} gradeModels={gradeModels} onGradeModelsChange={onGradeModelsChange} />
      </> : <p className="metrics-muted">Per-model testing and the grading matrix are on the <b>Grader</b> tab.</p>}

      {!grader && <>
      <div className="metrics-grid">
        <section className="metrics-card">
          <h3>Most-active models <span className="metrics-sub">by sample count</span></h3>
          <SamplesChart rows={rows} />
        </section>
        <section className="metrics-card">
          <div className="metrics-card-head">
            <h3>Throughput <span className="metrics-sub">tok/s by worker — mean of every recorded call (Σtokens / Σseconds), n beside it</span></h3>
          </div>
          <Legend palette={palette} />
          <WorkerChart rows={rows} palette={palette} />
        </section>
      </div>

      <section className="metrics-card">
        <h3>model_metrics <span className="metrics-sub">live sheet — db-hugpy, per model x quant x alloc_mode x worker (click a header to sort){hasActiveFilters(filters) ? ` · filtered: ${tableRows.length} of ${rows.length} rows` : ''}</span></h3>
        <LiveTable rows={tableRows} palette={palette} />
      </section>
      </>}
    </div>
  )
}

// Which run the workbook shows: the live one, or any archived run (whole).
function RunPicker({ runs, runsError, value, onChange, live }) {
  const when = (t) => (t ? new Date(Number(t) * 1000).toLocaleString() : 'not started')
  return <div className="metrics-runpicker">
    <label className="metrics-filter"><span>run</span>
      <select value={value} onChange={e => onChange(e.target.value)}>
        <option value="live">live · {live?.run_id || 'no run'} · {live?.status || 'idle'} · {(live?.models || []).join(', ') || 'all models'}</option>
        {runs.filter(r => r.run_id && r.run_id !== live?.run_id).map(r =>
          <option key={r.run_id} value={r.run_id}>{when(r.started)} · {r.run_id} · {r.status} · {(r.models || []).join(', ') || 'all models'} · {r.results} results · {r.calls} calls{r.error ? ` · ${r.error}` : ''}</option>)}
      </select></label>
    <span className="metrics-sub">{runs.length} archived run{runs.length === 1 ? '' : 's'}{runsError ? <span className="metrics-error"> · {runsError}</span> : null}</span>
    {value !== 'live' && <span className="metrics-sub"> · viewing an archived run (read-only); switch to live to test</span>}
  </div>
}

function FilterBar({ filters, setFilters, options, shown, total, suites }) {
  const set = (k, v) => setFilters(f => ({ ...f, [k]: v }))
  const toggleGrade = (g) => setFilters(f => ({ ...f, grade: f.grade.includes(g) ? f.grade.filter(x => x !== g) : [...f.grade, g] }))
  const sel = (k, label, values) => <label className="metrics-filter"><span>{label}</span>
    <select value={filters[k]} onChange={e => set(k, e.target.value)}>
      <option value="">any</option>{values.map(v => <option key={v} value={v}>{v}</option>)}
    </select></label>
  // Collapsed by default so the ~9 controls don't dominate the top of the panel
  // (they weren't in the pre-overhaul look). The summary line stays useful
  // collapsed: model count + whether filters are active. Every control is
  // unchanged, just tucked into the open state.
  const active = hasActiveFilters(filters)
  return <details className="metrics-filters-wrap" open={active}>
    <summary className="metrics-filters-summary">
      <span>Filters{active ? ' · active' : ''}</span>
      <span className="metrics-sub">{active ? `${shown} of ${total} models` : `${total} models`}</span>
    </summary>
    <div className="metrics-filters" role="group" aria-label="Model filters">
      <span className="metrics-filter"><span>grade</span>
        <span className="metrics-chipset">{GRADE_BUCKETS.map(([k, label]) =>
          <button key={k} type="button" aria-pressed={filters.grade.includes(k)} className={filters.grade.includes(k) ? 'active' : ''} onClick={() => toggleGrade(k)}>{label}</button>)}</span></span>
      {sel('pf', 'result', ['pass', 'fail'])}
      <label className="metrics-filter" title="the model's task (same nomenclature as the Models table); a task with no grading suite cannot be graded automatically and is said so"><span>task</span>
        <select value={filters.task} onChange={e => set('task', e.target.value)}>
          <option value="">any</option>
          {(options.tasks || []).map(({ task, count }) => { const su = suiteForTask(suites, task)
            return <option key={task} value={task === '(no task recorded)' ? '' : task}>{task} ({count}) — {su ? `suite ${su.name}` : 'NO SUITE: not gradable'}</option> })}
        </select></label>
      {sel('cat', 'grade category', ALL_SUITE_TASKS)}
      {sel('fw', 'framework', options.fw)}
      {sel('worker', 'worker', options.workers)}
      {!!options.verdicts.length && sel('verdict', 'verdict', options.verdicts)}
      {!!options.adm.length && sel('adm', 'admission', options.adm)}
      <label className="metrics-filter metrics-filter-check"><input type="checkbox" checked={filters.untested} onChange={e => set('untested', e.target.checked)} /><span>untested only</span></label>
      <span className="metrics-sub">{active ? `${shown} of ${total} models` : `${total} models`}</span>
      {active && <a className="metrics-link" href="#" onClick={e => { e.preventDefault(); setFilters({ ...EMPTY_FILTERS, grade: [] }) }}>clear filters</a>}
    </div>
  </details>
}

// The status read trims events to the last 200; GET /llm/benchmark/events
// relays every event of the live run, whole (2026-09-23).
function useAllBenchmarkEvents(benchmark) {
  const [all, setAll] = useState(null)
  const runId = benchmark?.run_id
  const n = (benchmark?.events || []).length
  useEffect(() => {
    if (!runId) { setAll(null); return }
    let live = true
    fetchJson('/api/llm/benchmark/events')
      .then(d => { if (live) setAll(d && d.run_id === runId && Array.isArray(d.events) ? d : null) })
      .catch(e => { if (live) setAll({ error: String(e?.message || e) }) })
    return () => { live = false }
  }, [runId, n])
  return all
}

function BenchmarkLiveOutput({ benchmark, onRefresh, onBenchmark }) {
  const [cancelling, setCancelling] = useState(false)
  const allEvents = useAllBenchmarkEvents(benchmark)
  // Collapsed-by-default output pane. Same run-collapse rule as the Fleet
  // capacity initiator: a NEW run auto-expands (and clears any prior manual
  // collapse); after it finishes the pane stays as the operator left it; a
  // manual toggle mid-run is respected until the next run starts; fresh load
  // with nothing running → collapsed.
  const [liveOpen, setLiveOpen] = useState(false)
  const livePrevRun = useRef(null)
  useEffect(() => {
    if (benchmark?.status === 'running' && benchmark?.run_id && benchmark.run_id !== livePrevRun.current) {
      livePrevRun.current = benchmark.run_id
      setLiveOpen(true)
    }
  }, [benchmark?.status, benchmark?.run_id])
  if (!benchmark) return null
  const progress = benchmark.progress || {}
  const calls = benchmark.calls || []
  const fullEvents = Array.isArray(allEvents?.events) && allEvents.events.length >= (benchmark.events || []).length
  const events = fullEvents ? allEvents.events : (benchmark.events || [])
  const results = benchmark.results || []
  const current = [progress.worker || progress.worker_id, progress.model, progress.quant,
    progress.config || progress.alloc_mode].filter(Boolean).join(' · ')
  const cancel = () => {
    if (benchmark.status !== 'running' || cancelling) return
    if (!confirm('Cancel the active capacity test? The result rows already recorded will be retained.')) return
    setCancelling(true)
    fetchJson('/api/llm/benchmark/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: 'execution' }) })
      .then(d => { if (d && d.status && onBenchmark) onBenchmark(d); onRefresh() })
      .catch(() => onRefresh()).finally(() => setCancelling(false))
  }
  if (benchmark.status === 'idle' && !events.length && !calls.length && !results.length) return null
  const passed = calls.filter(c => c.grade === 'PASS').length
  const failed = calls.filter(c => c.grade === 'FAIL').length
  const done = `${progress.completed || 0}/${progress.total || benchmark.plan?.runnable || 0}`
  const lastRun = benchmark.status !== 'running' && benchmark.finished
    ? ` · last run ${new Date(benchmark.finished * 1000).toLocaleTimeString()}` : ''
  return <details className={`metrics-card metrics-live metrics-live-collapse metrics-live-${benchmark.status || 'idle'}`}
    open={liveOpen} onToggle={e => setLiveOpen(e.currentTarget.open)}>
    <summary className="metrics-live-head metrics-live-summary">
      <div><h3>Live test output</h3>
        <span className="metrics-sub">{benchmark.run_id ? `run ${benchmark.run_id} · ` : ''}{benchmark.status || 'idle'} · {done} · {passed}✓ {failed}✗{lastRun}</span></div>
      <div className="metrics-live-actions">
        <button onClick={e => { e.preventDefault(); onRefresh() }}>↻ refresh</button>
        {benchmark.status === 'running' && <button className="metrics-cancel" onClick={e => { e.preventDefault(); cancel() }} disabled={cancelling}>
          {cancelling ? 'cancelling…' : '■ Cancel test'}
        </button>}
      </div>
    </summary>
    <div className="metrics-live-progress">
      <progress max="100" value={progress.percent || 0} />
      <b>{progress.completed || 0}/{progress.total || benchmark.plan?.runnable || 0}</b>
      <span>{progress.percent || 0}%</span>
    </div>
    {current && benchmark.status === 'running' && <div className="metrics-live-current"><span className="metrics-live-dot" />currently testing <code>{current}</code></div>}
    <RunError error={benchmark.error} />
    <div className="metrics-live-counts">
      <span><b>{results.length}</b> configurations</span><span><b>{calls.length}</b> graded calls</span>
      <span><b>{calls.filter(c => c.grade === 'PASS').length}</b> passed</span>
      <span><b>{calls.filter(c => c.grade === 'FAIL').length}</b> failed</span>
      {(() => { const s = calls.map(c => Number(c.tok_s)).filter(v => Number.isFinite(v) && v > 0)
        return <span><b>{s.length ? (s.reduce((a, b) => a + b, 0) / s.length).toFixed(1) : '—'}</b> avg tok/s</span> })()}
    </div>
    {!!events.length && <details className="metrics-live-log" open={benchmark.status === 'running'}>
      <summary>Activity log ({events.length}{fullEvents ? '' : ' — status read, last 200 at most'})</summary>
      {allEvents?.error && <div className="metrics-sub">GET /llm/benchmark/events failed: {allEvents.error} — showing the status read's events</div>}
      <LogBlock text={events.slice().reverse().map(e => `${e.at ? new Date(e.at * 1000).toLocaleTimeString() : '—'}  ${e.kind || 'event'}  ${e.message || (e.value == null ? '' : JSON.stringify(e.value))}`).join('\n')}
        source={fullEvents ? `GET /llm/benchmark/events (run ${benchmark.run_id}, ${events.length} events, newest first)` : `benchmark status events[] (run ${benchmark.run_id || '?'}, newest first)`}
        logRef={fullEvents ? 'benchmark:events' : ''} />
    </details>}
    {!!calls.length && <details className="metrics-live-log">
      <summary>Completed calls ({calls.length})</summary>
      <LogBlock text={calls.slice().reverse().map(c => `${c.grade || '—'}  ${c.worker || c.worker_id || '—'}  ${c.model || '—'} / ${c.quant || '—'}  ${c.task || '—'}  ${c.elapsed_s ?? '—'}s  ${c.tok_s ?? '—'} tok/s`).join('\n')}
        source={`benchmark calls[] (run ${benchmark.run_id || '?'}, ${calls.length} calls, newest first)`} />
    </details>}
  </details>
}

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
export function TierBlocks({ value }) {
  if (value == null) return 'N/A'
  const depth = Math.max(0, Math.min(3, tierDepth(value)))
  const history = Array.isArray(value?.history) ? value.history : []
  return <span className="metrics-tier" title={`${depth} tier${depth === 1 ? '' : 's'} passed`} aria-label={`${depth} of 3 tiers passed`}>
    {[0, 1, 2].map(i => <i key={i} title={['Easy', 'Medium', 'Hard'][i]} className={history[i]?.pass === true ? 'pass' : history[i]?.pass === false ? 'fail' : 'metrics-tier-untested'} />)}
    <em>{depth}</em>
  </span>
}

function WorkerSummary({ rows, calls, workers, lifetime = [] }) {
  // Throughput per worker = Σ tokens / Σ generation seconds over EVERY
  // recorded call of this model on that worker (the server's per-worker total
  // from the call ledger, `lifetime` = /llm/models/<m>/metrics worker_averages)
  // — including calls without a quant/alloc stamp, which are also shown as
  // their own bucket. No peak, no EMA. Falls back to combining the visible
  // cells only when the server sent no per-worker total, and says so.
  const stats = workers.map(w => {
    const life = lifetime.find(r => r.worker === w.id || r.worker === w.name)
    let tp
    if (life && life.n_calls) tp = { ...life, mean_tok_s: life.mean_tok_s ?? null, source: 'call ledger (every call on this worker)' }
    else {
      const mine = rows.filter(r => (r.worker_id || r.worker) === w.id)
      const seen = new Set()
      tp = { ...combineThroughput(mine.map(r => r.throughput).filter(t => t && !seen.has(t) && seen.add(t))),
        source: 'visible quant/alloc cells only (the server sent no per-worker total)' }
    }
    return { ...w, tp, mean: tp.mean_tok_s || 0, n: tp.n_calls || 0,
      calls: life?.n_calls ?? calls.filter(c => (c.worker_id || c.worker) === w.id).length }
  })
  // Model level: Σ/Σ over every worker's totals (exact: sums, not rates).
  const all = combineThroughput(stats.map(s => s.tp))
  const maxSpeed = Math.max(1, ...stats.map(s => s.mean)); const maxCalls = Math.max(1, ...stats.map(s => s.calls))
  // Headline (tok/s · n) sits beside the bar; the qualifiers (unrated calls,
  // unstamped bucket, absence reasons) go on their own full-width line.
  const speedText = s => {
    if (!s.n) return [s.tp?.reason ? 'no calls' : 'no calls recorded', s.tp?.reason ? `no calls on ${s.name}: ${s.tp.reason}` : null]
    const head = s.tp.mean_tok_s != null ? `${metricValue(s.tp.mean_tok_s)} tok/s · n=${s.n}` : `no tok/s · n=${s.n}`
    const notes = []
    if (s.tp.mean_tok_s == null) notes.push(s.tp.reason || 'no call had tokens and a generation window')
    else if (s.tp.n_rated != null && s.tp.n_rated !== s.n) notes.push(`${s.tp.n_rated} of ${s.n} with a generation window`)
    const u = s.tp.unstamped
    if (u && u.n_calls) notes.push(`+${u.n_calls} without quant/alloc stamp: ${u.mean_tok_s != null ? `${metricValue(u.mean_tok_s)} tok/s` : (u.reason || 'no tok/s')}`)
    return [head, notes.length ? notes.join(' · ') : null]
  }
  const chart = (field, max) => <div className="metrics-bars">{stats.map((s, i) => {
    const [head, note] = field === 'mean' ? speedText(s) : [s[field], null]
    return <div className="metrics-bar-row" key={s.id}>
      <b>{s.name}</b><span><i style={{ width: `${100 * s[field] / max}%`, background: colorFor(s.name, i) }} /></span>
      <em title={field === 'mean' ? `${throughputText(s.tp)}\nsource: ${s.tp?.source || 'n/a'}` : undefined}>{head}</em>
      {note && <small>{note}</small>}
    </div>
  })}</div>
  const modelLine = all.n_calls
    ? `model: ${all.mean_tok_s != null ? `${metricValue(all.mean_tok_s)} tok/s` : `no tok/s (${all.reason})`} · n=${all.n_calls}` +
      (all.unstamped?.n_calls ? ` · +${all.unstamped.n_calls} calls without quant/alloc stamp: ${all.unstamped.mean_tok_s != null ? `${metricValue(all.unstamped.mean_tok_s)} tok/s` : 'no tok/s'}` : '')
    : 'model: no calls recorded on the shown workers'
  return <div className="metrics-model-summary">
    <div><h3>Throughput <small>tok/s by worker — Σtokens / Σgeneration-seconds over every recorded call</small></h3>{chart('mean', maxSpeed)}
      <p className="metrics-muted" title={throughputText(all)}>{modelLine}</p></div>
    <div><h3>Total Calls <small>Selected-model evaluations</small></h3>{chart('calls', maxCalls)}</div>
  </div>
}

// A grade row's score/max: the suite's own denominator, counted from the
// per-item markers when recorded; a percent-only row says so.
function gradeFromHistory(h, detail, suites) {
  const suite = resolveSuite({ gradeSuite: h.grade_suite, detail, index: suites })
  if (h.grade == null) return { suite: suite.name, legacy: !!suite.legacy, score: null, max: suite.max, text: 'N/A' }
  const m = matrixFor(detail, suite)
  if (m.recorded) return { suite: suite.name, legacy: !!suite.legacy, score: m.score, max: m.max, text: m.text }
  const score = Math.round(Number(h.grade) * suite.max / 100)
  return { suite: suite.name, legacy: !!suite.legacy, score, max: suite.max,
    text: `${scoreText(score, suite.max)} · from ${Math.round(Number(h.grade))}%, no per-item detail${suite.legacy ? ' · legacy suite' : ''}` }
}

// Rows grouped by the suite that graded them — models graded by different
// suites never share a column set.
function suiteGroups(rows, suites, modelSuite) {
  const out = new Map()
  for (const r of rows) {
    // A row with no grade yet carries no suite of its own: it belongs to the
    // model's task suite (shown as "not graded"), never an empty "unknown suite".
    let suite = resolveSuite({ gradeSuite: r.grade_suite, detail: r.detail, index: suites })
    if (suite.unknown && !suite.tasks.length && modelSuite) suite = modelSuite
    if (!out.has(suite.name)) out.set(suite.name, [suite, []])
    out.get(suite.name)[1].push(r)
  }
  return [...out.values()]
}

// One task's tier markers with expected / actual / why on hover.
function SuiteMarkers({ markers }) {
  if (!markers) return 'not run'
  const passed = markers.filter(m => m.pass === true).length
  return <span className="metrics-tier" aria-label={`${passed} of ${markers.length} passed`}>
    {markers.map((m, i) => <i key={i} className={m.pass === true ? 'pass' : m.pass === false ? 'fail' : 'metrics-tier-untested'}
      title={`${m.tier}: ${m.pass === true ? 'PASS' : m.pass === false ? 'FAIL' : 'not run'}${m.expected ? `\nexpected: ${m.expected}` : ''}${m.actual != null && m.actual !== undefined ? `\nactual: ${m.actual}` : ''}${m.why ? `\nwhy: ${m.why}` : ''}`} />)}
    <em>{passed}/{markers.length}</em>
  </span>
}

function tpCell(tp) {
  // Σtok/Σgen-s with n; with no mean, the server's reason (which rows are
  // missing what), never a bare N/A.
  if (!tp) return 'row carries no throughput block'
  if (!tp.n_calls) return tp.reason || 'no calls recorded'
  if (tp.mean_tok_s == null) return `no tok/s · n=${tp.n_calls}: ${tp.reason || 'no call had tokens and a generation window'}`
  return `${metricValue(tp.mean_tok_s)} · n=${tp.n_calls}${tp.n_rated != null && tp.n_rated !== tp.n_calls ? ` (${tp.n_rated} timed)` : ''}`
}

function GradingMatrix({ rows, workers, suite }) {
  const cols = suite.tasks
  return <div className="metrics-sheetwrap"><table className="metrics-sheet metrics-grade-matrix"><thead><tr>
    <th>Quant / Size</th><th title="the last measured cold load for this quant (not a mean: per-load history is not yet kept)">Cold (last measured)</th><th>Grade Tier</th><th>Alloc</th><th title="the last measured hot load (not a mean: per-load history is not yet kept)">Hot (last measured)</th><th title="mean of every recorded call on this worker/quant/alloc (Σtokens / Σseconds)">tok/s (avg of calls)</th>
    {cols.map(t => <th key={t}>{t}</th>)}<th>task_best</th><th>task_worst</th><th>score</th>
  </tr></thead><tbody>{workers.flatMap((worker, workerIndex) => {
    const workerRows = rows.filter(r => (r.worker_id || r.worker) === worker.id)
    if (!workerRows.length) return []
    const quantNames = [...new Set(workerRows.map(r => r.quant))]
    const band = <tr className="metrics-band" key={`band:${worker.id}`}><th><span className="metrics-pill" style={{ color: colorFor(worker.name, workerIndex) }}>{worker.name}</span></th><td colSpan={8 + cols.length}>Worker allocation matrix</td></tr>
    const matrixRows = quantNames.flatMap(quant => {
      const isConstrained = r => r.status === 'hardware_constraint_failed'
      // skipped (hardware-constrained) allocations sort after the ones that ran
      const quantRows = workerRows.filter(r => r.quant === quant).sort((a, b) =>
        ((a.config === '4-bit') - (b.config === '4-bit')) || (isConstrained(a) - isConstrained(b)) ||
        String(a.alloc_mode).localeCompare(String(b.alloc_mode)))
      const coldRow = quantRows.find(r => !isConstrained(r) && r.cold_s != null) || quantRows.find(r => !isConstrained(r))
      const precisions = [...new Set(quantRows.map(r => r.config || 'standard'))]
      let quantOffset = 0
      return precisions.flatMap(precision => {
        const precisionRows = quantRows.filter(r => (r.config || 'standard') === precision)
        const gradeRow = precisionRows.find(r => r.detail && Object.keys(r.detail).length) ||
          precisionRows.find(r => !isConstrained(r)) || precisionRows[0]
        // The grade cells span every allocation of this precision: they are
        // N/A only when the row that carries the grade was itself skipped.
        const gradeConstrained = isConstrained(gradeRow)
        const mx = gradeRow.detail ? matrixFor(gradeRow.detail, suite) : null
        const byTask = mx ? Object.fromEntries(mx.cells.map(c => [c.task, c.markers])) : {}
        const graded = !!mx || (!!gradeRow.grade && gradeRow.grade !== 'N/A')
        const scoreLabel = mx && mx.recorded ? mx.text : graded ? gradeRow.grade : 'not graded'
        const rendered = precisionRows.map((r, allocationIndex) => {
          const constrained = r.status === 'hardware_constraint_failed'; const firstPrecision = allocationIndex === 0
          const firstQuant = quantOffset === 0 && firstPrecision
          const sourceTitle = constrained ? r.constraint_reason || r.error : r.metric_source ? `durable ${r.metric_source}` : 'normal central inference'
          return <tr key={`${worker.id}:${quant}:${precision}:${r.alloc_mode}`} className={`${constrained ? 'na ' : ''}${firstPrecision ? 'precision-start' : ''}`} title={sourceTitle}>
            {firstQuant && <td className="metrics-quant-cell" rowSpan={quantRows.length}><b>{quant}</b><small>{bytes(r.disk_bytes)}</small></td>}
            {firstQuant && <td rowSpan={quantRows.length}>{coldRow ? seconds(coldRow.cold_s) : 'N/A'}</td>}
            {firstPrecision && <td rowSpan={precisionRows.length}><span className={`metrics-tier-tag ${precision === '4-bit' ? 'bit4' : 'standard'}`}>{precision}</span></td>}
            <td>{r.alloc_mode}{constrained && <small>skipped</small>}</td><td>{constrained ? 'N/A' : seconds(r.hot_load_s)}</td>
            <td className="metrics-tp-cell" title={constrained ? sourceTitle : throughputText(r.throughput)}>{constrained ? `not run: ${r.constraint_reason || r.error || 'hardware constraint (no reason recorded)'}` : tpCell(r.throughput)}</td>
            {firstPrecision && cols.map(t => <td rowSpan={precisionRows.length} key={t}>{gradeConstrained ? 'N/A' : mx ? <SuiteMarkers markers={byTask[t]} /> : graded ? 'no per-item detail' : 'not graded'}</td>)}
            {firstPrecision && <><td rowSpan={precisionRows.length}>{gradeConstrained ? 'N/A' : gradeRow.task_best || (graded ? 'N/A' : 'not graded')}</td>
              <td rowSpan={precisionRows.length}>{gradeConstrained ? 'N/A' : gradeRow.task_worst || (graded ? 'N/A' : 'not graded')}</td>
              <td rowSpan={precisionRows.length} title={suite.legacy ? 'Legacy single-prompt suite. Rerun this model for Easy/Medium/Hard grading.' : `${suite.name}: PASS items / ${suite.max}`}><b>{gradeConstrained ? 'N/A' : scoreLabel}</b></td></>}
          </tr>
        })
        quantOffset += precisionRows.length
        return rendered
      })
    })
    return [band, ...matrixRows]
  })}</tbody></table></div>
}

// Every graded call as a row — the score is their visible PASS count over the
// suite's max. Live run calls + the persisted ledger rows, grouped by
// (suite, worker, quant, alloc); the latest call per task x tier counts.
function GradedCalls({ calls, suites, rows }) {
  const groups = useMemo(() => {
    const out = new Map()
    const suiteOfRow = (c) => rows.find(r => (r.worker_id || r.worker) === (c.worker_id || c.worker) && r.quant === c.quant)?.grade_suite
    for (const raw of calls || []) {
      const c = fromLedger(raw)
      if (!(c.benchmark || c.grade === 'PASS' || c.grade === 'FAIL' || c.grade === 'ERROR')) continue
      const suite = resolveSuite({ gradeSuite: c.grade_suite || suiteOfRow(c), index: suites })
      const key = [suite.name, c.worker || c.worker_id, c.quant, c.alloc_mode || c.config].join(' · ')
      if (!out.has(key)) out.set(key, { suite, calls: [] })
      out.get(key).calls.push(c)
    }
    return [...out.entries()].map(([key, g]) => [key, g.suite, gradedCallRows(g.calls, g.suite)])
  }, [calls, suites, rows])
  if (!groups.length) return <p className="metrics-muted">no graded items in the call ledger for this model ({(calls || []).length} call rows read, none carrying a PASS/FAIL/ERROR grade)</p>
  return <div className="metrics-call-section"><h3>Graded items <span className="metrics-sub">every item in full: prompt, expected response, model response, verdict — the score is the count of correct items; format is graded separately</span></h3>
    {groups.map(([key, suite, g]) => <details key={key} open={groups.length === 1} className="metrics-graded-group">
      <summary><b>{g.text}</b> correct · {g.rows.filter(r => r.format === true).length}/{g.max} format · {key} · {g.answered} of {g.max} items answered</summary>
      <div className="graded-list">{g.rows.map((r, i) => <GradedItem key={i} item={r}
        heading={`${r.task} · ${r.tier} · ${r.elapsed_s != null ? seconds(r.elapsed_s) : 'N/A'} · ${suite.name === 'hugpy-imagegen-v1' ? (r.s_per_image != null ? seconds(r.s_per_image) + '/image' : 'N/A') : metricValue(r.tok_s) + ' tok/s'} · ${r.worker} · ${r.quant} · ${r.alloc_mode} · ${r.ts ? new Date(r.ts * 1000).toLocaleString() : 'N/A'}`} />)}</div>
    </details>)}
  </div>
}

// Every call, every field. The ledger row is relayed whole: the summary
// columns, then the verdict, then the complete record (prompt / expected /
// response / judge / error / reason / evidence / log) with nothing cut.
const HIDE_IN_RECORD = new Set(['state'])
function CallRecord({ c }) {
  const entries = Object.entries(c).filter(([k, v]) => !HIDE_IN_RECORD.has(k) && v !== undefined)
  return <dl className="graded-qa">
    {entries.map(([k, v]) => <div key={k} className="call-record-row"><dt>{k}</dt>
      <dd><pre>{v == null ? 'null' : typeof v === 'object' ? JSON.stringify(v, null, 1) : String(v)}</pre></dd></div>)}
  </dl>
}

function EvaluationLog({ calls, model }) {
  return <div className="metrics-call-section"><div><h3>Evaluation Call Log</h3><span className="metrics-sub">every recorded call for {model}, every field — {calls.length} rows</span></div>
    {!calls.length && <p className="metrics-muted">no calls recorded in the ledger for {model}</p>}
    <div className="metrics-sheetwrap metrics-call-log"><table className="metrics-sheet"><thead><tr>
      <th>#</th><th>model</th><th>quant</th><th>alloc</th><th>worker</th><th>tok/s (this call)</th><th>elapsed</th><th>ctx-in</th><th>ctx-out</th><th>category (tier)</th><th>verdict</th><th>date</th><th>time</th><th>caller</th><th>log / error</th><th>record</th>
    </tr></thead><tbody>{calls.map((c, i) => { const when = c.ts || c.timestamp || c.finished; const date = when ? new Date(Number(when) * 1000) : null
      const err = c.error && c.error !== 'N/A' ? String(c.error) : ''
      const log = [err, c.reason, c.why, c.judge?.reason && `judge: ${c.judge.reason}`, c.log_ref && `log_ref: ${c.log_ref}`, c.evidence && JSON.stringify(c.evidence, null, 1)].filter(Boolean).join('\n')
      const verdict = c.grade || (c.passed === true ? 'PASS' : c.passed === false ? 'FAIL' : '—')
      return <tr key={`${c.request_id || c.task}:${i}`} className={verdict === 'PASS' ? '' : verdict === 'FAIL' || verdict === 'ERROR' ? 'metrics-verdict-fail' : ''}><td>{i + 1}</td><td>{scalar((typeof c.model === 'string' && c.model) || model)}</td><td>{scalar(c.quant)}</td><td>{scalar(c.alloc_mode || c.config || 'standard')}</td><td>{scalar(c.worker)}</td>
        <td>{metricValue(c.tok_per_s ?? c.tok_s)}</td><td>{c.elapsed_s != null ? seconds(c.elapsed_s) : 'N/A'}</td><td>{c.prompt_tokens ?? c.ctx_in ?? 'N/A'}</td><td>{c.completion_tokens ?? c.ctx_out ?? 'N/A'}</td><td>{scalar(c.task)}</td>
        <td><b className={verdict === 'PASS' ? 'metrics-pass' : verdict === 'FAIL' || verdict === 'ERROR' ? 'metrics-fail' : ''}>{verdict}</b>{c.format != null && <span className="graded-sub"> · format {c.format ? '✓' : '✗'}</span>}{c.revised && <span className="graded-sub graded-revised"> · revised</span>}</td>
        <td>{date ? date.toLocaleDateString() : 'N/A'}</td><td>{date ? date.toLocaleTimeString() : 'N/A'}</td><td>{scalar(c.caller || 'orchestrator')}</td>
        <td>{log ? <pre className="call-log">{log}</pre> : <span className="metrics-muted">none recorded</span>}</td>
        <td><details className="metrics-call-detail call-record"><summary>all fields</summary><CallRecord c={c} /></details></td></tr> })}
    </tbody></table></div></div>
}

function BenchmarkWorkbook({ benchmark, historical: historicalRaw = [], infos, canon, navList = [], model, onSelect, onBenchmark, onRefresh, benchmarkWorkers, gradeModels = [], onGradeModelsChange }) {
  // start = {phase: idle|starting|started|error|conflict, models, message, active}
  const [start, setStart] = useState({ phase: 'idle' })
  const [queue, setQueue] = useState(() => readStore(safeSession(), QUEUE_STORE, []).filter(Boolean))
  const [hiddenWorkers, setHiddenWorkers] = useState([])
  const [modelHistory, setModelHistory] = useState({ calls: [], worker_averages: [] })
  const [nonce, setNonce] = useState(0)      // bumps when a run ends -> refetch per-model data
  const inFlight = useRef(false)
  const suites = useSuites()
  useEffect(() => { writeStore(safeSession(), QUEUE_STORE, queue) }, [queue])

  const rows = useMemo(() => {
    // Integrity verdicts share model_metrics' grade columns (grade 100/0 +
    // an audit-log detail): keep those rows' load/throughput numbers but never
    // render their verdict as an aptitude score (it would read as 27/27).
    const historical = historicalRaw.map(h => (h.grade_suite === 'integrity' ? { ...h, grade: null, grade_detail: null } : h))
    const results = benchmark?.results || []
    const byKey = new Map(results.map(r => [benchKey(r), r]))
    const planned = (benchmark?.plan?.rows || []).map(p => ({ ...p, ...(byKey.get(benchKey(p)) || {}) }))
    const keys = new Set(planned.map(benchKey))
    const history = new Map(historical.map(h => [[h.worker, h.model_name, h.quant, h.alloc_mode].join('\0'), h]))
    const active = [...planned, ...results.filter(r => !keys.has(benchKey(r)))]
    // Run rows carry the worker's id; model_metrics rows only its name. Key the
    // database rows by the same id so one worker is one band, not two.
    const idByName = new Map(active.filter(r => r.worker && r.worker_id).map(r => [r.worker, r.worker_id]))
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
      const g = gradeFromHistory(h, detail, suites)
      return { model: h.model_name, worker: h.worker, worker_id: idByName.get(h.worker) || h.worker,
        quant: h.quant || 'runtime default', config: prefixed ? (parts[0] === '4bit' ? '4-bit' : parts[0]) : 'standard',
        alloc_mode: prefixed ? parts.slice(1).join(':') : h.alloc_mode || 'HugPy default',
        cold_s: h.cold_load_s, hot_load_s: h.hot_load_s, throughput: h.throughput,
        detail, legacy_grade: g.legacy, grade_suite: g.suite, score: g.score, max: g.max, grade: g.text,
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
        return { ...r, cold_s: h.cold_load_s, hot_load_s: h.hot_load_s, throughput: h.throughput,
          detail: historicDetail, grade: gradeFromHistory(h, historicDetail, suites).text,
          n_samples: h.n_samples, metric_source: 'database preserved after failed run', metric_updated_at: h.updated_at }
      }
      return { ...r, cold_s: missing(r.cold_s) ? h.cold_load_s : r.cold_s,
        hot_load_s: missing(r.hot_load_s) ? h.hot_load_s : r.hot_load_s,
        throughput: h.throughput,
        detail: r.detail || historicDetail, n_samples: r.n_samples ?? h.n_samples,
        grade: r.grade === 'N/A' && h.grade != null ? gradeFromHistory(h, historicDetail, suites).text : r.grade,
        metric_source: 'database fallback', metric_updated_at: h.updated_at }
    })
  }, [benchmark, historicalRaw, suites])
  const coverage = useMemo(() => {
    const suites = new Map()
    for (const call of (benchmark?.calls || [])) {
      const key = benchKey(call)
      if (!suites.has(key)) suites.set(key, new Set())
      suites.get(key).add(String(call.task || '').replace(/ \([^)]*\)$/, ''))
    }
    const out = {}
    for (const m of navList) {
      const planned = new Set(rows.filter(r => canon(r.model) === m).map(benchKey)).size
      const modelSuites = [...suites.entries()].filter(([key]) => canon(key.split('\0')[1]) === m)
      const suiteTasks = (key) => {
        const r = rows.find(x => benchKey(x) === key)
        return resolveSuite({ gradeSuite: r?.grade_suite, detail: r?.detail, index: suites }).tasks
      }
      const complete = modelSuites.filter(([key, tasks]) => suiteTasks(key).every(t => tasks.has(t))).length
      const calls = modelSuites.reduce((n, [, tasks]) => n + tasks.size, 0)
      out[m] = { planned, complete, calls,
        status: complete ? (complete >= planned ? 'tested' : 'tested · matrix incomplete')
          : calls ? 'attempted · suite incomplete' : 'not tested' }
    }
    return out
  }, [benchmark?.calls, navList, rows, canon, suites])
  // "Untested" = no aptitude grade on record AND no complete suite in the
  // current run, within the current filter view.
  const untestedModels = useMemo(() => navList.filter(m => !coverage[m]?.complete && isUntested(infos.get(m))), [navList, coverage, infos])

  const status = benchmark?.status
  const isRunning = status === 'running'
  const activeModels = isRunning ? (benchmark?.models?.length ? benchmark.models : [benchmark?.progress?.model].filter(Boolean)) : []
  const runningThis = isRunning && activeModels.map(canon).includes(model)

  // A run just ended -> refetch the selected model's history/why.
  const prevStatus = useRef(status)
  useEffect(() => {
    if (prevStatus.current === 'running' && status && status !== 'running') {
      setNonce(n => n + 1)
      setStart(s => (s.phase === 'started' ? { phase: 'idle' } : s))
    }
    prevStatus.current = status
  }, [status])

  useEffect(() => {
    if (!model) return
    let live = true
    fetchJson(`/api/llm/models/${encodeURIComponent(model)}/metrics?limit=2000`)
      .then(value => { if (live) setModelHistory(value || { calls: [], worker_averages: [] }) })
      .catch(() => { if (live) setModelHistory({ calls: [], worker_averages: [] }) })
    return () => { live = false }
  }, [model, nonce])

  const launch = useCallback(async (models, { fromQueue = false } = {}) => {
    if (!models.length || inFlight.current) return false
    // Same worker selection as the Test-all control (default all eligible). An
    // empty/absent selection means the operator unchecked everything — refuse
    // rather than silently testing on every worker.
    const workers = benchmarkWorkers || []
    if (!workers.length) { setStart({ phase: 'error', models, message: 'no workers selected — pick at least one worker above (default is all eligible)' }); return false }
    inFlight.current = true
    setStart({ phase: 'starting', models })
    const res = await postJson('/api/llm/benchmark/run', { executor: 'hugpy-central', tokens: 128, models, workers })
    inFlight.current = false
    if (res.ok) {
      onBenchmark(res.data)           // switch to the running view NOW, not on the next poll
      setStart({ phase: 'started', models })
      if (fromQueue) setQueue(q => q.slice(1))
      return true
    }
    if (res.status === 409) {
      if (res.data?.status) onBenchmark(res.data)   // the 409 body IS the active run
      const active = res.data?.models?.length ? res.data.models : [res.data?.progress?.model].filter(Boolean)
      setStart(fromQueue ? { phase: 'idle' } : { phase: 'conflict', models, active })
      return false
    }
    if (fromQueue) setQueue(q => q.slice(1))
    setStart({ phase: 'error', models, message: res.message })
    return false
  }, [onBenchmark, benchmarkWorkers])

  // Client-side queue: the server holds one global run; queued models start
  // when the active run ends (while this tab is open; survives reload).
  useEffect(() => {
    if (!queue.length || !status || isRunning || start.phase === 'starting') return
    launch([queue[0]], { fromQueue: true })
  }, [queue, status, isRunning, start.phase, launch])

  const enqueue = (m) => { setQueue(q => (q.includes(m) ? q : [...q, m])); setStart({ phase: 'idle' }) }
  const cancelActive = () => {
    postJson('/api/llm/benchmark/cancel', { scope: 'execution' }).then(res => {
      if (res.data?.status) onBenchmark(res.data)
      onRefresh()
    })
  }
  const runModel = () => {
    if (!model) return
    if (isRunning) { if (!runningThis) enqueue(model); return }
    launch([model])
  }
  const resumeUntested = () => {
    if (!untestedModels.length || isRunning || start.phase === 'starting') return
    if (!confirm(`Test ${untestedModels.length} untested models (current filter view) through normal HugPy inference?\n\nModels with a recorded grade are not included.`)) return
    launch(untestedModels)
  }

  const page = useMemo(() => rows.filter(r => canon(r.model) === model), [rows, model, canon])
  const workers = useMemo(() => {
    const out = new Map(); page.forEach(r => out.set(r.worker_id || r.worker, { id: r.worker_id || r.worker, name: r.worker || r.worker_id }))
    return [...out.values()].sort((a, b) => String(a.name).localeCompare(String(b.name)))
  }, [page])
  const visibleWorkers = workers.filter(w => !hiddenWorkers.includes(w.id))
  const visiblePage = page.filter(r => visibleWorkers.some(w => w.id === (r.worker_id || r.worker)))
  const liveCalls = (benchmark?.calls || []).filter(c => canon(c.model) === model)
  const modelCalls = [...liveCalls, ...(modelHistory.calls || []).map(fromLedger)]
    .filter(c => visibleWorkers.some(w => w.id === (c.worker_id || c.worker)))
  const index = navList.indexOf(model)
  const info = infos.get(model)
  const pct = benchmark?.progress?.percent || 0
  const queued = queue.includes(model)

  const modelSuite = suiteForTask(suites, info?.primary_task)
  const noSuite = !!info && !modelSuite
  const buttonLabel = noSuite ? 'no suite for this task'
    : start.phase === 'starting' ? 'starting…'
    : runningThis ? `running… ${pct}%`
    : isRunning ? (queued ? 'queued' : 'Queue test')
    : 'Test this model'
  const buttonTitle = noSuite ? `No grading suite covers ${info?.primary_task || 'an unrecorded task'}; this model cannot be graded automatically`
    : runningThis ? 'This model is being tested now — see Live test output above'
    : isRunning ? `A run is active (${activeModels.join(', ') || 'benchmark status lists no model'}); queue ${model} to start when it ends`
    : `Test ${model} only`

  // The model selector reuses the shared load-a-model popover component. Its
  // rows are the FilterBar-admitted set (navList), shaped so the shared grid can
  // render Model · Task · Lib · Grade · Suite from the one server-side index.
  const pickerModels = useMemo(() => navList.map(k => {
    const inf = infos.get(k) || {}
    let grade = 'not tested'
    if (inf.aptitude) {
      const suite = inf.aptitude.suiteDef || resolveSuite({ gradeSuite: inf.aptitude.suite, detail: inf.aptitude.detail })
      const mx = inf.aptitude.detail ? matrixFor(inf.aptitude.detail, suite) : null
      grade = mx && mx.recorded ? `graded ${mx.text}` : `graded ${Math.round(inf.aptitude.grade)}%`
    }
    const su = suiteForTask(suites, inf.primary_task)
    return { model_key: k, name: k, framework: inf.framework || '', primary_task: inf.primary_task || '',
      tasks: inf.primary_task ? [inf.primary_task] : [], _grade: grade, _suiteName: su?.name || null, _gradable: !!su }
  }), [navList, infos, suites])
  // Selection admitted by the filters iff it is in navList; if not, keep it and
  // pin it (never silently switch) — the shared picker shows it above the list.
  const selectedHidden = !!model && !navList.includes(model)
  const pinnedModels = selectedHidden
    ? [{ model_key: model, name: model, _note: 'current selection · hidden by the active filters', _pinned: true }]
    : null

  return <section className="metrics-card metrics-workbook">
    <div className="metrics-workbook-head"><div><h3>Benchmark workbook</h3>
      <span className="metrics-sub">{status || 'idle'} · {benchmark?.progress?.completed || 0}/{benchmark?.progress?.total || benchmark?.plan?.runnable || 0} allocation variations</span></div>
      <div className="metrics-model-nav"><button disabled={index <= 0} onClick={() => onSelect(navList[index - 1])} aria-label="previous model">←</button>
        <LoadModelPicker
          models={pickerModels}
          value={model}
          onPick={(k) => onSelect(k)}
          placeholder={navList.length ? (gradeModels.length ? `${gradeModels.length} ticked · pick to view…` : 'Pick a model to view / tick to grade…') : 'no models match the current filters'}
          header={<div className="mp-note">{navList.length} of {infos.size} models match filters · click a name to view it in the workbook, tick a box to include it in grading</div>}
          multiSelect={!!onGradeModelsChange}
          selectedKeys={gradeModels}
          onToggleKey={(k) => onGradeModelsChange?.(gradeModels.includes(k) ? gradeModels.filter(x => x !== k) : [...gradeModels, k])}
          onSelectAll={(keys) => onGradeModelsChange?.([...new Set([...gradeModels, ...keys])])}
          onClearSelected={() => onGradeModelsChange?.([])}
          pinned={pinnedModels}
          gridTemplate="minmax(150px,1.7fr) minmax(110px,1.1fr) 78px minmax(110px,1.2fr) minmax(120px,1.3fr)"
          headCells={['Model', 'Task', 'Lib', 'Grade', 'Suite']}
          emptyText={navList.length === 0
            ? (infos.size ? 'no models match the current filters' : 'no models loaded yet')
            : 'no model matches your search'}
          renderCells={(m) => m._pinned
            ? <span className="mp-pinned-note" style={{ gridColumn: '2 / -1' }}>{m._note}</span>
            : <>
                <span className="mp-task" title={m.primary_task}>{m.primary_task || '—'}</span>
                <span className={`mp-fw mp-fw-${m.framework || 'unknown'}`}>{m.framework || '—'}</span>
                <span className="mp-grade" title={m._grade}>{m._grade}</span>
                <span className={`mp-suite${m._gradable ? '' : ' mp-nosuite'}`} title={m._suiteName ? `suite ${m._suiteName}` : 'no grading suite for this task — not gradable'}>
                  {m._suiteName ? `suite ${m._suiteName}` : 'no suite'}
                </span>
              </>}
        />
        <span>{index >= 0 ? index + 1 : '—'}/{navList.length}</span><button disabled={index < 0 ? !navList.length : index >= navList.length - 1} onClick={() => onSelect(navList[index + 1] || navList[0])} aria-label="next model">→</button>
        <button className={`metrics-test-model${runningThis ? ' running' : ''}`} disabled={!model || noSuite || start.phase === 'starting' || runningThis || queued} onClick={runModel} title={buttonTitle}>
          {buttonLabel}</button></div>
    </div>
    {info && <p className={`metrics-suite-line${noSuite ? ' metrics-nosuite' : ''}`}>
      task: <b>{info.primary_task || 'not recorded'}</b> · {modelSuite
        ? <>graded by <b>{modelSuite.name}</b> ({modelSuite.tasks.length} grade categories × {modelSuite.tiers || 3} tiers = {modelSuite.max} items)</>
        : <>NO GRADING SUITE is registered for task {info.primary_task || '(no task recorded)'} ({Object.keys(suites || {}).length} suites known{_suitesErr ? `; ${_suitesErr}` : ''})</>}</p>}
    <ModelHeader model={model} info={info} benchmark={benchmark} canon={canon} nonce={nonce} />
    <RunNotice start={start} model={model} queue={queue} onCancelActive={cancelActive} onQueue={enqueue}
      onDismiss={() => setStart({ phase: 'idle' })} onUnqueue={(m) => setQueue(q => q.filter(x => x !== m))} />
    {!rows.length ? <p className="metrics-muted">No benchmark rows for {model || '(no model selected)'}: {(benchmark?.results || []).length} result row(s) in the current run, {historicalRaw.length} model_metrics row(s) scanned.</p> : <>
      {/* Display filter only — which workers' columns show in the matrix/summary
          below. NOT a run-target selector: what gets tested is chosen once, in
          the Fleet capacity test worker selector above. */}
      <div className="metrics-worker-toggles" title="Show or hide a worker's columns in the matrix below. To choose which workers a test runs on, use the worker selector in Fleet capacity test above.">
        <span>Show in matrix</span>{workers.map(w => <button key={w.id}
        className={hiddenWorkers.includes(w.id) ? '' : 'active'} onClick={() => setHiddenWorkers(old => old.includes(w.id) ? old.filter(id => id !== w.id) : [...old, w.id])}>{w.name}</button>)}</div>
      <div className="metrics-coverage">
        <span><b>{navList.filter(m => coverage[m]?.complete).length}</b> tested this run</span>
        <span><b>{navList.filter(m => !coverage[m]?.complete && coverage[m]?.calls).length}</b> incomplete suites</span>
        <span><b>{navList.filter(m => isUntested(infos.get(m))).length}</b> never graded</span>
        <span>selected: <b>{coverage[model]?.complete || 0}/{coverage[model]?.planned || 0}</b> complete suites · {coverage[model]?.calls || 0} distinct calls</span>
        <button disabled={isRunning || start.phase === 'starting' || !untestedModels.length} onClick={resumeUntested}
          title={isRunning ? 'Stop or finish the active run first' : 'Every model in the current filter view with no recorded grade'}>
          Test untested in view ({untestedModels.length})
        </button>
      </div>
      <WorkerSummary rows={visiblePage} calls={modelCalls} workers={visibleWorkers} lifetime={modelHistory.worker_averages || []} />
      <div className="metrics-matrix-head"><div><h3>Grading matrix</h3><span className="metrics-sub">grade categories of the suite that graded each row · every difficulty tested independently (Easy | Medium | Hard)</span></div>
        <div className="metrics-tier-legend"><span><i className="pass" />Pass</span><span><i className="fail" />Fail</span><span><i className="metrics-tier-untested" />Untested</span></div></div>
      {suiteGroups(visiblePage, suites, modelSuite).map(([suite, srows]) => <div key={suite.name} className="metrics-suite-block">
        <div className="metrics-sub"><b>{suite.name}</b>{suite.legacy ? ' (legacy suite — single prompt per category)' : ''} · {suite.tasks.length} grade categories × {suite.tiers || 3} tier{(suite.tiers || 3) === 1 ? '' : 's'} = {suite.max} items{suite.inferred ? ' · suite inferred from the grade detail' : ''}</div>
        <GradingMatrix rows={srows} workers={visibleWorkers} suite={suite} />
      </div>)}
      <GradedCalls calls={modelCalls} suites={suites} rows={visiblePage} />
      <EvaluationLog calls={modelCalls} model={model} />
    </>}
  </section>
}

function RunNotice({ start, model, queue, onCancelActive, onQueue, onDismiss, onUnqueue }) {
  const who = (start.models || []).length > 1 ? `${start.models.length} models` : (start.models || [])[0]
  return <>
    {start.phase === 'starting' && <p className="metrics-run-notice">starting test for {who}…</p>}
    {start.phase === 'started' && <p className="metrics-run-notice">started HugPy-native test for {who} — progress in Live test output above.</p>}
    {start.phase === 'error' && <p className="metrics-run-notice metrics-error">could not start {who}: {start.message} <a className="metrics-link" href="#" onClick={e => { e.preventDefault(); onDismiss() }}>dismiss</a></p>}
    {start.phase === 'conflict' && <div className="metrics-run-conflict" role="alert">
      <span>a run is already active: <b>{(start.active || []).join(', ') || '(the 409 reply named no active model)'}</b></span>
      <button onClick={onCancelActive}>■ Cancel active run</button>
      {(start.models || []).length === 1 && <button onClick={() => onQueue(start.models[0])}>Queue {start.models[0]}</button>}
      <a className="metrics-link" href="#" onClick={e => { e.preventDefault(); onDismiss() }}>dismiss</a>
    </div>}
    {!!queue.length && <p className="metrics-run-notice">queued (starts when the active run ends, while this page is open): {queue.map(m =>
      <span key={m} className="metrics-queue-item">{m === model ? <b>{m}</b> : m} <a className="metrics-link" href="#" title="remove from queue" onClick={e => { e.preventDefault(); onUnqueue(m) }}>×</a></span>)}</p>}
  </>
}

// Picker/header chips: the model's WORTH label + its per-worker live state
// (same vocabulary as the Models table) when central serves
// /llm/models/status; otherwise the metrics-derived chips as before.
const CHIP_TONE = { ok: 'ok', bad: 'bad', warn: 'warn', muted: 'muted', live: 'ok' }
function Chips({ info }) {
  const { index } = useModelStatus()
  const row = index.available && info ? statusFor(index, info.key) : null
  if (row) return <ModelLiveState modelKey={info.key} />
  return <>{statusChips(info).map(c => <span key={c.label} className={`metrics-chip ${c.tone}`}>{c.label}</span>)}</>
}

// Why a model is failing/held: integrity verdict (grade_detail why + log),
// admission reason, current-run errors and the latest recorded load failure.
function ModelHeader({ model, info, benchmark, canon, nonce }) {
  const [loadFail, setLoadFail] = useState(null)
  useEffect(() => {
    setLoadFail(null)
    if (!model) return
    let live = true
    fetchJson(`/api/llm/compute-actions?action=load&outcome=fail&limit=1&model=${encodeURIComponent(model)}`)
      .then(d => { if (live) setLoadFail((d?.actions || [])[0] || null) })
      .catch(e => { if (live) setLoadFail({ fetch_error: String(e?.message || e) }) })
    return () => { live = false }
  }, [model, nonce])
  if (!model) return null
  const reasons = []
  const integ = info?.integrity
  if (integ && integ.grade === 0) {
    reasons.push({ kind: `integrity: ${integ.verdict || 'fail'}`, line: firstLine(integ.why) || 'integrity grade 0 recorded with no why text',
      full: [integ.why, ...(integ.log || [])].filter(Boolean).join('\n') })
  }
  if (info?.admission && info.admission.status !== 'admitted') {
    reasons.push({ kind: `admission: ${info.admission.status}`, line: firstLine(info.admission.reason) || `admission record${info.admission.job ? ` job ${info.admission.job}` : ''} carries no reason text`, full: info.admission.reason })
  }
  if (info?.unserveable_reason) reasons.push({ kind: 'unserveable', line: firstLine(info.unserveable_reason), full: info.unserveable_reason })
  for (const r of (benchmark?.results || [])) {
    if (canon(r.model) !== model || !(r.status === 'error' || r.status === 'failed' || (r.error && r.error !== 'N/A'))) continue
    reasons.push({ kind: `this run · ${r.worker || r.worker_id || '—'} ${r.quant || ''}`, line: firstLine(r.error || r.constraint_reason || r.status), full: r.error || r.constraint_reason || '' })
  }
  if (loadFail?.fetch_error) {
    reasons.push({ kind: 'failure log read failed', line: loadFail.fetch_error, full: loadFail.fetch_error })
  } else if (loadFail) {
    const d = loadFail.detail || {}
    const text = d.loader_stderr || d.message || (typeof d === 'string' ? d : '')
    const when = loadFail.ts ? new Date(Number(loadFail.ts) * 1000).toLocaleString() : ''
    reasons.push({ kind: `last failed load${when ? ` · ${when}` : ''}${d.class ? ` · ${d.class}` : ''}`,
      line: firstLine(text) || `compute_actions row ${loadFail.id ?? '?'} (load/fail) carries no message`, full: [d.message, d.loader_stderr].filter(Boolean).join('\n\n') })
  }
  return <div className="metrics-model-header">
    <div className="metrics-model-title"><b title={info?.hub_id || model}>{model}</b>
      {info?.framework && <span className="metrics-sub">{info.framework}{info.hub_id ? ` · ${info.hub_id}` : ''}</span>}
      <Chips info={info} /></div>
    {reasons.map((r, i) => <div key={i} className="metrics-why">
      <span className="metrics-why-kind">{r.kind}</span> <span className="metrics-why-line">{r.line}</span>
      {r.full && r.full !== r.line && <details className="metrics-call-detail metrics-why-detail"><summary>details</summary><pre>{r.full}</pre></details>}
    </div>)}
  </div>
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
  if (!top.length) return <p className="metrics-muted">No samples: n_samples is 0 across all {rows.length} model_metrics rows.</p>
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

function WorkerChart({ rows, palette }) {
  const workers = Object.keys(palette)
  // (model, worker) -> mean over EVERY recorded call on that pair (the cells'
  // throughput, token-weighted across quant/alloc). The card's EMA fields are
  // placement state and are never drawn as the model's tok/s.
  const { grouped, counts } = useMemo(() => {
    const cells = {}
    for (const r of rows) ((cells[r.model_name] ||= {})[r.worker] ||= []).push(r.throughput)
    const g = {}, n = {}
    for (const [m, byW] of Object.entries(cells)) {
      for (const [w, list] of Object.entries(byW)) {
        const c = combineThroughput(list)
        if (c.n_calls) { (g[m] ||= {})[w] = c.mean_tok_s; (n[m] ||= {})[w] = c.n_calls }
      }
    }
    return { grouped: g, counts: n }
  }, [rows])
  const models = Object.keys(grouped)
    .sort((a, b) => Math.max(...vals(grouped[b])) - Math.max(...vals(grouped[a]))).slice(0, 12)
  if (!models.length) {
    const err = rows.map(r => r.throughput?.error).find(Boolean)
    return <p className="metrics-muted">{err ? `call ledger (model_calls) read failed: ${err}`
      : `No calls recorded in model_calls for any of the ${rows.length} model_metrics rows.`}</p>
  }
  const allv = models.flatMap(m => vals(grouped[m])).filter(v => v != null && !isBad(v))
  const max = Math.max(...allv, 1)
  const rowH = Math.max(24, workers.length * 15 + 10)
  const W = 500, padL = 160, padR = 52, bw = W - padL - padR
  const H = 12 + models.length * rowH
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="metrics-svg" role="img" aria-label="mean tok/s of recorded calls by worker">
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
                  <text className="m-val m-small" x={padL + ww + 5} y={by + bh / 2 + 3}>{bad ? '∞*' : `${fmt(v)} · n=${counts[m]?.[w] ?? 0}`}</text>
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
    ['worker', 'Worker'], ['tp_mean', 'tok/s (mean of calls)', 1], ['tp_n', 'calls', 1],
    ['task', 'Task'], ['updated_at', 'Updated', 1],
  ]
  const val = (r, k) => (k === 'tp_mean' ? r.throughput?.mean_tok_s : k === 'tp_n' ? r.throughput?.n_calls : r[k])
  const sorted = [...rows].sort((a, b) => {
    const x = val(a, sortKey), y = val(b, sortKey)
    if (typeof x === 'string' || typeof y === 'string') return dir * String(x || '').localeCompare(String(y || ''))
    return dir * ((x ?? -1) - (y ?? -1))
  })
  const click = (k) => { if (k === sortKey) setDir(-dir); else { setSortKey(k); setDir(-1) } }
  if (!rows.length) return <p className="metrics-muted">0 model_metrics rows (db-hugpy `model_metrics`) match the current view.</p>
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
              <td className="num" title={r.throughput ? throughputText(r.throughput) : 'row has no throughput field'}>{r.throughput?.n_calls ? (r.throughput.mean_tok_s != null ? fmt(r.throughput.mean_tok_s) : 'no tok/s (hover: why)') : r.throughput?.error ? 'ledger read failed' : (r.throughput?.reason || 'no calls recorded')}</td>
              <td className="num">{r.throughput ? r.throughput.n_calls : 'no throughput field in the response'}</td>
              <td>{fmtStr(r.task)}</td>
              <td className="num">{r.updated_at ? new Date(r.updated_at * 1000).toLocaleString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
