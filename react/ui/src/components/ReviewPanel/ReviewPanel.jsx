import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { runErrorText } from '../MetricsPanel/RunError'
import { GraderWorkbook } from '../MetricsPanel/MetricsPanel'
import ModelLiveState from '../ModelLiveState/ModelLiveState'
import { fetchJson } from '../../api'
import './ReviewPanel.css'

// The model grader/tester tab. Surfaces the existing review pipeline
// (abstract_hugpy_dev/review: screen -> smoke-load -> LLM-judge) over its
// /api/llm/review/* routes. The pipeline already runs in a background thread
// on the GPU box and persists every row to SQLite (reviews.db); this panel is
// purely the operator's window onto it — pick a criteria, kick a screen or a
// full run, and read the leaderboard / runs / per-model dossier that come back.
//
// A POLL MUST NEVER DESTROY GOOD DATA (same rule as CallsPanel): a failed or
// malformed reply keeps the last good list and shows a note.

const POLL_MS = 6000

// A benchmark is ACTIVE while it collects (running / resuming — central restarted
// mid-collection and is continuing the same run_id) or JUDGES (phase 2, the
// scheduled judge pass over the collected outputs) — the poll keeps going through
// all of them.
const BENCHMARK_ACTIVE = ['running', 'resuming', 'judging']
// A central restart (a package promotion) briefly 502s the status poll. The run
// is persisted and resumes when central is back, so this reads as "will resume",
// not a failure — and the poll keeps the last status and keeps polling.
const CENTRAL_RESTART_NOTE = 'central restarting — run will resume (kept the last status)'

// ── formatters ─────────────────────────────────────────────────────────────
const fmtBytes = n => {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}
const fmtParams = n => {
  if (!n) return '—'
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`
  return String(n)
}
const fmtClock = ts => (ts ? new Date(ts * 1000).toLocaleString() : '—')
const fmtAgo = ts => {
  if (!ts) return '—'
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return `${Math.round(s)}s ago`
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}
const fmtSecs = s => (s == null ? '—' : s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(1)} s`)
const fmtScore = s => (s == null ? '—' : Number(s).toFixed(2))

// Eligible = the exact predicate fleet_grading.eligible uses server-side:
// online, reachable, admission approved, serving not turned off. The benchmark
// grades every SELECTED eligible worker, defaulting to all of them.
const workerEligible = w => w && w.status === 'online' && !w.unreachable &&
  (w.admission === 'approved') && w.serve_mode !== 'off'
const workerId = w => w.id || w.name
const num = v => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v))

// verdict -> css class. adopt is the win, trial is a maybe, reject is a no.
const verdictClass = v => {
  if (v === 'adopt') return 'rv-vd-adopt'
  if (v === 'trial') return 'rv-vd-trial'
  if (v === 'reject') return 'rv-vd-reject'
  return 'rv-vd-none'
}
const stageClass = s => {
  if (s === 'smoked') return 'rv-stg-smoked'
  if (s === 'downloaded') return 'rv-stg-downloaded'
  return 'rv-stg-screened'
}

// The subset of ReviewCriteria the console lets you edit inline. The pipeline
// fills every other field from its dataclass defaults, so a criteria created
// here screens/downloads/smokes exactly like a hand-written one.
const CRIT_DEFAULTS = {
  name: '', query: '', task: 'text-generation',
  require_gguf: true, min_tokens_per_sec: 8, target_context: 16384,
  min_context: 8192, min_downloads: 500, max_params: null, min_params: null,
  pool_limit: 60, max_downloads_per_run: 2,
  smoke_test: true, judge: true, enabled: true, radar: false,
}

export default function ReviewPanel() {
  const [criteria, setCriteria] = useState([])     // saved criteria (/status): runnable
  const [runNames, setRunNames] = useState([])     // criteria names seen in /runs (may be history-only)
  const [selected, setSelected] = useState(null)
  const [runs, setRuns]       = useState([])
  const [results, setResults] = useState([])
  const [mode, setMode]       = useState('best')   // 'best' (leaderboard) | 'recent'
  const [stageFilter, setStage] = useState('')     // '' | screened | downloaded | smoked
  const [expanded, setExpanded] = useState({})     // hub_id -> open?
  const [dossiers, setDossiers] = useState({})     // hub_id -> dossier | {error}
  const [note, setNote]       = useState(null)
  const [busy, setBusy]       = useState(false)
  const [benchmark, setBenchmark] = useState({ status: 'idle', results: [], events: [] })
  // The last status the poll saw, read inside loadBenchmark's catch (which has a
  // stable [] identity) so a failed poll can tell a mid-run central restart from
  // an ordinary refresh failure.
  const benchmarkStatusRef = useRef(benchmark.status)
  const [benchmarkModel, setBenchmarkModel] = useState('')
  // Models ticked for grading in the workbook picker (multi-select). Empty = no
  // explicit pick; the run then falls back to the model text box, and if that is
  // blank too, to ALL central models (today's default, stated on the button).
  const [gradeModels, setGradeModels] = useState([])
  // The Fleet-capacity initiator (model input, worker selector, backup/restore
  // status, run button, meta) is collapsed by default and auto-expands only
  // while a run is active — see the run-collapse rule below.
  const [fcOpen, setFcOpen] = useState(false)
  // The fleet's workers (for the benchmark worker selector) and the operator's
  // explicit selection. Default: every ELIGIBLE worker checked — the benchmark
  // grades all selected eligible workers, never only designated seats.
  const [workersList, setWorkersList] = useState([])
  const [selectedWorkers, setSelectedWorkers] = useState(null) // null until first load = "all eligible"
  const [workersNote, setWorkersNote] = useState(null)

  // dialogs
  const [screenOpen, setScreenOpen] = useState(false)
  const [editorOpen, setEditorOpen] = useState(false)

  // ── loaders ───────────────────────────────────────────────────────────────
  const loadCriteria = useCallback(() => {
    fetchJson('/api/llm/review/status')
      .then(d => { if (d && Array.isArray(d.criteria)) setCriteria(d.criteria) })
      .catch(e => setNote(`criteria refresh failed: ${e.message} (kept the last list)`))
  }, [])

  const loadBenchmark = useCallback(() => {
    fetchJson('/api/llm/benchmark/status')
      .then(d => {
        if (d && d.status) {
          setBenchmark(d)
          setNote(n => (n === CENTRAL_RESTART_NOTE ? null : n))   // recovered
        }
      })
      .catch(e => setNote(BENCHMARK_ACTIVE.includes(benchmarkStatusRef.current)
        ? CENTRAL_RESTART_NOTE
        : `benchmark status refresh failed: ${e.message} (kept the last status)`))
  }, [])

  const loadWorkers = useCallback(() => {
    fetchJson('/api/llm/workers')
      .then(d => {
        if (!Array.isArray(d)) { setWorkersNote('worker list refresh failed: unexpected reply (kept the last list)'); return }
        setWorkersList(d)
        setWorkersNote(null)
        // Seed the selection ONCE to every eligible worker; after that the
        // operator's explicit choice is preserved across polls.
        setSelectedWorkers(prev => prev == null
          ? d.filter(workerEligible).map(workerId)
          : prev.filter(id => d.some(w => workerId(w) === id)))
      })
      .catch(e => setWorkersNote(`worker list refresh failed: ${e.message} (kept the last list)`))
  }, [])

  const loadRuns = useCallback((crit) => {
    if (!crit) { setRuns([]); return }
    fetchJson(`/api/llm/review/runs?criteria=${encodeURIComponent(crit)}&limit=25`)
      .then(d => { if (Array.isArray(d)) setRuns(d) })
      .catch(e => setNote(`runs refresh failed: ${e.message} (kept the last list)`))
  }, [])

  const loadRunNames = useCallback(() => {
    // /status only lists criteria with a saved .json file; the DB holds runs for
    // criteria whose files are gone. Union them so history stays viewable.
    fetchJson('/api/llm/review/runs?limit=200')
      .then(d => { if (Array.isArray(d)) setRunNames([...new Set(d.map(r => r.criteria).filter(Boolean))]) })
      .catch(e => setNote(`run history refresh failed: ${e.message} (kept the last list)`))
  }, [])

  const loadResults = useCallback((crit, m = mode, stg = stageFilter) => {
    if (!crit) { setResults([]); return }
    const best = m === 'best'
    const q = best
      ? `/api/llm/review/results?criteria=${encodeURIComponent(crit)}&best=1&limit=100`
      : `/api/llm/review/results?criteria=${encodeURIComponent(crit)}&limit=100${stg ? `&stage=${stg}` : ''}`
    fetchJson(q)
      .then(d => { if (Array.isArray(d)) { setResults(d); setNote(null) } })
      .catch(e => setNote(`results refresh failed: ${e.message} (kept the last list)`))
  }, [mode, stageFilter])

  // first load
  useEffect(() => { loadCriteria(); loadRunNames(); loadBenchmark(); loadWorkers() }, [loadCriteria, loadRunNames, loadBenchmark, loadWorkers])

  // Keep the ref in step with EVERY status change (poll, run start, cancel) so a
  // failed poll's catch reads the true last status.
  useEffect(() => { benchmarkStatusRef.current = benchmark.status }, [benchmark.status])

  useEffect(() => {
    if (!BENCHMARK_ACTIVE.includes(benchmark.status)) return undefined
    const t = setInterval(loadBenchmark, 2500)
    return () => clearInterval(t)
  }, [benchmark.status, loadBenchmark])

  // Run-collapse rule (shared with the Live test output pane): a NEW run
  // starting auto-expands the initiator and clears any prior manual collapse;
  // once it finishes the block stays as the operator left it (open until they
  // collapse or leave). A manual toggle mid-run is respected until the next run
  // starts. Fresh load with nothing running → collapsed. Deriving off run_id
  // means "another poll of the same finished run" never re-opens it.
  const fcPrevRun = useRef(null)
  useEffect(() => {
    const running = benchmark.status === 'running'
    if (running && benchmark.run_id && benchmark.run_id !== fcPrevRun.current) {
      fcPrevRun.current = benchmark.run_id
      setFcOpen(true)
    }
  }, [benchmark.status, benchmark.run_id])

  // pick a default criteria once names arrive
  useEffect(() => {
    if (selected) return
    const first = criteria[0]?.name || runNames[0]
    if (first) setSelected(first)
  }, [criteria, runNames, selected])

  // refetch runs + results when the selection / view changes
  useEffect(() => { loadRuns(selected); loadResults(selected) }, [selected, mode, stageFilter, loadRuns, loadResults])

  // A run in flight for this criteria (finished_at null) → live poll so counts
  // and the leaderboard fill in as the pipeline works.
  const runActive = useMemo(
    () => runs.some(r => r.finished_at == null) ||
          criteria.some(c => c.name === selected && c.running),
    [runs, criteria, selected])

  useEffect(() => {
    const t = setInterval(() => {
      loadCriteria()
      loadRuns(selected)
      if (runActive) loadResults(selected)
    }, POLL_MS)
    return () => clearInterval(t)
  }, [selected, runActive, loadCriteria, loadRuns, loadResults])

  const critNames = useMemo(
    () => [...new Set([...criteria.map(c => c.name), ...runNames])].sort(),
    [criteria, runNames])
  const selectedCrit = criteria.find(c => c.name === selected) || null
  const runnable = !!selectedCrit          // only a saved criteria file can Run
  const running  = !!selectedCrit?.running
  // Live progress, pass/fail counts and avg tok/s for a run are owned by the
  // single "Live test output" pane (GraderWorkbook / BenchmarkLiveOutput) below,
  // so this control surface no longer computes or renders its own copies.
  // ── actions ─────────────────────────────────────────────────────────────
  const runNow = useCallback(() => {
    if (!runnable || running || busy) return
    if (!confirm(`Run the full review pipeline for "${selected}"?\n\nThis downloads candidate weights and loads them on the GPU (minutes to hours). One run per criteria at a time.`)) return
    setBusy(true)
    fetchJson('/api/llm/review/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ criteria: selected }),
    })
      .then(d => { setNote(d?.status === 'already_running' ? `already running for ${selected}` : `run started for ${selected}`); loadCriteria(); loadRuns(selected) })
      .catch(e => setNote(`run failed: ${e.message}`))
      .finally(() => setBusy(false))
  }, [runnable, running, busy, selected, loadCriteria, loadRuns])

  const eligibleWorkers = useMemo(() => workersList.filter(workerEligible), [workersList])
  const chosenWorkers = selectedWorkers || []
  const toggleWorker = useCallback(id => setSelectedWorkers(prev => {
    const base = prev || eligibleWorkers.map(workerId)
    return base.includes(id) ? base.filter(x => x !== id) : [...base, id]
  }), [eligibleWorkers])

  const runBenchmark = useCallback(() => {
    if (benchmark.status === 'running') return
    // The benchmark runs on ALL SELECTED eligible workers. An empty selection is
    // refused rather than silently meaning "all" — the operator picks the seats.
    if (!chosenWorkers.length) { setNote('pick at least one worker to test on (default is every eligible worker)'); return }
    const nameOf = id => (workersList.find(w => workerId(w) === id)?.name) || id
    const wLabel = chosenWorkers.length === eligibleWorkers.length ? `all ${chosenWorkers.length} eligible workers` : `${chosenWorkers.length} worker(s): ${chosenWorkers.map(nameOf).join(', ')}`
    // Model set: ticked models win; else the exact-key text box; else all central
    // models. Never a silent default — the choice is spelled out in the prompt.
    const models = gradeModels.length ? gradeModels : (benchmarkModel.trim() ? [benchmarkModel.trim()] : [])
    const mLabel = gradeModels.length ? `${gradeModels.length} selected model(s): ${gradeModels.join(', ')}`
      : (benchmarkModel.trim() || 'ALL central models')
    if (!confirm(`Run the capacity test?\n\nModels: ${mLabel}\nWorkers: ${wLabel}\n\nBefore each involved worker is tested, its state (allocations/pins, per-model quant & levers, and which model files are on its drive) is backed up, and restored when the test ends. Cold quants may be copied from central to worker drives to measure cold loads; central llm_storage is never deleted.`)) return
    setNote('starting fleet capacity benchmark…')
    setFcOpen(true)   // expand the initiator immediately on Run, before the first poll
    fetchJson('/api/llm/benchmark/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ executor: 'hugpy-central', tokens: 128, models, workers: chosenWorkers }),
    }).then(d => { setBenchmark(d); setNote('fleet benchmark started') })
      .catch(e => setNote(`benchmark failed to start: ${e.message}`))
  }, [benchmark.status, benchmarkModel, gradeModels, chosenWorkers, eligibleWorkers, workersList])

  // PHASE 2 on demand: grade the collected outputs still pending (a finished or
  // partial run). Central refuses (409) while a run is still collecting/judging
  // or when nothing has been collected.
  const judgeNow = useCallback(() => {
    setNote('starting judging of collected outputs…')
    fetchJson('/api/llm/benchmark/judge', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
    }).then(d => { setBenchmark(d); setNote('judging started') })
      .catch(e => setNote(`judge failed to start: ${e.message}`))
  }, [])

  const cancelBenchmark = useCallback((scope, row = {}) => {
    fetchJson('/api/llm/benchmark/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope, worker_id: row.worker_id, model: row.model,
                             quant: row.quant, call: row.task }),
    }).then(setBenchmark).catch(e => setNote(`cancel failed: ${e.message}`))
  }, [])

  const toggleExpand = useCallback(k => setExpanded(o => ({ ...o, [k]: !o[k] })), [])

  const loadDossier = useCallback((hubId) => {
    if (!selected) return
    setDossiers(d => ({ ...d, [hubId]: { loading: true } }))
    fetchJson(`/api/llm/review/dossier?criteria=${encodeURIComponent(selected)}&hub_id=${encodeURIComponent(hubId)}`)
      .then(d => setDossiers(prev => ({ ...prev, [hubId]: d })))
      .catch(e => setDossiers(prev => ({ ...prev, [hubId]: { error: e.message } })))
  }, [selected])

  return (
    <div className="rv-panel">
      {/* ── criteria bar ─────────────────────────────────────────────────── */}
      <div className="rv-head">
        <strong title="Discovery pipeline for a saved criteria: screen → smoke-load → judge. Every row persists in reviews.db and appears in the leaderboard below.">Grader</strong>

        <select className="rv-crit" value={selected || ''}
                onChange={e => { setSelected(e.target.value); setExpanded({}); setDossiers({}) }}>
          {!critNames.length && <option value="">no criteria yet</option>}
          {critNames.map(n => {
            const c = criteria.find(x => x.name === n)
            const tag = !c ? ' (history)' : (c.running ? ' ● running' : (c.enabled === false ? ' (disabled)' : ''))
            return <option key={n} value={n}>{n}{tag}</option>
          })}
        </select>

        <button onClick={runNow} disabled={!runnable || running || busy}
                title={runnable ? 'Run the full pipeline in the background' : 'No saved criteria file — create one to run'}>
          {running ? '● running…' : '▶ Run now'}
        </button>
        <button onClick={() => setScreenOpen(true)}>🔍 Screen…</button>
        <button onClick={() => setEditorOpen(true)}>{runnable ? '✎ Edit' : '＋ New'} criteria</button>
        <button onClick={() => { loadCriteria(); loadRunNames(); loadRuns(selected); loadResults(selected); loadWorkers() }}>↻</button>
      </div>
      {note && <div className="rv-note">{note}</div>}

      <section className="rv-benchmark">
       <details className="rv-benchmark-collapse" open={fcOpen}
                onToggle={e => setFcOpen(e.currentTarget.open)}>
        <summary className="rv-benchmark-summary">
          <span className="rv-benchmark-title">Fleet capacity test</span>
          <span className="rv-dim">{benchmark.status === 'running' || benchmark.status === 'resuming'
            ? `● collecting ${benchmark.progress?.completed || 0}/${benchmark.progress?.total || benchmark.plan?.runnable || 0}`
            : benchmark.status === 'judging'
            ? `● judging ${benchmark.judge_progress?.judged || 0}/${benchmark.judge_progress?.total || 0}`
            : `${benchmark.status || 'idle'} · ${benchmark.results?.length || 0} config(s)${benchmark.finished ? ` · last run ${fmtAgo(benchmark.finished)}` : ''}`}</span>
        </summary>
        <div className="rv-benchmark-head">
          <span className="rv-dim">all central models · full quants, 4-bit and MoE · GPU / spill / RAM</span>
          <button onClick={runBenchmark} disabled={benchmark.status === 'running' || !chosenWorkers.length}>
            {benchmark.status === 'running' ? '● testing…'
              : gradeModels.length ? `▶ Grade ${gradeModels.length} model(s) on ${chosenWorkers.length} worker(s)`
              : benchmarkModel.trim() ? `▶ Test ${benchmarkModel.trim()} on ${chosenWorkers.length} worker(s)`
              : `▶ Grade ALL central models on ${chosenWorkers.length === eligibleWorkers.length ? 'all' : chosenWorkers.length} worker(s)`}
          </button>
          {(benchmark.status === 'running' || benchmark.status === 'resuming') && <button onClick={() => cancelBenchmark('execution')}>■ Cancel execution</button>}
          {benchmark.status === 'judging' && <button onClick={() => cancelBenchmark('execution')}>■ Cancel judging</button>}
          {!BENCHMARK_ACTIVE.includes(benchmark.status) && (benchmark.results?.length > 0) &&
            <button onClick={judgeNow} title="Phase 2: grade every collected output still awaiting the judge (the agent default brain, placed by central)">⚖ Judge now</button>}
        </div>
        <div className="rv-benchmark-meta">
          {gradeModels.length
            ? <span><strong>{gradeModels.length}</strong> model(s) ticked for grading in the workbook below
                <button type="button" className="rv-linkbtn" disabled={benchmark.status === 'running'}
                  onClick={() => setGradeModels([])}>clear selection</button></span>
            : <label>model <input value={benchmarkModel} onChange={e => setBenchmarkModel(e.target.value)}
                disabled={benchmark.status === 'running'} placeholder="all central models, or an exact model key" /></label>}
          <span className="rv-dim">tick models in the workbook to grade a chosen set; blank = all central models</span>
        </div>
        <div className="rv-benchmark-workers">
          <div className="rv-benchmark-workers-head">
            <span>workers to test</span>
            <span className="rv-dim">{chosenWorkers.length}/{eligibleWorkers.length} eligible selected</span>
            <button type="button" className="rv-linkbtn" disabled={benchmark.status === 'running'}
              onClick={() => setSelectedWorkers(eligibleWorkers.map(workerId))}>all</button>
            <button type="button" className="rv-linkbtn" disabled={benchmark.status === 'running'}
              onClick={() => setSelectedWorkers([])}>none</button>
          </div>
          <div className="rv-worker-checks">
            {workersList.map(w => {
              const id = workerId(w)
              const elig = workerEligible(w)
              const why = elig ? '' : [w.status !== 'online' && w.status, w.unreachable && 'unreachable',
                w.admission !== 'approved' && `admission ${w.admission || 'unknown'}`,
                w.serve_mode === 'off' && 'serving off'].filter(Boolean).join(', ')
              return <label key={id} className={`rv-worker-check${elig ? '' : ' rv-worker-inelig'}`}
                title={elig ? `${id} — eligible` : `${id} — not eligible: ${why}`}>
                <input type="checkbox" checked={chosenWorkers.includes(id)} disabled={!elig || benchmark.status === 'running'}
                  onChange={() => toggleWorker(id)} />
                {w.name || id}{elig ? '' : <span className="rv-dim"> ({why})</span>}
              </label>
            })}
            {!workersList.length && <span className="rv-dim">no workers reported by /llm/workers{workersNote ? ` — ${workersNote}` : ''}</span>}
          </div>
          {workersNote && !!workersList.length && <div className="rv-dim">{workersNote}</div>}
        </div>
        <BenchmarkStateStatus benchmark={benchmark} />
        <div className="rv-benchmark-meta">
          <span className={`rv-benchmark-status ${benchmark.status}`}>{benchmark.status || 'idle'}</span>
          {benchmark.executor && <span>executor: {benchmark.executor}</span>}
          <span>{benchmark.results?.length || 0} configurations recorded</span>
          {benchmark.report && <span title={benchmark.report}>report: {benchmark.report.split('/').pop()}</span>}
          {benchmark.error && <span className="rv-run-err">{runErrorText(benchmark.error)}</span>}
        </div>
        {(benchmark.status === 'judging' || benchmark.judge_summary) && benchmark.judge_progress &&
          <div className="rv-benchmark-meta">
            <span className="rv-dim">judge (phase 2): {benchmark.judge_progress.judged || 0}/{benchmark.judge_progress.total || 0} judged
              {benchmark.judge_progress.pending ? ` · ${benchmark.judge_progress.pending} unjudged` : ''}
              {benchmark.judge_progress.revised ? ` · ${benchmark.judge_progress.revised} revised by judge` : ''}
              {benchmark.judge_progress.refused ? ` · ${benchmark.judge_progress.refused} judge refusal(s)` : ''}</span>
          </div>}
        {benchmarkModel.trim() && <div className="rv-benchmark-meta"><ModelLiveState modelKey={benchmarkModel.trim()} /></div>}
       </details>
        {/* Live progress, pass/fail counts, avg tok/s, the activity log and the
            per-model grading matrix are all rendered ONCE by the workbook below:
            its "Live test output" pane owns the run's live numbers so this
            control surface (worker selector + run action) never competes with a
            second copy of them. The Benchmark workbook matrix stays visible (it
            is the results view); only the initiator above and the Live test
            output pane inside collapse by default. */}
        <GraderWorkbook benchmark={benchmark} benchmarkWorkers={chosenWorkers}
          gradeModels={gradeModels} onGradeModelsChange={setGradeModels} />
        {!!Object.keys(benchmark.summary || {}).length && <details className="rv-call-log">
          <summary>Worker, model, quant and execution totals</summary>
          <pre>{JSON.stringify(benchmark.summary, null, 2)}</pre>
        </details>}
      </section>

      {/* ── runs strip ───────────────────────────────────────────────────── */}
      {selected && (
        <div className="rv-runs">
          <span className="rv-runs-label">Runs</span>
          {!runs.length && <span className="rv-dim">no runs recorded for “{selected}”.</span>}
          {runs.slice(0, 12).map(r => (
            <span key={r.id} className={`rv-run${r.finished_at == null ? ' rv-run-live' : ''}${r.error ? ' rv-run-err' : ''}`}
                  title={`run #${r.id}\nstarted ${fmtClock(r.started_at)}\n${r.finished_at ? `finished ${fmtClock(r.finished_at)}` : 'in progress'}${r.error ? `\nerror: ${r.error}` : ''}${r.source_host ? `\nhost: ${r.source_host}` : ''}`}>
              <b>#{r.id}</b>
              <span className="rv-run-when">{r.finished_at == null ? '● live' : fmtAgo(r.finished_at)}</span>
              <span className="rv-run-counts">
                {r.screened || 0}s · {r.passed || 0}✓ · {r.downloaded || 0}↓ · {r.smoked || 0}🔥
              </span>
            </span>
          ))}
        </div>
      )}

      {/* ── results controls ─────────────────────────────────────────────── */}
      <div className="rv-controls">
        <div className="rv-seg">
          <button className={mode === 'best' ? 'on' : ''} onClick={() => setMode('best')} title="Best-scoring distinct models that passed">🏆 Leaderboard</button>
          <button className={mode === 'recent' ? 'on' : ''} onClick={() => setMode('recent')} title="Every recent review row, newest first">🕑 Recent</button>
        </div>
        {mode === 'recent' && (
          <div className="rv-seg">
            {['', 'screened', 'downloaded', 'smoked'].map(s => (
              <button key={s || 'all'} className={stageFilter === s ? 'on' : ''} onClick={() => setStage(s)}>
                {s ? s : 'all stages'}
              </button>
            ))}
          </div>
        )}
        <span className="rv-dim">{results.length} model{results.length === 1 ? '' : 's'}</span>
      </div>

      {/* ── results table ────────────────────────────────────────────────── */}
      <div className="rv-tablewrap">
        <table className="rv-table">
          <thead>
            <tr>
              <th>Model</th><th>Stage</th><th>Verdict</th><th className="rv-num">Score</th>
              <th className="rv-num">Gen tok/s</th><th className="rv-num">Load</th>
              <th className="rv-num">Params</th><th className="rv-num">Est VRAM</th>
              <th className="rv-num">Downloads</th><th>Reviewed</th>
            </tr>
          </thead>
          <tbody>
            {results.map(r => {
              const p = r.payload || {}
              const sm = p.smoke || {}
              const sc = p.screen || {}
              const open = !!expanded[r.hub_id]
              return (
                <ResultRow key={`${r.hub_id}:${r.id}`}
                           r={r} p={p} sm={sm} sc={sc} open={open}
                           onToggle={() => toggleExpand(r.hub_id)}
                           dossier={dossiers[r.hub_id]}
                           onLoadDossier={() => loadDossier(r.hub_id)} />
              )
            })}
            {!results.length && (
              <tr><td colSpan={10} className="rv-dim">
                {selected ? `no ${mode === 'best' ? 'passing models' : 'reviews'} recorded for “${selected}” yet — run a screen or the full pipeline.` : 'pick or create a criteria to begin.'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Only hand Screen a criteria NAME when it's a saved file — the /screen
          route loads it by name and 500s on a history-only name. Unsaved →
          the backend's built-in "adhoc" defaults. */}
      {screenOpen && <ScreenDialog criteria={runnable ? selected : null} onClose={() => setScreenOpen(false)} />}
      {editorOpen && (
        <CriteriaDialog
          initial={selectedCrit}
          onClose={() => setEditorOpen(false)}
          onSaved={(name) => { setEditorOpen(false); loadCriteria(); loadRunNames(); setSelected(name) }}
        />
      )}
    </div>
  )
}

// ── worker state snapshot / restore status ──────────────────────────────────
// The benchmark backs up each involved worker's state before the test and
// restores it after. Both halves ride the run's summary (summary.state) and its
// events; this surfaces them per the metrics/grading contract — every restore
// result, every failure with its reason, no truncation, absences explicit.
function BenchmarkStateStatus({ benchmark }) {
  const state = benchmark?.summary?.state || null
  // Fall back to the live events so status shows DURING the run too.
  const evSnap = (benchmark?.events || []).filter(e => e?.kind === 'state-snapshot').slice(-1)[0]?.value
  const evRest = (benchmark?.events || []).filter(e => e?.kind === 'state-restore').map(e => e.value).filter(Boolean)
  const snap = state?.snapshot || evSnap || null
  const restore = state?.restore || (evRest.length ? { workers: evRest } : null)
  if (!snap && !restore) return null
  return (
    <details className="rv-benchmark-state" open={!!restore}>
      <summary>Worker state backup &amp; restore{restore ? ` · restored ${(restore.workers || []).length} worker(s)` : ' · snapshot taken'}</summary>
      {snap && (
        <div className="rv-state-block">
          <b>Backed up</b> before the test:
          <span className="rv-dim"> {(snap.workers || []).length} worker(s) [{(snap.workers || []).join(', ') || 'none'}]
            {snap.models?.length ? ` · ${snap.models.length} model(s)` : ' · all central models'}
            {snap.error ? ` · SNAPSHOT ERROR: ${snap.error}` : ''}</span>
        </div>
      )}
      {restore && (
        <div className="rv-state-block">
          <b>Restored</b> after the test:
          {!(restore.workers || []).length && <span className="rv-dim"> nothing to restore (no state changed)</span>}
          <ul className="rv-state-list">
            {(restore.workers || []).map((w, i) => (
              <li key={w.worker_id || w.worker || i} className={(w.failures || []).length ? 'rv-state-fail' : ''}>
                <b>{w.worker || w.worker_id}</b>
                {' '}restored: {(w.restored || []).length ? w.restored.join('; ') : 'no changes needed'}
                {(w.redownloaded || []).length ? ` · re-downloaded to disk: ${w.redownloaded.join(', ')}` : ''}
                {(w.removed || []).length ? ` · removed to fit prior models: ${w.removed.join(', ')}` : ''}
                {(w.failures || []).map((f, j) => (
                  <div key={j} className="rv-state-failline">✗ {f.item ? `${f.item}: ` : ''}{f.error}</div>
                ))}
              </li>
            ))}
          </ul>
          {restore.error && <div className="rv-state-failline">✗ restore error: {restore.error}</div>}
        </div>
      )}
    </details>
  )
}

// ── one result row + its expandable detail ──────────────────────────────────
function ResultRow({ r, p, sm, sc, open, onToggle, dossier, onLoadDossier }) {
  return (
    <>
      <tr className={`rv-row ${open ? 'rv-row-open' : ''} ${r.passed ? '' : 'rv-row-dim'}`} onClick={onToggle}>
        <td className="rv-model">
          <span className="rv-caret">{open ? '▾' : '▸'}</span>
          <span title={r.hub_id}>{r.hub_id}</span>
        </td>
        <td><span className={`rv-stg ${stageClass(r.stage)}`}>{r.stage}</span></td>
        <td>{r.verdict ? <span className={`rv-vd ${verdictClass(r.verdict)}`}>{r.verdict}</span> : <span className="rv-dim">—</span>}</td>
        <td className="rv-num rv-score">{fmtScore(r.score)}</td>
        <td className="rv-num">{sm.gen_tokens_per_sec != null ? sm.gen_tokens_per_sec : '—'}</td>
        <td className="rv-num">{fmtSecs(sm.load_seconds)}</td>
        <td className="rv-num">{fmtParams(num(sm.n_params) || num(sc.params))}</td>
        <td className="rv-num">{fmtBytes(num(sc.est_vram_bytes))}</td>
        <td className="rv-num">{sc.downloads != null ? Number(sc.downloads).toLocaleString() : '—'}</td>
        <td className="rv-dim" title={fmtClock(r.reviewed_at)}>{fmtAgo(r.reviewed_at)}</td>
      </tr>
      {open && (
        <tr className="rv-detailrow">
          <td colSpan={10}>
            <div className="rv-detail">
              <DetailBlock title="Screen (metadata)">
                <Field k="architecture">{sc.architecture || '—'}</Field>
                <Field k="best quant">{sc.best_quant || '—'}</Field>
                <Field k="context">{sc.context_length != null ? sc.context_length.toLocaleString() : '—'}</Field>
                <Field k="KV cache">{fmtBytes(num(sc.kv_bytes))}</Field>
                <Field k="base model">{sc.base_model || '—'}</Field>
                <Field k="license">{sc.license || '—'}{sc.gated ? ' · gated' : ''}</Field>
                <Field k="age">{sc.age_days != null ? `${sc.age_days}d` : '—'}</Field>
                {Array.isArray(sc.reasons) && sc.reasons.length > 0 && (
                  <Field k="reasons" wide>{sc.reasons.join(' · ')}</Field>
                )}
              </DetailBlock>

              {(p.smoke) && (
                <DetailBlock title="Smoke (real load)">
                  <Field k="loaded">{sm.ok ? '✓ ok' : '✗ failed'}{sm.gpu_offloaded ? ' · GPU' : ''}</Field>
                  <Field k="gen tok/s">{sm.gen_tokens_per_sec ?? '—'}</Field>
                  <Field k="prompt tok/s">{sm.prompt_tokens_per_sec ?? '—'}</Field>
                  <Field k="load">{fmtSecs(sm.load_seconds)}</Field>
                  <Field k="ctx used">{sm.n_ctx_used != null ? `${sm.n_ctx_used}${sm.n_ctx_train ? ` / ${sm.n_ctx_train}` : ''}` : '—'}</Field>
                  {sm.coherence != null && <Field k="coherence">{String(sm.coherence)}</Field>}
                  {sm.error && <Field k="error" wide>{sm.error}</Field>}
                </DetailBlock>
              )}

              {p.judgement && (
                <DetailBlock title="Judge (agent verdict)">
                  <Field k="verdict"><span className={`rv-vd ${verdictClass(p.judgement.verdict)}`}>{p.judgement.verdict || '—'}</span></Field>
                  {p.judgement.confidence != null && <Field k="confidence">{p.judgement.confidence}</Field>}
                  {p.judgement.score != null && <Field k="score">{fmtScore(p.judgement.score)}</Field>}
                  {Array.isArray(p.judgement.reasons) && p.judgement.reasons.length > 0 && (
                    <Field k="reasons" wide>{p.judgement.reasons.join(' · ')}</Field>
                  )}
                </DetailBlock>
              )}

              {p.error && <div className="rv-detail-err">pipeline error: {String(p.error)}</div>}

              <div className="rv-detail-actions">
                <a className="rv-linkbtn" href={`https://huggingface.co/${r.hub_id}`} target="_blank" rel="noreferrer">↗ Hugging Face</a>
                {!dossier && <button onClick={onLoadDossier}>📄 Load dossier</button>}
                {dossier?.loading && <span className="rv-dim">loading dossier…</span>}
                {dossier?.error && <span className="rv-dim">dossier: {dossier.error}</span>}
              </div>

              {dossier && !dossier.loading && !dossier.error && <Dossier d={dossier} />}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function Dossier({ d }) {
  // The full dossier is deep; show the sections an operator scans first.
  const spec = d.specialization
  const trial = d.trial
  const verdict = d.verdict
  return (
    <div className="rv-dossier">
      <DetailBlock title="Dossier">
        {spec?.headline && <Field k="specialization" wide>{spec.headline}</Field>}
        {Array.isArray(spec?.domains) && spec.domains.length > 0 && <Field k="domains">{spec.domains.join(', ')}</Field>}
        {d.community?.heat != null && <Field k="community heat">{d.community.heat}</Field>}
        {d.research?.papers?.length != null && <Field k="papers">{d.research.papers.length}</Field>}
        {trial?.mean_quality != null && <Field k="mean quality">{fmtScore(trial.mean_quality)}</Field>}
        {trial?.depth && <Field k="trial depth">{trial.depth}</Field>}
        {verdict?.verdict && <Field k="verdict"><span className={`rv-vd ${verdictClass(verdict.verdict)}`}>{verdict.verdict}</span></Field>}
        {Array.isArray(verdict?.reasons) && verdict.reasons.length > 0 && <Field k="reasons" wide>{verdict.reasons.join(' · ')}</Field>}
      </DetailBlock>
    </div>
  )
}

function DetailBlock({ title, children }) {
  return (
    <div className="rv-dblock">
      <div className="rv-dtitle">{title}</div>
      <div className="rv-dfields">{children}</div>
    </div>
  )
}
function Field({ k, children, wide }) {
  return <span className={`rv-field${wide ? ' rv-field-wide' : ''}`}><span className="rv-field-k">{k}</span><span className="rv-field-v">{children}</span></span>
}

// ── metadata-only screen dialog ─────────────────────────────────────────────
function ScreenDialog({ criteria, onClose }) {
  const [text, setText] = useState('')
  const [running, setRunning] = useState(false)
  const [out, setOut] = useState(null)
  const [err, setErr] = useState(null)

  const run = useCallback(() => {
    const hub_ids = text.split(/[\s,]+/).map(s => s.trim()).filter(Boolean)
    if (!hub_ids.length) { setErr('enter at least one hub id (e.g. Qwen/Qwen2.5-7B-Instruct-GGUF)'); return }
    if (hub_ids.length > 50) { setErr('at most 50 hub ids per request'); return }
    setRunning(true); setErr(null)
    const body = { hub_ids }
    if (criteria) body.criteria_name = criteria
    fetchJson('/api/llm/review/screen', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(d => setOut(d))
      .catch(e => setErr(e.message))
      .finally(() => setRunning(false))
  }, [text, criteria])

  return (
    <div className="rv-modal-back" onClick={onClose}>
      <div className="rv-modal" onClick={e => e.stopPropagation()}>
        <div className="rv-modal-head">
          <strong>Screen models (metadata only)</strong>
          <button className="rv-x" onClick={onClose}>✕</button>
        </div>
        <p className="rv-dim">Fast paper check — fit, quant, downloads, recency. Fetches no weights. Uses criteria “{criteria || 'adhoc'}”.</p>
        <textarea className="rv-ta" rows={4} value={text} onChange={e => setText(e.target.value)}
                  placeholder={'One or more HF hub ids, whitespace/comma separated:\nQwen/Qwen2.5-7B-Instruct-GGUF  bartowski/…'} />
        {err && <div className="rv-note">{err}</div>}
        <div className="rv-modal-actions">
          <button onClick={run} disabled={running}>{running ? 'screening…' : '🔍 Screen'}</button>
          <button onClick={onClose}>Close</button>
        </div>
        {out && (
          <div className="rv-screen-out">
            <div className="rv-dim">criteria: {out.criteria} · {out.results?.length || 0} result(s)</div>
            {(out.results || []).map((s, i) => (
              <div key={i} className={`rv-screen-row ${s.passed ? 'ok' : 'no'}`}>
                <span className="rv-screen-verdict">{s.passed ? '✓ pass' : '✗ fail'}</span>
                <span className="rv-screen-hub" title={s.hub_id}>{s.hub_id}</span>
                <span className="rv-dim">
                  {s.best_quant ? `${s.best_quant} · ` : ''}{s.est_vram_bytes ? `${fmtBytes(s.est_vram_bytes)} · ` : ''}
                  {s.downloads != null ? `${Number(s.downloads).toLocaleString()} dl` : ''}
                </span>
                {Array.isArray(s.reasons) && s.reasons.length > 0 && <span className="rv-screen-reasons">{s.reasons.join(' · ')}</span>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── criteria create/edit dialog ─────────────────────────────────────────────
function CriteriaDialog({ initial, onClose, onSaved }) {
  const [f, setF] = useState(() => ({ ...CRIT_DEFAULTS, ...(initial || {}) }))
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState(null)
  const editing = !!initial
  const set = (k, v) => setF(o => ({ ...o, [k]: v }))

  const save = useCallback(() => {
    const name = (f.name || '').trim()
    if (!name) { setErr('name is required'); return }
    setSaving(true); setErr(null)
    // Send only the fields we edit; the backend keeps its defaults for the rest.
    const body = {
      query: f.query, task: f.task || null,
      require_gguf: !!f.require_gguf, min_tokens_per_sec: num(f.min_tokens_per_sec) ?? 8,
      target_context: num(f.target_context) ?? 16384, min_context: num(f.min_context) ?? 8192,
      min_downloads: num(f.min_downloads) ?? 0,
      max_params: num(f.max_params), min_params: num(f.min_params),
      pool_limit: num(f.pool_limit) ?? 60, max_downloads_per_run: num(f.max_downloads_per_run) ?? 2,
      smoke_test: !!f.smoke_test, judge: !!f.judge, enabled: !!f.enabled, radar: !!f.radar,
    }
    fetchJson(`/api/llm/review/criteria/${encodeURIComponent(name)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(() => onSaved(name))
      .catch(e => setErr(e.message))
      .finally(() => setSaving(false))
  }, [f, onSaved])

  return (
    <div className="rv-modal-back" onClick={onClose}>
      <div className="rv-modal" onClick={e => e.stopPropagation()}>
        <div className="rv-modal-head">
          <strong>{editing ? `Edit criteria “${initial.name}”` : 'New criteria'}</strong>
          <button className="rv-x" onClick={onClose}>✕</button>
        </div>
        <p className="rv-dim">A saved, re-runnable question: what to search HF for, what must fit the card, and how fast is “usable”. Advanced knobs keep their pipeline defaults.</p>
        <div className="rv-form">
          <label>Name<input value={f.name} disabled={editing} onChange={e => set('name', e.target.value)} placeholder="nightly" /></label>
          <label>Task<input value={f.task || ''} onChange={e => set('task', e.target.value)} placeholder="text-generation" /></label>
          <label className="rv-form-wide">HF search query<input value={f.query} onChange={e => set('query', e.target.value)} placeholder="reasoning coder gguf" /></label>
          <label>Min tok/s<input type="number" value={f.min_tokens_per_sec} onChange={e => set('min_tokens_per_sec', e.target.value)} /></label>
          <label>Target context<input type="number" value={f.target_context} onChange={e => set('target_context', e.target.value)} /></label>
          <label>Min context<input type="number" value={f.min_context} onChange={e => set('min_context', e.target.value)} /></label>
          <label>Min downloads<input type="number" value={f.min_downloads} onChange={e => set('min_downloads', e.target.value)} /></label>
          <label>Max params<input type="number" value={f.max_params ?? ''} onChange={e => set('max_params', e.target.value)} placeholder="(none)" /></label>
          <label>Pool limit<input type="number" value={f.pool_limit} onChange={e => set('pool_limit', e.target.value)} /></label>
          <label>Max downloads/run<input type="number" value={f.max_downloads_per_run} onChange={e => set('max_downloads_per_run', e.target.value)} /></label>
          <div className="rv-form-checks">
            <label className="rv-chk"><input type="checkbox" checked={f.require_gguf} onChange={e => set('require_gguf', e.target.checked)} />GGUF only</label>
            <label className="rv-chk"><input type="checkbox" checked={f.smoke_test} onChange={e => set('smoke_test', e.target.checked)} />Smoke test</label>
            <label className="rv-chk"><input type="checkbox" checked={f.judge} onChange={e => set('judge', e.target.checked)} />Judge</label>
            <label className="rv-chk"><input type="checkbox" checked={f.enabled} onChange={e => set('enabled', e.target.checked)} />Enabled</label>
            <label className="rv-chk"><input type="checkbox" checked={f.radar} onChange={e => set('radar', e.target.checked)} />Gem radar</label>
          </div>
        </div>
        {err && <div className="rv-note">{err}</div>}
        <div className="rv-modal-actions">
          <button onClick={save} disabled={saving}>{saving ? 'saving…' : '💾 Save criteria'}</button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  )
}
