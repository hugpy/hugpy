import { useCallback, useEffect, useMemo, useState } from 'react'
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
const num = v => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v))
const benchmarkKey = r => [r.worker_id || r.worker, r.model, r.quant, r.config, r.alloc_mode].join('\u0000')
const metric = (v, suffix = '') => {
  const n = num(v)
  return n == null ? 'N/A' : `${n.toFixed(1)}${suffix}`
}

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
  const [benchmarkModel, setBenchmarkModel] = useState('')
  const [benchmarkWorker, setBenchmarkWorker] = useState('')
  const [benchmarkPageModel, setBenchmarkPageModel] = useState('')

  // dialogs
  const [screenOpen, setScreenOpen] = useState(false)
  const [editorOpen, setEditorOpen] = useState(false)

  // ── loaders ───────────────────────────────────────────────────────────────
  const loadCriteria = useCallback(() => {
    fetchJson('/api/llm/review/status')
      .then(d => { if (d && Array.isArray(d.criteria)) setCriteria(d.criteria) })
      .catch(() => {})   // keep last good
  }, [])

  const loadBenchmark = useCallback(() => {
    fetchJson('/api/llm/benchmark/status')
      .then(d => { if (d && d.status) setBenchmark(d) })
      .catch(() => {})
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
      .catch(() => {})
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
  useEffect(() => { loadCriteria(); loadRunNames(); loadBenchmark() }, [loadCriteria, loadRunNames, loadBenchmark])

  useEffect(() => {
    if (benchmark.status !== 'running') return undefined
    const t = setInterval(loadBenchmark, 2500)
    return () => clearInterval(t)
  }, [benchmark.status, loadBenchmark])

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
  const benchmarkRows = useMemo(() => {
    const resultsByKey = new Map((benchmark.results || []).map(r => [benchmarkKey(r), r]))
    const planned = (benchmark.plan?.rows || []).map(p => ({ ...p, ...(resultsByKey.get(benchmarkKey(p)) || {}) }))
    const plannedKeys = new Set(planned.map(benchmarkKey))
    return [...planned, ...(benchmark.results || []).filter(r => !plannedKeys.has(benchmarkKey(r)))]
  }, [benchmark.plan, benchmark.results])
  const benchmarkStats = useMemo(() => {
    const calls = benchmark.calls || []
    const tested = benchmarkRows.filter(r => r.grade && r.grade !== 'N/A')
    const speeds = tested.map(r => num(r.tok_s_avg ?? r.tok_s)).filter(v => v != null)
    return {
      calls: calls.length,
      passed: calls.filter(c => c.grade === 'PASS').length,
      failed: calls.filter(c => c.grade === 'FAIL').length,
      tested: tested.length,
      avgSpeed: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null,
    }
  }, [benchmark.calls, benchmarkRows])
  const benchmarkModels = useMemo(() => [...new Set(benchmarkRows
    .map(r => r.model).filter(Boolean))].sort((a, b) => a.localeCompare(b)), [benchmarkRows])
  useEffect(() => {
    if (!benchmarkModels.length) {
      if (benchmarkPageModel) setBenchmarkPageModel('')
    } else if (!benchmarkModels.includes(benchmarkPageModel)) {
      const active = benchmark.progress?.model
      setBenchmarkPageModel(benchmarkModels.includes(active) ? active : benchmarkModels[0])
    }
  }, [benchmarkModels, benchmarkPageModel, benchmark.progress?.model])
  const benchmarkPageRows = useMemo(() => benchmarkRows
    .filter(r => r.model === benchmarkPageModel), [benchmarkRows, benchmarkPageModel])
  const benchmarkPageIndex = Math.max(0, benchmarkModels.indexOf(benchmarkPageModel))
  const benchmarkTasks = ['math', 'wordprob', 'factual', 'format_primes', 'logic',
    'exact_instruction', 'coding', 'json', 'letters']
  const benchmarkWorkers = useMemo(() => {
    const found = new Map()
    benchmarkPageRows.forEach(r => found.set(r.worker_id || r.worker,
      { id: r.worker_id || r.worker, name: r.worker || r.worker_id }))
    return [...found.values()].sort((a, b) => String(a.name).localeCompare(String(b.name)))
  }, [benchmarkPageRows])
  const benchmarkQuants = useMemo(() => [...new Set(benchmarkPageRows
    .map(r => r.quant).filter(Boolean))].sort((a, b) => a.localeCompare(b)), [benchmarkPageRows])
  const benchmarkSections = [
    { label: 'moe', config: 'moe', modes: null },
    { label: '4-bit:GPU', config: '4bit', modes: ['gpu_only'] },
    { label: '4-bit:RAM', config: '4bit', modes: ['ram_only'] },
    { label: '4-bit:SPLIT', config: '4bit', modes: ['max_gpu'] },
    { label: 'GPU', config: 'full', modes: ['gpu_only'] },
    { label: 'RAM', config: 'full', modes: ['ram_only'] },
    { label: 'SPLIT', config: 'full', modes: ['max_gpu'] },
  ]
  const sectionRow = (section, quant, worker) => benchmarkPageRows.find(r =>
    r.quant === quant && (r.worker_id || r.worker) === worker.id &&
    r.config === section.config && (!section.modes || section.modes.includes(r.alloc_mode)))
  const taskOutcome = (row, wanted) => {
    const entries = Object.entries(row.detail || {})
    const found = entries.find(([, value]) => Boolean(value) === wanted)
    return found ? found[0] : '—'
  }

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

  const runBenchmark = useCallback(() => {
    if (benchmark.status === 'running') return
    const scoped = benchmarkModel.trim() || benchmarkWorker.trim()
    if (!confirm(scoped
      ? `Run scoped capacity test?\n\nModel: ${benchmarkModel.trim() || 'all'}\nWorker: ${benchmarkWorker.trim() || 'all'}`
      : 'Test every feasible model quant/config on every eligible worker?\n\nCold quants will be copied from central to worker drives. Central llm_storage is never deleted. Workers and independent model lanes run concurrently.')) return
    setNote('starting fleet capacity benchmark…')
    fetchJson('/api/llm/benchmark/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ executor: 'hugpy-agent', tokens: 128,
        models: benchmarkModel.trim() ? [benchmarkModel.trim()] : [],
        workers: benchmarkWorker.trim() ? [benchmarkWorker.trim()] : [] }),
    }).then(d => { setBenchmark(d); setNote('fleet benchmark started') })
      .catch(e => setNote(`benchmark failed to start: ${e.message}`))
  }, [benchmark.status, benchmarkModel, benchmarkWorker])

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
        <strong>Grader</strong>
        <span className="rv-dim">screen → smoke-load → judge · results persist in reviews.db</span>

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
        <button onClick={() => { loadCriteria(); loadRunNames(); loadRuns(selected); loadResults(selected) }}>↻</button>
      </div>
      {note && <div className="rv-note">{note}</div>}

      <section className="rv-benchmark">
        <div className="rv-benchmark-head">
          <div>
            <strong>Fleet capacity test</strong>
            <span className="rv-dim"> all central models · full quants, 4-bit and MoE · GPU / spill / RAM</span>
          </div>
          <button onClick={runBenchmark} disabled={benchmark.status === 'running'}>
            {benchmark.status === 'running' ? '● testing…' : (benchmarkModel || benchmarkWorker) ? '▶ Test scope' : '▶ Test all models'}
          </button>
          {benchmark.status === 'running' && <button onClick={() => cancelBenchmark('execution')}>■ Cancel execution</button>}
        </div>
        <div className="rv-benchmark-meta">
          <label>model <input value={benchmarkModel} onChange={e => setBenchmarkModel(e.target.value)}
            disabled={benchmark.status === 'running'} placeholder="all, or exact model key" /></label>
          <label>worker <input value={benchmarkWorker} onChange={e => setBenchmarkWorker(e.target.value)}
            disabled={benchmark.status === 'running'} placeholder="all, or worker name/id" /></label>
        </div>
        <div className="rv-benchmark-meta">
          <span className={`rv-benchmark-status ${benchmark.status}`}>{benchmark.status || 'idle'}</span>
          {benchmark.executor && <span>executor: {benchmark.executor}</span>}
          <span>{benchmark.results?.length || 0} configurations recorded</span>
          {benchmark.report && <span title={benchmark.report}>report: {benchmark.report.split('/').pop()}</span>}
          {benchmark.error && <span className="rv-run-err">{benchmark.error}</span>}
        </div>
        <div className="rv-benchmark-progress">
          <progress max="100" value={benchmark.progress?.percent || 0} />
          <span>{benchmark.progress?.completed || 0}/{benchmark.progress?.total || benchmark.plan?.runnable || 0} · {benchmark.progress?.percent || 0}%</span>
        </div>
        <div className="rv-benchmark-cards">
          <span><b>{benchmarkStats.tested}</b><small>graded configs</small></span>
          <span><b>{benchmarkStats.calls}</b><small>logged calls</small></span>
          <span className="pass"><b>{benchmarkStats.passed}</b><small>passed calls</small></span>
          <span className="fail"><b>{benchmarkStats.failed}</b><small>failed calls</small></span>
          <span><b>{metric(benchmarkStats.avgSpeed)}</b><small>avg tok/s</small></span>
        </div>
        {!!benchmarkRows.length && <div className="rv-model-pager">
          <button disabled={benchmarkPageIndex <= 0}
            onClick={() => setBenchmarkPageModel(benchmarkModels[benchmarkPageIndex - 1])}>← Previous model</button>
          <label>Metrics page
            <select value={benchmarkPageModel} onChange={e => setBenchmarkPageModel(e.target.value)}>
              {benchmarkModels.map(model => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
          <span>{benchmarkPageIndex + 1} / {benchmarkModels.length}</span>
          <button disabled={benchmarkPageIndex >= benchmarkModels.length - 1}
            onClick={() => setBenchmarkPageModel(benchmarkModels[benchmarkPageIndex + 1])}>Next model →</button>
          {benchmark.progress?.model === benchmarkPageModel && benchmark.status === 'running' &&
            <span className="rv-benchmark-status running">● currently testing</span>}
        </div>}
        {!!benchmarkPageRows.length && <div className="rv-model-page-title">
          <strong>{benchmarkPageModel}</strong>
          <span>{benchmarkPageRows.length} worker / quant / configuration rows</span>
        </div>}
        {!!benchmarkPageRows.length && <div className="rv-benchmark-tablewrap">
          <table className="rv-benchmark-table rv-metrics-sheet">
            <thead>
              <tr><th rowSpan="2">{benchmarkPageModel}</th>
                {benchmarkWorkers.map(w => <th key={w.id} colSpan={benchmarkTasks.length + 7}>{w.name}</th>)}
              </tr>
              <tr>{benchmarkWorkers.flatMap(w => [
                <th key={`${w.id}:cold`}>cold</th>, <th key={`${w.id}:hot`}>hot</th>,
                <th key={`${w.id}:tps`}>tok_per_s</th>, <th key={`${w.id}:avg`}>tok_per_s_avg</th>,
                ...benchmarkTasks.map(t => <th key={`${w.id}:${t}`}>{t}</th>),
                <th key={`${w.id}:ideal`}>task_ideal</th>, <th key={`${w.id}:worst`}>task_worst</th>,
                <th key={`${w.id}:overall`}>overall</th>,
              ])}</tr>
            </thead>
            <tbody>{benchmarkSections.flatMap(section => [
              <tr className="rv-sheet-band" key={`${section.label}:band`}>
                <th>{section.label}</th>
                <td colSpan={benchmarkWorkers.length * (benchmarkTasks.length + 7)}> </td>
              </tr>,
              ...benchmarkQuants.map(quant => <tr key={`${section.label}:${quant}`}>
                <th>{quant}</th>
                {benchmarkWorkers.flatMap(worker => {
                  const r = sectionRow(section, quant, worker)
                  if (!r) return Array.from({ length: benchmarkTasks.length + 7 }, (_, n) =>
                    <td className="rv-sheet-na" key={`${worker.id}:${n}`}>N/A</td>)
                  return [
                    <td key={`${worker.id}:cold`}>{fmtSecs(num(r.cold_s))}</td>,
                    <td key={`${worker.id}:hot`}>{fmtSecs(num(r.inference_s ?? r.hot_s))}</td>,
                    <td key={`${worker.id}:tps`}>{metric(r.tok_s)}</td>,
                    <td key={`${worker.id}:avg`}>{metric(r.tok_s_avg)}</td>,
                    ...benchmarkTasks.map(task => <td className={r.detail?.[task] === 1 ? 'rv-sheet-pass' : r.detail?.[task] === 0 ? 'rv-sheet-fail' : ''}
                      key={`${worker.id}:${task}`}>{r.detail?.[task] ?? 'N/A'}</td>),
                    <td key={`${worker.id}:ideal`}>{taskOutcome(r, true)}</td>,
                    <td key={`${worker.id}:worst`}>{taskOutcome(r, false)}</td>,
                    <td className={r.ok ? 'rv-sheet-pass' : 'rv-sheet-fail'} key={`${worker.id}:overall`}>{r.grade || 'N/A'}</td>,
                  ]
                })}
              </tr>),
            ])}</tbody>
          </table>
        </div>}
        {!!benchmarkPageRows.length && <div className="rv-benchmark-tablewrap rv-worker-card-wrap">
          <table className="rv-benchmark-table">
            <thead><tr><th>worker_card</th><th>temperature</th><th>upload_time_s</th><th>tok_per_s</th>
              <th>n_samples</th><th>updated_at</th><th>task</th><th>media_bytes</th><th>analysis</th><th>runtime</th><th>disk</th><th>error</th></tr></thead>
            <tbody>{benchmarkWorkers.map(worker => {
              const rows = benchmarkPageRows.filter(r => (r.worker_id || r.worker) === worker.id)
              const tested = rows.filter(r => r.grade && r.grade !== 'N/A')
              const speeds = tested.map(r => num(r.tok_s_avg ?? r.tok_s)).filter(v => v != null)
              const latest = tested.slice().sort((a, b) => num(b.finished) - num(a.finished))[0] || rows[0] || {}
              return <tr key={worker.id}><th>{worker.name}</th><td>0</td>
                <td>{fmtSecs(num(latest.cold_s))}</td>
                <td>{speeds.length ? metric(speeds.reduce((a, b) => a + b, 0) / speeds.length) : 'N/A'}</td>
                <td>{tested.reduce((n, r) => n + (r.calls?.length || 0), 0)}</td>
                <td>{fmtClock(latest.finished)}</td><td>text-generation</td><td>N/A</td>
                <td>{tested.filter(r => r.metrics_complete).length}/{tested.length}</td>
                <td>{fmtBytes(latest.runtime_bytes)}</td><td>{fmtBytes(latest.disk_bytes)}</td>
                <td>{latest.error || 'N/A'}</td></tr>
            })}</tbody>
          </table>
        </div>}
        {!!benchmark.calls?.length && <details className="rv-call-log">
          <summary>Complete call log ({benchmark.calls.length})</summary>
          {benchmark.calls.slice().reverse().map((c, i) => <details key={`${c.worker_id}:${c.model}:${c.quant}:${c.task}:${i}`}>
            <summary>{c.grade} · {c.worker} · {c.model}/{c.quant} · {c.task} · {c.elapsed_s}s · {c.tok_s} tok/s ({c.tok_s_source || 'source N/A'}) · {c.finish_reason || 'finish N/A'}{c.truncated ? ' · truncated' : ''}</summary>
            {benchmark.status === 'running' && <button onClick={() => cancelBenchmark('call', c)}>skip this call</button>}
            <pre>{JSON.stringify(c, null, 2)}</pre>
          </details>)}
        </details>}
        {!!benchmark.events?.length && <details className="rv-call-log" open={benchmark.status === 'running'}>
          <summary>Live activity log ({benchmark.events.length})</summary>
          <pre>{benchmark.events.slice().reverse().map(e => {
            const stamp = e.at ? new Date(e.at * 1000).toLocaleTimeString() : '—'
            const detail = e.message || (e.value == null ? '' : JSON.stringify(e.value))
            return `${stamp}  ${e.kind || 'event'}  ${detail}`
          }).join('\n')}</pre>
        </details>}
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
