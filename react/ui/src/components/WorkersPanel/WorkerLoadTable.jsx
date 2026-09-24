import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchJson } from '../../api'
import { modelTask, modelTasks } from '../ModelTable/ModelTable'
import useSessionState from '../../hooks/useSessionState'
import { fmtBytes } from './formatters'
import { findCatalogRow } from './catalogRow'
import { archiveMark, archiveText } from '../ModelTable/archiveMark'
import { sizeView } from '../ModelTable/modelSize'

// Per-worker "load a model" as a SORTABLE, MULTI-SELECT table. Columns assort
// (click a header to sort); a checkbox column allocates a GROUP of models to
// THIS worker in one action. Anti-duplicate: a model already allocated to
// ANOTHER worker is locked ("on <worker>") to prevent accidental dual
// allocation — unless the operator flips the ⚡ breaker to deliberately
// replicate it (e.g. for scene fan-out across the fleet).
export function WorkerLoadTable({ models, allocation, workerId, worker, onAllocate, onCancel }) {
  const keyOf = (m) => m.model_key ?? m.key
  // TRUE per-model size: the effective serving quant for GGUF (size_bytes ||
  // effective_bytes from the /models feed), the on-disk footprint otherwise —
  // never the all-quants dir sum. Same rule the ModelPicker uses.
  const sizeOf = (m) => {
    const s = m.size_bytes != null ? m.size_bytes : m.effective_bytes
    return s != null ? Number(s) : null
  }
  const isGgufModel = (m) => {
    const fw = String(m.framework || '').toLowerCase()
    return fw === 'gguf' || fw === 'llama_cpp'
  }
  const [sel, setSel] = useState(() => new Set())
  const [sortKey, setSortKey] = useSessionState('hugpy.sess.wp.load.sort', 'name')
  const [sortDir, setSortDir] = useSessionState('hugpy.sess.wp.load.dir', 'asc')
  const [taskFilter, setTaskFilter] = useSessionState('hugpy.sess.wp.load.task', '')
  const [breaker, setBreaker] = useState(false)
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  // Central-provisioning readiness (AUTHORITATIVE — same check the loader
  // enforces; the manifest `status` is NOT reliable): a map keyed by model_key
  // → {state:'ready'|'incomplete'|'absent'|'error', reason}. A key absent from
  // the map = not in central's manifest → treat as 'absent'. Live download jobs
  // (from /api/jobs) overlay a `downloading` state with progress on top.
  const [provMap, setProvMap] = useState({})
  const [jobs, setJobs] = useState([])
  const [dlBusy, setDlBusy] = useState(() => new Set())   // model_keys with a POST in flight
  const [dlErr, setDlErr]   = useState({})                // model_key → last download error message
  const aliveRef = useRef(true)                            // guards async setState after unmount
  const prevActiveRef = useRef(new Set())                  // last poll's active (queued/running) job keys

  // Where else each model already lives (excludes THIS worker).
  const elsewhereOf = useCallback((m) => {
    const holders = allocation[keyOf(m)] || []
    return holders.filter(h => h.id !== workerId)
  }, [allocation, workerId])

  // Pull the authoritative central-readiness map. Cheap + idempotent; polled
  // slowly (a finished download only needs to flip to selectable eventually)
  // and also kicked whenever a job settles.
  const refetchProv = useCallback(async () => {
    try {
      const d = await fetchJson('/api/llm/central-provisioning')
      if (aliveRef.current) setProvMap(d && typeof d === 'object' ? d : {})
    } catch { /* readiness is advisory; a hiccup shouldn't break the picker */ }
  }, [])

  // Poll the download-job feed fast so ⏳ progress ticks live. When a job that
  // was active last poll is no longer active, a pull just settled → refresh the
  // readiness map so a completed model becomes selectable without reopening.
  const refetchJobs = useCallback(async () => {
    try {
      const d = await fetchJson('/api/jobs')
      if (!aliveRef.current) return
      const arr = Array.isArray(d) ? d : []
      setJobs(arr)
      const active = new Set(
        arr.filter(j => j && (j.status === 'queued' || j.status === 'running')).map(j => j.model_key)
      )
      let settled = false
      prevActiveRef.current.forEach(k => { if (!active.has(k)) settled = true })
      prevActiveRef.current = active
      if (settled) refetchProv()
    } catch { /* transient; next tick retries */ }
  }, [refetchProv])

  useEffect(() => {
    aliveRef.current = true
    // Self-scheduling for the same reason as the roster poll below: setInterval
    // fires whether or not the previous call returned, so a slow endpoint
    // MULTIPLIES its own load instead of backing off. /llm/jobs was measured at
    // 3-8s against this 2500ms tick — three-plus copies in flight, each holding
    // one of central's 24 slots, all of them competing with the roster poll
    // that actually renders the view. At most one of each is in flight now.
    let timers = []
    const every = (fn, ms) => {
      const tick = () => {
        Promise.resolve(fn()).finally(() => {
          if (aliveRef.current) timers.push(setTimeout(tick, ms))
        })
      }
      tick()
    }
    every(refetchJobs, 2500)    // fast: live download progress
    every(refetchProv, 10000)   // slow: readiness backstop
    return () => {
      aliveRef.current = false
      timers.forEach(clearTimeout)
    }
  }, [refetchProv, refetchJobs])

  // The active download job (queued/running) for a model, if any.
  const activeJobFor = useCallback((key) =>
    jobs.find(j => j && j.model_key === key && (j.status === 'queued' || j.status === 'running')) || null,
  [jobs])

  // Per-model provisioning state: a live download wins; else the readiness map;
  // else (key not in the manifest) → absent.
  const provStateOf = useCallback((m) => {
    const key = keyOf(m)
    const job = activeJobFor(key)
    if (job) return { state: 'downloading', job, reason: null }
    const p = provMap[key]
    if (!p) return { state: 'absent', reason: null }
    return { state: p.state || 'absent', reason: p.reason ?? null }
  }, [provMap, activeJobFor])

  // Kick (or resume) the central pull for a model, then optimistically refetch
  // jobs so the row flips to ⏳ immediately. Errors surface inline on the row.
  const download = async (key) => {
    setDlBusy(prev => new Set(prev).add(key))
    setDlErr(prev => { const n = { ...prev }; delete n[key]; return n })
    try {
      await fetchJson(`/api/models/${encodeURIComponent(key)}/download`, { method: 'POST' })
      await refetchJobs()
    } catch (e) {
      if (aliveRef.current) setDlErr(prev => ({ ...prev, [key]: e.message || `POST /api/models/${key}/download threw ${e?.name || 'an error'} with no message` }))
    } finally {
      if (aliveRef.current) setDlBusy(prev => { const n = new Set(prev); n.delete(key); return n })
    }
  }

  // Union of EVERY task across the pickable models (ModelTable's helper, so
  // multi-task models surface under each of their tasks — the same fix that
  // cured ModelTable's primary-task-only skew).
  const tasks = useMemo(() => [...new Set(models.flatMap(modelTasks))].sort(), [models])

  const COLUMNS = [
    { key: 'name', label: 'Model', get: m => m.name || keyOf(m) },
    // Central-provisioning readiness — sortable by state string. Sort is
    // point-in-time (provMap/jobs are intentionally NOT in `filtered` deps, so
    // the fast job poll doesn't reshuffle rows under the operator); the pills
    // themselves still update live because the row cells read provStateOf on
    // every render.
    { key: 'central', label: 'Central', get: m => provStateOf(m).state },
    { key: 'task', label: 'Task', get: m => modelTask(m) || '—' },
    { key: 'framework', label: 'Engine', get: m => m.framework || '—' },
    // Size sorts by true bytes; unknown (-1) sinks to the bottom in ascending sort.
    { key: 'size', label: 'Size', get: m => { const b = sizeOf(m); return b == null ? -1 : b }, num: true },
    { key: 'ctx', label: 'Ctx', get: m => m.model_max_length || 0, num: true },
    { key: 'elsewhere', label: 'Allocated on', get: m => elsewhereOf(m).map(h => h.name).join(', ') },
  ]

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    let rows = models
    if (needle) rows = rows.filter(m => (m.name || keyOf(m)).toLowerCase().includes(needle))
    // Full-list task match — a model counts under ANY task it advertises.
    if (taskFilter) rows = rows.filter(m => modelTasks(m).includes(taskFilter))
    const col = COLUMNS.find(c => c.key === sortKey) || COLUMNS[0]
    const dir = sortDir === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const va = col.get(a), vb = col.get(b)
      if (col.num) return (Number(va) - Number(vb)) * dir
      return String(va).localeCompare(String(vb)) * dir
    })
  }, [models, q, taskFilter, sortKey, sortDir, allocation, workerId])

  // Selectable only when central fully holds the model (ready) AND the
  // anti-duplicate breaker permits it. A not-ready model's checkbox is disabled;
  // its row offers a Download instead — so provisioning refusals disappear
  // before allocation ever happens.
  // A model the operator marked for archive is never selectable (central 409s it).
  const isEligible = useCallback((m) => !archiveMark(m) && provStateOf(m).state === 'ready'
    && (breaker || elsewhereOf(m).length === 0),
    [provStateOf, breaker, elsewhereOf])
  const eligibleKeys = useMemo(() => filtered.filter(isEligible).map(keyOf), [filtered, isEligible])
  const allSel = eligibleKeys.length > 0 && eligibleKeys.every(k => sel.has(k))

  const toggle = (k) => setSel(prev => { const n = new Set(prev); n.has(k) ? n.delete(k) : n.add(k); return n })
  const toggleAll = () => setSel(allSel ? new Set() : new Set(eligibleKeys))
  const toggleSort = (k) => {
    if (sortKey === k) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('asc') }
  }
  const arrow = (k) => sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''

  const allocate = async () => {
    const keys = [...sel]
    if (keys.length === 0) return
    setBusy(true)
    try { await onAllocate(keys, breaker); setSel(new Set()) }
    finally { setBusy(false) }
  }

  return (
    <div className="wp-loadtable">
      <div className="wp-loadtable-bar">
        <input className="wp-loadtable-q" placeholder="filter models…" value={q}
               onChange={e => setQ(e.target.value)} autoFocus />
        <select className="wp-loadtable-task" value={taskFilter}
                onChange={e => setTaskFilter(e.target.value)}
                title="Filter by task — matches ANY task a model advertises, not just its primary">
          <option value="">All tasks</option>
          {tasks.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <label className="wp-breaker" title="Anti-duplicate is ON by default: models already on another worker are locked. Flip this to deliberately allocate a duplicate (replicate across workers — e.g. scene fan-out).">
          <input type="checkbox" checked={breaker} onChange={e => { setBreaker(e.target.checked); setSel(new Set()) }} />
          ⚡ allow duplicate allocation
        </label>
        <button className="wp-load-cancel" title="Cancel" onClick={onCancel}>×</button>
      </div>
      <div className="wp-loadtable-scroll">
        <table className="wp-loadtable-t">
          <thead>
            <tr>
              <th className="wp-lt-check">
                <input type="checkbox" checked={allSel} onChange={toggleAll}
                       disabled={eligibleKeys.length === 0} title="Select all eligible" />
              </th>
              {COLUMNS.map(c => (
                <th key={c.key} className={`wp-lt-sortable${c.num ? ' wp-lt-num' : ''}`}
                    onClick={() => toggleSort(c.key)}>{c.label}{arrow(c.key)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={COLUMNS.length + 1} className="wp-none">No models to allocate.</td></tr>
            )}
            {filtered.map(m => {
              const k = keyOf(m)
              const elsewhere = elsewhereOf(m)
              const locked = elsewhere.length > 0 && !breaker
              const prov = provStateOf(m)
              const arch = archiveMark(m)
              return (
                // Only the anti-duplicate lock (and an archive mark) grays the
                // row; not-ready rows stay legible so their Download button is usable.
                <tr key={k} className={arch ? 'wp-lt-locked wp-lt-archived' : locked ? 'wp-lt-locked' : ''}
                    title={arch ? archiveText(arch) : undefined}>
                  <td className="wp-lt-check">
                    <input type="checkbox" checked={sel.has(k)} disabled={!isEligible(m)}
                           title={arch ? archiveText(arch) : undefined}
                           onChange={() => toggle(k)} />
                  </td>
                  <td>{m.name || k}{arch && <div className="wp-lt-archive-note">🗄 {archiveText(arch)}</div>}</td>
                  {/* Central-provisioning readiness pill (+ Download when the
                      model needs pulling; ⏳ progress while a pull runs). This is
                      what makes post-hoc "central doesn't have X on disk"
                      refusals visible up front. */}
                  <td className="wp-cprov-cell">
                    {prov.state === 'ready' && (
                      <span className="wp-state-pill wp-cprov-ready"
                            title="fully on central disk — allocatable">✓ on disk</span>
                    )}
                    {prov.state === 'downloading' && (() => {
                      const job = prov.job || {}
                      const pct = (job.total_bytes && job.total_bytes > 0 && job.progress != null)
                        ? Math.round(job.progress * 100) : null
                      return (
                        <span className="wp-state-pill wp-cprov-downloading"
                              title={job.total_bytes
                                ? `downloading to central — ${fmtBytes(job.downloaded_bytes)} / ${fmtBytes(job.total_bytes)}`
                                : 'downloading to central…'}>
                          ⏳ {pct != null ? `${pct}%` : '…'}
                        </span>
                      )
                    })()}
                    {prov.state === 'incomplete' && (
                      <>
                        <span className="wp-state-pill wp-cprov-incomplete"
                              title="directory present but files incomplete — download to finish">◐ incomplete</span>
                        <button className="wp-cprov-dl" disabled={dlBusy.has(k)} onClick={() => download(k)}
                                title="Finish downloading this model to central disk">
                          {dlBusy.has(k) ? '…' : '⬇ Download'}
                        </button>
                      </>
                    )}
                    {prov.state === 'absent' && (
                      <>
                        <span className="wp-state-pill wp-cprov-absent"
                              title="not on central disk / not in the manifest">○ not on central</span>
                        <button className="wp-cprov-dl" disabled={dlBusy.has(k)} onClick={() => download(k)}
                                title="Download this model to central disk">
                          {dlBusy.has(k) ? '…' : '⬇ Download'}
                        </button>
                      </>
                    )}
                    {prov.state === 'error' && (
                      <span className="wp-state-pill wp-cprov-error"
                            title={prov.reason || `provisioning map row for ${k} has state=error and no reason field`}>⚠ error</span>
                    )}
                    {dlErr[k] && <span className="wp-cprov-err" title={dlErr[k]}>failed</span>}
                  </td>
                  {/* first task + "+N" with the full list on hover — same
                      rendering ModelTable uses for multi-task models */}
                  <td className="wp-lt-muted" title={modelTasks(m).join(', ')}>
                    {modelTask(m)}{modelTasks(m).length > 1 ? ` +${modelTasks(m).length - 1}` : ''}
                  </td>
                  <td className="wp-lt-muted">{m.framework || '—'}</td>
                  <td className="wp-lt-num wp-lt-size"
                      title={sizeOf(m) == null ? sizeView(m, fmtBytes).title
                        : isGgufModel(m)
                          ? `effective quant${m.effective_gguf ? ` ${m.effective_gguf}` : ''} — ${fmtBytes(sizeOf(m))} (the ONE quant that serves, not the all-quants dir sum)`
                          : `${fmtBytes(sizeOf(m))} on disk`}>
                    {sizeOf(m) == null ? <em className="wp-lt-size-why">{sizeView(m, fmtBytes).text}</em> : fmtBytes(sizeOf(m))}
                  </td>
                  <td className="wp-lt-num">{m.model_max_length || '—'}</td>
                  <td className={elsewhere.length ? (breaker ? 'wp-lt-dup' : 'wp-lt-muted') : 'wp-lt-muted'}>
                    {elsewhere.length ? `${breaker ? '⚡ ' : ''}${elsewhere.map(h => h.name).join(', ')}` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {/* Measured headroom on THIS worker — the honest free space the fit-guard
          checks each pick against. VRAM free is nvidia-smi; RAM free is the
          worker's measured free (reserve-adjusted). "resident" is the SUM of the
          worker's MEASURED per-model VRAM allocations, not file sizes or envelope
          guesses. Per-model RAM RSS for in-process models isn't attributable yet
          (glibc-arena + page-cache gap, see CODE_GAPS), so RAM is worker-level. */}
      {worker && (worker.vram_total != null || worker.ram_total != null) && (() => {
        const allocs2 = Array.isArray(worker.allocations) ? worker.allocations.filter(a => a && a.model_key) : []
        const residentVram = allocs2.reduce((s, a) => s + (a.vram_bytes != null && a.vram_bytes > 0 ? a.vram_bytes : 0), 0)
        const selBytes = [...sel].reduce((s, k) => {
          const m = findCatalogRow(models, k)
          return s + (m ? (sizeOf(m) || 0) : 0)
        }, 0)
        return (
          <div className="wp-load-headroom"
               title="Measured headroom on this worker — what a model is fit-checked against before it loads. VRAM free is nvidia-smi; RAM free is the worker's measured free (reserve-adjusted). 'resident' sums measured per-model VRAM, never file sizes. Per-model RAM RSS for in-process models is not attributable yet (page-cache/arena gap), so RAM is shown at the worker level.">
            <span className="wp-load-headroom-label">Headroom (measured):</span>
            {worker.vram_total != null && (
              <span className="wp-load-headroom-item" title="Free VRAM from nvidia-smi (measured)">
                🖥 {fmtBytes(worker.vram_free)} free / {fmtBytes(worker.vram_total)} VRAM
                {residentVram > 0 && <em> · {fmtBytes(residentVram)} resident</em>}
              </span>
            )}
            {worker.ram_total != null && (
              <span className="wp-load-headroom-item" title="Worker's measured free RAM (reserve-adjusted). Per-model RAM attribution is not available yet.">
                🧠 {fmtBytes(worker.free_ram)} free / {fmtBytes(worker.ram_total)} RAM
              </span>
            )}
            {sel.size > 0 && selBytes > 0 && (
              <span className="wp-load-headroom-sel" title="Total on-disk size of the checked models (effective quant for GGUF). Resident memory use is typically at or below this; the fit-guard makes the final call per model.">
                selected {sel.size}: {fmtBytes(selBytes)} on disk
              </span>
            )}
          </div>
        )
      })()}
      <div className="wp-loadtable-actions">
        <span className="wp-load-hint" title="Each model is checked against this worker's free VRAM + RAM + disk before it loads — a model that won't fit is refused, not OOM'd">✓ fit-guarded</span>
        <button className="wp-loadtable-go" disabled={busy || sel.size === 0} onClick={allocate}>
          {busy ? 'Allocating…' : `Allocate ${sel.size} model${sel.size === 1 ? '' : 's'} to this worker`}
        </button>
      </div>
    </div>
  )
}
