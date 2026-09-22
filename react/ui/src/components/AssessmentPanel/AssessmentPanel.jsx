import { useEffect, useState, useMemo, useCallback } from 'react'
import { fetchJson } from '../../api'
import { useFeed } from '../../runtime/feeds'
import './AssessmentPanel.css'

// ── formatters ───────────────────────────────────────────────────────────────
function fmtBytes(n) {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}
function fmtParams(n) {
  if (!n) return null
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`
  return String(n)
}

// Filters mirror the category tags so the chips read 1:1 with the row badges.
const FILTERS = [
  { id: 'all',    label: 'All' },
  { id: 'media',  label: 'Media' },
  { id: 'worker', label: 'Worker' },
  { id: 'slot',   label: 'Slot' },
  { id: 'api',    label: 'API' },
  { id: 'active', label: 'Active' },
]

// Column model — drives the header, the cells, sorting, and resizing.
// `w` is the default width (px); `sort` maps a row to a comparable value.
const CATS_ORDER = ['media', 'worker', 'slot', 'api', 'processing']
const COLS = [
  { id: 'name',    label: 'Model',      w: 200, cls: 'as-c-name', sort: r => r.name.toLowerCase() },
  { id: 'loc',     label: 'Location',   w: 120, cls: 'as-c-loc',  sort: r => r.loc || '' },
  { id: 'pending', label: 'Pending',    w: 78,  cls: 'as-c-m', num: true, sort: r => r.qWaiting },
  { id: 'active',  label: 'Active',     w: 78,  cls: 'as-c-m', num: true, sort: r => r.qActive },
  { id: 'vram',    label: 'VRAM',       w: 90,  cls: 'as-c-m', num: true, sort: r => r.vram ?? -1 },
  { id: 'ram',     label: 'RAM',        w: 90,  cls: 'as-c-m', num: true, sort: r => r.ram ?? -1 },
  { id: 'cores',   label: 'Cores',      w: 78,  cls: 'as-c-m', num: true, sort: r => r.cores ?? -1 },
  // Categories pushed to the far right.
  { id: 'cats',    label: 'Categories', w: 220, cls: 'as-c-cats', sort: r => CATS_ORDER.filter(c => r.cats[c]).length },
]

function cellContent(id, r, isOpen) {
  switch (id) {
    case 'name':    return <><span className="as-caret">{isOpen ? '▾' : '▸'}</span>{r.name}</>
    case 'cats':    return <Tags r={r} />
    case 'loc':     return r.loc || '—'
    case 'pending': return r.qWaiting || '·'
    case 'active':  return r.qActive || '·'
    case 'vram':    return r.vram != null ? fmtBytes(r.vram) : '—'
    case 'ram':     return r.ram != null ? fmtBytes(r.ram) : '—'
    case 'cores':   return r.cores != null ? `${r.cores} thr` : '—'
    default:        return null
  }
}

// Cross-cutting fleet assessment: every model that is in media / allocated to a
// worker / serving in a slot / exposed to the API / actively processing, with
// the live numbers behind each. `models` + `workers` are lifted from the parent
// (already polled); slots / queue / serving / api-models are polled here while
// this tab is mounted.
//
// Every figure is a CUMULATIVE total of what is currently known: a model can
// occupy several slots/workers at once, so its VRAM/RAM/cores sum across all of
// them. The Pending / Active counts are the inference processing queue (the same
// /api/llm/queue the topbar ActivityQueue chip shows as "active · queued").
export default function AssessmentPanel({ models = [], workers = [] }) {
  const [slots, setSlots]     = useState({ enabled: null, slots: [], resources: null })
  const [queue, setQueue]     = useState({ active: [], counts: {} })
  const [serving, setServing] = useState([])
  const [apiModels, setApi]   = useState(new Map())   // model_key -> /api/v1/models entry
  const [filter, setFilter]   = useState('all')
  const [open, setOpen]       = useState({})          // model_key -> expanded?
  const [sort, setSort]       = useState({ col: 'loc', dir: -1 })   // default: Location descending (no-location rows sink to the bottom)
  const [widths, setWidths]   = useState(() => COLS.map(c => c.w))

  // slots / queue / serving come from the one live subscription (runtime/feeds.js);
  // only the /v1/models catalog view is still fetched here, once a minute.
  const fSlots = useFeed('slots', null)
  const fQueue = useFeed('queue', null)
  const fServing = useFeed('serving', null)
  useEffect(() => { if (fSlots) setSlots(fSlots) }, [fSlots])
  useEffect(() => { if (fQueue) setQueue(fQueue) }, [fQueue])
  useEffect(() => { if (Array.isArray(fServing)) setServing(fServing) }, [fServing])
  useEffect(() => {
    let alive = true
    const load = () => {
      fetchJson('/api/v1/models').then(d => alive && setApi(new Map((d?.data || []).map(m => [m.id, m])))).catch(() => {})
    }
    load()
    const t = setInterval(load, 60_000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const toggle = useCallback(key => setOpen(o => ({ ...o, [key]: !o[key] })), [])

  // Per-model assessment, then filtered to models that are in ≥1 category.
  const rows = useMemo(() => {
    const active = queue.active || []
    const slotsByModel = {}
    for (const s of (slots.slots || [])) if (s.model_key) (slotsByModel[s.model_key] ||= []).push(s)

    // Universe of models: the local registry PLUS any key referenced live by a
    // worker / slot / queue / the API — a worker can serve a model central
    // doesn't have in its own registry, and it must still show up here.
    const byKey = new Map()
    for (const m of models) byKey.set(m.model_key ?? m.key, m)
    const ensure = k => { if (k && !byKey.has(k)) byKey.set(k, { model_key: k, name: k, status: 'remote' }) }
    for (const w of workers) {
      (w.models || []).forEach(ensure)
      ;(w.loaded_models || []).forEach(ensure)
      ;(w.provisioning || []).forEach(ensure)
    }
    for (const s of (slots.slots || [])) ensure(s.model_key)
    for (const r of active) ensure(r.model_key)
    for (const k of apiModels.keys()) ensure(k)

    return [...byKey.values()].map(m => {
      const key = m.model_key ?? m.key
      const assignedOn  = workers.filter(w => (w.models || []).includes(key))
      const loadedOn    = workers.filter(w => (w.loaded_models || []).includes(key))
      const pendingOn   = workers.filter(w => (w.provisioning || []).includes(key))
      const mSlots      = slotsByModel[key] || []
      const slotServing = mSlots.some(s => s.healthy)
      const slotLoading = mSlots.some(s => s.model_key && !s.healthy)
      const apiEntry    = apiModels.get(key) || null

      const qWaiting = active.filter(r => r.model_key === key && r.state === 'waiting')
      const qActive  = active.filter(r => r.model_key === key && r.state === 'active')

      const cats = {
        media:      !!m.media,
        worker:     assignedOn.length > 0 || loadedOn.length > 0 || pendingOn.length > 0,
        loaded:     loadedOn.length > 0,
        slot:       mSlots.length > 0,
        serving:    slotServing,
        api:        !!apiEntry,
        processing: qActive.length > 0,
      }
      const loading = pendingOn.length > 0 || slotLoading

      // Cumulative "what is known": sum every slot footprint + every worker GPU
      // budget for VRAM; sum every slot RSS for RAM; sum every slot's threads for
      // cores. A model can live in several places at once — total them all.
      let vram = 0, vramKnown = false
      for (const s of mSlots) if (s.expected_bytes != null) { vram += s.expected_bytes; vramKnown = true }
      for (const w of assignedOn) {
        const gib = w.spill_by_model?.[key]?.gpu_mem_gib
        if (gib != null) { vram += gib * 1073741824; vramKnown = true }
      }
      let ram = 0, ramKnown = false
      // Prefer the honest pinned figure (rss_anon) — VmRSS counts the mmap'd
      // GGUF's file-backed pages (reclaimable cache) and overstates RAM ~28x.
      for (const s of mSlots) {
        const r = s.rss_anon_bytes ?? s.rss_bytes
        if (r) { ram += r; ramKnown = true }
      }
      // Slots aren't the only place a model holds RAM: an in-process (kind:'ram')
      // model on a worker reports MEASURED ram_resident_bytes (next-release
      // field). Counting slots only made those models read as holding nothing.
      // Absent on older workers → that worker simply contributes nothing here.
      for (const w of workers) {
        for (const a of (w.allocations || [])) {
          // kind:'slot' rows are the SAME processes mSlots already summed —
          // only in-process rows are additional, so never double-count.
          if (!a || a.model_key !== key || a.kind !== 'ram') continue
          const r = a.rss_anon_bytes ?? a.ram_resident_bytes
          if (r) { ram += r; ramKnown = true }
        }
      }
      let cores = 0, coresKnown = false
      for (const s of mSlots) if (s.threads != null) { cores += s.threads; coresKnown = true }

      const workersForModel = [...new Set([...loadedOn, ...pendingOn, ...assignedOn])]
      const locParts = [...mSlots.map(s => `slot ${s.slot_id}`), ...workersForModel.map(w => w.name)]
      const loc = locParts.length ? locParts[0] + (locParts.length > 1 ? ` +${locParts.length - 1}` : '') : null

      return {
        key, name: m.name || key, model: m, cats, loading,
        qWaiting: qWaiting.length, qActive: qActive.length, qActiveRows: qActive,
        vram: vramKnown ? vram : null,
        ram: ramKnown ? ram : null,
        cores: coresKnown ? cores : null,
        loc, mSlots, assignedOn, loadedOn, pendingOn, workersForModel, apiEntry,
      }
    }).filter(r => r.cats.media || r.cats.worker || r.cats.slot || r.cats.api || r.cats.processing)
  }, [models, workers, slots, queue, apiModels])

  const filtered = useMemo(() => rows.filter(r =>
    filter === 'all' ? true
    : filter === 'active' ? r.cats.processing
    : r.cats[filter]
  ), [rows, filter])

  const sorted = useMemo(() => {
    if (!sort.col) return filtered
    const col = COLS.find(c => c.id === sort.col)
    if (!col) return filtered
    return [...filtered].sort((a, b) => {
      const va = col.sort(a), vb = col.sort(b)
      if (va < vb) return -sort.dir
      if (va > vb) return sort.dir
      return 0
    })
  }, [filtered, sort])

  const onSort = useCallback(id =>
    setSort(s => s.col === id ? { col: id, dir: -s.dir } : { col: id, dir: 1 }), [])

  // Drag a header's right edge to resize that column; updates live, min 48px.
  const startResize = useCallback((i, e) => {
    e.preventDefault(); e.stopPropagation()
    const startX = e.clientX
    const startW = widths[i]
    const onMove = ev => setWidths(w => {
      const n = [...w]; n[i] = Math.max(48, startW + (ev.clientX - startX)); return n
    })
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    document.body.style.cursor = 'col-resize'
  }, [widths])

  const template = widths.map(w => `${w}px`).join(' ')

  return (
    <div className="as-panel">
      {/* Totals live in the persistent StatusBar above the tabs. */}
      {/* ── Filter chips ─────────────────────────────────────────────────── */}
      <div className="as-filters">
        {FILTERS.map(f => (
          <button key={f.id}
                  className={`as-chip${filter === f.id ? ' as-chip-on' : ''}`}
                  onClick={() => setFilter(f.id)}>
            {f.label}
            {f.id !== 'all' && (
              <span className="as-chip-n">
                {rows.filter(r => f.id === 'active' ? r.cats.processing : r.cats[f.id]).length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Model list — sortable + user-resizable columns ───────────────── */}
      <div className="as-list" style={{ '--as-cols': template }}>
        <div className="as-row as-head">
          {COLS.map((c, i) => (
            <div key={c.id}
                 className={`as-hcell ${c.cls}${c.num ? ' as-num' : ''}${sort.col === c.id ? ' as-sorted' : ''}`}
                 onClick={() => onSort(c.id)}
                 title="Click to sort · drag the edge to resize">
              <span className="as-hlabel">{c.label}</span>
              <span className="as-sort-ind">{sort.col === c.id ? (sort.dir > 0 ? '▲' : '▼') : ''}</span>
              {i < COLS.length - 1 && (
                <span className="as-resize"
                      onMouseDown={e => startResize(i, e)}
                      onClick={e => e.stopPropagation()} />
              )}
            </div>
          ))}
        </div>

        {sorted.length === 0 && (
          <div className="as-empty">No models match — nothing is in media, on a worker, in a slot, on the API, or processing.</div>
        )}

        {sorted.map(r => (
          <div key={r.key} className={`as-item${open[r.key] ? ' as-item-open' : ''}`}>
            <button className="as-row as-data" onClick={() => toggle(r.key)}>
              {COLS.map(c => (
                <span key={c.id}
                      className={`${c.cls}${c.id === 'pending' && r.qWaiting ? ' as-m-warn' : ''}${c.id === 'active' && r.qActive ? ' as-m-live' : ''}`}>
                  {cellContent(c.id, r, open[r.key])}
                </span>
              ))}
            </button>
            {open[r.key] && <Detail r={r} serving={serving} />}
          </div>
        ))}
      </div>
    </div>
  )
}

function Tags({ r }) {
  const t = []
  if (r.cats.media) t.push(['media', 'media', '🎬'])
  if (r.cats.slot) t.push(['slot', r.cats.serving ? 'slot' : 'slot·loading', '🧩'])
  if (r.cats.worker) t.push(['worker',
    r.cats.loaded ? 'loaded' : (r.loading ? 'loading' : 'worker'),
    r.cats.loaded ? '🔥' : (r.loading ? '⏳' : '○')])
  if (r.cats.api) t.push(['api', 'API', '🔌'])
  if (r.cats.processing) t.push(['proc', 'active', '⚡'])
  return <>{t.map(([cls, label, ico]) => <span key={cls + label} className={`as-tag as-tag-${cls}`}>{ico} {label}</span>)}</>
}

function Detail({ r, serving }) {
  const sv = serving.find(x => x.key === r.key)
  return (
    <div className="as-detail">
      <Row k="Model">
        {fmtParams(r.model.parameter_count) && <Field label="params">{fmtParams(r.model.parameter_count)}</Field>}
        {r.model.framework && <Field label="framework">{r.model.framework}</Field>}
        {r.model.primary_task && <Field label="task">{r.model.primary_task}</Field>}
        {r.model.total_bytes && <Field label="on disk">{fmtBytes(r.model.total_bytes)}</Field>}
        <Field label="status">{r.model.status}</Field>
        {r.cats.media && <Field label="media">selected</Field>}
      </Row>

      <Row k="Queue">
        <Field label="pending">{r.qWaiting}</Field>
        <Field label="active">{r.qActive}</Field>
        {r.qActiveRows.map(q => (
          <Field key={q.request_id} label={q.kind || 'req'}>
            {q.tokens != null ? `${q.tokens} tok` : ''} {q.elapsed != null ? `· ${q.elapsed}s` : ''}
          </Field>
        ))}
        {!r.qWaiting && !r.qActive && <span className="as-dim">idle — no in-flight requests</span>}
      </Row>

      {r.workersForModel.length > 0 && (
        <Row k="Workers">
          {r.workersForModel.map(w => {
            const loaded = (w.loaded_models || []).includes(r.key)
            const pend = (w.provisioning || []).includes(r.key)
            const gib = w.spill_by_model?.[r.key]?.gpu_mem_gib
            // MEASURED in-process host-RAM occupancy for this model on this
            // worker (next-release field; absent → nothing is claimed).
            const alloc = (w.allocations || []).find(
              a => a && a.model_key === r.key && a.kind === 'ram')
            const rres = alloc ? (alloc.rss_anon_bytes ?? alloc.ram_resident_bytes) : null
            return (
              <Field key={w.id} label={w.name}
                     title={rres != null ? 'measured resident host RAM (worker smaps read)' : undefined}>
                {pend ? '⏳ loading' : loaded ? '🔥 loaded' : '○ assigned'}
                {rres != null ? ` · ${fmtBytes(rres)} RAM resident` : ''}
                {gib != null ? ` · ${gib} GiB GPU budget` : ''}
                {` · ${w.gpus?.[0]?.name || (w.gpus?.length ? 'GPU' : 'CPU')}`}
              </Field>
            )
          })}
        </Row>
      )}

      {r.mSlots.map(s => (
        <Row key={s.slot_id} k={`Slot ${s.slot_id}`}>
          <Field label="state">{s.healthy ? 'serving' : (s.model_key ? 'loading' : 'idle')}</Field>
          {s.expected_bytes != null && <Field label="VRAM">{fmtBytes(s.expected_bytes)}</Field>}
          {s.rss_anon_bytes != null
            ? <Field label="RAM">{fmtBytes(s.rss_anon_bytes)}{s.rss_file_bytes > 0 ? ` + ${fmtBytes(s.rss_file_bytes)} cache` : ''}</Field>
            : (s.rss_bytes
                ? <Field label="RAM (VmRSS)"
                         title="VmRSS — includes mmap'd file pages, overstates pinned RAM">
                    ~{fmtBytes(s.rss_bytes)}
                  </Field>
                : null)}
          {s.n_gpu_layers != null && (
            <Field label="GPU layers">{String(s.n_gpu_layers)}{s.total_layers != null ? `/${s.total_layers}` : ''}</Field>
          )}
          {s.threads != null && <Field label="threads">{s.threads}</Field>}
          {(s.allowed_cpus || s.cpus) && <Field label="cores">{s.allowed_cpus || s.cpus}</Field>}
          {s.gpu != null && s.gpu !== '' && <Field label="GPU">{String(s.gpu)}</Field>}
          {s.ctx != null && <Field label="context">{s.ctx}</Field>}
          {s.free_vram_bytes != null && <Field label="VRAM free">{fmtBytes(s.free_vram_bytes)}</Field>}
        </Row>
      ))}

      {(r.cats.api || sv) && (
        <Row k="API">
          {r.apiEntry && <Field label="exposed">/api/v1/models</Field>}
          {r.apiEntry?.context_length && <Field label="context">{r.apiEntry.context_length}</Field>}
          {sv && <Field label="serve mode">{sv.always_on ? 'always-on' : (sv.mode || 'swap')}</Field>}
          {sv?.endpoint && <Field label="endpoint">{sv.endpoint}</Field>}
          {sv?.ttl_seconds && <Field label="ttl">{sv.ttl_seconds}s</Field>}
        </Row>
      )}
    </div>
  )
}

function Row({ k, children }) {
  return (
    <div className="as-drow">
      <span className="as-dk">{k}</span>
      <div className="as-dv">{children}</div>
    </div>
  )
}
function Field({ label, title, children }) {
  return <span className="as-field" title={title}><span className="as-field-k">{label}</span>{children}</span>
}
