import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { fetchJson } from '../../api'
import useSessionState from '../../hooks/useSessionState'
import DownloadsQueue from '../DownloadsQueue'
import './HFSearch.css'

// The full Hugging Face pipeline-tag taxonomy, grouped by modality so the (long)
// list stays navigable. Any tag here is a valid `pipeline_tag` filter the Hub
// search accepts, so the dropdown now covers everything the Hub can return —
// not just the handful hugpy ships models for.
const TASK_GROUPS = [
  ['Multimodal', [
    'image-text-to-text', 'visual-question-answering', 'document-question-answering',
    'video-text-to-text', 'image-to-text', 'any-to-any',
  ]],
  ['Natural Language Processing', [
    'text-generation', 'text2text-generation', 'summarization', 'translation',
    'text-classification', 'token-classification', 'fill-mask', 'question-answering',
    'table-question-answering', 'zero-shot-classification', 'sentence-similarity',
    'feature-extraction',
  ]],
  ['Computer Vision', [
    'image-classification', 'object-detection', 'image-segmentation', 'depth-estimation',
    'zero-shot-image-classification', 'zero-shot-object-detection', 'image-feature-extraction',
    'keypoint-detection', 'mask-generation', 'image-to-image', 'image-to-video',
    'text-to-image', 'unconditional-image-generation', 'video-classification',
    'text-to-3d', 'image-to-3d',
  ]],
  ['Audio', [
    'automatic-speech-recognition', 'text-to-speech', 'text-to-audio',
    'audio-classification', 'audio-to-audio', 'voice-activity-detection',
  ]],
  ['Tabular / Time Series', [
    'tabular-classification', 'tabular-regression', 'time-series-forecasting',
  ]],
  ['Reinforcement Learning & Other', [
    'reinforcement-learning', 'robotics', 'graph-ml',
  ]],
]
// Flat list retained for any callers that want a membership check.
const TASK_FILTERS = TASK_GROUPS.flatMap(([, tasks]) => tasks)

const LIBRARIES = ['transformers', 'gguf', 'diffusers', 'sentence-transformers', 'timm']

// Browse the whole Hub by default: no task filter, no query, no library. The
// panel opens on an unfiltered listing of Hugging Face and pages down through it
// (see PAGE_SIZE / "Load more"), so narrowing is an opt-in the operator makes,
// not a wall they start behind.
const DEFAULT_TASK = ''

// One request's worth of rows. Each page is a single Hub listing call plus the
// (permanently cached) per-repo size lookups the server fans out.
const PAGE_SIZE = 40

function inferFramework(result) {
  // HF-canonical vocabulary: "gguf" is HF Hub's library tag for GGUF repos
  // ("llama_cpp" retired as a framework value, 2026-07-05).
  const tags = (result.tags || []).map(t => String(t).toLowerCase())
  if (tags.some(t => t.includes('gguf'))) return 'gguf'
  if (result.library_name === 'gguf') return 'gguf'
  // Report the repo's ACTUAL library rather than assuming transformers. That
  // assumption was harmless while the panel only ever listed text-generation;
  // browsing the whole Hub surfaces keras/timm/diffusers/untagged repos, and
  // labelling those "transformers" would be a plain lie in the Lib column.
  return result.library_name || '–'
}

function inferTask(result) {
  // Untagged is its own answer. Most of the Hub carries no pipeline_tag, and
  // defaulting those to "text-generation" mislabels them now that "Any task"
  // is what the panel opens on.
  return result.pipeline_tag || '–'
}

function fmtCount(n) {
  if (n == null) return '–'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

function fmtBytes(n) {
  if (n == null) return '–'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// ── Automatic quant selection by free space (GGUF multi-select) ──────────────
// Given the spec `options` list and a free-space budget in bytes, return a Set
// of option ids to check — the largest useful ladder of quants that still fits.
// Greedy largest-first: the biggest quants carry the most quality, so they claim
// the budget first and smaller ones fill the remainder. The budget is the
// server's `free_bytes` (same MODELS_DIR headroom the per-option fits_disk
// flags were computed against), taken at 90% so a fit today isn't a full disk
// tomorrow. Returns null when the budget is unknown — caller keeps the current
// selection.
function autoSelectBySpace(options, freeBytes) {
  if (!Number.isFinite(freeBytes) || freeBytes <= 0) return null
  const candidates = (options ?? [])
    .filter(o => o.fits_disk !== false && o.total_bytes != null)
    .sort((a, b) => b.total_bytes - a.total_bytes)
  const picked = new Set()
  let budget = freeBytes * 0.9
  for (const o of candidates) {
    if (o.total_bytes <= budget) { picked.add(o.id); budget -= o.total_bytes }
  }
  return picked
}

// ── Worker-compatibility fit (operator ask 2026-09-10) ──────────────────────
// "Only show the models and quants that will fit on the selected worker."
// A model FITS a worker when it clears BOTH ceilings the fleet actually
// enforces: the SERVE ceiling (VRAM + RAM — the most a load can ever occupy,
// MoE/spill included) and the LAND ceiling (the worker's effective storage
// budget; eviction frees the whole budget, so headroom-now is not the bound).
// Unknown terms are skipped; nothing known -> null -> the filter is inert.
function workerFitBytes(w) {
  const vram = w?.vram_total ??
    (Array.isArray(w?.gpus) ? w.gpus.reduce((s, g) => s + (g?.memory_total ?? 0), 0) : 0)
  const ram = w?.ram_total ?? 0
  const serve = (vram || ram) ? vram + ram : null
  const st = w?.storage ?? {}
  const land = st.budget_effective_bytes ?? st.disk_free ?? null
  const known = [serve, land].filter(v => Number.isFinite(v) && v > 0)
  return known.length ? Math.min(...known) : null
}

function DownloadBar({ job, onCancel, onRetry }) {
  if (!job) return null
  const pct = Math.round((job.progress ?? 0) * 100)
  const indeterminate = job.status === 'running' && !job.total_bytes
  const active = job.status === 'running' || job.status === 'queued'
  const retryable = job.status === 'failed' || job.status === 'cancelled' ||
                    job.status === 'expired'
  const bps = job.bytes_per_second
  return (
    <div className="dl-bar-wrap">
      <div className={`dl-bar ${indeterminate ? 'dl-bar-indet' : ''} ${job.stalled ? 'dl-bar-stalled' : ''} dl-bar-${job.status}`}>
        <div className="dl-bar-fill" style={{ width: indeterminate ? '40%' : `${pct}%` }} />
      </div>
      <span className="dl-bar-label" title={job.error || job.message || ''}>
        {job.status === 'queued'    && (job.message || 'queued…')}
        {job.status === 'running'   && (job.stalled
          ? '⚠ stalled — resuming…'
          : indeterminate
            ? `downloading… ${fmtBytes(job.downloaded_bytes)}`
            : `${pct}% · ${fmtBytes(job.downloaded_bytes)} / ${fmtBytes(job.total_bytes)}`)}
        {job.status === 'running' && job.attempt > 1 && ` · try ${job.attempt}/${job.max_attempts}`}
        {job.status === 'running' && !job.stalled && bps > 0 && ` · ${fmtBytes(bps)}/s`}
        {job.status === 'completed' && '✓ installed'}
        {job.status === 'failed'    && `✗ ${job.error_reason ? `[${job.error_reason}] ` : ''}${job.error ?? 'failed'}`}
        {job.status === 'expired'   && `✗ expired — ${job.message || 'never ran'}`}
        {job.status === 'cancelled' && 'cancelled'}
      </span>
      {active && (
        <button className="dl-cancel" onClick={() => onCancel(job.id)} title="Cancel download">
          ✕ cancel
        </button>
      )}
      {retryable && onRetry && (
        <button className="dl-retry" onClick={() => onRetry(job.id)}
                title="Resume from where it stopped">
          ↻ retry
        </button>
      )}
    </div>
  )
}

const DL_ACTIVE   = new Set(['queued', 'running'])
// k121: `expired` included — the stalled-queue outcome must render, not vanish.
const DL_TERMINAL = new Set(['completed', 'failed', 'cancelled', 'expired'])

// Consolidated download list — every repo pull started from HF search, gathered
// in one place instead of hiding inside each expanded result row. Sourced from
// the app-wide jobsByHub map (keyed by repo), so it shows exactly what the
// per-row DownloadBars show. Active pulls (queued/running) list first; finished
// ones fall to a muted "recent" tail. The status-bar DownloadsQueue is the
// global, always-on twin of this; this one lives with the search that spawns them.
function DownloadsPanel({ jobsByHub, onCancelJob, onRetryJob }) {
  const [open, setOpen] = useState(true)
  const jobs = Object.values(jobsByHub ?? {})
  if (jobs.length === 0) return null
  const active = jobs.filter(j => DL_ACTIVE.has(j.status))
  const recent = jobs.filter(j => DL_TERMINAL.has(j.status))
  return (
    <div className="hf-downloads">
      <button type="button" className="hf-downloads-head" onClick={() => setOpen(o => !o)}
              title={open ? 'Collapse downloads' : 'Show current downloads'}>
        <span className="section-caret">{open ? '▾' : '▸'}</span>
        <span className="hf-downloads-title">Downloads</span>
        {active.length > 0
          ? <span className="hf-downloads-count hf-downloads-count-active">{active.length} downloading</span>
          : <span className="hf-downloads-count hf-downloads-count-idle">idle</span>}
        {recent.length > 0 && <span className="hf-downloads-count hf-downloads-recent">{recent.length} recent</span>}
      </button>
      {open && (
        <div className="hf-downloads-list">
          {[...active, ...recent].map(j => (
            <div key={j.id} className="hf-downloads-row">
              <a className="hf-downloads-name"
                 href={`https://huggingface.co/${j.hub_id}`} target="_blank" rel="noreferrer"
                 title={j.hub_id || j.model_key}>
                {j.hub_id || j.model_key || 'download'}
              </a>
              <DownloadBar job={j} onCancel={onCancelJob} onRetry={onRetryJob} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// An enqueue that returns 200 is NOT the same as an enqueue that landed. The
// API now answers {queued, queue_position, downloader_running, queue_error};
// before k119 the console showed a cheerful "queued" row for a job no daemon
// could ever see, and it just spun until the sweep expired it 30 minutes later.
// Surface the refusal at the moment the operator clicked.
function queueWarning(res) {
  if (!res) return null
  if (res.queue_error) return res.queue_error
  if (res.queued === false) return 'The download queue did not accept this job.'
  return null
}

function ResultRow({ r, fw, tk, onDisk, externallyQueued, job, onJobStarted, onCancelJob, onRetryJob, workerFit }) {
  const [open, setOpen]       = useState(false)
  const [data, setData]       = useState(null)
  const [choice, setChoice]   = useState(null)          // single-select id (non-GGUF path)
  const [choices, setChoices] = useState(() => new Set()) // multi-select ids (GGUF path)
  const [loading, setLoading] = useState(false)
  const [adding, setAdding]   = useState(false)
  // k121: an enqueue refusal stays ON the row until dismissed — the old
  // window.alert() was one-shot and unrecoverable once clicked away.
  const [warnMsg, setWarnMsg] = useState(null)

  const expand = useCallback(async () => {
    const next = !open
    setOpen(next)
    if (!next || data || loading) return
    setLoading(true)
    try {
      const d = await fetchJson(`/api/hf/spec?hub_id=${encodeURIComponent(r.hub_id)}`)
      setData(d)
      const recommended = d?.options?.recommended ?? null
      setChoice(recommended)
      // GGUF multi-select starts pre-seeded with the recommended quant.
      setChoices(new Set(recommended != null ? [recommended] : []))
    } catch (e) {
      setData({ error: e.message })
    } finally {
      setLoading(false)
    }
  }, [open, data, loading, r.hub_id])

  const addToLocal = useCallback(async () => {
    const options = data?.options
    if (!options) return
    const opt = options.options.find(o => o.id === choice)
    if (!opt) return
    setAdding(true)
    try {
      const res = await fetchJson('/api/llm/repos/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hub_id: r.hub_id, framework: opt.framework, task: options.task,
          filename: opt.filename ?? null, include: opt.include ?? null,
          total_bytes: opt.total_bytes ?? null, register: true,
        }),
      })
      onJobStarted?.({ ...res, hub_id: r.hub_id })
      setWarnMsg(queueWarning(res))
    } catch (e) {
      setWarnMsg(`Add failed: ${e.message}`)
    } finally {
      setAdding(false)
    }
  }, [data, choice, r.hub_id, onJobStarted])

  // GGUF multi-select: download EVERY checked quant. We await sequentially so we
  // don't hammer the download endpoint with N simultaneous POSTs; each response
  // is tagged with the repo's hub_id and handed to the shared job store.
  const addSelectedToLocal = useCallback(async () => {
    const opts = data?.options
    if (!opts) return
    const picked = opts.options.filter(o => choices.has(o.id))
    if (picked.length === 0) return
    setAdding(true)
    // One warning for the batch, not N identical alerts — the GGUF path enqueues
    // every checked quant, and 12 modal dialogs is not a better bug report.
    let warn = null
    try {
      for (const opt of picked) {
        const res = await fetchJson('/api/llm/repos/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            hub_id: r.hub_id, framework: opt.framework, task: opts.task,
            filename: opt.filename ?? null, include: opt.include ?? null,
            total_bytes: opt.total_bytes ?? null, register: true,
          }),
        })
        onJobStarted?.({ ...res, hub_id: r.hub_id })
        warn = warn || queueWarning(res)
      }
      setWarnMsg(warn)
    } catch (e) {
      setWarnMsg(`Add failed: ${e.message}`)
    } finally {
      setAdding(false)
    }
  }, [data, choices, r.hub_id, onJobStarted])

  const options = data?.options
  const selected = options?.options?.find(o => o.id === choice)
  const hasJob = !!job
  const jobActive = hasJob && (job.status === 'queued' || job.status === 'running')

  // Derived multi-select state (GGUF path).
  const optList = options?.options ?? []
  const freeBytes = data?.free_bytes ?? null
  // Worker-compatibility: an option is unfit when the selected worker's
  // capacity is known and this variant is bigger than it. Unknown sizes pass —
  // the filter must never hide what it cannot price.
  const unfitFor = (o) => !!workerFit && o.total_bytes != null && o.total_bytes > workerFit.bytes
  const fittable = optList.filter(o => o.fits_disk !== false && !unfitFor(o))
  const allSelected = fittable.length > 0 && fittable.every(o => choices.has(o.id))
  const selectedOpts = optList.filter(o => choices.has(o.id))
  const selectedCount = selectedOpts.length
  const selectedBytes = selectedOpts.reduce((s, o) => s + (o.total_bytes ?? 0), 0)

  const toggleChoice = (id) => setChoices(prev => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })
  const toggleSelectAll = () =>
    setChoices(allSelected ? new Set() : new Set(fittable.map(o => o.id)))

  return (
    <>
      <tr className={open ? 'hf-row-open' : ''}>
        <td className="hf-col-repo">
          <button className="hf-expand" onClick={expand} title="Show install options">
            {open ? '▾' : '▸'}
          </button>
          <a href={`https://huggingface.co/${r.hub_id}`} target="_blank" rel="noreferrer" title={r.hub_id}>
            {r.hub_id}
          </a>
          {r.trust === 'first-party' && (
            <span className="hf-trust hf-trust-first"
                  title="Curated trust tier (hugpy, not an official HF metric): canonical first-party publisher — the model's own vendor/org.">
              ✓ first-party
            </span>
          )}
          {r.trust === 'community' && (
            <span className="hf-trust hf-trust-community"
                  title="Curated trust tier (hugpy, not an official HF metric): reputable community repackager — a trusted re-uploader, not the canonical publisher.">
              community
            </span>
          )}
          {onDisk === 'installed' && (
            <span className="hf-ondisk hf-ondisk-yes"
                  title="Already on disk — this repo is installed on this box. Expanding still lets you add another variant / re-fetch.">
              ✓ on disk
            </span>
          )}
          {onDisk === 'partial' && (
            <span className="hf-ondisk hf-ondisk-partial"
                  title="Partially on disk — some files are present but the install is incomplete.">
              ◐ partial
            </span>
          )}
          {r.private && <span className="hf-private" title="private"> 🔒</span>}
        </td>
        <td className="hf-col-task">{tk}</td>
        <td className="hf-col-lib"><span className={`hf-fw fw-${fw}`}>{fw}</span></td>
        <td className="hf-col-num"
            title={fw === 'gguf'
              ? (open && selectedCount > 0
                  ? `${selectedCount} quant${selectedCount === 1 ? '' : 's'} selected · ${fmtBytes(selectedBytes)}. The whole repo (every quant) is ${fmtBytes(r.total_bytes)}.`
                  : 'Whole repo — every quantization. Expand to pick which (much smaller) variants to install.')
              : undefined}>
          {fw === 'gguf' && open && selectedCount > 0
            ? <>{fmtBytes(selectedBytes)}<em className="hf-size-note"> {selectedCount} quant{selectedCount === 1 ? '' : 's'}</em></>
            : <>{fmtBytes(r.total_bytes)}{fw === 'gguf' && <em className="hf-size-note"> all quants</em>}</>}
        </td>
        <td className="hf-col-num">{fmtCount(r.downloads)}</td>
        <td className="hf-col-num">{fmtCount(r.likes)}</td>
        <td className="hf-col-date">{(r.created_at || r.last_modified || '').slice(0, 10) || '—'}</td>
        <td className="hf-col-actions">
          <button className="btn-pull" onClick={expand} disabled={jobActive || externallyQueued}
                  title="Choose an install option">
            {jobActive ? '…' : externallyQueued ? 'queued' : '⬇ Options'}
          </button>
        </td>
      </tr>

      {open && (
        <tr className="hf-spec-row">
          <td colSpan={8}>
            {loading && <span className="hf-spec-loading">Loading specs…</span>}
            {data?.error && <span className="hf-search-error" title={data.error}>{data.error}</span>}
            {options && (
              <div className="hf-options">
                {data.spec && (
                  <div className="hf-spec-meta">
                    {data.spec.context_length != null && <span>ctx {data.spec.context_length}</span>}
                    {data.spec.total_bytes != null && <span>{fmtBytes(data.spec.total_bytes)}</span>}
                    {data.spec.license && <span>{data.spec.license}</span>}
                    {data.spec.gated && <span className="hf-gated">gated</span>}
                  </div>
                )}
                {options.options.length === 0 && <div className="hf-empty">No installable weights found.</div>}

                {fw === 'gguf' ? (
                  /* GGUF repos expose many quant variants — MULTI-select them. */
                  <>
                    {options.options.length > 0 && (
                      <div className="hf-opt-toolbar">
                        <label className="hf-opt hf-opt-all">
                          <input type="checkbox"
                                 checked={allSelected}
                                 disabled={jobActive || fittable.length === 0}
                                 onChange={toggleSelectAll} />
                          Select all{fittable.length > 0 ? ` (${fittable.length})` : ''}
                        </label>
                        <button type="button" className="hf-autofit"
                                disabled={jobActive || freeBytes == null || fittable.length === 0}
                                onClick={() => {
                                  const fit = autoSelectBySpace(optList, freeBytes)
                                  if (fit) setChoices(fit)
                                }}
                                title={freeBytes != null
                                  ? `Select the largest set of quants that fits this box's free disk (${fmtBytes(freeBytes)} free, 10% kept in reserve)`
                                  : 'Free-disk budget unavailable for this box'}>
                          ⤓ Auto-fit to space
                        </button>
                      </div>
                    )}
                    {options.options.map(o => (
                      <label key={o.id} className={`hf-opt ${(o.fits_disk === false || unfitFor(o)) ? 'hf-opt-toobig' : ''}`}
                             title={unfitFor(o) ? `Bigger than ${workerFit.name}'s capacity (${fmtBytes(workerFit.bytes)} = min of its VRAM+RAM ceiling and storage budget)` : undefined}>
                        <input type="checkbox"
                               checked={choices.has(o.id)}
                               disabled={o.fits_disk === false || unfitFor(o) || jobActive}
                               onChange={() => toggleChoice(o.id)} />
                        {o.label}{o.fits_disk === false ? ' · won’t fit' : unfitFor(o) ? ` · won’t fit on ${workerFit.name}` : ''}
                      </label>
                    ))}
                    {options.options.length > 0 && (
                      <div className="hf-opt-summary">
                        {selectedCount > 0
                          ? <>{selectedCount} quant{selectedCount === 1 ? '' : 's'} · {fmtBytes(selectedBytes)}</>
                          : 'nothing selected'}
                      </div>
                    )}
                  </>
                ) : (
                  /* Non-GGUF (transformers etc.): single-select radios, unchanged. */
                  options.options.map(o => (
                    <label key={o.id} className={`hf-opt ${(o.fits_disk === false || unfitFor(o)) ? 'hf-opt-toobig' : ''}`}
                           title={unfitFor(o) ? `Bigger than ${workerFit.name}'s capacity (${fmtBytes(workerFit.bytes)} = min of its VRAM+RAM ceiling and storage budget)` : undefined}>
                      <input type="radio" name={`opt-${r.hub_id}`} value={o.id}
                             checked={choice === o.id}
                             disabled={o.fits_disk === false || unfitFor(o) || jobActive}
                             onChange={() => setChoice(o.id)} />
                      {o.label}{o.fits_disk === false ? ' · won’t fit' : unfitFor(o) ? ` · won’t fit on ${workerFit.name}` : ''}
                    </label>
                  ))
                )}

                {warnMsg && (
                  <div className="hf-queue-warn" role="alert">
                    ⚠ {warnMsg}
                    <button className="hf-queue-warn-dismiss" title="Dismiss"
                            onClick={() => setWarnMsg(null)}>✕</button>
                  </div>
                )}
                {hasJob ? (
                  <DownloadBar job={job} onCancel={onCancelJob} onRetry={onRetryJob} />
                ) : options.options.length > 0 ? (
                  fw === 'gguf' ? (
                    <button className="btn-pull" onClick={addSelectedToLocal}
                            disabled={selectedCount === 0 || adding || externallyQueued}
                            title="Download and register every selected quant variant">
                      {adding ? '…' : `⬇ Add ${selectedCount} to local`}
                    </button>
                  ) : (
                    <button className="btn-pull" onClick={addToLocal}
                            disabled={!selected || adding || externallyQueued}
                            title="Download and register this variant">
                      {adding ? '…' : '⬇ Add to local'}
                    </button>
                  )
                ) : null}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  )
}

const COLUMNS = [
  { key: 'hub_id',     label: 'Repo',  get: r => r.hub_id, type: 'str' },
  { key: 'task',       label: 'Task',  get: r => inferTask(r), type: 'str' },
  { key: 'library',    label: 'Lib',   get: r => inferFramework(r), type: 'str' },
  { key: 'total_bytes',label: 'Size',  get: r => r.total_bytes ?? -1, type: 'num' },
  { key: 'downloads',  label: '↓',     get: r => r.downloads ?? -1, type: 'num' },
  { key: 'likes',      label: '♥',     get: r => r.likes ?? -1, type: 'num' },
  // Publish date — HF createdAt when present, else last-modified. ISO strings
  // slice to YYYY-MM-DD, which also sorts correctly as a plain string.
  { key: 'published',  label: 'Published', get: r => (r.created_at || r.last_modified || '').slice(0, 10), type: 'str' },
]

// ── Civitai panel — the SD-checkpoint habitat, wired to the comfy flow ──────
// Search via /api/civitai/search; the ⬇ action streams the single-file
// checkpoint into central's /checkpoints, where the registry sweep
// self-registers it as a comfy-<slug> model. The whole install IS the download.
// Civitai result columns — same clickable-sort contract as the HF COLUMNS.
const CV_COLUMNS = [
  { key: 'name',        label: 'Model',     get: r => r.name || '', type: 'str' },
  { key: 'base_model',  label: 'Base',      get: r => r.base_model || '', type: 'str' },
  { key: 'total_bytes', label: 'Size',      get: r => r.total_bytes ?? -1, type: 'num' },
  { key: 'downloads',   label: '↓',         get: r => r.downloads ?? -1, type: 'num' },
  { key: 'likes',       label: '♥',         get: r => r.likes ?? -1, type: 'num' },
  { key: 'published',   label: 'Published', get: r => r.published || '', type: 'str' },
]

function CivitaiPanel() {
  const [query, setQuery] = useSessionState('hugpy.sess.cv.query', '')
  const [base, setBase]   = useSessionState('hugpy.sess.cv.base', 'SD 1.5')
  const [sort, setSort]   = useSessionState('hugpy.sess.cv.sort', 'Highest Rated')
  const [rows, setRows]   = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dl, setDl]       = useState({})   // filename -> progress
  // Client-side column sort over the fetched page — session-sticky, exactly
  // like the HF table (the API `sort` select controls WHICH page comes back).
  const [sortKey, setSortKey] = useSessionState('hugpy.sess.cv.sortKey', 'downloads')
  const [sortDir, setSortDir] = useSessionState('hugpy.sess.cv.sortDir', 'desc')
  const reqRef = useRef(0)

  const sorted = useMemo(() => {
    const col = CV_COLUMNS.find(c => c.key === sortKey)
    if (!col) return rows
    const arr = [...rows]
    arr.sort((a, b) => {
      const av = col.get(a), bv = col.get(b)
      const cmp = col.type === 'num' ? av - bv : String(av).localeCompare(String(bv))
      return sortDir === 'asc' ? cmp : -cmp
    })
    return arr
  }, [rows, sortKey, sortDir])

  const toggleSort = useCallback((key) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }, [sortKey, setSortKey, setSortDir])

  useEffect(() => {
    const myReq = ++reqRef.current
    setLoading(true)
    const t = setTimeout(() => {
      const p = new URLSearchParams({ limit: '25', sort })
      if (query.trim()) p.set('query', query.trim())
      if (base) p.set('base', base)
      fetchJson(`/api/civitai/search?${p.toString()}`)
        .then(d => { if (myReq === reqRef.current) { setRows(Array.isArray(d) ? d : []); setError(null) } })
        .catch(e => { if (myReq === reqRef.current) { setError(e.message); setRows([]) } })
        .finally(() => { if (myReq === reqRef.current) setLoading(false) })
    }, 400)
    return () => clearTimeout(t)
  }, [query, base, sort])

  // Poll progress while anything is downloading.
  const active = Object.values(dl).some(s => s.status === 'downloading')
  useEffect(() => {
    const tick = () => fetchJson('/api/civitai/downloads').then(setDl).catch(() => {})
    tick()
    if (!active) return
    const iv = setInterval(tick, 2000)
    return () => clearInterval(iv)
  }, [active])

  const pull = useCallback(async (r) => {
    try {
      const res = await fetchJson('/api/civitai/download', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ download_url: r.download_url, filename: r.filename,
                               total_bytes: r.total_bytes,
                               // provenance stamp — the API writes these into a
                               // <file>.civitai.json sidecar + the central store
                               civitai_id: r.civitai_id, version_id: r.version_id,
                               name: r.name, base_model: r.base_model }),
      })
      if (res?.already) alert(`${r.filename} is already in /checkpoints`)
      setDl(d => ({ ...d, [r.filename]: { status: 'downloading', done_bytes: 0, total_bytes: r.total_bytes } }))
    } catch (e) { alert(`Download failed: ${e.message}`) }
  }, [])

  return (
    <>
      <div className="hf-search-bar">
        <input className="hf-search-input" placeholder="search Civitai checkpoints…"
               value={query} onChange={e => setQuery(e.target.value)} />
        <select value={base} onChange={e => setBase(e.target.value)}
                title="Base model — SD 1.5 / SDXL are what the vanilla comfy template serves">
          <option value="">Any base</option>
          <option>SD 1.5</option>
          <option>SDXL 1.0</option>
          <option>SD 2.1</option>
        </select>
        <select value={sort} onChange={e => setSort(e.target.value)}>
          <option>Highest Rated</option>
          <option>Most Downloaded</option>
          <option>Newest</option>
        </select>
        {loading && <span className="hf-search-spinner" aria-label="searching" />}
      </div>
      {error && <div className="hf-search-error">{error}</div>}
      {rows.length > 0 && (
        <div className="hf-results">
          <table className="hf-results-table">
            <thead>
              <tr>
                {CV_COLUMNS.map(c => (
                  <th key={c.key} className="hf-sortable" onClick={() => toggleSort(c.key)}>
                    {c.label}
                    {sortKey === c.key && <span className="hf-sort-arrow">{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                  </th>
                ))}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(r => {
                const st = dl[r.filename]
                const pct = st?.total_bytes ? Math.min(99, Math.round(100 * st.done_bytes / st.total_bytes)) : null
                return (
                  <tr key={r.civitai_id}>
                    <td className="hf-col-repo">
                      <a href={r.page_url} target="_blank" rel="noreferrer" title={r.filename}>{r.name}</a>
                      {!r.comfy_ready && <span title="base model outside the vanilla template's family (SD1.x/2.x/SDXL) — may not load"> ⚠</span>}
                    </td>
                    <td className="hf-col-task">{r.base_model}</td>
                    <td className="hf-col-num">{fmtBytes(r.total_bytes)}</td>
                    <td className="hf-col-num">{fmtCount(r.downloads)}</td>
                    <td className="hf-col-num">{fmtCount(r.likes)}</td>
                    <td className="hf-col-date">{r.published || '—'}</td>
                    <td className="hf-col-actions">
                      {st?.status === 'done' ? <span title="in /checkpoints — self-registered as a comfy model">✓ registered</span>
                        : st?.status === 'failed' ? <span className="hf-search-error" title={st.error}>failed</span>
                        : st?.status === 'downloading' ? <span>⏳ {pct != null ? `${pct}%` : '…'}</span>
                        : <button className="btn-pull" onClick={() => pull(r)}
                                  title="Stream into /checkpoints — registers as a comfy model automatically">
                            ⬇ → comfy
                          </button>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

export default function HFSearch({ models = [], onJobStarted, onCancelJob, onRetryJob, pendingByHub, jobsByHub, expanded, onToggleExpanded, embedded = false }) {
  // Session-sticky (defaults on a fresh tab; your picks survive reloads/tab
  // switches within the session — see useSessionState).
  const [query, setQuery]     = useSessionState('hugpy.sess.hf.query', '')
  // Key bumped to .v2 so tabs still carrying the old 'text-generation' default
  // pick up "Any task" (the same trick sortKey.v2 uses below).
  const [taskFilter, setTask] = useSessionState('hugpy.sess.hf.task.v2', DEFAULT_TASK)
  const [libFilter, setLib]   = useSessionState('hugpy.sess.hf.lib', '')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  // Default to 'relevance' = keep the server's ranked order (the `sorted` useMemo
  // returns `results` unchanged when sortKey matches no COLUMN). Key bumped to
  // .v2 so sessions carrying the old 'downloads' default pick up 'relevance'.
  const [sortKey, setSortKey] = useSessionState('hugpy.sess.hf.sortKey.v2', 'relevance')
  const [sortDir, setSortDir] = useSessionState('hugpy.sess.hf.sortDir.v2', 'desc')
  // Model source: HF Hub, or Civitai (the SD-checkpoint habitat — its files
  // stream into /checkpoints and self-register as comfy models).
  const [source, setSource] = useSessionState('hugpy.sess.hf.source', 'hf')
  // Worker-compatibility filter (2026-09-10): only show models/quants that fit
  // the selected worker. '' = no filter. The workers list is fetched when the
  // panel opens; a fetch failure just leaves the filter inert.
  const [fitWorkerId, setFitWorkerId] = useSessionState('hugpy.sess.hf.fitWorker', '')
  const [workers, setWorkers] = useState([])
  // How deep into the listing we've paged. Page 0 REPLACES the results (a fresh
  // search); every later page APPENDS, so the table grows as you walk the Hub.
  // Every control that changes what's being listed resets it via `rebrowse`.
  const [page, setPage] = useState(0)
  // A full page came back, so there is very likely another one behind it.
  const [hasMore, setHasMore] = useState(false)
  const reqIdRef = useRef(0)

  // Changing the query/filters/source starts the listing over from the top.
  // Resetting here rather than in an effect keeps it to a single fetch.
  const rebrowse = useCallback((apply) => { setPage(0); apply() }, [])

  useEffect(() => {
    if (source !== 'hf') return
    const q = query.trim()
    // No query and no filters is not "nothing to show" — it's "show me the whole
    // Hub", which is exactly what the Hub listing returns unfiltered.
    const fresh = page === 0
    const myReq = ++reqIdRef.current
    setLoading(true)
    const timer = setTimeout(() => {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(page * PAGE_SIZE),
      })
      if (q) params.set('q', q)
      if (taskFilter) params.set('task', taskFilter)
      if (libFilter) params.set('library', libFilter)
      fetchJson(`/api/search?${params.toString()}`)
        .then(data => {
          if (myReq !== reqIdRef.current) return
          const rows = Array.isArray(data) ? data : []
          setResults(prev => {
            if (fresh) return rows
            // Append, skipping any repo already listed — a re-ranked relevance
            // pool can straddle a page boundary and repeat a row.
            const seen = new Set(prev.map(r => r.hub_id))
            return [...prev, ...rows.filter(r => !seen.has(r.hub_id))]
          })
          setHasMore(rows.length === PAGE_SIZE)
          setError(null)
        })
        .catch(e => {
          if (myReq !== reqIdRef.current) return
          // A failed "load more" keeps what's already on screen; only a fresh
          // search clears the table.
          setError(e.message)
          if (fresh) setResults([])
          setHasMore(false)
        })
        .finally(() => { if (myReq === reqIdRef.current) setLoading(false) })
      // Debounce is for typing. "Load more" is a click — fire it immediately.
    }, fresh ? 400 : 0)
    return () => clearTimeout(timer)
  }, [source, query, taskFilter, libFilter, page])

  const sorted = useMemo(() => {
    const col = COLUMNS.find(c => c.key === sortKey)
    if (!col) return results
    const arr = [...results]
    arr.sort((a, b) => {
      const av = col.get(a), bv = col.get(b)
      let cmp = col.type === 'num' ? av - bv : String(av).localeCompare(String(bv))
      return sortDir === 'asc' ? cmp : -cmp
    })
    return arr
  }, [results, sortKey, sortDir])

  // Workers list for the compatibility filter — refreshed each time the HF
  // source is (re)opened, so capacities track the live fleet.
  useEffect(() => {
    if (!(embedded || expanded) || source !== 'hf') return
    fetchJson('/api/llm/workers')
      .then(ws => setWorkers(Array.isArray(ws) ? ws : (ws?.workers ?? [])))
      .catch(() => {})
  }, [embedded, expanded, source])

  // The selected worker's capacity: min(VRAM+RAM serve ceiling, storage
  // budget). null = filter off (none selected, or capacity unknowable).
  const workerFit = useMemo(() => {
    if (!fitWorkerId) return null
    const w = workers.find(x => x.id === fitWorkerId || x.name === fitWorkerId)
    if (!w) return null
    const bytes = workerFitBytes(w)
    return bytes != null ? { name: w.name || w.id, bytes } : null
  }, [fitWorkerId, workers])

  // Repo-level fit: non-GGUF repos are one weight set, so a repo bigger than
  // the worker is hidden outright. GGUF repos stay — their repo total is "all
  // quants", and the real per-quant filter lives in the expanded options.
  const fitSorted = useMemo(() => {
    if (!workerFit) return sorted
    return sorted.filter(r => {
      if (inferFramework(r) === 'gguf') return true
      return r.total_bytes == null || r.total_bytes <= workerFit.bytes
    })
  }, [sorted, workerFit])
  const fitHidden = sorted.length - fitSorted.length

  const toggleSort = useCallback((key) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }, [sortKey])

  // Which searched repos are ALREADY on this box — so a result you already have
  // is flagged instead of looking like a fresh find. Match on hub_id (the repo)
  // against the local registry; 'installed' = fully on disk, 'partial' = some
  // files present but incomplete. Case-insensitive.
  const installedByHub = useMemo(() => {
    const map = new Map()
    for (const mm of models || []) {
      const h = (mm.hub_id || '').toLowerCase()
      if (!h) continue
      if (mm.status === 'installed') map.set(h, 'installed')
      else if (mm.status === 'partial' && !map.has(h)) map.set(h, 'partial')
    }
    return map
  }, [models])

  // Downloads kicked off from here keep showing even when collapsed.
  const activeJobs = Object.values(jobsByHub ?? {}).filter(
    j => j.status === 'queued' || j.status === 'running'
  ).length

  const isOpen = embedded || expanded

  return (
    <section className={`hf-search ${isOpen ? 'hf-search-expanded' : ''}`}>
      {!embedded && (
      <div
        className="section-strip clickable"
        onClick={onToggleExpanded}
        title={expanded ? 'Collapse' : 'Search the Hugging Face Hub and add models'}
      >
        <span className="section-caret">{expanded ? '▾' : '▸'}</span>
        <span className="section-title">Add models · {source === 'civitai' ? 'Civitai' : source === 'downloads' ? 'Download queue' : 'Hugging Face Hub'}</span>
        {!expanded && (
          <span className="hf-strip-hint">search &amp; download new models</span>
        )}
        {activeJobs > 0 && (
          <span className="section-count hf-strip-active">{activeJobs} downloading</span>
        )}
        {expanded && fitSorted.length > 0 && (
          <span className="section-count">{fitSorted.length} results</span>
        )}
        {error && <span className="hf-search-error" title={error}>!</span>}
      </div>
      )}

      {isOpen && (
        <>
          {/* Source toggle: HF Hub ⇄ Civitai (SD checkpoints → /checkpoints →
              comfy) ⇄ the download queue (2026-08-13 — server-truth view of
              EVERY model download, not just this session's). */}
          <div className="hf-source-toggle" role="tablist" aria-label="Model source">
            <button type="button" role="tab" aria-selected={source !== 'civitai' && source !== 'downloads'}
                    className={`hf-source-btn${source !== 'civitai' && source !== 'downloads' ? ' hf-source-active' : ''}`}
                    onClick={() => rebrowse(() => setSource('hf'))}>🤗 Hugging Face</button>
            <button type="button" role="tab" aria-selected={source === 'civitai'}
                    className={`hf-source-btn${source === 'civitai' ? ' hf-source-active' : ''}`}
                    onClick={() => rebrowse(() => setSource('civitai'))}
                    title="SD-checkpoint habitat — downloads land in /checkpoints and self-register as comfy models">
              🧩 Civitai</button>
            <button type="button" role="tab" aria-selected={source === 'downloads'}
                    className={`hf-source-btn${source === 'downloads' ? ' hf-source-active' : ''}`}
                    onClick={() => rebrowse(() => setSource('downloads'))}
                    title="Every model download on this box — queued, running and recent, from any session — with cancel/retry">
              ⬇ Download queue</button>
          </div>
          {/* Every download started from HF search, gathered into one list —
              stays put whether you're on the HF or Civitai source view. Hidden
              on the queue tab, whose server-truth list supersedes it there. */}
          {source !== 'downloads' && (
            <DownloadsPanel jobsByHub={jobsByHub} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          )}
          {source === 'downloads' && <DownloadsQueue variant="panel" />}
          {source === 'civitai' && <CivitaiPanel />}
          {source !== 'civitai' && source !== 'downloads' && (<>
          <div className="hf-search-bar">
            <input className="hf-search-input" placeholder="filter by name (optional) — leave empty to list every model…"
                   value={query} onChange={e => rebrowse(() => setQuery(e.target.value))} />
            <select value={taskFilter} onChange={e => rebrowse(() => setTask(e.target.value))}>
              <option value="">Any task</option>
              {TASK_GROUPS.map(([label, tasks]) => (
                <optgroup key={label} label={label}>
                  {tasks.map(t => <option key={t} value={t}>{t}</option>)}
                </optgroup>
              ))}
            </select>
            <select value={libFilter} onChange={e => rebrowse(() => setLib(e.target.value))}>
              <option value="">Any library</option>
              {LIBRARIES.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
            <select value={fitWorkerId} onChange={e => setFitWorkerId(e.target.value)}
                    title="Worker compatibility — only show models and quants that can fit the selected worker: at or under min(its VRAM+RAM serve ceiling, its storage budget). Oversized quants inside GGUF repos are disabled with the reason.">
              <option value="">Fits: any worker</option>
              {workers.map(w => (
                <option key={w.id} value={w.id}>
                  fits {w.name || w.id}{workerFitBytes(w) != null ? ` (≤ ${fmtBytes(workerFitBytes(w))})` : ''}
                </option>
              ))}
            </select>
            {loading && <span className="hf-search-loading">…</span>}
          </div>
          {workerFit && fitHidden > 0 && (
            <div className="hf-fit-note" title="Repos whose single weight set is bigger than the selected worker's capacity are hidden. GGUF repos stay listed — expand one to see which quants fit.">
              {fitHidden} result{fitHidden === 1 ? '' : 's'} hidden — bigger than {workerFit.name}'s {fmtBytes(workerFit.bytes)} capacity
            </div>
          )}

          {fitSorted.length > 0 && (
            <>
            <div className="hf-relevance-bar">
              <span className="hf-relevance-label">Sort</span>
              {sortKey === 'relevance' ? (
                <span className="hf-relevance-pill hf-relevance-active"
                      title="Results are in the server's relevance ranking (closest name + uploader trust). Click any column to re-sort client-side.">
                  ✓ Relevance
                </span>
              ) : (
                <button type="button" className="hf-relevance-pill hf-relevance-reset"
                        onClick={() => { setSortKey('relevance'); setSortDir('desc') }}
                        title="Back to the server's relevance ranking (closest name + uploader trust)">
                  ↺ Relevance
                </button>
              )}
            </div>
            <div className="hf-results">
              <table className="hf-results-table">
                <thead>
                  <tr>
                    {COLUMNS.map(c => (
                      <th key={c.key} className="hf-sortable" onClick={() => toggleSort(c.key)}>
                        {c.label}
                        {sortKey === c.key && <span className="hf-sort-arrow">{sortDir === 'asc' ? ' ▲' : ' ▼'}</span>}
                      </th>
                    ))}
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {fitSorted.map(r => (
                    <ResultRow key={r.hub_id} r={r}
                      fw={inferFramework(r)} tk={inferTask(r)}
                      onDisk={installedByHub.get((r.hub_id || '').toLowerCase())}
                      externallyQueued={pendingByHub?.[r.hub_id]}
                      job={jobsByHub?.[r.hub_id]}
                      workerFit={workerFit}
                      onJobStarted={onJobStarted} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
                  ))}
                </tbody>
              </table>
            </div>
            </>
          )}

          {!loading && !error && fitSorted.length === 0 && (
            <div className="hf-empty">
              {workerFit && sorted.length > 0
                ? `No matches fit ${workerFit.name} (${fmtBytes(workerFit.bytes)} capacity).`
                : 'No matches.'}
            </div>
          )}

          {/* Walking the Hub: each click pulls the next page and appends it. The
              Hub publishes no total to count down from, so "more" just means the
              last page came back full. Deliberately OUTSIDE the results block —
              when the worker-fit filter hides a whole page, paging on is the only
              way to reach the models that do fit. */}
          {results.length > 0 && (
            <div className="hf-more-bar">
              {hasMore ? (
                <button type="button" className="hf-more-btn" disabled={loading}
                        onClick={() => setPage(p => p + 1)}
                        title="Fetch the next page of Hugging Face models and add it to the list">
                  {loading ? 'loading…' : `Load more — ${results.length} fetched so far`}
                </button>
              ) : (
                <span className="hf-more-end">
                  End of listing — {results.length} model{results.length === 1 ? '' : 's'} fetched
                </span>
              )}
            </div>
          )}
          </>)}
        </>
      )}
    </section>
  )
}
