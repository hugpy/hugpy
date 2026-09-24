import { useMemo, useState, useCallback, useRef, useEffect, Fragment } from 'react'

import { fetchJson } from '../../api'
import ServingControl from './ServingControl'
import QuantControl, { WorkerQuantSelect } from './QuantControl'
import { sizeView } from './modelSize'
import PlacementControl from './PlacementControl'
import GroupHeaderRow, { MemberVerdicts } from './GroupHeaderRow'
import ModelLogs from '../ModelLogs/ModelLogs'
import { useModelGroups } from './useModelGroups'
import useSessionState from '../../hooks/useSessionState'
import { useModelStatus, refreshModelStatus } from './useModelStatus'
import {
  BUCKETS, BUCKET_LABELS, EMPTY_STATUS_FILTERS, STATUS_FILTER_OPTIONS, STATUS_SORTS, WORTH_META, WORTH_ORDER,
  hasStatusFilters, matchesStatusFilters, parseStatusQuery, statusFor, summaryCounts, writeStatusQuery,
} from './modelStatus'
import {
  AdmissionCell, FailureCell, GradeCell, ServableCell, ThroughputCell, VerificationCell, WorkerStateChips, WorthCell,
} from './StatusCells'
import ModelStatusDetail from './ModelStatusDetail'
import { archiveMark, archiveText } from './archiveMark'
import './ModelTable.css'
import './ModelStatus.css'

function fmtCtx(n) {
  if (n == null) return '?'
  n = Number(n)
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

const isGgufFw = (fw) => { const f = String(fw || '').toLowerCase(); return f === 'gguf' || f === 'llama_cpp' }

function fmtBytes(n) {
  if (n == null) return '–'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// Short quant tag from a .gguf basename (Q4_K_M, IQ3_XS, F16, …); longest-match
// first so q4_k_m wins over a bare q4. Falls back to the basename sans extension.
const _QUANT_TOKENS = [
  'iq1_s', 'iq1_m', 'iq2_xxs', 'iq2_xs', 'iq2_s', 'iq2_m', 'iq3_xxs', 'iq3_xs',
  'iq3_s', 'iq3_m', 'iq4_xs', 'iq4_nl', 'q2_k', 'q3_k_l', 'q3_k_m', 'q3_k_s',
  'q4_k_m', 'q4_k_s', 'q5_k_m', 'q5_k_s', 'q6_k', 'q8_0', 'q4_0', 'q4_1',
  'q5_0', 'q5_1', 'bf16', 'f16', 'f32',
]
function quantTag(fn) {
  if (!fn) return ''
  const b = String(fn).toLowerCase()
  for (const q of _QUANT_TOKENS) if (b.includes(q)) return q.toUpperCase()
  return String(fn).replace(/\.gguf$/i, '')
}

function modelSearchText(m) {
  return [
    m.model_key,
    m.key,
    m.name,
    m.hub_id,
    m.framework,
    modelTask(m),
    m.status,
    ...(Array.isArray(m.tasks) ? m.tasks : []),
    ...(Array.isArray(m.tags) ? m.tags : []),
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
}

function modelFramework(m) {
  return m.framework ?? ''
}

function modelStatus(m) {
  return m.status ?? 'missing'
}
export function modelTask(m) {
  return (
    m.primary_task ??
    m.task ??
    m.pipeline_tag ??
    m.pipelineTag ??
    m.tags?.find?.(t => [
      'text-generation',
      'image-text-to-text',
      'automatic-speech-recognition',
      'feature-extraction',
      'summarization',
      'text-classification',
      'token-classification',
      'fill-mask',
      'zero-shot-classification',
      'image-classification',
      'object-detection',
    ].includes(t)) ??
    m.tasks?.[0] ??
    'unknown'
  )
}

// Every task a model advertises — the full `tasks` list plus the primary —
// so multi-task models (e.g. image-text-to-text + text-generation) are
// findable under each of their tasks, not just the primary one.
// Exported (with modelTask) so other tables (WorkerLoadTable) render tasks
// the same way instead of re-growing the primary-task-only skew.
export function modelTasks(m) {
  return [...new Set([...(Array.isArray(m.tasks) ? m.tasks : []), modelTask(m)])]
    .filter(Boolean)
}


function StatusBadge({ status }) {
  if (status === 'installed') return <span className="badge badge-green">✓ ready</span>
  if (status === 'partial') return <span className="badge badge-yellow">◐ partial</span>
  return <span className="badge badge-red">✗ missing</span>
}

function DownloadProgress({ job, onCancel, onRetry }) {
  const pct = Math.round((job.progress ?? 0) * 100)
  const indeterminate = job.status === 'running' && !job.total_bytes
  const active = job.status === 'running' || job.status === 'queued'
  const retryable = job.status === 'failed' || job.status === 'cancelled' ||
                    job.status === 'expired'
  const bps = job.bytes_per_second

  return (
    <div className="mt-dl">
      <div className={`mt-dl-bar ${indeterminate ? 'mt-dl-indet' : ''} ${job.stalled ? 'mt-dl-stalled' : ''} mt-dl-${job.status}`}>
        <div className="mt-dl-fill" style={{ width: indeterminate ? '40%' : `${pct}%` }} />
      </div>

      <span className="mt-dl-label" title={job.error || job.message || ''}>
        {job.status === 'queued' && (job.message || 'queued…')}
        {job.status === 'running' && (job.stalled
          ? '⚠ stalled — resuming…'
          : indeterminate
            ? `downloading… ${fmtBytes(job.downloaded_bytes)}`
            : `${pct}% · ${fmtBytes(job.downloaded_bytes)} / ${fmtBytes(job.total_bytes)}`
        )}
        {job.status === 'running' && job.attempt > 1 && ` · try ${job.attempt}/${job.max_attempts}`}
        {job.status === 'running' && !job.stalled && bps > 0 && ` · ${fmtBytes(bps)}/s`}
        {job.status === 'failed' && `✗ ${job.error_reason ? `[${job.error_reason}] ` : ''}${job.error ?? `job ${job.id} status=failed with no error text recorded (attempt ${job.attempt ?? '?'}/${job.max_attempts ?? '?'})`}`}
        {job.status === 'expired' && `✗ expired — ${job.message || `job ${job.id} expired with no message recorded (attempt ${job.attempt ?? 0}/${job.max_attempts ?? '?'})`}`}
        {job.status === 'cancelled' && 'cancelled'}
      </span>

      {active && (
        <button className="mt-dl-cancel" onClick={() => onCancel(job.id)} title="Cancel download">
          ✕
        </button>
      )}
      {retryable && onRetry && (
        <button className="mt-dl-retry" onClick={() => onRetry(job.id)}
                title="Resume from where it stopped">
          ↻
        </button>
      )}
    </div>
  )
}

// The spellings a model key may legitimately be named by — mirrors
// comms.priority_groups.key_forms (raw, lowercase, "/"-tail, "~"-tail) so the
// Group column matches membership exactly the way the server does.
function pgKeyForms(key) {
  const s = String(key || '').trim()
  if (!s) return []
  const out = new Set([s, s.toLowerCase()])
  const tail = s.split('/').pop()
  out.add(tail); out.add(tail.toLowerCase())
  if (s.includes('~')) {
    const base = s.split('~').slice(1).join('~')
    if (base) { out.add(base); out.add(base.toLowerCase()) }
  }
  return [...out]
}

const COLUMNS = [
  { key: 'index', label: '#', type: 'num', get: (_m, i) => i + 1 },
  // When two owners ship the same repo name the model_key is qualified
  // (`<owner>~<name>`); surface that owner so the duplicate names are distinct.
  { key: 'name', label: 'Name', type: 'str', get: m => {
      const k = m.model_key ?? m.key ?? ''
      const base = m.name ?? k
      return k.includes('~') ? `${base} (${k.split('~')[0]})` : base
    } },
  { key: 'media', label: 'Media', type: 'num', get: m => (m.media ? 1 : 0) },
  { key: 'framework', label: 'Framework', type: 'str', get: m => modelFramework(m) },
  { key: 'task', label: 'Task', type: 'str', get: m => modelTask(m) },
  { key: 'ctx', label: 'Ctx', type: 'num', get: m => Number(m.model_max_length ?? -1) },
  { key: 'status', label: 'Status', type: 'str', get: m => modelStatus(m) },
  // Explicit priority group (annotated in visibleModels; '' = ungrouped).
  { key: 'pgroup', label: 'Group', type: 'str', get: m => m.__pgroup ?? '' },
]

// "Worth my time" columns (GET /llm/models/status; annotated as m.__status in
// visibleModels). Only offered when the endpoint answers — feature-detected.
const STATUS_COLUMNS = [
  { key: 'worth', label: 'Worth', type: 'num', get: m => STATUS_SORTS.worth(m.__status), Cell: WorthCell },
  { key: 'verification', label: 'Verification', type: 'num', get: m => STATUS_SORTS.verification(m.__status), Cell: VerificationCell },
  { key: 'admission', label: 'Admission', type: 'num', get: m => STATUS_SORTS.admission(m.__status), Cell: AdmissionCell },
  { key: 'grade', label: 'Grade', type: 'num', get: m => STATUS_SORTS.grade(m.__status), Cell: GradeCell },
  { key: 'tps', label: 'tok/s (avg of calls)', type: 'num', get: m => m.__status?.throughput?.mean_tok_s ?? -1, Cell: ThroughputCell },
  { key: 'failure', label: 'Last failure', type: 'num', get: m => STATUS_SORTS.failure(m.__status), Cell: FailureCell },
  { key: 'servable', label: 'Servable now', type: 'num', get: m => STATUS_SORTS.servable(m.__status), Cell: ServableCell },
  { key: 'wstate', label: 'Workers', type: 'str', get: m => (m.__status?.workers || []).map(w => w.state).join(','), Cell: WorkerStateChips },
]

// URL <-> status filters (ms_* params), replaceState: shareable, no reload.
function readStatusFilters() {
  try { return parseStatusQuery(window.location.search) } catch { return { ...EMPTY_STATUS_FILTERS } }
}
function writeStatusFilters(f) {
  try {
    const { pathname, search, hash } = window.location
    const s = writeStatusQuery(search, f)
    if (s !== search) window.history.replaceState(window.history.state, '', `${pathname}${s}${hash}`)
  } catch { /* non-browser host */ }
}

export default function ModelTable({
  models,
  jobsByModel,
  activeChat,
  onDownload,
  onChat,
  onDelete,
  onPrune,
  onSetMedia,
  onSetMediaDefault,
  onCancel,
  onRetry,
  workers = [],
  onAssignWorker,
  onProbeWorker,
  onRefresh,
}) {
  const [probe, setProbe] = useState({})   // `${workerId}:${modelKey}` -> 'probing'|result
  const [logsOpen, setLogsOpen] = useState({})   // modelKey -> bool (failure log expanded)
  // ARCHIVE MARK (POST/DELETE /llm/models/<key>/archive). The console only
  // MARKS; the operator's `hugpy-model-archive --apply` sweep moves the files.
  // Outcome shown inline per model (never an alert loop).
  const [archiveNote, setArchiveNote] = useState({})   // modelKey -> {busy?, text}
  const setArchive = useCallback((modelKey, mark) => {
    let body
    if (mark) {
      // One prompt for the optional reason; Cancel aborts, empty = no reason.
      const reason = window.prompt(
        `Mark "${modelKey}" for archive?\n\nCentral stops placing and routing it at once; `
        + 'nothing is moved or deleted until the operator runs `hugpy-model-archive --apply`.'
        + '\n\nReason (optional):', '')
      if (reason === null) return
      body = JSON.stringify({ reason: reason.trim() || null })
    }
    setArchiveNote(n => ({ ...n, [modelKey]: { busy: true, text: mark ? 'marking…' : 'unmarking…' } }))
    fetchJson(`/api/llm/models/${encodeURIComponent(modelKey)}/archive`, {
      method: mark ? 'POST' : 'DELETE',
      ...(body ? { headers: { 'Content-Type': 'application/json' }, body } : {}),
    })
      .then(d => {
        const text = mark ? `✓ ${archiveText(d.archived)}`
          : d.was_marked ? `✓ unmarked (was ${archiveText(d.was)})` : '✓ was not marked'
        setArchiveNote(n => ({ ...n, [modelKey]: { text } }))
        onRefresh?.()
        refreshModelStatus()?.catch?.(() => {})
      })
      .catch(e => setArchiveNote(n => ({ ...n, [modelKey]: { text: `✗ ${e.message || e}` } })))
  }, [onRefresh])

  // Session-sticky filters/sort: defaults on a fresh tab, the operator's picks
  // survive reloads/tab-switches within the session (see useSessionState).
  const [query, setQuery] = useSessionState('hugpy.sess.mt.query', '')
  const [frameworkFilter, setFrameworkFilter] = useSessionState('hugpy.sess.mt.fw', '')
  const [taskFilter, setTaskFilter] = useSessionState('hugpy.sess.mt.task', '')
  const [statusFilter, setStatusFilter] = useSessionState('hugpy.sess.mt.status', 'installed')
  const [assignedOnly, setAssignedOnly] = useSessionState('hugpy.sess.mt.assigned', false)  // only models assigned to a worker
  const [sortKey, setSortKey] = useSessionState('hugpy.sess.mt.sortKey', 'name')
  const [sortDir, setSortDir] = useSessionState('hugpy.sess.mt.sortDir', 'asc')
  const [detailKey, setDetailKey] = useState(null)

  // WORTH MY TIME — one shared status snapshot (polls <=3 s while anything is
  // downloading/loading/serving, else 15 s). Filters live in the URL.
  const mstatus = useModelStatus()
  const sIndex = mstatus.index
  const statusOn = sIndex.available
  const [sfilters, setSfilters] = useState(readStatusFilters)
  useEffect(() => { writeStatusFilters(sfilters) }, [sfilters])
  const setSf = useCallback((k, v) => setSfilters(f => ({ ...f, [k]: f[k] === v ? '' : v })), [])
  const columns = useMemo(() => (statusOn ? [...COLUMNS, ...STATUS_COLUMNS] : COLUMNS), [statusOn])
  const summary = useMemo(() => summaryCounts(sIndex), [sIndex])

  // MODEL GROUPS. Purely additive: when the endpoint reports no multi-member
  // group (or isn't there at all) `groupedRows` degrades to the plain
  // `visibleModels` list and this table renders exactly as it did before.
  const groups = useModelGroups()
  const [groupBy, setGroupBy] = useSessionState('hugpy.sess.mt.groupBy', true)
  const [collapsed, setCollapsed] = useSessionState('hugpy.sess.mt.gcollapsed', {})
  const toggleCollapse = useCallback((gk) => {
    setCollapsed(c => ({ ...c, [gk]: !c?.[gk] }))
  }, [setCollapsed])

  // CON-03: size/quant/params/ctx + VRAM-annotated recommendations from THE
  // one metadata source (GET /models/<key>/meta), fetched lazily when a row
  // expands. Keys: "<model>" (base) and "<model>@<workerId>" (per-worker fit).
  // Cached in parent state (the detail row is rendered by renderModelDetail, a
  // plain function that holds no state of its own).
  const [modelMeta, setModelMeta] = useState({})
  const metaInflight = useRef(new Set())
  const loadModelMeta = useCallback((modelKey, workerIds = []) => {
    const want = [['', modelKey], ...workerIds.map(id => [id, `${modelKey}@${id}`])]
    for (const [workerId, cacheKey] of want) {
      if (metaInflight.current.has(cacheKey)) continue
      metaInflight.current.add(cacheKey)
      const q = workerId ? `?worker=${encodeURIComponent(workerId)}` : ''
      fetchJson(`/api/models/${encodeURIComponent(modelKey)}/meta${q}`)
        .then(m => setModelMeta(p => ({ ...p, [cacheKey]: m })))
        .catch(() => setModelMeta(p => ({ ...p, [cacheKey]: null })))
    }
  }, [])
  const CHATTABLE = new Set(['text-generation', 'image-text-to-text'])
  const isChattable = (m) => CHATTABLE.has(modelTask(m))

  // Disk discovery ("Discover models"): the server's discovery report can
  // shrink if a walk ran against a degraded storage mount — models vanish
  // from this list while their files still sit on disk. POST kicks a
  // server-side re-walk (slow: whole tree + hub enrichment), so poll the
  // state until it settles, then reload the list. Operator-gated server-side.
  const [discover, setDiscover] = useState({ running: false, error: null })
  const discoverTimer = useRef(null)
  useEffect(() => () => clearTimeout(discoverTimer.current), [])
  const pollDiscover = useCallback(function poll() {
    fetchJson('/api/models/discover')
      .then(s => {
        if (s.running) { discoverTimer.current = setTimeout(poll, 2500); return }
        setDiscover({ running: false, error: s.error || null })
        if (!s.error) onRefresh?.()
      })
      .catch(err => setDiscover({ running: false, error: String(err.message || err) }))
  }, [onRefresh])
  const startDiscover = useCallback(() => {
    setDiscover({ running: true, error: null })
    fetchJson('/api/models/discover', { method: 'POST' })
      .then(() => { discoverTimer.current = setTimeout(pollDiscover, 2500) })
      // POST can 409 when a sweep is already running server-side — either
      // way the state poll is the truth, so just follow it.
      .catch(() => { discoverTimer.current = setTimeout(pollDiscover, 1000) })
  }, [pollDiscover])

  const frameworks = useMemo(() => {
    return [...new Set(models.map(modelFramework).filter(Boolean))].sort()
  }, [models])

  const tasks = useMemo(() => {
    return [...new Set(models.flatMap(modelTasks))].sort()
  }, [models])

  const statuses = useMemo(() => {
    return [...new Set(models.map(modelStatus).filter(Boolean))].sort()
  }, [models])

  const assignedKeys = useMemo(() => new Set(workers.flatMap(w => w.models ?? [])), [workers])

  // PRIORITY GROUPS (explicit, /llm/model-groups) — the Group column. Distinct
  // from the DERIVED grouping above (useModelGroups): this is the operator's
  // hand-written registry, and the select WRITES membership (operator-gated
  // server-side). Matching mirrors comms.priority_groups.key_forms.
  const [pgroups, setPgroups] = useState([])
  const loadPgroups = useCallback(() => {
    fetchJson('/api/llm/model-groups')
      .then(d => setPgroups(d.groups || []))
      .catch(() => {})
  }, [])
  useEffect(() => { loadPgroups() }, [loadPgroups])
  const pgIndex = useMemo(() => {
    const idx = new Map()
    for (const g of pgroups) {
      if (!g.enabled) continue
      for (const m of (g.members || [])) {
        for (const f of pgKeyForms(m)) if (!idx.has(f)) idx.set(f, g)
      }
    }
    return idx
  }, [pgroups])
  const pgFor = useCallback(m => {
    for (const f of pgKeyForms(m.model_key ?? m.key ?? '')) {
      const g = pgIndex.get(f)
      if (g) return g
    }
    return null
  }, [pgIndex])
  const setModelGroup = useCallback((modelKey, gid) => {
    fetchJson('/api/llm/model-groups/member', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_key: modelKey, group_id: gid || null }),
    })
      .then(loadPgroups)
      .catch(e => window.alert(String(e.message || e)))
  }, [loadPgroups])
  // Allocate the whole group: to ONE worker (the header's "→ worker" select)
  // or to the group's own ordered workers list. Designation only — loading
  // stays the per-model gesture with its preflight.
  const [pgAllocNote, setPgAllocNote] = useState(null)  // {id, text}
  const allocatePgroup = useCallback((g, workerName) => {
    const body = workerName ? { workers: [workerName] } : {}
    fetchJson(`/api/llm/model-groups/${encodeURIComponent(g.id)}/allocate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(r => {
        const skips = (r.outcomes || []).filter(
          o => o.status !== 'designated' && o.status !== 'already')
        setPgAllocNote({
          id: g.id,
          text: `${r.designated} designated` +
            (skips.length ? `, ${skips.length} skipped` : ''),
        })
        onRefresh?.()
      })
      .catch(e => window.alert(String(e.message || e)))
  }, [onRefresh])

  const visibleModels = useMemo(() => {
    const q = query.trim().toLowerCase()

    const filtered = models.filter(m => {
      if (q && !modelSearchText(m).includes(q)) return false
      if (frameworkFilter && modelFramework(m) !== frameworkFilter) return false
      if (taskFilter && !modelTasks(m).includes(taskFilter)) return false
      if (statusFilter && modelStatus(m) !== statusFilter) return false
      if (assignedOnly && !assignedKeys.has(m.model_key ?? m.key)) return false
      if (statusOn && !matchesStatusFilters(statusFor(sIndex, m), sfilters)) return false
      return true
    }).map(m => ({ ...m, __pgroup: pgFor(m)?.name || '', __status: statusOn ? statusFor(sIndex, m) : null }))

    const col = columns.find(c => c.key === sortKey)
    if (!col) return filtered

    return [...filtered].sort((a, b) => {
      const av = col.get(a, 0)
      const bv = col.get(b, 0)

      let cmp
      if (col.type === 'num') {
        cmp = Number(av) - Number(bv)
      } else {
        cmp = String(av).localeCompare(String(bv))
      }

      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [models, query, frameworkFilter, taskFilter, statusFilter, assignedOnly, assignedKeys, sortKey, sortDir, pgFor,
      statusOn, sIndex, sfilters, columns])

  /**
   * MODEL GROUPS — the render list, as a flat array of
   * `{kind: 'group'|'model'}` rows.
   *
   * Grouping REORDERS: a group's members are pulled together under one header
   * so the operator can see the iterations side by side, which is the entire
   * point. Everything not in a multi-member group keeps the table's own sort
   * order and follows the groups. Ungrouped/toggled-off is the identity
   * transform over `visibleModels` — the pre-feature table, exactly.
   */
  const groupedRows = useMemo(() => {
    const plain = visibleModels.map(m => ({ kind: 'model', model: m }))

    const visibleKeys = new Set(visibleModels.map(m => m.model_key ?? m.key))
    const byKey = new Map(visibleModels.map(m => [m.model_key ?? m.key, m]))
    const claimed = new Set()
    const rows = []

    // EXPLICIT priority groups first — the operator wrote these, so they
    // cluster whenever any member is visible (even one: an explicit group is a
    // deliberate unit, unlike the derived <2 rule below), collapsible like a
    // single row. Members render in the GROUP's order — that order is the
    // fallback priority, and reordering it here would misstate it.
    const formToKey = new Map()
    for (const k of visibleKeys) {
      for (const f of pgKeyForms(k)) if (!formToKey.has(f)) formToKey.set(f, k)
    }
    for (const g of pgroups) {
      if (!g.enabled) continue
      const present = []
      for (const mem of (g.members || [])) {
        for (const f of pgKeyForms(mem)) {
          const k = formToKey.get(f)
          if (k && !claimed.has(k) && !present.includes(k)) { present.push(k); break }
        }
      }
      if (!present.length) continue
      rows.push({ kind: 'pgroup', pgroup: g, count: present.length })
      if (!collapsed?.[`pg:${g.id}`]) {
        for (const k of present) rows.push({ kind: 'model', model: byKey.get(k), pgroup: g })
      }
      present.forEach(k => claimed.add(k))
    }

    if (groupBy && groups.multiMember.length) {
      for (const g of groups.multiMember) {
        // Only group what the CURRENT filters already show. A group header for
        // rows the operator filtered away would be a phantom.
        const present = (g.members || [])
          .map(mem => mem.model_key)
          .filter(k => visibleKeys.has(k) && !claimed.has(k))
        if (present.length < 2) continue
        rows.push({ kind: 'group', group: g })
        if (!collapsed?.[g.group_key]) {
          for (const k of present) rows.push({ kind: 'model', model: byKey.get(k), group: g })
        }
        present.forEach(k => claimed.add(k))
      }
    }

    if (!rows.length) return plain
    for (const r of plain) {
      if (!claimed.has(r.model.model_key ?? r.model.key)) rows.push(r)
    }
    return rows
  }, [visibleModels, groupBy, groups.multiMember, collapsed, pgroups])

  const toggleSort = useCallback((key) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'ctx' ? 'desc' : 'asc')
    }
  }, [sortKey])

  const clearFilters = useCallback(() => {
    setQuery('')
    setFrameworkFilter('')
    setTaskFilter('')
    setStatusFilter('')
    setAssignedOnly(false)
    setSfilters({ ...EMPTY_STATUS_FILTERS })
  }, [])

  if (!models.length) {
    return <div className="empty">No models available.</div>
  }
// fields we want shown first, in this order, with a label + formatter
const DETAIL_FIELDS = [
  ['hub_id',           'Hub ID',        v => v],
  ['model_key',        'Model key',     v => v],
  ['key',              'Key (legacy)',  v => v],
  ['framework',        'Framework',     v => v],
  ['primary_task',     'Primary task',  v => v],
  ['tasks',            'Tasks',         v => (Array.isArray(v) ? v.join(', ') : v)],
  ['model_max_length', 'Context',       v => fmtCtx(v)],
  ['status',           'Status',        v => v],
  ['folder',           'Folder',        v => v],
  ['filename',         'Filename',      v => v],
  ['parameter_count',  'Parameters',    v => (v == null ? null : Number(v).toLocaleString())],
  ['license',          'License',       v => v],
  ['languages',        'Languages',     v => (Array.isArray(v) ? v.join(', ') : v)],
]

const SHOWN_KEYS = new Set(DETAIL_FIELDS.map(([k]) => k))

// A plain render FUNCTION, not a component: declared inside ModelTable's body,
// `<ModelDetail/>` was a NEW component type on every ModelTable render (status
// poll, workers poll), so React unmounted + remounted the whole expanded row
// each time — every child re-ran its mount effects (PlacementControl's
// GET /settings/shard_models/<key> per poll) and the row flickered. Called as
// a function its children keep stable types and stay mounted. It uses no hooks.
function renderModelDetail(model, colSpan) {
  // declared fields that actually have a value
  const rows = DETAIL_FIELDS
    .map(([k, label, fmt]) => [label, fmt(model[k])])
    .filter(([, val]) => val != null && val !== '' && !(Array.isArray(val) && !val.length))

  // anything discovery included that we didn't explicitly list
  const extras = Object.entries(model)
    .filter(([k, v]) =>
      !SHOWN_KEYS.has(k) &&
      v != null && v !== '' &&
      typeof v !== 'object'        // skip nested blobs; see note below
    )

  // Per-model action state, recomputed here (the detail row is the only place
  // actions live now that the per-row kebab is gone).
  const modelKey = model.model_key ?? model.key ?? model.hub_id ?? model.name
  const job = jobsByModel?.[modelKey] ?? jobsByModel?.[model.key] ?? jobsByModel?.[model.hub_id]
  const installed = model.status === 'installed'
  const onDisk = installed || model.status === 'partial'
  const downloading = job && (job.status === 'queued' || job.status === 'running')
  const dlLabel = installed
    ? '↻ Re-download'
    : model.status === 'partial'
      ? '⬇ Resume download'
      : '⬇ Download'
  const onlineWorkers = workers.filter(w => w.status === 'online')
  const arch = archiveMark(model)
  const archText = arch ? archiveText(arch) : ''
  const archNote = archiveNote[modelKey]

  return (
    <tr className={`mt-detail-row${arch ? ' mt-row-archived' : ''}`}>
      <td colSpan={colSpan}>
        <div className="mt-detail">
          <div className="mt-detail-actions">
            <button
              className="mt-act"
              disabled={!installed || !isChattable(model)}
              title={!installed ? 'Install model first'
                : !isChattable(model) ? 'Not a chat model' : 'Open chat'}
              onClick={() => onChat(modelKey)}
            >
              💬 Chat
            </button>

            <button
              className={`mt-act${logsOpen[modelKey] ? ' mt-act-on' : ''}`}
              title="Load failures, routing refusals, integrity verdict and admission reason for this model"
              onClick={() => setLogsOpen(o => ({ ...o, [modelKey]: !o[modelKey] }))}
            >
              📜 Logs
            </button>

            <button
              className="mt-act"
              disabled={downloading}
              onClick={() => onDownload(modelKey)}
            >
              {downloading ? '… downloading' : dlLabel}
            </button>

            {downloading && (
              <button className="mt-act mt-act-danger" onClick={() => onCancel(job.id)}>
                ✕ Cancel download
              </button>
            )}

            <button
              className="mt-act mt-act-danger"
              disabled={!onDisk || downloading}
              title={onDisk ? 'Remove downloaded files from disk' : 'Nothing downloaded'}
              onClick={() => onDelete(modelKey)}
            >
              🗑 Delete files
            </button>

            {arch ? (
              <button
                className="mt-act"
                disabled={!!archNote?.busy}
                title={`${archText} — clear the mark (the model returns to placement and routing)`}
                onClick={() => setArchive(modelKey, false)}
              >
                ↩ Unarchive
              </button>
            ) : (
              <button
                className="mt-act mt-act-danger"
                disabled={!!archNote?.busy}
                title="Mark for archive: central stops placing/routing it now; `hugpy-model-archive --apply` later moves it to the ARCHIVE"
                onClick={() => setArchive(modelKey, true)}
              >
                🗄 Archive
              </button>
            )}

            {!onDisk && onPrune && (
              <button
                className="mt-act mt-act-danger"
                disabled={downloading}
                title="Remove this not-installed model from the registry (clears the ghost entry)"
                onClick={() => onPrune(modelKey)}
              >
                ⊘ Prune entry
              </button>
            )}
          </div>

          {(arch || archNote?.text) && (
            <div className="mt-archive-note">
              {arch && <span>🗄 {archText}</span>}
              {archNote?.text && <span className="mt-serve-msg">{archNote.text}</span>}
            </div>
          )}

          {logsOpen[modelKey] && <ModelLogs modelKey={modelKey} model={model} />}

          {statusOn && <ModelStatusDetail modelKey={modelKey} row={statusFor(sIndex, model)} model={model} />}

          {(() => {
            const bm = modelMeta[modelKey]
            if (!bm) return null
            return (
              <div className="mt-meta-strip"
                   title="From GET /models/<key>/meta — the single model-metadata source">
                {bm.size_bytes == null && (() => {
                  const sv = sizeView(model, fmtBytes)
                  return <span className="mt-meta-chip" title={sv.title}>💾 {sv.text}</span>
                })()}
                {bm.size_bytes != null && (
                  <span className="mt-meta-chip"
                        title={bm.effective_gguf
                          ? `Effective quant ${bm.effective_gguf}${bm.mmproj_bytes ? ' + mmproj projector' : ''} — the one file that actually serves. Disk holds ${fmtBytes(bm.dir_bytes)} across all downloaded variants.`
                          : 'On-disk model footprint'}>
                    💾 {fmtBytes(bm.size_bytes)}
                    {bm.dir_bytes != null && bm.dir_bytes > bm.size_bytes * 1.05 && (
                      <em className="mt-meta-sub"> of {fmtBytes(bm.dir_bytes)} on disk</em>
                    )}
                  </span>
                )}
                {bm.quant && <span className="mt-meta-chip">{bm.quant}</span>}
                {bm.params_b != null && <span className="mt-meta-chip">{bm.params_b}B params</span>}
                {bm.ctx_max != null && <span className="mt-meta-chip">ctx {fmtCtx(bm.ctx_max)}</span>}
                {bm.recommended?.threads != null && <span className="mt-meta-chip">threads {bm.recommended.threads}</span>}
                {bm.recommended?.need_bytes != null && (
                  <span className="mt-meta-chip" title="Estimated VRAM need incl. context/overhead">
                    needs ≈{fmtBytes(bm.recommended.need_bytes)} VRAM
                  </span>
                )}
              </div>
            )
          })()}

          {onlineWorkers.length > 0 && (
            <div className="mt-worker-section">
              <div className="mt-serve-title">🖧 Run on worker</div>
              {onlineWorkers.map(w => {
                const serving = (w.models || []).includes(modelKey)
                const free = w.gpus?.[0]?.memory_free
                const need = model.effective_bytes ?? model.total_bytes
                const tight = (free != null && need != null && need > free)
                const pk = `${w.id}:${modelKey}`
                const pr = probe[pk]
                const wm = modelMeta[`${modelKey}@${w.id}`]
                const fit = wm?.recommended
                return (
                  <div key={w.id} className="mt-worker-row">
                    <button
                      className={serving ? 'mt-worker-on' : ''}
                      disabled={!!arch}
                      title={arch ? archText
                        : serving ? 'Already assigned — click to keep' : 'Assign this model to this worker'}
                      onClick={() => { onAssignWorker?.(w, modelKey) }}
                    >
                      {serving ? '✓ ' : '+ '}{w.name}
                      <span className="mt-worker-vram">
                        {free != null ? `${fmtBytes(free)} free` : 'GPU ?'}
                        {tight && ' ⚠'}
                      </span>
                    </button>
                    {/* Per-worker quant (GGUF) with its own fit on THIS worker;
                        a non-GGUF model shows its single artifact + size. */}
                    <WorkerQuantSelect modelKey={modelKey} worker={w} model={model}
                                       disabled={!!arch} disabledTitle={archText} />
                    {fit && !isGgufFw(model.framework) && (
                      <span className={`mt-worker-fit ${fit.fits_vram ? 'mt-fit-ok'
                        : fit.fits_vram === false ? 'mt-fit-partial' : ''}`}
                            title={fit.reason || ''}>
                        {fit.fits_vram ? '✓ fits in VRAM'
                          : fit.fits_vram === false
                            ? `◐ ${fit.gpu_fraction != null ? Math.round(fit.gpu_fraction * 100) + '% on GPU' : 'partial offload'}`
                            : ''}
                      </span>
                    )}
                    <button
                      className="mt-worker-probe"
                      title={arch ? archText : 'Load the model on this GPU and report whether it fits'}
                      disabled={!!arch || pr === 'probing'}
                      onClick={async () => {
                        setProbe(p => ({ ...p, [pk]: 'probing' }))
                        const res = await onProbeWorker?.(w, modelKey)
                        setProbe(p => ({ ...p, [pk]: res || { ok: false } }))
                      }}
                    >
                      {pr === 'probing' ? '…'
                        : pr ? (pr.fit ? '✓ fits' : (pr.ok ? '◐ spills' : '✗'))
                        : 'probe'}
                    </button>
                    {pr && pr !== 'probing' && (
                      <span className="mt-worker-probe-detail" title={pr.error || ''}>
                        {pr.vram_used != null ? `used ${fmtBytes(pr.vram_used)}` : (pr.error ? 'error' : '')}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {/* Which quantization this model IS — a first-class per-model choice,
              above (and independent of) the serving-mode config. */}
          <div className="mt-serve-section mt-quant-block">
            <QuantControl modelKey={modelKey} framework={model.framework} onChanged={onRefresh} />
          </div>

          {/* k56 — WHERE it runs (ordered worker preference) and HOW politely
              (never evict). Model-scoped, so it sits beside the per-model
              serving config rather than inside a single worker's row. */}
          <div className="mt-serve-section">
            <div className="mt-serve-title">🖧 Worker preference &amp; polite load</div>
            <PlacementControl modelKey={modelKey} workers={workers} archived={arch} />
          </div>

          <div className="mt-serve-section">
            <div className="mt-serve-title">Serving (GPU / CPU / context)</div>
            <ServingControl modelKey={model.model_key ?? model.key ?? model.hub_id} framework={model.framework} />
          </div>

          <div className="mt-detail-grid">
            {rows.map(([label, val]) => (
              <div className="mt-detail-item" key={label}>
                <span className="mt-detail-label">{label}</span>
                <span className="mt-detail-value">{String(val)}</span>
              </div>
            ))}
          </div>

          {model.tags?.length > 0 && (
            <div className="mt-detail-tags">
              {model.tags.map(t => <span className="mt-detail-tag" key={t}>{t}</span>)}
            </div>
          )}

          {extras.length > 0 && (
            <details className="mt-detail-raw">
              <summary>Other fields ({extras.length})</summary>
              <div className="mt-detail-grid">
                {extras.map(([k, v]) => (
                  <div className="mt-detail-item" key={k}>
                    <span className="mt-detail-label">{k}</span>
                    <span className="mt-detail-value">{String(v)}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </td>
    </tr>
  )
}
  return (
    <div className="model-table-panel">
      <div className="model-table-toolbar">
        {/* Primary row: the search box with its clear button right beside it,
            then the result count (+ a de-emphasized reload). */}
        <div className="model-table-filters">
          <input
            className="model-table-search"
            placeholder="filter by name, repo, task, framework…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <button
            className="btn-clear-filters"
            type="button"
            onClick={clearFilters}
            title="Clear the search and all filters"
          >
            clear
          </button>

          <span className="model-table-count">
            {visibleModels.length} / {models.length}
          </span>

          {/* MODEL GROUPS: only offered when there is actually a group to show,
              so a fleet with no multi-iteration model never sees the control. */}
          {groups.multiMember.length > 0 && (
            <label className="mt-groupby" title={
              'Pull iterations of the same base model under one group header'
              + (groups.enabled ? '' : `\n\n${groups.offHint}`)}>
              <input type="checkbox" checked={!!groupBy}
                     onChange={e => setGroupBy(e.target.checked)} />
              <span>group{groups.enabled ? '' : ' (off)'}</span>
            </label>
          )}
          {groups.notice && (
            <span className="mt-group-notice" title={groups.notice}
                  onClick={groups.clearNotice} role="alert">
              🔑 {groups.notice}
            </span>
          )}

          {/* Reload is only needed to pick up changes made OUTSIDE this view —
              everything done here (download / delete / prune / finished jobs)
              already refreshes the list on its own. So it's a quiet icon, not a
              primary control. */}
          {onRefresh && (
            <button
              className="model-table-refresh"
              type="button"
              onClick={onRefresh}
              title="Reload the model list (only needed for changes made outside this console)"
              aria-label="Reload model list"
            >
              ↻
            </button>
          )}

          {/* Discover re-registers models that exist on disk but fell out of
              the list (e.g. after a degraded-storage walk). Heavier than
              reload — it re-walks the whole model tree server-side. */}
          <button
            className="model-table-discover"
            type="button"
            onClick={startDiscover}
            disabled={discover.running}
            title="Re-scan model storage on disk and re-register anything missing from this list (can take a few minutes)"
          >
            {discover.running ? 'Discovering…' : 'Discover models'}
          </button>
          {discover.error && (
            <span className="model-table-discover-err" title={discover.error}>
              discover failed
            </span>
          )}
        </div>

        {/* Secondary row: the filter dropdowns, underneath the search. */}
        <div className="model-table-subfilters">
          <select value={frameworkFilter} onChange={e => setFrameworkFilter(e.target.value)}>
            <option value="">Any framework</option>
            {frameworks.map(f => <option key={f} value={f}>{f}</option>)}
          </select>

          <select value={taskFilter} onChange={e => setTaskFilter(e.target.value)}>
            <option value="">All tasks</option>
            {tasks.map(t => <option key={t} value={t}>{t}</option>)}
          </select>

          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="">Any status</option>
            {statuses.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <label className="model-table-assigned" title="Show only models currently assigned to a worker">
            <input
              type="checkbox"
              checked={assignedOnly}
              onChange={e => setAssignedOnly(e.target.checked)}
            />
            assigned{assignedKeys.size ? ` (${assignedKeys.size})` : ''}
          </label>
          {statusOn && Object.entries(STATUS_FILTER_OPTIONS).filter(([k]) => k !== 'worth' && k !== 'bucket').map(([k, opts]) => (
            <select key={k} value={sfilters[k]} title={`status filter: ${k}`}
                    onChange={e => setSfilters(f => ({ ...f, [k]: e.target.value }))}>
              <option value="">{{ ver: 'Any verification', adm: 'Any admission', grade: 'Any grade', fail: 'Any failures', serv: 'Any servability' }[k]}</option>
              {opts.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          ))}
        </div>

        {/* WORTH MY TIME — counts per label over the whole catalog; a click
            filters (URL-backed). The three buckets are the one-click answers. */}
        {statusOn ? (
          <div className="ms-summary" role="group" aria-label="Worth summary">
            {Object.keys(BUCKETS).map(b => (
              <button key={b} type="button" className={`ms-bucket ms-bucket-${b}${sfilters.bucket === b ? ' on' : ''}`}
                      aria-pressed={sfilters.bucket === b}
                      title={`${BUCKET_LABELS[b]}: ${BUCKETS[b].join(' + ')}`}
                      onClick={() => setSf('bucket', b)}>
                {BUCKET_LABELS[b]} <b>{summary.buckets[b]}</b>
              </button>
            ))}
            <span className="ms-summary-sep" />
            {WORTH_ORDER.map(l => (
              <button key={l} type="button" className={`ms-chip ms-${WORTH_META[l].tone} ms-lab${sfilters.worth === l ? ' on' : ''}`}
                      aria-pressed={sfilters.worth === l} onClick={() => setSf('worth', l)}>
                {WORTH_META[l].text} {summary.counts[l]}
              </button>
            ))}
            {hasStatusFilters(sfilters) && (
              <button type="button" className="btn-clear-filters" onClick={() => setSfilters({ ...EMPTY_STATUS_FILTERS })}>
                clear status filters
              </button>
            )}
            {mstatus.error && <span className="ms-note">status poll: {mstatus.error} (showing last good)</span>}
            {Object.entries(sIndex.sources || {}).filter(([, v]) => v && v.error).map(([k, v]) => (
              <span key={k} className="ms-note" title={v.error}>{k} source read failed: {v.error}</span>
            ))}
          </div>
        ) : mstatus.loaded && (
          <div className="ms-summary ms-note">
            {mstatus.unsupported
              ? `Worth / verification / grade columns empty: ${mstatus.unsupportedWhy || 'GET /llm/models/status not served by this central'}.`
              : `GET /llm/models/status failed: ${mstatus.error || 'no response body and no error recorded by the poller'}`}
          </div>
        )}
      </div>

      <div className="table-wrap">
        {visibleModels.length === 0 ? (
          <div className="empty">No models match the current filters.</div>
        ) : (
          <table className="model-table">
            <thead>
              <tr>
                {columns.map(c => (
                  <th
                    key={c.key}
                    className="mt-sortable"
                    onClick={() => toggleSort(c.key)}
                    title={`Sort by ${c.label}`}
                  >
                    {c.label}
                    {sortKey === c.key && (
                      <span className="mt-sort-arrow">
                        {sortDir === 'asc' ? ' ▲' : ' ▼'}
                      </span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {groupedRows.map((row, i) => {
              if (row.kind === 'pgroup') {
                const g = row.pgroup
                const isCollapsed = !!collapsed?.[`pg:${g.id}`]
                return (
                  <tr key={`pgroup:${g.id}`} className="mt-pgroup-row">
                    <td colSpan={columns.length}>
                      <div className="mt-pgroup-bar">
                        <button type="button" className="mt-pgroup-caret"
                                aria-expanded={!isCollapsed}
                                title={isCollapsed ? 'Expand group' : 'Collapse group'}
                                onClick={() => toggleCollapse(`pg:${g.id}`)}>
                          {isCollapsed ? '▸' : '▾'}
                        </button>
                        <span className="mt-pgroup-name" title={`priority group ${g.id}`}>
                          ▣ {g.name}
                        </span>
                        <span className="mt-pgroup-count">{row.count} model{row.count === 1 ? '' : 's'}</span>
                        {(g.workers || []).length > 0 && (
                          <span className="mt-pgroup-workers"
                                title="The group's worker allocation, in priority order">
                            → {g.workers.join(' → ')}
                          </span>
                        )}
                        <span className="mt-pgroup-spacer" />
                        {pgAllocNote?.id === g.id && (
                          <span className="mt-pgroup-note">{pgAllocNote.text}</span>
                        )}
                        <select className="mt-pgroup-select" value=""
                                title="Designate every model in this group to one worker"
                                onChange={e => { if (e.target.value) allocatePgroup(g, e.target.value); e.target.value = '' }}>
                          <option value="">assign group to worker…</option>
                          {workers.map(w => (
                            <option key={w.id} value={w.name || w.id}>{w.name || w.id}</option>
                          ))}
                        </select>
                        {(g.workers || []).length > 0 && (
                          <button type="button" className="mt-pgroup-allocate"
                                  title={`Designate every model to ${g.workers.join(', ')}`}
                                  onClick={() => allocatePgroup(g, null)}>
                            allocate
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              }
              if (row.kind === 'group') {
                return (
                  <GroupHeaderRow
                    key={`group:${row.group.group_key}`}
                    group={row.group}
                    colSpan={columns.length}
                    enabled={groups.enabled}
                    offHint={groups.offHint}
                    collapsed={!!collapsed?.[row.group.group_key]}
                    onToggleCollapse={() => toggleCollapse(row.group.group_key)}
                    onFlip={(tick, next) => groups.setTick(row.group.group_key, tick, next)}
                  />
                )
              }
              const m = row.model
              const rowKey = m.model_key ?? m.key ?? m.hub_id ?? m.name ?? `model-${i}`
              const modelKey = m.model_key ?? m.key ?? rowKey

              const job = jobsByModel?.[modelKey] ?? jobsByModel?.[rowKey]
              const isActive = modelKey === activeChat || rowKey === activeChat

              return (
  <Fragment key={rowKey}>
    <tr className={`${isActive ? 'row-active' : ''}${detailKey === rowKey ? ' row-expanded' : ''}`
      + (row.group ? ' mt-group-member' : '') + (archiveMark(m) ? ' mt-row-archived' : '')}
        title={archiveMark(m) ? archiveText(archiveMark(m)) : undefined}>
      <td className="col-num">
        <button
          type="button"
          className="mt-num-toggle"
          aria-expanded={detailKey === rowKey}
          title={detailKey === rowKey ? 'Collapse details' : 'Expand details & actions'}
          onClick={() => {
            const expanding = detailKey !== rowKey
            setDetailKey(expanding ? rowKey : null)
            if (expanding) loadModelMeta(modelKey,
              workers.filter(w => w.status === 'online').map(w => w.id))
          }}
        >
          {i + 1}
        </button>
      </td>

      <td className="col-name">
        <span
          className={`model-name${m.hub_id ? ' model-name-link' : ''}`}
          title={m.hub_id ? `${m.hub_id} — double-click to open on Hugging Face` : m.name}
          onDoubleClick={() => {
            if (m.hub_id) window.open(`https://huggingface.co/${m.hub_id}`, '_blank', 'noopener,noreferrer')
          }}
        >
          {m.name ?? m.key}
        </span>
        <span className="hub-id">{m.hub_id}</span>
        {archiveMark(m) && (
          <span className="mt-archive-tag">🗄 {archiveText(archiveMark(m))}</span>
        )}
        {!m.effective_gguf && (() => {
          const sv = sizeView(m, fmtBytes)
          return <span className={`mt-size-tag${sv.known ? '' : ' mt-size-unknown'}`} title={sv.title}>💾 {sv.text}</span>
        })()}
        {/* MODEL GROUPS: what this iteration does on each worker, or why it
            lost there. Only rendered for a grouped member row. */}
        {row.group && <MemberVerdicts group={row.group} modelKey={modelKey} />}
        {m.effective_gguf && (
          <button type="button" className="mt-quant-badge"
                  title={`Serving quant: ${m.effective_gguf}`
                    + (m.effective_bytes != null ? ` · ${fmtBytes(m.effective_bytes)}` : '')
                    + ((m.gguf_variants?.length || 0) > 1
                        ? ` · ${m.gguf_variants.length} variants downloaded — click to choose`
                        : ' — click for details')}
                  onClick={() => {
                    const expanding = detailKey !== rowKey
                    setDetailKey(expanding ? rowKey : null)
                    if (expanding) loadModelMeta(modelKey,
                      workers.filter(w => w.status === 'online').map(w => w.id))
                  }}>
            🧩 {quantTag(m.effective_gguf)}
            {m.effective_bytes != null ? ` · ${fmtBytes(m.effective_bytes)}` : ''}
            {(m.gguf_variants?.length || 0) > 1 ? ' ▾' : ''}
          </button>
        )}
      </td>

                    <td className="col-media">
                      {isChattable(m) ? (
                        <span className="media-cell">
                          <input
                            type="checkbox"
                            className="media-check"
                            checked={!!m.media}
                            title="Offer this model in the media-intelligence chat dropdown"
                            onChange={e => onSetMedia?.(modelKey, e.target.checked)}
                          />
                          {m.media && (
                            <button
                              type="button"
                              className={`media-default-btn${m.media_default ? ' is-default' : ''}`}
                              aria-pressed={!!m.media_default}
                              title={m.media_default
                                ? 'Default media model — shown first in the media chat list'
                                : 'Set as the default media model (first in the media chat list)'}
                              onClick={() => onSetMediaDefault?.(modelKey, !m.media_default)}
                            >
                              {m.media_default ? '★' : '☆'}
                            </button>
                          )}
                        </span>
                      ) : (
                        <span className="media-na" title="Chat-capable models only">—</span>
                      )}
                    </td>

                    <td>
                      <span className={`fw-tag fw-${m.framework}`}>{m.framework}</span>
                    </td>

                    <td className="col-task" title={modelTasks(m).join(', ')}>
                      {modelTask(m)}{modelTasks(m).length > 1 ? ` +${modelTasks(m).length - 1}` : ''}
                    </td>

                    <td className="col-ctx">{fmtCtx(m.model_max_length)}</td>

                    <td className="col-status">
                      {job && job.status !== 'completed' ? (
                        <DownloadProgress job={job} onCancel={onCancel} onRetry={onRetry} />
                      ) : (
                        <StatusBadge status={m.status} />
                      )}
                    </td>
                    <td className="col-pgroup">
                      <select className="mt-pgroup-member-select"
                              value={pgFor(m)?.id || ''}
                              title="Explicit priority group — change to move this model between groups"
                              onChange={e => setModelGroup(modelKey, e.target.value)}>
                        <option value="">—</option>
                        {pgroups.map(g => (
                          <option key={g.id} value={g.id}>{g.name}</option>
                        ))}
                      </select>
                    </td>
                    {statusOn && STATUS_COLUMNS.map(c => (
                      <td key={c.key} className={`ms-td ms-td-${c.key}`}><c.Cell row={m.__status} /></td>
                    ))}
                   </tr>

    {detailKey === rowKey && renderModelDetail(m, columns.length)}
  </Fragment>
)
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}