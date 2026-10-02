// Pure model-index / search / filter / selection logic for the Metrics panel.
//
// No React, no fetch, no DOM: everything here is plain data in, plain data out,
// so it runs unchanged under `node --test` (modelFilters.test.mjs) and in the
// webpack bundle. MetricsPanel.jsx owns the effects; this file owns the rules.

import { FALLBACK_SUITES, matrixFor, resolveSuite } from './suites.js'

// The text suite's tasks (kept for callers that name the text suite); every
// suite's tasks come from suites.js / GET /llm/benchmark/suites.
export const BENCH_TASKS = FALLBACK_SUITES['hugpy-native-v2'].tasks
export const ALL_SUITE_TASKS = [...new Set(Object.values(FALLBACK_SUITES).flatMap(s => s.tasks))]

// Grade buckets over the 0..100 aptitude grade. 'na' = never graded.
export const GRADE_BUCKETS = [
  ['na', 'N/A'], ['zero', '0'], ['lt50', '<50'], ['mid', '50–89'], ['high', '≥90'],
]

export const FILTER_KEYS = ['grade', 'pf', 'task', 'cat', 'fw', 'worker', 'verdict', 'adm', 'untested']
export const EMPTY_FILTERS = Object.freeze({
  // task = the MODEL's task (text-generation, image-text-to-text, …) — the same word as everywhere else.
  // cat  = a grade CATEGORY of the suite (math, coding, …), never called a task.
  grade: [], pf: '', task: '', cat: '', fw: '', worker: '', verdict: '', adm: '', untested: false,
})

export const parseDetail = (v) => {
  if (v == null || v === '') return null
  if (typeof v === 'object') return v
  try { return JSON.parse(v) } catch { return null }
}

const num = (v) => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v))

export function gradeBucket(grade) {
  const g = num(grade)
  if (g == null) return 'na'
  if (g <= 0) return 'zero'
  if (g < 50) return 'lt50'
  if (g < 90) return 'mid'
  return 'high'
}

// One task cell of a grade_detail -> 'pass' | 'fail' | 'partial' | null.
// Current suite: {tier, max:3, history:[{tier, pass}]} (every tier is tried
// independently, so one false == a failed tier). Legacy suites: bool or 0/1.
export function taskStatus(value) {
  if (value == null) return null
  if (typeof value === 'boolean') return value ? 'pass' : 'fail'
  if (typeof value === 'number') return value > 0 ? 'pass' : 'fail'
  if (typeof value !== 'object') return null
  const history = Array.isArray(value.history) ? value.history : []
  if (history.some(h => h && h.pass === false)) return 'fail'
  const max = Number(value.max) || 3
  if (history.length >= max && history.every(h => h && h.pass === true)) return 'pass'
  const tier = num(value.tier)
  if (!history.length && tier != null) return tier >= max ? 'pass' : 'fail'
  return history.length ? 'partial' : null
}

// Admission (being added server-side): a string or {status, reason}. Read
// defensively; absent -> null.
export function readAdmission(row) {
  const a = row && (row.admission ?? row.admission_status)
  if (a == null || a === '') return null
  if (typeof a === 'string') return { status: a, reason: row.admission_reason || '' }
  if (typeof a === 'object' && a.status) return { status: String(a.status), reason: a.reason || '' }
  return null
}

// Metrics rows carry alias forms (`owner~name` beside the catalog key). Map
// every form we can recognise onto the catalog model_key.
export function makeCanon(catalogRows = [], v1Models = []) {
  const alias = new Map()
  for (const c of catalogRows) {
    const key = c && (c.model_key || c.name)
    if (!key) continue
    alias.set(key, key)
    if (c.hub_id) { alias.set(c.hub_id, key); alias.set(c.hub_id.replace('/', '~'), key) }
  }
  for (const m of v1Models) {
    const id = typeof m === 'string' ? m : m && m.id
    if (!id) continue
    if (!alias.has(id)) alias.set(id, id)
    const hub = typeof m === 'object' && m.hub_id
    if (hub && !alias.has(hub.replace('/', '~'))) alias.set(hub.replace('/', '~'), alias.get(id))
  }
  return (name) => (name == null ? name : alias.get(name) || name)
}

const newer = (a, b) => (num(a?.graded_at) || 0) >= (num(b?.graded_at) || 0)

// Per-model summary used by the picker chips, the filters and the "why" line.
export function buildModelIndex({ metricsRows = [], catalogRows = [], v1Models = [], extraModels = [] } = {}) {
  const canon = makeCanon(catalogRows, v1Models)
  const infos = new Map()
  const get = (key) => {
    if (!infos.has(key)) {
      infos.set(key, { key, hub_id: '', framework: '', primary_task: '', quants: new Set(),
        workers: new Set(), aptitude: null, integrity: null, tasks: {}, admission: null,
        blocked: false, unserveable_reason: '', inCatalog: false, rows: 0 })
    }
    return infos.get(key)
  }
  for (const c of catalogRows) {
    const key = c && (c.model_key || c.name); if (!key) continue
    const info = get(key)
    Object.assign(info, { hub_id: c.hub_id || '', framework: c.framework || '',
      primary_task: c.primary_task || '', inCatalog: true, blocked: !!c.blocked,
      admission: readAdmission(c) })
    if (c.effective_gguf) info.quants.add(c.effective_gguf)
  }
  for (const m of v1Models) {
    const id = typeof m === 'string' ? m : m && m.id; if (!id) continue
    const info = get(canon(id))
    info.inCatalog = true
    if (typeof m === 'object') {
      if (!info.hub_id && m.hub_id) info.hub_id = m.hub_id
      if (!info.primary_task && m.task) info.primary_task = m.task
      if (m.blocked) info.blocked = true
      if (m.unserveable_reason) info.unserveable_reason = String(m.unserveable_reason)
      if (!info.admission) info.admission = readAdmission(m)
    }
  }
  for (const r of metricsRows) {
    if (!r || !r.model_name) continue
    const info = get(canon(r.model_name))
    info.rows++
    if (r.quant) info.quants.add(r.quant)
    if (r.worker) info.workers.add(r.worker)
    if (num(r.grade) == null) continue
    const detail = parseDetail(r.grade_detail)
    if (r.grade_suite === 'integrity') {
      const cand = { grade: num(r.grade), graded_at: r.graded_at, verdict: detail?.verdict || '',
        why: detail?.why || '', log: Array.isArray(detail?.log) ? detail.log : [] }
      if (!info.integrity || newer(cand, info.integrity)) info.integrity = cand
    } else {
      const cand = { grade: num(r.grade), suite: r.grade_suite || '', graded_at: r.graded_at, detail }
      if (!info.aptitude || newer(cand, info.aptitude)) info.aptitude = cand
    }
  }
  for (const k of extraModels) if (k) get(canon(k))
  for (const info of infos.values()) {
    const d = info.aptitude?.detail
    info.tasks = {}
    if (d && typeof d === 'object') {
      const suite = resolveSuite({ gradeSuite: info.aptitude?.suite, detail: d })
      info.aptitude.suiteDef = suite
      for (const t of suite.tasks) info.tasks[t] = taskStatus(d[t])
    }
  }
  return { canon, infos }
}

// Pass/fail over one model. `cat` narrows to that grade-category column only.
export function passFail(info, cat = '') {
  if (!info) return null
  if (cat) return info.tasks[cat] === 'fail' ? 'fail' : info.tasks[cat] === 'pass' ? 'pass' : null
  const vals = Object.values(info.tasks).filter(Boolean)
  if (vals.includes('fail') || info.integrity?.grade === 0) return 'fail'
  if ((vals.length && vals.every(v => v === 'pass')) || (!vals.length && info.integrity?.grade === 100)) return 'pass'
  return null
}

export const isUntested = (info) => !info || info.aptitude == null

export function haystack(info) {
  return [info.key, info.hub_id, info.framework, info.primary_task, ...info.quants,
    info.integrity?.verdict, info.admission?.status].filter(Boolean).join(' ').toLowerCase()
}

export function matchesSearch(info, query) {
  const q = String(query || '').trim().toLowerCase()
  if (!q) return true
  const hay = haystack(info)
  return q.split(/\s+/).every(tok => hay.includes(tok))
}

export function matchesFilters(info, f = EMPTY_FILTERS) {
  if (!info) return false
  if (f.grade?.length && !f.grade.includes(gradeBucket(info.aptitude?.grade))) return false
  if (f.cat && !f.pf && !info.tasks[f.cat]) return false
  if (f.task && (info.primary_task || '') !== f.task) return false
  if (f.pf && passFail(info, f.cat) !== f.pf) return false
  if (f.fw && info.framework !== f.fw) return false
  if (f.worker && !info.workers.has(f.worker)) return false
  if (f.verdict && (info.integrity?.verdict || '') !== f.verdict) return false
  if (f.adm && (info.admission?.status || '') !== f.adm) return false
  if (f.untested && !isUntested(info)) return false
  return true
}

export const hasActiveFilters = (f) =>
  !!(f && (f.grade?.length || f.pf || f.task || f.cat || f.fw || f.worker || f.verdict || f.adm || f.untested))

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Search + filter + rank: exact key, key prefix, whole query inside the key,
// every token at a word boundary of the key ("qwen 3b" -> Qwen2.5-3B before
// Qwen3.6-35B-A3B), every token in the key, then hub/framework/quant-only
// matches; alphabetical within a rank.
export function filterModels(infos, { query = '', filters = EMPTY_FILTERS } = {}) {
  const q = String(query || '').trim().toLowerCase()
  const toks = q ? q.split(/\s+/) : []
  const bounds = toks.map(t => new RegExp(`(^|[^a-z0-9])${escapeRe(t)}`))
  const list = [...(infos instanceof Map ? infos.values() : infos)]
    .filter(i => matchesFilters(i, filters) && matchesSearch(i, q))
  const rank = (i) => {
    if (!q) return 0
    const k = i.key.toLowerCase()
    if (k === q) return 0
    if (k.startsWith(q)) return 1
    if (k.includes(q)) return 2
    if (bounds.every(re => re.test(k))) return 3
    if (toks.every(t => k.includes(t))) return 4
    return 5
  }
  return list.map(i => [rank(i), i]).sort((a, b) => a[0] - b[0] || a[1].key.localeCompare(b[1].key)).map(([, i]) => i.key)
}

// Per-row filter for the model_metrics sheet: the model must pass the model
// filters and, when a worker filter is set, the row must be on that worker.
export function filterRows(rows, infos, canon, f = EMPTY_FILTERS) {
  if (!hasActiveFilters(f)) return rows
  return rows.filter(r => (!f.worker || r.worker === f.worker) && matchesFilters(infos.get(canon(r.model_name)), f))
}

// Distinct option values for the filter dropdowns.
export function filterOptions(infos) {
  const fw = new Set(), workers = new Set(), verdicts = new Set(), adm = new Set(), tasks = new Map()
  for (const i of infos.values()) {
    if (i.framework) fw.add(i.framework)
    tasks.set(i.primary_task || '(no task recorded)', (tasks.get(i.primary_task || '(no task recorded)') || 0) + 1)
    i.workers.forEach(w => workers.add(w))
    if (i.integrity?.verdict) verdicts.add(i.integrity.verdict)
    if (i.admission?.status) adm.add(i.admission.status)
  }
  const s = (x) => [...x].sort()
  return { fw: s(fw), workers: s(workers), verdicts: s(verdicts), adm: s(adm),
    tasks: [...tasks.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([task, count]) => ({ task, count })) }
}

// Status chips for one model: [{label, tone}] with tone ok|bad|warn|muted.
export function statusChips(info) {
  if (!info) return []
  const out = []
  if (info.aptitude) {
    const g = info.aptitude.grade
    const suite = info.aptitude.suiteDef || resolveSuite({ gradeSuite: info.aptitude.suite, detail: info.aptitude.detail })
    const mx = info.aptitude.detail ? matrixFor(info.aptitude.detail, suite) : null
    out.push({ label: mx && mx.recorded ? `graded ${mx.text}`
      : `graded ${Math.round(g * suite.max / 100)}/${suite.max} (${Math.round(g)}%)`, tone: g >= 50 ? 'ok' : g > 0 ? 'warn' : 'bad' })
  } else out.push({ label: 'untested', tone: 'muted' })
  if (info.integrity) out.push({ label: info.integrity.verdict || (info.integrity.grade ? 'integrity ok' : 'integrity fail'),
    tone: info.integrity.grade === 0 ? 'bad' : 'ok' })
  if (info.admission && info.admission.status !== 'admitted') out.push({ label: info.admission.status, tone: 'warn' })
  if (info.blocked) out.push({ label: 'blocked', tone: 'bad' })
  return out
}

// ---- URL query <-> state ---------------------------------------------------

export function parseQuery(search) {
  const q = new URLSearchParams(search || '')
  const grades = (q.get('grade') || '').split(',').filter(g => GRADE_BUCKETS.some(([k]) => k === g))
  const pf = ['pass', 'fail'].includes(q.get('pf')) ? q.get('pf') : ''
  const cat = ALL_SUITE_TASKS.includes(q.get('cat')) ? q.get('cat') : ''
  return {
    model: q.get('model') || '',
    filters: { grade: grades, pf, task: q.get('task') || '', cat, fw: q.get('fw') || '', worker: q.get('worker') || '',
      verdict: q.get('verdict') || '', adm: q.get('adm') || '', untested: q.get('untested') === '1' },
  }
}

// New search string: `model` + filters written, every unrelated param kept.
export function writeQuery(search, { model, filters = EMPTY_FILTERS }) {
  const q = new URLSearchParams(search || '')
  const set = (k, v) => { if (v) q.set(k, v); else q.delete(k) }
  set('model', model || '')
  set('grade', (filters.grade || []).join(','))
  for (const k of ['pf', 'task', 'cat', 'fw', 'worker', 'verdict', 'adm']) set(k, filters[k] || '')
  set('untested', filters.untested ? '1' : '')
  const s = q.toString()
  return s ? `?${s}` : ''
}

// ---- sticky selection -------------------------------------------------------
//
// current  the model in state ('' = nothing yet)
// chosen   true when it came from the user (click, URL, localStorage)
// known    Set of every model id the panel currently knows (catalog + rows)
// ready    the catalog AND the metrics sheet loaded cleanly (non-empty, no
//          error) — the only time "not in the list" means "gone"
// running  benchmark.progress.model while a run is active, else ''
// fallback first model of the visible list
//
// -> { model, chosen, dropped } ; dropped = the chosen id that truly vanished.
export function resolveSelection({ current = '', chosen = false, known = new Set(), ready = false,
  running = '', fallback = '' }) {
  if (current && chosen) {
    if (known.has(current) || !ready) return { model: current, chosen: true, dropped: '' }
    const auto = running && known.has(running) ? running : fallback
    return { model: auto || '', chosen: false, dropped: current }
  }
  // Not user-chosen: follow a running benchmark, else keep/assign the fallback.
  if (running && known.has(running)) return { model: running, chosen: false, dropped: '' }
  if (current && known.has(current)) return { model: current, chosen: false, dropped: '' }
  return { model: fallback || current || '', chosen: false, dropped: '' }
}

// First meaningful line of a load failure / integrity record (the inline
// "why"); the full text goes in the details expander.
export function firstLine(text) {
  const lines = String(text || '').split('\n').map(s => s.trim()).filter(Boolean)
  return lines.find(l => /\berror\b|failed|E llama/i.test(l)) || lines[0] || ''
}

// Metrics rows with model_name folded onto the catalog key, one row per
// (worker, model, quant, alloc_mode) — the newest-updated wins when two alias
// forms carry the same cell (the grade matrix keys rows on exactly that).
export function canonicalRows(rows = [], canon = (x) => x) {
  const out = new Map()
  for (const r of rows) {
    if (!r) continue
    const c = { ...r, model_name: canon(r.model_name) }
    const k = [c.worker, c.model_name, c.quant, c.alloc_mode].join('\0')
    const prev = out.get(k)
    if (!prev || (num(c.updated_at) || 0) > (num(prev.updated_at) || 0)) out.set(k, c)
  }
  return [...out.values()]
}
