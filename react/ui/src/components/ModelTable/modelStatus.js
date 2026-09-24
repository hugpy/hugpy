// "Is this model worth my time?" — pure view logic over GET /llm/models/status.
//
// No React, no fetch, no DOM: data in, data out, so it runs under `node --test`
// (modelStatus.test.mjs) and in the bundle. The server computes the facts
// (model_status_routes.py); this file only turns them into labelled cells,
// filters, sort keys and the "Verify + grade" action state machine.
//
// EXPLICIT MISSING: every view function returns a non-empty `text`. A fact that
// does not exist renders as a labelled state (NOT VERIFIED / NOT GRADED /
// NO GRADER FOR <task> / no failures recorded / ...) with the reason and the
// action that would produce it — never a blank cell, never a fake 0.

export const WORTH_ORDER = ['ready', 'weak', 'no-grader', 'ungraded', 'unverified', 'unservable', 'held', 'broken']
export const WORTH_META = {
  ready: { tone: 'ok', text: 'ready' },
  weak: { tone: 'bad', text: 'weak' },
  'no-grader': { tone: 'muted', text: 'no grader' },
  ungraded: { tone: 'warn', text: 'ungraded' },
  unverified: { tone: 'warn', text: 'unverified' },
  unservable: { tone: 'bad', text: 'unservable' },
  held: { tone: 'bad', text: 'held' },
  broken: { tone: 'bad', text: 'broken' },
}
export const BUCKETS = {
  ready: ['ready'],
  waste: ['broken', 'held', 'unservable', 'weak'],
  unknown: ['unverified', 'ungraded', 'no-grader'],
}
export const BUCKET_LABELS = { ready: 'ready', waste: 'wasting your time', unknown: 'unknown' }

// The per-(model, worker) vocabulary — identical to the server's
// model_worker_state (WORKER_STATES + the failed overlay).
export const WORKER_STATES = ['answering', 'serving', 'hot', 'loading', 'downloading from central', 'cold', 'on central', 'not allocated', 'missing', 'n/a', 'failed']
export const WORKER_STATE_META = {
  answering: { icon: '⚡', tone: 'ok' },
  serving: { icon: '▶', tone: 'ok' },
  hot: { icon: '●', tone: 'ok' },
  loading: { icon: '◐', tone: 'live' },
  'downloading from central': { icon: '⇣', tone: 'live' },
  cold: { icon: '○', tone: 'muted' },
  // on central: files on central's store, not on this worker's drive yet — they
  // copy on first call (lazy download). Not missing: the files exist somewhere.
  'on central': { icon: '◌', tone: 'muted' },
  // not allocated: on no drive and not assigned to this worker — neutral.
  'not allocated': { icon: '·', tone: 'muted' },
  // missing: on no drive anywhere AND allocated here — the only state that needs
  // an operator.
  missing: { icon: '∅', tone: 'muted' },
  'n/a': { icon: '–', tone: 'muted' },
  failed: { icon: '✗', tone: 'bad' },
}
export const LIVE_STATES = ['downloading from central', 'loading', 'serving', 'answering']

const num = (v) => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v))

export function fmtAgo(ts, now = Date.now() / 1000) {
  const t = num(ts)
  if (t == null) return 'never'
  const d = Math.max(0, now - (t > 1e12 ? t / 1000 : t))
  if (d < 60) return `${Math.round(d)}s ago`
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 172800) return `${(d / 3600).toFixed(1)}h ago`
  return `${Math.round(d / 86400)}d ago`
}

export function fmtGB(b) {
  const n = num(b)
  return n == null ? '?' : `${(n / 1e9).toFixed(1)}`
}

// ---- feature detection + index ---------------------------------------------

// The endpoint is new: an older central answers the SPA's index.html (200) or
// 404. "Available" = a JSON object with a models array came back.
export function indexStatus(payload) {
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.models)) {
    return { available: false, byKey: new Map(), counts: {}, buckets: {}, live: false, pollMs: 15000 }
  }
  const byKey = new Map()
  for (const r of payload.models) if (r && r.model_key) byKey.set(r.model_key, r)
  return {
    available: true, byKey, counts: payload.counts || {}, buckets: payload.buckets || {},
    live: !!payload.live, pollMs: pollInterval(payload), sources: payload.sources || {},
    rules: payload.rules || {}, generatedAt: payload.generated_at || null,
  }
}

// <= 3 s while anything is downloading/loading/serving/answering, else 15 s.
export function pollInterval(payload) {
  const rows = Array.isArray(payload?.models) ? payload.models : []
  const live = !!payload?.live || rows.some(r => (r.workers || []).some(w => LIVE_STATES.includes(w.base || w.state)))
  return live ? 3000 : 15000
}

// Look a catalog row up by any spelling it may carry.
export function statusFor(index, model) {
  if (!index?.available || !model) return null
  const keys = typeof model === 'string' ? [model] : [model.model_key, model.key, model.name]
  for (const k of keys) if (k && index.byKey.has(k)) return index.byKey.get(k)
  const tail = String(keys[0] || '').split('~').pop()
  for (const [k, r] of index.byKey) if (k.split('~').pop() === tail) return r
  return null
}

// ---- cell views: {text, tone, title} — text is never empty -----------------

export function worthView(row) {
  if (!row) return { text: 'NO STATUS', tone: 'muted', title: 'no row for this model in GET /llm/models/status' }
  const w = row.worth || {}
  const meta = WORTH_META[w.label] || { tone: 'muted', text: w.label || 'unknown' }
  return { text: meta.text, tone: meta.tone, title: [w.why, w.action && `→ ${w.action}`].filter(Boolean).join('\n'),
    label: w.label, bucket: w.bucket, why: w.why || '', action: w.action || '' }
}

const VERDICT_TONE = { static_ok: 'ok', working: 'ok', suite_mismatch: 'ok', mislabeled_task: 'warn',
  unsupported: 'muted', misconfigured: 'bad', broken_download: 'bad', faulty_model: 'bad', not_downloaded: 'bad' }

export function verificationView(row) {
  const v = row?.verification
  if (!v || v.source === 'none' || !v.verdict) {
    return { text: 'NOT VERIFIED', tone: 'warn', missing: true,
      title: v?.why || (v ? `verification source=${v.source}, no verdict recorded` : 'status row has no verification block') }
  }
  return { text: v.verdict.replace(/_/g, ' '), tone: VERDICT_TONE[v.verdict] || (v.ok ? 'ok' : 'bad'),
    title: `${v.why || `(the ${v.source} verdict carries no why text)`}\nsource: ${v.source}${v.at ? ` · ${fmtAgo(v.at)}` : ''}` }
}

export function admissionView(row) {
  const a = row?.admission
  if (!a || a.status === 'none') {
    return { text: 'NO RECORD', tone: 'warn', missing: true, title: a?.reason || 'status row has no admission block' }
  }
  const tone = a.status === 'admitted' ? 'ok' : a.status === 'held' ? 'bad' : 'warn'
  return { text: a.status, tone, title: `${a.reason || `(admission record${a.job ? ` job ${a.job}` : ''} carries no reason text)`}${a.at ? `\n${fmtAgo(a.at)}` : ''}` }
}

export function gradeView(row) {
  const g = row?.grade
  if (!g) return { text: 'NOT GRADED', tone: 'warn', missing: true, title: 'status row has no grade block' }
  const stray = (g.stray || []).length
    ? `\nignored: ${(g.stray || []).map(s => `${s.suite} ${s.value ?? '?'}`).join(', ')} (suite does not match the task)` : ''
  if (g.suite == null) {
    return { text: (g.reason || 'no grader').toUpperCase(), tone: 'muted', missing: true,
      title: `${g.reason || `no grading suite registered for task ${row?.task ?? '(no task recorded)'}`}${stray}` }
  }
  if (g.value == null) {
    return { text: 'NOT GRADED', tone: 'warn', missing: true,
      title: `${g.reason || `no ${g.suite} grade row recorded`}${stray}` }
  }
  const tone = g.value >= 50 ? 'ok' : g.value > 0 ? 'warn' : 'bad'
  const where = [g.worker, g.quant].filter(Boolean).join(' · ')
  // score/max (pct) — the server counts PASS items over the suite's own max.
  const text = g.text || (g.score != null && g.max ? `${g.score}/${g.max} (${Math.round(g.value)}%)` : `${Math.round(g.value)}%`)
  return { text, tone,
    sub: `${g.suite}${g.legacy ? ' (legacy)' : ''}`,
    title: `${g.suite}${g.legacy ? ' (legacy suite)' : ''} · ${fmtAgo(g.at)}${where ? ` · ${where}` : ''}\n${g.detail_summary || ''}${stray}` }
}

// tok/s = the mean over EVERY recorded call (never an EMA, never the last).
export function throughputView(row) {
  const tp = row?.throughput
  if (!tp || !tp.n_calls) {
    if (tp?.error) return { text: 'call ledger read failed', tone: 'bad', none: true, title: tp.reason || tp.error }
    return { text: 'no calls recorded', tone: 'muted', none: true,
      title: tp?.reason || 'status row has no throughput block' }
  }
  const f = (v) => (v == null ? '?' : Number(v).toFixed(1))
  return { text: `${f(tp.mean_tok_s)} tok/s · avg of ${tp.n_calls}`, tone: 'ok',
    title: `mean of ${tp.n_calls} recorded calls (Σtokens / Σseconds)\np50 ${f(tp.p50)} · p90 ${f(tp.p90)} · min ${f(tp.min)} · max ${f(tp.max)}\nfirst ${fmtAgo(tp.first_at)} · last ${fmtAgo(tp.last_at)}` }
}

export function failureView(row) {
  const f = row?.last_failure
  if (!f) return { text: 'none recorded', tone: 'muted', none: true, title: 'no load-fail or call-refused compute_actions row for this model in the window central reads (last 2000 load fails, last 500 refusals)' }
  return { text: f.first_line || `(${f.kind || 'failure'} row carries no message)`, tone: 'bad', kind: f.kind, cls: f.class,
    title: `${f.kind}${f.class ? ` · ${f.class}` : ''}${f.worker ? ` · ${f.worker}` : ''} · ${fmtAgo(f.ts)}${f.request_id ? `\nrequest ${f.request_id}` : ''}`,
    // the whole log behind the first line (server: last_failure.text, 2026-09-23)
    log: { text: f.text ?? '', source: f.text_source || 'last_failure (this central predates the whole-text field)',
      logRef: f.log_ref || (f.action_id != null ? `compute_actions#${f.action_id}` : ''), bytes: f.bytes } }
}

export function servableView(row) {
  const s = row?.servable
  if (!s) return { text: 'UNKNOWN', tone: 'muted', title: 'status row has no servable block' }
  if (s.now) return { text: `hot: ${s.where.join(', ')}`, tone: 'ok', title: s.reason }
  if ((s.provisioning || []).length) {
    const p = s.provisioning[0]
    return { text: `provisioning ${p.worker}${p.frac != null ? ` ${Math.round(p.frac * 100)}%` : ''}`, tone: 'live', title: s.reason }
  }
  if ((s.cold || []).length) return { text: `cold: ${s.cold.join(', ')}`, tone: 'warn', title: s.reason }
  if ((s.fits || []).length) return { text: `fits: ${s.fits.join(', ')}`, tone: 'warn', title: s.reason }
  return { text: 'not now', tone: 'bad', title: s.reason || 'servable block carries no reason' }
}

export function workerChips(row) {
  return (row?.workers || []).map(w => {
    const meta = WORKER_STATE_META[w.state] || WORKER_STATE_META.missing
    return { worker: w.worker, state: w.state, label: w.label || w.state, icon: meta.icon,
      tone: w.held ? 'bad' : meta.tone, online: w.online !== false,
      title: `${w.worker}: ${w.label || w.state}\n${w.detail || ''}` }
  })
}

// ---- provisioning ("aeb ← central: M 12.4/21.6 GB (57%)") ------------------

export function provisionLine(worker, model, p = {}) {
  const total = num(p.total_bytes)
  const done = num(p.done_bytes)
  let frac = num(p.frac)
  if (frac == null && total && done != null) frac = done / total
  const size = total ? ` ${fmtGB(done || 0)}/${fmtGB(total)} GB` : ' (size not reported yet)'
  return `${worker} ← central: ${model}${size}${frac != null ? ` (${Math.round(frac * 100)}%)` : ''}`
}

// Every in-flight worker pull off the /llm/workers roster (feature-detected:
// workers without the fields contribute nothing).
export function workerProvisioning(workers = []) {
  const out = []
  for (const w of Array.isArray(workers) ? workers : []) {
    if (!w || (w.status && w.status !== 'online')) continue
    const prog = (w.provision_progress && typeof w.provision_progress === 'object') ? w.provision_progress : {}
    const keys = new Set([...(Array.isArray(w.provisioning) ? w.provisioning : []), ...Object.keys(prog)])
    for (const k of keys) {
      const p = prog[k] || {}
      out.push({ worker: w.name || w.id, model: k, progress: p, line: provisionLine(w.name || w.id, k, p) })
    }
  }
  return out
}

// ---- filters (URL-backed) + sorting -----------------------------------------

export const STATUS_FILTER_KEYS = ['worth', 'bucket', 'ver', 'adm', 'grade', 'fail', 'serv']
export const EMPTY_STATUS_FILTERS = Object.freeze({ worth: '', bucket: '', ver: '', adm: '', grade: '', fail: '', serv: '' })
export const STATUS_FILTER_OPTIONS = {
  worth: WORTH_ORDER,
  bucket: Object.keys(BUCKETS),
  ver: ['verified', 'not-verified', 'bad'],
  adm: ['admitted', 'pending', 'held', 'none'],
  grade: ['graded', 'not-graded', 'no-grader', 'lt50', 'ge50'],
  fail: ['any', 'none'],
  serv: ['now', 'cold', 'fits', 'no'],
}

export function parseStatusQuery(search) {
  const q = new URLSearchParams(search || '')
  const out = { ...EMPTY_STATUS_FILTERS }
  for (const k of STATUS_FILTER_KEYS) {
    const v = q.get(`ms_${k}`) || ''
    if (v && STATUS_FILTER_OPTIONS[k].includes(v)) out[k] = v
  }
  return out
}

export function writeStatusQuery(search, f = EMPTY_STATUS_FILTERS) {
  const q = new URLSearchParams(search || '')
  for (const k of STATUS_FILTER_KEYS) { if (f[k]) q.set(`ms_${k}`, f[k]); else q.delete(`ms_${k}`) }
  const s = q.toString()
  return s ? `?${s}` : ''
}

export const hasStatusFilters = (f) => !!f && STATUS_FILTER_KEYS.some(k => !!f[k])

// `row` = the status row (null when the endpoint gives nothing for the model:
// then only an empty filter set matches, so a filtered view never lies).
export function matchesStatusFilters(row, f = EMPTY_STATUS_FILTERS) {
  if (!hasStatusFilters(f)) return true
  if (!row) return false
  const w = row.worth || {}
  if (f.worth && w.label !== f.worth) return false
  if (f.bucket && !(BUCKETS[f.bucket] || []).includes(w.label)) return false
  const v = row.verification || {}
  if (f.ver === 'verified' && !(v.source !== 'none' && v.ok)) return false
  if (f.ver === 'not-verified' && v.source !== 'none') return false
  if (f.ver === 'bad' && !(v.source !== 'none' && !v.ok)) return false
  if (f.adm && (row.admission?.status || 'none') !== f.adm) return false
  const g = row.grade || {}
  if (f.grade === 'graded' && g.value == null) return false
  if (f.grade === 'not-graded' && !(g.suite != null && g.value == null)) return false
  if (f.grade === 'no-grader' && g.suite != null) return false
  if (f.grade === 'lt50' && !(g.value != null && g.value < 50)) return false
  if (f.grade === 'ge50' && !(g.value != null && g.value >= 50)) return false
  if (f.fail === 'any' && !row.last_failure) return false
  if (f.fail === 'none' && row.last_failure) return false
  const s = row.servable || {}
  if (f.serv === 'now' && !s.now) return false
  if (f.serv === 'cold' && !(s.cold || []).length) return false
  if (f.serv === 'fits' && !(s.fits || []).length) return false
  if (f.serv === 'no' && (s.now || (s.fits || []).length || (s.cold || []).length)) return false
  return true
}

const VER_RANK = (r) => { const v = r?.verification; return !v || v.source === 'none' ? 0 : v.ok ? 2 : 1 }
const ADM_RANK = { admitted: 3, pending: 2, none: 1, held: 0 }
// Sort keys: a missing fact sorts BELOW every real value (never as 0).
export const STATUS_SORTS = {
  worth: (r) => (r ? WORTH_ORDER.length - WORTH_ORDER.indexOf(r.worth?.label) : -1),
  verification: (r) => (r ? VER_RANK(r) : -1),
  admission: (r) => (r ? ADM_RANK[r.admission?.status || 'none'] ?? 1 : -1),
  grade: (r) => (r?.grade?.value != null ? r.grade.value : r?.grade?.suite == null ? -2 : -1),
  failure: (r) => num(r?.last_failure?.ts) ?? -1,
  servable: (r) => (!r ? -1 : r.servable?.now ? 3 : (r.servable?.cold || []).length ? 2 : (r.servable?.fits || []).length ? 1 : 0),
}

export function summaryCounts(index) {
  const counts = {}
  for (const l of WORTH_ORDER) counts[l] = Number(index?.counts?.[l] || 0)
  const buckets = {}
  for (const [b, labels] of Object.entries(BUCKETS)) buckets[b] = labels.reduce((n, l) => n + counts[l], 0)
  return { counts, buckets }
}

// ---- "Verify + grade" action state machine ----------------------------------
//
// idle --click--> requesting --POST 202--> queued --poll running--> running
//   --poll done/failed--> done | failed ; any POST refusal --> error.
// `poll` events carry GET /llm/admission/<key>: {admission, jobs:[...]}.
export const VERIFY_IDLE = Object.freeze({ phase: 'idle' })

export function verifyReducer(state = VERIFY_IDLE, ev = {}) {
  switch (ev.type) {
    case 'click':
      return ['requesting', 'queued', 'running'].includes(state.phase) ? state : { phase: 'requesting', at: ev.at || Date.now() }
    case 'posted': {
      if (!ev.ok) {
        const who = ev.status === 401 || ev.status === 403 ? 'operator only' : 'refused'
        return { phase: 'error', message: `${who} (HTTP ${ev.status || 'network'}): ${ev.message || ''}`.trim() }
      }
      return { phase: 'queued', jobId: ev.job?.id || null, at: state.at || Date.now(), message: 'queued' }
    }
    case 'poll': {
      if (!['queued', 'running'].includes(state.phase)) return state
      const jobs = Array.isArray(ev.data?.jobs) ? ev.data.jobs : []
      const job = (state.jobId && jobs.find(j => j.id === state.jobId)) || jobs[0]
      if (!job) return state
      const log = String(job.log || '').split('\n').filter(Boolean)
      const last = log[log.length - 1] || ''
      if (job.status === 'queued') return { ...state, phase: 'queued', jobId: job.id, message: 'queued (waiting for the admission runner)' }
      if (job.status === 'running') return { ...state, phase: 'running', jobId: job.id, message: last || 'running' }
      const adm = ev.data?.admission || {}
      if (job.status === 'done') {
        return { phase: 'done', jobId: job.id, result: adm.status || 'done',
          message: `${adm.status || 'done'}${adm.reason ? `: ${adm.reason}` : ''}` }
      }
      if (job.status === 'failed') {
        const spent = state.at ? ` after ${Math.round((Date.now() - state.at) / 1000)}s` : ''
        return { phase: 'failed', jobId: job.id,
          message: `job ${job.id} failed${spent}: ${last || adm.reason || `empty job log and no admission reason (admission status ${adm.status || 'none'})`}` }
      }
      return state
    }
    case 'reset':
      return VERIFY_IDLE
    default:
      return state
  }
}

export const verifyActive = (s) => ['requesting', 'queued', 'running'].includes(s?.phase)

export function verifyLabel(s) {
  switch (s?.phase) {
    case 'requesting': return 'requesting…'
    case 'queued': return 'queued…'
    case 'running': return 'running…'
    case 'done': return `✓ ${s.result || 'done'}`
    case 'failed': return '✗ failed'
    case 'error': return '✗ refused'
    default: return 'Verify + grade'
  }
}
