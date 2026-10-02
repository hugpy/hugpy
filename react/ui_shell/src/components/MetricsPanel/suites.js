// Grade views driven by the SUITE that produced them — never by constants.
//
// The server serves the suite registry (GET /llm/benchmark/suites: name, task,
// ordered tasks, tiers, max, per-item prompt + expected). FALLBACK_SUITES is
// only the offline copy of that table for an older central. A score is always
// the visible count of PASS items over the suite's max: `n/max (pct)`.
// Pure (node --test: suites.test.mjs).

export const FALLBACK_SUITES = {
  'hugpy-native-v2': { name: 'hugpy-native-v2', task: 'text-generation', tiers: 3, max: 27,
    tasks: ['math', 'wordprob', 'factual', 'format_primes', 'logic', 'exact_instruction', 'coding', 'json', 'letters'] },
  'hugpy-vision-v1': { name: 'hugpy-vision-v1', task: 'image-text-to-text', tiers: 3, max: 18,
    tasks: ['color', 'count', 'spatial', 'read', 'shape', 'compare'] },
  'hugpy-imagegen-v1': { name: 'hugpy-imagegen-v1', task: 'text-to-image', tiers: 3, max: 18,
    tasks: ['render', 'size', 'color', 'layout', 'contrast', 'shape'] },
  'fleet-capacity-v1': { name: 'fleet-capacity-v1', task: 'text-generation', tiers: 1, max: 9, legacy: true,
    tasks: ['math', 'wordprob', 'factual', 'format_primes', 'logic', 'exact_instruction', 'coding', 'json', 'letters'] },
}
export const TIER_NAMES = ['easy', 'medium', 'hard']

// name -> suite; the server's registry wins, the fallback fills gaps.
export function suiteIndex(registry) {
  const out = { ...FALLBACK_SUITES }
  for (const s of Array.isArray(registry) ? registry : (registry?.suites || [])) {
    if (s && s.name && Array.isArray(s.tasks) && s.tasks.length) out[s.name] = { ...(out[s.name] || {}), ...s }
  }
  return out
}

const parse = (v) => {
  if (v == null || v === '') return null
  if (typeof v === 'object') return v
  try { return JSON.parse(v) } catch { return null }
}

// The suite for one grade row: the row's own declaration, else its
// grade_suite, else inferred from the detail's keys (flagged `inferred`).
export function resolveSuite({ gradeSuite, detail, index = FALLBACK_SUITES } = {}) {
  const d = parse(detail) || {}
  if (d.suite && index[d.suite]) return index[d.suite]
  if (Array.isArray(d.tasks) && d.max) return { name: d.suite || gradeSuite || 'unknown', tasks: d.tasks, tiers: d.tiers || 3, max: d.max }
  if (gradeSuite && index[gradeSuite]) return index[gradeSuite]
  const keys = Object.keys(d).filter(k => !['suite', 'max', 'tasks', 'tiers'].includes(k))
  if (keys.length && keys.every(k => typeof d[k] === 'number' || typeof d[k] === 'boolean')) return { ...index['fleet-capacity-v1'], inferred: true }
  const hit = Object.values(index).find(s => !s.legacy && keys.length && keys.every(k => s.tasks.includes(k)))
  if (hit) return { ...hit, inferred: true }
  return { name: gradeSuite || 'unknown suite', tasks: keys, tiers: 3, max: 3 * keys.length, inferred: true, unknown: true }
}

export function scoreText(score, max) {
  if (score == null || !max) return 'not graded'
  return `${score}/${max} (${Math.round((100 * score) / max)}%)`
}

const tierIndex = (t, i) => {
  const n = TIER_NAMES.indexOf(String(t).toLowerCase())
  if (n >= 0) return n
  const k = Number(t)
  return Number.isFinite(k) && k >= 1 ? k - 1 : i
}

// Pass/fail markers per task x tier for one grade_detail, in the suite's order.
// A task/tier with no recorded verdict is an explicit `null` ("not run").
export function matrixFor(detail, suite) {
  const d = parse(detail) || {}
  const tiers = suite.tiers || 3
  const cells = suite.tasks.map(task => {
    const v = d[task]
    const markers = Array.from({ length: tiers }, (_, i) => ({ tier: TIER_NAMES[i] || String(i + 1), pass: null }))
    if (typeof v === 'boolean' || typeof v === 'number') {
      markers[0] = { ...markers[0], pass: typeof v === 'boolean' ? v : v > 0 }
    } else if (v && typeof v === 'object') {
      const hist = Array.isArray(v.history) ? v.history : []
      hist.forEach((h, i) => {
        if (!h) return
        const at = tierIndex(h.tier, i)
        if (at < tiers) markers[at] = { tier: markers[at].tier, pass: h.pass === true ? true : h.pass === false ? false : null,
          prompt: h.prompt, expected: h.expected, expected_answer: h.expected_answer, actual: h.actual, why: h.why,
          check_pass: h.check_pass, format: h.format, revised: !!h.revised, judge: h.judge || null }
      })
      if (!hist.length && v.tier != null) for (let i = 0; i < tiers; i++) markers[i].pass = i < Number(v.tier)
    }
    return { task, markers }
  })
  const all = cells.flatMap(c => c.markers)
  const score = all.filter(m => m.pass === true).length
  const recorded = all.filter(m => m.pass !== null).length
  return { suite: suite.name, legacy: !!suite.legacy, columns: suite.tasks, cells, score, max: suite.max || all.length,
    recorded, text: recorded ? scoreText(score, suite.max || all.length) : 'no per-item verdicts recorded' }
}

// "color (easy)" -> ['color', 'easy']
export function splitTask(task) {
  const m = /^(.*?)\s*\(([^)]*)\)\s*$/.exec(String(task || ''))
  return m ? [m[1], m[2]] : [String(task || ''), '']
}

// One graded run's calls -> the item rows + the score as their visible sum.
// Keeps the LATEST call per task x tier (a re-run replaces, never adds).
// `suite` from resolveSuite / suiteIndex (items carry prompt + expected).
export function gradedCallRows(calls, suite) {
  const latest = new Map()
  for (const c of Array.isArray(calls) ? calls : []) {
    if (!c) continue
    const [task, tier] = splitTask(c.task)
    if (!suite.tasks.includes(task)) continue
    const key = `${task}\0${tier}`
    const ts = Number(c.timestamp ?? c.ts ?? 0)
    if (!latest.has(key) || ts >= latest.get(key)._ts) latest.set(key, { ...c, _ts: ts, _task: task, _tier: tier })
  }
  const order = (r) => suite.tasks.indexOf(r._task) * 10 + Math.max(0, TIER_NAMES.indexOf(r._tier))
  const rows = [...latest.values()].sort((a, b) => order(a) - order(b)).map(c => {
    const item = (suite.items?.[c._task] || []).find(i => String(i.tier) === c._tier) || {}
    const error = c.error && c.error !== 'N/A' ? String(c.error) : ''
    const verdict = error ? 'ERROR' : c.passed === true || c.grade === 'PASS' ? 'PASS' : c.passed === false || c.grade === 'FAIL' ? 'FAIL' : 'UNKNOWN'
    const expected = c.expected || (typeof c.check === 'string' ? c.check : '') || item.expected || '(expected answer not recorded)'
    return {
      task: c._task, tier: c._tier, prompt: c.prompt || item.prompt || '(prompt not recorded)', expected,
      expected_answer: c.expected_answer ?? item.expected_answer ?? null,
      actual: c.output ?? c.actual ?? '', verdict, pass: verdict === 'PASS' ? true : verdict === 'FAIL' ? false : null,
      check_pass: c.check_pass ?? null, format: c.format ?? null, revised: !!c.revised, judge: c.judge || null,
      why: verdict === 'FAIL' ? (c.why || `answer did not satisfy: ${expected}`) : verdict === 'ERROR' ? error : '',
      elapsed_s: c.elapsed_s ?? null, tok_s: c.tok_s ?? c.tok_per_s ?? null, s_per_image: c.s_per_image ?? null,
      worker: c.worker || c.worker_id || '', quant: c.quant || '', alloc_mode: c.alloc_mode || c.config || '', ts: c._ts || null,
    }
  })
  const score = rows.filter(r => r.verdict === 'PASS').length
  const max = suite.max || rows.length
  return { rows, score, max, answered: rows.length, text: scoreText(score, max) }
}

// Persisted model_calls rows (GET /llm/models/<key>/metrics) -> call shape.
// EVERY key of the persisted record is relayed (the ledger stores the whole
// call: prompt, expected, expected_answer, output, check_pass, format,
// revised, judge, why, failure_class, reason, evidence, timings); the row's
// own columns win only for the ledger's numeric fields.
export function fromLedger(c) {
  const st = parse(c?.state) || {}
  return { ...st, ...c, grade: st.grade ?? c.grade, passed: st.passed ?? c.passed, error: st.error ?? c.error,
    output: st.output ?? c.output, check: st.check ?? c.check, grade_suite: st.grade_suite ?? c.grade_suite,
    caller: st.caller ?? c.caller, benchmark: !!st.benchmark || st.grade != null, state: st }
}

// ---- throughput: the mean over every recorded call (never an EMA) ----------

const fmtN = (v, dp = 1) => (v == null || isNaN(Number(v)) ? '?' : Number(v).toFixed(dp))

export function throughputText(tp) {
  // Σ tokens / Σ generation seconds over every counted call, with n; every
  // absence says why (the server's reason, from the rows' own counts).
  if (!tp) return 'row carries no throughput block'
  if (!tp.n_calls) return tp.reason || 'no calls recorded'
  const nr = tp.n_rated ?? tp.n_calls
  const head = tp.mean_tok_s != null
    ? `${fmtN(tp.mean_tok_s)} tok/s · Σtokens/Σgeneration-s over ${nr} of ${tp.n_calls} call${tp.n_calls === 1 ? '' : 's'}`
    : `no tok/s · ${tp.reason || `${tp.n_calls} call(s), none with a generation window`}`
  const spread = tp.p50 != null ? ` · p50 ${fmtN(tp.p50)} · p90 ${fmtN(tp.p90)}` : ''
  const parts = []
  if (tp.n_no_tokens) parts.push(`${tp.n_no_tokens} without completion tokens`)
  if (tp.n_no_window) parts.push(`${tp.n_no_window} with no generation window (≤1 token)`)
  const basis = [['n_engine', 'engine-timed'], ['n_stream', 'stream-clock'], ['n_wall', 'wall-clock']]
    .filter(([k]) => tp[k]).map(([k, label]) => `${tp[k]} ${label}`)
  if (basis.length) parts.push(`window: ${basis.join(', ')}`)
  if (tp.n_bench_twins) parts.push(`${tp.n_bench_twins} graded call(s) counted once, on their relay row`)
  const u = tp.unstamped
  const ustr = u && u.n_calls
    ? `\n+${u.n_calls} call${u.n_calls === 1 ? '' : 's'} without quant/alloc stamp: ${u.mean_tok_s != null ? `${fmtN(u.mean_tok_s)} tok/s` : (u.reason || 'no tok/s')}` +
      ` (${u.n_no_quant ?? '?'} missing quant, ${u.n_no_alloc ?? '?'} missing alloc)`
    : ''
  return `${head}${spread}${parts.length ? ` · ${parts.join(' · ')}` : ''}${ustr}`
}

// Combine several throughput blocks (cells or per-worker totals) into one
// Σtokens / Σseconds mean. Blocks without the sums cannot be combined exactly:
// they are counted in n and named in `reason`, never averaged by rate.
export function combineThroughput(cells) {
  let tokens = 0, seconds = 0, n = 0, nr = 0, inexact = 0
  const u = { n_calls: 0, n_rated: 0, tokens: 0, seconds: 0, n_no_quant: 0, n_no_alloc: 0 }
  for (const c of cells || []) {
    if (!c || !c.n_calls) continue
    n += c.n_calls
    if (c.tokens != null && c.seconds != null) {
      tokens += Number(c.tokens) || 0; seconds += Number(c.seconds) || 0; nr += c.n_rated ?? c.n_calls
    } else if (c.mean_tok_s != null) inexact += c.n_calls
    const cu = c.unstamped
    if (cu && typeof cu === 'object' && cu.n_calls) {
      u.n_calls += cu.n_calls; u.n_rated += cu.n_rated ?? 0
      u.tokens += Number(cu.tokens) || 0; u.seconds += Number(cu.seconds) || 0
      u.n_no_quant += cu.n_no_quant || 0; u.n_no_alloc += cu.n_no_alloc || 0
    }
  }
  if (!n) return { n_calls: 0, mean_tok_s: null, reason: 'no calls recorded' }
  const mean = tokens > 0 && seconds > 0 ? tokens / seconds : null
  const out = { n_calls: n, n_rated: nr, tokens, seconds, mean_tok_s: mean,
    label: `Σtok/Σgen-s over ${nr} of ${n} call${n === 1 ? '' : 's'}` }
  if (mean == null) out.reason = `${n} call(s), none with tokens and a generation window`
  if (inexact) out.reason = `${out.reason ? out.reason + '; ' : ''}${inexact} call(s) from blocks without token/second sums are counted in n but not in the mean`
  if (u.n_calls) out.unstamped = { ...u, mean_tok_s: u.tokens > 0 && u.seconds > 0 ? u.tokens / u.seconds : null,
    reason: u.tokens > 0 && u.seconds > 0 ? undefined : 'none with a generation window' }
  return out
}


// Which suite grades a MODEL TASK (text-generation, image-text-to-text, ...).
// Reads the server registry's `serves_tasks` (plus each suite's own `task`);
// null means NO suite covers the task — the model cannot be graded
// automatically, and every view must say so rather than grade it 0.
export function suiteForTask(index, task) {
  if (!task) return null
  for (const s of Object.values(index || {})) {
    if (s.legacy) continue
    if (s.task === task || (Array.isArray(s.serves_tasks) && s.serves_tasks.includes(task))) return s
  }
  return null
}
