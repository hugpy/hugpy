// node --test src/components/MetricsPanel/modelFilters.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  EMPTY_FILTERS, buildModelIndex, filterModels, filterRows, firstLine, gradeBucket,
  parseQuery, passFail, resolveSelection, statusChips, taskStatus, writeQuery,
} from './modelFilters.js'

const tier = (...passes) => ({ tier: passes.filter(Boolean).length, max: 3,
  history: passes.map((pass, i) => ({ tier: i + 1, pass })) })
const allPass = { math: tier(true, true, true), coding: tier(true, true, true) }
const catalogRows = [
  { model_key: 'Qwen2.5-3B-Instruct-GGUF', hub_id: 'Qwen/Qwen2.5-3B-Instruct-GGUF', framework: 'gguf', effective_gguf: 'q4_k_m.gguf' },
  { model_key: 'Qwen2.5-VL-7B-Instruct', hub_id: 'Qwen/Qwen2.5-VL-7B-Instruct', framework: 'transformers' },
  { model_key: 'Echo-Mini', hub_id: 'Jershone/Echo-Mini', framework: 'gguf', admission: { status: 'held', reason: 'bad tensor dims' } },
  { model_key: 'dreamshaper-8', hub_id: 'comfy/dreamshaper-8', framework: 'comfy' },
]
const metricsRows = [
  { model_name: 'Qwen2.5-3B-Instruct-GGUF', worker: 'aeb', quant: 'q4', grade: 100, grade_suite: 'hugpy-native-v2', graded_at: '10', grade_detail: JSON.stringify(allPass) },
  { model_name: 'Qwen~Qwen2.5-VL-7B-Instruct', worker: 'computron', quant: 'bf16', grade: 40, grade_suite: 'hugpy-native-v2', graded_at: '5',
    grade_detail: JSON.stringify({ math: tier(true, false, true), coding: tier(true, true, true) }) },
  { model_name: 'Echo-Mini', worker: 'aeb', grade: 0, grade_suite: 'integrity', graded_at: '7',
    grade_detail: JSON.stringify({ verdict: 'faulty_model', why: 'wrong tensor shape', log: ['a', 'b'] }) },
]
const idx = () => buildModelIndex({ metricsRows, catalogRows })

test('grade buckets', () => {
  assert.deepEqual([null, 'N/A', 0, 10, 49.9, 50, 89, 90, 100].map(gradeBucket),
    ['na', 'na', 'zero', 'lt50', 'lt50', 'mid', 'mid', 'high', 'high'])
})

test('task status from tier history and legacy shapes', () => {
  assert.equal(taskStatus(tier(true, true, true)), 'pass')
  assert.equal(taskStatus(tier(true, false, true)), 'fail')
  assert.equal(taskStatus(tier(true)), 'partial')
  assert.equal(taskStatus(true), 'pass')
  assert.equal(taskStatus(0), 'fail')
  assert.equal(taskStatus(undefined), null)
})

test('alias rows fold onto the catalog key; integrity kept apart from aptitude', () => {
  const { infos, canon } = idx()
  assert.equal(canon('Qwen~Qwen2.5-VL-7B-Instruct'), 'Qwen2.5-VL-7B-Instruct')
  assert.equal(infos.has('Qwen~Qwen2.5-VL-7B-Instruct'), false)
  assert.equal(infos.get('Qwen2.5-VL-7B-Instruct').aptitude.grade, 40)
  const echo = infos.get('Echo-Mini')
  assert.equal(echo.aptitude, null)
  assert.equal(echo.integrity.verdict, 'faulty_model')
  assert.equal(echo.admission.status, 'held')
  assert.deepEqual(statusChips(echo).map(c => c.label), ['untested', 'faulty_model', 'held'])
})

test('pass/fail: any failed tier or integrity 0 fails; per-task narrows', () => {
  const { infos } = idx()
  assert.equal(passFail(infos.get('Qwen2.5-3B-Instruct-GGUF')), 'pass')
  assert.equal(passFail(infos.get('Qwen2.5-VL-7B-Instruct')), 'fail')
  assert.equal(passFail(infos.get('Qwen2.5-VL-7B-Instruct'), 'coding'), 'pass')
  assert.equal(passFail(infos.get('Echo-Mini')), 'fail')
  assert.equal(passFail(infos.get('dreamshaper-8')), null)
})

test('search: every token must match key/hub/framework/quant', () => {
  const { infos } = idx()
  assert.deepEqual(filterModels(infos, { query: 'qwen gguf' }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(filterModels(infos, { query: 'comfy' }), ['dreamshaper-8'])
  assert.deepEqual(filterModels(infos, { query: 'Q4_K_M' }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(filterModels(infos, { query: 'echo-mini' })[0], 'Echo-Mini')
})

test('filters: grade buckets, pass/fail+task, framework, worker, verdict, admission, untested', () => {
  const { infos } = idx()
  const f = (o) => filterModels(infos, { filters: { ...EMPTY_FILTERS, ...o } })
  assert.deepEqual(f({ grade: ['high'] }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(f({ grade: ['na'] }), ['dreamshaper-8', 'Echo-Mini'])
  assert.deepEqual(f({ pf: 'fail' }), ['Echo-Mini', 'Qwen2.5-VL-7B-Instruct'])
  assert.deepEqual(f({ pf: 'fail', task: 'math' }), ['Qwen2.5-VL-7B-Instruct'])
  assert.deepEqual(f({ task: 'coding' }), ['Qwen2.5-3B-Instruct-GGUF', 'Qwen2.5-VL-7B-Instruct'])
  assert.deepEqual(f({ fw: 'comfy' }), ['dreamshaper-8'])
  assert.deepEqual(f({ worker: 'computron' }), ['Qwen2.5-VL-7B-Instruct'])
  assert.deepEqual(f({ verdict: 'faulty_model' }), ['Echo-Mini'])
  assert.deepEqual(f({ adm: 'held' }), ['Echo-Mini'])
  assert.deepEqual(f({ untested: true, fw: 'gguf' }), ['Echo-Mini'])
})

test('row filter keeps only rows of matching models (and the chosen worker)', () => {
  const { infos, canon } = idx()
  const rows = filterRows(metricsRows, infos, canon, { ...EMPTY_FILTERS, fw: 'transformers' })
  assert.deepEqual(rows.map(r => r.model_name), ['Qwen~Qwen2.5-VL-7B-Instruct'])
  assert.equal(filterRows(metricsRows, infos, canon, EMPTY_FILTERS), metricsRows)
})

test('query round-trip keeps unrelated params and drops empties', () => {
  const filters = { ...EMPTY_FILTERS, grade: ['lt50', 'zero'], pf: 'fail', task: 'coding', untested: true }
  const s = writeQuery('?tab=metrics&foo=1', { model: 'Echo-Mini', filters })
  const back = parseQuery(s)
  assert.equal(back.model, 'Echo-Mini')
  assert.deepEqual(back.filters, filters)
  assert.match(s, /tab=metrics/); assert.match(s, /foo=1/)
  assert.equal(writeQuery('', { model: '', filters: EMPTY_FILTERS }), '')
  assert.deepEqual(parseQuery('?grade=bogus,high&pf=maybe&task=nope').filters.grade, ['high'])
  assert.equal(parseQuery('?pf=maybe').filters.pf, '')
})

test('selection is sticky across polls, empty lists and a running benchmark', () => {
  const known = new Set(['a', 'b', 'c'])
  // user choice survives a running benchmark on another model
  assert.deepEqual(resolveSelection({ current: 'b', chosen: true, known, ready: true, running: 'c', fallback: 'a' }),
    { model: 'b', chosen: true, dropped: '' })
  // survives a poll that transiently lost the model / an empty list while loading
  assert.equal(resolveSelection({ current: 'b', chosen: true, known: new Set(['a']), ready: false, fallback: 'a' }).model, 'b')
  assert.equal(resolveSelection({ current: 'b', chosen: true, known: new Set(), ready: false }).model, 'b')
  // only a clean catalog that truly lacks it drops it
  assert.deepEqual(resolveSelection({ current: 'z', chosen: true, known, ready: true, fallback: 'a' }),
    { model: 'a', chosen: false, dropped: 'z' })
  // no user choice: follow the running model, else the first visible model
  assert.equal(resolveSelection({ current: '', known, ready: true, running: 'c', fallback: 'a' }).model, 'c')
  assert.equal(resolveSelection({ current: '', known, ready: true, fallback: 'a' }).model, 'a')
  assert.equal(resolveSelection({ current: 'b', known, ready: true, fallback: 'a' }).model, 'b')
})

test('firstLine prefers the error line', () => {
  assert.equal(firstLine('\n0.00 I loading\n0.01 E llama_model_load: error loading model: bad\nmore'),
    '0.01 E llama_model_load: error loading model: bad')
  assert.equal(firstLine('just text\nsecond'), 'just text')
  assert.equal(firstLine(null), '')
})

test('canonicalRows folds alias forms and keeps the newest duplicate', async () => {
  const { canonicalRows, makeCanon } = await import('./modelFilters.js')
  const canon = makeCanon(catalogRows)
  const rows = canonicalRows([
    { model_name: 'Qwen~Qwen2.5-VL-7B-Instruct', worker: 'aeb', quant: 'q', alloc_mode: 'x', updated_at: '1' },
    { model_name: 'Qwen2.5-VL-7B-Instruct', worker: 'aeb', quant: 'q', alloc_mode: 'x', updated_at: '2' },
    { model_name: 'Echo-Mini', worker: 'aeb', quant: 'q', alloc_mode: 'x', updated_at: '1' },
  ], canon)
  assert.equal(rows.length, 2)
  assert.equal(rows.find(r => r.model_name === 'Qwen2.5-VL-7B-Instruct').updated_at, '2')
})

test('ranking prefers word-boundary token matches in the key', () => {
  const { infos } = buildModelIndex({ catalogRows: [
    { model_key: 'LuffyTheFox-Qwen3.6-35B-A3B-GGUF', framework: 'gguf' },
    { model_key: 'Qwen2.5-3B-Instruct-GGUF', framework: 'gguf' },
    { model_key: 'Qwen2.5-Coder-3B-Instruct-GGUF', framework: 'gguf' },
  ] })
  assert.deepEqual(filterModels(infos, { query: 'qwen 3b' }),
    ['Qwen2.5-3B-Instruct-GGUF', 'Qwen2.5-Coder-3B-Instruct-GGUF', 'LuffyTheFox-Qwen3.6-35B-A3B-GGUF'])
  assert.equal(filterModels(infos, { query: 'qwen2.5-coder' })[0], 'Qwen2.5-Coder-3B-Instruct-GGUF')
})
