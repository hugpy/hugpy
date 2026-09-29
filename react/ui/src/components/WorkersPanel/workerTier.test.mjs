// node --test src/components/WorkersPanel/workerTier.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import { TIER_RANK, compareByWorkerTier, modelSpellings, storageRowsByKey, workerTierOf } from './workerTier.js'

// Shapes lifted from the live roster (GET /llm/workers) and catalog (GET /models).
const aeb = {
  id: 'w-aeb', name: 'ae-worker',
  models: ['Qwen3-Coder-Next-GGUF'],
  models_local: ['Qwen3-Coder-Next-GGUF'],
  storage: { models: [
    { model_key: 'Qwen3-Coder-Next-GGUF', bytes: 93_311_284_363, store: 'reapable', assigned: true, last_picked: 1790643277 },
    { model_key: 'Qwen3.6-35B-A3B', bytes: 71_926_873_205, store: 'reapable', assigned: false, last_picked: null },
    { model_key: 'Qwen3-32B', bytes: 65_540_298_841, store: 'reapable', assigned: false, last_picked: 1790643277 },
  ] },
}
const abrain = {
  id: 'w-abrain', name: 'a-brain', models: [], models_local: [],
  storage: { models: [
    { model_key: 'GLM-5.3-Flash-Q4_K', bytes: 190_875_467_776, store: 'shared', protected: true, why: 'shared/central storage — never reaped' },
  ] },
}
const cat = (model_key, extra = {}) => ({ model_key, name: model_key, status: 'installed', size_bytes: 1, ...extra })

test('hot: an UNASSIGNED model on the worker\'s own store root is hot (models_local alone would miss it)', () => {
  const t = workerTierOf(cat('Qwen3.6-35B-A3B'), aeb)
  assert.equal(t.tier, 'hot'); assert.equal(t.source, 'storage'); assert.equal(t.row.bytes, 71_926_873_205)
})

test('shared: a store=shared survey row is NOT this box\'s disk', () => {
  const t = workerTierOf(cat('GLM-5.3-Flash-Q4_K'), abrain)
  assert.equal(t.tier, 'shared'); assert.equal(t.rank, TIER_RANK.shared)
})

test('central: on central only when neither the survey nor models_local nor hot_workers name this worker', () => {
  const t = workerTierOf(cat('Qwen2.5-3B-Instruct-GGUF', { hot_workers: ['computron'] }), aeb)
  assert.equal(t.tier, 'central'); assert.equal(t.source, 'catalog')
})

test('none: no catalog files anywhere', () => {
  const t = workerTierOf({ model_key: 'ghost', status: 'missing', dir_bytes: 0 }, aeb)
  assert.equal(t.tier, 'none')
})

test('hot_workers by worker NAME is honoured when the survey has no row', () => {
  const t = workerTierOf(cat('X', { hot_workers: ['ae-worker'] }), { ...aeb, storage: {}, models_local: [] })
  assert.equal(t.tier, 'hot'); assert.equal(t.source, 'hot_workers')
})

test('owner~name / hub_id spellings resolve to the survey\'s bare key', () => {
  const m = cat('Qwen~Qwen3-32B', { hub_id: 'Qwen/Qwen3-32B' })
  assert.ok(modelSpellings(m).has('Qwen3-32B'))
  assert.equal(workerTierOf(m, aeb).tier, 'hot')
})

test('storageRowsByKey ignores junk rows and tolerates a worker without a survey', () => {
  assert.equal(storageRowsByKey({ storage: { models: [null, {}, { model_key: 'a' }] } }).size, 1)
  assert.equal(storageRowsByKey({}).size, 0)
  assert.equal(workerTierOf(cat('a'), {}).tier, 'central')
})

test('sort: hot first, then shared, then central; within hot the bigger on-worker copy first; then name', () => {
  const rows = [
    cat('Qwen2.5-3B-Instruct-GGUF', { size_bytes: 2_104_932_768 }),     // central
    cat('Qwen3-32B'),                                                    // hot 65G
    cat('Qwen3.6-35B-A3B'),                                              // hot 71G
    cat('GLM-5.3-Flash-Q4_K', { size_bytes: 190_875_467_776 }),       // central on aeb (bigger than the 3B)
  ]
  const rowsByKey = storageRowsByKey(aeb)
  const tiers = new Map(rows.map(m => [m, workerTierOf(m, aeb, rowsByKey)]))
  const sizeOf = m => m.size_bytes ?? null
  const sorted = [...rows].sort((a, b) => compareByWorkerTier(a, b, tiers.get(a), tiers.get(b), sizeOf, 1))
  assert.deepEqual(sorted.map(m => m.model_key),
    ['Qwen3.6-35B-A3B', 'Qwen3-32B', 'GLM-5.3-Flash-Q4_K', 'Qwen2.5-3B-Instruct-GGUF'])
  const desc = [...rows].sort((a, b) => compareByWorkerTier(a, b, tiers.get(a), tiers.get(b), sizeOf, -1))
  assert.deepEqual(desc.map(m => m.model_key), sorted.map(m => m.model_key).reverse())
})
