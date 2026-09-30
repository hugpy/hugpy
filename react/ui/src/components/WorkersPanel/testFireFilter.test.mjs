// node --test src/components/WorkersPanel/testFireFilter.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import { TF_FILTERS, testFireRows, testFireCounts, filterTestFireRows, loadTestFireFilter, saveTestFireFilter } from './testFireFilter.js'

const job = {
  models: ['tg-a', 'Qwen2.5-VL-3B', 'tg-b'],
  summary: { per_model: { 'tg-a': { last_status: 'ok', ok: 1, failed: 0 },
                          'Qwen2.5-VL-3B': { last_status: 'fail', ok: 0, failed: 1 } } },
  skipped: [{ model_key: 'emb', reason: 'not text-gen: feature-extraction' }],
}
const mem = () => { const m = new Map(); return { getItem: k => m.has(k) ? m.get(k) : null, setItem: (k, v) => m.set(k, String(v)) } }

test('chip labels and counts; filters narrow to that status; skipped carries its reason', () => {
  assert.deepEqual(TF_FILTERS.map(f => f.label), ['All', 'Passed', 'Failed', 'Skipped'])
  const rows = testFireRows(job)
  assert.deepEqual(testFireCounts(rows), { all: 4, passed: 1, failed: 1, skipped: 1 })
  assert.deepEqual(filterTestFireRows(rows, 'all').map(r => r.key), ['tg-a', 'Qwen2.5-VL-3B', 'tg-b', 'emb'])
  assert.deepEqual(filterTestFireRows(rows, 'passed').map(r => r.key), ['tg-a'])
  assert.deepEqual(filterTestFireRows(rows, 'failed').map(r => r.key), ['Qwen2.5-VL-3B'])
  const sk = filterTestFireRows(rows, 'skipped')
  assert.equal(sk.length, 1); assert.equal(sk[0].skip.reason, 'not text-gen: feature-extraction')
})

test('filter choice persists per worker, defaults to All, ignores junk', () => {
  const s = mem()
  assert.equal(loadTestFireFilter('w1', s), 'all')
  saveTestFireFilter('w1', 'failed', s)
  assert.equal(loadTestFireFilter('w1', s), 'failed')
  assert.equal(loadTestFireFilter('w2', s), 'all')
  s.setItem('hugpy.testFire.filter.w2', 'bogus')
  assert.equal(loadTestFireFilter('w2', s), 'all')
  assert.equal(loadTestFireFilter('w1', undefined), 'all')
})
