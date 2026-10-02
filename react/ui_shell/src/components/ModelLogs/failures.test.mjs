import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeFailures, tsOf } from './failures.js'
import { holdingWorkers } from '../ChatPanel/workers.js'

test('tsOf accepts epoch s, epoch ms and ISO', () => {
  assert.equal(tsOf(1790000000), 1790000000000)
  assert.equal(tsOf('1790000000.5'), 1790000000500)
  assert.equal(tsOf(1790000000000), 1790000000000)
  assert.equal(tsOf('2026-09-23T00:00:00Z'), Date.parse('2026-09-23T00:00:00Z'))
  assert.equal(tsOf(null), null)
})

test('normalizeFailures: compute-actions rows, refusals, newest first, full text', () => {
  const stderr = 'line1 E llama_model_load: error loading model\nline2'
  const items = normalizeFailures({
    actions: [{ ts: 10, worker_card: 'aeb', detail: { class: 'hard_load_failure', loader_stderr: stderr, message: 'HardLoadFailure: x' } }],
    refusals: [{ ts: 20, predicate: 'holds_model', request_id: 'r1', reason: 'not on disk' }],
  })
  assert.equal(items.length, 2)
  assert.equal(items[0].kind, 'routing refusal')
  assert.equal(items[0].predicate, 'holds_model')
  assert.equal(items[0].request_id, 'r1')
  assert.equal(items[1].kind, 'load failure')
  assert.equal(items[1].cls, 'hard_load_failure')
  assert.equal(items[1].text, stderr)            // never truncated
  assert.equal(items[1].message, 'HardLoadFailure: x')
  assert.deepEqual(normalizeFailures(null), [])
  assert.equal(normalizeFailures([{ kind: 'load_failure', at: 5 }])[0].kind, 'load failure')
})

test('holdingWorkers: only workers with bytes on disk; hot first', () => {
  const model = { hot_workers: ['computron'], workers: [
    { worker: 'aeb', on_disk_bytes: 5, status: 'online' },
    { worker: 'computron', on_disk_bytes: 5, status: 'online' },
    { worker: 'op', on_disk_bytes: 0, status: 'online' },
    { worker: 'zed', on_disk_bytes: 9, status: 'offline' },
  ] }
  assert.deepEqual(holdingWorkers(model).map(w => [w.name, w.hot, w.online]),
    [['computron', true, true], ['aeb', false, true], ['zed', false, false]])
  assert.deepEqual(holdingWorkers(undefined), [])
})
