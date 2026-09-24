// node --test src/components/HelpPanel/helpModel.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import { buildHelpContext, recordsToView, activityLine } from './helpModel.js'

test('context strip: route + tab, model, worker from URL, last error', () => {
  const c = buildHelpContext({
    pathname: '/console', search: '?tab=compute&worker=aeb', model: 'Echo-Mini',
    error: 'HTTP 500 · /api/llm/workers · boom', surface: 'console',
  })
  assert.deepEqual(c.items.map(i => i.key), ['route', 'model', 'worker', 'error'])
  assert.equal(c.payload.route, '/console · compute')
  assert.equal(c.payload.worker, 'aeb')
  assert.equal(c.payload.model, 'Echo-Mini')
  assert.equal(c.payload.surface, 'console')
})

test('explicit tab wins over URL; empty values are dropped', () => {
  const c = buildHelpContext({ pathname: '/console', search: '?tab=models', tab: 'status', model: '', error: null })
  assert.deepEqual(c.items, [{ key: 'route', label: 'route', value: '/console · status' }])
})

test('long errors are clipped; extra context is carried', () => {
  const c = buildHelpContext({ pathname: '/c', error: 'x'.repeat(2000), extra: { request_id: 'r-1', n: 3 } })
  assert.ok(c.payload.error.length <= 601)
  assert.equal(c.payload.request_id, 'r-1')
  assert.equal(c.payload.n, '3')
})

test('no location at all is safe', () => {
  assert.deepEqual(buildHelpContext().items, [])
})

test('activity lines', () => {
  assert.equal(activityLine({ type: 'tool', name: 'Bash', summary: 'journalctl -u 7002_hugpy_api -n 200' }),
    'Bash: journalctl -u 7002_hugpy_api -n 200')
  assert.equal(activityLine({ type: 'tool_result', is_error: false, text: '\n12 passed in 1.2s\n' }), '✓ 12 passed in 1.2s')
  assert.equal(activityLine({ type: 'tool_result', is_error: true, text: '' }), '✗ error')
})

test('records fold into bubbles: text streams into one agent bubble per turn', () => {
  const recs = [
    { i: 0, kind: 'meta' }, { i: 1, kind: 'backend', backend: 'claude-arm', label: 'claude arm' },
    { i: 2, kind: 'ref', ref: 'cs-1' },
    { i: 3, kind: 'user', text: 'what failed?', context: '- route: /console' },
    { i: 4, kind: 'event', type: 'status', state: 'compiling' },
    { i: 5, kind: 'event', type: 'tool', name: 'Bash', summary: 'curl compute-actions' },
    { i: 6, kind: 'event', type: 'tool_result', is_error: false, text: '{"actions":[]}' },
    { i: 7, kind: 'event', type: 'text', text: 'Echo-Mini ' },
    { i: 8, kind: 'event', type: 'text', text: 'failed.' },
    { i: 9, kind: 'event', type: 'done', rc: 0, result: 'Echo-Mini failed.' },
    { i: 10, kind: 'user', text: 'why?' },
    { i: 11, kind: 'event', type: 'done', rc: 1, error: 'held' },
  ]
  const v = recordsToView(recs)
  assert.deepEqual(v.map(x => x.kind), ['note', 'user', 'activity', 'activity', 'agent', 'user', 'error'])
  assert.equal(v[4].text, 'Echo-Mini failed.')
  assert.equal(v[1].context, '- route: /console')
})

test('done with only a result (no streamed text) still shows the answer', () => {
  const v = recordsToView([{ i: 1, kind: 'user', text: 'q' }, { i: 2, kind: 'event', type: 'done', rc: 0, result: 'answer' }])
  assert.deepEqual(v.map(x => [x.kind, x.text]), [['user', 'q'], ['agent', 'answer']])
})
