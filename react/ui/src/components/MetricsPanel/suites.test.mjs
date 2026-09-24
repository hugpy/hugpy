// node --test src/components/MetricsPanel/suites.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  FALLBACK_SUITES, combineThroughput, fromLedger, gradedCallRows, matrixFor, resolveSuite, scoreText,
  splitTask, suiteIndex, throughputText,
} from './suites.js'

const tiers = (passes) => ({ tier: passes.filter(Boolean).length, max: 3,
  history: passes.map((pass, i) => ({ tier: ['easy', 'medium', 'hard'][i], pass })) })

const VISION = FALLBACK_SUITES['hugpy-vision-v1']
const TEXT = FALLBACK_SUITES['hugpy-native-v2']

test('score text is n/max (pct)', () => {
  assert.equal(scoreText(9, 18), '9/18 (50%)')
  assert.equal(scoreText(27, 27), '27/27 (100%)')
  assert.equal(scoreText(null, 18), 'not graded')
})

test('a vision-suite row renders 6 columns, 18 markers, 9/18 (50%)', () => {
  const detail = { color: tiers([true, true, false]), count: tiers([true, false, false]), spatial: tiers([true, true, true]),
    read: tiers([true, false, false]), shape: tiers([false, true, false]), compare: tiers([true, false, false]) }
  const suite = resolveSuite({ gradeSuite: 'hugpy-vision-v1', detail })
  const m = matrixFor(detail, suite)
  assert.deepEqual(m.columns, ['color', 'count', 'spatial', 'read', 'shape', 'compare'])
  assert.equal(m.cells.flatMap(c => c.markers).length, 18)
  assert.equal(m.score, 9)
  assert.equal(m.text, '9/18 (50%)')
})

test('a text-suite row renders 9 columns, 27 markers; never rescaled onto another suite', () => {
  const detail = Object.fromEntries(TEXT.tasks.map(t => [t, tiers([true, true, true])]))
  const m = matrixFor(detail, resolveSuite({ gradeSuite: 'hugpy-native-v2', detail }))
  assert.equal(m.columns.length, 9)
  assert.equal(m.cells.flatMap(c => c.markers).length, 27)
  assert.equal(m.text, '27/27 (100%)')
  // a vision detail never lands in text columns
  const v = matrixFor({ color: tiers([true, true, true]) }, resolveSuite({ gradeSuite: 'hugpy-vision-v1' }))
  assert.equal(v.max, 18)
  assert.equal(v.text, '3/18 (17%)')
})

test('suite inference: declared > grade_suite > detail keys; legacy 9-point rows say so', () => {
  assert.equal(resolveSuite({ detail: { color: tiers([true]) } }).name, 'hugpy-vision-v1')
  const legacy = resolveSuite({ detail: { math: true, json: false } })
  assert.equal(legacy.name, 'fleet-capacity-v1')
  assert.ok(legacy.legacy)
  const lm = matrixFor({ math: true, json: false }, legacy)
  assert.equal(lm.cells[0].markers.length, 1)
  assert.equal(lm.text, '1/9 (11%)')
  const idx = suiteIndex([{ name: 'hugpy-vision-v1', tasks: ['color', 'count'], tiers: 3, max: 6 }])
  assert.equal(resolveSuite({ gradeSuite: 'hugpy-vision-v1', index: idx }).max, 6)   // server registry wins
  assert.equal(matrixFor({}, VISION).text, 'no per-item verdicts recorded')
})

test('hover data: expected/actual/why ride on each marker when recorded', () => {
  const detail = { count: { tier: 0, max: 3, history: [{ tier: 'easy', pass: false, expected: 'first number == 2', actual: '3', why: 'said 3' }] } }
  const mk = matrixFor(detail, VISION).cells.find(c => c.task === 'count').markers[0]
  assert.deepEqual([mk.pass, mk.expected, mk.actual, mk.why], [false, 'first number == 2', '3', 'said 3'])
})

test('a run with 18 graded calls: denominator 18, 18 rows, score = PASS count', () => {
  const calls = []
  let t = 1000
  for (const task of VISION.tasks) {
    for (const tier of ['easy', 'medium', 'hard']) {
      const passed = (calls.length % 2) === 0
      calls.push({ task: `${task} (${tier})`, grade: passed ? 'PASS' : 'FAIL', passed, output: passed ? 'ok' : 'nope',
        check: `rule for ${task} ${tier}`, elapsed_s: 1.2, tok_s: 80, worker: 'aeb', quant: 'Q4_K_M',
        alloc_mode: 'gpu_only', timestamp: t++, error: 'N/A' })
    }
  }
  // an older call for the same item is replaced, not added
  calls.unshift({ task: 'color (easy)', grade: 'FAIL', passed: false, timestamp: 1, error: 'N/A' })
  const g = gradedCallRows(calls, VISION)
  assert.equal(g.rows.length, 18)
  assert.equal(g.max, 18)
  assert.equal(g.score, g.rows.filter(r => r.verdict === 'PASS').length)
  assert.equal(g.score, 9)
  assert.equal(g.text, '9/18 (50%)')
  const fail = g.rows.find(r => r.verdict === 'FAIL')
  assert.match(fail.why, /answer did not satisfy: rule for/)
  assert.equal(fail.actual, 'nope')
  assert.deepEqual(g.rows.slice(0, 3).map(r => `${r.task} ${r.tier}`), ['color easy', 'color medium', 'color hard'])
})

test('errors are ERROR rows with their reason; unknown items never counted', () => {
  const g = gradedCallRows([{ task: 'count (easy)', error: 'timeout after 60s', timestamp: 1 },
    { task: 'bogus (easy)', grade: 'PASS', timestamp: 2 }], VISION)
  assert.deepEqual(g.rows.map(r => [r.verdict, r.why]), [['ERROR', 'timeout after 60s']])
  assert.equal(g.text, '0/18 (0%)')
})

test('ledger rows carry their verdict from state', () => {
  const c = fromLedger({ task: 'color (easy)', state: JSON.stringify({ grade: 'PASS', passed: true, output: 'red', benchmark: true }) })
  assert.deepEqual([c.grade, c.output, c.benchmark], ['PASS', 'red', true])
  assert.equal(splitTask('color (easy)')[1], 'easy')
})

test('throughput: avg of N calls or explicit none; never an EMA', () => {
  assert.equal(throughputText({ n_calls: 20, mean_tok_s: 71.24, p50: 70, p90: 80 }), '71.2 tok/s · avg of 20 calls · p50 70.0 · p90 80.0')
  assert.equal(throughputText({ n_calls: 0 }), 'no calls recorded')
  assert.equal(throughputText(null), 'no calls recorded')
  const c = combineThroughput([{ n_calls: 2, tokens: 200, seconds: 4 }, { n_calls: 1, tokens: 100, seconds: 1 }])
  assert.deepEqual([c.n_calls, c.mean_tok_s, c.label], [3, 60, 'avg of 3 calls'])
  assert.equal(combineThroughput([]).reason, 'no calls recorded')
})
