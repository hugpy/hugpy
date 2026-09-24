// node --test src/components/ModelTable/modelStatus.test.mjs
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  EMPTY_STATUS_FILTERS, STATUS_SORTS, VERIFY_IDLE, admissionView, failureView, gradeView, indexStatus,
  matchesStatusFilters, parseStatusQuery, throughputView, pollInterval, provisionLine, servableView, statusFor, summaryCounts,
  verificationView, verifyActive, verifyLabel, verifyReducer, workerChips, workerProvisioning, worthView,
  writeStatusQuery,
} from './modelStatus.js'

const ready = {
  model_key: 'Qwen2.5-3B-Instruct-GGUF', task: 'text-generation',
  verification: { verdict: 'static_ok', why: 'static checks passed', at: 1790000000, source: 'audit', ok: true },
  admission: { status: 'admitted', reason: 'static_ok; graded 88', at: 1790000100 },
  grade: { suite: 'hugpy-native-v2', value: 92.6, score: 25, max: 27, text: '25/27 (93%)', at: 1790000100, worker: 'aeb',
    quant: 'Q4_K_M', detail_summary: '25/27 tiers · 8/9 tasks full', stray: [] },
  throughput: { n_calls: 20, mean_tok_s: 71.24, p50: 70, p90: 80, min: 40, max: 88, first_at: 1, last_at: 2 },
  last_failure: null,
  servable: { now: true, where: ['aeb'], cold: [], fits: ['aeb'], fits_offline: [], provisioning: [], reason: 'loaded on aeb' },
  workers: [{ worker: 'aeb', state: 'hot', base: 'hot', label: 'hot', held: false, online: true, detail: 'loaded in slot 1 · idle' },
    { worker: 'computron', state: 'missing', base: 'missing', label: 'missing', held: false, online: true, detail: "not on this worker's disk" }],
  worth: { label: 'ready', bucket: 'ready', why: 'verified, graded 88/100', action: 'use it' },
}
const unverified = {
  model_key: 'owner~Mystery', task: 'text-generation',
  verification: { verdict: null, why: 'never verified', at: null, source: 'none', ok: false },
  admission: { status: 'none', reason: 'no admission record', at: null },
  grade: { suite: 'hugpy-native-v2', value: null, max: 100, reason: 'not graded', stray: [] },
  last_failure: { kind: 'load failure', class: 'hard_load_failure', first_line: 'E llama_model_load: error loading model', worker: 'aeb', ts: 1790000000 },
  servable: { now: false, where: [], cold: [], fits: [], fits_offline: [], provisioning: [], reason: 'fits no known worker' },
  workers: [{ worker: 'aeb', state: 'failed', base: 'cold', label: 'failed: hard_load_failure', held: true, online: true, detail: 'x' }],
  worth: { label: 'unverified', bucket: 'unknown', why: 'never verified', action: 'Verify + grade' },
}
const video = {
  ...unverified, model_key: 'Wan2.1-T2V', task: 'text-to-video',
  verification: { verdict: 'suite_mismatch', why: 'non-text', at: 1, source: 'audit', ok: true },
  grade: { suite: null, value: null, max: null, reason: 'no grader for text-to-video',
    stray: [{ suite: 'hugpy-native-v2', value: 0 }] },
  worth: { label: 'no-grader', bucket: 'unknown', why: 'no suite', action: 'judge it by hand' },
}
const payload = { models: [ready, unverified, video], counts: { ready: 1, unverified: 1, 'no-grader': 1 }, live: false }

test('feature detection: SPA html / 404 body / missing models array -> unavailable', () => {
  assert.equal(indexStatus(null).available, false)
  assert.equal(indexStatus('<!doctype html>').available, false)
  assert.equal(indexStatus({ rows: [] }).available, false)
  const idx = indexStatus(payload)
  assert.equal(idx.available, true)
  assert.equal(statusFor(idx, { model_key: 'Qwen2.5-3B-Instruct-GGUF' }), ready)
  assert.equal(statusFor(idx, 'Mystery'), unverified)                 // owner~ tail form
  assert.equal(statusFor(indexStatus(null), 'x'), null)
})

test('worth rendering: label, tone and why for every bucket', () => {
  assert.deepEqual([worthView(ready).text, worthView(ready).tone], ['ready', 'ok'])
  assert.equal(worthView(unverified).tone, 'warn')
  assert.equal(worthView(video).text, 'no grader')
  assert.match(worthView(ready).title, /→ use it/)
  assert.equal(worthView(null).text, 'NO STATUS')
})

test('explicit missing: never a blank cell, never a fake 0', () => {
  assert.equal(verificationView(unverified).text, 'NOT VERIFIED')
  assert.equal(admissionView(unverified).text, 'NO RECORD')
  assert.equal(gradeView(unverified).text, 'NOT GRADED')
  assert.equal(gradeView(video).text, 'NO GRADER FOR TEXT-TO-VIDEO')
  assert.match(gradeView(video).title, /ignored: hugpy-native-v2 0/)
  assert.equal(failureView(ready).text, 'none recorded')
  for (const row of [ready, unverified, video, null]) {
    for (const v of [verificationView, admissionView, gradeView, failureView, servableView, worthView]) {
      assert.ok(String(v(row).text).trim(), `${v.name} blank for ${row?.model_key}`)
    }
  }
  assert.equal(gradeView(ready).text, '25/27 (93%)')
  assert.equal(throughputView(ready).text, '71.2 tok/s · avg of 20')
  assert.equal(throughputView(unverified).text, 'no calls recorded')
  assert.equal(servableView(ready).text, 'hot: aeb')
  assert.equal(servableView(unverified).text, 'not now')
})

test('worker chips use the shared vocabulary; held turns the chip red', () => {
  const c = workerChips(ready)
  assert.deepEqual(c.map(x => [x.worker, x.label]), [['aeb', 'hot'], ['computron', 'missing']])
  const f = workerChips(unverified)[0]
  assert.equal(f.label, 'failed: hard_load_failure')
  assert.equal(f.tone, 'bad')
})

test('worker chips: on central and not allocated render as muted, distinct icons', () => {
  const row = { workers: [
    { worker: 'aeb', state: 'on central', base: 'on central', label: 'on central', held: false, online: true,
      detail: 'on central storage, copies on first call' },
    { worker: 'computron', state: 'not allocated', base: 'not allocated', label: 'not allocated', held: false, online: true,
      detail: 'on no drive and not allocated here' },
    { worker: 'k61', state: 'missing', base: 'missing', label: 'missing', held: false, online: true,
      detail: 'allocated but on no drive' },
  ] }
  const chips = workerChips(row)
  assert.deepEqual(chips.map(x => [x.state, x.icon, x.tone]), [
    ['on central', '◌', 'muted'],
    ['not allocated', '·', 'muted'],
    ['missing', '∅', 'muted'],
  ])
  // a failed overlay over on-central still turns the chip red
  const failedOnCentral = workerChips({ workers: [
    { worker: 'aeb', state: 'failed', base: 'on central', label: 'failed: hard_load_failure', held: false, online: true, detail: 'x' },
  ] })[0]
  assert.equal(failedOnCentral.tone, 'bad')
})

test('status filters: bucket, verification, grade, failures, servability', () => {
  const rows = [ready, unverified, video]
  const pick = (f) => rows.filter(r => matchesStatusFilters(r, { ...EMPTY_STATUS_FILTERS, ...f })).map(r => r.model_key)
  assert.deepEqual(pick({}), rows.map(r => r.model_key))
  assert.deepEqual(pick({ bucket: 'ready' }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(pick({ bucket: 'unknown' }), ['owner~Mystery', 'Wan2.1-T2V'])
  assert.deepEqual(pick({ worth: 'no-grader' }), ['Wan2.1-T2V'])
  assert.deepEqual(pick({ ver: 'not-verified' }), ['owner~Mystery'])
  assert.deepEqual(pick({ grade: 'not-graded' }), ['owner~Mystery'])
  assert.deepEqual(pick({ grade: 'no-grader' }), ['Wan2.1-T2V'])
  assert.deepEqual(pick({ grade: 'ge50' }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(pick({ fail: 'any' }), ['owner~Mystery', 'Wan2.1-T2V'])
  assert.deepEqual(pick({ serv: 'now' }), ['Qwen2.5-3B-Instruct-GGUF'])
  assert.deepEqual(pick({ adm: 'none' }), ['owner~Mystery', 'Wan2.1-T2V'])
  // no status row: only an empty filter set matches
  assert.equal(matchesStatusFilters(null, EMPTY_STATUS_FILTERS), true)
  assert.equal(matchesStatusFilters(null, { ...EMPTY_STATUS_FILTERS, bucket: 'ready' }), false)
})

test('filters round-trip through the URL and keep unrelated params', () => {
  const s = writeStatusQuery('?tab=models&x=1', { ...EMPTY_STATUS_FILTERS, bucket: 'waste', grade: 'not-graded' })
  const q = new URLSearchParams(s)
  assert.equal(q.get('tab'), 'models')
  assert.equal(q.get('ms_bucket'), 'waste')
  assert.deepEqual(parseStatusQuery(s), { ...EMPTY_STATUS_FILTERS, bucket: 'waste', grade: 'not-graded' })
  assert.deepEqual(parseStatusQuery('?ms_bucket=bogus&ms_worth=ready'), { ...EMPTY_STATUS_FILTERS, worth: 'ready' })
  assert.equal(writeStatusQuery('?ms_worth=ready', EMPTY_STATUS_FILTERS), '')
})

test('sorting: missing facts sort below every real value', () => {
  assert.ok(STATUS_SORTS.grade(ready) > STATUS_SORTS.grade(unverified))
  assert.ok(STATUS_SORTS.grade(unverified) > STATUS_SORTS.grade(video))
  assert.ok(STATUS_SORTS.grade({ grade: { suite: 's', value: 0 } }) > STATUS_SORTS.grade(unverified))
  assert.ok(STATUS_SORTS.worth(ready) > STATUS_SORTS.worth(unverified))
  assert.equal(STATUS_SORTS.worth(null), -1)
})

test('summary counts and buckets', () => {
  const { counts, buckets } = summaryCounts(indexStatus({ models: [], counts: { ready: 2, held: 3, broken: 1, ungraded: 4 } }))
  assert.equal(counts.ready, 2)
  assert.deepEqual(buckets, { ready: 2, waste: 4, unknown: 4 })
})

test('poll cadence: 3 s while any worker is live, else 15 s', () => {
  assert.equal(pollInterval(payload), 15000)
  const live = { models: [{ ...ready, workers: [{ worker: 'aeb', state: 'downloading from central', base: 'downloading from central' }] }] }
  assert.equal(pollInterval(live), 3000)
  assert.equal(pollInterval({ models: [], live: true }), 3000)
})

test('verify + grade action state machine', () => {
  let s = verifyReducer(VERIFY_IDLE, { type: 'click', at: 1 })
  assert.equal(s.phase, 'requesting')
  assert.equal(verifyReducer(s, { type: 'click' }), s)                   // no double submit
  const refused = verifyReducer(s, { type: 'posted', ok: false, status: 403, message: 'forbidden' })
  assert.equal(refused.phase, 'error')
  assert.match(refused.message, /operator only \(HTTP 403\)/)
  s = verifyReducer(s, { type: 'posted', ok: true, status: 202, job: { id: 'j1' } })
  assert.deepEqual([s.phase, s.jobId], ['queued', 'j1'])
  assert.ok(verifyActive(s))
  s = verifyReducer(s, { type: 'poll', data: { jobs: [{ id: 'j1', status: 'running', log: 'a\nstatic: static_ok' }] } })
  assert.deepEqual([s.phase, s.message], ['running', 'static: static_ok'])
  const other = verifyReducer(s, { type: 'poll', data: { jobs: [] } })
  assert.equal(other, s)                                                   // empty poll keeps state
  s = verifyReducer(s, { type: 'poll', data: { admission: { status: 'held', reason: 'graded 0' },
    jobs: [{ id: 'j1', status: 'done' }] } })
  assert.deepEqual([s.phase, s.result, s.message], ['done', 'held', 'held: graded 0'])
  assert.equal(verifyLabel(s), '✓ held')
  assert.equal(verifyReducer(s, { type: 'poll', data: { jobs: [{ id: 'j1', status: 'running' }] } }), s)
  assert.equal(verifyReducer(s, { type: 'reset' }).phase, 'idle')
  const failed = verifyReducer({ phase: 'running', jobId: 'j2' }, { type: 'poll', data: { jobs: [{ id: 'j2', status: 'failed', log: 'crash: X' }] } })
  assert.deepEqual([failed.phase, failed.message], ['failed', 'crash: X'])
})

test('worker provisioning line (aeb ← central)', () => {
  assert.equal(provisionLine('aeb', 'M', { done_bytes: 12.4e9, total_bytes: 21.6e9, frac: 0.574 }),
    'aeb ← central: M 12.4/21.6 GB (57%)')
  assert.equal(provisionLine('aeb', 'M', {}), 'aeb ← central: M (size not reported yet)')
  const rows = workerProvisioning([
    { name: 'aeb', status: 'online', provisioning: ['M'], provision_progress: { M: { done_bytes: 1e9, total_bytes: 2e9 } } },
    { name: 'op', status: 'offline', provisioning: ['N'] },
    { name: 'computron', status: 'online' },
  ])
  assert.deepEqual(rows.map(r => r.line), ['aeb ← central: M 1.0/2.0 GB (50%)'])
})
