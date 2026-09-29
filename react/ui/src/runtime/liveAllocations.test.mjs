// node --test src/runtime/liveAllocations.test.mjs
// LOAD-STATE-LATENCY (2026-09-29): the liveness feed's per-allocation load
// state must patch the cached roster rows so "loading" -> "serving" (with
// measured VRAM) shows within a beat, not after the ≤30 s roster rebuild.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mergeAllocations, anyLoadingIn } from './liveAllocations.js'

const rosterSlot = { kind: 'slot', slot_id: '1', model_key: 'coder', healthy: true, busy: false,
                     vram_bytes: 18_000_000_000, model_bytes: 48_000_000_000, n_gpu_layers: -1 }
const rosterRam  = { kind: 'ram', model_key: 'qwythos', materialized: true, vram_bytes: 1_300_000_000 }

test('no live rows -> roster untouched (older worker without allocations on the feed)', () => {
  const roster = [rosterSlot, rosterRam]
  assert.equal(mergeAllocations(roster, undefined), roster)
  assert.equal(mergeAllocations(roster, []), roster)
})

test('a slot mid-/load patches healthy/loading/vram onto the roster row', () => {
  const live = [{ kind: 'slot', slot_id: '1', model_key: 'coder', healthy: false, loading: true,
                  materialized: false, vram_bytes: 4_000_000_000 }]
  const out = mergeAllocations([rosterSlot, rosterRam], live)
  const row = out.find(a => a.model_key === 'coder')
  assert.equal(row.healthy, false)
  assert.equal(row.loading, true)
  assert.equal(row.materialized, false)
  assert.equal(row.vram_bytes, 4_000_000_000)
  assert.equal(row.model_bytes, 48_000_000_000, 'roster-only fields are kept')
  assert.equal(out.find(a => a.model_key === 'qwythos').loading, undefined, 'untouched rows keep shape')
})

test('load done: healthy flips true and the loading flag clears even when the feed omits it', () => {
  const loading = mergeAllocations([rosterSlot], [{ kind: 'slot', slot_id: '1', model_key: 'coder', healthy: false, loading: true }])
  assert.equal(loading[0].loading, true)
  const done = mergeAllocations(loading, [{ kind: 'slot', slot_id: '1', model_key: 'coder', healthy: true, vram_bytes: 18_800_000_000 }])
  assert.equal(done[0].healthy, true)
  assert.equal(done[0].loading, false)
  assert.equal(done[0].vram_bytes, 18_800_000_000)
})

test('a seat that changed hands is replaced by the live row (no stale occupant + newcomer pair)', () => {
  const live = [{ kind: 'slot', slot_id: '1', model_key: 'deepseek', healthy: false, loading: true }]
  const out = mergeAllocations([rosterSlot], live)
  assert.equal(out.length, 1)
  assert.equal(out[0].model_key, 'deepseek')
  assert.equal(out[0].loading, true)
})

test('a loading model with no roster row yet is appended (pre-claim / cold admission)', () => {
  const live = [{ kind: 'ram', model_key: 'newbie', loading: true, materialized: false }]
  const out = mergeAllocations([rosterRam], live)
  assert.equal(out.length, 2)
  assert.deepEqual(out[1], { kind: 'ram', model_key: 'newbie', loading: true, materialized: false })
})

test('a roster row still flagged loading but absent from the live feed is un-flagged (refused/finished)', () => {
  const out = mergeAllocations([{ ...rosterRam, loading: true }], [{ kind: 'slot', slot_id: '2', model_key: 'other', healthy: true }])
  assert.equal(out.find(a => a.model_key === 'qwythos').loading, false)
})

test('anyLoadingIn reads the liveness rows first, then the roster', () => {
  assert.equal(anyLoadingIn({ liveness: [{ loading: ['x'] }] }), true)
  assert.equal(anyLoadingIn({ liveness: [{ loading: [] }], workers: [{ loading: ['x'] }] }), false)
  assert.equal(anyLoadingIn({ workers: [{ loading: ['x'] }] }), true)
  assert.equal(anyLoadingIn({}), false)
})
