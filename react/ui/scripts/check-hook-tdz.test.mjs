// node --test scripts/check-hook-tdz.test.mjs — the hook-order TDZ scanner
// (a) flags the CallsPanel shape that crashed the live console 2026-09-29,
// (b) does not flag references from callback bodies, (c) finds nothing in src/.
import test from 'node:test'
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { scanPaths, scanSource } from './check-hook-tdz.mjs'

const here = dirname(fileURLToPath(import.meta.url))

test('flags a hook deps array naming a const declared further down (CallsPanel c769a19)', () => {
  const src = `
    import { useCallback, useEffect } from 'react'
    export default function CallsPanel() {
      const cancelCall = useCallback(async () => { load() }, [load])
      const load = useCallback(() => {}, [])
      useEffect(() => { load() }, [load])
      return null
    }`
  const f = scanSource(src, 'CallsPanel.jsx')
  assert.equal(f.length, 1)
  assert.equal(f[0].name, 'load')
  assert.match(f[0].msg, /before its declaration/)
})

test('flags a useMemo factory that reads a later const (factory runs during render)', () => {
  const src = `
    import { useMemo } from 'react'
    function P() {
      const rows = useMemo(() => filtered.slice(0, 5), [])
      const filtered = []
      return rows
    }`
  assert.equal(scanSource(src).length, 1)
})

test('does not flag references from bodies that run later (useCallback / useEffect / handlers)', () => {
  const src = `
    import { useCallback, useEffect } from 'react'
    function P() {
      const cancel = useCallback(() => { load() }, [])
      useEffect(() => { load() }, [])
      const el = <button onClick={() => load()}>x</button>
      const load = useCallback(() => {}, [])
      return el
    }`
  assert.deepEqual(scanSource(src), [])
})

test('react/ui/src has no hook-order TDZ', () => {
  const findings = scanPaths([resolve(here, '../src')])
  assert.deepEqual(findings.map(f => f.msg), [])
})
