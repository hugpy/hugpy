// Test-fire strip filter (operator ask 2026-09-29): chips "All · Passed ·
// Failed · Skipped" with counts over the worker's test-fire rows (the job's
// selected models + its skipped rows — i.e. the models allocated to that
// worker). Choice persists per worker in localStorage.
export const TF_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'passed', label: 'Passed' },
  { id: 'failed', label: 'Failed' },
  { id: 'skipped', label: 'Skipped' },
]
const IDS = new Set(TF_FILTERS.map(f => f.id))

// One row per model: {key, status: 'passed'|'failed'|'pending'|'skipped', pm?, skip?}
export function testFireRows(job) {
  if (!job) return []
  const per = (job.summary && job.summary.per_model) || {}
  const rows = (job.models || []).map(key => {
    const pm = per[key]
    const st = pm && pm.last_status
    return { key, pm, status: st === 'ok' ? 'passed' : st === 'fail' ? 'failed' : 'pending' }
  })
  for (const s of job.skipped || []) rows.push({ key: s.model_key, skip: s, status: 'skipped' })
  return rows
}

export function testFireCounts(rows) {
  const c = { all: rows.length, passed: 0, failed: 0, skipped: 0 }
  for (const r of rows) if (r.status in c) c[r.status] += 1
  return c
}

export function filterTestFireRows(rows, filter) {
  return !filter || filter === 'all' ? rows : rows.filter(r => r.status === filter)
}

const storeKey = workerId => `hugpy.testFire.filter.${workerId}`

export function loadTestFireFilter(workerId, storage = globalThis.localStorage) {
  try {
    const v = storage && workerId ? storage.getItem(storeKey(workerId)) : null
    return IDS.has(v) ? v : 'all'
  } catch { return 'all' }
}

export function saveTestFireFilter(workerId, filter, storage = globalThis.localStorage) {
  try { if (storage && workerId && IDS.has(filter)) storage.setItem(storeKey(workerId), filter) } catch { /* private mode */ }
}
