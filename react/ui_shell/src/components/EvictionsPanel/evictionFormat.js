// Eviction telemetry — the pure display helpers, shared by the full Evictions
// panel and the compact feed embedded in the Models tab.
//
// Extracted so the two views cannot drift: the same bytes are labelled the same
// way, the same run folds into the same rows, and a refusal reason is flattened
// once. Nothing here touches React or the network.

// Binary units, correctly labelled (KiB/MiB/GiB — 1024-based).
export function fmtBytes(n) {
  if (n == null) return '?'
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

export function fmtMs(ms) {
  if (ms == null) return null
  const n = Number(ms)
  if (n < 1000) return `${Math.round(n)} ms`
  if (n < 60_000) return `${(n / 1000).toFixed(1)}s`
  const m = Math.floor(n / 60_000)
  return `${m}m ${String(Math.round((n % 60_000) / 1000)).padStart(2, '0')}s`
}

export function fmtClock(ts) {
  if (!ts) return ''
  try { return new Date(Number(ts) * 1000).toLocaleTimeString() } catch { return '' }
}

// `reason` is a string on the simple paths and the honest-refusal DICT on the
// interesting ones. Flatten either into one readable line — never let an object
// reach the DOM as [object Object].
export function fmtReason(r) {
  if (r == null) return ''
  if (typeof r === 'string') return r
  if (typeof r === 'number' || typeof r === 'boolean') return String(r)
  if (Array.isArray(r)) return r.map(fmtReason).filter(Boolean).join(', ')
  if (typeof r === 'object') {
    return Object.entries(r)
      .map(([k, v]) => {
        if (v == null) return null
        const vs = (typeof v === 'object') ? fmtReason(v) : String(v)
        return vs === '' ? null : `${k}: ${vs}`
      })
      .filter(Boolean)
      .join(' · ')
  }
  return String(r)
}

// De-dup key. `_id` is authoritative; (worker_id, seq) covers events that
// arrive without one so the SSE replay can't double-render the backfill.
export function eventKey(ev) {
  if (ev && ev._id) return `id:${ev._id}`
  return `sq:${ev?.worker_id ?? '?'}:${ev?.seq ?? Math.random()}`
}

export const OUTCOME_CLASS = {
  fit: 'ev-out-fit',
  partial: 'ev-out-partial',
  refused: 'ev-out-refused',
  'proceeded-unfit': 'ev-out-unfit',
}

export function tierClass(tier) {
  return `ev-tier ev-tier-${String(tier || 'unknown').replace(/[^a-z0-9]+/gi, '-')}`
}

// Last in-flight victim row for a model (hand-rolled rather than
// Array.findLastIndex, which is ES2023 and outside this build's lib target).
function lastInFlight(rows, modelKey) {
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i]
    if (r.kind === 'victim' && r.model_key === modelKey && r.state === 'running') return i
  }
  return -1
}

// Same walk for the serve-pipeline rows. `match` narrows beyond the model_key
// because a single run tries SEVERAL provision sources in sequence for the same
// model (local, then central-transfer, then hf) — keyed on model_key alone, the
// second source's .done would upgrade the first source's row.
function lastInFlightOf(rows, kind, match) {
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i]
    if (r.kind === kind && r.state === 'running' && match(r)) return i
  }
  return -1
}

// A provision.fail's operator line. The backend composes `human` whenever it has
// enough to say something true ("disk full (ENOSPC) on /mnt/storage — 0 B free of
// 938 GB"); it knows the mount and the free/total bytes, so it always beats
// anything we could reassemble here. Fall back to the parts only when it didn't.
export function provisionFailText(r) {
  if (!r) return ''
  if (r.human) return r.human
  const bits = [r.errno_name, r.detail || r.error_class].filter(Boolean)
  return bits.join(': ') || 'provisioning failed'
}

// The short code an operator scans for — ENOSPC, EACCES, the exception class.
export function failCode(r) {
  if (!r) return null
  return r.errno_name || r.error_class || null
}

// ONE LINE, no detail: what the feed shows without expanding. The whole point of
// the 2026-07-28 incident is that "ENOSPC" must be visible from the list.
export function failSummary(r) {
  if (!r) return null
  if (r.kind === 'provision') {
    return ['provision failed', r.source, failCode(r)].filter(Boolean).join(' · ')
  }
  if (r.kind === 'resolve') {
    return ['resolve failed', fmtReason(r.reason) || 'no path'].filter(Boolean).join(' · ')
  }
  if (r.kind === 'load') {
    return ['load failed', r.engine, fmtReason(r.error)].filter(Boolean).join(' · ')
  }
  if (r.kind === 'victim') {
    return ['eviction failed', r.model_key, fmtReason(r.error)].filter(Boolean).join(' · ')
  }
  return null
}

// The full sentence — the `title` tooltip and the expanded detail line.
export function failDetail(r) {
  if (!r) return null
  if (r.kind === 'provision') return provisionFailText(r)
  if (r.kind === 'resolve') {
    const why = fmtReason(r.reason) || 'could not resolve a path'
    return r.resolved_path ? `${why} (${r.resolved_path})` : why
  }
  if (r.kind === 'load') return fmtReason(r.error) || 'load failed'
  if (r.kind === 'victim') return fmtReason(r.error) || 'eviction failed'
  return null
}

// Every row kind that can carry a failure, in the order the pipeline runs them.
function isFailRow(r) {
  return r.state === 'failed'
    && (r.kind === 'provision' || r.kind === 'resolve' || r.kind === 'load' || r.kind === 'victim')
}

// ── Fold a run's raw event list into display rows ───────────────────────────
// Rows stay in EVENT ORDER. An evict.start opens an in-flight victim row that
// the matching evict.done / evict.fail upgrades in place, so a completed
// eviction is one row and not three.
//
// SERVE PIPELINE (added 2026-07-28). The same run_id also carries the stages
// that run BEFORE any eviction does — provision (fetch the weights), resolve
// (find them on disk), load (hand them to the engine). They are here because of
// a real incident: a worker's disk was 100% full, provisioning died with ENOSPC
// before the headroom pass ever ran, and this feed showed only "trigger load /
// no candidates walked". Nothing in the console named the disk; the operator had
// to ssh the box and read journalctl to find out why a request failed. A failed
// request must say WHERE it died and WHY, in the card.
export function buildRun(run) {
  const evs = run.events
  const rows = []
  let incoming = null, trigger = null, tier = null
  let done = null, needBytes = null
  // MODEL GROUPS: set when a group's tick DEMANDED this pass ({group_key, tick}).
  let group = null

  for (const ev of evs) {
    if (ev.incoming_model && !incoming) incoming = ev.incoming_model
    if (ev.need_bytes != null && needBytes == null) needBytes = ev.need_bytes

    switch (ev.stage) {
      // MODEL GROUPS. These are the FIRST stages of a run when a group was
      // consulted — which iteration of a base model is serving, and why every
      // other one lost. Without them the card would open with a
      // provision.start for a model key the operator never typed.
      case 'member.select':
        rows.push({
          kind: 'member', key: eventKey(ev), state: 'done',
          group_key: ev.group_key, model_key: ev.model_key,
          as: ev.as, reason: ev.reason, verdict: ev.verdict,
          need_bytes: ev.need_bytes, demanded_by: ev.demanded_by,
        })
        break
      case 'member.skip':
        rows.push({
          kind: 'memberskip', key: eventKey(ev),
          group_key: ev.group_key, model_key: ev.model_key, reason: ev.reason,
        })
        break
      case 'headroom.start':
        trigger = ev.trigger || trigger
        tier = ev.tier || tier
        // A pass DEMANDED by a group's tick says so, so an operator can tell it
        // from ordinary contention. Absent on every non-group eviction.
        if (ev.group) group = ev.group
        break
      case 'provision.start':
        rows.push({
          kind: 'provision', key: eventKey(ev), state: 'running',
          model_key: ev.model_key, source: ev.source, dest_path: ev.dest_path,
        })
        break
      case 'provision.done': {
        const at = lastInFlightOf(rows, 'provision', r => r.model_key === ev.model_key && r.source === ev.source)
        const patch = { state: 'done', bytes: ev.bytes, duration_ms: ev.duration_ms }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'provision', key: eventKey(ev), model_key: ev.model_key, source: ev.source, ...patch })
        break
      }
      case 'provision.fail': {
        // A .fail with no preceding .start still pushes its own row — same
        // defensive shape as evict.done. The one event that MUST never be
        // dropped is the one that explains the failure.
        const at = lastInFlightOf(rows, 'provision', r => r.model_key === ev.model_key && r.source === ev.source)
        const patch = {
          state: 'failed',
          error_class: ev.error_class, errno_name: ev.errno_name,
          detail: ev.detail, human: ev.human,
          disk_free_bytes: ev.disk_free_bytes, disk_total_bytes: ev.disk_total_bytes,
          disk_mount: ev.disk_mount,
        }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'provision', key: eventKey(ev), model_key: ev.model_key, source: ev.source, ...patch })
        break
      }
      case 'resolve.fail':
        // Resolve only ever reports when it FAILS — a found path is silent, so
        // this row has no in-flight half to upgrade.
        rows.push({
          kind: 'resolve', key: eventKey(ev), state: 'failed',
          model_key: ev.model_key, resolved_path: ev.resolved_path, reason: ev.reason,
        })
        break
      case 'load.start':
        rows.push({
          kind: 'load', key: eventKey(ev), state: 'running',
          model_key: ev.model_key, engine: ev.engine,
        })
        break
      case 'load.done': {
        const at = lastInFlightOf(rows, 'load', r => r.model_key === ev.model_key)
        const patch = { state: 'done', duration_ms: ev.duration_ms, engine: ev.engine || (at >= 0 ? rows[at].engine : null) }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'load', key: eventKey(ev), model_key: ev.model_key, ...patch })
        break
      }
      case 'load.fail': {
        const at = lastInFlightOf(rows, 'load', r => r.model_key === ev.model_key)
        const patch = { state: 'failed', error: ev.error, engine: ev.engine || (at >= 0 ? rows[at].engine : null) }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'load', key: eventKey(ev), model_key: ev.model_key, ...patch })
        break
      }
      case 'fit.fail':
        rows.push({ kind: 'fitfail', key: eventKey(ev), need: ev.need_bytes, free: ev.free_bytes })
        break
      case 'candidate.skip':
        rows.push({
          kind: 'skip', key: eventKey(ev),
          model_key: ev.model_key, tier: ev.tier,
          reason: ev.reason, vram_bytes: ev.vram_bytes, idle_s: ev.idle_s,
        })
        break
      case 'evict.start':
        rows.push({ kind: 'victim', key: eventKey(ev), model_key: ev.model_key, tier: ev.tier, state: 'running' })
        break
      case 'evict.done': {
        const at = lastInFlight(rows, ev.model_key)
        const patch = { state: 'done', freed_bytes: ev.freed_bytes, duration_ms: ev.duration_ms, tier: ev.tier || (at >= 0 ? rows[at].tier : null) }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'victim', key: eventKey(ev), model_key: ev.model_key, ...patch })
        break
      }
      case 'evict.fail': {
        const at = lastInFlight(rows, ev.model_key)
        const patch = { state: 'failed', error: ev.error, tier: ev.tier || (at >= 0 ? rows[at].tier : null) }
        if (at >= 0) rows[at] = { ...rows[at], ...patch }
        else rows.push({ kind: 'victim', key: eventKey(ev), model_key: ev.model_key, ...patch })
        break
      }
      case 'reclaim.done':
        rows.push({ kind: 'reclaim', key: eventKey(ev) })
        break
      case 'makeroom.verdict':
        rows.push({
          kind: 'verdict', key: eventKey(ev),
          action: ev.action, reason: ev.reason,
          evicted: ev.evicted, freed_bytes: ev.freed_bytes,
        })
        break
      case 'headroom.done':
        done = ev
        break
      default:
        break
    }
  }

  // Every model_key this pass TOUCHED, in any role — the incoming model, every
  // skipped candidate, every victim. This is what the Models tab highlights
  // against: the operator calls a model from chat and wants its eviction
  // activity to pop whether it was the thing loading or the thing unloaded.
  const touched = new Set()
  if (incoming) touched.add(incoming)
  for (const r of rows) if (r.model_key) touched.add(r.model_key)
  if (done && Array.isArray(done.evicted)) for (const m of done.evicted) touched.add(m)

  // The run's OVERALL failure. headroom.done is the only thing that closes a run,
  // and a request that dies in provisioning never reaches it — so before this
  // existed such a run sat "running…" forever and read as merely slow. Take the
  // FIRST fail: the pipeline is sequential, so the first thing to break is the
  // cause and everything after it is fallout.
  const failRow = rows.find(isFailRow) || null

  return {
    id: run.id,
    worker_id: run.worker_id,
    startTs: run.startTs,
    endTs: done ? done.ts : null,
    incoming: incoming || (done && done.incoming_model) || null,
    trigger, tier, needBytes, rows, done, touched, group,
    failure: failSummary(failRow),          // one line, for the collapsed views
    failureDetail: failDetail(failRow),     // the full human string, for titles
    failureRow: failRow,
    // everything the filter box searches over. The serve-pipeline parts are in
    // here deliberately: after the ENOSPC incident an operator's first move is to
    // type "enospc" or "central-transfer" and expect the run to surface.
    haystack: [
      run.worker_id, incoming,
      ...rows.map(r => r.model_key),
      ...rows.map(r => r.source),
      ...rows.map(r => r.engine),
      ...rows.map(r => r.errno_name),
      ...rows.map(r => r.error_class),
      // MODEL GROUPS: an operator debugging a group types its key or a tick
      // name ("quality") and expects the runs it caused to surface.
      ...rows.map(r => r.group_key),
      ...rows.map(r => r.as),
      ...rows.map(r => r.reason),
      group && group.group_key, group && group.tick,
    ].filter(Boolean).join(' ').toLowerCase(),
  }
}

// The victims of a folded run, for the compact one-line summary.
export function runVictims(run) {
  return run.rows.filter(r => r.kind === 'victim' && r.state === 'done')
}

export function runFreedBytes(run) {
  let total = 0, any = false
  for (const r of runVictims(run)) {
    if (r.freed_bytes != null) { total += Number(r.freed_bytes); any = true }
  }
  return any ? total : null
}
