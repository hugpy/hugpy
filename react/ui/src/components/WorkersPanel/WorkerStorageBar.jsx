import FixDoc from '../FixDoc/FixDoc'
import { BudgetTrack } from './BudgetBars'
import { fmtBytes, fmtServed } from './formatters'
import { storageBadge } from './storageBadge'

// ── Per-worker local STORAGE: what model cache this box holds ─────────────────
// The always-on monitoring depiction (sibling to the VRAM/RAM WorkerBudgetBar):
// cache_used vs budget, a per-model cached-files list (size + last-served +
// protection), and — only when the worker is over budget — a proposal-only
// eviction review. Everything here is derived server-side in storage_proposal;
// this component renders `worker.storage` and computes nothing. Nothing deletes
// without the explicit Approve click (human-in-the-loop, central-approved).
export function WorkerStorageBar({ worker, onApproveEvictions, sizeByKey, detailsOnly = false }) {
  const GIB = 2 ** 30
  const s = worker.storage
  // Degrade cleanly for a pre-roll worker that hasn't reported the storage
  // survey yet (older agent) — the RAM row already shows raw disk-free.
  if (!s || !s.reported) return null

  const allModels = Array.isArray(s.models) ? s.models : []
  // ── SHARED CATALOG vs THIS WORKER'S CACHE (k60, operator 2026-07-31) ──────
  // A row on the shared/central catalog (or on a store the box never declared
  // reapable) is on disk here but is NOT in this worker's eviction economy: it
  // can never be deleted from here, so central prices it at zero. Split it out
  // of the priced list and render it in its own section — otherwise the used
  // figure shrinking looks like models vanished, when they simply stopped being
  // billed to a budget that was never theirs.
  const sharedModels = allModels.filter(m => m.counts_toward_budget === false)
  const models = allModels.filter(m => m.counts_toward_budget !== false)
  const sharedBytes = s.unbudgeted_bytes != null
    ? s.unbudgeted_bytes
    : sharedModels.reduce((a, m) => a + (m.bytes || 0), 0)
  // RESIDENT bytes — what is actually ON DISK. The gauge fills to this, NEVER to
  // the attributed/assigned set: assignment is lazy (bytes arrive on first call),
  // so an over-subscribed ATTRIBUTION must never render as disk pressure. Central
  // now sends an explicit hot-drive figure (gauge_used_bytes / hot_bytes,
  // canonical STATE-MODEL.md #7); fall back to cache_used_bytes.
  const used = (s.gauge_used_bytes != null ? s.gauge_used_bytes
               : (s.hot_bytes != null ? s.hot_bytes
               : (s.cache_used_bytes || 0)))
  const cap = s.budget_basis === 'cap'
  const over = !!s.over_budget
  const proposed = Array.isArray(s.proposed_evictions) ? s.proposed_evictions : []
  const proposedKeys = new Set(proposed.map(p => p.model_key))

  // Bar total: reserve-mode carves the held-out reserve from the free tail so a
  // shrinking free area visibly collides with the reserve segment as it fills;
  // cap-mode caps at the ceiling. Never below `used`, so an over-budget bar
  // reads full rather than overflowing.
  const total = cap
    ? Math.max(s.budget || 0, used, 1)
    : Math.max(used + (s.disk_free || 0), used, 1)
  const reserveBytes = cap ? 0 : (s.reserve || 0)

  // "on disk" not "cached" — the gauge is RESIDENT bytes. The assigned/attributed
  // set is reported separately (the "assigned … over allocation" pill below), so
  // the two can never be conflated into a false disk-pressure reading.
  const head = `${fmtBytes(used)} / ${s.budget != null ? fmtBytes(s.budget) : '?'} on disk`
    + ` · ${models.length} model${models.length === 1 ? '' : 's'} resident`

  // ── ALLOCATION-LEVEL view (operator, 2026-07-16) ──────────────────────────
  // `used` is what LANDED on disk; lazy-download means an assigned model often
  // has no files yet, so a worker can read comfortably-under while its ASSIGNED
  // SET cannot possibly fit. That deficit is STRUCTURAL — no eviction order
  // rescues it — so surface it BEFORE some unlucky call wedges, not only in a
  // refusal. Central computes these (storage_proposal.allocated_totals).
  const allocOver = s.allocated_over_budget_bytes || 0
  const allocUnknown = s.allocated_unknown_count || 0
  const allocTotal = s.allocated_total_bytes || 0
  const overSubscribed = allocOver > 0

  // ── UNATTRIBUTED ON DISK (2026-07-17 addendum) ────────────────────────────
  // A THIRD class beside attributed and resident-attributed: bytes on disk that
  // match NO current assignment — a leftover model dir or a STALLED *.part set
  // from an abandoned pull (computron held 5.7G of Qwen2.5-VL-3B .part junk that
  // was invisible because the panel rendered the allocation ledger only). Code
  // calls it "orphaned"; the UI says "unattributed on disk" (attribution vocab).
  const orphanBytes = s.orphaned_bytes || 0
  const orphanCount = s.orphaned_count || 0
  const orphanItems = Array.isArray(s.orphaned_items) ? s.orphaned_items : []
  const orphanTitle = orphanCount === 0 ? '' : (
    `${fmtBytes(orphanBytes)} across ${orphanCount} item(s) sit on this worker's disk `
    + `but are attributed to NO current model — leftover dirs or stalled partial `
    + `downloads (*.part) from an abandoned pull. This is NOT in the assigned set `
    + `and NOT a resident model; it is residue eating the drive.\n\n`
    + orphanItems.slice(0, 12).map(o =>
        `  • ${o.path} — ${fmtBytes(o.bytes)}${o.kind === 'partial' ? ' (stalled .part)' : ''}`
      ).join('\n')
    + (orphanItems.length > 12 ? `\n  …and ${orphanItems.length - 12} more` : ''))
  // "≥" when some sizes are unknown: the total is a FLOOR, never a precise
  // claim. Unknowns are counted and shown — a silent 0 would make an
  // over-subscribed set look fine, which is the dishonesty this exists to end.
  const allocFigs = `${allocUnknown ? '≥' : ''}${fmtBytes(allocTotal)}`
  const allocTitle = `The ${s.allocated_count} model(s) ASSIGNED to this worker total `
    + `${allocFigs}${allocUnknown ? ` (${allocUnknown} of unknown size — this total is a floor)` : ''}, `
    + `against a ${fmtBytes(s.budget)} budget — ${fmtBytes(allocOver)} OVER.\n\n`
    + `This is the assignment set, not what is on disk: models download lazily, `
    + `on first call. The set as a whole cannot fit, so eviction cannot save it — `
    + `some call will eventually be refused no matter which model asks.\n\n`
    + `Unassign models from this worker, raise its allocation (disk_cache_gib), `
    + `or route some of them to another box.`

  // Why the used figure excludes the shared rows — spelled out, because the
  // operator's alarm was a number, and the fix is a number getting smaller.
  const sharedTitle = sharedModels.length === 0 ? '' : (
    `${fmtBytes(sharedBytes)} across ${sharedModels.length} model(s) sit on a store this worker `
    + `may never delete from — the SHARED central catalog (${s.store_root_shared ? 'this box\'s model root IS that catalog' : 'mounted and read through from here'})`
    + `, or a store the box never declared local & disposable.\n\n`
    + `They are shown for visibility but count ZERO toward "${fmtBytes(used)} on disk" and toward the `
    + `over-budget math: the eviction economy is this worker's OWN cache. They can never be proposed `
    + `for eviction, and every delete-time guard refuses them independently.`)

  // Models the worker REFUSED to pull for lack of storage. They have NO files
  // on disk (that is the point — the download never started), so they are not
  // in s.models and must be synthesized as rows; otherwise a model the operator
  // asked for would simply be absent from the console with no explanation.
  const refusedMap = (s.refused && typeof s.refused === 'object') ? s.refused : {}
  const refusedRows = Object.entries(refusedMap).map(([model_key, reason]) => ({
    model_key, bytes: 0, refused: reason, protected: false, last_picked: null,
  }))

  // List order: refused FIRST (missing + actionable — the operator asked for it
  // and it isn't there), then proposed/evictable-cold (what they act on next),
  // protected last; within a tier, least-recently-served first.
  const rows = [...refusedRows, ...models].sort((a, b) => {
    const ap = a.refused ? -1 : (proposedKeys.has(a.model_key) ? 0 : (a.protected ? 2 : 1))
    const bp = b.refused ? -1 : (proposedKeys.has(b.model_key) ? 0 : (b.protected ? 2 : 1))
    if (ap !== bp) return ap - bp
    return (a.last_picked || 0) - (b.last_picked || 0)
  })

  return (
    <div className={`wp-storage${over ? ' wp-storage-over' : ''}`}>
      {!detailsOnly && (
        <div className="wp-storage-head"
             title={`Model weights RESIDENT (on disk) on ${worker.disk?.root || 'the model-root volume'}: `
               + `${fmtBytes(used)} across ${models.length} model(s) — NOT the assigned set `
               + `(models download lazily, so attribution ≠ disk usage). `
               + (cap ? `explicit cap ${fmtBytes(s.budget)}` : `budget ${fmtBytes(s.budget)} (keeps ${fmtBytes(reserveBytes)} disk free in reserve)`)
               + '. Loaded / 🔒static / assigned models are protected and never proposed. '
               + '📌 Pinned models are NOT protected — pin keeps the allocation/routing, not the files, so a pinned model can be proposed for eviction (its bytes re-pull on next call).'}>
          <span className="wp-storage-icon">💾 storage</span>
          <span className="wp-storage-figs">{head}</span>
          <span className="wp-storage-basis">{cap ? 'cap' : 'reserve'}</span>
          {over && (
            <span className="wp-storage-warn"
                  title={`Over budget by ${fmtBytes(s.need_bytes)} — the review below proposes evicting the coldest unprotected models to get back under.`}>
              ⚠ over budget · {fmtBytes(s.need_bytes)} over<FixDoc doc="worker-over-budget" />
            </span>
          )}
          {overSubscribed && (
            <span className="wp-storage-warn" title={allocTitle}>
              ⚠ assigned {allocFigs} · {fmtBytes(allocOver)} over allocation
            </span>
          )}
          {orphanCount > 0 && (
            <span className="wp-storage-orphan" title={orphanTitle}>
              🧹 unattributed on disk: {fmtBytes(orphanBytes)} · {orphanCount} item{orphanCount === 1 ? '' : 's'}
            </span>
          )}
          {sharedModels.length > 0 && (
            <span className="wp-storage-shared-chip" title={sharedTitle}>
              🔗 shared catalog: {fmtBytes(sharedBytes)} · {sharedModels.length} model{sharedModels.length === 1 ? '' : 's'} (never evicted)
            </span>
          )}
        </div>
      )}

      {!detailsOnly && s.budget != null && (
        <BudgetTrack label="DISK" total={total} used={used}
                     reserve={reserveBytes} reserveGib={reserveBytes ? Math.round(reserveBytes / GIB) : 0}
                     note={cap ? 'model cache vs. explicit per-worker cap'
                               : 'model cache; hatched tail = disk reserve kept free'} />
      )}

      {detailsOnly && over && (
        <div className="wp-storage-warn wp-storage-warn-solo"
             title="Over budget — the proposal below evicts the coldest unprotected models to get back under.">
          ⚠ over budget · {fmtBytes(s.need_bytes)} over<FixDoc doc="worker-over-budget" />
        </div>
      )}
      {detailsOnly && rows.length === 0 && sharedModels.length === 0 && (
        <div className="wp-res-empty">No model files cached on this worker yet.</div>
      )}
      {detailsOnly && rows.length === 0 && sharedModels.length > 0 && (
        <div className="wp-res-empty" title={sharedTitle}>
          No model files in this worker&apos;s own cache — everything below is on the shared catalog.
        </div>
      )}
      {refusedRows.length > 0 && (
        <div className="wp-storage-warn wp-storage-warn-solo"
             title={'These models were REQUESTED but could not be downloaded: even after '
               + 'evicting every cold, unprotected model, they would not fit under this '
               + "worker's storage allocation. The pulls were refused BEFORE they started, "
               + 'so no partial files and no wasted disk. Hover a row for its exact numbers.'}>
          ⊘ {refusedRows.length} model{refusedRows.length === 1 ? '' : 's'} missing — won&apos;t fit under this worker&apos;s storage budget
        </div>
      )}

      {rows.length > 0 && (
        <div className="wp-storage-list">
          {rows.map(m => {
            const isProp = proposedKeys.has(m.model_key)
            const badge = storageBadge(m, isProp)
            // A GGUF dir may hold several quants; the one that serves is `eff`.
            // Show the true on-disk total (what a reclaim frees) + the quant note.
            const eff = sizeByKey && sizeByKey.get(m.model_key)
            const showQuant = eff && m.bytes > eff.eff * 1.05
            return (
              <div key={m.model_key}
                   className={`wp-storage-row${m.protected ? ' wp-storage-protected' : ''}${isProp ? ' wp-storage-proposed' : ''}${m.refused ? ' wp-storage-refused' : ''}`}>
                <span className={`wp-state-pill ${badge.pill}`} title={badge.title}>{badge.glyph}</span>
                <span className="wp-storage-name" title={m.model_key}>{m.model_key}</span>
                {m.refused ? (
                  // Not "0 B" (which reads as a real, empty file-set): show what
                  // it WOULD need, so the row explains itself without a hover.
                  <span className="wp-storage-size" title={badge.title}>
                    <em className="wp-storage-quant">needs {fmtBytes(m.refused.needs_bytes || 0)}</em>
                  </span>
                ) : (
                  <span className="wp-storage-size"
                        title={showQuant ? `${fmtBytes(m.bytes)} on disk (all variants); effective quant ${eff.effGguf || ''} ${fmtBytes(eff.eff)}` : undefined}>
                    {fmtBytes(m.bytes)}
                    {showQuant && <em className="wp-storage-quant"> · quant {fmtBytes(eff.eff)}</em>}
                  </span>
                )}
                <span className="wp-storage-served" title={m.refused
                  ? 'never served — the files were never downloaded'
                  : 'Last time central routed a request to this (worker, model)'}>
                  {m.refused ? '—' : fmtServed(m.last_picked)}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {sharedModels.length > 0 && (
        <div className="wp-storage-shared">
          <div className="wp-storage-shared-head" title={sharedTitle}>
            🔗 shared catalog (never evicted) — {fmtBytes(sharedBytes)} across {sharedModels.length} model
            {sharedModels.length === 1 ? '' : 's'}, on a store this worker may not delete from.
            Not counted toward the budget above.
          </div>
          <div className="wp-storage-list">
            {sharedModels
              .slice()
              .sort((a, b) => (b.bytes || 0) - (a.bytes || 0))
              .map(m => {
                const badge = storageBadge(m, false)
                return (
                  <div key={m.model_key} className="wp-storage-row wp-storage-protected">
                    <span className={`wp-state-pill ${badge.pill}`} title={badge.title}>{badge.glyph}</span>
                    <span className="wp-storage-name" title={m.model_key}>{m.model_key}</span>
                    <span className="wp-storage-size" title="on the shared store — not billed to this worker's budget">
                      {fmtBytes(m.bytes)}
                    </span>
                    <span className="wp-storage-served" title="Last time central routed a request to this (worker, model)">
                      {fmtServed(m.last_picked)}
                    </span>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {over && proposed.length > 0 && (
        <div className="wp-storage-review">
          <div className="wp-storage-review-head">
            Eviction proposal — frees {fmtBytes(s.proposed_free_bytes)} by deleting {proposed.length} cold,
            unprotected model{proposed.length === 1 ? '' : 's'} (least-recently-served first):
          </div>
          <div className="wp-storage-review-list">
            {proposed.map(p => (
              <div key={p.model_key} className="wp-storage-review-row">
                <span className="wp-storage-name" title={p.model_key}>{p.model_key}</span>
                <span className="wp-storage-size">{fmtBytes(p.bytes)}</span>
                <span className="wp-storage-served">{fmtServed(p.last_picked)}</span>
              </div>
            ))}
          </div>
          <button className="wp-storage-approve" onClick={() => onApproveEvictions && onApproveEvictions(worker)}
                  title="Delete these files now. Central re-checks the proposal at approval time and the worker re-proves every guard per model before deleting — protected (loaded / 🔒static / assigned) files are never touched. 📌 Pinned files ARE eligible: pin keeps the allocation, not the bytes, which re-pull on next call.">
            ✓ Approve &amp; free {fmtBytes(s.proposed_free_bytes)}
          </button>
        </div>
      )}
    </div>
  )
}
