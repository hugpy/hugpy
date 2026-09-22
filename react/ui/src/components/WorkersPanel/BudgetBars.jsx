import { fmtBytes } from './formatters'

// One resource track: a used fill + an optional right-aligned striped reserve
// segment (held out of the pool, not allocatable). The caller only renders this
// when total > 0, so the width math never goes NaN.
export function BudgetTrack({ label, total, used, reserve, reserveGib, note }) {
  const usedPct    = total ? Math.min(Math.max((used || 0) / total * 100, 0), 100) : 0
  const reservePct = (reserve && total) ? Math.min(reserve / total * 100, 100) : 0
  const free = used != null ? Math.max(total - used, 0) : null
  const title = `${label}: ${fmtBytes(used || 0)} used / ${fmtBytes(total)}`
    + (free != null ? ` · ${fmtBytes(free)} free` : '')
    + (note ? ` — ${note}` : '')
  return (
    <div className="wp-budget-row" title={title}>
      <span className="wp-budget-label">{label}</span>
      <div className="wp-budget-bar">
        <div className="wp-budget-fill" style={{ width: `${usedPct}%` }} />
        {reservePct > 0 && (
          <div className="wp-budget-reserve" style={{ width: `${reservePct}%` }}
               title={`reserve: ${reserveGib} GiB held out of the pool (not allocatable)`} />
        )}
      </div>
    </div>
  )
}

export function WorkerBudgetBar({ worker }) {
  const GIB = 2 ** 30
  // Residents (engine-agnostic): the unified allocations view is the truth when
  // present — a PRESENT-but-empty [] is honest idle ("0 resident"). Older agents
  // (undefined allocations) fall back to loaded_models so a mixed-version fleet
  // doesn't read every old worker as "0 resident".
  const allocs = Array.isArray(worker.allocations) ? worker.allocations : null
  const residents = allocs
    ? allocs.filter(a => a && a.model_key)
    : (worker.loaded_models || []).map(k => ({ model_key: k }))
  const count = residents.length
  const residentLabel = count === 0
    ? '0 models resident — pool idle'
    : `${count} model${count === 1 ? '' : 's'} resident`

  // The bar FILL is the flat used/total delta — it already folds in reserve,
  // ComfyUI and any non-model process use, a truthful "committed" figure. The
  // per-resident footprint sums drive the chips ONLY; putting them on the bar
  // too would visibly disagree with this. None (never 0) where unknown, so the
  // track hides rather than drawing a fake 0/None bar.
  const vramTotal = worker.vram_total
  const ramTotal  = worker.ram_total
  const hasVram = vramTotal != null && vramTotal > 0
  const hasRam  = ramTotal != null && ramTotal > 0

  // Reserve held out of the pool — from the worker's caps snapshot (optional:
  // {} or missing on unconfigured/older boxes → no reserve segment). ComfyUI is
  // out-of-pool yet still inside vram_used/ram_used, so we say so in the tooltip.
  const caps = worker.caps || {}
  const comfy = !!worker.comfy?.available

  const parts = [residentLabel]
  if (hasVram) parts.push(`${fmtBytes(worker.vram_used || 0)} / ${fmtBytes(vramTotal)} VRAM`)
  if (hasRam)  parts.push(`${fmtBytes(worker.ram_used || 0)} / ${fmtBytes(ramTotal)} RAM`)
  else if (worker.free_ram != null) parts.push(`${fmtBytes(worker.free_ram)} RAM free`)

  return (
    <div className="wp-budget">
      <div className="wp-budget-head" title="Resource pool = VRAM + host RAM. Occupancy is emergent — how many models are packed in right now, not a fixed seat count.">
        {parts.join(' · ')}
      </div>
      {/* VRAM lives solely in the per-GPU chips below (GpuChips) — no duplicate
          GPU bar here. This bar is the host-RAM half of the pool. The per-model
          residents are itemized in the engine-agnostic Slots row, not re-listed
          here (they'd double the residency view). */}
      {hasRam && (
        <BudgetTrack label="RAM" total={ramTotal} used={worker.ram_used}
                     reserve={(caps.ram_reserve_gib || 0) * GIB} reserveGib={caps.ram_reserve_gib}
                     note={`used incl. reserve/headroom + non-model RAM${comfy ? ' + ComfyUI' : ''}`} />
      )}
      {comfy && (
        <div className="wp-budget-comfy" title="ComfyUI is an adopted, externally-owned process (image/video generation). Its checkpoints are NOT worker pool residents — they load on-demand inside ComfyUI. Its memory folds into the used/total figures above as an out-of-pool reservation; it is never itemized as a slot.">
          🧩 ComfyUI attached — external (out-of-pool)
        </div>
      )}
    </div>
  )
}
