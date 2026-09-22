import { useState } from 'react'
import { allocModeLabel, deriveAllocMode, resolvedSeatMode } from './allocation'
import { fmtBytes } from './formatters'
import { residentState, vramBytesFor } from './workerMetrics'
import { WorkerStorageBar } from './WorkerStorageBar'

// Per-GPU chip with a used/free VRAM bar.
export function GpuChips({ gpus }) {
  if (!gpus || !gpus.length) return <span className="wp-nogpu">no GPU reported</span>
  return (
    <div className="wp-gpus">
      {gpus.map((g, i) => {
        const total = g.memory_total, free = g.memory_free
        const used = (total != null && free != null) ? Math.max(total - free, 0) : null
        const pct = (used != null && total) ? Math.min((used / total) * 100, 100) : 0
        return (
          <div key={i} className="wp-gpu" title={g.name || ''}>
            <div className="wp-gpu-head">
              🖥 {g.name || `GPU ${g.index ?? i}`}
              {used != null && (
                <em> · {fmtBytes(used)} used / {fmtBytes(total)}</em>
              )}
            </div>
            {used != null && (
              <div className="wp-vram-bar" title={`${fmtBytes(free)} free`}>
                <div className="wp-vram-fill" style={{ width: `${pct}%` }} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Per-worker PID registry (model→pid→VRAM log + foreign GPU squatters) ─────
// Lives in the VRAM (GPU-card) expansion. Two lists, visually distinct:
//   • ATTRIBUTED (pid_registry.models): models THIS worker spawned — each carries
//     a model_key, its owning pid, host_mode, measured VRAM and an alive flag.
//     Each gets an "×" evict button → POST /llm/workers/<id>/evict {model_key},
//     which frees the model's VRAM/RAM (idempotent; reloads on next request).
//   • UNATTRIBUTED (pid_registry.unattributed): foreign GPU processes the worker
//     did NOT spawn (e.g. ComfyUI, a rogue script). Shown muted + clearly labelled
//     "not owned by hugpy". Their "×" is DISABLED — the evict verb only frees
//     hugpy-owned models; killing a foreign pid needs a privileged worker helper
//     that does not exist yet. See the TODO below.
// Degrades to nothing when the worker reports no pid_registry (older agent / no GPU).
export function PidRegistry({ worker, onEvict }) {
  const reg = worker && worker.pid_registry
  const [evicting, setEvicting] = useState(null)  // model_key currently in-flight
  if (!reg) return null
  const models = Array.isArray(reg.models) ? reg.models : []
  const foreign = Array.isArray(reg.unattributed) ? reg.unattributed : []
  if (!models.length && !foreign.length) return null

  const doEvict = async (mk) => {
    if (!onEvict || evicting) return
    setEvicting(mk)
    try { await onEvict(worker, mk) } finally { setEvicting(null) }
  }

  // E/M (k65): this IS the "mirror nvidia-smi exactly" view. The SIZE of every
  // row is the worker's MEASURED per-PID mib (attribution adds only the label),
  // so the rows sum BYTE-FOR-BYTE to nvidia-smi's compute-apps total — the
  // footer proves it. Raw MiB is the summed value (MIB below), GiB is render-only
  // via fmtBytes (which divides by 1024) — never round-trip a rounded GiB back.
  const MIB = 1024 * 1024
  const measuredSum = models.reduce((s, m) => s + (Number(m.vram_bytes) || 0), 0)
  const foreignSum = foreign.reduce((s, p) => s + (Number(p.mib) || 0) * MIB, 0)
  const smiTotal = measuredSum + foreignSum
  // A row's human label: a served model shows its key; a worker-infra / comfy /
  // idle row (model_key null) shows its host-mode label instead of a blank.
  const rowLabel = (m) => m.model_key || m.label || m.display_label
    || (m.host_mode === 'cuda_context' ? 'agent CUDA context'
      : m.host_mode === 'comfy' ? 'ComfyUI' : (m.host_mode || 'unknown'))

  return (
    <div className="wp-pidreg">
      {models.length > 0 && (
        <>
          <div className="wp-res-detail-label">GPU process registry — model → pid → measured VRAM (mirrors nvidia-smi · {models.length})</div>
          <div className="wp-pidreg-list">
            {models.map((m, i) => {
              const isModel = !!m.model_key
              const busy = isModel && evicting === m.model_key
              const alive = m.alive !== false
              // PLANNED beside MEASURED (E/M): an in-process row carries its torch
              // weight estimate; the measured figure also holds the CUDA context /
              // KV, so they legitimately differ. Show the disagreement — never blend.
              const planned = m.vram_bytes_planned
              const measured = m.vram_bytes
              const showPlanned = planned != null && measured != null
                && Math.abs(Number(planned) - Number(measured)) > MIB
              return (
                <div key={`${m.model_key || m.host_mode}-${m.pid}-${i}`} className="wp-pidreg-row" title={`${rowLabel(m)} · pid ${m.pid} · ${m.host_mode || 'unknown host'}`}>
                  <span className={`wp-pidreg-dot ${alive ? 'wp-pidreg-alive' : 'wp-pidreg-dead'}`}
                        title={alive ? 'process alive' : 'process gone (stale entry)'}>{alive ? '●' : '○'}</span>
                  <span className="wp-pidreg-name">{rowLabel(m)}</span>
                  <span className="wp-pidreg-meta">pid {m.pid}</span>
                  <span className="wp-pidreg-mode" title={`host mode: ${m.host_mode || 'unknown'}`}>{m.host_mode || '—'}</span>
                  <span className="wp-pidreg-vram" title={showPlanned
                        ? `measured ${fmtBytes(measured)} (nvidia-smi) vs planned ${fmtBytes(planned)} (declared weights) — the gap is CUDA context / KV`
                        : 'measured VRAM from nvidia-smi per-PID'}>
                    {measured != null ? fmtBytes(measured) : '—'}
                    {showPlanned && <span className="wp-fact-est"> (plan ~{fmtBytes(planned)})</span>}
                  </span>
                  <button className="wp-model-x wp-pidreg-x" disabled={busy || !onEvict || !isModel}
                          title={!isModel ? 'worker infrastructure / external — not an evictable model'
                            : busy ? 'evicting…' : `Evict ${m.model_key} — free its VRAM/RAM (reloads on next request)`}
                          onClick={() => isModel && doEvict(m.model_key)}>{busy ? '⏳' : '×'}</button>
                </div>
              )
            })}
          </div>
        </>
      )}

      {foreign.length > 0 && (
        <>
          <div className="wp-res-detail-label wp-pidreg-foreign-label"
               title="Unattributed GPU processes this worker did NOT spawn and could not recognize — hugpy cannot manage their lifecycle. Item I's 22 GiB zombie would surface HERE.">
            Unattributed / foreign GPU processes ({foreign.length})
          </div>
          <div className="wp-pidreg-list wp-pidreg-foreign">
            {foreign.map((p, i) => (
              <div key={`${p.pid}-${i}`} className="wp-pidreg-row wp-pidreg-foreign-row"
                   title="Unattributed — this worker did not spawn it and could not recognize it. Its VRAM folds into the GPU's used total but it is not a pool resident.">
                <span className="wp-pidreg-dot wp-pidreg-foreign-dot">◈</span>
                <span className="wp-pidreg-name">{p.name || 'unknown process'}</span>
                <span className="wp-pidreg-meta">pid {p.pid}</span>
                <span className="wp-pidreg-vram" title="measured VRAM from nvidia-smi per-PID">{p.mib != null ? fmtBytes(Number(p.mib) * MIB) : '—'}</span>
                {/* TODO(keeper): kill-by-pid needs the privileged worker helper (operator privilege decision pending) */}
                <button className="wp-model-x wp-pidreg-x" disabled
                        title="foreign process — kill requires a privileged worker helper (not yet enabled)">×</button>
              </div>
            ))}
          </div>
        </>
      )}

      {/* BYTE-EXACT ledger (E/M acceptance, k65): attributed + foreign rows sum
          to nvidia-smi's compute-apps total by construction (every PID counted
          once, sized by its measured mib). If this ever disagrees with the GPU
          chip's used bar, the difference is KV/activations not tied to a
          compute-app PID — never a rounding artifact. */}
      <div className="wp-pidreg-total" title="Sum of every attributed model row + unattributed row, each sized by its measured nvidia-smi per-PID mib. Mirrors nvidia-smi compute-apps byte-for-byte.">
        Σ measured = {fmtBytes(smiTotal)}
        {foreignSum > 0 && <span className="wp-fact-est"> ({fmtBytes(measuredSum)} attributed + {fmtBytes(foreignSum)} unattributed)</span>}
        {' — mirrors nvidia-smi'}
      </div>
    </div>
  )
}

// Per-model VRAM byte footprint (weights resident on the GPU). A GGUF SLOT's
// unified allocation entry carries only its host RSS, so the weight bytes come
// from loaded_detail. Fully offloaded (n_gpu_layers === -1) → the whole weight;
// partial → prorate by the offloaded fraction when total layers are known;
// null when the model isn't on the GPU (it lives in host RAM). EXCLUDES the KV
// cache (not reported per model) — the reconcile line accounts for that gap.
// The row renderer and the reconcile sum both call this, so they can't drift.
//
// PROVENANCE (2026-07-28 ruling — worker measurements are the truth): the
// return carries WHERE the number came from, because the fallback below is not
// a measurement — it is declared placement × file size, and rendering it the
// same as an nvidia-smi read launders a guess into a fact. Returns
// null | { bytes, estimated }: estimated === false → measured by the worker;
// estimated === true → derived from declared bytes. The fallback STAYS (on a
// box with a broken nvidia-smi it is the only VRAM signal there is) — every
// caller must just mark it: `~` prefix, muted style, honest tooltip.
export const ESTIMATED_VRAM_TITLE =
  'estimated from declared placement × file size — not measured'

// One resident row (a model occupying VRAM or host RAM). Shared by the VRAM and
// RAM resource details; engine-agnostic via residentState.
export function ResidentList({ items, loadedSet, worker, resource, emptyLabel, sizeByKey }) {
  if (!items || !items.length) return <div className="wp-res-empty">{emptyLabel}</div>
  return (
    <div className="wp-models wp-res-models">
      {items.map((a, i) => {
        const st = residentState(a, loadedSet)
        const res = worker.config?.residency?.[a.model_key]
        // Per-resource footprint — the honest split. A GGUF slot's host-RAM use is
        // its process RSS; its GPU use is the offloaded layer count (exact per-slot
        // VRAM isn't reported). So the SAME slot shows a real RAM figure under RAM
        // and a layer count under VRAM — never its full disk size as if it all sat
        // in VRAM (why a 15 GB model on an 8 GB GPU no longer looks GPU-resident).
        let facts
        let factsEst = false   // this row's byte figure is derived, not measured
        if (resource === 'vram') {
          // Per-model VRAM = resident weights (vramBytesFor); the KV cache is not
          // reported per model, so the rows won't sum to the GPU's used bar — the
          // reconcile line below closes that gap. Same helper drives that sum, so
          // rows and total can't disagree.
          const det = worker.loaded_detail?.[a.model_key] || {}
          const ngl = a.n_gpu_layers
          const total = a.total_layers ?? det.total_layers
          const vr = vramBytesFor(a, worker, sizeByKey)
          const vb = vr ? vr.bytes : null
          // Estimated figures never render like measurements: `~` + the muted
          // wp-fact-est class + the honest tooltip on the whole row's facts.
          const est = !!(vr && vr.estimated)
          const vtxt = vb != null ? `${est ? '~' : ''}${fmtBytes(vb)} VRAM` : null
          const bits = []
          // RESOLVED-mode label (item M/D, k65): name the seat by what actually
          // happened. A 0<ngl<total split is a max-gpu SPILL — never "gpu only".
          // When the operator's CONFIGURED mode disagrees with the resolved
          // placement (e.g. gpu-only configured but the worker spilled to 42/81),
          // surface the disagreement — never blend the two into one comforting word.
          const resolved = resolvedSeatMode(ngl, total, a.gpu_pct)
          const configured = deriveAllocMode(worker.spill_by_model?.[a.model_key])
          if (resolved.mode) {
            const disagrees = configured && configured !== resolved.mode
            bits.push(disagrees
              ? `⚠ ${allocModeLabel(resolved.mode)} (configured ${allocModeLabel(configured)})`
              : allocModeLabel(resolved.mode))
          }
          if (ngl === -1) {
            bits.push(vtxt || 'all layers on GPU')
          } else if (ngl != null && ngl > 0) {
            // partial offload — exact VRAM needs a per-layer split we don't get,
            // so estimate from the offloaded fraction when total layers are known,
            // else fall back to the honest layer count. Never render a missing
            // total as the literal string "undefined" (old workers omit it).
            const layerTxt = total != null ? `${ngl}/${total} layers`
              : `${ngl} layer${ngl === 1 ? '' : 's'} on GPU`
            bits.push(vtxt ? `${vtxt} · ${layerTxt}` : layerTxt)
          } else if (a.gpu_pct != null && a.gpu_pct > 0) {
            bits.push(vtxt
              ? `${vtxt} · ${Math.round(a.gpu_pct)}% on GPU`
              : `${Math.round(a.gpu_pct)}% on GPU`)
          } else {
            // loaded, but no layers on the GPU — it lives in host RAM
            bits.push('host RAM — not in VRAM')
          }
          if (a.ctx != null) bits.push(`ctx ${a.ctx}`)
          facts = bits.join(' · ')
          factsEst = est && vb != null
        } else {
          // HOST RAM footprint = MEASURED resident memory only. A model's file
          // size is NOT its RAM usage — an idle or mmap'd model isn't holding its
          // weights in RAM, which is how a 29 GB box showed 125 GB "resident".
          // So: real rss when the worker measured it; otherwise mark it idle and
          // show the effective on-disk size (never the dir sum) as context only.
          const reg = sizeByKey?.get?.(a.model_key)
          const disk = (reg && reg.eff != null) ? reg.eff : (a.model_bytes ?? a.weight_bytes)
          const bits = []
          if (a.rss_anon_bytes != null) {
            // HONEST split (new workers): VmRSS counts the mmap'd GGUF's
            // file-backed pages — reclaimable page cache, not pinned RAM — so a
            // 1.5 GB process read as "42.9 GB RSS" (~28x overstated). Lead with
            // the anon (truly-pinned) figure; the mmap'd share is labeled cache.
            const cache = a.rss_file_bytes
            bits.push(cache != null && cache > 0
              ? `${fmtBytes(a.rss_anon_bytes)} RAM + ${fmtBytes(cache)} cache`
              : `${fmtBytes(a.rss_anon_bytes)} RAM`)
          } else if (a.ram_resident_bytes != null) {
            // MEASURED host-RAM occupancy of an in-process (ram-kind) model —
            // the worker's smaps read, the supply side of the 2026-07-28
            // measurements-are-truth ruling. Absent on pre-release workers, so
            // this arm simply doesn't fire there (the honest lines below do).
            bits.push(`${fmtBytes(a.ram_resident_bytes)} RAM resident`)
          } else if (a.rss_bytes != null) {
            bits.push(`${fmtBytes(a.rss_bytes)}${a.kind === 'slot' ? ' RSS' : ''}`)
          } else if (st.state === 'allocated') {
            // NOT resident — an allocation is attributed here with no measurement
            // behind it. This line used to read "resident · RAM size not measured",
            // which asserted residency the worker never established: on 2026-07-28
            // it described a model whose provisioning had died with ENOSPC. Say
            // what is actually known — an allocation exists, nothing measured it —
            // and keep the on-disk size as context only.
            bits.push(disk != null ? `allocated (unmeasured) · ${fmtBytes(disk)} on disk` : 'allocated (unmeasured)')
          } else if (st.state !== 'idle') {
            // Measured-RESIDENT (torch saw materialized weights, e.g. device 'cpu')
            // but this worker build can't attribute a per-model RSS byte count —
            // the glibc-arena + page-cache gap (see CODE_GAPS). Say residency is
            // real and the SIZE is unmeasured, with the on-disk size only as
            // context — never present the file size as RAM use.
            bits.push(disk != null ? `resident · RAM size not measured · ${fmtBytes(disk)} on disk` : 'resident · RAM size not measured')
          } else {
            bits.push(disk != null ? `idle · ${fmtBytes(disk)} on disk` : 'idle')
          }
          if (a.ctx != null) bits.push(`ctx ${a.ctx}`)
          facts = bits.join(' · ')
        }
        return (
          <span key={(a.model_key || '') + i} className={`wp-model wp-st-${st.state}`}>
            <span className={`wp-state-pill wp-pill-${st.state}`} title={st.title}>{st.glyph}</span>
            <span className="wp-model-name">{a.model_key}</span>
            {res === 'static' && (
              <span className="wp-slot-res" title="occupant residency: static — locked in, never swapped out">🔒</span>
            )}
            <span className={`wp-model-facts${factsEst ? ' wp-fact-est' : ''}`}
                  title={factsEst ? ESTIMATED_VRAM_TITLE : undefined}>{facts}</span>
          </span>
        )
      })}
    </div>
  )
}

// Closes the itemized weights back to the GPU's used bar — on EVERY GPU, even
// one with no hugpy model resident (ae's 3090 shows GBs held by ComfyUI while
// the model list is empty; leaving that unexplained is the bug). The per-model
// rows show resident WEIGHTS only; the used figure also includes the KV cache,
// the CUDA context, ComfyUI (when attached — deliberately out-of-pool, not
// itemized), and any other non-model GPU use. So we show weights + the
// remainder, named honestly for what most likely holds it, and never pretend
// the remainder is all KV cache. Hidden only when there's no GPU or used is 0.
export function VramReconcile({ worker, gpuRes, comfy, sizeByKey }) {
  const used = worker.vram_used
  if (used == null || used <= 0) return null
  // A reconciliation of MEASUREMENTS never silently contains guesses (ruling
  // 2026-07-28). Measured per-process weights and estimated ones (declared
  // placement × file size) are summed SEPARATELY and shown as distinct terms —
  // the estimate still appears (on a broken-nvidia-smi box it's the only signal)
  // but it is named, `~`-marked and muted, never folded into the measured sum.
  let sum = 0, estSum = 0, estCount = 0, exact = true
  for (const a of gpuRes) {
    const vr = vramBytesFor(a, worker, sizeByKey)
    if (vr == null) exact = false     // a weight we couldn't measure — mark ~
    else if (vr.estimated) { estSum += vr.bytes; estCount += 1 }
    else sum += vr.bytes
  }
  const overhead = Math.max(used - sum - estSum, 0)

  // VRAM in use but nothing the UI can see as GPU-resident. This is the case
  // nvidia-smi exposed: the worker's in-process (transformers/vision) models sit
  // on the GPU but report n_gpu_layers=null, so they fall out of gpuRes. Don't
  // guess a holder (it is NOT necessarily ComfyUI — measured at 256 MiB idle);
  // say plainly that the per-process read is what will itemize it.
  if (!gpuRes.length) {
    return (
      <div className="wp-vram-reconcile"
           title="VRAM is in use but no model is reported as GPU-resident here — in-process models load onto the GPU without saying so. The worker's per-process (nvidia-smi) read will attribute this per model.">
        <span className="wp-vram-used">{fmtBytes(used)} used</span>
        {' — per-model VRAM pending the worker per-process read'}
      </div>
    )
  }

  // t13/t14: when the worker reports the pid_registry split, close the ledger
  // HONESTLY — attributed (hugpy's own served models + CUDA context) vs
  // unattributed (foreign / non-hugpy squatters), surfaced as a distinct labeled
  // term the way the storage chip surfaces 'unattributed on disk'. This replaces
  // the guessed "KV cache / context / other" remainder with a measured split.
  const attributed = worker.vram_attributed_bytes
  const unattributed = worker.vram_unattributed_bytes
  if (attributed != null) {
    const residual = Math.max(used - attributed - (unattributed || 0), 0)
    return (
      <div className="wp-vram-reconcile"
           title="Ledger from the worker's per-process (nvidia-smi + pid_registry) read: attributed = hugpy's own served models and CUDA context; unattributed = foreign / non-hugpy GPU use (the honest overhead the bar counts as external); residual = KV cache / activations not tied to a process.">
        <span className="wp-vram-attributed">{fmtBytes(attributed)} attributed</span>
        {unattributed > 0 && (
          <span className="wp-vram-overhead"> + {fmtBytes(unattributed)} unattributed / foreign</span>
        )}
        {residual > 0 && (
          <span className="wp-vram-overhead"> + {fmtBytes(residual)} KV cache / activations</span>
        )}
        <span className="wp-vram-used"> = {fmtBytes(used)} used</span>
      </div>
    )
  }

  return (
    <div className="wp-vram-reconcile"
         title={'Measured weights come from the worker\'s per-process read. '
           + (estCount > 0
             ? `${estCount} model${estCount === 1 ? ' is' : 's are'} shown as a separate estimated term (${ESTIMATED_VRAM_TITLE}) — never added into the measured sum. `
             : '')
           + 'The remainder is KV cache, CUDA context, and any other GPU use — a residual, not an independently measured number.'}>
      {`${exact ? '' : '~'}${fmtBytes(sum)} measured weights`}
      {estCount > 0 && (
        <span className="wp-vram-est"> + ~{fmtBytes(estSum)} estimated ({estCount} model{estCount === 1 ? '' : 's'})</span>
      )}
      {overhead > 0 && <span className="wp-vram-overhead"> + {fmtBytes(overhead)} KV cache / context / other</span>}
      <span className="wp-vram-used"> = {fmtBytes(used)} used</span>
    </div>
  )
}

// A compact resource summary chip — same anatomy as the per-GPU wp-gpu chip
// (head line + wp-vram-bar), but clickable to expand its detail below.
// Honest resource bar (t13/t14). `bar` is the central summary's spec projection
// (bar_semantics / bar_used / bar_total / bar_remaining / bar_over_limit +
// encroachment/raw). When present it is AUTHORITATIVE — used, total, and the
// over-limit warning all come from it, so the chip's numerator and denominator
// share one universe (the operator's spec). When absent or bar_semantics ===
// 'legacy' (a pre-slice worker that never reported the honest inputs) the chip
// falls back to the passed used/total but LABELS the bar 'legacy est.' rather
// than pretending the mixed-universe number is exact.
export function ResourceChip({ icon, name, used, total, count, active, onClick, disabled, physicalTotal, bar, extraTerm }) {
  const semantics = bar?.semantics
  const honest = !!bar && semantics !== 'legacy'
  // Honest path: everything from the spec. Legacy/no-bar: the caller's figures.
  const barUsed  = honest && bar.used != null ? bar.used : used
  const barTotal = honest && bar.total != null ? bar.total : total
  const rawUsed  = honest && bar.rawUsed != null ? bar.rawUsed : barUsed
  const overLimit = honest && !!bar.overLimit
  const encroach  = honest ? (bar.encroachment || 0) : 0
  const has = barTotal != null && barTotal > 0 && barUsed != null
  // Fill: clamped 0..100. An over-limit box pins at 100 (the raw truth lives in
  // the warning + title, never a >100% bar).
  const pct = has ? Math.min(Math.max((barUsed / barTotal) * 100, 0), 100) : 0
  const free = honest && bar.remaining != null ? bar.remaining
             : (has ? Math.max(barTotal - barUsed, 0) : null)
  // The denominator IS a central limit (below the physical total) — box delegation.
  const limited = semantics === 'central'
    || (physicalTotal != null && barTotal != null && barTotal < physicalTotal)
  const legacy = !!bar && semantics === 'legacy'
  const title = has
    ? `${name}: ${fmtBytes(barUsed)} used / ${fmtBytes(barTotal)}`
      + (limited ? ` central limit${physicalTotal != null ? ` (${fmtBytes(physicalTotal)} physical)` : ''}` : ' physical')
      + (free != null ? ` · ${fmtBytes(free)} free` : '')
      + (encroach > 0 ? `\n\n⚠ ${fmtBytes(encroach)} of the worker's budget is encroached by external (non-hugpy) usage that exceeded the physical headroom above the limit.` : '')
      + (overLimit ? `\n\n⛔ OVER LIMIT by ${fmtBytes(bar.overBy || Math.max(rawUsed - barTotal, 0))} — raw usage ${fmtBytes(rawUsed)} exceeds the ${fmtBytes(barTotal)} budget. The bar is pinned full; admission is paused until it drains.` : '')
      + (legacy ? '\n\n(legacy estimate — this worker predates the honest budget-bar inputs; the figure mixes physical and limit universes.)' : '')
      + ' — click for resident detail'
    : `${name}: no data`
  return (
    <button type="button" disabled={disabled}
            className={`wp-gpu wp-res-chip${active ? ' wp-res-active' : ''}${disabled ? ' wp-res-disabled' : ''}${overLimit ? ' wp-res-overlimit' : ''}`}
            onClick={onClick}
            title={title}>
      <div className="wp-gpu-head">
        <span className="wp-res-name">{icon} {name}</span>
        {has ? <em> · {fmtBytes(barUsed)} / {fmtBytes(barTotal)}</em> : <em className="wp-res-nodata"> · —</em>}
        {limited && !legacy && (
          <span className="wp-res-limit"
                title={`Central limit ${fmtBytes(barTotal)}${physicalTotal != null ? ` of ${fmtBytes(physicalTotal)} physical` : ''} — the budget the fleet respects.`}>
            limit
          </span>
        )}
        {legacy && (
          <span className="wp-res-legacy"
                title="This worker predates the honest budget-bar inputs (t13/t14). The bar is a legacy estimate mixing physical and central-limit figures — update the worker to get the true bar.">
            legacy est.
          </span>
        )}
        {overLimit && (
          <span className="wp-res-warn"
                title={`Over the ${fmtBytes(barTotal)} budget by ${fmtBytes(bar.overBy || Math.max(rawUsed - barTotal, 0))} — raw usage ${fmtBytes(rawUsed)}. Admission is paused until it drains.`}>
            ⛔ over
          </span>
        )}
        {!overLimit && encroach > 0 && (
          <span className="wp-res-warn wp-res-encroach"
                title={`${fmtBytes(encroach)} of this worker's budget is taken by external (non-hugpy) usage that spilled past the physical headroom above the limit.`}>
            ⚠ encroached {fmtBytes(encroach)}
          </span>
        )}
        {count != null && count > 0 && <span className="wp-res-count">{count}</span>}
        {!disabled && <span className="wp-res-caret">{active ? '▾' : '▸'}</span>}
      </div>
      <div className={`wp-vram-bar${overLimit ? ' wp-vram-bar-over' : ''}`}>
        {has && <div className={`wp-vram-fill${overLimit ? ' wp-vram-fill-over' : ''}`} style={{ width: `${pct}%` }} />}
        {/* The encroachment slice, drawn as a distinct segment at the tail of
            the fill so the operator sees how much of the budget is being eaten
            by external usage vs the worker's own models. */}
        {has && encroach > 0 && !overLimit && (
          <div className="wp-vram-encroach"
               style={{ width: `${Math.min(Math.max((encroach / barTotal) * 100, 0), 100)}%` }} />
        )}
      </div>
      {extraTerm && <div className="wp-res-extraterm">{extraTerm}</div>}
    </button>
  )
}

// The worker card's resource header: a row of three compact chips (VRAM · RAM ·
// Storage) sharing the wp-gpu look, each expanding a detail panel below on click.
// VRAM → per-GPU breakdown + GPU-seated models; RAM → in-process resident models;
// Storage → the cached-file list + eviction proposal. Splits the engine-agnostic
// residents by where they live (slot = GPU, ram = host process).
export function ResourceStrip({ worker, models, onApproveEvictions, onEvict }) {
  const [open, setOpen] = useState(null)  // 'vram' | 'ram' | 'storage' | null
  const gpus = Array.isArray(worker.gpus) ? worker.gpus : []
  const s = worker.storage || {}

  // Model-level effective-quant sizes (from the /models feed) — a GGUF's real
  // size is the ONE quant that serves, not its on-disk dir sum. Keyed by model_key
  // so the resident list and the storage detail can show the true model size.
  const ggufSize = new Map()
  for (const mm of (models || [])) {
    const k = mm && (mm.model_key || mm.key)
    const fw = String((mm && mm.framework) || '').toLowerCase()
    if (k && (fw === 'gguf' || fw === 'llama_cpp') && mm.effective_bytes != null) {
      ggufSize.set(k, { eff: mm.effective_bytes, effGguf: mm.effective_gguf })
    }
  }

  const GIB = 2 ** 30
  const limits = worker.limits || {}
  const allocs = Array.isArray(worker.allocations) ? worker.allocations.filter(a => a && a.model_key) : []
  const loadedSet = new Set(worker.loaded_models || [])
  // Resource-aware split — a GGUF slot lives in host RAM (its process rss) AND, when
  // layers are offloaded, on the GPU, so it appears under BOTH (each with the
  // footprint that fits). This is why a 15 GB model on an 8 GB GPU now shows in RAM
  // instead of pretending to sit entirely in VRAM, and why the RAM detail is no
  // longer empty while host RAM is clearly in use.
  // Device truth first: the worker's per-process (nvidia-smi + torch) read stamps
  // device/vram_bytes, so a cuda-resident transformers/vision model is GPU-resident
  // even though its n_gpu_layers is null (that's a llama.cpp field — the old
  // "host RAM — not in VRAM" mislabel). Fall back to the layer/gpu_pct heuristic
  // for pre-read workers. A slot always keeps its host-RSS row in RAM.
  const onGpu = a => a.device === 'cuda' || (a.vram_bytes != null && a.vram_bytes > 0)
                   || (a.kind === 'slot' && a.n_gpu_layers != null && a.n_gpu_layers !== 0)
                   || (a.gpu_pct != null && a.gpu_pct > 0)
  const inRam = a => a.kind === 'slot' ? true
                   : (a.device === 'cuda' ? false
                      : (a.gpu_pct == null || a.gpu_pct < 99))
  const gpuRes = allocs.filter(onGpu)
  const ramRes = allocs.filter(inRam)
  const comfy = !!worker.comfy?.available

  const vramName = gpus.length === 1 ? (gpus[0].name || 'VRAM')
    : (gpus.length > 1 ? `VRAM · ${gpus.length} GPUs` : 'VRAM')
  const hasVram = (worker.vram_total != null && worker.vram_total > 0) || gpus.length > 0
  // Budget against central limits when set (box delegation wins) — the fleet
  // respects the limit, so the chip measures usage against it, physical as context.
  const ramBudget = limits.ram_max_gib != null ? limits.ram_max_gib * GIB : worker.ram_total
  const vramBudget = limits.gpu_mem_gib != null ? limits.gpu_mem_gib * GIB : worker.vram_total
  // The 💾 Storage chip shows the worker's PHYSICAL DRIVE — the hot drive when
  // present, else the root drive. The worker picks that volume server-side and
  // reports it as disk_total / disk_free (used = total − free). The cache-budget
  // / eviction economy (cache_used vs budget) is the expanded detail below, not
  // this top-line drive gauge.
  const stUsed = s.reported ? Math.max((s.disk_total ?? 0) - (s.disk_free ?? 0), 0) : null
  const stTotal = s.reported ? (s.disk_total ?? null) : null

  // Honest budget-bar (t13/t14): central computed the spec fields onto the
  // worker, PREFIXED per resource (ram_bar_* / vram_bar_*) so RAM and VRAM never
  // collide. Map the flat wire fields to the chip's `bar` prop. Absent (older
  // central / pre-slice worker) -> undefined, and the chip falls back to legacy.
  const mkBar = (p) => {
    const sem = worker[`${p}bar_semantics`]
    if (sem == null) return undefined
    return {
      semantics: sem,
      used: worker[`${p}bar_used`],
      total: worker[`${p}bar_total`],
      remaining: worker[`${p}bar_remaining`],
      rawUsed: worker[`${p}bar_raw_used`],
      encroachment: worker[`${p}bar_encroachment`] || 0,
      overLimit: !!worker[`${p}bar_over_limit`],
      overBy: worker[`${p}bar_over_by`] || 0,
    }
  }
  const ramBar = mkBar('ram_')
  const vramBar = mkBar('vram_')
  // VRAM overhead term: unattributed / non-model GPU bytes the console surfaces
  // as a labeled term on the chip (mirror the storage 'unattributed on disk').
  const vramOverhead = worker.vram_unattributed_bytes
  const vramExtra = (vramBar && vramBar.semantics !== 'legacy' && vramOverhead != null && vramOverhead > 0)
    ? `+ ${fmtBytes(vramOverhead)} unattributed / overhead`
    : null

  const toggle = k => setOpen(o => (o === k ? null : k))

  return (
    <div className="wp-resources-wrap">
      <div className="wp-resources">
        <ResourceChip icon="🖥" name={vramName} used={worker.vram_used} total={vramBudget}
                      physicalTotal={worker.vram_total} count={gpuRes.length} active={open === 'vram'}
                      bar={vramBar} extraTerm={vramExtra}
                      disabled={!hasVram} onClick={() => toggle('vram')} />
        <ResourceChip icon="🧠" name="RAM" used={worker.ram_used} total={ramBudget}
                      physicalTotal={worker.ram_total} count={ramRes.length} active={open === 'ram'}
                      bar={ramBar}
                      onClick={() => toggle('ram')} />
        <ResourceChip icon="💾" name="Storage" used={stUsed} total={stTotal}
                      count={s.reported ? (s.models?.length ?? 0) : null} active={open === 'storage'}
                      disabled={!s.reported} onClick={() => toggle('storage')} />
      </div>

      {open === 'vram' && (
        <div className="wp-res-detail">
          {/* Per-GPU breakdown ONLY when there is something to break down: on a
              single-GPU worker the collapsed chip already IS that GPU's summary
              (same name, same used/total, same bar), so repeating it here read
              as a phantom second GPU (operator ask, 2026-07-11). */}
          {gpus.length > 1 && <GpuChips gpus={gpus} />}
          {/* Only models actually resident in this GPU's VRAM, each with its VRAM
              allocation. Host-RAM-only residents belong under the RAM chip, not
              here — listing a "not in VRAM" model under a VRAM heading reads as a
              contradiction. So this is the honest "models within the VRAM" view,
              and its count matches the chip's in-VRAM badge. */}
          <div className="wp-res-detail-label">Models in VRAM — allocation each ({gpuRes.length})</div>
          <ResidentList items={gpuRes} loadedSet={loadedSet} worker={worker} resource="vram"
                        sizeByKey={ggufSize}
                        emptyLabel="No models are using this GPU's VRAM right now." />
          <VramReconcile worker={worker} gpuRes={gpuRes} comfy={comfy} sizeByKey={ggufSize} />
          <PidRegistry worker={worker} onEvict={onEvict} />
        </div>
      )}

      {open === 'ram' && (
        <div className="wp-res-detail">
          <div className="wp-res-detail-label">Models in host RAM ({ramRes.length})</div>
          <ResidentList items={ramRes} loadedSet={loadedSet} worker={worker} resource="ram"
                        sizeByKey={ggufSize}
                        emptyLabel="No models resident in host RAM right now." />
          {comfy && (
            <div className="wp-budget-comfy"
                 title="ComfyUI is an adopted, externally-owned process. Its checkpoints load on-demand inside ComfyUI and are NOT worker pool residents; its memory folds into the used/total above as an out-of-pool reservation.">
              🧩 ComfyUI attached — external (out-of-pool)
            </div>
          )}
        </div>
      )}

      {open === 'storage' && (
        <div className="wp-res-detail">
          {s.reported
            ? <WorkerStorageBar worker={worker} onApproveEvictions={onApproveEvictions} sizeByKey={ggufSize} detailsOnly />
            : <div className="wp-res-empty">This worker hasn’t reported a storage survey yet.</div>}
        </div>
      )}
    </div>
  )
}
