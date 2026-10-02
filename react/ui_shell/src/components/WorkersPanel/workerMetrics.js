// ── Resource-pool budget bar (read-only; slot-budget redesign, slice 1) ──────
// A worker's UNIFIED pool is its VRAM + host RAM. A "slot" is EMERGENT: the
// occupancy is simply how many models are packed into the pool right now — never
// a fixed seat count — so this reads "N models resident" and "0 models resident
// — pool idle" (not empty seats) when nothing is loaded. This is the row
// headline; the per-GPU chips and the free-RAM line below it are the breakdown.
// STRICTLY DISPLAY: no controls, no onClick, no mutation (the reserve/static
// controls the redesign mentions belong to a later slice).

// Live state of one pool resident — engine-agnostic, so a GGUF slot seat and an
// in-process (RAM) model are described the same way. Returns a state suffix
// (answering|serving|warming|idle) that drives both the pill class and the row
// class, so the bar and the rows never disagree.
//   slot: healthy+busy → answering, healthy → serving, else warming.
//   ram/legacy: the worker's own `serving` flag (answered within its serving
//     window) → serving, else idle. `serving` is worker-computed to dodge
//     client clock skew; only pre-roll agents that omit it fall back to
//     loaded_models membership — else every resident of a churned test pool
//     (ae's leftovers) would falsely read "serving".
// ── Measured residency: the honest "is this model ACTUALLY loaded" test ───────
// "Loaded" must derive from a MEASURED footprint the worker reports for a model,
// NEVER from a bare runner-cache membership (that intent-not-residency signal is
// what flapped rows purple on every reconcile pass). Measured signals:
//   vram_bytes > 0   — nvidia-smi per-process VRAM (a slot child) or torch
//                      per-model CUDA bytes (an in-process transformers/vision model)
//   rss_bytes  > 0    — a slot child's measured process RSS (host RAM)
//   device === 'cpu'  — torch walked MATERIALIZED cpu params: the weights are
//                       genuinely resident in host RAM (its per-model RSS bytes
//                       aren't attributable yet — the glibc-arena + page-cache gap,
//                       see CODE_GAPS — but its RESIDENCY is measured, not guessed)
//                       ...ONLY when device_source says 'measured'. Since
//                       2026-07-25 the worker also reports an INFERRED device
//                       (from the placement it declared at launch) on boxes
//                       where nvidia-smi/torch can't see the model — that is a
//                       placement claim, NOT evidence of residency, so it must
//                       never satisfy this test. device_source is omitted by
//                       pre-2026-07-25 workers, whose device was always measured.
// measuredFootprint = the model is occupying VRAM/RAM right now.
export function measuredFootprint(a) {
  return !!a && ((a.vram_bytes != null && a.vram_bytes > 0)
              || (a.rss_bytes != null && a.rss_bytes > 0)
              || (a.device === 'cpu' && a.device_source !== 'inferred'))
}

// isMeasuredResident = the model's residency was MEASURED. Nothing else counts.
//
// This used to be `measuredFootprint(a) || a.serving === true`, and that OR was a
// conflation that cost an operator an ssh session (2026-07-28): computron's disk
// was 100% full, provisioning died with ENOSPC before a single weight was read,
// but a hollow runner OBJECT had already been constructed and its `last_used`
// stamped — so the worker emitted an allocation row with `serving: true` and the
// compute tab rendered "🔥 serving Qwen2.5-7B-Instruct-GGUF · resident" for a
// model that had NEVER loaded. `serving` is worker-side "touched within the last
// 180s": it is RECENCY, an observation about REQUESTS, and a request that fails
// touches the clock exactly as hard as one that succeeds. Recency is not
// measurement, so it can never be evidence of residency — doctrine: residency
// must be MEASURED. A row with no measurement now renders "allocated
// (unmeasured)" (see residentState) so a phantom can never wear the flame.
export function isMeasuredResident(a) {
  if (!a) return false
  // A worker build that can tell says so outright: `materialized: false` means a
  // runner object exists but no weights were ever loaded through it. That is a
  // definitive NO and outranks everything else. null/absent = an older worker
  // that cannot tell, which must behave exactly as it always did — no regression.
  if (a.materialized === false) return false
  return measuredFootprint(a)
}

export function residentState(a) {
  if (a.kind === 'slot') {
    if (a.healthy && a.busy) return { state: 'answering', glyph: '⚡ answering', title: 'actively processing a request right now' }
    if (a.healthy)           return { state: 'serving',   glyph: '🔥 serving',   title: 'hosted in a slot on this worker — routable' }
    return { state: 'warming', glyph: '⏳ warming', title: 'seated but not yet healthy — warming' }
  }
  if (isMeasuredResident(a)) {
    return { state: 'serving', glyph: '🔥 serving', title: 'measured residency — the worker sees this model occupying VRAM/RAM right now' }
  }
  // ALLOCATED, not resident. Something on the worker is attributing an allocation
  // to this model — a hollow runner object (materialized:false), or a recency
  // flag with no measurement behind it — but nothing measured a footprint. The
  // load may have died in provisioning and never happened at all. Distinct from
  // idle because there IS an attribution to explain, and pointedly NOT a flame.
  if (a.materialized === false || a.serving === true) {
    return { state: 'allocated', glyph: '◌ allocated', title: 'an allocation is attributed to this model but nothing measured its residency — it may never have loaded' }
  }
  return { state: 'idle', glyph: '○ idle', title: 'a runner exists but holds no measured VRAM/RAM — not actually resident' }
}

export function vramBytesFor(a, worker, sizeByKey) {
  // Real measured VRAM (worker's nvidia-smi per-process read, or comfy) ALWAYS
  // wins — that's the ground truth we want to reflect, no estimating.
  if (a?.vram_bytes != null) return { bytes: a.vram_bytes, estimated: false }
  const det = worker?.loaded_detail?.[a?.model_key] || {}
  const reg = sizeByKey?.get?.(a?.model_key)
  // Until the per-process read lands: the model's KNOWN weight — the worker's
  // loaded facts if present, else the registry's effective size (always there).
  // The registry fallback is the fix for slots whose loaded_detail the worker
  // drops, which made the figure read 0 and dumped everything into "overhead".
  const weights = a?.weight_bytes ?? a?.model_bytes ?? det.weight_bytes ?? det.model_bytes
    ?? (reg && reg.eff != null ? reg.eff : null)
  if (weights == null) return null
  const ngl = a?.n_gpu_layers
  const total = a?.total_layers ?? det.total_layers
  const est = b => ({ bytes: b, estimated: true })
  if (ngl === -1) return est(weights)
  if (ngl != null && ngl > 0) return total ? est(weights * ngl / total) : null
  if (a?.gpu_pct != null && a.gpu_pct > 0) return est(weights * a.gpu_pct / 100)
  return null
}
