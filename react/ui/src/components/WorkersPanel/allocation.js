import { ALLOC_MODE_OPTIONS, WP_GIB } from './constants.js'

// ── GPU allocation: concise spill <-> mode mapping ──────────────────────────
// A worker's per-(model) override is an opaque "spill" dict the backend applies
// when it loads the model. We expose it as one mode + an optional VRAM budget.
export function spillToMode(spill) {
  const empty = { mode: 'auto', gib: '', ram: '', threads: '',
                  gpuBand: '', ramBand: '', ctxPct: '', ctxBand: '', priority: '' }
  if (!spill || !Object.keys(spill).length) return empty
  const ngl = spill.n_gpu_layers
  if (ngl === -1 || ngl === '-1') return { ...empty, mode: 'gpu' }
  if (ngl === 'off' || ngl === 0 || ngl === '0') return { ...empty, mode: 'cpu' }
  return {
    mode: 'custom',
    gib: spill.gpu_mem_gib != null ? String(spill.gpu_mem_gib) : '',
    ram: spill.cpu_mem_gib != null ? String(spill.cpu_mem_gib) : '',
    threads: spill.threads != null ? String(spill.threads) : '',
    // t21 tolerance-band fields (custom/explicit-budget only, GGUF-gated).
    gpuBand: spill.gpu_mem_gib_deviation_pct != null ? String(spill.gpu_mem_gib_deviation_pct) : '',
    ramBand: spill.cpu_mem_gib_deviation_pct != null ? String(spill.cpu_mem_gib_deviation_pct) : '',
    ctxPct: spill.ctx_pct != null ? String(spill.ctx_pct) : '',
    ctxBand: spill.ctx_deviation_pct != null ? String(spill.ctx_deviation_pct) : '',
    priority: spill.priority != null ? String(spill.priority) : '',
  }
}

export function modeToSpill(mode, gib, ram, threads, gpuBand, ramBand, ctxPct, ctxBand, priority) {
  if (mode === 'gpu') return { n_gpu_layers: -1 }
  if (mode === 'cpu') return { n_gpu_layers: 'off' }
  if (mode === 'custom') {
    // Explicit per-model budgets: VRAM / RAM / cores — the model's resource
    // contract on this worker (enforced at load: autofit plans within the
    // VRAM budget, the CPU-resident share must fit the RAM budget, threads
    // cap generation cores). t21 tolerance bands let this allocation flex ±
    // a percent of the worker's capacity under contention before eviction;
    // ctx_pct/ctx_deviation_pct do the same for the KV context reservation
    // (the cheapest thing to flex); priority lets a higher-priority
    // allocation compress lower-priority neighbors within their bands first.
    const s = {}
    if (gib !== '' && gib != null) s.gpu_mem_gib = Number(gib)
    if (ram !== '' && ram != null) s.cpu_mem_gib = Number(ram)
    if (threads !== '' && threads != null) s.threads = Number(threads)
    if (gpuBand !== '' && gpuBand != null) s.gpu_mem_gib_deviation_pct = Number(gpuBand)
    if (ramBand !== '' && ramBand != null) s.cpu_mem_gib_deviation_pct = Number(ramBand)
    if (ctxPct !== '' && ctxPct != null) s.ctx_pct = Math.round(Number(ctxPct))
    if (ctxBand !== '' && ctxBand != null) s.ctx_deviation_pct = Number(ctxBand)
    if (priority !== '' && priority != null) s.priority = Math.round(Number(priority))
    return s
  }
  return {}   // 'auto' → empty = autofit (clears any override)
}

// Is this spill a GGUF-ONLY allocation? NARROWED by t26 (operator: "the non
// ggufs should still have options, just not explicit — autofit, maxgpu and cpu
// only should still be on the table"). The EXPLICIT-BUDGET class (gpu_mem_gib /
// cpu_mem_gib / threads / tensor_split + bands + explicit-only companions) is
// GGUF-exclusive, AND ``alloc_mode: 'explicit'`` by value. Autofit ({}), the
// placement-intent modes Max GPU / CPU only (which carry ONLY n_gpu_layers), and
// ``alloc_mode: 'max-ram'`` (opened for non-GGUF 2026-07-24) are engine-agnostic
// and apply to every engine. Mirrors the backend's _alloc_is_gguf_only so the UI
// split preview and the server gate agree.
export const EXPLICIT_BUDGET_KEYS = ['gpu_mem_gib', 'cpu_mem_gib', 'threads', 'tensor_split',
  'gpu_mem_gib_deviation_pct', 'cpu_mem_gib_deviation_pct', 'ctx_pct', 'ctx_deviation_pct',
  'priority', 'leniency_pct', 'priority_device']

export function allocIsGgufOnly(spill) {
  if (!spill || Object.keys(spill).length === 0) return false   // autofit — universal
  if (EXPLICIT_BUDGET_KEYS.some(k => k in spill)) return true
  // alloc_mode is value-sensitive: explicit is GGUF-only, max-ram is not.
  return String(spill.alloc_mode || '').trim().toLowerCase() === 'explicit'
}

export function spillLabel(spill) {
  const { mode, gib, ram, threads, gpuBand, ramBand, ctxPct, ctxBand, priority } = spillToMode(spill)
  // k37 nomenclature (labels only): gpu→gpu-only, cpu→ram-only, auto→max-gpu.
  if (mode === 'gpu') return 'GPU only'
  if (mode === 'cpu') return 'RAM only'
  if (mode === 'custom') {
    const parts = []
    if (gib) parts.push(`${gib}G VRAM`)
    if (ram) parts.push(`${ram}G RAM`)
    if (threads) parts.push(`${threads} cores`)
    if (gpuBand) parts.push(`±${gpuBand}%V`)
    if (ramBand) parts.push(`±${ramBand}%R`)
    if (ctxPct || ctxBand) parts.push(`ctx${ctxPct || '?'}±${ctxBand || 0}%`)
    if (priority) parts.push(`p${priority}`)
    return parts.join(' · ') || 'explicit'
  }
  return 'max-gpu'
}

export function workerCapacity(worker, which) {
  const limits = (worker && worker.limits) || {}
  if (which === 'vram') {
    if (limits.gpu_mem_gib != null) return { bytes: limits.gpu_mem_gib * WP_GIB, gib: limits.gpu_mem_gib, basis: 'central limit' }
    if (worker && worker.vram_total != null) return { bytes: worker.vram_total, gib: worker.vram_total / WP_GIB, basis: 'card' }
    return null
  }
  if (limits.ram_max_gib != null) return { bytes: limits.ram_max_gib * WP_GIB, gib: limits.ram_max_gib, basis: 'central limit' }
  if (worker && worker.ram_total != null) return { bytes: worker.ram_total, gib: worker.ram_total / WP_GIB, basis: 'physical' }
  return null
}

// The model's EFFECTIVE flat mode from its persisted override/spill — the JS
// mirror of managers.alloc_modes.derive_alloc_mode (the WorkersPanel serving
// rows carry only worker.spill_by_model[key], not a server-derived alloc_mode
// field, so the derivation happens here). A blank model == max-gpu (the
// default: fits-and-spills, never OOMs — defaults-are-promises).
export function deriveAllocMode(spill) {
  const ov = spill || {}
  const am = ov.alloc_mode != null ? String(ov.alloc_mode).trim().toLowerCase() : ''
  // Persisted alloc_mode wins (canonical only reaches us for max-ram/explicit;
  // the coarse trio was rewritten onto the wire, but honor it if present).
  if (am === 'max-ram' || am === 'explicit' || am === 'gpu-only'
      || am === 'ram-only' || am === 'max-gpu') return am
  if (am === 'autofit') return 'max-gpu'
  if (am === 'cpu-only' || am === 'cpu_only' || am === 'ram_only') return 'ram-only'
  if (am === 'gpu_only') return 'gpu-only'
  if (am === 'max_ram') return 'max-ram'
  if (am === 'budget' || am === 'bands') return 'explicit'
  const ngl = ov.n_gpu_layers
  if (ngl != null) {
    const s = String(ngl).trim().toLowerCase()
    if (s === '-1') return 'gpu-only'
    if (s === '0' || s === 'off' || s === 'cpu' || s === 'none') return 'ram-only'
    // positive layer count / "auto": a fit-and-spill flavor → max-gpu
  }
  for (const k of ['leniency_pct', 'gpu_mem_gib', 'cpu_mem_gib',
                   'gpu_mem_gib_deviation_pct', 'cpu_mem_gib_deviation_pct']) {
    if (ov[k] != null) return 'explicit'
  }
  return 'max-gpu'
}

// Short label for a mode name (the alloc cell reads this).
export function allocModeLabel(mode) {
  const opt = ALLOC_MODE_OPTIONS.find(o => o[0] === mode)
  return opt ? opt[1] : mode
}

// RESOLVED mode of a SEAT from its ACTUAL placement — what the worker really did,
// not what the override intended (item M/D, k65). A 42/81 layer split is a
// GPU-maximal SPILL (max-gpu), never strict gpu-only (all-or-bust): labeling it
// "gpu only" is the lie the operator caught. Derived purely from the measured
// allocation row:
//   ngl === -1, or ngl === total   → all layers on GPU  (gpu-only shape)
//   0 < ngl < total                → 'max-gpu' SPLIT     (hybrid — NOT gpu-only)
//   ngl === 0 (or gpu_pct === 0)   → ram-only
//   unknown                        → null (say nothing rather than guess)
// Returns { mode, split } where split is "N/T" for a hybrid, else null.
export function resolvedSeatMode(ngl, total, gpuPct) {
  const n = ngl == null ? null : Number(ngl)
  const t = total == null ? null : Number(total)
  if (n != null && Number.isFinite(n)) {
    if (n === 0) return { mode: 'ram-only', split: null }
    if (n === -1) return { mode: 'gpu-only', split: null }
    if (t != null && Number.isFinite(t) && t > 0) {
      if (n >= t) return { mode: 'gpu-only', split: null }
      if (n > 0) return { mode: 'max-gpu', split: `${n}/${t}` }
    }
    if (n > 0) return { mode: 'max-gpu', split: null }  // partial, total unknown
  }
  if (gpuPct != null && Number.isFinite(Number(gpuPct))) {
    const p = Number(gpuPct)
    if (p <= 0) return { mode: 'ram-only', split: null }
    if (p >= 100) return { mode: 'gpu-only', split: null }
    return { mode: 'max-gpu', split: null }
  }
  return { mode: null, split: null }
}

// A short GiB rendering for the numbered disable tooltips — mirrors the backend
// 409's _fmt_gib ("68.0GiB"), binary units, so the UI reason matches the wire's.
export function fmtGiB(bytes) {
  if (bytes == null) return '?'
  return `${(Number(bytes) / (2 ** 30)).toFixed(1)}GiB`
}

// The honest, numbers-naming reason a given mode is INFEASIBLE for this
// (model, worker) — the UI mirror of the backend's per-mode feasibility_modes
// rules + the 409 wording ("model 68.0GiB exceeds GPU 24.0GiB"). ctx carries
// {modelBytes, vramTotal, ramTotal} from the row/worker. Returns a string when a
// number-backed reason exists, else a generic capability line (engine gate for
// explicit), else null (no specific reason to show). NEVER used to
// DECIDE disabling — the backend's feasible[] set is authoritative for that;
// this only EXPLAINS a disable the feasible set already made.
export function allocDisableReason(value, ctx, engineGguf) {
  const mb = ctx && ctx.modelBytes
  const gpu = ctx && ctx.vramTotal
  const ram = ctx && ctx.ramTotal
  const HEADROOM = 0.95   // mirrors alloc_modes._GPU_FIT_HEADROOM (approx.)
  if (value === 'explicit' && engineGguf === false) {
    return 'explicit is GGUF-only — banded leniency has no transformers analogue'
  }
  if (mb == null) return null   // no size → no numbered reason (fail-open anyway)
  if ((value === 'gpu-only' || (value === 'max-gpu' && engineGguf === false))
      && gpu != null && mb > HEADROOM * gpu) {
    return `model ${fmtGiB(mb)} exceeds GPU ${fmtGiB(gpu)}`
  }
  if (value === 'ram-only' && ram != null && mb > ram) {
    return `model ${fmtGiB(mb)} exceeds RAM ${fmtGiB(ram)}`
  }
  if ((value === 'max-ram' || value === 'explicit')
      && gpu != null && ram != null && mb > gpu + ram) {
    return `model ${fmtGiB(mb)} exceeds GPU+RAM ${fmtGiB(gpu + ram)}`
  }
  return null
}
