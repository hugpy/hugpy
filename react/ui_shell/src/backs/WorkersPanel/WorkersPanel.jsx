import { useEffect, useState, useCallback, useMemo, useRef, Fragment } from 'react'
import { fetchJson } from '../../api'
import { resolveApiOrigin } from '../../runtime/config'
import ModelPicker from '../ModelPicker/ModelPicker'
import { modelTask, modelTasks } from '../ModelTable/ModelTable'
import FixDoc from '../FixDoc/FixDoc'
import useSessionState from '../../hooks/useSessionState'
import useColumnLayout from '../../hooks/useColumnLayout'
import './WorkersPanel.css'

// Serving-table layout persistence: the localStorage key + the DEFAULT column
// order (left-to-right, after the fixed select-checkbox). Operator ask
// 2026-07-28 REPLACES the 07-18 order: Model leads (it is now the FROZEN
// identity column — sticky-left during horizontal scroll, and excluded from
// drag-reorder so it always stays first), then the sizing/cost columns
// (Memory, Alloc, Size, Ctx), then the routing identifiers (Task, Engine,
// 4-bit, MoE), then the live/placement columns (State, Seat, Residency, 📌),
// with Actions last — ⛔ block is a button INSIDE the Actions cell, not a
// column of its own, so the requested trailing "block" lands there.
// The order/widths are user-adjustable (drag to reorder, drag a header edge to
// resize) and persisted per browser; useColumnLayout merges this list against a
// stored layout so a future column add/remove needs no version bump (see hook).
// The key is bumped to v2 BECAUSE the default ORDER changed: the hook's merge
// deliberately honours a stored order, so operators carrying a v1 layout would
// otherwise never see the new default.
const SERV_LAYOUT_KEY = 'hugpy.workers.servtable.layout.v2'
const SERV_DEFAULT_ORDER = [
  'name', 'memory', 'alloc', 'size', 'ctx', 'task', 'framework',
  'fourbit', 'moe', 'state', 'seat', 'residency', 'pin', 'actions',
]
// The one column that is PINNED left (frozen during horizontal scroll) and
// therefore not a drag-reorder source or drop target. Its sort click still works.
// It is ALSO the fluid column (operator ask 2026-07-28): no manual resize, width
// is a clamp() in CSS, and any persisted px width for it is ignored (see the
// <colgroup> below) — deliberately WITHOUT bumping SERV_LAYOUT_KEY, so every
// other column's stored width survives.
const SERV_FROZEN_COL = 'name'

// ── Compact (narrow-viewport) mode ───────────────────────────────────────────
// Below SERV_COMPACT_PX of TABLE-WRAPPER width the serving table drops to three
// columns and moves everything else into a per-row drawer. The trigger is the
// WRAPPER's width, not the viewport's (a ResizeObserver, not matchMedia): the
// panel can be narrow inside a wide window (a split tab pane, a narrow card),
// and a viewport media query would leave that case unusably wide-scrolling.
const SERV_COMPACT_PX = 720
// The only columns that stay in the table in compact mode, left-to-right. Every
// other column def is rendered as a labeled chip in the row drawer instead.
const SERV_COMPACT_COLS = ['name', 'state', 'memory']

// useNarrowContainer — true while the observed element is narrower than `px`.
// Degrades to false (= desktop layout) wherever ResizeObserver is missing, so
// an old browser gets the full table rather than a broken one.
function useNarrowContainer(ref, px) {
  const [narrow, setNarrow] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        const w = e.contentRect ? e.contentRect.width : el.clientWidth
        setNarrow(w > 0 && w < px)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref, px])
  return narrow
}

// midTrunc — MIDDLE ellipsis for a model name. Model keys differentiate at BOTH
// ends ("DavidAU~MN-GRAND-23.5B-…-NEO-Imatrix-GGUF" vs its non-NEO sibling), so
// a plain CSS tail-ellipsis hides exactly the distinguishing part. The budget is
// a CHARACTER count derived from the column (not a measured pixel width): cheap,
// deterministic, and stable across re-renders. No-ops when the name already fits
// (head + tail + 2 chars — never replaces 1–2 characters with a 1-char ellipsis).
// The FULL name always rides the cell's title, and the compact drawer shows it
// untruncated.
//
// BUDGET: the column caps at 220px; at the table's 12px font (~6.2px/char) less
// 16px of cell padding that is ~32 characters, so 12 + 1 + 14 = 27 sits inside
// it with margin and the CSS ellipsis backstop never fires. The TAIL is the
// longer half on purpose — the operator's own case is
// "…-NEO-Imatrix-GGUF" vs "…-Imatrix-GGUF", where the discriminator is ~16
// characters from the end; a 10-char tail would show "atrix-GGUF" for both.
function midTrunc(name, head = 12, tail = 14) {
  const s = String(name ?? '')
  if (s.length <= head + tail + 2) return s
  return `${s.slice(0, head)}…${s.slice(-tail)}`
}

// Honest binary units (ruling 5, 2026-07-24): this formatter divides by 1024
// (binary), so the LABEL must be the binary unit (KiB/MiB/GiB), not the decimal
// SI one (KB/MB/GB). Before, a 45.09 GiB model (effective_bytes 48.41e9 —
// exactly the sum of coder-next's shards) rendered "45.1 GB", conflating the
// two universes: the number was GiB, the suffix said GB. The whole stack speaks
// GiB (effective quant, budgets, box caps all say "GiB"), so labeling the /1024
// math as GiB makes every call site (~80 of them) honest at once. The reported
// "41.6 GB" was NOT this formatter's output for the size cell — it is the model's
// MoE non-expert GPU-split figure (feasibility prices the card against that, not
// the full file); the size cell always showed the full effective_bytes, which is
// correct data, only mislabeled by one unit-suffix character each.
function fmtBytes(n) {
  if (n == null) return '?'
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// last-served relative time from the central per-(worker,model) `last_picked`
// epoch (seconds). null / 0 = never routed through central — the coldest, and
// exactly the never-served leftovers the eviction proposal frees first.
function fmtServed(epoch) {
  if (!epoch) return 'never served'
  const secs = Math.max(0, Date.now() / 1000 - Number(epoch))
  if (secs < 45) return 'just now'
  const m = secs / 60
  if (m < 60) return `${Math.round(m)}m ago`
  const h = m / 60
  if (h < 24) return `${Math.round(h)}h ago`
  const d = h / 24
  if (d < 30) return `${Math.round(d)}d ago`
  return `${Math.round(d / 30)}mo ago`
}

// Protection/eviction badge for a per-model storage row. Reuses the Serving-row
// state-pill palette. Protected models CANNOT be deleted; evictable ones can,
// and any in the current proposal are flagged for freeing.
function storageBadge(m, proposed) {
  // REFUSED (storage): the pull never started — even a full FIFO of the cold,
  // unprotected models couldn't free enough room under this worker's budget.
  // Doctrine "defaults are promises": a model that cannot fit must read as
  // MISSING with a reason, never as a pull stuck at 7%. `title` carries the
  // worker's own honest reason string (needs / budget / reclaimable / blocked).
  if (m.refused) {
    return {
      pill: 'wp-pill-refused', glyph: '⊘ missing',
      title: `won't fit on this worker — the download was REFUSED before it `
        + `started (nothing was deleted, no partial file). ${m.refused.reason || ''}`
        + `\n\nRaise this worker's storage allocation (disk_cache_gib), free a `
        + `protected model, or route this model to another box.`,
    }
  }
  if (proposed) return { pill: 'wp-pill-evict', glyph: '🗑 will free', title: 'in the eviction proposal — freed on approval. 📌 A pinned model can appear here: pin keeps its allocation/routing, not its files.' }
  // SHARED CATALOG / UNREAPABLE STORE (k60, 2026-07-31). These bytes are on a
  // store this worker may NEVER delete from, so they are shown but charged to
  // nothing: they contribute 0 to used/over-budget and can never be proposed.
  // Checked ahead of every other state — it is a filesystem fact, not a policy
  // label, and it outranks whatever else the row happens to be.
  if (m.store === 'shared') return { pill: 'wp-pill-shared', glyph: '🔗 shared', title: 'on the SHARED central catalog (the fleet\'s source-of-truth copies, read through from here). Never evicted from this worker and never counted against its storage budget — it is not this box\'s cache.' }
  if (m.store === 'unreapable') return { pill: 'wp-pill-shared', glyph: '🔗 unreapable store', title: 'on a model store this box has not declared local & disposable (HUGPY_MODEL_STORE_REAPABLE unset), so nothing here can be reaped. Shown for visibility; never counted against this worker\'s storage budget.' }
  // NOTE (2026-07-17): 📌 pin no longer protects files, so the PROTECTIVE
  // states are checked FIRST — a pinned model that is also static/loaded/…
  // shows that (real) protection. A bare pinned model falls through to the
  // attribution-only badge below (protected:false → still a candidate).
  if (m.why === 'static') return { pill: 'wp-pill-loaded', glyph: '🔒 static', title: 'static residency — a locked seat, never evicted, files kept on disk (the only tier that blocks eviction/reaping)' }
  if (m.loaded) return { pill: 'wp-pill-serving', glyph: '🔥 loaded', title: 'resident/serving right now — protected' }
  if (m.loading) return { pill: 'wp-pill-heating', glyph: '🔶 heating', title: 'weights loading — protected' }
  // Only a GENUINELY LIVE pull renders as a transfer. Central derives this
  // read-side (owner alive AND bytes moving) — a dead-owner/stalled entry
  // never reaches us as `provisioning`, so it can't show a phantom ⏳ forever.
  if (m.provisioning) return { pill: 'wp-pill-pulling', glyph: '⏳ pulling', title: 'files transferring right now — protected while the bytes land' }
  if (m.assigned) return { pill: 'wp-pill-idle', glyph: '📎 assigned', title: 'designated to this worker — protected in the operator-gated bulk reaper' }
  if (m.pinned) return { pill: 'wp-pill-loaded', glyph: '📌 pinned', title: 'this allocation survives restarts (routing to this worker is durable). Does NOT download the model and does NOT protect its files from eviction — bytes arrive on call and can be evicted to make room (routing is unaffected). Only 🔒 static keeps files on disk.' }
  if (m.protected) return { pill: 'wp-pill-idle', glyph: '🛡 protected', title: m.why || 'protected' }
  return { pill: 'wp-pill-cold', glyph: '○ evictable', title: 'on disk, unassigned & cold — reclaimable' }
}

// ── GPU allocation: concise spill <-> mode mapping ──────────────────────────
// A worker's per-(model) override is an opaque "spill" dict the backend applies
// when it loads the model. We expose it as one mode + an optional VRAM budget.
function spillToMode(spill) {
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

function modeToSpill(mode, gib, ram, threads, gpuBand, ramBand, ctxPct, ctxBand, priority) {
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
const EXPLICIT_BUDGET_KEYS = ['gpu_mem_gib', 'cpu_mem_gib', 'threads', 'tensor_split',
  'gpu_mem_gib_deviation_pct', 'cpu_mem_gib_deviation_pct', 'ctx_pct', 'ctx_deviation_pct',
  'priority', 'leniency_pct', 'priority_device']
function allocIsGgufOnly(spill) {
  if (!spill || Object.keys(spill).length === 0) return false   // autofit — universal
  if (EXPLICIT_BUDGET_KEYS.some(k => k in spill)) return true
  // alloc_mode is value-sensitive: explicit is GGUF-only, max-ram is not.
  return String(spill.alloc_mode || '').trim().toLowerCase() === 'explicit'
}

function spillLabel(spill) {
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

// One concise allocation editor: a mode dropdown + explicit per-model budgets
// (VRAM / RAM / cores) in custom mode — the model's resource contract.
// The worker's EFFECTIVE capacity (bytes) for a resource — the SAME universe the
// honest bars use (limit when set, else physical): VRAM = limits.gpu_mem_gib
// else vram_total; RAM = limits.ram_max_gib else ram_total. This is the honest
// WHOLE the explicit-budget percentages resolve against (never a mixed
// denominator). Returns {bytes, gib, basis} — basis names what the whole is.
const WP_GIB = 2 ** 30
function workerCapacity(worker, which) {
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

// t39 (operator: "the up and down arrows are useless and a hindrance" — every
// tolerance band AND the value it loosens becomes a slider, no number-spinner
// inputs anywhere in this editor). One editable numeric readout shared by every
// slider below: shows the live value; click it to type an exact number (Enter
// commits + re-syncs the slider, Escape/blur cancels). The slider stays the
// PRIMARY control — this is the escape hatch for precise entry.
function SliderReadout({ text, value, onCommit, title }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  if (editing) {
    return (
      <input type="text" inputMode="decimal" className="wp-slider-edit" autoFocus
             value={draft} title={title}
             onChange={e => setDraft(e.target.value)}
             onBlur={() => setEditing(false)}
             onKeyDown={e => {
               if (e.key === 'Enter') { onCommit(draft); setEditing(false) }
               else if (e.key === 'Escape') setEditing(false)
             }} />
    )
  }
  return (
    <button type="button" className="wp-slider-readout"
            title={title || 'Click to type an exact value'}
            onClick={() => { setDraft(value === '' || value == null ? '' : String(value)); setEditing(true) }}>
      {text}
    </button>
  )
}

// A single-value slider (no unit toggle): tolerance bands, CTX target, cores,
// priority. `value` is the raw state string — '' means unset/untouched and
// renders the handle at `min` WITHOUT writing a value (dragging or typing is
// what actually sets it, so an unset band stays unset until the operator
// deliberately picks one — same tri-state the old blank number input had).
// `clampMax` off (priority) lets a typed value exceed the slider's UI ceiling,
// since the backend accepts any priority >= 0 with no hard cap.
function PlainSlider({ label, title, value, onChange, min, max, step = 1, formatValue, clampMax = true, className = '' }) {
  const empty = value === '' || value == null
  // `raw` is the TRUTH (what's actually stored / what Apply will send) — only
  // floor-clamped to `min`, ceiling-clamped to `max` too when clampMax. The
  // <input type=range> value attribute is separately capped to `max` (native
  // range inputs can't render past their own ceiling) so a clampMax=false
  // field whose stored value exceeds the UI ceiling still shows its REAL
  // number in the readout — the thumb just pins at the far end.
  const raw = empty ? null : Math.max(min, clampMax ? Math.min(max, Number(value)) : Number(value))
  const trackVal = raw == null ? min : Math.min(max, raw)
  const text = formatValue ? formatValue(raw) : `${label} ${raw == null ? '—' : raw}`
  return (
    <span className={`wp-plain-slider ${className}`} title={title}>
      {label && <span className="wp-slider-label">{label}</span>}
      <input type="range" min={min} max={max} step={step} value={trackVal} className="wp-range"
             onChange={e => onChange(e.target.value)} />
      <SliderReadout text={text} value={raw == null ? '' : raw} title="Click to type an exact value"
                      onCommit={raw => {
                        if (raw === '') { onChange(''); return }
                        const v = Number(raw)
                        if (!Number.isFinite(v)) return
                        const clamped = Math.max(min, clampMax ? Math.min(max, v) : v)
                        onChange(String(Math.round(clamped)))
                      }} />
    </span>
  )
}

// One VRAM/RAM budget SLIDER with a GiB ⇄ % toggle (operator addendum): plain GiB
// entry OR a percentage of a capacity BASIS. The % resolves to concrete GiB AT
// APPLY TIME (live-previewed here); the wire always carries the resolved GiB
// (no schema change). The slider's own range IS the clamp (GiB ranges to `cap`
// when known, else `fallbackMax` — a UI sanity bound only, not enforcement; %
// ranges 1..100), so a value can never leave the slider needing a separate
// "clamped to..." correction note.
//
// k16 (operator): percent has TWO possible bases — "of worker" (the original
// behavior; `cap` = workerCapacity, e.g. card VRAM) and "of model" (`cap` =
// this model's own size/`need`, e.g. a 10 GiB GGUF). The CALLER picks which
// `cap` to hand in (based on `basis`) — this component doesn't choose; it just
// renders the toggle button (`modelCap`/`basis`/`onBasisChange`) so the
// operator can flip it, and reads whatever `cap` it was given for the slider
// range + live preview math, unchanged from before k16. When `modelCap` is
// null (model size unknown — non-GGUF or unmeasured), the toggle button is
// omitted entirely and the field silently stays on worker-basis — the honest
// degrade (t49's need-null handling, reused here).
function BudgetInput({ label, title, unit, value, onChange, cap, fallbackMax = 128,
                       modelCap = null, basis = 'worker', onBasisChange = null }) {
  // unit: 'gib' | 'pct'. value is the raw string in the current unit.
  const capGib = cap ? cap.gib : null
  const isPct = unit === 'pct'
  const min = isPct ? 1 : 0
  const max = isPct ? 100 : (capGib != null ? +capGib.toFixed(2) : fallbackMax)
  const empty = value === '' || value == null
  const n = empty ? min : Math.max(min, Math.min(max, Number(value)))
  const resolvedGib = (() => {
    if (empty) return null
    if (isPct) {
      if (capGib == null) return null
      return +(capGib * n / 100).toFixed(2)
    }
    return n
  })()
  return (
    <span className="wp-alloc-value" title={title}>
      <span className="wp-slider-label">{label}</span>
      <input type="range" min={min} max={max} step={isPct ? 1 : 0.5} value={n} className="wp-range"
             onChange={e => onChange(e.target.value, unit)} />
      <SliderReadout text={empty ? `${label} —` : (isPct ? `${n}%` : `${n.toFixed(1)} GiB`)}
                      value={empty ? '' : n}
                      title={`Click to type an exact ${isPct ? 'percent' : 'GiB'} value`}
                      onCommit={raw => {
                        if (raw === '') { onChange('', unit); return }
                        const v = Number(raw)
                        if (!Number.isFinite(v)) return
                        onChange(String(Math.max(min, Math.min(max, v))), unit)
                      }} />
      <button type="button" className="wp-budget-unit"
              title={`Toggle ${label} between GiB and % of its current capacity basis`}
              onClick={() => onChange(value, unit === 'gib' ? 'pct' : 'gib')}>
        {unit === 'pct' ? '%' : 'GiB'}
      </button>
      {/* k16: percent BASIS toggle — worker capacity (default, byte-identical
          to pre-k16 behavior) vs this model's own size. Only offered in pct
          units, and only when the model's size is actually known. */}
      {isPct && modelCap && onBasisChange && (
        <button type="button" className="wp-budget-basis"
                title={basis === 'model'
                  ? `Percent of this model's own size (${modelCap.gib.toFixed(1)} GiB). Click for percent of the worker instead.`
                  : `Percent of the worker's effective ${label} capacity. Click for percent of this model's own size (${modelCap.gib.toFixed(1)} GiB) instead.`}
                onClick={() => onBasisChange(basis === 'model' ? 'worker' : 'model')}>
          {basis === 'model' ? 'of model' : 'of worker'}
        </button>
      )}
      {isPct && (
        <span className="wp-budget-preview">
          {cap == null
            ? '— capacity unknown'
            : (empty ? `of ${capGib.toFixed(1)} GiB ${cap.basis}`
               : `= ${resolvedGib != null ? resolvedGib.toFixed(1) : '—'} GiB of ${capGib.toFixed(1)} (${cap.basis})`)}
        </span>
      )}
    </span>
  )
}

// AllocControl — RETIRED 2026-07-24 (no live call sites). The bulk multi-select
// bar now uses BulkAllocControl (parity with the per-model AllocModeMenu: the
// five flat modes + a first-class "Default (derived)" clear + a model-denominated
// explicit split); the per-row cell uses AllocModeMenu. This component (and its
// AllocControl-only helpers BudgetInput / workerCapacity / modeToSpill) is kept
// only as reference for the old device-denominated budget editor and is safe to
// delete wholesale. NOTHING renders it.
//   engineGguf: true  = a gguf model (offer every mode incl. explicit budgets)
//               false = a non-gguf single model — offer Autofit / Max GPU / CPU
//                       only (engine-agnostic PLACEMENT INTENT; the worker maps
//                       n_gpu_layers to transformers placement), but HIDE the
//                       Explicit-budgets mode (the sole GGUF-exclusive class, t26)
//               null  = MIXED/bulk selection (offer all modes; the caller warns
//                       that the explicit-budget mode applies to gguf members only)
//   worker: the worker record, for the % ⇄ GiB capacity denominators.
//   need (t49): {bytes, gib} — THIS model's own total requirement (GGUF
//     effective quant), when known. When present, the VRAM/RAM value sliders
//     COUPLE: together they always sum to 100% of `need` (moving one moves the
//     other). Single-model site only — a bulk selection has no single
//     denominator, so it's left null there (honest degrade to independent
//     sliders, the pre-t49 behavior).
//   bulkKeys / getModelBytes (t48): when this editor is driving a BULK
//     selection, bulkKeys is the full list of selected model keys and
//     getModelBytes(key) looks up that model's own {bytes} — used ONLY at
//     Apply time, and ONLY when a VRAM/RAM budget is in PERCENT units, to
//     resolve the percent against EACH member's own size instead of baking one
//     absolute (from the worker's capacity) and stamping it on every member.
//   k16 (operator): a percent slider's BASIS — what 100% actually MEANS — is
//     now selectable per field (VRAM/RAM independently): "of worker" (default,
//     `workerCapacity` — pre-k16 behavior, byte-identical) or "of model" (this
//     model's own `need` — e.g. VRAM 60% of a 10 GiB model resolves straight
//     to gpu_mem_gib≈6, and via the t49 coupling the RAM side becomes the
//     ≈4 GiB remainder). Wire-level this changes NOTHING — still absolute
//     gpu_mem_gib/cpu_mem_gib (+ t21 bands on top); basis is purely how the UI
//     turns a percent into that absolute. Single-model site only (needGib is
//     null for a bulk selection, so the toggle is hidden there — bulk keeps
//     its existing t48 per-member resolution, which is already model-relative
//     for percent, just without a live coupled preview).
function AllocControl({ spill, onApply, onCancel, applyLabel = 'Apply', engineGguf = null, worker = null,
                        need = null, bulkKeys = null, getModelBytes = null }) {
  const init = spillToMode(spill)
  const [mode, setMode] = useState(init.mode)
  const [gib, setGib]   = useState(init.gib)
  const [ram, setRam]   = useState(init.ram)
  const [threads, setThreads] = useState(init.threads)
  // t21 tolerance-band + priority fields (custom/explicit-budget only).
  const [gpuBand, setGpuBand]   = useState(init.gpuBand)
  const [ramBand, setRamBand]   = useState(init.ramBand)
  const [ctxPct, setCtxPct]     = useState(init.ctxPct)
  const [ctxBand, setCtxBand]   = useState(init.ctxBand)
  const [priority, setPriority] = useState(init.priority)
  // Per-field unit (GiB | %) for VRAM and RAM — the addendum's dual input.
  const [vramUnit, setVramUnit] = useState('gib')
  const [ramUnit, setRamUnit]   = useState('gib')
  // k16: per-field percent BASIS — 'worker' (default, pre-k16 behavior,
  // BYTE-IDENTICAL for anyone who never touches this) or 'model' (this
  // model's own size). Independent per field so a mixed VRAM-of-model /
  // RAM-of-worker choice is possible, though the common case (needGib known)
  // is to flip both — the t49 coupling keeps the two sides summing to `need`
  // in ABSOLUTE GiB regardless of which basis either field displays in.
  const [vramBasis, setVramBasis] = useState('worker')
  const [ramBasis, setRamBasis]   = useState('worker')
  const vramCap = workerCapacity(worker, 'vram')
  const ramCap  = workerCapacity(worker, 'ram')
  // t49: the coupling denominator — this model's own required GiB, or null
  // (no denominator → no coupling, per-field sliders stay independent).
  const needGib = (need && need.gib > 0) ? need.gib : null
  // k16: this model's own size as a selectable percent BASIS — same shape as
  // workerCapacity's return ({bytes, gib, basis}) so it drops into BudgetInput's
  // `cap` prop with no special-casing. null when unknown (non-GGUF / unmeasured
  // — the same gate t49 already applies to needGib), which is also what hides
  // the "of model" toggle button (BudgetInput only renders it when modelCap is
  // non-null) — the honest degrade.
  const modelSizeCap = needGib != null ? { bytes: need.bytes, gib: needGib, basis: 'model size' } : null
  // Effective cap actually fed to each BudgetInput / used to resolve that
  // field's percent — worker capacity unless this field's basis is flipped to
  // 'model' AND a model size is actually known. This is the ONLY place basis
  // changes behavior; a field left at the default 'worker' basis resolves
  // exactly as it did before k16.
  const vramEffCap = (vramBasis === 'model' && modelSizeCap) ? modelSizeCap : vramCap
  const ramEffCap  = (ramBasis === 'model' && modelSizeCap) ? modelSizeCap : ramCap

  // Resolve a budget field to concrete GiB (the wire value): % → cap·pct/100;
  // GiB clamped to cap. Pure — usable both live (coupling, t49) and at Apply
  // time (t48/wire). Mirrors BudgetInput's own preview math.
  const resolveGib = (raw, unit, cap) => {
    if (raw === '' || raw == null) return ''
    const n = Number(raw)
    if (!Number.isFinite(n)) return ''
    if (unit === 'pct') {
      if (!cap) return ''   // no denominator → drop the field (can't resolve %)
      const pct = Math.min(100, Math.max(1, n))
      return String(+(cap.gib * pct / 100).toFixed(2))
    }
    return String(cap ? Math.min(n, +cap.gib.toFixed(2)) : n)
  }
  // Same resolution, as a Number-or-null (for live coupling math — resolveGib
  // returns '' on "can't resolve", which is fine for the wire but awkward for
  // arithmetic here).
  const liveAbs = (raw, unit, cap) => {
    const s = resolveGib(raw, unit, cap)
    return s === '' ? null : Number(s)
  }
  // The inverse: an absolute GiB amount, expressed back in a field's CURRENT
  // unit (a literal GiB string, or a percent of THAT field's own cap) — so
  // coupling can write into the partner field without disturbing whichever
  // unit it's already displaying in.
  const absToUnit = (abs, unit, cap) => {
    if (abs == null) return ''
    if (unit === 'pct') {
      if (!cap || !cap.gib) return ''   // can't express as % without a cap
      return String(Math.max(1, Math.min(100, Math.round(abs / cap.gib * 100))))
    }
    return String(+Math.max(0, abs).toFixed(2))
  }

  // A field's change carries both the value and the (possibly toggled) unit.
  // t49: when this model's own `need` is known, VRAM and RAM are a COUPLED
  // split of it — moving one recomputes the other so the two always sum to
  // needGib (both the drag path AND the SliderReadout click-to-type path funnel
  // through here, so both stay coupled). No denominator → falls through
  // untouched (today's independent-slider behavior).
  // k16: coupling resolves each side against its OWN effective cap (worker or
  // model, per that field's basis) — a field on model-basis resolves its %
  // against `need` directly (pct × need = gib, exactly the operator's "10 GiB
  // model, VRAM 60% → 6 GiB GPU / 4 GiB RAM" spec); a field left on
  // worker-basis resolves as it always has. Either way the ABSOLUTE GiB values
  // still sum to needGib — basis only changes how each side's percent maps to
  // GiB, never the coupling invariant itself.
  const onVram = (v, unit) => {
    setVramUnit(unit); setGib(v)
    if (needGib != null) {
      const vAbs = liveAbs(v, unit, vramEffCap)
      if (vAbs != null) {
        const rAbs = Math.max(0, +(needGib - Math.min(vAbs, needGib)).toFixed(2))
        setRam(absToUnit(rAbs, ramUnit, ramEffCap))
      }
    }
  }
  const onRam = (v, unit) => {
    setRamUnit(unit); setRam(v)
    if (needGib != null) {
      const rAbs = liveAbs(v, unit, ramEffCap)
      if (rAbs != null) {
        const vAbs = Math.max(0, +(needGib - Math.min(rAbs, needGib)).toFixed(2))
        setGib(absToUnit(vAbs, vramUnit, vramEffCap))
      }
    }
  }
  // Live "share of required" readouts (t49) — display only.
  const vramAbsNow = needGib != null ? liveAbs(gib, vramUnit, vramEffCap) : null
  const ramAbsNow  = needGib != null ? liveAbs(ram, ramUnit, ramEffCap) : null

  // t48: this SAME model-key's own capacity (for a bulk PERCENT resolution) —
  // mirrors workerCapacity's return shape but denominated on the model's own
  // size instead of the worker's. null when the feed doesn't know this model's
  // size (falls back to the worker cap at the call site, same as today).
  const modelCapFor = (key) => {
    const info = getModelBytes ? getModelBytes(key) : null
    const bytes = info && info.bytes != null ? info.bytes : null
    return bytes != null ? { bytes, gib: bytes / WP_GIB, basis: 'model size' } : null
  }

  // Cores slider ceiling: the worker's own physical thread count when the
  // heartbeat reports it (caps.threads), else its central limit, else a sane
  // UI fallback — never a hard rule, just a slider bound (t39).
  const coresCap = (worker && worker.caps && worker.caps.threads)
    || (worker && worker.limits && worker.limits.threads) || 64

  // Only the EXPLICIT-BUDGET mode is gguf-gated now (t26). Autofit / Max GPU /
  // CPU only are placement intent offered for every engine.
  const explicitOffered = engineGguf !== false   // hide "custom" only on a known non-gguf
  // If a non-gguf model somehow arrives already in custom mode, fall the select
  // back to autofit so it never shows a mode it can't offer.
  const effMode = (!explicitOffered && mode === 'custom') ? 'auto' : mode
  return (
    <div className="wp-alloc">
      <select value={effMode} onChange={e => setMode(e.target.value)}
              title="How much of this worker to give this model">
        {/* k37 rename (labels only; the internal values keep the legacy spill
            semantics the bulk builder emits): auto=max-gpu, gpu=gpu-only,
            cpu=ram-only, custom=explicit. */}
        <option value="auto">Max GPU (fit &amp; spill)</option>
        <option value="gpu">GPU only (all layers)</option>
        <option value="cpu">RAM only</option>
        {explicitOffered && <option value="custom">Explicit budgets…</option>}
      </select>
      {!explicitOffered && (
        <span className="wp-alloc-note" title="Explicit per-model budgets are a GGUF-only concept; the transformers loader takes only a device_map placement, not a per-model budget. Autofit / Max GPU / CPU only DO apply — the worker maps them to transformers placement.">
          non-GGUF — no explicit budgets (placement modes apply)
        </span>
      )}
      {effMode === 'custom' && explicitOffered && (
        <>
          {engineGguf == null && (
            <span className="wp-alloc-note" title="Explicit budgets are a GGUF concept — in a mixed selection they apply to the GGUF models only; transformers/comfy members are skipped. Autofit / Max GPU / CPU only apply to everyone.">
              GGUF-only — transformers members skipped
            </span>
          )}
          {/* t39 (operator): every tolerance band is a slider, paired DIRECTLY
              BELOW the value it loosens — one bordered stack per variable
              (VRAM / RAM / CTX). The value control is a slider too; no
              number-spinner inputs anywhere in this editor. */}
          {needGib != null && (
            <span className="wp-alloc-note wp-alloc-coupled"
                  title={`This model's own requirement is ${needGib.toFixed(1)} GiB (effective quant). VRAM and RAM below are COUPLED — they always sum to 100% of it; moving one moves the other.`}>
              coupled to {needGib.toFixed(1)} GiB required
            </span>
          )}
          <div className="wp-alloc-group">
            <BudgetInput label="VRAM" unit={vramUnit} value={gib} onChange={onVram} cap={vramEffCap}
                         fallbackMax={96}
                         modelCap={modelSizeCap} basis={vramBasis} onBasisChange={setVramBasis}
                         title={needGib != null
                           ? `VRAM budget — coupled with RAM to this model's ${needGib.toFixed(1)} GiB requirement`
                           : "VRAM budget — autofit plans layers within this"} />
            {needGib != null && vramAbsNow != null && (
              <span className="wp-alloc-share" title="This side's share of the model's total requirement.">
                {Math.round(vramAbsNow / needGib * 100)}% of required
              </span>
            )}
            <PlainSlider label="±" min={0} max={100} className="wp-alloc-band"
                         value={gpuBand} onChange={setGpuBand}
                         formatValue={v => v == null ? 'no band' : `±${v}%`}
                         title="Tolerance band: under contention this allocation may deviate ± this percent of the worker's VRAM capacity from its target before eviction." />
          </div>
          <div className="wp-alloc-group">
            <BudgetInput label="RAM" unit={ramUnit} value={ram} onChange={onRam} cap={ramEffCap}
                         fallbackMax={512}
                         modelCap={modelSizeCap} basis={ramBasis} onBasisChange={setRamBasis}
                         title={needGib != null
                           ? `RAM budget — coupled with VRAM to this model's ${needGib.toFixed(1)} GiB requirement`
                           : "RAM budget — the CPU-resident share must fit this or the load is refused"} />
            {needGib != null && ramAbsNow != null && (
              <span className="wp-alloc-share" title="This side's share of the model's total requirement.">
                {Math.round(ramAbsNow / needGib * 100)}% of required
              </span>
            )}
            <PlainSlider label="±" min={0} max={100} className="wp-alloc-band"
                         value={ramBand} onChange={setRamBand}
                         formatValue={v => v == null ? 'no band' : `±${v}%`}
                         title="Tolerance band: under contention this allocation may deviate ± this percent of the worker's RAM capacity from its target before eviction." />
          </div>
          <div className="wp-alloc-group">
            <PlainSlider label="CTX" min={1} max={100} className="wp-alloc-value"
                         value={ctxPct} onChange={setCtxPct}
                         formatValue={v => v == null ? 'CTX —' : `${v}%`}
                         title="Context allocation: percent of the model's max context window reserved for KV." />
            <PlainSlider label="±" min={0} max={100} className="wp-alloc-band"
                         value={ctxBand} onChange={setCtxBand}
                         formatValue={v => v == null ? 'no band' : `±${v}%`}
                         title="Tolerance band for the context allocation under contention (ctx is the cheapest thing to flex)." />
          </div>
          <PlainSlider label="cores" min={1} max={coresCap} className="wp-alloc-solo"
                       value={threads} onChange={setThreads}
                       formatValue={v => v == null ? 'cores —' : `${v} cores`}
                       title="Generation threads for this model" />
          <PlainSlider label="priority" min={0} max={10} clampMax={false} className="wp-alloc-solo"
                       value={priority} onChange={setPriority}
                       formatValue={v => v == null ? 'p —' : `p${v}`}
                       title="Higher priority may compress lower-priority neighbors within their bands before anything is evicted. 0 = normal. Slider tops out at 10 for reach; type a higher number if you need one — the backend accepts any non-negative integer." />
        </>
      )}
      <button className="wp-alloc-apply"
              onClick={() => {
                const flatSpill = modeToSpill(
                  effMode,
                  resolveGib(gib, vramUnit, vramEffCap),
                  resolveGib(ram, ramUnit, ramEffCap),
                  threads,
                  gpuBand,
                  ramBand,
                  ctxPct,
                  ctxBand,
                  priority,
                )
                // t48: a BULK selection (bulkKeys) with a PERCENT VRAM/RAM
                // budget must resolve that percent against EACH model's OWN
                // size — not once against the worker's capacity, baked into
                // one absolute, then stamped identically on every member (the
                // reported bug: every model got the number that only happened
                // to be right for whichever model it was computed against).
                // Only built when it can actually differ per model (custom +
                // at least one field in % units); otherwise every member
                // legitimately wants the SAME contract (autofit / max GPU /
                // CPU only / an operator-typed absolute GiB) and flatSpill
                // alone is correct — perModel stays null and the caller keeps
                // using the one shared spill.
                //
                // k16 note: the "of model" percent BASIS toggle is single-model
                // only (needGib — and so modelSizeCap — is always null at the
                // bulk call site, since `need` is never passed there; see the
                // AllocControl docstring). vramCap/ramCap below are therefore
                // already correct (== vramEffCap/ramEffCap when there's no
                // model basis to apply) — bulk keeps its existing t48
                // per-member-size resolution, which already IS "of model"
                // semantics for percent, just without a live toggle/preview
                // (there's no single model to preview a coupled split against).
                let perModel = null
                if (bulkKeys && bulkKeys.length > 0 && effMode === 'custom'
                    && (vramUnit === 'pct' || ramUnit === 'pct')) {
                  perModel = {}
                  for (const key of bulkKeys) {
                    const ownVramCap = vramUnit === 'pct' ? (modelCapFor(key) || vramCap) : vramCap
                    const ownRamCap  = ramUnit === 'pct' ? (modelCapFor(key) || ramCap) : ramCap
                    perModel[key] = modeToSpill(
                      effMode,
                      resolveGib(gib, vramUnit, ownVramCap),
                      resolveGib(ram, ramUnit, ownRamCap),
                      threads,
                      gpuBand,
                      ramBand,
                      ctxPct,
                      ctxBand,
                      priority,
                    )
                  }
                }
                onApply(flatSpill, perModel)
              }}>
        {applyLabel}
      </button>
      {onCancel && <button className="wp-alloc-cancel" onClick={onCancel} title="Cancel">×</button>}
    </div>
  )
}

// Residency picker (slice 8): clicking the residency tag NEVER changes
// anything — it opens this menu (same inline-expansion idiom as AllocControl).
// Only picking a DIFFERENT state fires a config change, so each ~5s agent
// restart is a deliberate choice.
//
// v3 final semantics (operator-locked): the POLICY axis has exactly two
// tiers — on-demand (the default; no stored override) and static (locked
// seat; permanent with 📌 pin). "Serving" is purely a STATE (a model in a
// slot) — the live pills (🔥 serving / ⚡ answering / ○ cold) tell that
// truth; no policy is ever called serving.
const RESIDENCY_OPTIONS = [
  ['on-demand', '⏲ on-demand', 'loads on call; holds its slot until another model needs the seat (default)'],
  ['static', '🔒 static', 'always kept on this worker: downloaded eagerly, never evicted (the only tier that keeps files on disk)'],
]

function ResidencyMenu({ mode, onPick, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose() }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [onClose])
  return (
    <div className="wp-res-menu" ref={ref} role="menu"
         onKeyDown={e => { if (e.key === 'Escape') onClose() }}>
      {RESIDENCY_OPTIONS.map(([value, label, desc]) => (
        <button key={value} role="menuitemradio" aria-checked={value === mode}
                className={`wp-res-opt${value === mode ? ' wp-res-opt-on' : ''}`}
                title={value === mode ? 'current state — click to close'
                  : 'applies via a ~5s agent restart'}
                onClick={() => (value === mode ? onClose() : onPick(value))}>
          <span className="wp-res-opt-label">{label}</span>
          <span className="wp-res-opt-desc">— {desc}</span>
          {value === mode && <span className="wp-res-opt-mark">✓</span>}
        </button>
      ))}
    </div>
  )
}

// ── The FIVE flat allocation modes (k37, Slice B) ───────────────────────────
// Operator ask 2026-07-24: the alloc value is picked IN PLACE — a compact menu
// anchored at the click site. Four of the five modes are ZERO-KNOB (apply the
// instant they're picked); ONLY "explicit" opens extra chrome (its knobs).
//
// These are the operator-facing NAMES (managers/alloc_modes.ALLOC_MODES). The
// wire encoding is handled server-side (/assign runs normalize_spill), so the
// UI just sends {alloc_mode: <name>, …} and central rewrites the coarse trio
// onto the legacy n_gpu_layers wire — the UI never touches n_gpu_layers again.
//   gpu-only  all layers on the GPU, no spill (old console "Max GPU", -1)
//   ram-only  all in RAM, never the GPU              (old console "CPU only")
//   max-gpu   as much GPU as fits, spill the rest — THE DEFAULT (old "autofit")
//   max-ram   as much RAM as fits, spill the rest to GPU              (NEW)
//   explicit  target VRAM/RAM budgets + leniency%% + device priority  (NEW)
const ALLOC_MODE_OPTIONS = [
  ['gpu-only', '🖥 GPU only', 'all layers on the GPU, no spill — won’t fit the GPU (after evict) → refused'],
  ['ram-only', '🧠 RAM only', 'all in host RAM, never the GPU (binds CPU even with a GPU present)'],
  ['max-gpu',  '⚡ Max GPU',  'as much GPU as fits, spill the rest to RAM — the DEFAULT (serves-and-spills, never OOMs)'],
  ['max-ram',  '💾 Max RAM',  'as much RAM as fits, spill the rest to the GPU'],
  ['explicit', '🎛 Explicit…', 'target VRAM/RAM budgets + a leniency %% + a device priority — the only mode with knobs'],
]
// Modes a non-GGUF (transformers/comfy) model may pick. The four non-explicit
// modes have a working non-GGUF meaning: the coarse trio rides accelerate, and
// max-ram was opened for non-GGUF 2026-07-24 (transformers RAM-priority
// max_memory, diffusers cpu-offload — Slice C wired the loaders). Only
// ``explicit`` stays GGUF-only (its banded leniency floor has no transformers
// analogue), so the menu DISABLES only explicit for a known non-gguf model.
// Mirrors alloc_modes.NONGGUF_ALLOWED_MODES.
const NONGGUF_ALLOC_MODES = new Set(['gpu-only', 'ram-only', 'max-gpu', 'max-ram'])
const GGUF_ONLY_MODE_TIP = 'explicit is GGUF-only — banded leniency has no transformers analogue'

// The model's EFFECTIVE flat mode from its persisted override/spill — the JS
// mirror of managers.alloc_modes.derive_alloc_mode (the WorkersPanel serving
// rows carry only worker.spill_by_model[key], not a server-derived alloc_mode
// field, so the derivation happens here). A blank model == max-gpu (the
// default: fits-and-spills, never OOMs — defaults-are-promises).
function deriveAllocMode(spill) {
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
function allocModeLabel(mode) {
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
function resolvedSeatMode(ngl, total, gpuPct) {
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

// Explicit-budget knobs — the ONLY extra chrome, shown inside the anchored
// popover when "explicit" is chosen.
//
// MODEL-DENOMINATED (operator ruling 2026-07-24): the split is ONE complementary
// slider OF THE MODEL's own footprint — "% of this model on GPU", with the RAM
// share the auto-shown complement (gpu% + ram% = 100% of the MODEL, always).
// NEVER two independent device-percentages: the old bands editor denominated
// each side against a DIFFERENT device (card VRAM vs box RAM), so "1% of VRAM"
// left "20% of RAM" — nonsensical. Here 1% GPU ⇒ 99% RAM, both of the SAME whole.
//   • Default = 100% on the priority device (gpu default).
//   • `need` (the model's effective-quant GiB, threaded from the row) turns the
//     split into live absolute GiB shown next to the percentages, and those
//     absolutes are what POSTs (gpu_mem_gib / cpu_mem_gib — the wire is
//     absolute; the %% framing is UI-side, Slice B). No size known → the split
//     still POSTs the %% intent but shows no GiB (blank, never card capacity).
//   • Leniency is % OF THE MODEL with the floor shown live.
function ExplicitPanel({ spill, worker, need, onApply, onCancel }) {
  const needGib = (need && need.gib > 0) ? need.gib : null
  // Seed the GPU share from any persisted absolute budgets (as a % of need when
  // we know it), else from the priority device (100% on the favored side).
  const seedPriority = (spill && spill.priority_device === 'ram') ? 'ram' : 'gpu'
  const seedGpuPct = (() => {
    if (needGib != null && spill && spill.gpu_mem_gib != null) {
      return Math.max(0, Math.min(100, Math.round(spill.gpu_mem_gib / needGib * 100)))
    }
    if (needGib != null && spill && spill.cpu_mem_gib != null) {
      return Math.max(0, Math.min(100, 100 - Math.round(spill.cpu_mem_gib / needGib * 100)))
    }
    return seedPriority === 'ram' ? 0 : 100   // 100% on the priority device
  })()
  const [gpuPct, setGpuPct] = useState(seedGpuPct)
  const [leniency, setLeniency] = useState(
    (spill && spill.leniency_pct != null) ? String(spill.leniency_pct) : '')
  const [priorityDevice, setPriorityDevice] = useState(seedPriority)
  const ramPct = 100 - gpuPct
  // Live absolute GiB from the split (only when the model's size is known).
  const gpuGib = needGib != null ? +(needGib * gpuPct / 100).toFixed(1) : null
  const ramGib = needGib != null ? +(needGib * ramPct / 100).toFixed(1) : null
  // Leniency floor readout: N% of the model may shift OFF its preferred device,
  // so the floor on the priority device is (share − leniency), the other side
  // gains it. Display-only — the backend does the real step-down.
  const lenN = (leniency === '' || leniency == null) ? 0 : Math.max(0, Math.min(100, Number(leniency)))
  const floorLine = (() => {
    if (lenN <= 0) return 'floor: exactly the split above (strict)'
    // The preferred device may cede up to lenN% of the model to the other.
    const gpuFloor = priorityDevice === 'gpu' ? Math.max(0, gpuPct - lenN) : Math.min(100, gpuPct + lenN)
    const ramFloor = 100 - gpuFloor
    return `floor: ${gpuFloor}% GPU / ${ramFloor}% RAM`
  })()
  return (
    <div className="wp-allocmode-explicit">
      <div className="wp-allocmode-explicit-head" title={GGUF_ONLY_MODE_TIP}>
        🎛 Explicit — split of {needGib != null ? `${needGib.toFixed(1)} GiB model` : 'the model'}
      </div>
      {/* ONE complementary split of the MODEL: GPU share; RAM is the remainder. */}
      <PlainSlider label="on GPU" min={0} max={100} className="wp-alloc-value"
                   value={String(gpuPct)} onChange={v => setGpuPct(Math.max(0, Math.min(100, Math.round(Number(v) || 0))))}
                   formatValue={v => `${v == null ? 0 : v}% GPU`}
                   title="Share of THIS MODEL's footprint placed on the GPU. The rest goes to host RAM — the two always sum to 100% of the model (not of any device)." />
      <div className="wp-allocmode-split" title="The complementary split of the model's own footprint.">
        <span className="wp-allocmode-split-gpu">
          {gpuPct}% GPU{gpuGib != null ? ` · ${gpuGib.toFixed(1)} GiB` : ''}
        </span>
        <span className="wp-allocmode-split-sep">/</span>
        <span className="wp-allocmode-split-ram">
          {ramPct}% RAM{ramGib != null ? ` · ${ramGib.toFixed(1)} GiB` : ''}
        </span>
        {needGib != null && (
          <span className="wp-allocmode-split-need">
            = {needGib.toFixed(1)} GiB model
          </span>
        )}
      </div>
      <PlainSlider label="leniency" min={0} max={100} className="wp-alloc-solo"
                   value={leniency} onChange={setLeniency}
                   formatValue={v => v == null ? 'leniency —' : `${v}%`}
                   title="Leniency: up to this percent of the model may shift off its preferred device before the load is refused. 0 = strict." />
      <div className="wp-allocmode-floor" title="The worst-case placement the load will still accept before refusing.">{floorLine}</div>
      <span className="wp-allocmode-prio" title="Priority device: which side the split favors and which way leniency degrades. GPU default.">
        <span className="wp-slider-label">priority</span>
        <button type="button"
                className={`wp-allocmode-prio-btn${priorityDevice === 'gpu' ? ' wp-allocmode-prio-on' : ''}`}
                onClick={() => setPriorityDevice('gpu')} title="Favor the GPU (default)">GPU</button>
        <button type="button"
                className={`wp-allocmode-prio-btn${priorityDevice === 'ram' ? ' wp-allocmode-prio-on' : ''}`}
                onClick={() => setPriorityDevice('ram')} title="Favor host RAM">RAM</button>
      </span>
      <div className="wp-allocmode-explicit-actions">
        <button className="wp-alloc-apply" onClick={() => {
          const s = { alloc_mode: 'explicit' }
          // Wire is ABSOLUTE GiB (Slice B). Send the derived absolutes when the
          // model size is known; otherwise send nothing for the GiB targets and
          // let the backend fall back to its own defaults (never card capacity).
          if (gpuGib != null) s.gpu_mem_gib = gpuGib
          if (ramGib != null) s.cpu_mem_gib = ramGib
          if (leniency !== '' && leniency != null) s.leniency_pct = Number(leniency)
          s.priority_device = priorityDevice
          onApply(s)
        }}>Apply</button>
        <button className="wp-alloc-cancel" onClick={onCancel} title="Cancel">×</button>
      </div>
    </div>
  )
}

// A short GiB rendering for the numbered disable tooltips — mirrors the backend
// 409's _fmt_gib ("68.0GiB"), binary units, so the UI reason matches the wire's.
function fmtGiB(bytes) {
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
function allocDisableReason(value, ctx, engineGguf) {
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

// AllocModeMenu — the in-place, click-site alloc picker (operator ask
// 2026-07-24). A compact anchored popover listing the five flat modes with the
// current one checked, PLUS a sixth "Auto — derived: <mode>" entry at the top
// (ruling 2): selecting it POSTs an empty spill {} to CLEAR the contract, so the
// model reverts to tracking the read-time derivation. Picking any of the four
// zero-knob concrete modes fires onPick immediately (menu closes, cell updates
// optimistically). Picking "explicit" swaps the popover body to ExplicitPanel —
// the ONLY mode that needs room. Outside-click / Esc close it (mirrors
// ResidencyMenu).
//
// FEASIBILITY DISABLES (ruling 1): `feasible` (the per-(model,worker) set from
// alloc_by_worker[wid].feasible, or the model-level union, or null) disables any
// mode NOT in it; a disabled option's tooltip names the reason with numbers when
// they're known (feasibleCtx). MISSING feasibility (feasible==null) disables
// NOTHING — fail-open, exactly like the backend. engineGguf=false remains a
// belt-and-suspenders disable for max-ram/explicit even if feasibility is absent.
function AllocModeMenu({ mode, spill, worker, need, engineGguf, feasible, feasibleCtx,
                        derivedMode, anchorRef, onPick, onApplyExplicit, onRevertDerived, onClose }) {
  const ref = useRef(null)
  const [explicitOpen, setExplicitOpen] = useState(false)
  // The serving table lives in an overflow:auto scroll box, which would CLIP an
  // absolutely-positioned popover. Position the menu with position:fixed, from
  // the trigger button's viewport rect, so it floats over the clip and still
  // reads as anchored AT THE CLICK SITE. Recomputed on open + on scroll/resize.
  const [pos, setPos] = useState(null)
  useEffect(() => {
    const place = () => {
      const el = anchorRef && anchorRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      // CLAMP to the viewport (compact mode, 2026-07-28): position:fixed from the
      // trigger's left edge runs off-screen on a narrow phone viewport, where the
      // menu (min-width 240px) is wider than the space to the right of the click.
      // Measure the menu when it exists, fall back to its min-width before the
      // first paint; 4px gutters both sides, never negative.
      const mw = (ref.current && ref.current.offsetWidth) || 240
      const maxLeft = Math.max(4, (window.innerWidth || 0) - mw - 4)
      setPos({ top: r.bottom + 4, left: Math.max(4, Math.min(r.left, maxLeft)) })
    }
    place()
    // Second pass after the first paint, when the menu has a real width to clamp
    // against (the first pass can only use the 240px min-width guess).
    const raf = typeof requestAnimationFrame === 'function' ? requestAnimationFrame(place) : null
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      if (raf != null) cancelAnimationFrame(raf)
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [anchorRef])
  useEffect(() => {
    const onDown = (e) => {
      if (ref.current && ref.current.contains(e.target)) return
      if (anchorRef && anchorRef.current && anchorRef.current.contains(e.target)) return
      onClose()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [onClose, anchorRef])
  return (
    <div className="wp-allocmode-menu" ref={ref} role="menu"
         style={pos ? { position: 'fixed', top: pos.top, left: pos.left } : undefined}
         onKeyDown={e => { if (e.key === 'Escape') onClose() }}>
      {explicitOpen ? (
        <ExplicitPanel spill={spill} worker={worker} need={need}
                       onApply={(s) => { onApplyExplicit(s) }}
                       onCancel={() => setExplicitOpen(false)} />
      ) : (
        <>
          {/* Ruling 2 — the SIXTH, top entry: "Auto — derived: <mode>". Clears
              the contract (POST spill {}) so the model tracks the read-time
              derivation. Shown checked when NO contract is pinned (isDerived is
              decided at the cell; here we mark it current when the effective mode
              equals the derived one AND onRevertDerived is the no-op we'd apply).
              The derived mode name comes from alloc_by_worker[wid].derived_default
              (or the model-level derivation) — falls back to the current mode. */}
          {onRevertDerived && (() => {
            const dm = derivedMode || mode
            return (
              <button key="__auto__" role="menuitem"
                      className="wp-allocmode-opt wp-allocmode-opt-auto"
                      title="Revert to the DERIVED default — clears any pinned contract so this model tracks the derivation (and improves with it, as measured values land). This is the always-available revert-to-default."
                      onClick={() => onRevertDerived()}>
                <span className="wp-allocmode-opt-label">↺ Auto</span>
                <span className="wp-allocmode-opt-desc">— derived: {allocModeLabel(dm)}</span>
              </button>
            )
          })()}
          {ALLOC_MODE_OPTIONS.map(([value, label, desc]) => {
            // Feasibility DISABLES (ruling 1): a mode not in the per-(model,worker)
            // feasible set is disabled with a numbered reason; MISSING feasibility
            // (feasible==null) disables nothing (fail-open, like the backend). The
            // engine gate remains as a fallback disable when feasibility is absent.
            const feasDisabled = Array.isArray(feasible)
              ? !feasible.includes(value)
              : (engineGguf === false && !NONGGUF_ALLOC_MODES.has(value))
            const reason = feasDisabled
              ? (allocDisableReason(value, feasibleCtx, engineGguf) || 'not feasible for this model on this worker')
              : null
            const disabled = feasDisabled
            const isCur = value === mode
            return (
              <button key={value} role="menuitemradio" aria-checked={isCur} disabled={disabled}
                      className={`wp-allocmode-opt${isCur ? ' wp-allocmode-opt-on' : ''}`}
                      title={disabled ? reason
                        : isCur ? 'current mode — click to close'
                        : (value === 'explicit' ? 'opens the explicit-budget knobs' : 'applies immediately')}
                      onClick={() => {
                        if (disabled) return
                        if (value === 'explicit') { setExplicitOpen(true); return }
                        if (isCur) { onClose(); return }
                        onPick(value)
                      }}>
                <span className="wp-allocmode-opt-label">{label}</span>
                <span className="wp-allocmode-opt-desc">— {desc}</span>
                {isCur && <span className="wp-allocmode-opt-mark">✓</span>}
              </button>
            )
          })}
        </>
      )}
    </div>
  )
}

// ── BULK alloc parity (operator ask 2026-07-24) ─────────────────────────────
// The multi-select bar's alloc editor, brought to PARITY with the per-model
// AllocModeMenu that just shipped. It offers the SAME five flat modes
// (ALLOC_MODE_OPTIONS) PLUS a first-class "Default (derived)" entry that CLEARS
// each selected model's contract back to the derivation — the bulk analogue of
// the menu's "↺ Auto — derived" (POST an empty spill {} per selected key, which
// assign_model interprets as clear; see worker_routes._apply_alloc_map: "{}
// clears it"). It is a flat INLINE editor (not the anchored click-popover the
// per-row cell uses) because the bulk bar has no single cell to anchor to.
//
// WIRE (mirrors the per-model path):
//   • Default (derived) → onApply({}, null)                    → spill:{} broadcast
//   • the four zero-knob modes → onApply({alloc_mode:<name>}, null) → broadcast
//   • explicit → onApply(null, perModel) where perModel is {key: absolute-GiB
//     spill}, each key's %-split resolved against THAT model's OWN size (the
//     `spills:` per-model map path — the only honest way to apply a single
//     "% of the model" split across a heterogeneous selection whose members
//     differ in size). GGUF-only members are skipped server-side with a reason.
//
// BULK EXPLICIT FRAMING (deliverable 2): the per-model ExplicitPanel is a
// complementary split OF THE MODEL ("% on GPU", RAM the remainder — never two
// device-denominated percentages). That % framing stays well-defined for a
// heterogeneous SELECTION: the SAME percentage applies per model, resolved to
// each member's own absolute GiB at apply. Absolute-GiB inputs would be
// ambiguous across mixed sizes (one GiB number can't be right for a 4 GiB and a
// 40 GiB model at once), so bulk explicit is PERCENT-ONLY by construction — no
// GiB entry field is offered. The per-model absolutes are shown as a live range
// (min/max across the selection) so the operator sees the honest spread.
function BulkExplicitPanel({ bulkKeys, getModelBytes, onApply, onCancel }) {
  const [gpuPct, setGpuPct] = useState(100)
  const [leniency, setLeniency] = useState('')
  const [priorityDevice, setPriorityDevice] = useState('gpu')
  const ramPct = 100 - gpuPct
  // Per-model own sizes (GiB) for the live absolute-range readout. Only the
  // members whose size the feed actually knows contribute; unknown-size members
  // still POST the %-split intent (no GiB baked) — the honest degrade.
  const sizesGib = (bulkKeys || [])
    .map(k => { const i = getModelBytes ? getModelBytes(k) : null; return i && i.bytes != null ? i.bytes / WP_GIB : null })
    .filter(g => g != null && g > 0)
  const gpuGibRange = (() => {
    if (!sizesGib.length) return null
    const g = sizesGib.map(s => s * gpuPct / 100)
    const lo = Math.min(...g), hi = Math.max(...g)
    return { lo: +lo.toFixed(1), hi: +hi.toFixed(1) }
  })()
  const lenN = (leniency === '' || leniency == null) ? 0 : Math.max(0, Math.min(100, Number(leniency)))
  const floorLine = (() => {
    if (lenN <= 0) return 'floor: exactly the split above (strict)'
    const gpuFloor = priorityDevice === 'gpu' ? Math.max(0, gpuPct - lenN) : Math.min(100, gpuPct + lenN)
    return `floor: ${gpuFloor}% GPU / ${100 - gpuFloor}% RAM`
  })()
  return (
    <div className="wp-allocmode-explicit">
      <div className="wp-allocmode-explicit-head" title={GGUF_ONLY_MODE_TIP}>
        🎛 Explicit — split of EACH selected model (% of-the-model; GGUF members only)
      </div>
      <PlainSlider label="on GPU" min={0} max={100} className="wp-alloc-value"
                   value={String(gpuPct)} onChange={v => setGpuPct(Math.max(0, Math.min(100, Math.round(Number(v) || 0))))}
                   formatValue={v => `${v == null ? 0 : v}% GPU`}
                   title="Share of EACH selected model's own footprint placed on the GPU. The rest goes to host RAM — the two always sum to 100% of that model (resolved against each member's own size at apply, so the GiB differs per model)." />
      <div className="wp-allocmode-split" title="The complementary split, applied per model against its own size.">
        <span className="wp-allocmode-split-gpu">
          {gpuPct}% GPU{gpuGibRange != null
            ? ` · ${gpuGibRange.lo === gpuGibRange.hi ? `${gpuGibRange.lo.toFixed(1)} GiB` : `${gpuGibRange.lo.toFixed(1)}–${gpuGibRange.hi.toFixed(1)} GiB`}`
            : ''}
        </span>
        <span className="wp-allocmode-split-sep">/</span>
        <span className="wp-allocmode-split-ram">{ramPct}% RAM</span>
        <span className="wp-allocmode-split-need">
          {gpuGibRange != null ? 'per model — GiB range across the selection' : 'per model (sizes unknown → % intent only)'}
        </span>
      </div>
      <PlainSlider label="leniency" min={0} max={100} className="wp-alloc-solo"
                   value={leniency} onChange={setLeniency}
                   formatValue={v => v == null ? 'leniency —' : `${v}%`}
                   title="Leniency: up to this percent of each model may shift off its preferred device before that load is refused. 0 = strict." />
      <div className="wp-allocmode-floor" title="The worst-case placement each load will still accept before refusing.">{floorLine}</div>
      <span className="wp-allocmode-prio" title="Priority device: which side each split favors and which way leniency degrades. GPU default.">
        <span className="wp-slider-label">priority</span>
        <button type="button"
                className={`wp-allocmode-prio-btn${priorityDevice === 'gpu' ? ' wp-allocmode-prio-on' : ''}`}
                onClick={() => setPriorityDevice('gpu')} title="Favor the GPU (default)">GPU</button>
        <button type="button"
                className={`wp-allocmode-prio-btn${priorityDevice === 'ram' ? ' wp-allocmode-prio-on' : ''}`}
                onClick={() => setPriorityDevice('ram')} title="Favor host RAM">RAM</button>
      </span>
      <div className="wp-allocmode-explicit-actions">
        <button className="wp-alloc-apply" onClick={() => {
          // Fan the ONE %-split out to a per-model absolute-GiB contract, each
          // resolved against that member's own size. Members with an unknown
          // size still POST the %-split intent (leniency + priority + no GiB
          // targets → the backend falls back to its own defaults, never card
          // capacity — same posture as the single-model ExplicitPanel).
          const perModel = {}
          for (const key of (bulkKeys || [])) {
            const info = getModelBytes ? getModelBytes(key) : null
            const gib = info && info.bytes != null ? info.bytes / WP_GIB : null
            const s = { alloc_mode: 'explicit', priority_device: priorityDevice }
            if (gib != null && gib > 0) {
              s.gpu_mem_gib = +(gib * gpuPct / 100).toFixed(1)
              s.cpu_mem_gib = +(gib * ramPct / 100).toFixed(1)
            }
            if (leniency !== '' && leniency != null) s.leniency_pct = Number(leniency)
            perModel[key] = s
          }
          onApply(null, perModel)
        }}>Apply to {(bulkKeys || []).length}</button>
        <button className="wp-alloc-cancel" onClick={onCancel} title="Cancel">×</button>
      </div>
    </div>
  )
}

// The bulk multi-select alloc editor — parity with AllocModeMenu. A flat inline
// list: "Default (derived)" (clear) on top, then the five ALLOC_MODE_OPTIONS.
// Picking any of the four zero-knob concrete modes applies immediately (broadcast
// {alloc_mode}); picking explicit swaps in BulkExplicitPanel; picking Default
// broadcasts spill {} (clear-to-derived). A mixed selection may include non-GGUF
// members: ``explicit`` is GGUF-only (max-ram was opened for non-GGUF
// 2026-07-24), so it's NOT hard-disabled here (the selection is heterogeneous
// and the backend skips non-GGUF members with an honest reason surfaced by
// setAllocMany) — instead a note warns which modes are GGUF-only, matching the
// "GGUF-only — transformers members skipped" affordance the old control showed.
// onApply(flatSpill, perModel): exactly one is set.
function BulkAllocControl({ count, bulkKeys, getModelBytes, onApply, onCancel }) {
  const [explicitOpen, setExplicitOpen] = useState(false)
  if (explicitOpen) {
    return (
      <div className="wp-alloc wp-bulkalloc">
        <BulkExplicitPanel bulkKeys={bulkKeys} getModelBytes={getModelBytes}
                           onApply={onApply} onCancel={() => setExplicitOpen(false)} />
      </div>
    )
  }
  return (
    <div className="wp-alloc wp-bulkalloc">
      <div className="wp-allocmode-menu wp-bulkalloc-menu" role="menu">
        {/* Parity with AllocModeMenu's "↺ Auto — derived": the top, always-
            available revert. Broadcasts spill {} → each selected model clears its
            pinned contract and tracks the derivation again (the cells then render
            the dimmed/auto style automatically). */}
        <button key="__default__" role="menuitem"
                className="wp-allocmode-opt wp-allocmode-opt-auto"
                title="Default (derived) — clears any pinned contract on every selected model so each tracks its derived default (and improves with it as measured values land). The bulk revert-to-default."
                onClick={() => onApply({}, null)}>
          <span className="wp-allocmode-opt-label">↺ Default (derived)</span>
          <span className="wp-allocmode-opt-desc">— clear overrides; track the derivation</span>
        </button>
        {ALLOC_MODE_OPTIONS.map(([value, label, desc]) => {
          const ggufOnly = !NONGGUF_ALLOC_MODES.has(value)   // max-ram / explicit
          return (
            <button key={value} role="menuitem"
                    className="wp-allocmode-opt"
                    title={value === 'explicit'
                      ? 'opens the explicit-budget knobs (a % split of each model — GGUF members only)'
                      : ggufOnly
                        ? `${desc} — GGUF-only; transformers/comfy members in the selection are skipped with a reason`
                        : `${desc} — applies to every selected model immediately`}
                    onClick={() => {
                      if (value === 'explicit') { setExplicitOpen(true); return }
                      onApply({ alloc_mode: value }, null)
                    }}>
              <span className="wp-allocmode-opt-label">{label}</span>
              <span className="wp-allocmode-opt-desc">— {desc}</span>
              {ggufOnly && <span className="wp-allocmode-opt-mark" title="GGUF-only mode">GGUF</span>}
            </button>
          )
        })}
      </div>
      <div className="wp-bulkalloc-foot">
        <span className="wp-alloc-note" title="explicit is a GGUF-only concept — its banded leniency floor has no transformers analogue. In a mixed selection the backend applies it to the GGUF members and skips the rest with an honest reason. Default / Max GPU / GPU only / RAM only / Max RAM apply to every engine.">
          applies to {count} selected · explicit skips transformers members
        </span>
        {onCancel && <button className="wp-alloc-cancel" onClick={onCancel} title="Cancel">×</button>}
      </div>
    </div>
  )
}

// Per-GPU chip with a used/free VRAM bar.
function GpuChips({ gpus }) {
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
function PidRegistry({ worker, onEvict }) {
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
function measuredFootprint(a) {
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
function isMeasuredResident(a) {
  if (!a) return false
  // A worker build that can tell says so outright: `materialized: false` means a
  // runner object exists but no weights were ever loaded through it. That is a
  // definitive NO and outranks everything else. null/absent = an older worker
  // that cannot tell, which must behave exactly as it always did — no regression.
  if (a.materialized === false) return false
  return measuredFootprint(a)
}

function residentState(a) {
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

// One resource track: a used fill + an optional right-aligned striped reserve
// segment (held out of the pool, not allocatable). The caller only renders this
// when total > 0, so the width math never goes NaN.
function BudgetTrack({ label, total, used, reserve, reserveGib, note }) {
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

function WorkerBudgetBar({ worker }) {
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

// ── Per-worker local STORAGE: what model cache this box holds ─────────────────
// The always-on monitoring depiction (sibling to the VRAM/RAM WorkerBudgetBar):
// cache_used vs budget, a per-model cached-files list (size + last-served +
// protection), and — only when the worker is over budget — a proposal-only
// eviction review. Everything here is derived server-side in storage_proposal;
// this component renders `worker.storage` and computes nothing. Nothing deletes
// without the explicit Approve click (human-in-the-loop, central-approved).
function WorkerStorageBar({ worker, onApproveEvictions, sizeByKey, detailsOnly = false }) {
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
  // now sends an explicit resident figure (gauge_used_bytes / resident_bytes);
  // fall back to cache_used_bytes for a pre-2026-07-17 central.
  const used = (s.gauge_used_bytes != null ? s.gauge_used_bytes
               : (s.resident_bytes != null ? s.resident_bytes
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
const ESTIMATED_VRAM_TITLE =
  'estimated from declared placement × file size — not measured'
function vramBytesFor(a, worker, sizeByKey) {
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

// One resident row (a model occupying VRAM or host RAM). Shared by the VRAM and
// RAM resource details; engine-agnostic via residentState.
function ResidentList({ items, loadedSet, worker, resource, emptyLabel, sizeByKey }) {
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
function VramReconcile({ worker, gpuRes, comfy, sizeByKey }) {
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
function ResourceChip({ icon, name, used, total, count, active, onClick, disabled, physicalTotal, bar, extraTerm }) {
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
function ResourceStrip({ worker, models, onApproveEvictions, onEvict }) {
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
  const stUsed = s.reported ? (s.cache_used_bytes || 0) : null
  const stTotal = s.reported ? (s.budget ?? null) : null

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

function SpillBadge({ spill }) {
  if (!spill || !spill.mode) return null
  const free = spill.free_vram_bytes
  const label = spill.mode === 'auto' ? 'autofit' : spill.mode
  return (
    <span className="wp-spill" title="GPU/CPU split mode reported by the worker">
      spill: {label}
      {free != null && <em> · {fmtBytes(free)} VRAM free</em>}
    </span>
  )
}

// Per-worker "load a model" as a SORTABLE, MULTI-SELECT table. Columns assort
// (click a header to sort); a checkbox column allocates a GROUP of models to
// THIS worker in one action. Anti-duplicate: a model already allocated to
// ANOTHER worker is locked ("on <worker>") to prevent accidental dual
// allocation — unless the operator flips the ⚡ breaker to deliberately
// replicate it (e.g. for scene fan-out across the fleet).
function WorkerLoadTable({ models, allocation, workerId, worker, onAllocate, onCancel }) {
  const keyOf = (m) => m.model_key ?? m.key
  // TRUE per-model size: the effective serving quant for GGUF (size_bytes ||
  // effective_bytes from the /models feed), the on-disk footprint otherwise —
  // never the all-quants dir sum. Same rule the ModelPicker uses.
  const sizeOf = (m) => {
    const s = m.size_bytes != null ? m.size_bytes : m.effective_bytes
    return s != null ? Number(s) : null
  }
  const isGgufModel = (m) => {
    const fw = String(m.framework || '').toLowerCase()
    return fw === 'gguf' || fw === 'llama_cpp'
  }
  const [sel, setSel] = useState(() => new Set())
  const [sortKey, setSortKey] = useSessionState('hugpy.sess.wp.load.sort', 'name')
  const [sortDir, setSortDir] = useSessionState('hugpy.sess.wp.load.dir', 'asc')
  const [taskFilter, setTaskFilter] = useSessionState('hugpy.sess.wp.load.task', '')
  const [breaker, setBreaker] = useState(false)
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  // Central-provisioning readiness (AUTHORITATIVE — same check the loader
  // enforces; the manifest `status` is NOT reliable): a map keyed by model_key
  // → {state:'ready'|'incomplete'|'absent'|'error', reason}. A key absent from
  // the map = not in central's manifest → treat as 'absent'. Live download jobs
  // (from /api/jobs) overlay a `downloading` state with progress on top.
  const [provMap, setProvMap] = useState({})
  const [jobs, setJobs] = useState([])
  const [dlBusy, setDlBusy] = useState(() => new Set())   // model_keys with a POST in flight
  const [dlErr, setDlErr]   = useState({})                // model_key → last download error message
  const aliveRef = useRef(true)                            // guards async setState after unmount
  const prevActiveRef = useRef(new Set())                  // last poll's active (queued/running) job keys

  // Where else each model already lives (excludes THIS worker).
  const elsewhereOf = useCallback((m) => {
    const holders = allocation[keyOf(m)] || []
    return holders.filter(h => h.id !== workerId)
  }, [allocation, workerId])

  // Pull the authoritative central-readiness map. Cheap + idempotent; polled
  // slowly (a finished download only needs to flip to selectable eventually)
  // and also kicked whenever a job settles.
  const refetchProv = useCallback(async () => {
    try {
      const d = await fetchJson('/api/llm/central-provisioning')
      if (aliveRef.current) setProvMap(d && typeof d === 'object' ? d : {})
    } catch { /* readiness is advisory; a hiccup shouldn't break the picker */ }
  }, [])

  // Poll the download-job feed fast so ⏳ progress ticks live. When a job that
  // was active last poll is no longer active, a pull just settled → refresh the
  // readiness map so a completed model becomes selectable without reopening.
  const refetchJobs = useCallback(async () => {
    try {
      const d = await fetchJson('/api/jobs')
      if (!aliveRef.current) return
      const arr = Array.isArray(d) ? d : []
      setJobs(arr)
      const active = new Set(
        arr.filter(j => j && (j.status === 'queued' || j.status === 'running')).map(j => j.model_key)
      )
      let settled = false
      prevActiveRef.current.forEach(k => { if (!active.has(k)) settled = true })
      prevActiveRef.current = active
      if (settled) refetchProv()
    } catch { /* transient; next tick retries */ }
  }, [refetchProv])

  useEffect(() => {
    aliveRef.current = true
    // Self-scheduling for the same reason as the roster poll below: setInterval
    // fires whether or not the previous call returned, so a slow endpoint
    // MULTIPLIES its own load instead of backing off. /llm/jobs was measured at
    // 3-8s against this 2500ms tick — three-plus copies in flight, each holding
    // one of central's 24 slots, all of them competing with the roster poll
    // that actually renders the view. At most one of each is in flight now.
    let timers = []
    const every = (fn, ms) => {
      const tick = () => {
        Promise.resolve(fn()).finally(() => {
          if (aliveRef.current) timers.push(setTimeout(tick, ms))
        })
      }
      tick()
    }
    every(refetchJobs, 2500)    // fast: live download progress
    every(refetchProv, 10000)   // slow: readiness backstop
    return () => {
      aliveRef.current = false
      timers.forEach(clearTimeout)
    }
  }, [refetchProv, refetchJobs])

  // The active download job (queued/running) for a model, if any.
  const activeJobFor = useCallback((key) =>
    jobs.find(j => j && j.model_key === key && (j.status === 'queued' || j.status === 'running')) || null,
  [jobs])

  // Per-model provisioning state: a live download wins; else the readiness map;
  // else (key not in the manifest) → absent.
  const provStateOf = useCallback((m) => {
    const key = keyOf(m)
    const job = activeJobFor(key)
    if (job) return { state: 'downloading', job, reason: null }
    const p = provMap[key]
    if (!p) return { state: 'absent', reason: null }
    return { state: p.state || 'absent', reason: p.reason ?? null }
  }, [provMap, activeJobFor])

  // Kick (or resume) the central pull for a model, then optimistically refetch
  // jobs so the row flips to ⏳ immediately. Errors surface inline on the row.
  const download = async (key) => {
    setDlBusy(prev => new Set(prev).add(key))
    setDlErr(prev => { const n = { ...prev }; delete n[key]; return n })
    try {
      await fetchJson(`/api/models/${encodeURIComponent(key)}/download`, { method: 'POST' })
      await refetchJobs()
    } catch (e) {
      if (aliveRef.current) setDlErr(prev => ({ ...prev, [key]: e.message || 'download failed' }))
    } finally {
      if (aliveRef.current) setDlBusy(prev => { const n = new Set(prev); n.delete(key); return n })
    }
  }

  // Union of EVERY task across the pickable models (ModelTable's helper, so
  // multi-task models surface under each of their tasks — the same fix that
  // cured ModelTable's primary-task-only skew).
  const tasks = useMemo(() => [...new Set(models.flatMap(modelTasks))].sort(), [models])

  const COLUMNS = [
    { key: 'name', label: 'Model', get: m => m.name || keyOf(m) },
    // Central-provisioning readiness — sortable by state string. Sort is
    // point-in-time (provMap/jobs are intentionally NOT in `filtered` deps, so
    // the fast job poll doesn't reshuffle rows under the operator); the pills
    // themselves still update live because the row cells read provStateOf on
    // every render.
    { key: 'central', label: 'Central', get: m => provStateOf(m).state },
    { key: 'task', label: 'Task', get: m => modelTask(m) || '—' },
    { key: 'framework', label: 'Engine', get: m => m.framework || '—' },
    // Size sorts by true bytes; unknown (-1) sinks to the bottom in ascending sort.
    { key: 'size', label: 'Size', get: m => { const b = sizeOf(m); return b == null ? -1 : b }, num: true },
    { key: 'ctx', label: 'Ctx', get: m => m.model_max_length || 0, num: true },
    { key: 'elsewhere', label: 'Allocated on', get: m => elsewhereOf(m).map(h => h.name).join(', ') },
  ]

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    let rows = models
    if (needle) rows = rows.filter(m => (m.name || keyOf(m)).toLowerCase().includes(needle))
    // Full-list task match — a model counts under ANY task it advertises.
    if (taskFilter) rows = rows.filter(m => modelTasks(m).includes(taskFilter))
    const col = COLUMNS.find(c => c.key === sortKey) || COLUMNS[0]
    const dir = sortDir === 'asc' ? 1 : -1
    return [...rows].sort((a, b) => {
      const va = col.get(a), vb = col.get(b)
      if (col.num) return (Number(va) - Number(vb)) * dir
      return String(va).localeCompare(String(vb)) * dir
    })
  }, [models, q, taskFilter, sortKey, sortDir, allocation, workerId])

  // Selectable only when central fully holds the model (ready) AND the
  // anti-duplicate breaker permits it. A not-ready model's checkbox is disabled;
  // its row offers a Download instead — so provisioning refusals disappear
  // before allocation ever happens.
  const isEligible = useCallback((m) => provStateOf(m).state === 'ready' && (breaker || elsewhereOf(m).length === 0),
    [provStateOf, breaker, elsewhereOf])
  const eligibleKeys = useMemo(() => filtered.filter(isEligible).map(keyOf), [filtered, isEligible])
  const allSel = eligibleKeys.length > 0 && eligibleKeys.every(k => sel.has(k))

  const toggle = (k) => setSel(prev => { const n = new Set(prev); n.has(k) ? n.delete(k) : n.add(k); return n })
  const toggleAll = () => setSel(allSel ? new Set() : new Set(eligibleKeys))
  const toggleSort = (k) => {
    if (sortKey === k) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('asc') }
  }
  const arrow = (k) => sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''

  const allocate = async () => {
    const keys = [...sel]
    if (keys.length === 0) return
    setBusy(true)
    try { await onAllocate(keys, breaker); setSel(new Set()) }
    finally { setBusy(false) }
  }

  return (
    <div className="wp-loadtable">
      <div className="wp-loadtable-bar">
        <input className="wp-loadtable-q" placeholder="filter models…" value={q}
               onChange={e => setQ(e.target.value)} autoFocus />
        <select className="wp-loadtable-task" value={taskFilter}
                onChange={e => setTaskFilter(e.target.value)}
                title="Filter by task — matches ANY task a model advertises, not just its primary">
          <option value="">All tasks</option>
          {tasks.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <label className="wp-breaker" title="Anti-duplicate is ON by default: models already on another worker are locked. Flip this to deliberately allocate a duplicate (replicate across workers — e.g. scene fan-out).">
          <input type="checkbox" checked={breaker} onChange={e => { setBreaker(e.target.checked); setSel(new Set()) }} />
          ⚡ allow duplicate allocation
        </label>
        <button className="wp-load-cancel" title="Cancel" onClick={onCancel}>×</button>
      </div>
      <div className="wp-loadtable-scroll">
        <table className="wp-loadtable-t">
          <thead>
            <tr>
              <th className="wp-lt-check">
                <input type="checkbox" checked={allSel} onChange={toggleAll}
                       disabled={eligibleKeys.length === 0} title="Select all eligible" />
              </th>
              {COLUMNS.map(c => (
                <th key={c.key} className={`wp-lt-sortable${c.num ? ' wp-lt-num' : ''}`}
                    onClick={() => toggleSort(c.key)}>{c.label}{arrow(c.key)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={COLUMNS.length + 1} className="wp-none">No models to allocate.</td></tr>
            )}
            {filtered.map(m => {
              const k = keyOf(m)
              const elsewhere = elsewhereOf(m)
              const locked = elsewhere.length > 0 && !breaker
              const prov = provStateOf(m)
              return (
                // Only the anti-duplicate lock grays the row; not-ready rows stay
                // legible so their Download button is usable.
                <tr key={k} className={locked ? 'wp-lt-locked' : ''}>
                  <td className="wp-lt-check">
                    <input type="checkbox" checked={sel.has(k)} disabled={!isEligible(m)}
                           onChange={() => toggle(k)} />
                  </td>
                  <td>{m.name || k}</td>
                  {/* Central-provisioning readiness pill (+ Download when the
                      model needs pulling; ⏳ progress while a pull runs). This is
                      what makes post-hoc "central doesn't have X on disk"
                      refusals visible up front. */}
                  <td className="wp-cprov-cell">
                    {prov.state === 'ready' && (
                      <span className="wp-state-pill wp-cprov-ready"
                            title="fully on central disk — allocatable">✓ on disk</span>
                    )}
                    {prov.state === 'downloading' && (() => {
                      const job = prov.job || {}
                      const pct = (job.total_bytes && job.total_bytes > 0 && job.progress != null)
                        ? Math.round(job.progress * 100) : null
                      return (
                        <span className="wp-state-pill wp-cprov-downloading"
                              title={job.total_bytes
                                ? `downloading to central — ${fmtBytes(job.downloaded_bytes)} / ${fmtBytes(job.total_bytes)}`
                                : 'downloading to central…'}>
                          ⏳ {pct != null ? `${pct}%` : '…'}
                        </span>
                      )
                    })()}
                    {prov.state === 'incomplete' && (
                      <>
                        <span className="wp-state-pill wp-cprov-incomplete"
                              title="directory present but files incomplete — download to finish">◐ incomplete</span>
                        <button className="wp-cprov-dl" disabled={dlBusy.has(k)} onClick={() => download(k)}
                                title="Finish downloading this model to central disk">
                          {dlBusy.has(k) ? '…' : '⬇ Download'}
                        </button>
                      </>
                    )}
                    {prov.state === 'absent' && (
                      <>
                        <span className="wp-state-pill wp-cprov-absent"
                              title="not on central disk / not in the manifest">○ not on central</span>
                        <button className="wp-cprov-dl" disabled={dlBusy.has(k)} onClick={() => download(k)}
                                title="Download this model to central disk">
                          {dlBusy.has(k) ? '…' : '⬇ Download'}
                        </button>
                      </>
                    )}
                    {prov.state === 'error' && (
                      <span className="wp-state-pill wp-cprov-error"
                            title={prov.reason || 'central could not resolve this model'}>⚠ error</span>
                    )}
                    {dlErr[k] && <span className="wp-cprov-err" title={dlErr[k]}>failed</span>}
                  </td>
                  {/* first task + "+N" with the full list on hover — same
                      rendering ModelTable uses for multi-task models */}
                  <td className="wp-lt-muted" title={modelTasks(m).join(', ')}>
                    {modelTask(m)}{modelTasks(m).length > 1 ? ` +${modelTasks(m).length - 1}` : ''}
                  </td>
                  <td className="wp-lt-muted">{m.framework || '—'}</td>
                  <td className="wp-lt-num wp-lt-size"
                      title={sizeOf(m) == null ? 'size unknown — not on disk / not reported by the feed'
                        : isGgufModel(m)
                          ? `effective quant${m.effective_gguf ? ` ${m.effective_gguf}` : ''} — ${fmtBytes(sizeOf(m))} (the ONE quant that serves, not the all-quants dir sum)`
                          : `${fmtBytes(sizeOf(m))} on disk`}>
                    {sizeOf(m) == null ? '—' : fmtBytes(sizeOf(m))}
                  </td>
                  <td className="wp-lt-num">{m.model_max_length || '—'}</td>
                  <td className={elsewhere.length ? (breaker ? 'wp-lt-dup' : 'wp-lt-muted') : 'wp-lt-muted'}>
                    {elsewhere.length ? `${breaker ? '⚡ ' : ''}${elsewhere.map(h => h.name).join(', ')}` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {/* Measured headroom on THIS worker — the honest free space the fit-guard
          checks each pick against. VRAM free is nvidia-smi; RAM free is the
          worker's measured free (reserve-adjusted). "resident" is the SUM of the
          worker's MEASURED per-model VRAM allocations, not file sizes or envelope
          guesses. Per-model RAM RSS for in-process models isn't attributable yet
          (glibc-arena + page-cache gap, see CODE_GAPS), so RAM is worker-level. */}
      {worker && (worker.vram_total != null || worker.ram_total != null) && (() => {
        const allocs2 = Array.isArray(worker.allocations) ? worker.allocations.filter(a => a && a.model_key) : []
        const residentVram = allocs2.reduce((s, a) => s + (a.vram_bytes != null && a.vram_bytes > 0 ? a.vram_bytes : 0), 0)
        const selBytes = [...sel].reduce((s, k) => {
          const m = models.find(mm => keyOf(mm) === k)
          return s + (m ? (sizeOf(m) || 0) : 0)
        }, 0)
        return (
          <div className="wp-load-headroom"
               title="Measured headroom on this worker — what a model is fit-checked against before it loads. VRAM free is nvidia-smi; RAM free is the worker's measured free (reserve-adjusted). 'resident' sums measured per-model VRAM, never file sizes. Per-model RAM RSS for in-process models is not attributable yet (page-cache/arena gap), so RAM is shown at the worker level.">
            <span className="wp-load-headroom-label">Headroom (measured):</span>
            {worker.vram_total != null && (
              <span className="wp-load-headroom-item" title="Free VRAM from nvidia-smi (measured)">
                🖥 {fmtBytes(worker.vram_free)} free / {fmtBytes(worker.vram_total)} VRAM
                {residentVram > 0 && <em> · {fmtBytes(residentVram)} resident</em>}
              </span>
            )}
            {worker.ram_total != null && (
              <span className="wp-load-headroom-item" title="Worker's measured free RAM (reserve-adjusted). Per-model RAM attribution is not available yet.">
                🧠 {fmtBytes(worker.free_ram)} free / {fmtBytes(worker.ram_total)} RAM
              </span>
            )}
            {sel.size > 0 && selBytes > 0 && (
              <span className="wp-load-headroom-sel" title="Total on-disk size of the checked models (effective quant for GGUF). Resident memory use is typically at or below this; the fit-guard makes the final call per model.">
                selected {sel.size}: {fmtBytes(selBytes)} on disk
              </span>
            )}
          </div>
        )
      })()}
      <div className="wp-loadtable-actions">
        <span className="wp-load-hint" title="Each model is checked against this worker's free VRAM + RAM + disk before it loads — a model that won't fit is refused, not OOM'd">✓ fit-guarded</span>
        <button className="wp-loadtable-go" disabled={busy || sel.size === 0} onClick={allocate}>
          {busy ? 'Allocating…' : `Allocate ${sel.size} model${sel.size === 1 ? '' : 's'} to this worker`}
        </button>
      </div>
    </div>
  )
}

// A worker row: status + GPUs (with used/free) + provisioning state, the models
// it serves with per-model load state + concise GPU allocation + free controls.
function WorkerRow({ worker, models, allocation, onAssign, onLoad, onUnassign, onRemove, onFree, onFreeAll, onFreeRam, onRestart, onUpdate = null, onAdmit, onBlock, onSetPool, onSetLimits, onSetConfig, onSetResidency, onSetResidencyMany, onSetAllocMany, onTogglePin, onPinAll, onUnpinAll, onReap, onApproveEvictions, onEvict, onAllocateMany, onRefresh = null, applying = false, restarting = false, updating = false, blockedKeys = null, onToggleBlock = null }) {
  const [pick, setPick]       = useState('')
  const [newSpill, setNewSpill] = useState({})   // allocation for the next assign
  const [allocMenu, setAllocMenu] = useState(null)   // model key whose in-place alloc menu is open
  const allocAnchorRef = useRef(null)                // the open menu's trigger button (for fixed positioning)
  // Optimistic per-model alloc override: key -> spill dict, applied on top of
  // the derived mode so the cell updates the instant a mode is picked; reverted
  // (deleted) if the /assign POST rejects, and cleared once the refetch lands.
  const [allocOptimistic, setAllocOptimistic] = useState({})
  // Optimistic bitsandbytes toggles: key -> bool, applied on top of the worker
  // payload so the checkbox flips instantly while the POST + refetch land. The
  // Alloc cell beside it re-derives from the SERVER's answer, so the two settle
  // together on the next workers refresh.
  const [bnbOptimistic, setBnbOptimistic] = useState({})
  // The models on THIS worker that can actually take the specialization. Drives
  // both the bulk buttons' visibility and their count, so the label never
  // promises to change models it will skip.
  const bnbEligibleKeys = useMemo(
    () => Object.keys(worker.bnb_available || {}),
    [worker.bnb_available])

  // Bulk apply/clear. Sequential rather than Promise.all: each POST is a
  // registry write on the same worker record, and firing 44 concurrent writes
  // at one file-locked store is how you get lost updates. 44 small writes take
  // ~a second and the optimistic ticks make it feel instant anyway.
  const [moeOptimistic, setMoeOptimistic] = useState({})
  const setMoe = useCallback(async (modelKey, value) => {
    setMoeOptimistic(o => ({ ...o, [modelKey]: value === null ? undefined : value }))
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/moe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey, value }),
      })
    } catch (e) {
      setMoeOptimistic(o => { const n = { ...o }; delete n[modelKey]; return n })
      alert(`MoE split failed: ${e.message}`)
    }
    // Refetch: the Alloc column re-derives off this (forcing the split off drops
    // coder-next from explicit to max-ram), so the two must settle together.
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, onRefresh])

  const moeCapableKeys = useMemo(
    () => Object.keys(worker.moe_capable || {}), [worker.moe_capable])

  const setMoeMany = useCallback(async (value) => {
    if (!moeCapableKeys.length) return
    try {
      // One server-side sweep over the capable set — same reasoning as the 4-bit
      // bulk: the client's payload can lag a just-applied change.
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/moe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true, value }),
      })
    } catch (e) {
      alert(`MoE bulk failed: ${e.message}`)
    }
    setMoeOptimistic({})
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, moeCapableKeys, onRefresh])

  const setBnbMany = useCallback(async (enabled) => {
    const keys = Object.keys(worker.bnb_available || {})
    if (!keys.length) return
    if (enabled && !confirm(
      `Load ${keys.length} model(s) on ${worker.name || 'this worker'} at 4-bit `
      + '(bitsandbytes nf4)?\n\nEach is re-priced at ~30% of its fp16 size, so '
      + 'their allocations re-derive — several may move from RAM onto the GPU. '
      + 'Quantization costs some output quality.')) return
    setBnbOptimistic(o => {
      const n = { ...o }
      for (const k of keys) n[k] = enabled
      return n
    })
    try {
      // ONE server-side sweep, not N client POSTs. The server resolves the
      // eligible set itself, so the result cannot depend on how stale this
      // render's worker payload is — the bug that left 19 of 44 rows enabled
      // while the UI showed none ticked.
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/bnb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true, enabled }),
      })
    } catch (e) {
      setBnbOptimistic({})
      alert(`4-bit bulk failed: ${e.message}`)
    }
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, worker.name, worker.bnb_available, onRefresh])

  const setBnb = useCallback(async (modelKey, enabled) => {
    setBnbOptimistic(o => ({ ...o, [modelKey]: enabled }))
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/bnb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey, enabled }),
      })
      // Refetch so the Alloc column picks up the RE-DERIVED mode; the optimistic
      // entry is dropped once the authoritative payload carries the new value.
      if (typeof onRefresh === 'function') onRefresh()
    } catch (e) {
      setBnbOptimistic(o => { const n = { ...o }; delete n[modelKey]; return n })
      alert(`Specialization failed: ${e.message}`)
    }
  }, [worker.id, onRefresh])
  // Serving-detail cache (rulings 1+2, 2026-07-24): the per-(model,worker)
  // FEASIBLE mode set and the feasibility-DERIVED default live ONLY on
  // /llm/serving/<key> (alloc_by_worker / alloc_mode_derived), NOT on the
  // /llm/workers list this panel polls. Rather than fetch that for every serving
  // row (a per-row request storm on a worker with 100+ models), we fetch it
  // LAZILY — once, the first time a model's alloc menu is opened — and cache it
  // by model_key. The menu is the only place feasibility/derived_default are
  // needed; a closed row shows the derived-vs-pinned distinction from the local
  // override alone (see the alloc cell), no fetch required. Cache survives the
  // 10s poll (it's keyed by model_key, orthogonal to the worker refresh).
  const [servDetail, setServDetail] = useState({})   // key -> serving-GET row (or {error})
  const servFetchedRef = useRef(new Set())           // keys already fetched (dedupe)
  const fetchServDetail = useCallback((key) => {
    if (servFetchedRef.current.has(key)) return       // fetch each key at most once
    servFetchedRef.current.add(key)
    setServDetail(prev => ({ ...prev, [key]: { loading: true } }))
    fetchJson(`/api/llm/serving/${encodeURIComponent(key)}`)
      .then(row => setServDetail(prev => ({ ...prev, [key]: row || {} })))
      .catch(e => {
        // On failure, forget the key so a later open retries — feasibility is a
        // fail-OPEN read (missing data offers every mode), never a hard error.
        servFetchedRef.current.delete(key)
        setServDetail(prev => ({ ...prev, [key]: { error: e.message } }))
      })
  }, [])
  const [resMenu, setResMenu] = useState(null)   // model key whose residency picker is open
  const [limitsOpen, setLimitsOpen] = useState(false)
  const [limitsForm, setLimitsForm] = useState({ ram_max_gib: '', gpu_mem_gib: '', disk_cache_gib: '', threads: '' })
  const [ping, setPing]       = useState(null)   // null | 'checking' | {reachable, error}
  const [showLoad, setShowLoad] = useState(false) // reveal the load-a-model editor
  const [activating, setActivating] = useState(null) // model key being activated (loaded now)
  // Serving TABLE controls — a comprehensive, sortable view of the models
  // designated to THIS worker (same idiom as the "load a model" table). Search
  // and task-filter are per-mount (ephemeral, reset on unmount); the SORT is
  // session-sticky so the operator's chosen ordering survives a re-render.
  const [servQ, setServQ] = useState('')          // text search: model name / key
  const [servTask, setServTask] = useState('')    // task dropdown ('' = all tasks)
  const [servSort, setServSort] = useSessionState('hugpy.sess.wp.serving.sort', 'name')
  const [servDir, setServDir]   = useSessionState('hugpy.sess.wp.serving.dir', 'asc')
  // Multi-SELECT for bulk residency (todo t12): a Set of selected model_keys,
  // per-worker (this component instance IS one worker). Kept in local state
  // keyed by model_key, so it survives the panel's 10s load() refresh untouched
  // (the poll replaces the `worker` prop's data, never this selection); a model
  // that's since been unassigned is pruned lazily at apply time (the backend
  // also drops off-worker keys). Ephemeral: resets on unmount, like servQ.
  const [servSel, setServSel] = useState(() => new Set())
  const toggleSel = useCallback((key) => {
    setServSel(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }, [])
  const clearSel = useCallback(() => setServSel(new Set()), [])
  // Bulk-alloc editor (todo t15): the inline AllocControl expansion in the bulk
  // bar is open/closed by this flag (mirrors the per-row `editing` toggle).
  const [bulkAllocOpen, setBulkAllocOpen] = useState(false)

  // In-place alloc apply (operator ask 2026-07-24): a picked mode POSTs the
  // {alloc_mode, …} spill via the SAME /assign path the old editor used
  // (onAssign). Optimistic: stamp the spill locally so the cell flips instantly,
  // then reconcile — on success the refetch's spill_by_model becomes the truth
  // and we drop the optimistic entry; on error we drop it too (revert to the
  // unchanged server value; onAssign already surfaced the reason).
  const applyAllocMode = useCallback((key, spill) => {
    setAllocMenu(null)
    setAllocOptimistic(prev => ({ ...prev, [key]: spill }))
    Promise.resolve(onAssign(worker, key, spill))
      .finally(() => setAllocOptimistic(prev => {
        const next = { ...prev }; delete next[key]; return next
      }))
  }, [onAssign, worker])

  const checkHealth = useCallback(async () => {
    setPing('checking')
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/health`)
      setPing(r)
    } catch (e) {
      setPing({ reachable: false, error: e.message })
    }
  }, [worker.id])

  const assignable = useMemo(() => {
    const assigned = new Set(worker.models || [])
    return models.filter(m => !assigned.has(m.model_key ?? m.key))
  }, [models, worker.models])

  const nameFor = useCallback(
    key => models.find(m => (m.model_key ?? m.key) === key)?.name || key,
    [models],
  )

  // A model's TRUE on-disk size from the /models feed: for GGUF the ONE effective
  // quant that serves (size_bytes/effective_bytes), never the all-quants dir sum.
  // Used to label serving rows with the real footprint instead of loaded_detail's
  // dir bytes (which double-counts sibling quants for GGUF).
  const sizeInfo = useCallback((key) => {
    const m = models.find(mm => (mm.model_key ?? mm.key) === key)
    if (!m) return null
    const fw = String(m.framework || '').toLowerCase()
    const eff = m.size_bytes != null ? m.size_bytes : m.effective_bytes
    return {
      bytes: eff != null ? Number(eff) : null,
      isGguf: fw === 'gguf' || fw === 'llama_cpp',
      effGguf: m.effective_gguf,
    }
  }, [models])

  const loaded      = new Set(worker.loaded_models || [])
  const provisioning = new Set(worker.provisioning || [])
  const heating     = new Set(worker.loading || [])   // weights load in flight
  // Unified, engine-agnostic allocation view (new agents): {kind:"slot"|"ram"}
  // per resident model. When present it is the source of truth for a model's
  // residency KIND — a slot occupant (GGUF seat) and an in-RAM transformers
  // resident are reported the same way. When ABSENT (older agents) we fall
  // back to the legacy slots+loaded_models derivation below.
  const allocByKey = Array.isArray(worker.allocations)
    ? Object.fromEntries(worker.allocations.filter(a => a && a.model_key).map(a => [a.model_key, a]))
    : null
  // Residency KIND (legacy fallback): loaded_models is a merge of in-process
  // residents and slot-hosted models — the slots list tells them apart.
  // Slot-hosted = "serving" (routable supervised child); in-process =
  // "loaded" (resident in the agent itself, not in a slot).
  const slotServed  = new Set((worker.slots || [])
    .filter(s => s && s.model_key && s.healthy).map(s => s.model_key))
  // Live attribution: models whose slot is mid-request RIGHT NOW.
  const slotBusy    = new Set((worker.slots || [])
    .filter(s => s && s.model_key && s.busy).map(s => s.model_key))
  // UTIL-08 disk-truth: null until the worker reports it (older agents).
  const localSet = worker.models_local ? new Set(worker.models_local) : null

  // Per-model live-state derivation for THIS worker. Lifted verbatim out of the
  // Serving row render so the serving TABLE can BOTH sort by the derived state
  // (severity order) and render the same pill — the attribution ladder below is
  // byte-for-byte the operator-vetted logic it always was, only relocated.
  const deriveModelState = (key) => {
    const isPulling = provisioning.has(key)
    const isHeating = heating.has(key)
    const override = worker.spill_by_model?.[key]
    // The BACKEND's derived default for this (worker, model). Without it the
    // alloc cell fell back to deriveAllocMode's hardcoded 'max-gpu' for every
    // model with no persisted contract — 62 of ae's 64 — so a 67 GiB
    // transformers model that the operator's decision tree correctly resolves
    // to ram-only still displayed "⚡ Max GPU · auto". The tree was right and
    // already shipped; its answer simply never reached this cell.
    const derivedMode = worker.model_alloc_modes?.[key] || null
    // 📌 pin = PERMANENT attribution to this worker — blocks unassign.
    const isPinned = !!worker.config?.pinned?.[key]
    // Full attribution ladder (each stage reported by the worker):
    //   pulling n%  — files downloading from central/HF (live progress)
    //   heating     — weights loading into VRAM/RAM right now
    //   serving     — hosted in a SLOT (routable supervised child)
    //   loaded      — resident IN-PROCESS on this machine (no slot)
    //   cold        — assigned; loads on first request or next warm pass
    // Residency KIND, engine-agnostic: prefer the unified allocations
    // view (a slot occupant OR an in-RAM transformers resident both count
    // as "resident"); fall back to the legacy slots+loaded_models sets.
    const alloc = allocByKey ? allocByKey[key] : undefined
    let inSlot, isAnswering, isServing
    // isIdleResident: a ram allocation EXISTS but the worker reports NO
    // measured footprint for it (no VRAM, no RSS, no cuda/cpu device, not
    // served recently). That is a hollow runner-cache entry — the exact
    // thing a reconcile /probe leaves behind, and the flap the operator saw:
    // it must read as a distinct, subtle idle state, NEVER as purple loaded.
    let isIdleResident = false
    if (allocByKey) {
      inSlot = !!alloc && alloc.kind === 'slot' && !!alloc.healthy
      isAnswering = inSlot && !!alloc.busy
      const ramResident = !!alloc && alloc.kind === 'ram' && isMeasuredResident(alloc)
      // "Loaded" derives from MEASURED residency, not from the mere presence
      // of a ram allocation (its runner-cache membership) — see the flap note.
      isServing = inSlot || ramResident
      isIdleResident = !!alloc && alloc.kind === 'ram' && !ramResident
    } else {
      isServing = loaded.has(key)
      inSlot = slotServed.has(key)
      isAnswering = inSlot && slotBusy.has(key)
    }
    const prog = worker.provision_progress?.[key]
    const pct = prog && prog.total_bytes > 0
      ? Math.min(99, Math.round(100 * (prog.done_bytes ?? prog.frac * prog.total_bytes) / prog.total_bytes))
      : null
    // Assigned but files ABSENT on the worker. Under the LAZY-DOWNLOAD
    // doctrine (7f0e6e8 + 2a3baeb) assignment/pin is ATTRIBUTION, not a
    // transfer order — so this is the model's CORRECT RESTING STATE, not
    // drift: it waits here until called, then downloads on first use.
    //
    // `isPulling` is now live-only (central gates it on owner-alive AND
    // bytes-moving), so a dead/queued entry correctly FALLS THROUGH to this
    // state instead of masking it as a phantom ⏳ pulling forever.
    const isMissing = localSet != null && !localSet.has(key)
      && !isPulling && !isHeating && !isServing && !isIdleResident
    // Order: an idle-resident (hollow) ram alloc is NOT missing and NOT
    // loaded — it sits between serving and cold as its own honest state.
    const state = isPulling ? 'pulling' : isHeating ? 'heating'
      : isServing ? (inSlot ? (isAnswering ? 'answering' : 'serving') : 'loaded')
      : isIdleResident ? 'idle'
      : isMissing ? 'missing' : 'cold'
    const stateTitle = isPulling ? `downloading files from central/HF${pct != null ? ` — ${pct}%` : ''}`
      : isHeating ? 'weights loading into VRAM/RAM right now'
      : isServing ? (inSlot
        ? (isAnswering
          ? 'actively processing a request right now'
          : 'hosted in a slot on this worker — routable, crash-isolated server child')
        : 'resident in this worker\'s own process — dedicated to this machine, not in a slot')
      : isIdleResident ? 'a runner is cached on this worker but holds NO measured VRAM/RAM and hasn\'t served recently — not actually resident (a reconcile warm may have just instantiated it, or its weights were freed). Its measured residency is the truth here, not the runner-cache membership.'
      : isMissing ? 'assigned but NOT YET DOWNLOADED — this is the normal resting state, not an error. Assignment attributes the model to this worker; the weights transfer on the FIRST CALL (lazy download). Nothing is transferring right now.'
      : 'assigned — loads on first request or the next warm pass'
    // load_reports[key] is the recorded outcome of the LAST warm/probe attempt
    // central made on this (worker, model) — additive, central-side. We annotate
    // it ONLY where the model is NOT resident right now (cold/missing/idle/
    // heating), so a "▶ activate did nothing" always carries a visible why; a
    // live serving/answering/loaded row needs none. The pill stays DERIVED from
    // measured residency (doctrine above) — this is an annotation, not a state.
    const loadReport = worker.load_reports?.[key]
    const showLoadWhy = !!loadReport
      && (state === 'cold' || state === 'missing' || state === 'idle' || state === 'heating')
    // failed = the probe reported not-ok OR reported it won't fit; ok-stale = the
    // last warm succeeded yet the model has since gone non-resident (cold again).
    const loadFailed = showLoadWhy && (loadReport?.ok === false || loadReport?.fit === false)
    const loadStale  = showLoadWhy && !loadFailed && loadReport?.ok === true
    return { isPulling, isHeating, override, isPinned, alloc, inSlot, derivedMode,
             isAnswering, isServing, isIdleResident, pct, isMissing, state, stateTitle,
             loadReport, loadFailed, loadStale }
  }

  // Severity order for the State column sort: the more "live" a model is, the
  // higher it ranks (answering ▸ serving ▸ loaded ▸ heating ▸ pulling ▸ idle ▸
  // missing ▸ cold) — the same ladder the pills read top-to-bottom.
  const SERV_STATE_RANK = { answering: 7, serving: 6, loaded: 5, heating: 4, pulling: 3, idle: 2, missing: 1, cold: 0 }
  // ONE unified, ORDERED column model — the single source both the <th> row and
  // every <td> render from, so a column can be dragged anywhere across the whole
  // set (the data columns and the control columns are no longer two frozen
  // groups). Each def carries: whether it's SORTABLE (the six load-table
  // identifiers stay sortable via the existing servSort/servDir; the control
  // columns stay non-sortable), `num` (right-aligned numeric), the <td> class +
  // optional dynamic title, and a `render(cx)` that returns the cell body. `cx`
  // is the per-row context { key, m, d, isSel, isBlocked, need } (d = the
  // deriveModelState bundle). The renders are the SAME JSX the fixed layout used
  // — only relocated, so pills / Alloc editor / action buttons behave identically.
  const SERV_COL_DEFS = {
    task: {
      label: 'Task', sortable: true, cls: 'wp-lt-muted',
      title: ({ m }) => (m ? modelTasks(m).join(', ') : ''),
      // first task + "+N", full list on hover; '—' when not in the catalog feed
      render: ({ m }) => (m ? `${modelTask(m)}${modelTasks(m).length > 1 ? ` +${modelTasks(m).length - 1}` : ''}` : '—'),
    },
    framework: {
      label: 'Engine', sortable: true, cls: 'wp-lt-muted',
      render: ({ m }) => m?.framework || '—',
    },
    seat: {
      // Seat — the allocation-kind badge (engine-agnostic): a slot seat or the
      // worker's own RAM. Only present when the worker reports the unified
      // allocations view; '—' otherwise.
      label: 'Seat', sortable: false, cls: 'wp-servtable-seat',
      render: ({ d }) => (d.alloc ? (
        <span className={`wp-alloc-kind wp-alloc-${d.alloc.kind}`}
              title={d.alloc.kind === 'slot'
                ? 'allocated a slot (a resource seat) on this worker'
                : 'resident in the worker’s own process (RAM allocation)'}>
          {d.alloc.kind === 'slot' ? '🎰 slot' : '🧠 RAM'}
        </span>
      ) : <span className="wp-lt-muted">—</span>),
    },
    residency: {
      // Residency POLICY tag (v3 + slice 8): on-demand (default) or static.
      // Clicking only OPENS the picker (rendered in the expansion row).
      label: 'Residency', sortable: false, cls: '',
      render: ({ key }) => (worker.config && onSetResidency ? (() => {
        const mode = worker.config.residency?.[key] === 'static' ? 'static' : 'on-demand'
        const desc = mode === 'static'
          ? ' (locked seat — never swapped out; permanent with 📌 pin)'
          : ' (default — loads on call; holds its slot until another model needs the seat)'
        return (
          <button className={`wp-residency wp-residency-${mode}`}
                  disabled={applying}
                  title={applying
                    ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                    : `Residency policy: ${mode}${desc} — click to choose (a change applies via a ~5s agent restart)`}
                  onClick={() => setResMenu(resMenu === key ? null : key)}>
            {mode === 'static' ? '🔒 static' : '⏲ on-demand'}
          </button>
        )
      })() : <span className="wp-lt-muted">—</span>),
    },
    pin: {
      // Tiers v3 — 📌 pin = PERMANENT ATTRIBUTION of the model to this worker:
      // blocks unassign, files never reaped, residency overrides survive.
      label: '📌', sortable: false, cls: '',
      render: ({ key, d }) => (worker.config && onTogglePin ? (
        <button className={`wp-pin${d.isPinned ? ' wp-pin-on' : ''}`}
                disabled={applying}
                title={applying
                  ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                  : d.isPinned
                    ? 'Pinned: this allocation survives restarts — routing to this worker is durable and unassign is refused (409). Does not download the model and does not protect its files from eviction (bytes re-pull on call). Click to unpin.'
                    : 'Pin: make this allocation survive restarts — routing to this worker becomes durable and unassign is refused (409). Does not download the model or protect its files from eviction.'}
                onClick={() => onTogglePin(worker, key, !d.isPinned)}>
          📌{d.isPinned ? '' : '?'}
        </button>
      ) : <span className="wp-lt-muted">—</span>),
    },
    name: {
      // Model — nameFor(key), MIDDLE-truncated (both ends of a model key carry
      // the distinguishing part); full key on hover, and full untruncated in the
      // compact drawer. A ⛔ blocked chip rides here when the model is blocked
      // from the serving pool. The character budget tightens in compact mode,
      // where the column is only ~28vw of a phone-width table.
      label: 'Model', sortable: true, cls: 'wp-servtable-name',
      title: ({ key }) => key,
      render: ({ key, isBlocked, compact }) => (
        <>
          {/* Compact keeps the full 14-char TAIL (it is what discriminates
              sibling repos) and gives up head characters instead — the compact
              cell wraps rather than ellipsising, so 23 chars costs two lines,
              not a cut-off tail. */}
          {compact ? midTrunc(nameFor(key), 8, 14) : midTrunc(nameFor(key))}
          {isBlocked && (
            <span className="wp-blocked-chip"
                  title="⛔ Blocked from the serving pool by the operator — not routed to, assigned, warmed, or used as a fallback anywhere. This designation stays recorded but inert (block outranks pin). Files are untouched. Use the ⛔ action to unblock.">
              ⛔ blocked
            </span>
          )}
        </>
      ),
    },
    size: {
      // Size — SAME effective-quant rule as the serving facts (sizeInfo → the
      // ONE quant that serves for GGUF), honest on-disk title; '—' when unknown.
      label: 'Size', sortable: true, num: true, cls: 'wp-lt-num wp-lt-size',
      title: ({ key }) => {
        const si = sizeInfo(key)
        const det = worker.loaded_detail?.[key]
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        return declared == null ? 'size unknown — not on disk / not reported by the feed'
          : `${fmtBytes(declared)} on disk${si && si.isGguf ? ` (effective quant${si.effGguf ? ` ${si.effGguf}` : ''}, not the all-quants dir sum)` : ''}`
      },
      render: ({ key }) => {
        const si = sizeInfo(key)
        const det = worker.loaded_detail?.[key]
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        return declared == null ? '—' : fmtBytes(declared)
      },
    },
    ctx: {
      label: 'Ctx', sortable: true, num: true, cls: 'wp-lt-num',
      render: ({ m }) => m?.model_max_length || '—',
    },
    state: {
      // State — the EXISTING pill, verbatim (FixDoc on missing, live pulling %,
      // load_reports annotation when not resident).
      label: 'State', sortable: true, cls: 'wp-servtable-state',
      render: ({ d }) => {
        const { isPulling, isHeating, isServing, inSlot, isAnswering, isIdleResident,
                isMissing, pct, state, stateTitle, loadFailed, loadReport, loadStale } = d
        return (
          <>
            <span className={`wp-state-pill wp-pill-${state}`} title={stateTitle}>
              {isPulling ? `⏳ pulling${pct != null ? ` ${pct}%` : ''}`
                : isHeating ? '🔶 heating'
                : isServing ? (inSlot ? (isAnswering ? '⚡ answering' : '🔥 serving') : '📌 loaded')
                : isIdleResident ? '◍ idle'
                : isMissing ? '○ missing'
                : '○ cold'}
              {isMissing && <FixDoc doc="worker-model-missing" />}
            </span>
            {loadFailed && (
              <span className="wp-loadwhy wp-loadwhy-bad"
                    title={`${loadReport?.error || (loadReport?.fit === false ? 'probe: fit=false' : 'warm failed — the model never became resident')}${loadReport?.ts ? ` · ${fmtServed(loadReport?.ts)}` : ''}`}>
                ⚠
              </span>
            )}
            {loadStale && (
              <span className="wp-loadwhy wp-loadwhy-ok"
                    title={`last warm succeeded${loadReport?.ts ? ` ${fmtServed(loadReport?.ts)}` : ''} — not resident now`}>
                ⓘ
              </span>
            )}
          </>
        )
      },
    },
    moe: {
      // MoE — the EXPERT-SPLIT lever (operator ask 2026-07-26). Tri-state, but
      // presented as a plain checkbox on purpose: AUTO renders TICKED whenever
      // the derivation actually produced a split, so the operator sees the real
      // behaviour instead of an empty box that secretly means "on"
      // ("defaults can remain auto and should, but that also should entail a
      // checked box under the correct column, that could be switched by the
      // user"). Clicking pins the opposite state; ⟲ returns it to auto.
      label: 'MoE', sortable: false, cls: '',
      render: ({ key }) => {
        if (!worker.moe_capable?.[key]) {
          return <span className="wp-4bit-na" title={
            'No expert structure — this model is dense, so there is nothing to '
            + 'split.'}>—</span>
        }
        const ov = worker.moe_by_model?.[key]
        const pinned = ov !== undefined && ov !== null
        const on = key in moeOptimistic ? moeOptimistic[key]
                 : (pinned ? !!ov : !!worker.moe_effective?.[key])
        return (
          <span className="wp-moe">
            <label title={
              pinned
                ? `Expert split PINNED ${on ? 'on' : 'off'} by you. Click to flip; ⟲ restores auto.`
                : (on
                  ? 'Expert split ACTIVE, derived automatically — experts in RAM, everything else on the GPU. Untick to force it off.'
                  : 'Capable of an expert split, but the derivation did not apply one here (transformers MoE has no split path yet). Tick to force it on.')}>
              <input type="checkbox" className="wp-moe-box" checked={on}
                     disabled={applying}
                     onChange={e => setMoe(key, e.target.checked)} />
            </label>
            {pinned && (
              <button className="wp-moe-auto" disabled={applying}
                      title="Restore AUTO — follow the derivation for this model."
                      onClick={() => setMoe(key, null)}>⟲</button>
            )}
          </span>
        )
      },
    },
    fourbit: {
      // 4-BIT (operator ask 2026-07-26) — the bitsandbytes lever.
      // NAMED "4-bit", NOT "Quantize": the console already uses
      // "Quantization (GGUF variant)" on the Models tab (QuantControl) for a
      // DIFFERENT thing — picking WHICH pre-built GGUF quant file to download
      // (Q4_K_M vs Q8_0). This is load-time bitsandbytes on transformers
      // weights. Two distinct concepts must not share a name (operator ruling
      // 2026-07-26, keeper owns nomenclature).
      // Sits next to Alloc deliberately: it is a COMPRESSION choice that
      // re-prices the model, so the Alloc cell beside it re-derives the moment
      // this is ticked (67 GiB transformers: ram-only -> max-gpu at ~20 GiB).
      // Only rendered where it can actually work — not GGUF (llama.cpp carries
      // its own quantization), not a CPU-only worker (the 4-bit kernels are
      // CUDA-only), not an already-quantized repo. Elsewhere the cell is a
      // quiet em-dash rather than a disabled control nobody can use.
      label: '4-bit', sortable: false, cls: '',
      render: ({ key }) => {
        const avail = !!worker.bnb_available?.[key]
        if (!avail) {
          return <span className="wp-4bit-na" title={
            'No bitsandbytes specialization here: GGUF models carry their own '
            + 'quantization, the 4-bit kernels need a CUDA worker, and an '
            + 'already-quantized repo cannot be re-quantized.'}>—</span>
        }
        const on = key in bnbOptimistic ? bnbOptimistic[key] : !!worker.bnb_by_model?.[key]
        return (
          <label className="wp-4bit" title={
            on ? 'bitsandbytes 4-bit (nf4) ON — the model is priced at ~30% of '
               + 'its fp16 size, so its allocation re-derives. Untick to restore '
               + 'full precision.'
               : 'Load this model with bitsandbytes 4-bit (nf4). It is priced at '
               + '~30% of its fp16 size, so the Alloc column re-derives — often '
               + 'turning a RAM-only model into one that fits the GPU. Costs some '
               + 'quality.'}>
            <input type="checkbox" className="wp-4bit-box" checked={on}
                   disabled={applying}
                   onChange={e => setBnb(key, e.target.checked)} />
            <span className="wp-4bit-tag">{on ? '4-bit' : ''}</span>
          </label>
        )
      },
    },
    alloc: {
      // Alloc — the IN-PLACE mode picker (operator ask 2026-07-24). Clicking the
      // cell opens a compact menu anchored AT THE CLICK SITE (the five flat
      // modes); picking a zero-knob mode applies immediately, only "explicit"
      // opens extra chrome. The old full-width AllocControl expansion is retired.
      label: 'Alloc', sortable: false, cls: '',
      render: ({ key, m, d, need }) => {
        // Effective spill: an in-flight optimistic pick wins, else the persisted
        // override — so the cell flips the instant a mode is chosen.
        const effSpill = key in allocOptimistic ? allocOptimistic[key] : d.override
        // Blank (no persisted contract) => show the BACKEND's derived default,
        // not deriveAllocMode's 'max-gpu' fallback. Only a real spill is read
        // locally; an optimistic pick still wins so the cell stays responsive.
        const hasSpill = !!(effSpill && Object.keys(effSpill).length > 0)
        const mode = hasSpill ? deriveAllocMode(effSpill)
                              : (d.derivedMode || deriveAllocMode(effSpill))
        const engineGguf = /^(gguf|llama_cpp)$/.test(String(m?.framework || '').toLowerCase())
        const isOpen = allocMenu === key
        // Ruling 2 (2026-07-24) — DERIVED vs PINNED at a glance. A model with NO
        // persisted contract TRACKS the derivation ("Auto"): the cell shows the
        // derived mode dimmed/italic with an "auto" affix. A persisted pick shows
        // solid. We decide this LOCALLY from the override (no fetch): an empty /
        // absent override == tracking-the-derivation; any non-empty override ==
        // an operator pin. (This is stricter-but-honester than the backend's
        // narrow alloc_mode_derived, which only inspects the alloc_mode key and
        // so calls a gpu-only/ram-only pin "derived" too — here an operator's
        // n_gpu_layers:-1 correctly reads as a pin. When the serving fetch has
        // landed we PREFER its alloc_mode_derived for the exact backend truth.)
        // NOTE the subtlety: explicit max-gpu ≠ blank. Blank tracks the
        // derivation as it improves (measured values land later via the
        // calibration evaluator); an explicit pick freezes the mode.
        const sd = servDetail[key]
        const isDerived = (sd && typeof sd.alloc_mode_derived === 'boolean' && !(key in allocOptimistic))
          ? sd.alloc_mode_derived
          : !(effSpill && Object.keys(effSpill).length > 0)
        // Per-(model,worker) feasibility + derived default, once the serving GET
        // for this key has landed. Fail-OPEN: undefined feasible ⇒ no disables.
        const byWorker = (sd && sd.alloc_by_worker && sd.alloc_by_worker[worker.id]) || null
        const feasibleUnion = (sd && Array.isArray(sd.alloc_modes_feasible)) ? sd.alloc_modes_feasible : null
        const feasible = byWorker && Array.isArray(byWorker.feasible) ? byWorker.feasible
          : feasibleUnion   // fall back to the model-level union when unscoped
        const derivedMode = (byWorker && byWorker.derived_default)
          || (sd && sd.alloc_mode_derived && sd.alloc_mode) || null
        return (
          <span className="wp-allocmode-anchor">
            <button className={`wp-alloc-edit${isDerived ? ' wp-alloc-derived' : ''}`} disabled={applying}
                    ref={isOpen ? allocAnchorRef : undefined}
                    title={applying
                      ? 'Agent is applying the previous change — retry in a few seconds.'
                      : isDerived
                        ? `Allocation: ${allocModeLabel(mode)} — DERIVED (no pinned contract; tracks the default as it improves). Click to change or pin.`
                        : `Allocation: ${allocModeLabel(mode)} — pinned. Click to change or revert to the derived default.`}
                    onClick={() => { setAllocMenu(isOpen ? null : key); if (!isOpen) fetchServDetail(key) }}>
              {allocModeLabel(mode)}
              {isDerived && <span className="wp-alloc-auto-affix"> · auto</span>}
            </button>
            {isOpen && (
              <AllocModeMenu
                mode={mode}
                spill={effSpill}
                worker={worker}
                need={need}
                engineGguf={engineGguf}
                feasible={feasible}
                feasibleCtx={{ modelBytes: need?.bytes ?? null,
                               vramTotal: worker.vram_total ?? null,
                               ramTotal: worker.ram_total ?? null }}
                derivedMode={derivedMode}
                anchorRef={allocAnchorRef}
                onPick={(next) => applyAllocMode(key, { alloc_mode: next })}
                onApplyExplicit={(s) => applyAllocMode(key, s)}
                onRevertDerived={() => applyAllocMode(key, {})}
                onClose={() => setAllocMenu(null)}
              />
            )}
          </span>
        )
      },
    },
    memory: {
      // Memory — the HONEST THREE-QUANTITY display (ruling 3, 2026-07-24):
      //   "<disk> disk · <resident> resident (<vram> VRAM + <anon> RAM) · <n>/<total> layers"
      // when the shipped 0.1.198+ fields are present on the row (vram_bytes,
      // rss_anon_bytes, n_gpu_layers/total_layers — all on the measured
      // allocations view d.alloc, with loaded_detail as fallback). Resident =
      // VRAM + anon RAM (the true footprint; disk is a separate universe). The
      // cell stays COMPACT — "disk · resident" — and the full breakdown rides
      // the title. Absent fields degrade to what exists (never invented): VRAM
      // 0 means resident-but-on-CPU; no measured figures fall back to the
      // declared-split line the cell always showed.
      label: 'Memory', sortable: false, cls: 'wp-servtable-mem',
      render: ({ key, d }) => {
        const det = worker.loaded_detail?.[key]
        const a = d.alloc || null
        // Prefer the measured allocations view; fall back to loaded_detail.
        const vram = a && a.vram_bytes != null ? a.vram_bytes
          : det && det.vram_bytes != null ? det.vram_bytes : null
        // Host-RAM occupancy: a slot child's anon RSS, or — for an in-process
        // (kind:'ram') model — the worker's MEASURED ram_resident_bytes (ships
        // with the next release; absent on older workers, which then degrade to
        // the disk-only line exactly as before). Either way this is a
        // measurement, never a declared/file figure.
        const anon = a && a.rss_anon_bytes != null ? a.rss_anon_bytes
          : a && a.ram_resident_bytes != null ? a.ram_resident_bytes
          : det && det.rss_anon_bytes != null ? det.rss_anon_bytes
          : det && det.ram_resident_bytes != null ? det.ram_resident_bytes : null
        const ngl = a && a.n_gpu_layers != null ? a.n_gpu_layers
          : det && det.n_gpu_layers != null ? det.n_gpu_layers : null
        const totalLayers = a && a.total_layers != null ? a.total_layers
          : det && det.total_layers != null ? det.total_layers : null
        const measuredVram = vram   // legacy name kept for the declared-split fallback below
        const si = sizeInfo(key)
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        if (declared == null && det?.gpu_pct == null && measuredVram == null) return <span className="wp-lt-muted">—</span>
        // Resident footprint = measured VRAM + anon RAM (only when at least one
        // measured figure exists; a bare declared size is NOT residency).
        const resident = (vram != null || anon != null)
          ? (vram || 0) + (anon || 0) : null
        const layersTxt = ngl == null ? ''
          : `${ngl === -1 ? 'all' : ngl}${totalLayers ? `/${totalLayers}` : ''} layers on GPU`
        const declaredTitle = declared != null
          ? `${fmtBytes(declared)} on disk${si && si.isGguf ? ` (effective quant${si.effGguf ? ` ${si.effGguf}` : ''}, not the all-quants dir sum)` : ''}`
          : ''
        // The FULL breakdown for the tooltip — the honest three quantities named.
        const fullTitle = [
          declaredTitle,
          resident != null ? `${fmtBytes(resident)} resident = ${vram != null ? fmtBytes(vram) : '0'} VRAM + ${anon != null ? fmtBytes(anon) : '0'} anon RAM` : '',
          layersTxt,
          (vram != null || anon != null)
            ? 'VRAM/RAM figures are MEASURED (nvidia-smi / rss_anon / ram_resident); VRAM 0 = running on CPU'
            : (det?.gpu_pct != null ? 'GPU split is DECLARED by the loader, not a measured VRAM read' : ''),
        ].filter(Boolean).join(' · ')
        // NOT-YET-RESIDENT: show the PLANNED division, not the on-disk size.
        // The old fallback rendered "<X> disk" — the exact number the Size
        // column already shows, which is the redundancy the operator flagged.
        // planned_split answers the question this column exists for ("where will
        // this actually go") and moves with the Alloc mode, the 4-bit lever and
        // the MoE lever, so flipping any switch visibly updates the row.
        const plan = worker.planned_split?.[key]
        if (resident == null && measuredVram == null && plan
            && (plan.gpu_bytes != null || plan.ram_bytes != null)) {
          const g = plan.gpu_bytes, r = plan.ram_bytes
          const parts = []
          if (g) parts.push(`${fmtBytes(g)} VRAM`)
          if (r) parts.push(`${fmtBytes(r)} RAM`)
          return (
            <span className="wp-model-facts wp-fact-planned" title={
              (plan.split
                ? `PLANNED expert split: ${fmtBytes(g || 0)} of non-expert tensors on the GPU, `
                  + `${fmtBytes(r || 0)} of experts in RAM. `
                : `PLANNED placement under '${plan.mode}': `)
              + (plan.split ? '' : `${fmtBytes(g || r || 0)} on ${g ? 'the GPU' : 'the CPU'}`
                 + (plan.mode === 'max-gpu' ? ' (spills whatever will not fit at load time)' : '')
                 + '. ')
              + 'Not resident yet — this is what the current Alloc mode and the '
              + '4-bit / MoE switches add up to, not a measurement.'}>
              {parts.join(' + ')}
              <span className="wp-fact-planned-tag"> planned</span>
            </span>
          )
        }
        return (
          <span className="wp-model-facts" title={fullTitle}>
            {declared != null && <span className="wp-fact-disk">{fmtBytes(declared)} disk</span>}
            {resident != null ? (
              <span className="wp-fact-resident">
                {' · '}{fmtBytes(resident)} resident
                <span className="wp-fact-split"> ({vram != null ? fmtBytes(vram) : '0'} VRAM + {anon != null ? fmtBytes(anon) : '0'} RAM)</span>
                {totalLayers != null && ngl != null && (
                  <span className="wp-fact-layers"> · {ngl === -1 ? totalLayers : ngl}/{totalLayers} layers</span>
                )}
              </span>
            ) : measuredVram != null
              ? (measuredVram > 0 ? ` · ${fmtBytes(measuredVram)} VRAM` : ' · 0 VRAM · on CPU')
              : det?.gpu_pct != null ? ` · ~${det.gpu_pct}% GPU / ${100 - det.gpu_pct}% spill` : ''}
          </span>
        )
      },
    },
    actions: {
      // Actions — ▶ activate (cold/missing) / ⏏ free (resident or hollow-idle) /
      // × unassign (blocked while pinned) / ⛔ block toggle.
      label: 'Actions', sortable: false, cls: 'wp-servtable-actions',
      render: ({ key, d, isBlocked }) => {
        const { state, isServing, isIdleResident, isPinned } = d
        return (
          <>
            {onLoad && (state === 'cold' || state === 'missing') && (
              <button className="wp-activate" disabled={activating === key}
                      title={activating === key
                        ? 'seating this model on the worker…'
                        : 'Activate: allocate this worker’s resources to the model now so it serves immediately. Files transfer first if not local yet.'}
                      onClick={async () => {
                        setActivating(key)
                        try { await onLoad(worker, key) }
                        finally { setActivating(null) }
                      }}>
                {activating === key ? '⏳ activating…' : '▶ activate'}
              </button>
            )}
            {(isServing || isIdleResident) && (
              <button className="wp-free"
                      title={isIdleResident
                        ? 'Clear this idle runner shell from the worker (stays assigned)'
                        : 'Unload from VRAM (stays assigned)'}
                      onClick={() => onFree(worker, key)}>⏏</button>
            )}
            <button className="wp-model-x" disabled={isPinned}
                    title={isPinned ? 'pinned — unpin first' : 'Unassign'}
                    onClick={() => onUnassign(worker, key)}>×</button>
            {onToggleBlock && (
              <button className={`wp-model-block${isBlocked ? ' wp-model-block-on' : ''}`}
                      title={isBlocked
                        ? 'Blocked from the serving pool — click to UNBLOCK (return it to routing). Block outranks pin; the designation is unchanged.'
                        : 'Block this model from the serving pool (global): never routed to / assigned / warmed / a fallback default anywhere. Files stay; designations stay (inert). Reversible.'}
                      onClick={() => onToggleBlock(key, !isBlocked)}>⛔</button>
            )}
          </>
        )
      },
    },
  }
  // Persistent, user-adjustable column order + widths (localStorage, per browser).
  const { order: servOrder, widths: servWidths, moveColumn: servMoveCol,
          setWidth: servSetWidth, reset: servResetLayout } = useColumnLayout(SERV_LAYOUT_KEY, SERV_DEFAULT_ORDER)
  const servColsAll = servOrder.map(k => ({ key: k, ...SERV_COL_DEFS[k] })).filter(c => c.label != null)
  // COMPACT MODE (operator ask 2026-07-28) — measured on the table's wrapper, so
  // a narrow panel inside a wide window compacts too. In compact the table keeps
  // only checkbox / Model / State / Memory (no horizontal scroll at all) and
  // every other column def moves into the per-row drawer, rendered from the SAME
  // def.render(cx) — the cell logic is never forked.
  const servWrapRef = useRef(null)
  const servCompact = useNarrowContainer(servWrapRef, SERV_COMPACT_PX)
  const servCols = servCompact
    // keep the operator's own left-to-right order among the surviving three
    ? servColsAll.filter(c => SERV_COMPACT_COLS.includes(c.key))
    : servColsAll
  // The columns that moved OUT of the table and into the drawer, in the same
  // order the operator arranged them (Actions is rendered last, on its own row).
  const servDrawerCols = servCompact
    ? servColsAll.filter(c => !SERV_COMPACT_COLS.includes(c.key) && c.key !== 'actions')
    : []
  const servActionsCol = SERV_COL_DEFS.actions
  // +1 for the leading select-checkbox column (todo t12 bulk residency).
  const SERV_COLSPAN = 1 + servCols.length
  // Which row's drawer is open (compact mode only) — ONE at a time: tapping the
  // same row closes it, tapping another moves it. Cleared whenever we leave
  // compact mode so a returning desktop layout never carries a stray row.
  const [servDrawer, setServDrawer] = useState(null)
  useEffect(() => { if (!servCompact) setServDrawer(null) }, [servCompact])
  // A row tap opens/closes the drawer — but ONLY when the tap landed on the row
  // BODY. Anything interactive (the select checkbox, a button, an input, a
  // select, a label, a link) keeps its own behaviour untouched.
  const servRowTap = (key) => (e) => {
    if (!servCompact) return
    const t = e.target
    if (t && typeof t.closest === 'function'
        && t.closest('button, input, select, textarea, label, a, .wp-servtable-selcol')) return
    setServDrawer(prev => (prev === key ? null : key))
  }
  // Drag-to-reorder: the key of the header currently being dragged, and the one
  // it's hovering over (for the drop-indicator). Native HTML5 drag on the <th>s;
  // the resize grip cancels its own dragstart so the two gestures never collide.
  const [servDragCol, setServDragCol] = useState(null)
  const [servDragOver, setServDragOver] = useState(null)
  // Drag-to-resize: pointer-drag a header's right edge, writing px width to the
  // hook (→ the <colgroup>). Listens on window so the drag survives leaving the
  // 6px grip.
  const startServResize = (e, key) => {
    e.preventDefault(); e.stopPropagation()
    const th = e.currentTarget.closest('th')
    const startX = e.clientX
    const startW = th ? th.getBoundingClientRect().width : (servWidths[key] || 120)
    const onMove = (ev) => servSetWidth(key, startW + (ev.clientX - startX))
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  // Task dropdown options: the union of EVERY task advertised by the models
  // ASSIGNED to this worker (a multi-task model surfaces under each of its
  // tasks — the same union rule the load table uses), narrowed to what's here.
  const servTasks = useMemo(() => {
    const assigned = new Set(worker.models || [])
    const here = models.filter(m => assigned.has(m.model_key ?? m.key))
    return [...new Set(here.flatMap(modelTasks))].sort()
  }, [models, worker.models])
  // The Size sort value MUST match the Size cell: the true effective-quant
  // footprint (sizeInfo → the ONE quant that serves for GGUF), falling back to
  // the worker's loaded_detail dir bytes; unknown (-1) sinks to the bottom asc.
  const servSizeOf = (key) => {
    const si = sizeInfo(key)
    const det = worker.loaded_detail?.[key]
    const b = (si && si.bytes != null) ? si.bytes : det?.model_bytes
    return b == null ? -1 : Number(b)
  }
  // One row per assigned model (state derived once so it can be BOTH sorted and
  // rendered), then the search + task filter, then the sort.
  const servRowsAll = (worker.models || []).map(key => ({
    key,
    m: models.find(mm => (mm.model_key ?? mm.key) === key),
    d: deriveModelState(key),
  }))
  const servNeedle = servQ.trim().toLowerCase()
  let servRows = servRowsAll
  if (servNeedle) servRows = servRows.filter(({ key, m }) =>
    (m?.name || key).toLowerCase().includes(servNeedle) || key.toLowerCase().includes(servNeedle))
  // Full-list task match — a model counts under ANY task it advertises.
  if (servTask) servRows = servRows.filter(({ m }) => m && modelTasks(m).includes(servTask))
  const servGet = ({ key, m, d }) => {
    switch (servSort) {
      case 'task': return m ? (modelTask(m) || '') : ''
      case 'framework': return m?.framework || ''
      case 'size': return servSizeOf(key)
      case 'ctx': return m?.model_max_length || 0
      case 'state': return SERV_STATE_RANK[d.state] ?? -1
      case 'name':
      default: return m?.name || key
    }
  }
  const servNumeric = servSort === 'size' || servSort === 'ctx' || servSort === 'state'
  const servDirMul = servDir === 'asc' ? 1 : -1
  servRows = [...servRows].sort((a, b) => {
    const va = servGet(a), vb = servGet(b)
    if (servNumeric) return (Number(va) - Number(vb)) * servDirMul
    return String(va).localeCompare(String(vb)) * servDirMul
  })
  const toggleServSort = (k) => {
    if (servSort === k) setServDir(servDir === 'asc' ? 'desc' : 'asc')
    else { setServSort(k); setServDir('asc') }
  }
  const servArrow = (k) => servSort === k ? (servDir === 'asc' ? ' ▲' : ' ▼') : ''

  // Bulk-residency selection derivations (todo t12), computed off the CURRENT
  // filter (servRows) so "select all" and the header checkbox track exactly
  // what the operator can see. selCount counts only selected keys still visible
  // AND still designated — a stale selection (model unassigned in another tab)
  // never inflates the count or the "N selected" bar.
  const servVisibleKeys = servRows.map(r => r.key)
  const selVisible = servVisibleKeys.filter(k => servSel.has(k))
  const selCount = selVisible.length
  const allVisibleSelected = servVisibleKeys.length > 0 && selCount === servVisibleKeys.length
  const someVisibleSelected = selCount > 0 && !allVisibleSelected
  const selectAllVisible = () => setServSel(prev => {
    const next = new Set(prev)
    if (allVisibleSelected) servVisibleKeys.forEach(k => next.delete(k))
    else servVisibleKeys.forEach(k => next.add(k))
    return next
  })
  const canBulkResidency = !!worker.config && !!onSetResidencyMany
  const canBulkAlloc = !!onSetAllocMany

  return (
    <div className={`wp-worker wp-${worker.status} wp-adm-${worker.admission || 'approved'}`}>
      <div className="wp-worker-head">
        <span className="wp-dot" />
        <span className="wp-name">{worker.name}</span>
        <span className="wp-status">{worker.status}</span>
        {/* Version pill — control-plane/worker skew is silent behavior drift, so
            the running package version rides in the header next to the name.
            GREEN = in sync with central's required_pkg_version, RED = skewed
            (⬆ Update converges it now), NEUTRAL = the worker didn't report a
            version, or central has no pin, or the row predates version_ok
            (older workers / showroom fixtures) — never guess red from absence. */}
        {worker.pkg_version && (
          <span className={`wp-ver ${worker.version_ok === true ? 'wp-ver-ok'
            : worker.version_ok === false ? 'wp-ver-skew' : 'wp-ver-unknown'}`}
                title={worker.version_ok === true
                  ? `abstract_hugpy_dev ${worker.pkg_version} — in sync with central`
                  : worker.version_ok === false
                    ? `worker on ${worker.pkg_version}, central requires ${worker.required_pkg_version || '?'} — use ⬆ Update to converge now (or it self-updates on its next heartbeat)`
                    : `abstract_hugpy_dev ${worker.pkg_version} — central reported no required version to compare against`}>
            v{worker.pkg_version}
          </span>
        )}
        {worker.engine_build && (
          // ENGINE build id beside the pkg version (item L, k65): the native
          // llama-server commit, so engine skew across the fleet is visible next
          // to version skew. The spawn now probes engine capability; this makes
          // the build it probes against legible.
          <span className="wp-ver wp-ver-engine"
                title={`native llama-server engine build ${worker.engine_build} — surfaced so engine skew across the fleet is visible (item L). The fat-arch rebuild is tracked separately.`}>
            ⚙ {worker.engine_build}
          </span>
        )}
        <span className={`wp-adm wp-adm-pill-${worker.admission || 'approved'}`}
              title="Operator admission gate — only approved workers serve traffic">
          {worker.admission || 'approved'}
          {/* pending/blocked are warnings — link the fix; approved needs none */}
          {worker.admission && worker.admission !== 'approved' && <FixDoc doc="worker-admission" />}
        </span>
        <span className={`wp-pool ${worker.pool ? 'wp-pool-set' : ''}`} role="button"
              title="Dedicated pool — this worker serves ONLY requests tagged for it (general traffic never lands here). Click to set/clear."
              onClick={() => onSetPool(worker)}>
          🏷 {worker.pool || 'general'}
        </span>
        {worker.role === 'rpc' && <span className="wp-role" title="shard backend (lends GPU via rpc-server)">rpc</span>}
        {worker.comfy?.available && (
          <span className="wp-role" title={`ComfyUI running on this worker${worker.comfy.version ? ` (v${worker.comfy.version})` : ''} at ${worker.comfy.url} — comfy-templated generation routes here (engine slice B)`}>
            🧩 comfy
          </span>
        )}
        <span className="wp-url" title={worker.url}>{worker.url}</span>
        <SpillBadge spill={worker.spill} />
        {(worker.gpus || []).length > 0 && worker.engine?.supports_gpu_offload === false && (
          <span className="wp-cpu-only"
                title={'This worker\'s llama-cpp-python is a CPU-only build: GGUF models run on CPU and n_gpu_layers is silently ignored, so VRAM stays idle. Rebuilding it with GPU support has real traps (missing nvcc, missing CUDA runtime libs, AVX512 SIGILL) — click 📖 for the diagnostic and the rebuild recipe that actually works.'}>
            ⚠ CPU-only engine
            <FixDoc doc="engine-cpu-only" />
          </span>
        )}
        {worker.install && worker.install.canonical === false && (
          <span className="wp-noncanon"
                title={`Non-canonical install — this worker did NOT come from the standard bootstrap/installer.\n`
                       + `unit: ${worker.install.unit || (worker.install.via_systemd ? 'systemd (name unknown)' : 'not a systemd unit')}\n`
                       + `venv: ${worker.install.venv || 'unknown'}\n`
                       + `Canonical = a hugpy-worker.service (or legacy abstract-hugpy-worker.service) user unit running from ~/hugpy-worker/venv. See WORKER-SETUP.md §1.`}>
            ⚠ non-canonical install
          </span>
        )}
        {ping && ping !== 'checking' && (
          <span className={`wp-ping ${ping.reachable ? 'wp-ping-ok' : 'wp-ping-bad'}`}
                title={ping.reachable ? 'central can reach this worker' : (ping.error || 'unreachable')}>
            {ping.reachable ? '✓ reachable' : '✗ unreachable'}
            {!ping.reachable && <FixDoc doc="worker-unreachable" />}
          </span>
        )}
        <button className="wp-ping-btn" title="Ping the worker's /health from central"
                onClick={checkHealth} disabled={ping === 'checking'}>
          {ping === 'checking' ? '…' : 'ping'}
        </button>
        {loaded.size > 0 && (
          <button className="wp-free-all" title="Unload every model from this GPU (stays assigned)"
                  onClick={() => onFreeAll(worker)}>
            ⏏ free VRAM
          </button>
        )}
        {/* Maintenance controls — both shown ALWAYS (Free RAM is the fix for an
            orphaned arena that holds RAM with nothing loaded, so never gate it). */}
        <button className="wp-free-ram" title="Return reclaimable host RAM to the OS — non-destructive: loaded models stay resident"
                onClick={() => onFreeRam(worker)}>
          🧹 Free RAM
        </button>
        <button className="wp-restart" title="Restart the worker agent — drops all loaded models and re-execs the agent process"
                onClick={() => onRestart(worker)} disabled={restarting}>
          {restarting ? '↻ restarting…' : '↻ Restart'}
        </button>
        {/* ⬆ Update — converge this worker onto central's required version now
            instead of waiting for its next heartbeat. It pip-installs and
            restarts itself, so it shares the restart transient (and its flag). */}
        {onUpdate && (
          <button className={`wp-update${worker.version_ok === false ? ' wp-update-skew' : ' wp-update-noop'}`}
                  title={worker.version_ok === false
                    ? `worker on ${worker.pkg_version || '?'}, central requires ${worker.required_pkg_version || '?'} — update pip-installs the required version and restarts the agent`
                    : 'already at central’s required version — update is a no-op'}
                  onClick={() => onUpdate(worker)} disabled={updating || restarting}>
            {updating ? '⬆ updating…' : '⬆ Update'}
          </button>
        )}
        {worker.admission === 'approved'
          ? <button className="wp-block" title="Block: stop serving; the agent exits on its next contact and won't respawn"
                    onClick={() => onBlock(worker)}>⛔ block</button>
          : <button className="wp-admit" title={worker.admission === 'blocked' ? 'Unblock and allow serving' : 'Admit: allow this worker to serve'}
                    onClick={() => onAdmit(worker)}>✓ {worker.admission === 'blocked' ? 'unblock' : 'admit'}</button>}
        <button className="wp-remove" title="Remove worker (forget — a live agent re-appears as pending; use Block to evict)" onClick={() => onRemove(worker)}>✕</button>
      </div>

      {/* Resource header: VRAM · RAM · Storage as compact chips, each expanding
          its resident/detail panel below on click. Replaces the old budget bar +
          storage bar + per-GPU chips + free-RAM line. */}
      <ResourceStrip worker={worker} models={models} onApproveEvictions={onApproveEvictions} onEvict={onEvict} />

      {/* Two-tier resource governance: the box's OWN config (caps) is the hard
          ceiling; central-set limits are clamped to it server-side. */}
      <div className="wp-caps">
        {worker.caps && Object.keys(worker.caps).length > 0 && (
          <span className="wp-cap-chip" title="Configured on the worker box itself (unit env) — central can only set limits at or below these.">
            box caps:{worker.caps.ram_max_gib != null && ` RAM ${worker.caps.ram_max_gib}GiB`}
            {worker.caps.gpu_mem_gib != null && ` · VRAM ${worker.caps.gpu_mem_gib}GiB`}
            {worker.caps.disk_cache_gib != null && ` · disk ${worker.caps.disk_cache_gib}GiB`}
            {worker.caps.threads != null && ` · ${worker.caps.threads} threads`}
          </span>
        )}
        {worker.limits && Object.keys(worker.limits).length > 0 && (
          <span className="wp-cap-chip wp-limit-chip" title="Central-set limits (≤ box caps); the worker adopts them on its next heartbeat.">
            central limits:{worker.limits.ram_max_gib != null && ` RAM ${worker.limits.ram_max_gib}GiB`}
            {worker.limits.gpu_mem_gib != null && ` · VRAM ${worker.limits.gpu_mem_gib}GiB`}
            {worker.limits.disk_cache_gib != null && ` · disk ${worker.limits.disk_cache_gib}GiB`}
            {worker.limits.threads != null && ` · ${worker.limits.threads} threads`}
          </span>
        )}
        {/* Console-managed serving config (daylight item 3): slot count lives
            in the AGENT's own settings (beats env drop-ins); the chip shows
            the EFFECTIVE value + where it came from. */}
        {worker.config?.slot_count != null && (
          <span className={`wp-cap-chip${applying ? ' wp-chip-applying' : ''}`} role="button"
                title={applying
                  ? 'The agent is restarting (~5s) to apply the previous config change — controls unlock when the new config arrives in a heartbeat.'
                  : `Worker slot pool: ${worker.config.slot_count} slot(s) — source: ${worker.config.slot_count_source || '?'}. Click to change (persists in the agent's runtime settings; applies via a ~5s agent restart).`}
                onClick={() => !applying && onSetConfig && onSetConfig(worker)}>
            🎛 slots: {worker.config.slot_count}
            {worker.config.slot_count_source && worker.config.slot_count_source !== 'settings' &&
              <em> ({worker.config.slot_count_source})</em>}
          </span>
        )}
        {applying && (
          <span className="wp-applying"
                title="The last pin/residency/slot-count change was accepted; the agent re-execs (~5s) to apply it and this worker's config controls are paused until the new config shows up in a heartbeat.">
            ⏳ applying…
          </span>
        )}
        {onSetLimits && (
          <button className="wp-limits-edit" title="Worker budget — this box's resource ceiling for ALL models combined (VRAM / RAM / disk / threads), clamped to its box caps. This is NOT a per-model allocation: per-model VRAM/RAM placement is the Alloc column on each serving row. Central-set (≤ box caps)."
                  onClick={() => {
                    setLimitsForm({
                      ram_max_gib: worker.limits?.ram_max_gib ?? '',
                      gpu_mem_gib: worker.limits?.gpu_mem_gib ?? '',
                      disk_cache_gib: worker.limits?.disk_cache_gib ?? '',
                      threads: worker.limits?.threads ?? '',
                    })
                    setLimitsOpen(o => !o)
                  }}>
            ⚙ worker budget
          </button>
        )}
        {limitsOpen && (
          <span className="wp-limits-form">
            {/* Ruling 4 (2026-07-24): an unmistakable label so this box-level
                ceiling can never again be read as per-model allocation (the
                operator conflated them). The per-model contract is the Alloc
                column; THIS is the whole-box budget. */}
            <span className="wp-limits-title" title="This box's resource ceiling across ALL models it hosts — not a per-model budget.">
              worker budget — this box's resource ceiling (all models combined):
            </span>
            <input type="number" step="1" min="0" placeholder="RAM GiB" value={limitsForm.ram_max_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, ram_max_gib: e.target.value }))} />
            <input type="number" step="1" min="0" placeholder="VRAM GiB" value={limitsForm.gpu_mem_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, gpu_mem_gib: e.target.value }))} />
            <input type="number" step="1" min="0" placeholder="disk cache GiB" title="Local model-cache ceiling for this worker. Over it, cold local models become eviction candidates in the storage proposal. Clamped to the box's own caps.disk_cache_gib — the worker's stated delegation wins."
                   value={limitsForm.disk_cache_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, disk_cache_gib: e.target.value }))} />
            <input type="number" step="1" min="1" placeholder="threads" value={limitsForm.threads}
                   onChange={e => setLimitsForm(f => ({ ...f, threads: e.target.value }))} />
            <button className="wp-alloc-apply" onClick={() => {
              const limits = {}
              for (const k of ['ram_max_gib', 'gpu_mem_gib', 'disk_cache_gib', 'threads']) {
                if (limitsForm[k] !== '' && limitsForm[k] != null) limits[k] = Number(limitsForm[k])
              }
              onSetLimits(worker, limits)
              setLimitsOpen(false)
            }}>Set</button>
            <button className="wp-alloc-cancel" title="Clear all central limits"
                    onClick={() => { onSetLimits(worker, {}); setLimitsOpen(false) }}>clear</button>
          </span>
        )}
      </div>


      {/* Serving: every model DESIGNATED to this worker, as a comprehensive,
          sortable TABLE (same visual system as the "load a model" table). The
          load-table column identifiers (Model/Task/Engine/Size/Ctx) come first,
          then the serving-specific ones (State/Seat/Alloc/Residency/📌/Memory/
          Actions). "Serving" here is the whole assignment at EVERY stage of the
          attribution ladder — not only the models actively hosted right now. */}
      <div className={`wp-models wp-servtable${servCompact ? ' wp-servtable-compact' : ''}`} ref={servWrapRef}>
        <div className="wp-servtable-bar">
          <span className="wp-models-label">Serving:</span>
          {(worker.models || []).length > 0 && (
            <>
              <input className="wp-servtable-q" placeholder="filter models…" value={servQ}
                     onChange={e => setServQ(e.target.value)} />
              <select className="wp-servtable-task" value={servTask}
                      onChange={e => setServTask(e.target.value)}
                      title="Filter by task — matches ANY task a model advertises, not just its primary">
                <option value="">All tasks</option>
                {servTasks.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              {/* COMPACT-ONLY sort control. In compact mode most sortable
                  headers are not on screen to click, so the same servSort /
                  servDir state gets a select + a direction toggle here. Desktop
                  keeps header-click sorting and never renders this. */}
              {servCompact && (
                <span className="wp-servtable-sortsel">
                  <select value={servSort} onChange={e => setServSort(e.target.value)}
                          title="Sort the serving list (the sortable columns are hidden in this narrow layout)">
                    {servColsAll.filter(c => c.sortable).map(c => (
                      <option key={c.key} value={c.key}>sort: {c.label}</option>
                    ))}
                  </select>
                  <button className="wp-servtable-sortdir"
                          title={servDir === 'asc' ? 'Ascending — click for descending' : 'Descending — click for ascending'}
                          onClick={() => setServDir(servDir === 'asc' ? 'desc' : 'asc')}>
                    {servDir === 'asc' ? '▲' : '▼'}
                  </button>
                </span>
              )}
              {/* Reset the per-user column layout (order + widths) back to the
                  default. Only appears once the layout diverges from default, so
                  it's silent for operators who never touch the headers. Hidden in
                  compact mode — there is no drag/resize there to undo. */}
              {!servCompact && (servOrder.join(',') !== SERV_DEFAULT_ORDER.join(',') || Object.keys(servWidths).length > 0) && (
                <button className="wp-servtable-resetcols" onClick={servResetLayout}
                        title="Reset the serving-table columns (order + widths) to the default layout. Drag a header to reorder, or a header's right edge to resize — your layout is remembered per browser.">
                  ⟲ reset columns
                </button>
              )}
            </>
          )}
          {/* Bulk residency (todo t12): appears once one or more models in the
              current filter are selected (via the row checkboxes). Sets the
              RESIDENCY tier of the whole selection in ONE agent restart (central
              relays a single /ops/config residency map — same batching pinAll
              does). Confirms first, surfaces per-model results. Residency ONLY —
              NOT 📌 pin (that's the separate control to the right). */}
          {(canBulkResidency || canBulkAlloc) && selCount > 0 && (
            <span className="wp-bulkres wp-servtable-bulkres">
              <span className="wp-bulkres-count" title="models selected in the current filter">
                {selCount} selected
              </span>
              {/* Bulk RESIDENCY (todo t12): a ~5s agent restart. */}
              {canBulkResidency && (
                <>
                  <span className="wp-bulkres-label">set residency →</span>
                  <button className="wp-bulkres-btn wp-bulkres-ondemand" disabled={applying}
                          title={applying
                            ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                            : `Set the ${selCount} selected model${selCount === 1 ? '' : 's'} to ⏲ on-demand (the default — loads on call, yields its seat under contention). Confirms first; one ~5s agent restart.`}
                          onClick={() => onSetResidencyMany(worker, selVisible, 'on-demand')}>
                    ⏲ on-demand
                  </button>
                  <button className="wp-bulkres-btn wp-bulkres-static" disabled={applying}
                          title={applying
                            ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                            : `Set the ${selCount} selected model${selCount === 1 ? '' : 's'} to 🔒 static (locked seat — kept on this worker, never evicted). Confirms first; one ~5s agent restart.`}
                          onClick={() => onSetResidencyMany(worker, selVisible, 'static')}>
                    🔒 static
                  </button>
                </>
              )}
              {/* Bulk ALLOC (todo t15): a registry write — NO restart. The ⚙
                  button toggles the inline AllocControl below the bar (same
                  editor the per-row ⚙ uses), applied to the whole selection. */}
              {canBulkAlloc && (
                <>
                  {canBulkResidency && <span className="wp-bulkres-sep" aria-hidden="true">·</span>}
                  <span className="wp-bulkres-label">set alloc →</span>
                  <button className={`wp-bulkres-btn wp-bulkres-alloc${bulkAllocOpen ? ' wp-bulkres-alloc-open' : ''}`}
                          title={`Set the GPU allocation (Default / Max GPU / GPU only / RAM only / Max RAM / Explicit) for the ${selCount} selected model${selCount === 1 ? '' : 's'}. A registry contract applied on next load — no agent restart. Confirms first.`}
                          onClick={() => setBulkAllocOpen(o => !o)}>
                    ⚙ allocation {bulkAllocOpen ? '▾' : '▸'}
                  </button>
                </>
              )}
              <button className="wp-bulkres-clear" title="Clear the selection"
                      onClick={() => { clearSel(); setBulkAllocOpen(false) }}>clear</button>
            </span>
          )}
          {/* The bulk-alloc editor, expanded below the bar (full-width, so the
              custom-budget inputs have room) — mirrors the per-row expansion. */}
          {canBulkAlloc && selCount > 0 && bulkAllocOpen && (
            <div className="wp-bulkalloc-editor">
              <span className="wp-bulkalloc-hint">
                Apply this allocation to the {selCount} selected model{selCount === 1 ? '' : 's'}:
              </span>
              <BulkAllocControl
                count={selCount}
                bulkKeys={selVisible}
                getModelBytes={sizeInfo}
                onApply={(spill, perModel) => { setBulkAllocOpen(false); onSetAllocMany(worker, selVisible, spill, perModel) }}
                onCancel={() => setBulkAllocOpen(false)}
              />
            </div>
          )}
          {/* Bulk pin: 📌 pin (or unpin) EVERY model designated to this worker in
              one settings-write. Sticky, so pinAll confirms first. Kept on the
              RIGHT of the filter bar. */}
          {worker.config && (onPinAll || onUnpinAll) && (worker.models || []).length > 0 && (
            <span className="wp-bulkpin wp-servtable-bulk">
              {onPinAll && (
                <button className="wp-bulkpin-btn" disabled={applying}
                        title={applying
                          ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                          : 'Pin ALL of this worker’s models — permanent attribution; each then blocks unassign until unpinned. Confirms first.'}
                        onClick={() => onPinAll(worker)}>📌 pin all</button>
              )}
              {onUnpinAll && (
                <button className="wp-bulkpin-btn wp-bulkunpin-btn" disabled={applying}
                        title={applying
                          ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                          : 'Unpin ALL of this worker’s models — the undo for Pin all; lets them be unassigned again.'}
                        onClick={() => onUnpinAll(worker)}>📌✕ unpin all</button>
              )}
              {/* Bulk 4-bit (operator ask 2026-07-26), in the same row as pin
                  all / unpin all. Acts ONLY on the ELIGIBLE set — GGUF,
                  CPU-only workers and already-quantized repos are skipped
                  rather than attempted, so the count in the label is the real
                  number of models that will change. Hidden entirely when this
                  worker has nothing eligible (e.g. op, no GPU). */}
              {moeCapableKeys.length > 0 && (
                <button className="wp-bulkpin-btn wp-bulkmoe-btn" disabled={applying}
                        title={`Restore AUTO expert-split handling on all ${moeCapableKeys.length} MoE-capable model(s) — clears any forced on/off and lets the derivation decide.`}
                        onClick={() => setMoeMany(null)}>
                  ⟲ MoE auto ({moeCapableKeys.length})
                </button>
              )}
              {bnbEligibleKeys.length > 0 && (
                <>
                  <button className="wp-bulkpin-btn wp-bulk4bit-btn" disabled={applying}
                          title={`Load all ${bnbEligibleKeys.length} eligible model(s) at bitsandbytes 4-bit (nf4). Each is then priced at ~30% of its fp16 size, so their Alloc modes re-derive — models too big for the GPU can become GPU-resident. Costs some quality.`}
                          onClick={() => setBnbMany(true)}>
                    ▦ 4-bit all ({bnbEligibleKeys.length})
                  </button>
                  <button className="wp-bulkpin-btn wp-bulkunpin-btn" disabled={applying}
                          title="Restore full precision on every model that currently has the 4-bit specialization — the undo for 4-bit all."
                          onClick={() => setBnbMany(false)}>
                    ▦✕ clear 4-bit
                  </button>
                </>
              )}
            </span>
          )}
        </div>
        {(worker.models || []).length === 0 ? (
          <span className="wp-none">— nothing assigned —</span>
        ) : (
          <div className="wp-servtable-scroll">
            <table className="wp-servtable-t">
              {/* Column widths live in the <colgroup> so a resize touches ONE
                  place, not every <td>. The checkbox column is fixed; each data
                  column gets its persisted px width, or auto when never resized.
                  The FROZEN Model column is the exception: it is FLUID (a CSS
                  clamp() on .wp-servtable-namecol) and carries no manual resize,
                  so any px width a previous layout stored for it is IGNORED here
                  — the stored value is simply never read, which leaves every
                  other column's remembered width intact (no storage-key bump). */}
              <colgroup>
                <col className="wp-servtable-selcol-col" />
                {servCols.map(c => {
                  const fluid = c.key === SERV_FROZEN_COL
                  return (
                    <col key={c.key}
                         className={fluid ? 'wp-servtable-namecol' : undefined}
                         style={!fluid && servWidths[c.key] ? { width: servWidths[c.key] } : undefined} />
                  )
                })}
              </colgroup>
              <thead>
                <tr>
                  {/* Select-all-in-current-filter checkbox (todo t12). Only the
                      visible/filtered rows are (de)selected — matches what the
                      operator can see. Indeterminate when a subset is picked.
                      FIXED left — not reorderable or resizable. */}
                  <th className="wp-servtable-h wp-servtable-selcol">
                    {canBulkResidency ? (
                      <input type="checkbox" className="wp-servtable-selall"
                             checked={allVisibleSelected}
                             ref={el => { if (el) el.indeterminate = someVisibleSelected }}
                             disabled={servVisibleKeys.length === 0}
                             onChange={selectAllVisible}
                             title="Select all models in the current filter (for a bulk residency change)" />
                    ) : null}
                  </th>
                  {/* Every other column renders from the ONE ordered model, so a
                      drag reorders it anywhere. Sortable columns keep their click-
                      to-sort + arrow; all columns get a drag-reorder grip and a
                      right-edge resize handle. */}
                  {servCols.map(c => {
                    // The Model column is FROZEN left (sticky during horizontal
                    // scroll): it is neither a drag SOURCE nor a drop TARGET, so
                    // it can never leave slot 1 and nothing can land before it.
                    // Its sort click stays live; its RESIZE GRIP is gone (2026-
                    // 07-28) — the column is fluid (clamp) and mid-ellipsised, so
                    // a manual px width would fight the clamp.
                    const frozen = c.key === SERV_FROZEN_COL
                    // Compact mode has nothing to reorder into and nothing to
                    // freeze against (three columns, no horizontal scroll), so
                    // drag/resize/sticky all go inert.
                    const dragOk = !frozen && !servCompact
                    return (
                    <th key={c.key}
                        className={`wp-col-th${c.sortable ? ' wp-lt-sortable' : ' wp-servtable-h'}${c.num ? ' wp-lt-num' : ''}${frozen ? ' wp-servtable-stickycol wp-servtable-stickyname' : ''}${dragOk && servDragOver === c.key && servDragCol && servDragCol !== c.key ? ' wp-col-dragover' : ''}${servDragCol === c.key ? ' wp-col-dragging' : ''}`}
                        draggable={dragOk}
                        onDragStart={!dragOk ? undefined : (e) => { setServDragCol(c.key); e.dataTransfer.effectAllowed = 'move' }}
                        onDragOver={!dragOk ? undefined : (e) => { if (servDragCol && servDragCol !== c.key) { e.preventDefault(); setServDragOver(c.key) } }}
                        onDrop={!dragOk ? undefined : (e) => { e.preventDefault(); if (servDragCol && servDragCol !== c.key) servMoveCol(servDragCol, c.key); setServDragCol(null); setServDragOver(null) }}
                        onDragEnd={!dragOk ? undefined : () => { setServDragCol(null); setServDragOver(null) }}
                        title={frozen ? 'Click to sort · this column is pinned left (frozen while you scroll sideways) · its width is fluid — long names are shortened in the MIDDLE, hover for the full key'
                          : !dragOk ? (c.sortable ? 'Click to sort' : undefined)
                          : c.sortable ? 'Click to sort · drag to reorder · drag the right edge to resize' : 'Drag to reorder · drag the right edge to resize'}
                        onClick={c.sortable ? () => toggleServSort(c.key) : undefined}>
                      {c.label}{c.sortable ? servArrow(c.key) : ''}
                      {/* Resize grip: a pointer-drag that never triggers the sort
                          click or the column drag (both are stopped here). NOT
                          rendered for the fluid Model column, nor in compact mode
                          (where there is no spare width to hand out). */}
                      {dragOk && (
                        <span className="wp-col-resize" draggable={false}
                              onDragStart={(e) => e.preventDefault()}
                              onClick={(e) => e.stopPropagation()}
                              onPointerDown={(e) => startServResize(e, c.key)} />
                      )}
                    </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {/* Assigned models exist, but the search/task filter hid them
                    ALL — a DISTINCT empty state, never the "nothing assigned"
                    message (which would misreport the worker as empty). */}
                {servRows.length === 0 && (
                  <tr><td colSpan={SERV_COLSPAN} className="wp-servtable-nomatch">
                    no match — clear the filter to see all {(worker.models || []).length} assigned
                  </td></tr>
                )}
                {servRows.map(({ key, m, d }) => {
                  const state = d.state
                  const isSel = servSel.has(key)
                  // Operator model BLOCK (global): this model is removed from the
                  // serving pool everywhere. The designation row still renders
                  // (inert + labeled) — block does not unassign and outranks pin.
                  const isBlocked = !!(blockedKeys && blockedKeys.has(key))
                  // t49: this model's own requirement (GGUF effective quant),
                  // for the AllocControl VRAM/RAM slider coupling. GGUF-only —
                  // explicit budgets (what the coupling lives inside) are
                  // already a GGUF-exclusive concept, so gate the denominator
                  // the same way rather than offering a misleading coupling on
                  // a transformers/comfy model's dir-size total.
                  const need = (() => {
                    const si = sizeInfo(key)
                    return (si && si.isGguf && si.bytes != null)
                      ? { bytes: si.bytes, gib: si.bytes / WP_GIB } : null
                  })()
                  // `compact` rides the cell context so a def can tighten its own
                  // rendering (only Model does today: a smaller midTrunc budget).
                  const cx = { key, m, d, isSel, isBlocked, need, compact: servCompact }
                  const drawerOpen = servCompact && servDrawer === key
                  return (
                    <Fragment key={key}>
                      <tr className={`wp-servtable-row wp-st-${state}${isSel ? ' wp-servtable-sel' : ''}${isBlocked ? ' wp-servtable-blocked' : ''}${drawerOpen ? ' wp-servtable-rowopen' : ''}`}
                          onClick={servRowTap(key)}>
                        {/* Row select checkbox (todo t12) — FIXED left, keyed by
                            model_key; survives the 10s refresh. Not reorderable. */}
                        <td className="wp-servtable-selcol">
                          {canBulkResidency ? (
                            <input type="checkbox" className="wp-servtable-selrow"
                                   checked={isSel}
                                   onChange={() => toggleSel(key)}
                                   title="Select this model for a bulk residency change" />
                          ) : null}
                        </td>
                        {/* Every other cell renders from the SAME ordered column
                            model as the header, so a reorder/resize moves the whole
                            column (header + body) together. */}
                        {servCols.map(c => (
                          <td key={c.key}
                              className={`${c.cls || ''}${c.key === SERV_FROZEN_COL ? ' wp-servtable-stickycol wp-servtable-stickyname' : ''}`.trim() || undefined}
                              title={c.title ? c.title(cx) : undefined}>
                            {c.render(cx)}
                          </td>
                        ))}
                      </tr>
                      {/* COMPACT DRAWER (operator ask 2026-07-28) — the SAME
                          colspanned expansion-row mechanism the residency picker
                          uses, one row at a time. It carries the FULL untruncated
                          model name, then every column that left the table as a
                          labeled chip, then the actions as a touch-target row.
                          Each chip's body is the column def's OWN render(cx): the
                          Alloc menu, residency button, 📌 pin, 4-bit and MoE
                          checkboxes are literally the same controls as desktop,
                          not a second implementation. */}
                      {drawerOpen && (
                        <tr className="wp-servtable-expand wp-servtable-drawer">
                          <td colSpan={SERV_COLSPAN}>
                            <div className="wp-servdrawer">
                              <div className="wp-servdrawer-name" title={key}>{nameFor(key)}</div>
                              {/* The model KEY when it differs from the display
                                  name — the key is what every API call and log
                                  line uses, so it must be readable somewhere. */}
                              {nameFor(key) !== key && (
                                <div className="wp-servdrawer-key">{key}</div>
                              )}
                              <div className="wp-servdrawer-chips">
                                {servDrawerCols.map(c => (
                                  <div key={c.key} className="wp-servdrawer-chip"
                                       title={c.title ? c.title(cx) : undefined}>
                                    <span className="wp-servdrawer-chip-label">{c.label}</span>
                                    <span className="wp-servdrawer-chip-body">{c.render(cx)}</span>
                                  </div>
                                ))}
                              </div>
                              {/* Actions (▶ activate / ⏏ free / × unassign / ⛔
                                  block) — the same def render, laid out as a
                                  full-width row of ≥40px touch targets. */}
                              <div className="wp-servdrawer-actions">
                                {servActionsCol.render(cx)}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {/* Full-width expansion row: the residency picker (the
                          alloc editor is now the in-place, click-site
                          AllocModeMenu — no lower-row expansion, operator ask
                          2026-07-24). Rendered as its own colspanned row so it
                          never disturbs the table's column layout. */}
                      {resMenu === key && worker.config && onSetResidency && (
                        <tr className="wp-servtable-expand">
                          <td colSpan={SERV_COLSPAN}>
                            <ResidencyMenu mode={worker.config.residency?.[key] === 'static' ? 'static' : 'on-demand'}
                                           onClose={() => setResMenu(null)}
                                           onPick={(next) => {
                                             setResMenu(null)
                                             onSetResidency(worker, key, next)
                                           }} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {!showLoad ? (
        <button className="wp-load-toggle" onClick={() => setShowLoad(true)}
                title="Load another model onto this worker">
          ＋ load a model
        </button>
      ) : (
        <WorkerLoadTable
          models={assignable}
          allocation={allocation || {}}
          workerId={worker.id}
          worker={worker}
          onAllocate={async (modelKeys, breaker) => {
            await onAllocateMany(worker, modelKeys, breaker)
            setShowLoad(false)
          }}
          onCancel={() => setShowLoad(false)}
        />
      )}
    </div>
  )
}

// Group-assign: pick a model, tick a set of workers (with select-all), dedicate
// it to all of them at once. Workers that already hold the model are shown
// checked+locked ("already here") and never re-fired — that IS the no-double-
// booking guard. Offline/blocked workers can't take an assignment, so they're
// locked out too. Placed at the fleet level, above the per-worker rows.
function GroupAssignPanel({ models, workers, onGroupAssign }) {
  const [open, setOpen] = useSessionState('hugpy.sess.wp.group.open', false)
  const [pick, setPick] = useState('')
  const [sel, setSel] = useState(() => new Set())
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState(null)

  const keyOf = (m) => m.model_key ?? m.key

  // Per-worker eligibility for the picked model.
  const rows = useMemo(() => (workers || []).map(w => {
    const already = pick ? (w.models || []).includes(pick) : false
    const offline = w.status === 'offline' || w.admission === 'blocked' || w.admission === 'pending'
    return {
      w,
      already,
      offline,
      eligible: !!pick && !already && !offline,
      reason: already ? '✓ already here' : offline ? `(${w.admission === 'blocked' ? 'blocked' : w.status === 'offline' ? 'offline' : 'not admitted'})` : '',
    }
  }), [workers, pick])

  const eligibleIds = useMemo(() => rows.filter(r => r.eligible).map(r => r.w.id), [rows])
  const allSelected = eligibleIds.length > 0 && eligibleIds.every(id => sel.has(id))
  const chosen = useMemo(() => rows.filter(r => r.eligible && sel.has(r.w.id)).map(r => r.w), [rows, sel])

  const toggle = (id) => setSel(prev => {
    const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n
  })
  const toggleAll = () => setSel(allSelected ? new Set() : new Set(eligibleIds))

  const dedicate = async () => {
    if (!pick || chosen.length === 0) return
    setBusy(true); setResults(null)
    try {
      const out = await onGroupAssign(pick, chosen, false)
      setResults(out)
      // Drop the successes from the selection so a re-run only retries failures.
      const failed = new Set(out.filter(r => !r.ok).map(r => r.id))
      setSel(prev => new Set([...prev].filter(id => failed.has(id))))
    } finally { setBusy(false) }
  }

  return (
    <div className="wp-group">
      {!open ? (
        <button className="wp-group-toggle" onClick={() => setOpen(true)}
                title="Dedicate one model to several workers at once">
          ⧉ Assign a model to a group of workers…
        </button>
      ) : (
        <div className="wp-group-body">
          <div className="wp-group-head">
            <strong>Group assign</strong>
            <button className="wp-group-x" onClick={() => { setOpen(false); setPick(''); setSel(new Set()); setResults(null) }}>×</button>
          </div>
          <ModelPicker
            models={models}
            value={pick}
            onPick={(k) => { setPick(k); setSel(new Set()); setResults(null) }}
            placeholder="1. Pick a model to dedicate…"
          />
          {pick && (
            <>
              <div className="wp-group-selall">
                <label className="wp-group-check">
                  <input type="checkbox" checked={allSelected} onChange={toggleAll}
                         disabled={eligibleIds.length === 0} />
                  <span>Select all eligible ({eligibleIds.length})</span>
                </label>
              </div>
              <div className="wp-group-list">
                {rows.length === 0 && <div className="wp-none">No workers in the pool.</div>}
                {rows.map(({ w, already, eligible, reason }) => (
                  <label key={w.id}
                         className={`wp-group-row${eligible ? '' : ' wp-group-locked'}`}
                         title={eligible ? '' : reason}>
                    <input type="checkbox"
                           checked={already || sel.has(w.id)}
                           disabled={!eligible}
                           onChange={() => toggle(w.id)} />
                    <span className="wp-group-wname">{w.name}</span>
                    {reason && <span className="wp-group-note">{reason}</span>}
                    {w.disk && typeof w.disk.free_bytes === 'number' && (
                      <span className="wp-group-disk">{fmtBytes(w.disk.free_bytes)} free</span>
                    )}
                  </label>
                ))}
              </div>
              <div className="wp-group-actions">
                <button className="wp-group-go" disabled={busy || chosen.length === 0}
                        onClick={dedicate}
                        title="Preflight (VRAM+RAM+disk) then assign to each selected worker">
                  {busy ? 'Dedicating…' : `Dedicate to ${chosen.length} worker${chosen.length === 1 ? '' : 's'}`}
                </button>
              </div>
              {results && (
                <div className="wp-group-results">
                  {results.length === 0 && <span className="wp-group-ok">Nothing to do — all selected workers already had it.</span>}
                  {results.map(r => (
                    <div key={r.id} className={r.ok ? 'wp-group-ok' : 'wp-group-bad'}>
                      {r.ok ? '✓' : '✗'} {r.name}{r.note ? ` — ${r.note}` : ''}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function WorkersPanel({ models = [], embedded = false }) {
  const [workers, setWorkers] = useState([])
  const [error, setError]     = useState(null)
  const [open, setOpen]       = useSessionState('hugpy.sess.wp.open', false)  // session-sticky expansion
  const [form, setForm]       = useState({ name: '', url: '', models: '' })
  const [busy, setBusy]       = useState(false)
  const [tokens, setTokens]   = useState([])
  const [central, setCentral] = useState(null)     // /llm/slots — central's own row
  // Central-as-worker load controls (replaced the separate Model Slots panel).
  const [showCentralLoad, setShowCentralLoad] = useState(false)
  const [centralPick, setCentralPick] = useState('')
  const [centralBusy, setCentralBusy] = useState(false)
  const [newToken, setNewToken] = useState(null)   // freshly-minted plaintext, shown once
  // Transient per-worker "restarting…" flag. A restart re-execs the agent, so
  // the row goes offline→online on its own via the 10s load() poll; this just
  // reflects the click until the poll (or a fallback timeout) retires it.
  const [restarting, setRestarting] = useState({})  // worker.id -> true
  // Same transient for ⬆ Update: the worker pip-installs then restarts itself,
  // so it wears its own flag (a longer fallback — pip is slower than a re-exec).
  const [updating, setUpdating] = useState({})      // worker.id -> true

  // ── Per-worker TABS (operator, 2026-07-26) ────────────────────────────────
  // The pool used to render every worker's card stacked down the page, which
  // got long fast once the serving list stopped being a 260px box (b1b2222).
  // One card at a time instead: a tab strip picks the worker, and the selection
  // persists per browser like the other console view state.
  const [activeWorker, setActiveWorker] = useSessionState('hugpy.sess.wp.tab', '')

  // Resolve the tab selection defensively: a worker can be removed, renamed or
  // simply not be in the pool this session, and a stale id in sessionStorage
  // must never render an empty panel. Falls back to the first worker.
  const selectedWorkerId = useMemo(() => {
    if (activeWorker && workers.some(w => w.id === activeWorker)) return activeWorker
    return workers[0]?.id || ''
  }, [activeWorker, workers])
  // With a single worker there is no strip, so nothing is filtered out.
  const shownWorkers = useMemo(
    () => (workers.length > 1 ? workers.filter(w => w.id === selectedWorkerId) : workers),
    [workers, selectedWorkerId])
  // Operator model BLOCK (global serving-pool primitive): the set of model_keys
  // blocked everywhere. Seeded from the /models feed (`blocked` field on each
  // row) and kept as local state so a toggle reflects instantly without waiting
  // on the parent's models refetch (App fetches /models once). See toggleBlock.
  const propBlocked = useMemo(
    () => new Set((models || [])
      .filter(m => m && m.blocked)
      .map(m => m.model_key ?? m.key)),
    [models])
  const [blockedKeys, setBlockedKeys] = useState(propBlocked)
  useEffect(() => { setBlockedKeys(propBlocked) }, [propBlocked])
  // Setup chrome (install command, tokens, manual add) is collapsed by default so
  // the live worker list is the focus — but auto-opens when there are no workers
  // yet, since the install command is the only way to add the first one.
  const [setupOpen, setSetupOpen] = useSessionState('hugpy.sess.wp.setupOpen', false)
  // Apply-blip guard (pin / residency / slot-count): a successful /config POST
  // replies {restarting: true} and the agent re-execs ~0.5s later, so its
  // reported config stays stale for a heartbeat or two and a re-click during
  // the blip hits a dead socket (surfaced as a bare 502 — "pin is broken").
  // Track an "applying" window per worker: disable that worker's config
  // controls and soften errors until a refresh shows the change (or ~10s).
  const [applying, setApplying] = useState({})  // worker.id -> {t, kind, model?, value}
  const markApplying = useCallback((workerId, exp) => {
    setApplying(a => ({ ...a, [workerId]: { t: Date.now(), ...exp } }))
  }, [])
  useEffect(() => {
    if (Object.keys(applying).length === 0) return undefined
    const met = (w, exp) => {
      if (!w || !w.config) return false
      if (exp.kind === 'residency') {
        // Two-tier policy: static is the only stored override; everything
        // else (no entry / legacy values) reads as the on-demand default.
        const cur = w.config.residency?.[exp.model] ?? null
        return exp.value === 'static' ? cur === 'static' : cur !== 'static'
      }
      if (exp.kind === 'pinned') return !!w.config.pinned?.[exp.model] === !!exp.value
      if (exp.kind === 'pin_all') {
        // Bulk pin/unpin: cleared once every affected model matches the target.
        const p = w.config.pinned || {}
        return (exp.models || []).every(mk => !!p[mk] === !!exp.value)
      }
      if (exp.kind === 'residency_all') {
        // Bulk residency: cleared once every selected model's stored tier
        // matches the target (static stored verbatim; on-demand = no entry).
        const res = w.config.residency || {}
        return (exp.models || []).every(mk =>
          exp.value === 'static' ? res[mk] === 'static' : res[mk] !== 'static')
      }
      if (exp.kind === 'slot_count') return w.config.slot_count === exp.value
      return false
    }
    const sweep = () => setApplying(a => {
      let changed = false
      const n = { ...a }
      for (const [wid, exp] of Object.entries(a)) {
        const w = workers.find(x => x.id === wid)
        if (Date.now() - exp.t > 10_000 || met(w, exp)) { delete n[wid]; changed = true }
      }
      return changed ? n : a
    })
    sweep()                                   // a refresh may already show it
    const t = setTimeout(sweep, 10_500)       // …or age the window out
    return () => clearTimeout(t)
  }, [workers, applying])

  // Returns a promise so the caller can WAIT before scheduling the next poll
  // (see the self-scheduling effect below). Without this the backpressure loop
  // would resolve instantly and degenerate back into setInterval.
  const load = useCallback(() => {
    const pWorkers = fetchJson('/api/llm/workers')
      .then(data => {
        // A POLL MUST NEVER DESTROY GOOD DATA. This used to be
        //   setWorkers(Array.isArray(data) ? data : [])
        // so ANY non-array reply — an error object, a partial, a body returned
        // while a worker agent is mid-re-exec — blanked the whole pool list.
        // The operator saw pools "constantly disappearing simply because they
        // are being called", from a strictly sequential one-model-at-a-time
        // script: not load, just a single degraded reply landing between good
        // ones. Keep the last known roster and surface the problem instead;
        // stale-but-labelled beats empty-and-silent.
        if (Array.isArray(data)) { setWorkers(data); setError(null) }
        else setError('worker list unavailable (kept the last known roster)')
      })
      .catch(e => setError(e.message))
    // Central-as-worker: its compute (the local slot pool) renders as a
    // footer card BELOW every remote worker row, same anatomy as a remote
    // worker — it's secondary context (this console's own host), not the
    // fleet being managed.
    const pSlots = fetchJson('/api/llm/slots')
      // Same rule as the worker roster above: a falsy/degraded reply keeps the
      // last known card rather than blanking it. Only a well-formed object
      // replaces what is on screen.
      .then(d => { if (d && typeof d === 'object') setCentral(d) })
      .catch(() => {})
    // Both settle before the next poll is scheduled — neither can pile up.
    return Promise.all([pWorkers, pSlots])
  }, [])

  const loadTokens = useCallback(() => {
    fetchJson('/api/llm/enroll-tokens')
      .then(data => setTokens(Array.isArray(data) ? data : []))
      .catch(() => {})   // tokens are a secondary panel; don't surface as a registry error
  }, [])

  useEffect(() => {
    // SELF-SCHEDULING, NOT setInterval — this is backpressure, not polish.
    //
    // setInterval(load, 10_000) fired every 10s whether or not the previous
    // call had returned. With /llm/workers at 31s (measured: list_workers()
    // alone is 11.8s cold for THREE workers) that means 3-4 copies in flight
    // permanently, each holding one of central's 24 gunicorn slots. The console
    // then starves itself: the browser log showed every /api/llm/workers line
    // with NO status and NO timing while every other endpoint returned.
    //
    // That is how ONE slow backend call takes down the whole view — it
    // MULTIPLIES instead of backing off. Waiting for the response before
    // scheduling the next makes the loop self-limiting: a slow endpoint
    // lowers its own poll rate to at most one in flight, so the panel degrades
    // to "less fresh" instead of "never loads", and it can never be the reason
    // other panels starve.
    //
    // Fixing the server side (making /llm/workers fast) is necessary but not
    // sufficient — without this, any future slow endpoint reproduces it.
    let cancelled = false
    let timer = null
    const tick = () => {
      Promise.resolve(load()).finally(() => {
        if (!cancelled) timer = setTimeout(tick, 10_000)
      })
    }
    tick()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [load])

  useEffect(() => { loadTokens() }, [loadTokens])

  // Load into a free central slot (409 with a clear reason when none is free)
  // — the same backend the retired Model Slots panel drove.
  const centralLoad = useCallback(async (modelKey) => {
    setCentralBusy(true)
    try {
      const r = await fetchJson('/api/llm/slots/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.loaded === false) alert(`Not loaded: ${r.reason || 'no free slot'}`)
      setShowCentralLoad(false); setCentralPick('')
      load()
    } catch (err) { alert(`Load failed: ${err.message}`) }
    finally { setCentralBusy(false) }
  }, [load])

  const centralUnload = useCallback(async (control) => {
    try {
      await fetchJson('/api/llm/slots/unload', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ control }),
      })
      load()
    } catch (err) { alert(`Unload failed: ${err.message}`) }
  }, [load])

  const issueToken = useCallback(async () => {
    const label = prompt('Label for this enrollment token (e.g. gpu-box-2):', '')
    if (label === null) return
    try {
      const r = await fetchJson('/api/llm/enroll-tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      })
      setNewToken(r)   // contains the one-time plaintext `token`
      loadTokens()
    } catch (err) { alert(`Could not issue token: ${err.message}`) }
  }, [loadTokens])

  const revokeToken = useCallback(async (tok) => {
    if (!confirm(`Revoke token "${tok.label || tok.id}"? Workers using it are refused and their agents stop.`)) return
    try {
      await fetchJson(`/api/llm/enroll-tokens/${encodeURIComponent(tok.id)}`, { method: 'DELETE' })
      loadTokens()
    } catch (err) { alert(`Revoke failed: ${err.message}`) }
  }, [loadTokens])

  const register = useCallback(async (e) => {
    e.preventDefault()
    if (!form.name.trim()) return   // url is optional; central uses source IP
    setBusy(true)
    try {
      const body = {
        name: form.name.trim(),
        models: form.models.split(',').map(s => s.trim()).filter(Boolean),
      }
      if (form.url.trim()) body.url = form.url.trim()
      await fetchJson('/api/llm/workers/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setForm({ name: '', url: '', models: '' })
      load()
    } catch (err) {
      alert(`Could not register worker: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }, [form, load])

  const assign = useCallback(async (worker, modelKey, spill) => {
    try {
      const body = { model_key: modelKey }
      if (spill) body.spill = spill
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      await load()
    } catch (err) { alert(`Assign failed: ${err.message}`) }
  }, [load])

  // Guarded place-on-worker: VRAM+RAM preflight → assign → background warm.
  // The backend refuses (409) a model that won't fit the worker's free VRAM+RAM
  // (fetchJson throws with that reason); on a pass it returns immediately and the
  // worker warms the model — residency shows up via the next heartbeat.
  const loadModel = useCallback(async (worker, modelKey, spill, force) => {
    try {
      const body = { model_key: modelKey }
      if (spill) body.spill = spill
      if (force) body.force = true
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      // Fits, but only by spilling to CPU (RAM+VRAM ok, VRAM alone not) — warn.
      if (r && r.preflight && r.preflight.gpu_resident === false && r.preflight.reason) {
        alert(`Placed — note: ${r.preflight.reason}`)
      }
      load()
    } catch (err) {
      // 409 preflight refuse arrives here carrying the clear "won't fit …" reason.
      // Offer an override for the rare case the operator knows better.
      if (confirm(`${err.message}\n\nForce-load anyway? (may OOM the worker)`)) {
        loadModel(worker, modelKey, spill, true)
      }
    }
  }, [load])

  // Group-assign: dedicate ONE model to a SET of workers in a single action.
  // Each worker goes through the same /load preflight (VRAM+RAM+disk); a worker
  // that already has the model is skipped upstream in the UI (no double-booking)
  // and defended here too. Returns per-worker outcomes for the summary.
  const groupAssign = useCallback(async (modelKey, workerList, force) => {
    const targets = workerList.filter(w => !((w.models || []).includes(modelKey)))
    const settled = await Promise.all(targets.map(async (w) => {
      try {
        const body = { model_key: modelKey }
        if (force) body.force = true
        const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(w.id)}/load`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        const spilled = r && r.preflight && r.preflight.gpu_resident === false
        return { id: w.id, name: w.name, ok: true, note: spilled ? 'CPU-spilled' : '' }
      } catch (err) {
        return { id: w.id, name: w.name, ok: false, note: err.message }
      }
    }))
    load()
    return settled
  }, [load])

  // Per-worker bulk allocate (worker <- group of models). Each model via the
  // same /load preflight; anti-duplicate is enforced in the table UI (the
  // breaker flag rides along only for the summary). Reports per-model outcome.
  const allocateMany = useCallback(async (worker, modelKeys, breaker) => {
    const settled = await Promise.all((modelKeys || []).map(async (mk) => {
      try {
        const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/load`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_key: mk }),
        })
        const spilled = r && r.preflight && r.preflight.gpu_resident === false
        return { mk, ok: true, note: spilled ? 'CPU-spilled' : '' }
      } catch (err) { return { mk, ok: false, note: err.message } }
    }))
    load()
    const okN = settled.filter(r => r.ok).length
    const failed = settled.filter(r => !r.ok)
    let msg = `Allocated ${okN}/${settled.length} to ${worker.name}${breaker ? ' (duplicate-allocation breaker was ON)' : ''}.`
    if (failed.length) msg += `\n\nRefused:\n` + failed.map(r => `  • ${r.mk} — ${r.note}`).join('\n')
    if (failed.length || okN !== settled.length) alert(msg)
  }, [load])

  const unassign = useCallback(async (worker, modelKey) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unassign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      load()
    } catch (err) { alert(`Unassign failed: ${err.message}`) }
  }, [load])

  // Operator model BLOCK / UNBLOCK (global): removes a model from the serving
  // pool everywhere (never routed / assigned / warmed / a fallback default) or
  // returns it. Optimistically updates blockedKeys so the ⛔ chip flips at once;
  // reverts on error. Block does NOT unassign — the designation row stays (inert)
  // and pin is unaffected (block outranks pin at routing time). Operator token is
  // merged by hugpyFetch, same as every mutation here.
  const toggleBlock = useCallback(async (modelKey, block) => {
    if (block && !confirm(
      `Block "${modelKey}" from the serving pool?\n\n` +
      `It will no longer be routed to, assigned, warmed, or used as a fallback ` +
      `default anywhere — files stay on disk and existing designations stay ` +
      `recorded (inert). Reversible.`)) return
    setBlockedKeys(prev => {
      const next = new Set(prev)
      if (block) next.add(modelKey); else next.delete(modelKey)
      return next
    })
    try {
      await fetchJson(
        `/api/llm/models/${encodeURIComponent(modelKey)}/${block ? 'block' : 'unblock'}`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}) })
      load()
    } catch (err) {
      // revert the optimistic flip on failure
      setBlockedKeys(prev => {
        const next = new Set(prev)
        if (block) next.delete(modelKey); else next.add(modelKey)
        return next
      })
      alert(`${block ? 'Block' : 'Unblock'} failed: ${err.message}`)
    }
  }, [load])

  // Free GPU VRAM: evict a loaded model from the worker's cache. It stays
  // assigned — only the live VRAM is reclaimed (reloads on next request).
  const freeModel = useCallback(async (worker, modelKey) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.ok === false) alert(`Free failed: ${r.error || 'unknown error'}`)
      load()
    } catch (err) { alert(`Free failed: ${err.message}`) }
  }, [load])

  // Evict an attributed model from a worker's GPU process registry: POST the
  // operator-gated evict verb (the operator token is merged by hugpyFetch, same
  // as every other mutation here). Frees the model's VRAM/RAM by killing/unloading
  // its owning pid. Idempotent — returns evicted:false + a reason when the model
  // isn't resident. Surfaces the reason, then refetches so the freed VRAM and the
  // removed registry row reflect immediately. Returns the result so the calling
  // row can clear its in-flight state.
  const evictModel = useCallback(async (worker, modelKey) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/evict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.ok === false) {
        alert(`Evict failed: ${r.error || r.reason || 'unknown error'}`)
      } else if (r && r.evicted === false) {
        alert(`Not evicted: ${r.reason || 'model is not resident on this worker'}`)
      }
      load()
      return r
    } catch (err) { alert(`Evict failed: ${err.message}`) }
  }, [load])

  const freeAll = useCallback(async (worker) => {
    if (!confirm(`Unload all models from ${worker.name}'s GPU? They stay assigned and reload on demand.`)) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true }),
      })
      if (r && r.ok === false) alert(`Free failed: ${r.error || 'unknown error'}`)
      load()
    } catch (err) { alert(`Free failed: ${err.message}`) }
  }, [load])

  // Free host RAM: ask the agent to return reclaimable process/arena memory to
  // the OS. NON-destructive — loaded models stay resident (unlike freeAll), so
  // there's no scary confirm. Surfaces how much was reclaimed.
  const freeRam = useCallback(async (worker) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/free-ram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (r && r.ok === false) { alert(`Free RAM failed: ${r.error || 'unknown error'}`); return }
      const freed = r && typeof r.ram_freed === 'number' ? r.ram_freed : null
      if (freed && freed > 0) {
        alert(`Freed ${(freed / 1073741824).toFixed(1)} GiB RAM on ${worker.name}`)
      } else {
        alert(`No reclaimable RAM on ${worker.name} right now.`)
      }
      load()
    } catch (err) { alert(`Free RAM failed: ${err.message}`) }
  }, [load])

  // Restart the worker's agent: drops all loaded models and re-execs the agent
  // process. During re-exec the relay may answer 503 {error:{code:"AgentRestarting"}}
  // — that's the expected transient, not a failure. The row flips offline→online
  // on its own via the 10s load() poll; we flag "restarting…" briefly meanwhile.
  const restart = useCallback(async (worker) => {
    if (!confirm(`Restart ${worker.name}'s worker agent? It drops all loaded models and re-execs the agent.`)) return
    setRestarting(s => ({ ...s, [worker.id]: true }))
    const clear = () => setRestarting(s => { const n = { ...s }; delete n[worker.id]; return n })
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    } catch (err) {
      // 503 AgentRestarting is the agent re-execing — expected, treat as success.
      if (!/\b503\b|AgentRestarting/i.test(err.message || '')) {
        alert(`Restart failed: ${err.message}`)
        clear()
        return
      }
    }
    // Let the offline→online poll retire the indicator; clear on a fallback timeout.
    setTimeout(clear, 12_000)
    load()
  }, [load])

  // Converge a worker onto central's required version NOW (instead of waiting
  // for its next heartbeat's self-update). The worker pip-installs --no-deps and
  // re-execs, so the SAME 503 {error:{code:"AgentRestarting"}} transient applies
  // — just with a longer fallback timeout, because pip install runs first. An
  // empty body means "central's required version" server-side.
  const update = useCallback(async (worker) => {
    if (!confirm(`Update ${worker.name} to central's required version (${worker.required_pkg_version || 'unknown'})? The worker pip-installs and restarts itself.`)) return
    setUpdating(s => ({ ...s, [worker.id]: true }))
    const clear = () => setUpdating(s => { const n = { ...s }; delete n[worker.id]; return n })
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    } catch (err) {
      // 503 AgentRestarting is the agent re-execing after the install — expected.
      if (!/\b503\b|AgentRestarting/i.test(err.message || '')) {
        alert(`Update failed: ${err.message}`)
        clear()
        return
      }
    }
    setTimeout(clear, 20_000)
    load()
  }, [load])

  // Reaper (tiers-v2 slice 4): preview first, then reclaim on confirm. The
  // worker re-proves every guard at delete time; pinned/assigned/loaded/comfy
  // are never reaped.
  const reap = useCallback(async (worker) => {
    try {
      const prev = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/reap`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: true }),
      })
      const items = (prev && prev.reclaimable) || []
      if (items.length === 0) {
        alert(`Nothing to reclaim on ${worker.name}. Everything on disk is assigned, loaded, or 📌pinned.`)
        return
      }
      // Honest binary units (ruling 5): the variable is named `gib`, so divide
      // by 2^30 (GiB) — matching fmtBytes and the rest of the panel — not the
      // decimal 1e9 it used before (which computed GB but labeled GiB).
      const gib = (prev.reclaimable_bytes || 0) / 1073741824
      const list = items.slice(0, 12).map(r => `  • ${r.model_key} (${(r.bytes / 1073741824).toFixed(1)} GiB)`).join('\n')
      const more = items.length > 12 ? `\n  …and ${items.length - 12} more` : ''
      if (!confirm(`Reclaim ${gib.toFixed(1)} GiB from ${worker.name} by deleting ${items.length} unassigned model(s)?\n\n${list}${more}\n\nProtected (assigned / loaded / 🔒static) files are left untouched. 📌 Pin does not protect files — it keeps the allocation/routing only.`)) return
      const res = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/reap`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true }),
      })
      const freed = ((res && res.freed_bytes) || 0) / 1073741824
      const failed = ((res && res.results) || []).filter(r => !r.ok)
      let msg = `Reclaimed ${freed.toFixed(1)} GiB from ${worker.name}.`
      if (failed.length) msg += `\n\nSkipped ${failed.length}:\n` + failed.map(r => `  • ${r.model_key} — ${r.reason}`).join('\n')
      alert(msg)
      load()
    } catch (err) { alert(`Reclaim failed: ${err.message}`) }
  }, [load])

  // Storage over-budget: approve the LRU eviction proposal central computed.
  // Human-in-the-loop — this fires ONLY on the explicit click; nothing is
  // auto-deleted. Central re-computes + intersects the proposal at approval
  // time (drops anything since loaded/assigned/static — NOT pinned; pin no
  // longer protects files as of 2026-07-17), then the worker re-proves every
  // guard per model before deleting. Prefers the dedicated
  // /reap-approve route (adds the central intersection guard); falls back to
  // /reap — which already re-guards worker-side — if it isn't deployed yet.
  const approveEvictions = useCallback(async (worker) => {
    const proposed = (worker.storage && worker.storage.proposed_evictions) || []
    if (proposed.length === 0) {
      alert(`Nothing proposed for eviction on ${worker.name} — it isn't over budget.`)
      return
    }
    const keys = proposed.map(p => p.model_key)
    const freed = ((worker.storage && worker.storage.proposed_free_bytes) || 0) / 1e9
    const list = proposed.slice(0, 12).map(p => `  • ${p.model_key} (${(p.bytes / 1e9).toFixed(1)} GB)`).join('\n')
    const more = proposed.length > 12 ? `\n  …and ${proposed.length - 12} more` : ''
    if (!confirm(`Approve eviction on ${worker.name}?\n\nFrees ~${freed.toFixed(1)} GB by deleting ${keys.length} cold, unprotected model(s):\n\n${list}${more}\n\nLoaded / 🔒static / assigned files are never touched. 📌 Pinned files ARE eligible — pin keeps the allocation, not the bytes (they re-pull on next call). Central re-checks this list and the worker re-proves each model before deleting.`)) return
    const url = `/api/llm/workers/${encodeURIComponent(worker.id)}/reap-approve`
    const body = JSON.stringify({ model_keys: keys })
    const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }
    try {
      let res
      try {
        res = await fetchJson(url, opts)
      } catch (err) {
        // NEVER fall back to the un-intersected /reap: that path relays the
        // client's keys WITHOUT central's re-derive+intersect second guard, so a
        // stale proposal could widen the delete. /reap-approve ships in 0.1.137;
        // if it isn't live, refuse rather than weaken the guarantee.
        if (/\b404\b|not found/i.test(err.message || '')) {
          throw new Error('reap-approve is not available on this central/worker (needs 0.1.137+). Refusing to fall back to the un-guarded reaper.')
        }
        throw err
      }
      const freedGb = ((res && res.freed_bytes) || 0) / 1e9
      const failed = ((res && res.results) || []).filter(r => !r.ok)
      let msg = `Freed ${freedGb.toFixed(1)} GB from ${worker.name}.`
      if (res && res.note) msg += `\n\n${res.note}`
      if (failed.length) msg += `\n\nSkipped ${failed.length}:\n` + failed.map(r => `  • ${r.model_key} — ${r.reason}`).join('\n')
      alert(msg)
      load()
    } catch (err) { alert(`Eviction failed: ${err.message}`) }
  }, [load])

  const remove = useCallback(async (worker) => {
    if (!confirm(`Remove worker ${worker.name} from the pool?`)) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}`, { method: 'DELETE' })
      load()
    } catch (err) { alert(`Remove failed: ${err.message}`) }
  }, [load])

  // Admission gate (the persistent switch, unlike Remove which a heartbeat undoes).
  const admit = useCallback(async (worker) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/admit`, { method: 'POST' })
      load()
    } catch (err) { alert(`Admit failed: ${err.message}`) }
  }, [load])

  const block = useCallback(async (worker) => {
    if (!confirm(`Block ${worker.name}? It stops serving and its agent exits on next contact.`)) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/block`, { method: 'POST' })
      load()
    } catch (err) { alert(`Block failed: ${err.message}`) }
  }, [load])

  // Dedicate this worker to a pool (reserved for an app's traffic). A worker that
  // declares WORKER_POOL on its box re-asserts that on its next heartbeat.
  const setPool = useCallback(async (worker) => {
    const next = window.prompt(
      `Dedicated pool for "${worker.name}" (blank = general). Requests tagged for this pool route here; general traffic won't.`,
      worker.pool || '')
    if (next === null) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/pool`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pool: next.trim() }),
      })
      load()
    } catch (err) { alert(`Set pool failed: ${err.message}`) }
  }, [load])

  // Console-managed serving config (daylight item 3). Persists in the AGENT's
  // settings file (source of truth over env drop-ins); applies via a short
  // agent re-exec — the row blips offline→online, same worker id.
  const setConfig = useCallback(async (worker) => {
    const cur = worker.config?.slot_count
    const next = window.prompt(
      `Slot count for "${worker.name}" (0–16; currently ${cur ?? '?'}${worker.config?.slot_count_source ? ` from ${worker.config.slot_count_source}` : ''}).\n` +
      '0 = no slots (in-process only). Applies via a ~5s agent restart.',
      cur != null ? String(cur) : '')
    if (next === null || next.trim() === '') return
    const n = Number(next)
    if (!Number.isInteger(n) || n < 0 || n > 16) { alert('slot count must be an integer 0–16'); return }
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot_count: n }),
      })
      if (r && r.ok === false) alert(`Config failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'slot_count', value: n })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Config failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Per-model residency (v3 final): deep-merged into the agent's settings.
  // on-demand IS the default and is represented by NO stored override, so
  // anything except 'static' normalizes to null (clears the override).
  const setResidency = useCallback(async (worker, modelKey, mode) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ residency: { [modelKey]: mode === 'static' ? 'static' : null } }),
      })
      if (r && r.ok === false) alert(`Residency failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'residency', model: modelKey, value: mode })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Residency failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk residency (todo t12): set the RESIDENCY tier of a SELECTED set of a
  // worker's models in ONE settings-write. Central relays a single /ops/config
  // with the whole residency map (same code path + same one agent re-exec as
  // pin-all) — never one restart per model. Confirms first, surfaces per-model
  // results + counts exactly like pinAll. Residency ONLY — never touches 📌 pin.
  const setResidencyMany = useCallback(async (worker, modelKeys, mode) => {
    const keys = (modelKeys || []).filter(k => (worker.models || []).includes(k))
    if (keys.length === 0) { alert('No models selected.'); return }
    const label = mode === 'static' ? '🔒 static' : '⏲ on-demand'
    const desc = mode === 'static'
      ? 'Static is a locked seat: the model is kept on this worker and never evicted (the only tier that keeps files on disk). '
      : 'On-demand is the default: the model loads on call and yields its seat when another model needs it. '
    if (!confirm(
      `Set residency → ${label} for ${keys.length} selected model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      desc + 'The worker agent restarts once (~5s) to apply all of them together.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/residency-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_keys: keys, mode }),
      })
      if (r && r.ok === false && !r.results) {
        // A pre-relay reject (bad body/mode) — no per-model map to show.
        alert(`Residency change failed: ${r.error?.message || 'unknown'}`)
        return
      }
      if (r && r.restarting) {
        markApplying(worker.id, { kind: 'residency_all', models: keys,
                                  value: mode === 'static' ? 'static' : 'on-demand' })
      }
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Set residency → ${label}: ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Set residency failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk ALLOC (todo t15; engine gate narrowed t26; per-model % t48): set the
  // GPU allocation (spill contract) of a SELECTED set of a worker's models in
  // ONE request. Unlike residency/pin this is a central REGISTRY write
  // (assign_model, same as the single ⚙ editor) — it does NOT restart the
  // agent; the spill applies on next load. `spill` is what AllocControl
  // produces (autofit {} / max GPU {n_gpu_layers:-1} / CPU only
  // {n_gpu_layers:"off"} / explicit budgets) — the ONE contract to broadcast
  // to every selected model. `perModel` (t48, optional) is AllocControl's
  // PER-MODEL resolution of a PERCENT VRAM/RAM budget — {model_key: spill},
  // each entry already resolved against THAT model's own size — sent instead
  // of `spill` whenever it's non-empty (a percent field was actually in play);
  // otherwise every member wants the identical flat contract and `spill` alone
  // is correct, same as before t48.
  // Autofit / Max GPU / CPU only are engine-agnostic PLACEMENT INTENT and apply
  // to EVERYONE; only the explicit-budget class is GGUF-only and splits the
  // selection. Confirms first; surfaces per-model results like setResidencyMany.
  const setAllocMany = useCallback(async (worker, modelKeys, spill, perModel) => {
    const keys = (modelKeys || []).filter(k => (worker.models || []).includes(k))
    if (keys.length === 0) { alert('No models selected.'); return }
    const perModelActive = !!(perModel && Object.keys(perModel).length > 0)
    // Parity label (2026-07-24 bulk parity): BulkAllocControl posts the new
    // vocabulary — {alloc_mode:<name>} for the four zero-knob modes, {} for
    // "Default (derived)" (clear-to-derivation), and a perModel map of
    // {alloc_mode:'explicit',…} for the bulk explicit split. spillLabel() only
    // reads the legacy n_gpu_layers/budget keys, so derive the human label from
    // whichever the caller actually sent.
    const flatMode = spill && spill.alloc_mode ? String(spill.alloc_mode) : null
    const isClear = !perModelActive && (!spill || Object.keys(spill).length === 0)
    const label = perModelActive ? 'Explicit'
      : isClear ? 'Default (derived)'
      : flatMode ? allocModeLabel(flatMode)
      : spillLabel(spill)
    // Engine split (t26 + 2026-07-24): the GGUF-only class is now the
    // explicit-budget spills AND the flat modes max-ram / explicit (which ride
    // alloc_mode). When the chosen alloc is GGUF-only, pre-state the split so the
    // operator sees which members are touched before apply. (The backend
    // re-enforces this — the UI number is a courtesy, not the gate.) The KEY set
    // is identical whether broadcasting `spill` or fanning out `perModel`.
    const fwOf = (k) => String(models.find(m => (m.model_key ?? m.key) === k)?.framework || '').toLowerCase()
    const isGgufFw = (fw) => fw === 'gguf' || fw === 'llama_cpp'
    const ggufKeys = keys.filter(k => isGgufFw(fwOf(k)))
    const modeGgufOnly = flatMode === 'max-ram' || flatMode === 'explicit'
      || (perModelActive && Object.values(perModel).some(s => s && s.alloc_mode === 'explicit'))
    const gO = modeGgufOnly || allocIsGgufOnly(spill)
    let confirmMsg = isClear
      ? `Revert ${keys.length} selected model${keys.length === 1 ? '' : 's'} to the DERIVED default on ${worker.name}?\n\n` +
        'Clears any pinned allocation contract so each model tracks its derived default again (and improves with it as measured values land). Applies on next load — no agent restart.'
      : `Set GPU allocation → ${label} for ${keys.length} selected model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
        'The allocation is the model\'s resource contract on this worker; it applies the next time each model loads (no agent restart).'
    if (gO) {
      const m = keys.length - ggufKeys.length
      confirmMsg =
        `Set GPU allocation → ${label} (GGUF-only) on ${worker.name}?\n\n` +
        `Applies to ${ggufKeys.length} GGUF model${ggufKeys.length === 1 ? '' : 's'}; ` +
        `${m} transformers/comfy model${m === 1 ? '' : 's'} skipped with a reason (${label} is a GGUF concept; Default / Max GPU / GPU only / RAM only would apply to all).\n\n` +
        'Applies on next load — no agent restart.'
      if (ggufKeys.length === 0) {
        alert(`None of the ${keys.length} selected models are GGUF — "${label}" is GGUF-only and would touch nothing. Use Default, Max GPU, GPU only, or RAM only to affect non-GGUF models.`)
        return
      }
    }
    if (perModelActive) {
      confirmMsg += '\n\n(An explicit % split is in play — each model resolves it ' +
        'against its OWN size, so the actual GiB numbers differ per model.)'
    }
    if (!confirm(confirmMsg)) return
    try {
      // t48: send the per-model map when the percent resolution actually
      // produced one; otherwise the single shared `spill` exactly as before.
      const body = perModelActive
        ? { model_keys: keys, spills: Object.fromEntries(keys.map(k => [k, perModel[k] || {}])) }
        : { model_keys: keys, spill: spill || {} }
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/alloc-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (r && r.ok === false && !r.results) {
        alert(`Allocation change failed: ${r.error?.message || 'unknown'}`)
        return
      }
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0, skN = r?.counts?.skipped ?? 0
      // Surface skipped-with-reason (engine mismatch) + any hard errors together.
      const notable = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
      if (notable.length > 0) {
        alert(`Set alloc → ${label}: ${okN} applied` +
          (skN ? `, ${skN} skipped` : '') + (errN ? `, ${errN} failed` : '') +
          ` on ${worker.name}.\n\n` +
          notable.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) { alert(`Set alloc failed: ${err.message}`) }
  }, [load, models])

  // Tiers v2 FILES axis: pin toggles ride the same settings channel.
  const togglePin = useCallback(async (worker, modelKey, pin) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: { [modelKey]: pin ? true : null } }),
      })
      if (r && r.ok === false) alert(`Pin failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'pinned', model: modelKey, value: pin })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Pin failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk pin: 📌 pin EVERY model designated to this worker in one settings-write
  // (central relays a single /ops/config with the full pinned map — same code
  // path as the per-model pin). Sticky, so we CONFIRM first: each pinned model
  // then blocks unassign until Unpin all. Surfaces per-model results + counts.
  const pinAll = useCallback(async (worker) => {
    const keys = worker.models || []
    if (keys.length === 0) { alert(`${worker.name} has no assigned models to pin.`); return }
    const unpinned = keys.filter(k => !worker.config?.pinned?.[k])
    if (unpinned.length === 0) {
      alert(`All ${keys.length} model${keys.length === 1 ? '' : 's'} on ${worker.name} are already pinned.`)
      return
    }
    if (!confirm(
      `📌 Pin all ${keys.length} model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      'Pinning is PERMANENT attribution: each pinned model then refuses unassign ' +
      '("unpin first") until you Unpin all. The worker agent restarts (~5s) to apply.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/pin-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      })
      if (r && r.restarting) markApplying(worker.id, { kind: 'pin_all', models: keys, value: true })
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Pinned ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Pin all failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Unpin all — the undo for Pin all. Unpins every model on the worker in one
  // /ops/config write (same relay); afterward the models can be unassigned.
  const unpinAll = useCallback(async (worker) => {
    const keys = worker.models || []
    const pinnedKeys = keys.filter(k => worker.config?.pinned?.[k])
    if (pinnedKeys.length === 0) { alert(`No pinned models on ${worker.name}.`); return }
    if (!confirm(
      `Unpin all ${pinnedKeys.length} pinned model${pinnedKeys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      'This is the undo for Pin all — the models can be unassigned again afterward. ' +
      'The worker agent restarts (~5s) to apply.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unpin-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      })
      if (r && r.restarting) markApplying(worker.id, { kind: 'pin_all', models: keys, value: false })
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Unpinned ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Unpin all failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  const setLimits = useCallback(async (worker, limits) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/limits`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limits }),
      })
      load()
    } catch (err) { alert(`Set limits failed: ${err.message}`) }
  }, [load])

  // Drop every offline worker (e.g. stale entries left by an old agent).
  const pruneOffline = useCallback(async () => {
    const stale = workers.filter(w => w.status !== 'online')
    if (!stale.length) return
    if (!confirm(`Remove ${stale.length} offline worker(s) from the pool?`)) return
    await Promise.all(stale.map(w =>
      fetchJson(`/api/llm/workers/${encodeURIComponent(w.id)}`, { method: 'DELETE' }).catch(() => {})
    ))
    load()
  }, [workers, load])

  const onlineCount = workers.filter(w => w.status === 'online').length
  const offlineCount = workers.length - onlineCount
  const pendingCount = workers.filter(w => w.admission === 'pending').length

  // Fleet rollup: VRAM used/total and models loaded across ONLINE workers.
  const fleet = useMemo(() => {
    let total = 0, free = 0, serving = 0, hasVram = false
    for (const w of workers) {
      if (w.status !== 'online') continue
      serving += (w.loaded_models || []).length
      for (const g of (w.gpus || [])) {
        if (g.memory_total != null) { total += g.memory_total; hasVram = true }
        if (g.memory_free != null) free += g.memory_free
      }
    }
    return { total, free, used: Math.max(total - free, 0), serving, hasVram }
  }, [workers])

  // The setup disclosure is forced open while the pool is empty (the install
  // command is the only first-worker path) and otherwise follows the user's
  // toggle — defaulting closed so the live worker list leads.
  const isEmpty = workers.length === 0

  // One-line installer served by THIS central node. The command must be
  // fetchable FROM THE WORKER BOX, but by default all we have is the browser's
  // origin — browsing the console on central itself (http://localhost:7002)
  // would render a command that points a remote box at its own loopback. When
  // the origin is loopback, ask central for its LAN address and render that
  // instead (the curl host is also what the script bakes as CENTRAL, so fixing
  // the display fixes the install). If central can't tell, keep the command
  // but warn.
  const browserOrigin = resolveApiOrigin()
  const originIsLoopback = useMemo(() => {
    try {
      const h = new URL(browserOrigin).hostname.replace(/^\[|\]$/g, '')
      return h === 'localhost' || h === '::1' || h === '0.0.0.0' || h.startsWith('127.')
    } catch { return false }
  }, [browserOrigin])
  const [lanOrigin, setLanOrigin] = useState(null)
  useEffect(() => {
    if (!originIsLoopback) return
    fetchJson('/api/llm/workers/central-address')
      .then(d => { if (d && d.lan_ip && d.base_url) setLanOrigin(d.base_url) })
      .catch(() => {})   // fall through to the loopback warning
  }, [originIsLoopback])
  const installCmd = `curl -fsSL ${(originIsLoopback && lanOrigin) || browserOrigin}/api/llm/workers/install.sh | bash`

  // models picked in the "Add manually" form (form.models stays a comma-joined string)
  const selectedModels = form.models.split(',').map(s => s.trim()).filter(Boolean)
  const addModel = (key) => setForm(f => {
    const cur = f.models.split(',').map(s => s.trim()).filter(Boolean)
    return cur.includes(key) ? f : { ...f, models: [...cur, key].join(',') }
  })
  const removeModel = (key) => setForm(f => ({
    ...f,
    models: f.models.split(',').map(s => s.trim()).filter(Boolean).filter(k => k !== key).join(','),
  }))

  // Cross-worker allocation map (model_key -> [{id,name}]) — powers the load
  // table's anti-duplicate lockout: a model already on another worker is shown
  // "on <worker>" and can't be re-allocated unless the breaker is flipped.
  const allocationMap = useMemo(() => {
    const map = {}
    for (const w of workers) {
      for (const mk of (w.models || [])) {
        (map[mk] ??= []).push({ id: w.id, name: w.name })
      }
    }
    return map
  }, [workers])

  return (
    <div className="workers-panel">
      <div className={`wp-bar${embedded ? ' wp-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="wp-title">🖧 GPU Workers</span>
        <span className="wp-count">{onlineCount} online / {workers.length} total</span>
        {fleet.hasVram && (
          <span className="wp-fleet" title="VRAM used / total across online workers">
            VRAM {fmtBytes(fleet.used)} / {fmtBytes(fleet.total)} · {fmtBytes(fleet.free)} free
          </span>
        )}
        {fleet.serving > 0 && (
          <span className="wp-fleet-loaded" title="models serving (resident in VRAM) across the fleet">
            🔥 {fleet.serving} serving
          </span>
        )}
        {pendingCount > 0 && (
          <span className="wp-pending-chip" title="Workers awaiting your approval — they don't serve until admitted">
            ⏳ {pendingCount} pending
          </span>
        )}
        {error && <span className="wp-err" title={error}>registry error<FixDoc doc="registry-error" /></span>}
        {offlineCount > 0 && (
          <button
            className="wp-prune"
            title="Remove all offline workers"
            onClick={e => { e.stopPropagation(); pruneOffline() }}
          >clear {offlineCount} offline</button>
        )}
        {!embedded && <span className="wp-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="wp-body">
          {/* t22 (operator): the page reads live workers first, then group
              actions, then the "Add & manage workers" tooling — so the per-worker
              cards + central footer come first, GroupAssignPanel next, and the
              wp-setup section LAST (moved below, after the central footer). */}
          {workers.length === 0 && <div className="wp-empty">No workers have joined the pool yet.</div>}
          {/* Tab strip: one worker's card at a time. `selected` falls back to the
              first worker so a stale/removed id from sessionStorage can never
              leave the panel blank. Each tab carries the worker's live status dot
              and its assignment count, so the strip stays informative without
              expanding the cards. */}
          {workers.length > 1 && (
            <div className="wp-tabs" role="tablist">
              {workers.map(w => (
                <button
                  key={w.id}
                  role="tab"
                  aria-selected={w.id === selectedWorkerId}
                  className={`wp-tab wp-tab-${w.status || 'unknown'}${w.id === selectedWorkerId ? ' wp-tab-on' : ''}`}
                  onClick={() => setActiveWorker(w.id)}
                  title={`${w.name || w.id} — ${w.status || 'unknown'}, ${(w.models || []).length} assigned`}
                >
                  <span className="wp-tab-dot" />
                  <span className="wp-tab-name">{w.name || w.id.slice(0, 8)}</span>
                  <span className="wp-tab-count">{(w.models || []).length}</span>
                </button>
              ))}
            </div>
          )}
          {shownWorkers.map(w => (
            <WorkerRow
              key={w.id}
              worker={w}
              models={models}
              allocation={allocationMap}
              onAssign={assign}
              onRefresh={load}
              onLoad={loadModel}
              onUnassign={unassign}
              onRemove={remove}
              onFree={freeModel}
              onFreeAll={freeAll}
              onFreeRam={freeRam}
              onRestart={restart}
              restarting={!!restarting[w.id]}
              onUpdate={update}
              updating={!!updating[w.id]}
              onAdmit={admit}
              onBlock={block}
              onSetPool={setPool}
              onSetLimits={setLimits}
              onSetConfig={setConfig}
              onSetResidency={setResidency}
              onSetResidencyMany={setResidencyMany}
              onSetAllocMany={setAllocMany}
              onTogglePin={togglePin}
              onPinAll={pinAll}
              onUnpinAll={unpinAll}
              onReap={reap}
              onApproveEvictions={approveEvictions}
              onEvict={evictModel}
              onAllocateMany={allocateMany}
              applying={!!applying[w.id]}
              blockedKeys={blockedKeys}
              onToggleBlock={toggleBlock}
            />
          ))}

          {/* Central itself, represented as a worker like the others — its
              compute is the local slot pool (+ in-process fallback), so it
              renders with the same row anatomy: identity, resources, and its
              "serving" list (= the slots). Rendered LAST, below every remote
              worker row: it's the console's own host, not fleet capacity —
              secondary context, not the headline. */}
          {central && (
            <div className="wp-central-footer">
              <div className="wp-worker wp-online wp-central">
                <div className="wp-worker-head">
                  <span className="wp-dot" />
                  <span className="wp-name">central</span>
                  <span className="wp-status">online</span>
                  <span className="wp-adm wp-adm-pill-approved" title="The console's own server">this host</span>
                  <span className="wp-url" title="Serves via its local slot pool; also the fallback when no worker can take a model">
                    local slot pool{central.enabled === false ? ' (disabled)' : ''}
                  </span>
                </div>
                {central.resources && (
                  <div className="wp-ram" title="Central's own RAM — what the slot preflight budgets against (minus reserves)">
                    🧠 RAM {fmtBytes(central.resources.available_bytes ?? central.resources.free_bytes)} free
                    {central.resources.total_bytes != null && <> of {fmtBytes(central.resources.total_bytes)}</>}
                    {central.resources.cpu_count != null && <> · {central.resources.cpu_count} cores</>}
                  </div>
                )}
                <div className="wp-models">
                  <span className="wp-models-label">Slots:</span>
                  {(central.slots || []).length === 0 && <span className="wp-none">— no slots —</span>}
                  {(central.slots || []).map(s => {
                    const state = s.model_key ? (s.healthy ? 'serving' : 'warming') : 'idle'
                    return (
                      <span key={s.slot_id} className={`wp-model wp-st-${state}`}>
                        <span className={`wp-state-pill wp-pill-${state}`}>
                          {s.model_key ? (s.healthy ? '🔥 serving' : '⏳ warming') : `○ slot ${s.slot_id}`}
                        </span>
                        <span className="wp-model-name">{s.model_key || '— idle —'}</span>
                        <span className="wp-model-facts">
                          {s.rss_bytes ? `${fmtBytes(s.rss_bytes)} RSS` : ''}
                          {s.n_gpu_layers != null ? ` · ngl ${s.n_gpu_layers}` : ''}
                          {s.ctx != null ? ` · ctx ${s.ctx}` : ''}
                        </span>
                        {s.model_key && (
                          <button className="wp-free" title="Unload this slot (frees its RAM/VRAM)"
                                  onClick={() => centralUnload(s._control)}>⏏</button>
                        )}
                      </span>
                    )
                  })}
                </div>
                {/* Load-a-model — same affordance as every other worker row. This
                    made the separate central-only "Model Slots" panel redundant
                    (retired); central presents as just another worker. */}
                {!showCentralLoad ? (
                  <button className="wp-load-toggle" onClick={() => setShowCentralLoad(true)}
                          title="Load a model into a free central slot">
                    ＋ load a model
                  </button>
                ) : (
                  <div className="wp-assign-row">
                    <ModelPicker
                      models={models}
                      value={centralPick}
                      onPick={setCentralPick}
                      placeholder="Load a model into a free central slot…"
                      autoFocus
                    />
                    <button className="wp-alloc-apply" disabled={!centralPick || centralBusy}
                            onClick={() => centralLoad(centralPick)}>
                      {centralBusy ? '…' : '+ Load'}
                    </button>
                    <button className="wp-load-cancel" title="Cancel"
                            onClick={() => { setShowCentralLoad(false); setCentralPick('') }}>×</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Group actions — below the live workers (t22). */}
          {workers.length > 0 && (
            <GroupAssignPanel models={models} workers={workers} onGroupAssign={groupAssign} />
          )}

          {/* Add & manage workers — install command, enrollment tokens, manual
              add. Moved BELOW the workers + group actions (t22). Stays collapsed
              by default (setupOpen defaults false); it is force-OPEN only when the
              fleet is empty (isEmpty), since then there are no workers above it and
              the install command is the only way to add the first box. */}
          <details className="wp-setup" open={setupOpen || isEmpty}
                   onToggle={e => { if (!isEmpty) setSetupOpen(e.currentTarget.open) }}>
            <summary className="wp-setup-summary">
              ＋ Add &amp; manage workers — install command, enrollment tokens, manual add
            </summary>
          <div className="wp-install">
            <span className="wp-install-label">Add a GPU box — run on the worker:</span>
            <code className="wp-install-cmd" title="Click to copy"
                  onClick={() => navigator.clipboard?.writeText(installCmd)}>
              {installCmd}
            </code>
            <span className="wp-install-note">
              Central fills in its own address and this box's reachable IP — no per-worker config.
            </span>
            {originIsLoopback && !lanOrigin && (
              <span className="wp-install-warn">
                ⚠ You're browsing central at {browserOrigin} — a worker box can't reach that
                address. Replace it with one the worker can reach (e.g. central's LAN IP).
                <FixDoc doc="worker-join" />
              </span>
            )}
          </div>

          <details className="wp-tokens">
            <summary>Enrollment tokens ({tokens.filter(t => !t.revoked).length} active) — admit machines to the fleet</summary>
            <div className="wp-tokens-body">
              <button className="wp-token-issue" onClick={issueToken}>+ Issue enrollment token</button>
              {newToken && (
                <div className="wp-token-new">
                  <strong>Copy this token now — it is shown only once:</strong>
                  <code className="wp-token-secret" title="Click to copy"
                        onClick={() => navigator.clipboard?.writeText(newToken.token)}>
                    {newToken.token}
                  </code>
                  <span className="wp-install-note">Run on the worker (bakes central + token into its unit):</span>
                  <code className="wp-install-cmd" title="Click to copy"
                        onClick={() => navigator.clipboard?.writeText(
                          `WORKER_ENROLL_TOKEN=${newToken.token} ${installCmd}`)}>
                    {`WORKER_ENROLL_TOKEN=${newToken.token} ${installCmd}`}
                  </code>
                  <button className="wp-token-dismiss" onClick={() => setNewToken(null)}>Done</button>
                </div>
              )}
              {tokens.length === 0 && <div className="wp-none">No enrollment tokens issued.</div>}
              {tokens.map(t => (
                <div key={t.id} className={`wp-token-row${t.revoked ? ' wp-token-revoked' : ''}`}>
                  <span className="wp-token-label">{t.label || '(no label)'}</span>
                  <span className="wp-token-id" title="token id">{t.id}</span>
                  {t.revoked
                    ? <span className="wp-token-state">revoked</span>
                    : <button className="wp-token-revoke" title="Revoke — its workers are refused and stop"
                              onClick={() => revokeToken(t)}>revoke</button>}
                </div>
              ))}
            </div>
          </details>

          <details className="wp-manual">
            <summary>Add manually</summary>
            <form className="wp-register" onSubmit={register}>
              <input
                placeholder="worker name (e.g. gpu-box-1)"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              />
              <input
                placeholder="worker URL (optional — central uses source IP)"
                value={form.url}
                onChange={e => setForm(f => ({ ...f, url: e.target.value }))}
              />
              <div className="wp-models-pick">
                <ModelPicker
                  models={models.filter(m => !selectedModels.includes(m.model_key ?? m.key))}
                  value=""
                  onPick={addModel}
                  placeholder={models.length ? 'models to serve (optional)…' : 'no models registered yet'}
                  disabled={!models.length}
                />
                {selectedModels.length > 0 && (
                  <div className="wp-models-chips">
                    {selectedModels.map(k => (
                      <span key={k} className="wp-models-chip">
                        {models.find(m => (m.model_key ?? m.key) === k)?.name || k}
                        <button type="button" className="wp-chip-x" title="Remove"
                                onClick={() => removeModel(k)}>×</button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <button type="submit" disabled={busy}>+ Add worker</button>
            </form>
          </details>
          </details>
        </div>
      )}
    </div>
  )
}
