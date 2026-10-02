import { useEffect, useRef, useState } from 'react'
import {
  ALLOC_MODE_OPTIONS,
  GGUF_ONLY_MODE_TIP,
  NONGGUF_ALLOC_MODES,
  WP_GIB,
} from './constants'
import {
  allocDisableReason,
  allocModeLabel,
  modeToSpill,
  spillToMode,
  workerCapacity,
} from './allocation'

// t39 (operator: "the up and down arrows are useless and a hindrance" — every
// tolerance band AND the value it loosens becomes a slider, no number-spinner
// inputs anywhere in this editor). One editable numeric readout shared by every
// slider below: shows the live value; click it to type an exact number (Enter
// commits + re-syncs the slider, Escape/blur cancels). The slider stays the
// PRIMARY control — this is the escape hatch for precise entry.
export function SliderReadout({ text, value, onCommit, title }) {
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
export function PlainSlider({ label, title, value, onChange, min, max, step = 1, formatValue, clampMax = true, className = '' }) {
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
export function BudgetInput({ label, title, unit, value, onChange, cap, fallbackMax = 128,
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
export function AllocControl({ spill, onApply, onCancel, applyLabel = 'Apply', engineGguf = null, worker = null,
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
export function ExplicitPanel({ spill, worker, need, onApply, onCancel }) {
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
export function AllocModeMenu({ mode, spill, worker, need, engineGguf, feasible, feasibleCtx,
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
export function BulkExplicitPanel({ bulkKeys, getModelBytes, onApply, onCancel }) {
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
export function BulkAllocControl({ count, bulkKeys, getModelBytes, onApply, onCancel }) {
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
