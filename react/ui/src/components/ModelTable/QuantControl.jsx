import { useEffect, useState, useCallback } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { getServing, invalidateServing, primeServing } from './servingCache'
import { fetchJson } from '../../api'
import { sizeView } from './modelSize'
import { responseReason } from '../responseReason'

// Per-model GGUF QUANTIZATION picker — a first-class model setting, not a serving
// sub-option. A GGUF repo may hold many quants (Q4_K_M, Q5_K_M, Q8_0, …) but only
// ONE serves; this picks it (persisted as the global gguf_file override, honored by
// every worker/runner) and — since the effective quant also defines the model's
// size — drives the size shown across the console. GGUF (llama.cpp) only.
//
// Talks straight to /api/llm/serving/<key>; on change it calls onChanged so the
// parent reloads /models (row badge + sizes update). If the model is served via
// systemd/swap, a "reload now" applies the new quant to the running unit.

function fmtBytes(n) {
  if (n == null || !isFinite(n)) return ''
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0, v = Number(n)
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${u[i]}`
}

export default function QuantControl({ modelKey, framework, onChanged }) {
  const [variants, setVariants] = useState([]) // [{filename, bytes, is_effective}]
  const [sel, setSel] = useState('')           // current gguf_file override ('' = auto)
  const [effGguf, setEffGguf] = useState(null)
  const [effBytes, setEffBytes] = useState(null)
  const [mode, setMode] = useState('off')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [ready, setReady] = useState(false)

  const apply = useCallback((d) => {
    setVariants(Array.isArray(d.available_gguf_detail) ? d.available_gguf_detail : [])
    setSel(d.gguf_file || '')
    setEffGguf(d.effective_gguf ?? null)
    setEffBytes(d.effective_bytes ?? null)
    setMode(d.mode ?? 'off')
  }, [])

  const load = useCallback(async () => {
    try {
      const d = await getServing(modelKey)   // shared, deduped, 20 s cache (see servingCache.js)
      apply(d)
    } catch (e) {
      setMsg(String(e.message || e))
    } finally {
      setReady(true)
    }
  }, [modelKey, apply])

  useEffect(() => { load() }, [load])

  // GGUF only. Framework vocab is HF-canonical ('gguf'); accept legacy 'llama_cpp'.
  if (framework !== 'gguf' && framework !== 'llama_cpp') return null

  const save = async (value, doApply) => {
    setBusy(true); setMsg(doApply ? 'reloading…' : 'saving…')
    try {
      const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gguf_file: value, apply: doApply }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
      apply(d)
      if (doApply) {
        const a = d.apply || {}
        setMsg(a.applied ? '✓ reloaded with this quant' : `saved — not applied: ${responseReason(a)}`)
      } else {
        setMsg('✓ saved')
      }
      onChanged?.()
    } catch (e) {
      setMsg(`✗ ${e.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  const served = mode === 'systemd' || mode === 'swap'

  return (
    <div className="mt-quant">
      <div className="mt-serve-title">Quantization (GGUF variant)</div>
      {!ready ? (
        <div className="mt-quant-none">loading quants…</div>
      ) : variants.length === 0 ? (
        <div className="mt-quant-none">No .gguf variants downloaded for this model yet.</div>
      ) : (
        <div className="mt-quant-row">
          <select value={sel} disabled={busy}
                  title="Which downloaded quantization this model serves — global for the model. 'auto' lets the resolver pick (q4_k_m first)."
                  onChange={(e) => save(e.target.value, false)}>
            <option value="">auto{effGguf ? ` → ${effGguf}` : ''}</option>
            {variants.map(v => (
              <option key={v.filename} value={v.filename}>
                {v.filename}{v.bytes ? ` · ${fmtBytes(v.bytes)}` : ''}{v.is_effective ? ' ✓ effective' : ''}
              </option>
            ))}
          </select>
          {effBytes != null && (
            <span className="mt-quant-eff"
                  title="Size the model actually serves — this quant plus its mmproj projector (vision).">
              serves {fmtBytes(effBytes)}
            </span>
          )}
          {served && (
            <button className="mt-quant-apply" disabled={busy}
                    title="Reload the running unit now with the selected quant (systemd / swap serving modes)."
                    onClick={() => save(sel, true)}>
              ⟳ reload now
            </button>
          )}
          {msg && <span className="mt-serve-msg">{msg}</span>}
        </div>
      )}
    </div>
  )
}


// ── Per-WORKER quant pin (2026-09-23) ──────────────────────────────────────
// One "Run on worker" row's quant: a dropdown of the model's COMPLETE GGUF
// variants (incomplete shard groups are listed by the model-wide picker as
// litter, never offered here — same filter as fleet_grading._quants), the
// model-wide effective one marked. A change POSTs gguf_file_by_worker for THIS
// worker only (the whole map is sent — the overrides layer replaces the map —
// with every other worker's pin carried over). The fit readout is the server's
// own verdict for the SHOWN quant on this worker
// (GET /models/<key>/meta?worker=<id>&gguf=<file>), fetched only when the
// (model, worker, quant) triple changes — never per render.
// Non-GGUF models have exactly one artifact: shown, not selectable.
const _isGguf = (fw) => fw === 'gguf' || fw === 'llama_cpp'

export function WorkerQuantSelect({ modelKey, worker, model, disabled = false, disabledTitle = '' }) {
  const gg = _isGguf(String(model?.framework || '').toLowerCase())
  const [row, setRow] = useState(null)     // the shared serving row
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [fit, setFit] = useState(null)
  const wid = worker?.id || ''
  const wname = worker?.name || ''

  useEffect(() => {
    if (!gg) return undefined
    let alive = true
    getServing(modelKey)
      .then(d => { if (alive) setRow(d) })
      .catch(e => { if (alive) setErr(`serving read failed: ${e.message || e}`) })
    return () => { alive = false }
  }, [modelKey, gg])

  const forms = [wid, wname].filter(Boolean).map(x => String(x).toLowerCase())
  const pins = (row && row.gguf_file_by_worker) || {}
  const pinKey = Object.keys(pins).find(k => forms.includes(String(k).toLowerCase()))
  const pin = pinKey ? String(pins[pinKey] || '') : ''
  const variants = ((row && row.available_gguf_detail) || []).filter(v => v && v.complete !== false)
  const matches = (v, token) => {
    const f = String(v.filename || '').toLowerCase(), t = String(token || '').toLowerCase()
    return !!t && (f === t || f.includes(t))
  }
  const pinned = pin ? (variants.find(v => String(v.filename).toLowerCase() === pin.toLowerCase())
                        || variants.find(v => matches(v, pin))) : null
  const shown = (pinned && pinned.filename) || (row && row.effective_gguf) || ''

  useEffect(() => {
    if (!gg || !shown || !wid) return undefined
    let alive = true
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}/meta?worker=${encodeURIComponent(wid)}&gguf=${encodeURIComponent(shown)}`)
      .then(m => { if (alive) setFit(m) })
      .catch(e => { if (alive) setFit({ error: String(e.message || e) }) })
    return () => { alive = false }
  }, [modelKey, wid, shown, gg])

  if (!gg) {
    const sv = sizeView(model, fmtBytes)
    const artifact = model?.filename || model?.hub_id || modelKey
    return (
      <span className="mt-wq mt-wq-single" title={`${model?.framework || 'non-GGUF'} model: one artifact, no quant choice — ${sv.title}`}>
        {artifact} · {sv.text}
      </span>
    )
  }

  const save = async (filename) => {
    const next = {}
    for (const [k, v] of Object.entries(pins)) {
      if (!forms.includes(String(k).toLowerCase())) next[k] = v
    }
    if (filename) next[wid || wname] = filename
    setBusy(true); setErr('')
    try {
      const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gguf_file_by_worker: next }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
      primeServing(modelKey, d)
      setRow(d)
    } catch (e) {
      setErr(`✗ ${e.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  const rec = fit && fit.recommended
  const fitText = !fit ? '' : fit.error ? `fit read failed: ${fit.error}`
    : fit.selected_gguf_error ? fit.selected_gguf_error
    : rec?.fits_vram ? `✓ ${shown} fits in VRAM`
    : rec?.fits_vram === false
      ? `◐ ${shown}: ${rec.gpu_fraction != null ? Math.round(rec.gpu_fraction * 100) + '% on GPU' : 'partial offload'}`
      : (rec?.reason || '')
  const unpinnedPin = pin && !pinned
    ? `pinned "${pin}" matches no complete variant on central` : ''

  return (
    <span className="mt-wq">
      <select value={pinned ? pinned.filename : ''}
              disabled={disabled || busy || !row || variants.length === 0}
              title={disabled ? disabledTitle
                : `Quant this model loads on ${wname || wid} (gguf_file_by_worker). `
                  + '"model-wide" follows the model\'s own quant choice.'}
              onChange={e => save(e.target.value)}>
        <option value="">model-wide{row && row.effective_gguf ? `: ${row.effective_gguf}` : ''}</option>
        {variants.map(v => (
          <option key={v.filename} value={v.filename}>
            {v.filename}{v.bytes ? ` · ${fmtBytes(v.bytes)}` : ''}{v.is_effective ? ' (model-wide effective)' : ''}
          </option>
        ))}
      </select>
      {fitText && (
        <span className={`mt-worker-fit ${rec?.fits_vram ? 'mt-fit-ok' : rec?.fits_vram === false ? 'mt-fit-partial' : ''}`}
              title={rec?.reason || fit?.selected_gguf_error || fit?.error || ''}>
          {fitText}
        </span>
      )}
      {unpinnedPin && <span className="mt-serve-msg">{unpinnedPin}</span>}
      {err && <span className="mt-serve-msg">{err}</span>}
    </span>
  )
}
