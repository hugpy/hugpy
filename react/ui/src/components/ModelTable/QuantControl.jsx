import { useEffect, useState, useCallback } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { getServing, invalidateServing } from './servingCache'

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
        setMsg(a.applied ? '✓ reloaded with this quant' : `saved — ${a.reason || 'not applied'}`)
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
