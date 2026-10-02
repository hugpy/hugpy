import { useEffect, useState, useCallback } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { getServing, invalidateServing } from './servingCache'
import { responseReason } from '../responseReason'

// Per-model serving control: mode + GPU layers + CPU threads + context.
// Self-contained — talks straight to /api/llm/serving/<key>, so it needs no
// wiring through App state. Shown in a model's detail row (GGUF only). The GGUF
// VARIANT choice lives in the sibling QuantControl (a first-class per-model
// setting), so this control never sends gguf_file and can't clobber that choice.

const MODES = [
  ['off', 'off (in-process)'],
  ['systemd', 'systemd (always-on)'],
  ['swap', 'swap (on-demand)'],
]

export default function ServingControl({ modelKey, framework }) {
  const [row, setRow] = useState(null)
  const [form, setForm] = useState(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const load = useCallback(async () => {
    setMsg('')
    try {
      const d = await getServing(modelKey)   // shared, deduped, 20 s cache (see servingCache.js)
      setRow(d)
      setForm({
        serve_mode: d.mode ?? 'off',
        n_gpu_layers: d.n_gpu_layers ?? '',
        threads: d.threads ?? '',
        llama_ctx: d.ctx_size ?? '',
      })
    } catch (e) {
      setMsg(String(e.message || e))
    }
  }, [modelKey])

  useEffect(() => { load() }, [load])

  const save = async (apply) => {
    setBusy(true); setMsg(apply ? 'applying…' : 'saving…')
    try {
      const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, apply }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
      setRow(d)
      if (apply) {
        const a = d.apply || {}
        setMsg(a.applied ? '✓ applied (unit written + restarted)'
                          : `saved — not applied: ${responseReason(a)}`)
      } else {
        setMsg('✓ saved (apply to (re)write the unit)')
      }
    } catch (e) {
      setMsg(`✗ ${e.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  // GGUF only. Framework vocab is HF-canonical ('gguf'); accept legacy 'llama_cpp'.
  if (framework !== 'gguf' && framework !== 'llama_cpp') {
    return (
      <div className="mt-serve mt-serve-na">
        Serving control applies to GGUF (llama.cpp) models; this one runs
        in-process.
      </div>
    )
  }
  if (!form) {
    return <div className="mt-serve">{msg || 'loading serving…'}</div>
  }

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  return (
    <div className="mt-serve">
      <div className="mt-serve-fields">
        <label className="mt-serve-field">
          <span>Mode</span>
          <select value={form.serve_mode} onChange={set('serve_mode')}>
            {MODES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
          </select>
        </label>
        <label className="mt-serve-field">
          <span>GPU layers</span>
          <input type="number" value={form.n_gpu_layers}
                 onChange={set('n_gpu_layers')} placeholder="-1 = all" />
        </label>
        <label className="mt-serve-field">
          <span>CPU threads</span>
          <input type="number" value={form.threads} onChange={set('threads')} />
        </label>
        <label className="mt-serve-field">
          <span>Context</span>
          <input type="number" value={form.llama_ctx} onChange={set('llama_ctx')} />
        </label>
      </div>

      <div className="mt-serve-actions">
        <button disabled={busy} onClick={() => save(false)}>Save</button>
        <button disabled={busy} onClick={() => save(true)}>Save &amp; apply</button>
        {row?.endpoint && <span className="mt-serve-ep">→ {row.endpoint}</span>}
        {!row?.endpoint && <span className="mt-serve-ep">→ in-process</span>}
        {msg && <span className="mt-serve-msg">{msg}</span>}
      </div>
    </div>
  )
}
