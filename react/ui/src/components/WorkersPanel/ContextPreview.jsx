import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import { fmtBytes } from './formatters'

// The size per context setting (operator, 2026-09-29): as the Ctx slider moves,
// show "at 25%: 23.4 GB of 23.6 GB free" from central's priced preview
// (GET /llm/workers/<id>/context-preview — weights + KV at that ctx vs the
// worker's budget) and the largest pct that fits. Debounced; never guesses.
export function ContextPreview({ workerId, modelKey, pct }) {
  const [view, setView] = useState(null)
  useEffect(() => {
    if (!workerId || !modelKey) return undefined
    let live = true
    const t = setTimeout(() => {
      fetchJson(`/api/llm/workers/${encodeURIComponent(workerId)}/context-preview?model_key=${encodeURIComponent(modelKey)}&pct=${pct}`)
        .then(r => { if (live) setView(r) })
        .catch(e => { if (live) setView({ error: e.message }) })
    }, 150)
    return () => { live = false; clearTimeout(t) }
  }, [workerId, modelKey, pct])
  if (!view) return <div className="wp-context-preview">pricing…</div>
  if (view.error) return <div className="wp-context-preview">{view.error}</div>
  const r = view.requested
  if (!r || r.need_bytes == null) return <div className="wp-context-preview">{view.reason || 'not priced'}</div>
  const budget = view.budget_bytes
  return (
    <div className="wp-context-preview"
         title={`weights ${fmtBytes(view.weights_bytes)} + KV ${fmtBytes(r.kv_bytes || 0)} at ctx ${(r.ctx || 0).toLocaleString()}; budget = ${fmtBytes(view.gpu_vram_free || 0)} free − ${fmtBytes(view.reserve_bytes || 0)} cushion`}>
      at {r.pct}%: {fmtBytes(r.need_bytes)}{budget != null ? ` of ${fmtBytes(budget)} free` : ''}
      {r.fits === false ? ' — will not fit' : ''}
      {view.max_fitting_pct != null ? ` · largest fitting: ${view.max_fitting_pct}%` : (budget != null ? ' · no ctx setting fits' : '')}
    </div>
  )
}
