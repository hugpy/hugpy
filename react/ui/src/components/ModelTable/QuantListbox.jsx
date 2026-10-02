import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'

// QUANT LISTBOX (operator 2026-10-02): "checkable quants in the dropdown as a
// custom sort listbox". The pair knob `quants` is an ORDERED preference list;
// the server derives gguf_file = the first listed quant whose verdict fits at
// the pair's ctx target + KV type. Rules in code:
//   * the ALLOCATED quant (the derived gguf_file) sits at the TOP, marked;
//   * checked quants follow in preference order (▲▼ reorder);
//   * unchecked quants come last; a non-fitting quant is KEPT and greyed with
//     the verdict's sentence — never refused (budgets change);
//   * empty list = the verdict default (server unsets quants + gguf_file);
//   * a checked-but-untouched quant runs at AUTO allocation if the walk lands
//     on it (server binds placement knobs to the quant they were tuned under).
export function fmtGiB(n) {
  if (n == null || !isFinite(n)) return ''
  return `${(Number(n) / 2 ** 30).toFixed(1)} GiB`
}

// JS mirror of query_registry.quant_fit_walk for ONE pair's verdicts.
export function quantFitReason(memory, { ctxPct = null, trained = null, kvType = 'f16', bnbOn = false } = {}) {
  if (!memory) return 'no verdict for this quant on this worker yet'
  const table = bnbOn ? (memory.bnb_4bit || {}) : memory
  const kt = kvType === 'bf16' ? 'f16' : (kvType || 'f16')
  const want = (ctxPct != null && trained) ? Math.max(1024, Math.floor(Number(trained) * Number(ctxPct) / 100 / 1024) * 1024) : null
  let anyFit = false
  for (const mm of Object.values(table)) {
    if (!mm || typeof mm !== 'object' || mm.fits !== true) continue
    anyFit = true
    const cap = mm.ctx_max?.[kt]
    if (want != null && cap != null && (Number(cap) <= 0 || want > Number(cap))) continue
    return null
  }
  return (anyFit && want != null) ? `fits, but no mode holds the context target ${want.toLocaleString()} at KV ${kt}` : 'does not fit this worker in any mode'
}

// The console model key → the DB display row (name / hub_id forms, as the
// Workers panel matches them). DB-only route: never queues behind the store lock.
export async function fetchDbModelRow(modelKey) {
  const tail = String(modelKey || '').split('/').pop()
  const needle = tail.includes('~') ? tail.split('~').pop() : tail
  const d = await fetchJson(`/api/models/database?q=${encodeURIComponent(needle.slice(0, 48))}&limit=50`)
  const rows = (d && d.rows) || []
  const forms = [modelKey, tail, tail.replace('~', '/')].filter(Boolean).map(x => String(x).toLowerCase())
  return rows.find(r => {
    const names = [r.name, r.hub_id, r.hub_id && r.hub_id.includes('/') ? r.hub_id.replace('/', '~') : null]
      .filter(Boolean).map(x => String(x).toLowerCase())
    return names.some(n => forms.includes(n))
  }) || null
}

export function useDbModelRow(modelKey, refreshKey = 0) {
  const [row, setRow] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    let alive = true
    fetchDbModelRow(modelKey).then(r => { if (alive) { setRow(r); setErr('') } })
      .catch(e => { if (alive) setErr(String(e.message || e)) })
    return () => { alive = false }
  }, [modelKey, refreshKey])
  return [row, err]
}

export default function QuantListbox({ variants, order, allocated, fitFor, disabled = false, busy = false, onChange, title = '', minWidth = 320 }) {
  // RANKED PRIORITY LIST, NAME-SORTED DISPLAY (operator 2026-10-02: "sort by
  // name and keep the numbered designation … that's cleaner"). The checked
  // quants form a ranked preference — #1 is served when it fits; the ranks
  // below it are the evict-to-fit ladder (a polite eviction may step down to a
  // smaller quant rather than evict a neighbour). The ROWS never move: every
  // variant sits in filename order; the rank badge (#n) carries the priority
  // and ▲▼ change it. The allocated quant is marked in place (→ allocated).
  const files = (variants || []).map(v => v.filename).slice().sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }))
  const byFile = Object.fromEntries((variants || []).map(v => [v.filename, v]))
  const checked = (order || []).filter(f => files.includes(f))
  const locked = disabled || busy
  const move = (f, dir) => {
    const i = checked.indexOf(f); const j = i + dir
    if (i < 0 || j < 0 || j >= checked.length) return
    const next = checked.slice(); [next[i], next[j]] = [next[j], next[i]]
    onChange(next)
  }
  const toggle = (f) => onChange(checked.includes(f) ? checked.filter(x => x !== f) : [...checked, f])
  const Row = ({ f }) => {
    const v = byFile[f] || {}
    const idx = checked.indexOf(f)
    const on = idx >= 0
    const fit = fitFor ? fitFor(f) : { ok: null, text: '' }
    const isAlloc = allocated && f === allocated
    const grey = fit && fit.ok === false
    return (
      <div className={`mt-ql-row${on ? ' mt-ql-on' : ''}${isAlloc ? ' mt-ql-alloc' : ''}`}
           style={{ display: 'flex', alignItems: 'center', gap: 6, opacity: grey ? 0.55 : 1, padding: '2px 0' }}
           title={fit?.text || (on ? `priority #${idx + 1}` : 'not in the list')}>
        <input type="checkbox" checked={on} disabled={locked} onChange={() => toggle(f)} />
        <span style={{ width: 34, textAlign: 'right', fontSize: 10, color: 'var(--muted)' }}>
          {on ? `#${idx + 1}` : ''}
        </span>
        <span className="mt-quant-eff" style={{ width: 12, fontSize: 11 }} title={isAlloc ? 'the quant this pair serves now' : ''}>{isAlloc ? '→' : ''}</span>
        <span className="mt-ql-file" style={{ fontWeight: isAlloc ? 600 : 400 }}>{f}</span>
        {v.bytes ? <span style={{ fontSize: 11, color: 'var(--muted)' }}>{fmtGiB(v.bytes)}</span> : null}
        {isAlloc && <span className="mt-quant-eff" style={{ fontSize: 11 }}>allocated</span>}
        {fit?.text && <span style={{ fontSize: 11, color: grey ? 'var(--danger, #d33)' : 'var(--muted)' }}>{fit.text}</span>}
        <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 2, visibility: on ? 'visible' : 'hidden' }}>
          <button disabled={locked || idx <= 0} onClick={() => move(f, -1)} title="raise priority">▲</button>
          <button disabled={locked || idx >= checked.length - 1} onClick={() => move(f, +1)} title="lower priority">▼</button>
        </span>
      </div>
    )
  }
  return (
    <div className="mt-ql" role="listbox" aria-label="quant priority" title={title}
         style={{ display: 'flex', flexDirection: 'column', minWidth }}>
      {files.map(f => <Row key={f} f={f} />)}
      <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>
        {checked.length
          ? `#1 serves when it fits; lower ranks are the evict-to-fit ladder${busy ? ' · saving…' : ''}`
          : 'nothing checked = the verdict default'}
      </div>
    </div>
  )
}
