// Dev-grade rendering of server diagnostics objects (chat errors, model
// failure logs). Shape-agnostic on purpose — the server's `diagnostics`
// payload is still evolving — so: scalars -> key/value rows, arrays of
// objects -> one table (union of keys as columns), objects of objects
// (e.g. {workerName: {...}}) -> one table keyed by the outer key, anything
// else -> pretty JSON. Nothing is ever truncated.
import { useState } from 'react'
import { hugpyFetch, resolveApiUrl } from '../../runtime/config'
import './Diagnostics.css'

const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v)
const cell = (v) => (v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v))

export function toJsonText(value) {
  if (typeof value === 'string') return value
  try { return JSON.stringify(value, null, 2) } catch { return String(value) }
}

export function CopyButton({ text, label = 'copy', className = '' }) {
  const [state, setState] = useState('')
  const copy = async () => {
    const t = typeof text === 'function' ? text() : text
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(t)
      else {
        const ta = document.createElement('textarea')
        ta.value = t; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove()
      }
      setState('copied')
    } catch (e) { setState(`copy failed: ${e?.message || e}`) }
    setTimeout(() => setState(''), 1800)
  }
  return <button type="button" className={`diag-copy ${className}`} onClick={copy}>{state || label}</button>
}

function RowsTable({ rows, keyLabel }) {
  const cols = []
  for (const [, r] of rows) for (const k of Object.keys(isObj(r) ? r : { value: r })) if (!cols.includes(k)) cols.push(k)
  return <div className="diag-tablewrap"><table className="diag-table"><thead><tr>
    {keyLabel && <th>{keyLabel}</th>}{cols.map(c => <th key={c}>{c}</th>)}
  </tr></thead><tbody>{rows.map(([k, r], i) => {
    const o = isObj(r) ? r : { value: r }
    return <tr key={`${k}:${i}`}>{keyLabel && <th>{k}</th>}{cols.map(c => <td key={c}>{cell(o[c])}</td>)}</tr>
  })}</tbody></table></div>
}

export function DiagnosticsView({ value }) {
  if (value == null) return null
  if (!isObj(value) && !Array.isArray(value)) return <pre className="diag-pre">{String(value)}</pre>
  if (Array.isArray(value)) {
    return value.every(isObj) && value.length
      ? <RowsTable rows={value.map((r, i) => [i, r])} />
      : <pre className="diag-pre">{toJsonText(value)}</pre>
  }
  const scalars = Object.entries(value).filter(([, v]) => !isObj(v) && !Array.isArray(v))
  const nested = Object.entries(value).filter(([, v]) => isObj(v) || Array.isArray(v))
  return <div className="diag-view">
    {!!scalars.length && <table className="diag-kv"><tbody>{scalars.map(([k, v]) =>
      <tr key={k}><th>{k}</th><td>{cell(v)}</td></tr>)}</tbody></table>}
    {nested.map(([k, v]) => {
      let body
      if (Array.isArray(v) && v.length && v.every(isObj)) body = <RowsTable rows={v.map((r, i) => [i, r])} />
      else if (isObj(v) && Object.keys(v).length && Object.values(v).every(isObj)) body = <RowsTable rows={Object.entries(v)} keyLabel="key" />
      else body = <pre className="diag-pre">{toJsonText(v)}</pre>
      return <div key={k} className="diag-section"><div className="diag-section-title">{k}</div>{body}</div>
    })}
  </div>
}

// ── whole logs (2026-09-23) ─────────────────────────────────────────────────
// "everything that says logs needs to be logs": a log renders WHOLE in a
// scrollable box, with where it was read from, its byte count, a copy button
// and an "open raw" link to the server's text/plain copy (GET /llm/logs).
export const logRawPath = (ref) => `/api/llm/logs?ref=${encodeURIComponent(ref)}`
export const logRawUrl = (ref) => resolveApiUrl(logRawPath(ref))
const utf8Bytes = (t) => { try { return new TextEncoder().encode(t).length } catch { return t.length } }

export function RawLink({ logRef, label = 'open raw' }) {
  const [err, setErr] = useState('')
  if (!logRef) return null
  // Fetch through hugpyFetch (carries configured auth headers) and open the
  // text as a blob; the href still works for middle-click / copy-link.
  const open = async (e) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return
    e.preventDefault()
    setErr('')
    try {
      const r = await hugpyFetch(logRawPath(logRef))
      const text = await r.text()
      const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
      window.open(url, '_blank', 'noopener')
      setTimeout(() => URL.revokeObjectURL(url), 60000)
      if (!r.ok) setErr(`HTTP ${r.status}`)
    } catch (x) { setErr(String(x?.message || x)) }
  }
  return <a className="diag-raw" href={logRawUrl(logRef)} target="_blank" rel="noopener noreferrer"
    onClick={open} title={`GET /llm/logs?ref=${logRef} (text/plain, whole)`}>{label}{err ? ` (${err})` : ''}</a>
}

// ``text``: the log itself. ``source``: what was read (e.g. "compute_actions#12
// detail.loader_stderr"). ``logRef``: the server-side ref for the raw link.
// ``title``: an optional heading (a verdict/summary) shown ABOVE the log.
export function LogBlock({ text, source, logRef, bytes, title, className = '' }) {
  const t = text == null ? '' : typeof text === 'string' ? text : toJsonText(text)
  const n = bytes ?? utf8Bytes(t)
  return <div className={`diag-log ${className}`}>
    {title && <div className="diag-log-title">{title}</div>}
    <div className="diag-log-meta">
      <span>source {source || 'unspecified'} · {n} bytes{logRef ? ` · log_ref ${logRef}` : ''}</span>
      {t && <CopyButton text={t} />}
      <RawLink logRef={logRef} />
    </div>
    {t ? <pre className="diag-pre diag-log-pre">{t}</pre>
      : <pre className="diag-pre diag-log-pre diag-log-empty">{`(empty: read ${source || 'the log field'}${logRef ? ` (${logRef})` : ''}, bytes=0)`}</pre>}
  </div>
}
