import React, { useEffect, useMemo, useState } from 'react'
import { fetchJson } from '../../api'
import { consoleTraceEnabled, sendConsoleTraceEvent, setConsoleTraceEnabled, resolveApiUrl } from '../../runtime/config'
import './ConsoleTracePanel.css'

export default function ConsoleTracePanel() {
  const [enabled, setEnabled] = useState(consoleTraceEnabled())
  const [events, setEvents] = useState([])
  const [requests, setRequests] = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState('')
  const [browserUrl, setBrowserUrl] = useState(() => window.location.href)
  useEffect(() => {
    if (!enabled) return undefined
    sendConsoleTraceEvent('browser.capture.started', { href: window.location.href })
    let source
    try {
      source = new EventSource(resolveApiUrl('/api/console/trace/stream'))
      source.onmessage = (event) => {
        try { const row = JSON.parse(event.data); setEvents(v => [...v.slice(-499), row]) } catch { /* ignore malformed */ }
      }
      source.onerror = () => setError('live stream reconnecting')
    } catch (e) { setError(String(e)) }
    const poll = () => fetchJson('/api/console/trace?limit=100').then(r => setRequests(r.requests || [])).catch(e => setError(String(e)))
    poll(); const timer = setInterval(poll, 2000)
    return () => { clearInterval(timer); source?.close() }
  }, [enabled])
  const toggle = () => { const next = !enabled; setConsoleTraceEnabled(next); setEnabled(next) }
  const load = (id) => fetchJson(`/api/console/trace?trace_id=${encodeURIComponent(id)}`).then(setSelected).catch(e => setError(String(e)))
  const spanCount = useMemo(() => selected?.spans?.length || 0, [selected])
  return <section className="ct-panel">
    <div className="ct-head"><div><h2>Console Trace</h2><p>Persistent browser → Flask → Python call/return feed in the console DB.</p></div>
      <div className="ct-actions"><button className={enabled ? 'ct-stop' : 'ct-start'} onClick={toggle}>{enabled ? 'Stop capture' : 'Start capture'}</button>
      <input value={browserUrl} onChange={e => setBrowserUrl(e.target.value)} aria-label="browser URL" />
      <button className="ct-open" onClick={() => { setConsoleTraceEnabled(true); setEnabled(true); window.open(browserUrl, '_blank', 'noopener,noreferrer') }}>Open traced browser</button></div></div>
    {error && <div className="ct-error">{error}</div>}
    <div className="ct-grid">
      <div className="ct-card"><h3>Live feed <span>{events.length}</span></h3><div className="ct-feed">{events.length ? events.slice().reverse().map((e, i) => <div className="ct-event" key={`${e.id}-${i}`}><time>{new Date(e.at * 1000).toLocaleTimeString()}</time><b>{e.kind}</b><code>{JSON.stringify(e.payload)}</code></div>) : <div className="ct-empty">Start capture, then use the console.</div>}</div></div>
      <div className="ct-card"><h3>Flask requests</h3><div className="ct-feed">{requests.map(r => <button className="ct-request" key={r.id} onClick={() => load(r.id)}><b>{r.status || '…'} {r.method}</b> <span>{r.path}</span><small>{r.duration_ms == null ? 'running' : `${Math.round(r.duration_ms)} ms`} · {r.id.slice(0, 10)}</small></button>)}{!requests.length && <div className="ct-empty">No traced requests yet.</div>}</div></div>
    </div>
    {selected && <div className="ct-card ct-detail"><h3>Execution tree · {selected.request?.method} {selected.request?.path} <span>{spanCount} events</span></h3><pre>{selected.spans.map(s => `${'  '.repeat(Math.min(s.depth, 30))}${s.event} ${s.function} — ${s.file}:${s.line}${s.duration_ms == null ? '' : ` (${s.duration_ms.toFixed(2)} ms)`}${s.error ? ` :: ${s.error}` : ''}`).join('\n')}</pre></div>}
  </section>
}
