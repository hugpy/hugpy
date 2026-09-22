import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchJson } from '../../api'
import './CallsPanel.css'

// The call log: who called what, from where, when, served by whom, how it
// ended. Read from GET /api/llm/calls (central appends one JSON line per
// request start/end; the route merges them). Added 2026-09-10 after a day of
// hunting callers by hand through nginx logs and job databases.
//
// A poll MUST NEVER DESTROY GOOD DATA: a failed or malformed reply keeps the
// last good list and shows a note.

const POLL_MS = 5000
const fmtClock = ts => (ts ? new Date(ts * 1000).toLocaleTimeString() : '—')
const fmtDur = ms => (ms == null ? '—' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`)
const short = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s || '—')

function statusClass(s) {
  if (!s) return 'cp-st-unknown'
  if (s === 'done' || s === 'completed') return 'cp-st-ok'
  if (s === 'failed' || s === 'error' || s === 'expired') return 'cp-st-bad'
  if (s === 'cancelled') return 'cp-st-warn'
  return 'cp-st-live'
}

export default function CallsPanel() {
  const [rows, setRows] = useState([])
  const [note, setNote] = useState(null)
  const [path, setPath] = useState('')
  const [q, setQ] = useState('')
  const [paused, setPaused] = useState(false)

  const load = useCallback(() => {
    fetchJson('/api/llm/calls?limit=400')
      .then(d => {
        if (d && Array.isArray(d.calls)) { setRows(d.calls); setPath(d.path || ''); setNote(null) }
        else setNote('call log reply was not a list (kept the last one)')
      })
      .catch(e => setNote(`call log refresh failed: ${e.message} (kept the last list)`))
  }, [])

  useEffect(() => {
    load()
    if (paused) return
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load, paused])

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter(r => [r.model_key, r.model, r.client, r.ua, r.route, r.principal, r.worker, r.transport, r.kind, r.locality, r.status, r.error]
      .some(v => v && String(v).toLowerCase().includes(needle)))
  }, [rows, q])

  // "what is calling" at a glance: callers grouped by client + user-agent.
  const byCaller = useMemo(() => {
    const m = new Map()
    for (const r of filtered) {
      const k = `${r.client || '?'} · ${short(r.ua, 40)}`
      const e = m.get(k) || { n: 0, models: new Set(), last: 0 }
      e.n++; if (r.model_key) e.models.add(r.model_key); e.last = Math.max(e.last, r.started_ts || 0)
      m.set(k, e)
    }
    return [...m.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 8)
  }, [filtered])

  return (
    <div className="cp-panel">
      <div className="cp-head">
        <strong>Calls</strong>
        <span className="cp-dim">{filtered.length} shown{q ? ` of ${rows.length}` : ''} · newest first · refreshes every {POLL_MS / 1000} s</span>
        <input className="cp-filter" placeholder="filter: model, client, user-agent, route, worker, status…" value={q} onChange={e => setQ(e.target.value)} />
        <button onClick={() => setPaused(p => !p)}>{paused ? '▶ resume' : '⏸ pause'}</button>
        <button onClick={load}>↻</button>
      </div>
      {note && <div className="cp-note">{note}</div>}

      {byCaller.length > 0 && (
        <div className="cp-callers">
          {byCaller.map(([k, e]) => (
            <span key={k} className="cp-caller" title={[...e.models].join('\n')}>
              <b>{e.n}</b> {k} <span className="cp-dim">· {e.models.size} model{e.models.size === 1 ? '' : 's'} · last {fmtClock(e.last)}</span>
            </span>
          ))}
        </div>
      )}

      <div className="cp-tablewrap">
        <table className="cp-table">
          <thead>
            <tr>
              <th>When</th><th>Status</th><th>Via</th><th>Client</th><th>User-agent</th><th>Who</th>
              <th>Op</th><th>Model</th><th>Cache</th><th>Worker</th><th className="cp-num">Tokens</th><th className="cp-num">Took</th><th>Error</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(r => (
              <tr key={r.id} className={r.error ? 'cp-row-bad' : ''}>
                <td className="cp-mono" title={r.started_ts ? new Date(r.started_ts * 1000).toISOString() : ''}>{fmtClock(r.started_ts)}</td>
                <td><span className={`cp-st ${statusClass(r.status)}`}>{r.status || '—'}</span></td>
                <td className="cp-mono" title={r.route || ''}>{r.transport || r.kind || '—'}{r.route ? ` ${short(r.route, 28)}` : ''}</td>
                <td className="cp-mono">{r.client || '—'}</td>
                <td className="cp-mono" title={r.ua || ''}>{short(r.ua, 34)}</td>
                <td>{r.principal || '—'}</td>
                <td className="cp-mono" title={r.kind || ''}>{r.kind || '—'}</td>
                <td className="cp-model" title={r.model_key || ''}>{r.model || short(r.model_key, 42)}</td>
                <td className={`cp-mono cp-loc cp-loc-${r.locality || 'na'}`} title="model weights already on the worker's local hot drive (hot) vs served off the shared array (cold), at job start">{r.locality || '—'}</td>
                <td>{r.worker || '—'}</td>
                <td className="cp-num">{r.tokens || 0}</td>
                <td className="cp-num">{fmtDur(r.duration_ms)}</td>
                <td className="cp-err" title={r.error || ''}>{short(r.error, 60)}</td>
              </tr>
            ))}
            {!filtered.length && <tr><td colSpan={13} className="cp-dim">no calls recorded yet{path ? ` (log: ${path})` : ''}</td></tr>}
          </tbody>
        </table>
      </div>
      {path && <div className="cp-dim cp-foot">log file: {path}</div>}
    </div>
  )
}
