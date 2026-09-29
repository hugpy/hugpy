import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
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
// Stage stamps (2026-09-29): queued -> processing -> first token -> processed,
// each written by central AT the transition (not reconstructed at the end), so
// a call that is stuck shows WHERE it is stuck. Rendered as offsets from queued.
const fmtOff = (ts, base) => (ts && base ? `+${(ts - base).toFixed(1)}s` : '—')
function stagesText(r) {
  const q = r.queued_ts || r.started_ts
  const parts = [
    `q ${fmtClock(q)}`,
    `proc ${fmtOff(r.processing_ts, q)}`,
    `tok1 ${fmtOff(r.first_token_ts, q)}`,
    `${r.processed_status || 'done'} ${fmtOff(r.processed_ts || r.ended_ts, q)}`,
  ]
  return parts.join(' → ')
}
function liveStage(r) {
  if (r.processed_ts || r.ended_ts) return null
  if (r.stage === 'prefill') return 'prefill'
  if (r.first_token_ts) return 'streaming'
  if (r.processing_ts) return 'processing'
  return 'queued'
}

function statusClass(s) {
  if (!s) return 'cp-st-unknown'
  if (s === 'done' || s === 'completed') return 'cp-st-ok'
  if (s === 'failed' || s === 'error' || s === 'expired' || s === 'interrupted') return 'cp-st-bad'
  if (s === 'cancelled') return 'cp-st-warn'
  return 'cp-st-live'
}

function jsonLabel(value) {
  if (Array.isArray(value)) return `Array (${value.length})`
  if (value && typeof value === 'object') return `Object (${Object.keys(value).length} keys)`
  const encoded = JSON.stringify(value)
  return short(encoded === undefined ? String(value) : encoded, 120)
}

function JsonNode({ name, value, root = false }) {
  const isArray = Array.isArray(value)
  const isObject = value !== null && typeof value === 'object'
  const entries = isArray ? value.map((v, i) => [String(i), v]) : (isObject ? Object.entries(value) : [])
  return (
    <details className={`cp-json-node${root ? ' cp-json-root' : ''}`} open={root || undefined}>
      <summary><b>{name}</b><span className="cp-json-summary">{jsonLabel(value)}</span></summary>
      {isObject && entries.length > 0
        ? <div className="cp-json-children">{entries.map(([key, child]) => <JsonNode key={key} name={key} value={child} />)}</div>
        : <pre>{JSON.stringify(value, null, 2)}</pre>}
    </details>
  )
}

export default function CallsPanel() {
  const [rows, setRows] = useState([])
  const [expanded, setExpanded] = useState({})
  const [note, setNote] = useState(null)
  const [path, setPath] = useState('')
  const [q, setQ] = useState('')
  const [paused, setPaused] = useState(false)
  const [cancelling, setCancelling] = useState({})

  // Declared BEFORE cancelCall: its deps array names `load`, and a hook's deps
  // array is evaluated during render — a `const` further down is still in its
  // temporal dead zone there ("can't access lexical declaration 'load' before
  // initialization", the live crash of 2026-09-29). Hooks that name each other
  // must be declared in dependency order.
  const load = useCallback(() => {
    fetchJson('/api/llm/calls?limit=400')
      .then(d => {
        if (d && Array.isArray(d.calls)) { setRows(d.calls); setPath(d.path || ''); setNote(null) }
        else setNote('call log reply was not a list (kept the last one)')
      })
      .catch(e => setNote(`call log refresh failed: ${e.message} (kept the last list)`))
  }, [])

  const cancelCall = useCallback(async (row, event) => {
    event.stopPropagation()
    const requestId = row.client_request || row.request_id || row.id
    if (!requestId || cancelling[requestId]) return
    if (!window.confirm(`Cancel this active call on ${row.model || row.model_key || 'the selected model'}?`)) return
    setCancelling(v => ({ ...v, [requestId]: true }))
    try {
      await fetchJson(`/api/llm/chat/cancel/${encodeURIComponent(requestId)}`, { method: 'POST' })
      load()
    } catch (e) {
      setNote(`call cancellation failed: ${e.message}`)
    } finally {
      setCancelling(v => { const next = { ...v }; delete next[requestId]; return next })
    }
  }, [cancelling, load])

  useEffect(() => {
    load()
    if (paused) return
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load, paused])

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter(r => [r.model_key, r.model, r.client, r.peer, r.ua, r.route, r.host, r.principal, r.client_user, r.client_process, r.client_session, r.client_turn, r.client_request, r.client_task, r.worker, r.transport, r.kind, r.locality, r.status, r.error]
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
              <th aria-hidden="true" className="cp-expand-head" />
              <th>When</th><th>Status</th><th>Via</th><th>Client</th><th>User-agent</th><th>Who</th>
              <th>Op</th><th>Model</th><th>Cache</th><th>Worker</th><th className="cp-num">Tokens</th><th className="cp-num">Took</th><th title="queued → processing → first token → processed (offsets from queued; stamped at each transition)">Stages</th><th>Error</th><th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(r => (
            <Fragment key={r.id}>
              <tr className={`cp-call-row${r.error ? ' cp-row-bad' : ''}`} tabIndex={0} aria-expanded={!!expanded[r.id]} title="Click to show the full request" onClick={() => setExpanded(v => ({ ...v, [r.id]: !v[r.id] }))} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded(v => ({ ...v, [r.id]: !v[r.id] })) } }}>
                <td className="cp-expand">{expanded[r.id] ? '▾' : '▸'}</td>
                <td className="cp-mono" title={r.started_ts ? new Date(r.started_ts * 1000).toISOString() : ''}>{fmtClock(r.started_ts)}</td>
                <td><span className={`cp-st ${statusClass(r.status)}`} title={r.processed_status ? `terminal: ${r.processed_status}` : (liveStage(r) ? `live: ${liveStage(r)}` : '')}>{r.status || '—'}{liveStage(r) && liveStage(r) !== r.status ? ` · ${liveStage(r)}` : ''}</span></td>
                <td className="cp-mono" title={r.route || ''}>{r.transport || r.kind || '—'}{r.route ? ` ${short(r.route, 28)}` : ''}</td>
                <td className="cp-mono">{r.client || '—'}</td>
                <td className="cp-mono" title={r.ua || ''}>{short(r.ua, 34)}</td>
                <td>{r.principal || '—'}</td>
                <td className="cp-mono" title={r.kind || ''}>{r.kind || '—'}</td>
                <td className="cp-model" title={r.model_key || ''}>{r.model || short(r.model_key, 42)}</td>
                {/* Storage tier at job start, in the fleet's one vocabulary: the
                    ledger's wire value 'cold' means "served off the shared
                    array", i.e. tier `shared` (workerTier.js). Class keeps the
                    wire value for the existing colours. */}
                <td className={`cp-mono cp-loc cp-loc-${r.locality || 'na'}`} title="storage tier at job start: hot = weights already on the worker's own drive; shared = served off the shared/central array (wire value: cold)">{r.locality === 'cold' ? 'shared' : (r.locality || '—')}</td>
                <td>{r.worker || '—'}</td>
                <td className="cp-num" title={r.total_tokens != null ? `${r.input_tokens || 0} input + ${r.output_tokens || 0} output` : 'streamed token count'}>{r.total_tokens ?? r.tokens ?? 0}</td>
                <td className="cp-num">{fmtDur(r.duration_ms)}</td>
                <td className="cp-mono cp-stages" title={stagesText(r)}>{stagesText(r)}</td>
                <td className="cp-err" title={r.error || ''}>{short(r.error, 60)}</td>
                <td className="cp-action">{(r.status === 'active' || r.status === 'running' || r.status === 'answering' || r.status === 'processing') &&
                  <button onClick={e => cancelCall(r, e)} disabled={!!cancelling[r.client_request || r.request_id || r.id]}>
                    {cancelling[r.client_request || r.request_id || r.id] ? 'cancelling…' : 'cancel'}
                  </button>}</td>
              </tr>
              {expanded[r.id] && <tr className="cp-request-row"><td colSpan={16}>
                <div className="cp-request-meta">
                  <div><b>Caller:</b> {r.client_process || 'process not reported'}{r.client_pid ? ` (pid ${r.client_pid})` : ''} · <b>OS user:</b> {r.client_user || 'not reported'} · <b>HugPy identity:</b> {r.principal || 'unauthenticated'}</div>
                  <div><b>Source:</b> {r.client || '—'} · <b>direct peer:</b> {r.peer || '—'} · <b>forwarded for:</b> {r.forwarded_for || '—'}</div>
                  <div><b>Via:</b> {r.method || '—'} {r.host || ''}{r.route || ''} · <b>client:</b> {r.ua || '—'} · <b>platform:</b> {r.client_platform || '—'}</div>
                  <div><b>Hermes session:</b> {r.client_session || '—'} · <b>turn:</b> {r.client_turn || '—'} · <b>request:</b> {r.client_request || '—'} · <b>task:</b> {r.client_task || '—'}</div>
                </div>
                {r.request
                  ? <div className="cp-json-tree"><JsonNode name="request" value={r.request} root /></div>
                  : <pre>Request body was not recorded for this call.</pre>}
                {r.request && <details className="cp-json-raw"><summary>Raw JSON</summary><pre>{JSON.stringify(r.request, null, 2)}</pre></details>}
              </td></tr>}
            </Fragment>
            ))}
            {!filtered.length && <tr><td colSpan={16} className="cp-dim">no calls recorded yet{path ? ` (log: ${path})` : ''}</td></tr>}
          </tbody>
        </table>
      </div>
      {path && <div className="cp-dim cp-foot">log file: {path}</div>}
    </div>
  )
}
