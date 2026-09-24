// HelpPanel — the console Help button's slide-over: a persistent chat with the
// hugpy help agent (POST/GET /api/llm/help/sessions…, routes/help_routes.py).
//
// * Operator-only: the server gate decides; a 401/403 is shown plainly here.
// * The session id persists in localStorage (and `?help=<id>` in the URL wins),
//   so reloading the console reopens the same conversation.
// * Replies stream from /sessions/<id>/stream (SSE over fetch, so the configured
//   auth headers ride along); a broken stream falls back to polling.
// * The context strip is built from where the operator is (route/tab, selected
//   model, worker from the URL, last console error) — pure buildHelpContext().
//   openHelpPanel({error}) from anywhere puts THAT error into the strip.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { buildHelpContext, recordsToView } from './helpModel'
import { OPEN_EVENT, getLastConsoleError } from './helpBus'
import { openHelpWidget } from '../../../../ui_shared/help/helpWidget'
import './HelpPanel.css'

const LS_KEY = 'hugpy.help.session'
const BASE = '/api/llm/help'

function readSaved() {
  try {
    const q = new URLSearchParams(window.location.search).get('help')
    if (q && /^hs-[0-9a-f]{12}$/.test(q)) return q
    return localStorage.getItem(LS_KEY) || ''
  } catch { return '' }
}

function save(id) {
  try { id ? localStorage.setItem(LS_KEY, id) : localStorage.removeItem(LS_KEY) } catch { /* private mode */ }
}

async function call(path, init) {
  const r = await hugpyFetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...((init && init.headers) || {}) },
  })
  let body = null
  try { body = await r.json() } catch { body = null }
  return { status: r.status, ok: r.ok, body }
}

function gateMessage(status, body) {
  if (status === 401) return 'Operator sign-in required. The hugpy help agent can read logs and edit code, so it is operator-only.'
  if (status === 403) return 'Your account is not an operator. The hugpy help agent is operator-only (it can read logs and edit code).'
  if (status === 503) return (body && body.error) || 'The help agent is disabled on this deployment.'
  return (body && body.error) || `HTTP ${status}`
}

export default function HelpPanel({ tab = '', model = '', error = '' }) {
  const [open, setOpen] = useState(false)
  const [sid, setSid] = useState(readSaved)
  const [records, setRecords] = useState([])
  const [meta, setMeta] = useState(null)          // {backend, backend_label, tools, status}
  const [gate, setGate] = useState('')            // non-empty => refused
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessions, setSessions] = useState([])
  const [explicit, setExplicit] = useState({})    // context handed in by openHelpPanel
  const [useCtx, setUseCtx] = useState(true)
  const [dropped, setDropped] = useState({})
  const followRef = useRef(0)
  const bodyRef = useRef(null)
  const inputRef = useRef(null)

  const ctx = useMemo(() => {
    const loc = typeof window !== 'undefined' ? window.location : {}
    return buildHelpContext({
      pathname: loc.pathname, search: loc.search, tab, model,
      error: explicit.error || error || getLastConsoleError(),
      worker: explicit.worker, extra: explicit.context, surface: 'console',
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, tab, model, error, explicit, records.length])
  const ctxItems = ctx.items.filter(i => !dropped[i.key])
  const ctxPayload = useMemo(() => {
    const p = {}
    ctxItems.forEach(i => { p[i.key] = i.value })
    return p
  }, [ctxItems])

  // ---- open via Navbar button / openHelpPanel() / FAB override -------------
  useEffect(() => {
    const onOpen = (e) => {
      const d = (e && e.detail) || {}
      setExplicit({ error: d.error || '', worker: d.worker || '', context: d.context || null })
      setDropped({})
      if (d.prompt) setInput(String(d.prompt))
      setOpen(true)
      setTimeout(() => inputRef.current && inputRef.current.focus(), 0)
    }
    window.addEventListener(OPEN_EVENT, onOpen)
    return () => window.removeEventListener(OPEN_EVENT, onOpen)
  }, [])

  const recordsRef = useRef([])
  useEffect(() => { recordsRef.current = records }, [records])

  // t: a full transcript ({backend, status, records}) or a bare {records:[…]}
  // from the stream — only a full transcript touches the header meta.
  const merge = useCallback((t) => {
    if (!t) return
    if ('backend' in t) {
      setMeta({ backend: t.backend, backend_label: t.backend_label, tools: t.tools, status: t.status })
    }
    setRecords(prev => {
      const byI = new Map(prev.map(r => [r.i, r]))
      ;(t.records || []).forEach(r => byI.set(r.i, r))
      return [...byI.values()].sort((a, b) => a.i - b.i)
    })
  }, [])

  const load = useCallback(async (id) => {
    const r = await call(`/sessions/${id}`)
    if (r.status === 401 || r.status === 403 || r.status === 503) { setGate(gateMessage(r.status, r.body)); return null }
    if (r.status === 404) { save(''); setSid(''); setRecords([]); setMeta(null); return null }
    if (!r.ok) return null
    setGate('')
    setRecords([])
    merge(r.body)
    return r.body
  }, [merge])

  // ---- follow a busy session: SSE stream, polling fallback -----------------
  const follow = useCallback(async (id) => {
    const token = ++followRef.current
    const alive = () => followRef.current === token
    const have = recordsRef.current.filter(r => Number.isInteger(r.i))
    let since = have.length ? have[have.length - 1].i + 1 : 0
    for (let round = 0; round < 400 && alive(); round++) {
      let status = 'busy'
      try {
        const resp = await hugpyFetch(`${BASE}/sessions/${id}/stream?since=${since}`)
        if (!resp.ok || !resp.body) throw new Error(`stream ${resp.status}`)
        const reader = resp.body.getReader()
        const dec = new TextDecoder()
        let buf = ''
        while (alive()) {
          const { value, done } = await reader.read()
          if (done) break
          buf += dec.decode(value, { stream: true })
          let cut
          while ((cut = buf.indexOf('\n\n')) >= 0) {
            const frame = buf.slice(0, cut)
            buf = buf.slice(cut + 2)
            const ev = (frame.match(/^event: (.*)$/m) || [])[1] || 'message'
            const data = frame.split('\n').filter(l => l.startsWith('data: ')).map(l => l.slice(6)).join('\n')
            if (!data) continue
            let obj
            try { obj = JSON.parse(data) } catch { continue }
            if (ev === 'status') {
              status = obj.status
              setMeta(m => (m ? { ...m, status: obj.status } : m))
            } else {
              since = Math.max(since, obj.i + 1)
              merge({ records: [obj] })
            }
          }
        }
        if (!alive()) return
        // re-read meta (backend label etc.) and final status
        const r = await call(`/sessions/${id}?since=${since}`)
        if (r.ok) { merge(r.body); status = r.body.status; since = r.body.count }
      } catch {
        // polling fallback
        await new Promise(res => setTimeout(res, 1500))
        const r = await call(`/sessions/${id}?since=${since}`)
        if (r.ok) { merge(r.body); status = r.body.status; since = r.body.count }
      }
      if (status !== 'busy') return
    }
  }, [merge])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    ;(async () => {
      const list = await call('/sessions')
      if (cancelled) return
      if (list.status === 401 || list.status === 403 || list.status === 503) { setGate(gateMessage(list.status, list.body)); return }
      setGate('')
      if (list.ok) setSessions((list.body && list.body.sessions) || [])
      if (sid) {
        const t = await load(sid)
        if (t && t.status === 'busy') follow(sid)
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, sid])

  useEffect(() => () => { followRef.current++ }, [])
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [records.length, open])

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setBusy(true)
    try {
      const context = useCtx ? ctxPayload : null
      const r = sid
        ? await call(`/sessions/${sid}/messages`, { method: 'POST', body: JSON.stringify({ text, context }) })
        : await call('/sessions', { method: 'POST', body: JSON.stringify({ prompt: text, context }) })
      if (r.status === 401 || r.status === 403 || r.status === 503) { setGate(gateMessage(r.status, r.body)); return }
      if (!r.ok) {
        setRecords(prev => [...prev, { kind: 'error', i: (prev.length ? prev[prev.length - 1].i : 0) + 0.5, error: gateMessage(r.status, r.body) }])
        return
      }
      setInput('')
      const id = r.body.id
      if (id !== sid) { save(id); setSid(id); setRecords([]) }
      merge(r.body)
      setExplicit({})
      follow(id)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    if (!sid) return
    const r = await call(`/sessions/${sid}/stop`, { method: 'POST' })
    if (r.ok) merge(r.body)
  }

  const newSession = () => {
    followRef.current++
    save('')
    setSid('')
    setRecords([])
    setMeta(null)
    setTimeout(() => inputRef.current && inputRef.current.focus(), 0)
  }

  const pick = (id) => {
    followRef.current++
    save(id)
    setRecords([])
    setSid(id)
  }

  const view = recordsToView(records)
  const status = meta && meta.status
  if (!open) return null

  return (
    <div className="hp-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) setOpen(false) }}>
      <aside className="hp-panel" role="dialog" aria-label="hugpy help agent">
        <header className="hp-head">
          <div>
            <div className="hp-title">hugpy help agent</div>
            <div className="hp-sub">
              {meta && meta.backend
                ? <span className={meta.tools ? 'hp-tag tools' : 'hp-tag ro'}>{meta.tools ? 'claude arm · logs + code' : 'local model, read-only'}</span>
                : <span className="hp-tag">operator · logs, code, tests</span>}
              {status && <span className={`hp-status ${status}`}>{status}</span>}
            </div>
          </div>
          <div className="hp-head-actions">
            {sessions.length > 0 && (
              <select className="hp-select" value={sid} onChange={e => (e.target.value ? pick(e.target.value) : newSession())} aria-label="Help sessions">
                <option value="">new session…</option>
                {sessions.map(s => <option key={s.id} value={s.id}>{(s.title || s.id).slice(0, 48)}</option>)}
              </select>
            )}
            <button type="button" className="hp-btn ghost" onClick={newSession} title="Start a new help session">New</button>
            <button type="button" className="hp-x" onClick={() => setOpen(false)} aria-label="Close help">×</button>
          </div>
        </header>

        {gate ? (
          <div className="hp-body">
            <div className="hp-msg error">{gate}</div>
            <div className="hp-msg note">
              Members can still ask the Keeper (read-only answers, fix requests go to an operator):{' '}
              <button type="button" className="hp-btn ghost" onClick={() => { setOpen(false); openHelpWidget() }}>Ask the Keeper</button>
            </div>
          </div>
        ) : (
          <>
            <div className="hp-body" ref={bodyRef}>
              {!view.length && (
                <div className="hp-msg note">
                  Ask about anything in hugpy: a failed load, an error on screen, a worker acting up. The agent
                  reads the central/worker logs, compute-actions and the audit report, reads and edits code in
                  /srv/hugpy/src/hugpy, runs tests and can queue a verify. It never restarts services, promotes,
                  touches worker boxes or deletes files; promotion stays your call.
                </div>
              )}
              {view.map((v, n) => (
                <div key={`${v.i}-${n}`} className={`hp-msg ${v.kind}${v.error ? ' bad' : ''}`}>
                  {v.kind === 'activity' ? <span className="hp-act">▸ {v.text}</span> : v.text}
                  {v.kind === 'user' && v.context && <div className="hp-ctx-echo">{v.context}</div>}
                </div>
              ))}
              {status === 'busy' && <div className="hp-msg note hp-working">working…</div>}
            </div>

            <div className="hp-ctx">
              <label className="hp-check">
                <input type="checkbox" checked={useCtx} onChange={e => setUseCtx(e.target.checked)} />
                <span>context</span>
              </label>
              {useCtx && ctxItems.map(i => (
                <span key={i.key} className={`hp-chip ${i.key === 'error' ? 'err' : ''}`} title={i.value}>
                  <b>{i.label}</b> {i.value.length > 60 ? i.value.slice(0, 60) + '…' : i.value}
                  <button type="button" aria-label={`drop ${i.label}`} onClick={() => setDropped(d => ({ ...d, [i.key]: true }))}>×</button>
                </span>
              ))}
              {useCtx && !ctxItems.length && <span className="hp-chip empty">none</span>}
            </div>

            <footer className="hp-foot">
              <textarea
                ref={inputRef}
                rows={3}
                value={input}
                placeholder="What failed to load on aeb today and why?"
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                aria-label="Ask the hugpy help agent"
              />
              <div className="hp-row">
                {sid && <span className="hp-sid" title="session id (transcript kept under PROJECTS_HOME/help_sessions)">{sid}</span>}
                <span style={{ flex: 1 }} />
                {status === 'busy' && <button type="button" className="hp-btn ghost" onClick={stop}>Stop</button>}
                <button type="button" className="hp-btn" onClick={send} disabled={busy || !input.trim()}>{busy ? '…' : 'Send'}</button>
              </div>
            </footer>
          </>
        )}
      </aside>
    </div>
  )
}
