// CentralConnect — the console shell shown when there is no same-origin hugpy
// central (the public hugpy.ai brochure, or any build served without a backend).
//
// Principle: never a dead end, never a raw 502. The shell ALWAYS loads. hugpy is
// "bring your own central" — this page connects to a central the visitor runs
// (default http://localhost:7002) and shows it as connected or disconnected. If
// disconnected, it shows exactly how to start one (pip install hugpy && hugpy
// serve) and an always-available "open" link — so there is always a way forward.
//
// The probe is a `no-cors` fetch: it resolves if something is listening (even
// without CORS headers) and rejects on connection-refused, which is all we need
// for a connected/disconnected signal. A real local central is same-origin when
// the visitor opens it directly, so the full console there has no CORS issues.
import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import Navbar from '../Navbar/Navbar'
import './CentralConnect.css'

const DEFAULT_CENTRAL = 'http://localhost:7002'
const LS_KEY = 'hugpy.centralUrl'

function normalize(u) {
  let s = (u || '').trim().replace(/\/+$/, '')
  if (s && !/^https?:\/\//i.test(s)) s = 'http://' + s
  return s
}

export default function CentralConnect() {
  const [url, setUrl] = useState(() => {
    try { return localStorage.getItem(LS_KEY) || DEFAULT_CENTRAL } catch { return DEFAULT_CENTRAL }
  })
  const [status, setStatus] = useState('checking') // checking | connected | disconnected

  const probe = useCallback(async (raw) => {
    const base = normalize(raw)
    if (!base) { setStatus('disconnected'); return }
    setStatus('checking')
    try {
      const ctrl = new AbortController()
      const t = setTimeout(() => ctrl.abort(), 2500)
      // no-cors: resolves if a server answers, rejects on connection-refused.
      await fetch(base + '/readiness', { mode: 'no-cors', cache: 'no-store', signal: ctrl.signal })
      clearTimeout(t)
      setStatus('connected')
    } catch {
      setStatus('disconnected')
    }
  }, [])

  useEffect(() => { probe(url) }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const connect = useCallback(() => {
    const base = normalize(url)
    setUrl(base)
    try { localStorage.setItem(LS_KEY, base) } catch { /* private mode */ }
    probe(base)
  }, [url, probe])

  const target = normalize(url) || DEFAULT_CENTRAL

  return (
    <div className="cc">
      <Navbar>
        <span className={`cc-pill cc-${status}`}>
          <span className="cc-dot" />
          central&nbsp;·&nbsp;{status === 'checking' ? 'checking…' : status}
        </span>
      </Navbar>

      <div className="cc-body">
        <h1 className="cc-h1">
          {status === 'connected' ? 'Your hugpy central is up.' : 'Connect a hugpy central.'}
        </h1>
        <p className="cc-lede">
          This is the hugpy front door — it runs no models itself. The console talks to a
          central <em>you</em> run. Point it at one below.
        </p>

        <div className="cc-connect">
          <input
            className="cc-input"
            value={url}
            spellCheck={false}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') connect() }}
            aria-label="hugpy central URL"
          />
          <button className="cc-btn" onClick={connect}>Check</button>
          {/* Always available — never a dead end. Opens the central in place. */}
          <a className={`cc-btn ${status === 'connected' ? 'primary' : ''}`} href={target}>
            Open console →
          </a>
        </div>

        {status === 'disconnected' && (
          <div className="cc-card">
            <div className="cc-card-h">No central is answering at <code>{target}</code>.</div>
            <p>Start one — it's a single pip package (console + API in one process):</p>
            <pre className="cc-cmd">{`pip install hugpy
hugpy serve --port 7002`}</pre>
            <div className="cc-card-f">
              then <button className="cc-link" onClick={connect}>re-check</button>, or
              <a className="cc-link" href={target}> open it</a> once it's running.
            </div>
          </div>
        )}

        {status === 'connected' && (
          <div className="cc-card cc-ok">
            <div className="cc-card-h">Connected to <code>{target}</code>.</div>
            <p>Open the console running on your central:</p>
            <a className="cc-btn primary" href={target}>Open your console →</a>
          </div>
        )}

        <p className="cc-foot">
          <Link to="/">← back to overview</Link>
          <a href="https://pypi.org/project/hugpy/" target="_blank" rel="noreferrer">hugpy on PyPI</a>
        </p>
      </div>
    </div>
  )
}
