// Embeds the SEPARATE @hugpy/console broker (terminals / files / keeper) into
// the hugpy UI as a tab. hugpy only *renders* it — the broker stays its own
// WG-only service (failure-domain / attack-surface separation preserved).
//
// Heavy deps (@hugpy/console + xterm) are pulled in here; App.jsx lazy-loads
// this component so they never weigh on the rest of the console.
import { useEffect, useRef, useState, useCallback } from 'react'
import { mountConsole } from '@hugpy/console'
import '@hugpy/console/style.css'
import '@xterm/xterm/css/xterm.css'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { CanvasAddon } from '@xterm/addon-canvas'
import './KeeperConsole.css'

const lsGet = (k, d = '') => { try { return localStorage.getItem('hugpy_console_' + k) ?? d } catch { return d } }
const lsSet = (k, v) => { try { localStorage.setItem('hugpy_console_' + k, v) } catch { /* ignore */ } }

export default function KeeperConsole() {
  const elRef = useRef(null)
  const apiRef = useRef(null)
  const [backend, setBackend] = useState(() => lsGet('backend'))
  const [token, setToken]     = useState(() => lsGet('token'))
  const [target, setTarget]   = useState(() => lsGet('target', 'hugpy'))
  const [mounted, setMounted] = useState(false)
  const [err, setErr]         = useState(null)

  const unmount = useCallback(() => {
    if (apiRef.current) { try { apiRef.current.destroy() } catch { /* ignore */ } apiRef.current = null }
    setMounted(false)
  }, [])

  const mount = useCallback(() => {
    setErr(null)
    unmount()
    lsSet('backend', backend); lsSet('token', token); lsSet('target', target)
    try {
      apiRef.current = mountConsole(elRef.current, {
        backend: backend.trim(),          // '' = same origin (recommended: reverse-proxy the broker here)
        token: token.trim(),
        vm: target.trim() || 'hugpy',
        namespace: 'hugpy-embed',
        xterm: { Terminal, FitAddon, CanvasAddon },
      })
      setMounted(true)
    } catch (e) { setErr(e.message || String(e)) }
  }, [backend, token, target, unmount])

  useEffect(() => () => unmount(), [unmount])  // tear down when the tab unmounts

  return (
    <div className="keeper-console">
      <div className="kc-bar">
        <span className="kc-title">🖥 Station console</span>
        <input className="kc-in" placeholder="broker URL — blank = same origin" value={backend}
               onChange={e => setBackend(e.target.value)} size={32} />
        <input className="kc-in" placeholder="token" type="password" value={token}
               onChange={e => setToken(e.target.value)} size={12} />
        <input className="kc-in" placeholder="target" value={target}
               onChange={e => setTarget(e.target.value)} size={9} />
        {!mounted
          ? <button className="btn-primary" onClick={mount}>Mount</button>
          : <button onClick={unmount}>Unmount</button>}
        {err && <span className="kc-err" title={err}>{err.slice(0, 60)}</span>}
      </div>
      <div className="kc-howto">
        Renders the standalone <code>@hugpy/console</code> broker (terminals · files · keeper). The broker is a
        separate WG-only service — hugpy embeds it, it doesn't host it. For a clean connection, reverse-proxy the
        broker under this same origin (blank URL); otherwise the broker must be reachable over HTTPS with CORS
        allowing this origin.
      </div>
      <div ref={elRef} className="kc-mount" />
    </div>
  )
}
