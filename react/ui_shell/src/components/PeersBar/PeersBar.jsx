import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import { useFeed } from '../../runtime/feeds'
import './PeersBar.css'

function fmtBytes(n) {
  if (n == null) return '?'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  let v = Number(n)
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${units[i]}`
}

export default function PeersBar() {
  const [peers, setPeers] = useState([])
  const [error, setError] = useState(null)

  // Fed by the one live subscription (runtime/feeds.js), not a 15 s timer.
  const fPeers = useFeed('peers', null)
  useEffect(() => { if (Array.isArray(fPeers)) { setPeers(fPeers); setError(null) } }, [fPeers])

  if (error) {
    return <div className="peers-bar peers-error" title={error}>peers offline</div>
  }
  if (!peers.length) return null

  return (
    <div className="peers-bar">
      <span className="section-title">Peers</span>
      {peers.map(p => {
        const free = p.disk?.free
        const total = p.disk?.total
        const pct = (free != null && total) ? Math.round((1 - free / total) * 100) : null
        const mountClass = p.storage_mounted ? 'peer-mount-ok' : 'peer-mount-bad'
        return (
          <div key={p.name} className={`peer peer-${p.status}`} title={p.storage_root}>
            <span className="peer-dot" />
            <span className="peer-name">{p.name}</span>
            <span className="peer-role">{p.role}</span>
            <span className={`peer-mount ${mountClass}`}>
              {p.storage_mounted ? '✓' : '✗'} {p.storage_root}
            </span>
            {pct != null && (
              <span className="peer-disk">
                {fmtBytes(free)} free · {pct}% used
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
