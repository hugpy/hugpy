// StationDemo — demo.hugpy.ai/station: the hugpy-station showroom as a React
// page (operator 2026-09-02: the demo needs no VM; it lives in /srv/hugpy as a
// component). The canned station SPA (the real 1.0.39 console + demo-shim.js,
// served statically from /srv/hugpy/demo/www/hugpy-station/) is embedded here;
// its "local" surface bridges LIVE to /agent/ (ttyd → `hugpy-agent console`
// running on the host under the jailed hugpy-demo user). The frontier surface is
// canned playback; nothing here reaches the real fleet.
import { useState } from 'react'
import { Helmet } from 'react-helmet-async'
import { Link } from 'react-router-dom'

const TABS = [
  { id: 'station', label: 'hugpy Station', src: '/hugpy-station/', blurb: 'the desktop cockpit — loci, seats, board, steward. Canned data; the local seat is live.' },
  { id: 'agent',   label: 'agent terminal', src: '/agent/',         blurb: 'a live `hugpy-agent console` on a throwaway box, reset nightly.' },
]

export default function StationDemo() {
  const [tab, setTab] = useState(TABS[0])
  return (
    <div className="station-demo" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Helmet><title>hugpy Station — demo</title></Helmet>
      <header style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '8px 14px', borderBottom: '1px solid var(--edge, #30363d)' }}>
        <Link to="/" style={{ fontWeight: 600 }}>hugpy</Link>
        <span style={{ opacity: .6 }}>demo</span>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t)} title={t.blurb}
            style={{ padding: '4px 10px', borderRadius: 6, border: '1px solid var(--edge, #30363d)',
                     background: tab.id === t.id ? 'var(--accent, #4c8dff)' : 'transparent',
                     color: tab.id === t.id ? '#fff' : 'inherit', cursor: 'pointer' }}>{t.label}</button>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: 12, opacity: .7 }}>{tab.blurb}</span>
        <Link to="/console" style={{ fontSize: 12 }}>console showroom →</Link>
      </header>
      <iframe key={tab.id} title={tab.label} src={tab.src}
        style={{ flex: 1, border: 0, width: '100%', background: '#0d1117' }}
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups" />
    </div>
  )
}
