// Live "Your instance" panel for the Landing — the self-hoster onboarding.
//
// Fetches GET /api/readiness (never-5xx backend probe; see
// abstract_hugpy_dev/flask_app/app/routes/welcome_routes.py) and renders what's
// ready vs missing with the exact remediation command. It GUIDES instead of
// letting the user walk into errors.
//
// Shown only on an OPEN instance (a local `hugpy serve`). On the public,
// external-auth dev.hugpy.ai the Landing stays a clean marketing front door, and
// instance internals aren't surfaced to anonymous guests. If readiness is
// unreachable we render nothing — the marketing page is never broken by it.
import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import './ReadinessPanel.css'

const DOT = { ok: 'rp-ok', warn: 'rp-warn', bad: 'rp-bad', muted: 'rp-muted' }

function Row({ state, title, desc, fix, link }) {
  return (
    <div className="rp-row">
      <span className={`rp-dot ${DOT[state] || 'rp-muted'}`} />
      <div className="rp-bodyc">
        <div className="rp-title">{title}</div>
        {desc && <div className="rp-desc">{desc}</div>}
        {fix && <pre className="rp-fix">{fix}</pre>}
        {link && <a className="rp-link" href={link.href}>{link.label}</a>}
      </div>
    </div>
  )
}

export default function ReadinessPanel() {
  const [r, setR] = useState(null)

  useEffect(() => {
    let alive = true
    fetchJson('/api/readiness')
      .then(d => { if (alive) setR(d) })
      .catch(() => { /* unreachable → stay hidden, never break the page */ })
    return () => { alive = false }
  }, [])

  if (!r) return null
  // Only on an open (local self-host) instance.
  if (r.central && r.central.auth_mode && r.central.auth_mode !== 'open') return null

  // Prefer the real origin the user is viewing (reliable behind a proxy) over
  // the server's request.host_url, which reads 127.0.0.1:7002 behind nginx.
  const base = (typeof window !== 'undefined' && window.location && window.location.origin)
            || (r.connect && r.connect.base_url) || ''
  const st = r.storage || {}
  const sv = r.serving || {}
  const cs = r.console || {}
  const hasModels = (st.model_count || 0) > 0

  return (
    <section className="rp" id="your-instance">
      <h2 className="landing-section-title">
        Your instance{r.central && r.central.version ? ` · v${r.central.version}` : ''}
      </h2>
      <div className="rp-cards">
        <Row
          state={st.error ? 'bad' : hasModels ? 'ok' : st.exists ? 'warn' : 'bad'}
          title={`Model storage${st.free_gb != null ? ` — ${st.free_gb} GB free` : ''}`}
          desc={st.error ? `Could not read storage: ${st.error}`
              : st.exists ? `${st.root} — ${st.model_count || 0} model(s)${st.writable ? '' : ' (not writable)'}`
              : `Storage root ${st.root || 'unset'} does not exist yet.`}
          fix={!st.error && !hasModels
              ? 'export DEFAULT_ROOT=/path/with/space\nhugpy models pull <model-key>   # or drop a .gguf there'
              : null}
        />
        <Row
          state={sv.any_serving ? 'ok' : 'warn'}
          title="A model is serving"
          desc={sv.any_serving ? 'At least one slot is live.'
              : sv.enabled === false ? 'Model slots are not enabled on this install.'
              : 'No slot is serving a model yet.'}
          fix={sv.any_serving ? null : `curl -s ${base}/api/llm/slots   # check slot health`}
        />
        <Row
          state="ok"
          title="Connect a bot or worker"
          desc="Point any hugpy arm at this central:"
          fix={`export HUGPY_BASE_URL=${base}\nhugpy bot       # or: hugpy worker --central ${base}`}
        />
        {cs.configured
          ? <Row state="ok" title="Console"
                 desc="A delegated console endpoint is configured."
                 link={{ href: cs.url, label: 'Open console →' }} />
          : <Row state="muted" title="Console (optional)"
                 desc="Run the separate @hugpy/console broker on its own port and point hugpy at it — hugpy never spawns the shell itself, it only links to the delegated endpoint."
                 fix={'python3 pty_broker.py --resolver local --port 8801\nexport HUGPY_CONSOLE_URL=http://<host>:8801'} />}
      </div>
    </section>
  )
}
