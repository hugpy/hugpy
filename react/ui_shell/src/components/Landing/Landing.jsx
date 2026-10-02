// The product front door, baked into the SPA: guests on an external-auth
// instance land here (PrivateRoute bounces to /welcome); "Open the console"
// continues into /login. Open-mode instances never bounce here, but the page
// stays reachable at /welcome.
import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { getAuthConfig } from '../../Auth/authConfig'
import hugpyLockup from '../../assets/hugpy.png'
import hugpyMark from '../../assets/hugpy-mark.png'
import ReadinessPanel from './ReadinessPanel'
import Navbar from '../Navbar/Navbar'
import './Landing.css'

/* Fleet band — the GPU/phone/API/Discord nodes wired to hugpy central.
   Kept from the "Mesh" landing exploration, restyled to the terminal theme. */
const ico = { width: 22, height: 22, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' }
const GpuIcon   = () => <svg {...ico}><rect x="3" y="6" width="18" height="12" rx="2" /><rect x="6" y="9" width="6" height="6" rx="1" /><path d="M15 9v6M18 9v6" /></svg>
const PhoneIcon = () => <svg {...ico}><rect x="7" y="3" width="10" height="18" rx="2" /><path d="M11 18h2" /></svg>
const KeyIcon   = () => <svg {...ico}><circle cx="8" cy="14" r="4" /><path d="M11 11l8-8M16 6l2 2M19 3l2 2" /></svg>
const ChatIcon  = () => <svg {...ico}><path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12z" /></svg>

function FleetNode({ icon, label, sub }) {
  return (
    <div className="lf-node">
      <div className="lf-ic">{icon}</div>
      <div className="lf-lb">{label}</div>
      <div className="lf-sb">{sub}</div>
    </div>
  )
}

const FEATURES = [
  ['Model registry & downloads',
   'Search the Hugging Face Hub, pick a quantization, and pull with resumable, cancellable jobs. Everything lands in one manifest-backed registry.'],
  ['Streaming chat, never truncated',
   'Token-streamed chat across transformers and llama.cpp backends, with unbounded auto-continuation past per-pass token caps.'],
  ['OpenAI-compatible API',
   '/v1/chat/completions and /v1/models on your own box. Point any OpenAI SDK at it. Mint and revoke API keys from the console — or run it open.'],
  ['GPU worker fleet',
   'Join any box with `hugpy worker`. Central registers, heartbeats, assigns models, probes VRAM fit, and routes requests with local fallback.'],
  ['Cross-machine sharding',
   'Models too big for any single GPU split across the fleet via llama.cpp RPC — a deterministic allocator picks the placement, hugpy runs the lead.'],
  ['One process, one port',
   'Console and API ship in a single pip package. `hugpy serve` and you’re at http://localhost:7002 — no nginx, no node.'],
  ['Phones as a video-analytics pool',
   'Add phones as workers, not just GPUs. The phone-brick pool runs an ONNX-YOLO vision worker on each handset; fan one frame across the fleet and the console resolves a live consensus verdict with per-phone class, confidence, and detections.'],
  ['Discord bot & cross-machine comms',
   'Bind any model — or a Claude keeper — to a Discord channel or user, and the hugpy bot relays both ways. Console↔Discord bridges plus any↔any comms mailboxes let models, keepers, and people talk across the whole fleet.'],
  ['Portable agent fleet',
   'Enroll any box as an autonomous agent that uses your hugpy fleet as its brain — a workspace-jailed toolset, crash-safe memory, and policy gates with operator approval over Discord. One command to install, no local GPU required.'],
]

export default function Landing() {
  const [copied, setCopied] = useState(false)
  // "Console" always goes to /console — never a dead end. There, the gate shows
  // either the live console (backend reachable) or the CentralConnect shell with
  // connect/install info (no backend). Only an external-auth instance needs the
  // login form first. Resolved without useAuth() so <Landing> stays usable
  // outside AuthProvider.
  const [consoleTo, setConsoleTo] = useState('/console')
  useEffect(() => {
    let alive = true
    getAuthConfig().then(c => { if (alive) setConsoleTo(c.mode === 'external' ? '/login' : '/console') })
    return () => { alive = false }
  }, [])
  const copyPip = () => {
    navigator.clipboard?.writeText('pip install hugpy').then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="landing">
      <Helmet>
        <title>Hugpy — Inference You Own · Self-Hosted LLM Server</title>
        <meta name="description" content="A self-hosted LLM console: pull Hugging Face models, expose an OpenAI-compatible API with keys you mint yourself, and pool GPUs across machines — all on your hardware." />
        <link rel="canonical" href="https://hugpy.ai/" />
        <meta property="og:title" content="Hugpy — Inference You Own" />
        <meta property="og:url" content="https://hugpy.ai/" />
      </Helmet>
      {/* No brand on the welcome nav — the hero lockup below already carries it. */}
      <Navbar brand={false} />

      <div className="landing-hero">
        <img className="landing-lockup" src={hugpyLockup} alt="HUGPY AI" />
        <h1>Inference you own<span className="landing-dim">.</span></h1>
        <p className="landing-sub">
          A self-hosted LLM console. Pull models from the Hugging Face Hub, chat with
          streaming auto-continuation, expose an OpenAI-compatible API with keys you
          mint yourself, and pool GPUs across machines — all on your hardware.
        </p>
        <div className="landing-cta-row">
          <Link className="landing-btn primary" to={consoleTo}>Open the console →</Link>
          <a className="landing-btn" href="#quickstart">Run your own</a>
        </div>
        <div className="landing-pip" onClick={copyPip} title="copy">
          <span className="landing-dollar">$</span> pip install hugpy{copied && <span className="landing-copied">✓ copied</span>}
        </div>
      </div>

      {/* Live setup state — only renders on an open (local) instance. */}
      <ReadinessPanel />

      <div className="landing-fleet">
        <FleetNode icon={<GpuIcon />} label="GPU box" sub="--role rpc" />
        <div className="lf-wire" />
        <FleetNode icon={<PhoneIcon />} label="Phone" sub="ONNX-YOLO" />
        <div className="lf-wire" />
        <div className="lf-core">
          <div className="lf-ring"><img src={hugpyMark} width={48} height={48} alt="hugpy" /></div>
          <div className="lf-corelb">hugpy central</div>
        </div>
        <div className="lf-wire" />
        <FleetNode icon={<KeyIcon />} label="OpenAI API" sub="/v1" />
        <div className="lf-wire" />
        <FleetNode icon={<ChatIcon />} label="Discord" sub="relay" />
      </div>

      <section id="features">
        <h2 className="landing-section-title">What's in the box</h2>
        <div className="landing-grid">
          {FEATURES.map(([title, body]) => (
            <div className="landing-card" key={title}>
              <h3><span className="landing-glyph">⬢</span>{title}</h3>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="quickstart">
        <h2 className="landing-section-title">Quickstart</h2>
        <div className="landing-steps">
          <div className="landing-step">
            <div className="landing-step-head">1 · serve</div>
            <pre>{`# console + API in one process
pip install hugpy
hugpy serve --port 7002`}</pre>
          </div>
          <div className="landing-step">
            <div className="landing-step-head">2 · call it like OpenAI</div>
            <pre>{`client = OpenAI(
  base_url="http://localhost:7002/api/v1",
  api_key="hp_…")  # or open mode`}</pre>
          </div>
          <div className="landing-step">
            <div className="landing-step-head">3 · grow the fleet</div>
            <pre>{`# on any GPU box
hugpy worker --central http://your-hugpy:7002
# or lend the GPU to the shard pool
hugpy worker --role rpc`}</pre>
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="landing-footer-links">
          <a href="https://hugpy.ai">hugpy.ai</a>
          <a href="https://x.com/hugpyai" target="_blank" rel="noreferrer">X</a>
          <a href="https://www.instagram.com/hugpy.ai/" target="_blank" rel="noreferrer">Instagram</a>
          <a href="https://www.threads.com/@hugpy.ai" target="_blank" rel="noreferrer">Threads</a>
          <a href="https://www.reddit.com/user/hugpy/" target="_blank" rel="noreferrer">Reddit</a>
          <a href="https://www.minds.com/hugpy/" target="_blank" rel="noreferrer">Minds</a>
          <a href="https://huggingface.co/hugpy-ai" target="_blank" rel="noreferrer">Hugging Face</a>
          <a href="https://github.com/hugpy" target="_blank" rel="noreferrer">GitHub</a>
          <a href="https://pypi.org/project/hugpy/" target="_blank" rel="noreferrer">PyPI</a>
          <a href="https://www.npmjs.com/org/hugpy" target="_blank" rel="noreferrer">npm</a>
        </div>
        <div className="landing-footer-emails">
          <a href="mailto:hello@hugpy.ai">hello@hugpy.ai</a><span className="em-note">social</span>
          <a href="mailto:support@hugpy.ai">support@hugpy.ai</a><span className="em-note">support</span>
          <a href="mailto:abuse@hugpy.ai">abuse@hugpy.ai</a><span className="em-note">abuse reports</span>
        </div>
        © 2026 hugpy · source-available · inference you own.
      </footer>
    </div>
  )
}
