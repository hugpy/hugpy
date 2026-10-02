// Landing.jsx — HUGPY product front door (enriched)
// Merge of the "Mesh" hero + fleet band + bento features and the "Atlas"
// reference section. Drop-in for the SPA: same react-router wiring and the
// app's CSS custom properties (--bg, --accent, --surface-1, …) so it inherits
// the live theme automatically. The Constellation mark is inline SVG (no new
// image asset); the backdrop canvas reads --accent at runtime.
import { useState, useRef, useEffect, useId } from 'react'
import { Link } from 'react-router-dom'
import hugpyLockup from '../../assets/hugpy.png'
import './Landing.css'

/* ------------------------------------------------------------------ *
 * Brand "Constellation" symbol — inline so it scales + glows crisply  *
 * ------------------------------------------------------------------ */
function HugpyMark({ size = 120, className, style }) {
  const uid = useId().replace(/:/g, '')
  const sd = `sd-${uid}`, ad = `ad-${uid}`
  return (
    <svg className={className} style={style} width={size} height={size}
      viewBox="0 0 240 240" role="img" aria-label="hugpy">
      <defs>
        <linearGradient id={sd} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4f6f8" /><stop offset=".52" stopColor="#c6cbd1" /><stop offset="1" stopColor="#878d95" />
        </linearGradient>
        <linearGradient id={ad} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#9fd9e4" /><stop offset="1" stopColor="#4d93ac" />
        </linearGradient>
      </defs>
      <g fill="none" stroke="#9aa1a9" strokeOpacity="0.3" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
        <line x1="120" y1="40" x2="104" y2="152" /><line x1="184" y1="80" x2="96" y2="96" />
        <line x1="184" y1="160" x2="86" y2="124" /><line x1="56" y1="160" x2="150" y2="92" />
        <line x1="56" y1="80" x2="160" y2="140" /><line x1="120" y1="200" x2="150" y2="92" />
        <line x1="96" y1="96" x2="160" y2="140" /><line x1="150" y1="92" x2="104" y2="152" />
      </g>
      <g fill="none" stroke="#9aa1a9" strokeOpacity="0.5" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="96" y1="96" x2="150" y2="92" /><line x1="150" y1="92" x2="140" y2="118" />
        <line x1="140" y1="118" x2="160" y2="140" /><line x1="160" y1="140" x2="104" y2="152" />
        <line x1="104" y1="152" x2="86" y2="124" /><line x1="86" y1="124" x2="96" y2="96" />
      </g>
      <g fill="none" stroke={`url(#${sd})`} strokeWidth="4.4" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="120,40 184,80 184,160 120,200 56,160 56,80" />
      </g>
      <g fill="#9aa1a9" fillOpacity="0.7">
        <circle cx="96" cy="96" r="2.5" /><circle cx="150" cy="92" r="2.5" /><circle cx="160" cy="140" r="2.5" />
        <circle cx="104" cy="152" r="2.5" /><circle cx="86" cy="124" r="2.5" /><circle cx="140" cy="118" r="2.5" />
      </g>
      <circle cx="120" cy="120" r="10.5" fill={`url(#${ad})`} fillOpacity=".14" />
      <circle cx="120" cy="120" r="5.5" fill={`url(#${ad})`} />
      <circle cx="120" cy="120" r="5.5" fill="none" stroke="#dfeef2" strokeOpacity=".5" strokeWidth="1" />
    </svg>
  )
}

/* ------------------------------------------------------------------ *
 * Animated constellation field — drifting nodes linking to a lit core *
 * Reads --accent off the canvas so it tracks the app theme.           *
 * ------------------------------------------------------------------ */
function toRgb(ctx, color, fallback) {
  try { ctx.fillStyle = color; const h = ctx.fillStyle
    if (h[0] === '#') { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255] } } catch {}
  return fallback
}
function Constellation({ core = true, coreY = 0.36, density = 0.00008 }) {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current; if (!cv) return
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = cv.getContext('2d')
    const cs = getComputedStyle(cv)
    const accent = toRgb(ctx, cs.getPropertyValue('--accent').trim() || '#56d4e8', [86, 212, 232])
    const fil = [154, 161, 169]
    let raf, w, h, nodes, dpr
    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      const r = cv.getBoundingClientRect(); w = r.width; h = r.height
      cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const n = Math.max(24, Math.round(w * h * density))
      nodes = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.16, vy: (Math.random() - 0.5) * 0.16, r: Math.random() * 1.3 + 0.6 }))
    }
    const link = 130
    const draw = () => {
      ctx.clearRect(0, 0, w, h); const cx = w * 0.5, cy = h * coreY
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]; a.x += a.vx; a.y += a.vy
        if (a.x < 0 || a.x > w) a.vx *= -1; if (a.y < 0 || a.y > h) a.vy *= -1
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy)
          if (d < link) { ctx.strokeStyle = `rgba(${fil},${(1 - d / link) * 0.22})`; ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke() }
        }
        if (core) { const dc = Math.hypot(a.x - cx, a.y - cy)
          if (dc < link * 1.5) { const o = (1 - dc / (link * 1.5)) * 0.4
            const g = ctx.createLinearGradient(a.x, a.y, cx, cy)
            g.addColorStop(0, `rgba(${fil},${o * 0.4})`); g.addColorStop(1, `rgba(${accent},${o})`)
            ctx.strokeStyle = g; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(cx, cy); ctx.stroke() }
        }
      }
      for (const a of nodes) { ctx.fillStyle = 'rgba(198,203,209,.55)'; ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, 7); ctx.fill() }
      if (core) { const p = 0.5 + Math.sin(Date.now() / 625) * 0.5
        const rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60 + p * 14)
        rg.addColorStop(0, `rgba(${accent},${0.22 + p * 0.12})`); rg.addColorStop(0.5, `rgba(${accent},.07)`); rg.addColorStop(1, `rgba(${accent},0)`)
        ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(cx, cy, 60 + p * 14, 0, 7); ctx.fill()
        ctx.fillStyle = `rgb(${accent})`; ctx.beginPath(); ctx.arc(cx, cy, 3.2, 0, 7); ctx.fill() }
      raf = requestAnimationFrame(draw)
    }
    resize(); draw()
    const ro = new ResizeObserver(resize); ro.observe(cv)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [core, coreY, density])
  return <canvas ref={ref} className="landing-constellation" aria-hidden="true" />
}

/* tiny line icons for the fleet band */
const ICO = { width: 22, height: 22, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' }
const GpuIcon = () => <svg {...ICO}><rect x="3" y="6" width="18" height="12" rx="2" /><rect x="6" y="9" width="6" height="6" rx="1" /><path d="M15 9v6M18 9v6" /></svg>
const PhoneIcon = () => <svg {...ICO}><rect x="7" y="3" width="10" height="18" rx="2" /><path d="M11 18h2" /></svg>
const KeyIcon = () => <svg {...ICO}><circle cx="8" cy="14" r="4" /><path d="M11 11l8-8M16 6l2 2M19 3l2 2" /></svg>
const ChatIcon = () => <svg {...ICO}><path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12z" /></svg>

const FLEET = [
  [<GpuIcon key="g" />, 'GPU box', '--role rpc'],
  [<PhoneIcon key="p" />, 'Phone', 'ONNX-YOLO'],
  [<KeyIcon key="k" />, 'OpenAI API', '/v1'],
  [<ChatIcon key="c" />, 'Discord', 'relay'],
]

// [glyph, title, body, span]
const FEATURES = [
  ['Sharding', 'Cross-machine sharding', 'Models too big for any single GPU split across the fleet via llama.cpp RPC — a deterministic allocator picks the placement, and hugpy runs the lead.', 'feat span3'],
  ['API', 'OpenAI-compatible API', '/v1/chat/completions and /v1/models on your own box. Point any OpenAI SDK at it. Mint and revoke API keys from the console — or run it open.', 'feat span3'],
  ['Registry', 'Model registry', 'Search the Hub, pick a quantization, pull with resumable, cancellable jobs into one manifest-backed registry.', 'span2'],
  ['Chat', 'Streaming, never truncated', 'Token-streamed chat with unbounded auto-continuation past per-pass token caps.', 'span2'],
  ['Fleet', 'GPU worker fleet', 'Join any box with hugpy worker. Central heartbeats, assigns models, probes VRAM, routes with local fallback.', 'span2'],
  ['Edge', 'Phones as a video-analytics pool', 'Add phones as workers, not just GPUs. Each runs an ONNX-YOLO vision worker; fan one frame across the fleet and resolve a live consensus verdict.', 'span3'],
  ['Comms', 'Discord bot & cross-machine comms', 'Bind any model — or a Claude keeper — to a Discord channel, and the hugpy bot relays both ways across the whole fleet.', 'span3'],
]

const REFS = [
  ['Documentation', 'Guides, config, deployment', 'hugpy.ai/docs', '/docs'],
  ['PyPI', 'pip install hugpy', 'pypi.org/project/hugpy', 'https://pypi.org/project/hugpy/'],
  ['GitHub', 'Source & issues', 'github.com/hugpy', 'https://github.com/hugpy'],
  ['API reference', 'OpenAI-compatible routes', '/api/v1', '#quickstart'],
  ['Discord', 'Community & support', 'discord.gg/hugpy', '#'],
  ['Hugging Face', 'Models & org', 'huggingface.co/hugpy-ai', 'https://huggingface.co/hugpy-ai'],
]

export default function Landing() {
  const [copied, setCopied] = useState(false)
  const copyPip = () => {
    navigator.clipboard?.writeText('pip install hugpy').then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="landing">
      <nav className="landing-nav">
        <span className="landing-brand"><img src={hugpyLockup} alt="" />hugpy</span>
        <span className="landing-links">
          <a href="#features">Features</a>
          <a href="#quickstart">Quickstart</a>
          <a href="#reference">Reference</a>
          <Link to="/login">Console</Link>
          <a href="https://pypi.org/project/hugpy/" target="_blank" rel="noreferrer">PyPI</a>
        </span>
      </nav>

      {/* ---- Hero (Mesh) ---- */}
      <header className="landing-hero">
        <Constellation coreY={0.34} />
        <div className="landing-hero-inner">
          <div className="landing-bigmark">
            <span className="landing-glow" /><span className="landing-ring" />
            <HugpyMark size={120} style={{ position: 'relative' }} />
          </div>
          <div className="landing-eyebrow">Self-hosted · GPU-pooling · open</div>
          <h1>Your models.<span className="landing-accentline">Your machines.</span></h1>
          <p className="landing-sub">
            A self-hosted LLM console in one pip package — pull from the Hugging Face Hub,
            serve an OpenAI-compatible API, and pool GPUs and phones across machines into a
            single private endpoint.
          </p>
          <div className="landing-cta-row">
            <Link className="landing-btn primary" to="/login">Open the console →</Link>
            <a className="landing-btn" href="#quickstart">Run your own</a>
          </div>
          <div className="landing-pip" onClick={copyPip} title="copy">
            <span className="landing-dollar">$</span> pip install hugpy{copied && <span className="landing-copied">✓ copied</span>}
          </div>
        </div>
      </header>

      {/* ---- Fleet band (Mesh) ---- */}
      <div className="landing-fleet">
        <div className="landing-node">
          <span className="ic"><GpuIcon /></span><span className="lb">GPU box</span><span className="sb">--role rpc</span>
        </div>
        <span className="landing-wire" />
        <div className="landing-node">
          <span className="ic"><PhoneIcon /></span><span className="lb">Phone</span><span className="sb">ONNX-YOLO</span>
        </div>
        <span className="landing-wire" />
        <div className="landing-core">
          <span className="ring"><HugpyMark size={54} /></span><span className="lb">hugpy central</span>
        </div>
        <span className="landing-wire" />
        <div className="landing-node">
          <span className="ic"><KeyIcon /></span><span className="lb">OpenAI API</span><span className="sb">/v1</span>
        </div>
        <span className="landing-wire" />
        <div className="landing-node">
          <span className="ic"><ChatIcon /></span><span className="lb">Discord</span><span className="sb">relay</span>
        </div>
      </div>

      {/* ---- Features bento (Mesh) ---- */}
      <section id="features">
        <div className="landing-sec-head">
          <span className="k">What's in the box</span>
          <h2>One process. The whole fleet.</h2>
        </div>
        <div className="landing-bento">
          {FEATURES.map(([glyph, title, body, span]) => (
            <div className={`landing-tile ${span}`} key={title}>
              <span className="gx" /><span className="tile-glyph">{glyph}</span>
              <h3>{title}</h3><p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- Quickstart (retained) ---- */}
      <section id="quickstart">
        <div className="landing-sec-head">
          <span className="k">Quickstart</span><h2>Three commands.</h2>
        </div>
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

      {/* ---- Reference (Atlas) ---- */}
      <section id="reference">
        <div className="landing-sec-head">
          <span className="k">Reference</span><h2>Where to go next.</h2>
        </div>
        <div className="landing-refs">
          {REFS.map(([h, p, a, href]) => (
            <a className="landing-ref" href={href} target={href.startsWith('http') ? '_blank' : undefined} rel="noreferrer" key={h}>
              <span className="l"><h4>{h}</h4><p>{p}</p></span>
              <span className="a">{a}</span>
            </a>
          ))}
        </div>
      </section>

      {/* ---- CTA band (Mesh) ---- */}
      <div className="landing-band">
        <h2>Inference you own.</h2>
        <p>No nginx, no node — one pip package, your hardware, your keys.</p>
        <div className="landing-cta-row">
          <span className="landing-btn primary" onClick={copyPip} role="button" tabIndex={0}>pip install hugpy</span>
          <a className="landing-btn" href="/docs">Read the docs</a>
        </div>
      </div>

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
