// demoFetch — the showroom's demo-mode fetch shim.
//
// Installed via configureHugpy({ fetch: demoFetch }) so EVERY console panel call
// (all go through hugpyFetch/fetchJson) is answered from canned fixtures instead
// of a real backend. The real <Console/> renders unchanged and fully populated on
// the public, backend-less front door — zero backend exposure, no /api origin.
//
// Deliberately NOT a window.fetch patch: CentralConnect probes the visitor's own
// http://localhost:7002 with a raw fetch, and that must stay REAL for the
// "bring your own inference" go-live path.
//
// Reads -> canned JSON. POST /api/chat/stream -> a real ReadableStream replaying
// a scripted SSE conversation. Mutating writes -> a friendly, schema-shaped demo
// body + a "this is a demo" toast (and native alert()/confirm() are routed to the
// same toast). A Phone-Brick run replays a recorded consensus stream via a patched
// EventSource. Everything is restored on uninstall.

import { configureHugpy, resetHugpyConfig } from '../runtime/config.ts'
import * as fx from './fixtures.js'

export const DEMO_MSG = 'This is a demo — install hugpy (pip install hugpy) and run the console to do this for real.'

// ── tiny toast bus ──────────────────────────────────────────────────────────
const listeners = new Set()
export function onDemoToast(fn) { listeners.add(fn); return () => { listeners.delete(fn) } }
function toast(msg) { for (const fn of [...listeners]) { try { fn(msg || DEMO_MSG) } catch { /* ignore */ } } }

// ── helpers ─────────────────────────────────────────────────────────────────
const enc = new TextEncoder()
const jsonResp = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

function pathOf(input) {
  let url = typeof input === 'string' ? input : (input && input.url) || ''
  if (/^https?:\/\//i.test(url)) {
    try { const u = new URL(url); url = u.pathname + u.search } catch { /* keep */ }
  }
  return url
}
function methodOf(input, init) {
  return String((init && init.method) || (input && typeof input === 'object' && input.method) || 'GET').toUpperCase()
}

// ── scripted streaming chat ─────────────────────────────────────────────────
let chatTurn = 0
function chatStreamResponse(init) {
  const turn = fx.CHAT_REPLAY[chatTurn % fx.CHAT_REPLAY.length]
  chatTurn += 1
  const signal = init && init.signal
  let cancelled = false
  let timer = null
  const stream = new ReadableStream({
    start(controller) {
      const frames = turn.frames
      let i = 0
      const step = () => {
        if (cancelled) return
        if (i >= frames.length) { try { controller.close() } catch { /* closed */ } return }
        const f = frames[i++]
        timer = setTimeout(() => {
          if (cancelled) return
          try { controller.enqueue(enc.encode('data: ' + JSON.stringify(f.event) + '\n\n')) } catch { /* closed */ }
          step()
        }, f.delay_ms)
      }
      const onAbort = () => { cancelled = true; if (timer) clearTimeout(timer); try { controller.close() } catch { /* closed */ } }
      if (signal) {
        if (signal.aborted) { onAbort(); return }
        signal.addEventListener('abort', onAbort, { once: true })
      }
      step()
    },
    cancel() { cancelled = true; if (timer) clearTimeout(timer) },
  })
  return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' } })
}

// ── GET routes -> canned fixtures ───────────────────────────────────────────
function getRoute(p) {
  if (p === '/api/auth/config') return fx.AUTH_CONFIG
  if (p === '/api/version') return fx.VERSION
  if (p === '/api/readiness') return fx.READINESS
  if (/^\/api\/auth-svc\/me$/.test(p)) return fx.AUTH_ME
  if (p === '/api/models') return fx.MODELS
  if (p === '/api/v1/models') return fx.V1_MODELS
  if (/^\/api\/models\/[^/]+$/.test(p)) return fx.MODEL_BY_KEY
  if (p === '/api/ml') return fx.ML
  if (p === '/api/ml/gate') return fx.ML_GATE
  if (p === '/api/keys') return fx.KEYS
  if (p === '/api/prompt/tasks') return fx.PROMPT_TASKS
  if (p === '/api/jobs') return fx.JOBS
  if (/^\/api\/jobs\/[^/]+$/.test(p)) return fx.MUTATING.download
  if (p === '/api/llm/jobs') return fx.LLM_JOBS
  if (p === '/api/llm/central-provisioning') return fx.CENTRAL_PROVISIONING
  if (p === '/api/llm/workers/central-address') return fx.CENTRAL_ADDRESS
  if (p === '/api/llm/hf/auth') return fx.HF_AUTH
  if (p === '/api/discord/sessions') return fx.DISCORD_SESSIONS
  if (p === '/api/civitai/downloads') return fx.CIVITAI_DOWNLOADS
  if (p.startsWith('/api/civitai/search')) return fx.CIVITAI_SEARCH
  if (p === '/api/llm/queue') return fx.QUEUE
  if (p === '/api/llm/slots') return fx.SLOTS
  if (p === '/api/llm/slots/install') return fx.SLOTS_INSTALL
  if (p === '/api/llm/cache') return fx.CACHE
  if (p === '/api/llm/serving') return fx.SERVING
  if (/^\/api\/llm\/serving\/[^/]+$/.test(p)) return fx.SERVING_BY_KEY
  if (p === '/api/llm/groups') return fx.LLM_GROUPS
  if (p === '/api/llm/workers') return fx.WORKERS
  if (/^\/api\/llm\/workers\/[^/]+\/health$/.test(p)) return fx.WORKER_HEALTH
  if (/^\/api\/llm\/workers\/[^/]+$/.test(p)) return fx.WORKERS[0]
  if (p === '/api/llm/enroll-tokens') return fx.ENROLL_TOKENS
  if (p === '/api/llm/peers') return fx.PEERS
  if (p.startsWith('/api/search')) return fx.SEARCH
  if (p.startsWith('/api/hf/spec')) return fx.HF_SPEC
  if (p === '/api/phone-brick/phones') return fx.PHONES
  if (/^\/api\/phone-brick\/phones\/[^/]+\/health$/.test(p)) return fx.PHONE_HEALTH
  if (p === '/api/discord/bridges') return fx.DISCORD_BRIDGES
  if (p === '/api/discord/channels') return fx.DISCORD_CHANNELS
  if (/^\/api\/discord\/bridges\/[^/]+\/messages$/.test(p)) return fx.DISCORD_MESSAGES
  if (p === '/api/discord/bindings') return fx.DISCORD_BINDINGS
  if (p === '/api/discord/users') return fx.DISCORD_USERS
  return undefined
}

// ── mutating routes -> friendly demo bodies (+ toast) ───────────────────────
// returns { body, msg? } | undefined
function mutateRoute(method, p) {
  const M = fx.MUTATING
  const fleet = 'Demo mode — install hugpy to manage the worker fleet.'
  // model lifecycle (download/repo-download return a TERMINAL job -> 100% bar, no toast)
  if (method === 'POST' && /^\/api\/models\/[^/]+\/download$/.test(p)) return { body: M.download }
  if (method === 'POST' && p === '/api/llm/repos/download') return { body: M.repoDownload }
  if (method === 'DELETE' && /^\/api\/models\/[^/]+$/.test(p)) return { body: M.delete, msg: M.delete.message }
  if (method === 'POST' && /^\/api\/models\/[^/]+\/prune$/.test(p)) return { body: M.prune, msg: M.prune.message }
  if (method === 'POST' && /^\/api\/models\/[^/]+\/media$/.test(p)) return { body: M.media }
  if (method === 'POST' && /^\/api\/models\/[^/]+\/media-default$/.test(p)) return { body: { ok: true, demo: true, media_default: true }, msg: 'Demo mode — default media model not persisted.' }
  if (method === 'POST' && /^\/api\/jobs\/[^/]+\/cancel$/.test(p)) return { body: M.jobCancel }
  if (method === 'POST' && /^\/api\/jobs\/[^/]+\/retry$/.test(p)) return { body: M.jobRetry }
  // uploads (chat attach) — silent success so the attach flow works
  if (method === 'POST' && p === '/api/uploads') return { body: fx.UPLOAD }
  // api keys + gates
  if (method === 'POST' && p === '/api/keys') return { body: M.keyMint, msg: M.keyMint.message }
  if (method === 'DELETE' && /^\/api\/keys\/[^/]+$/.test(p)) return { body: M.keyRevoke, msg: M.keyRevoke.message }
  if (method === 'PUT' && p === '/api/keys/require') return { body: M.keyRequire, msg: M.keyRequire.message }
  if (method === 'PUT' && p === '/api/ml/gate') return { body: M.mlGate, msg: M.mlGate.message }
  // serving / slots / cache
  if (method === 'POST' && /^\/api\/llm\/serving\/[^/]+$/.test(p)) return { body: M.servingSave, msg: 'Demo mode — install hugpy to write + restart the serving unit.' }
  if (method === 'POST' && p === '/api/llm/slots/load') return { body: M.slotLoad, msg: M.slotLoad.reason }
  if (method === 'POST' && p === '/api/llm/slots/unload') return { body: M.slotUnload, msg: M.slotUnload.message }
  if (method === 'POST' && p === '/api/llm/cache/warm') return { body: M.cacheWarm, msg: M.cacheWarm.message }
  if (method === 'POST' && p === '/api/llm/free-worker') return { body: M.freeWorker, msg: M.freeWorker.message }
  // chat cancel (Stop) — responsive, no toast
  if (method === 'POST' && /^\/api\/llm\/chat\/cancel\/[^/]+$/.test(p)) return { body: M.chatCancel }
  // worker fleet (showcase only)
  if (method === 'POST' && /^\/api\/llm\/workers\/[^/]+\/probe$/.test(p)) return { body: M.workerProbe, msg: M.workerProbe.message }
  if (method === 'POST' && /^\/api\/llm\/workers\/[^/]+\/unload$/.test(p)) return { body: M.workerUnload, msg: M.workerUnload.message }
  if (method === 'POST' && /^\/api\/llm\/workers\/[^/]+\/(assign|unassign|admit|block|pool)$/.test(p)) return { body: fx.MUTATING_DEFAULT, msg: fleet }
  if (method === 'POST' && p === '/api/llm/workers/register') return { body: { id: 'demo-worker', demo: true, message: fleet }, msg: fleet }
  if (method === 'DELETE' && /^\/api\/llm\/workers\/[^/]+$/.test(p)) return { body: { ok: false, demo: true, message: 'Demo mode — install hugpy to deregister workers.' } }
  if (method === 'POST' && p === '/api/llm/enroll-tokens') return { body: M.enrollMint, msg: M.enrollMint.message }
  if (method === 'DELETE' && /^\/api\/llm\/enroll-tokens\/[^/]+$/.test(p)) return { body: { revoked: false, demo: true, message: 'Demo mode — install hugpy to revoke tokens.' } }
  // discord
  if (method === 'POST' && p === '/api/discord/bridges') return { body: { id: 'demo-bridge', demo: true }, msg: 'Demo mode — install hugpy to create Discord bridges.' }
  if (method === 'DELETE' && /^\/api\/discord\/bridges\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: 'Demo mode — install hugpy to delete bridges.' }
  if (method === 'POST' && /^\/api\/discord\/bridges\/[^/]+\/(approve|reject|send)$/.test(p)) return { body: { ok: true, demo: true }, msg: 'Demo mode — install hugpy to manage Discord.' }
  if (method === 'POST' && p === '/api/discord/bindings') return { body: { id: 'demo-binding', demo: true }, msg: 'Demo mode — install hugpy to create bindings.' }
  if (method === 'DELETE' && /^\/api\/discord\/bindings\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: 'Demo mode — install hugpy to delete bindings.' }
  if (method === 'POST' && p === '/api/discord/outbox') return { body: { ok: true, demo: true }, msg: 'Demo mode — install hugpy to post to Discord.' }
  // phone-brick (run opens the patched EventSource -> recorded consensus stream)
  if (method === 'POST' && p === '/api/phone-brick/run') return { body: fx.PHONE_RUN }
  if (method === 'POST' && /^\/api\/phone-brick\/runs\/[^/]+\/cancel$/.test(p)) return { body: { ok: true, demo: true } }
  if (method === 'DELETE' && /^\/api\/phone-brick\/phones\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: 'Demo mode — install hugpy to manage phones.' }
  // HF credential status writes (ApiAccess) — save/clear the HF token
  if (method === 'POST' && p === '/api/llm/hf/auth') return { body: { configured: false, valid: false, user: null, demo: true }, msg: 'Demo mode — install hugpy to store a Hugging Face token.' }
  if (method === 'DELETE' && p === '/api/llm/hf/auth') return { body: { configured: false, valid: false, user: null, demo: true }, msg: 'Demo mode — no token to clear.' }
  // Discord sessions (SessionsPanel)
  if (method === 'POST' && p === '/api/discord/sessions') return { body: { id: 'demo-session', token: 'hpds_demo0000000000000000000000000000000000', endpoint: 'https://demo.hugpy.ai', demo: true }, msg: 'Demo mode — install hugpy to mint Discord sessions.' }
  if (method === 'DELETE' && /^\/api\/discord\/sessions\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: 'Demo mode — install hugpy to revoke sessions.' }
  // Civitai downloads (HFSearch civitai tab)
  if (method === 'POST' && p === '/api/civitai/download') return { body: { ok: true, demo: true }, msg: 'Demo mode — install hugpy to download from Civitai.' }
  // Central-side model discovery / provisioning triggers
  if (method === 'POST' && p === '/api/models/discover') return { body: { discovered: 0, demo: true }, msg: 'Demo mode — install hugpy to re-scan the model store.' }
  // auth-svc writes never fire in open mode, but answer harmlessly
  if (/^\/api\/auth-svc\//.test(p)) return { body: { ok: true, demo: true } }
  return undefined
}

// ── the shim ────────────────────────────────────────────────────────────────
export async function demoFetch(input, init) {
  const path = pathOf(input)
  const method = methodOf(input, init)
  try {
    if (method === 'POST' && /^\/api\/chat\/stream$/.test(path)) return chatStreamResponse(init)
    if (method === 'GET') {
      const g = getRoute(path)
      if (g !== undefined) return jsonResp(g)
      if (path.startsWith('/api/')) { try { console.warn('[showroom] unmatched demo GET', path) } catch { /* */ } return jsonResp([]) }
      return jsonResp({})
    }
    const m = mutateRoute(method, path)
    if (m) { if (m.msg) toast(m.msg); return jsonResp(m.body) }
    try { console.warn('[showroom] unmatched demo write', method, path) } catch { /* */ }
    toast(DEMO_MSG)
    return jsonResp(fx.MUTATING_DEFAULT)
  } catch (e) {
    return jsonResp({ error: 'demo shim error', detail: String((e && e.message) || e) }, 500)
  }
}

// ── EventSource patch (Phone-Brick run consensus replay) ────────────────────
function makeDemoEventSource(Real) {
  return class DemoEventSource {
    constructor(url, opts) {
      const u = String(url)
      // Only intercept phone-brick run streams; delegate anything else.
      if (!/\/phone-brick\/runs\/[^/]+\/stream/.test(u)) return new Real(u, opts)
      this.url = u
      this.readyState = 0
      this.withCredentials = !!(opts && opts.withCredentials)
      this.onmessage = null
      this.onerror = null
      this.onopen = null
      this._l = {}
      this._timers = []
      this._closed = false
      this._start()
    }
    addEventListener(type, fn) { (this._l[type] || (this._l[type] = [])).push(fn) }
    removeEventListener(type, fn) { const a = this._l[type]; if (a) this._l[type] = a.filter((f) => f !== fn) }
    _emit(type, ev) {
      const h = type === 'open' ? this.onopen : type === 'message' ? this.onmessage : type === 'error' ? this.onerror : null
      if (h) { try { h(ev) } catch { /* */ } }
      ;(this._l[type] || []).forEach((fn) => { try { fn(ev) } catch { /* */ } })
    }
    _start() {
      this.readyState = 1
      this._emit('open', { type: 'open' })
      let acc = 0
      for (const f of fx.PHONE_RUN_FRAMES) {
        acc += f.delay_ms
        this._timers.push(setTimeout(() => {
          if (this._closed) return
          this._emit('message', { type: 'message', data: JSON.stringify(f.event) })
        }, acc))
      }
    }
    close() { this._closed = true; this.readyState = 2; this._timers.forEach(clearTimeout); this._timers = [] }
  }
}

// ── install / uninstall (idempotent, fully restorable) ──────────────────────
let installed = false
const saved = {}
export function installDemoMode() {
  if (installed) return
  installed = true
  configureHugpy({ fetch: demoFetch })
  if (typeof window !== 'undefined') {
    saved.confirm = window.confirm
    saved.alert = window.alert
    saved.EventSource = window.EventSource
    window.confirm = (msg) => {
      const first = typeof msg === 'string' && msg ? msg.split('\n')[0] : ''
      toast(first ? `${first} — not available in the demo. Install hugpy to do this for real.` : DEMO_MSG)
      return false
    }
    window.alert = (msg) => { toast(typeof msg === 'string' ? msg : DEMO_MSG) }
    try { if (saved.EventSource) window.EventSource = makeDemoEventSource(saved.EventSource) } catch { /* */ }
  }
}
export function uninstallDemoMode() {
  if (!installed) return
  installed = false
  resetHugpyConfig()
  if (typeof window !== 'undefined') {
    if (saved.confirm) window.confirm = saved.confirm
    if (saved.alert) window.alert = saved.alert
    if (saved.EventSource) window.EventSource = saved.EventSource
  }
}
