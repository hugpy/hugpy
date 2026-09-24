// Module-level chat store — the single source of truth for per-model
// conversations AND the in-flight request that fills them in.
//
// t142: previously the conversation lived in Console's `useState` (via the
// old useChats.js) and the actual fetch + SSE-stream-reading loop lived
// inside ChatPanel's `send()` closure, with its AbortController/request-id in
// ChatPanel-local refs. Both are anchored to a React fiber that dies the
// moment its owner unmounts — which happens on ANY top-level route change
// away from /console (Console, and therefore its useChats() state, unmounts
// entirely: see App/App.jsx's <Route path="/console/*" element={<ConsoleGate/>} />)
// and on every model switch (ChatPanel is remounted via `key={activeChat}` in
// App.jsx). Once a component unmounts, calling its state setter is a silent
// no-op — so any tokens that streamed in after that point were dropped, never
// reached localStorage, and the Stop button lost its only handle on the real
// AbortController.
//
// Fix: none of this lives in a component anymore. `chats`/`streaming`/
// `allocation`/the AbortController/the request id are plain module-scope
// state that exists for the lifetime of the page (not any component tree).
// `sendMessage()` is a plain async function, not a hook/effect/callback tied
// to a render — it keeps running to completion (or an explicit stopMessage())
// no matter what mounts or unmounts around it. Components only ever *read* a
// snapshot (via useSyncExternalStore, see useChats.js) and *dispatch*
// (sendMessage/stopMessage/clearChat/setMessages) — they never own the
// request or the only copy of the data.

import { hugpyFetch } from '../../runtime/config.ts'

const KEY = 'hugpy.chats.v1'
const EMPTY = []   // stable reference so "no messages yet" doesn't churn effects

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {} } catch { return {} }
}

// Drop anything heavy or purely transient before persisting: base64 image
// dataUrls would blow the ~5MB localStorage quota, and `status` is only
// meaningful mid-stream.
function sanitize(chats) {
  const out = {}
  for (const [modelKey, msgs] of Object.entries(chats)) {
    if (!Array.isArray(msgs) || msgs.length === 0) continue
    out[modelKey] = msgs.map(m => {
      const { status, ...rest } = m
      if (rest.attachment?.dataUrl) {
        rest.attachment = { ...rest.attachment, dataUrl: undefined }
      }
      return rest
    })
  }
  return out
}

// ---- module-level state: outlives every component ----
let chats = load()                 // { [modelKey]: Message[] }
const streaming = {}               // { [modelKey]: boolean }
const allocation = {}              // { [modelKey]: {servedBy, workerId, workerName} | null }
const controllers = {}             // { [modelKey]: AbortController }  (in-memory only, on purpose)
const requestIds = {}              // { [modelKey]: string }

const listeners = new Set()

// One snapshot object per version so useSyncExternalStore's Object.is check
// short-circuits renders that don't touch the parts that changed.
let snapshot = { chats, streaming: {}, allocation: {} }
function commit() {
  snapshot = { chats, streaming: { ...streaming }, allocation: { ...allocation } }
  for (const l of listeners) l()
}
function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(sanitize(chats))) } catch { /* quota / private mode */ }
}

export function subscribe(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}
export function getSnapshot() { return snapshot }

export function getMessages(modelKey) { return chats[modelKey] || EMPTY }

// Per-model worker pin for the chat box: '' = "system decides" (no alloc in
// the body); a worker name = `alloc: {worker}` (an explicit pin is a contract
// server-side: it fails naming why rather than rerouting). Persisted per model.
const PIN_KEY = 'hugpy.chat.worker.v1'
function loadPins() { try { return JSON.parse(localStorage.getItem(PIN_KEY)) || {} } catch { return {} } }
export function getWorkerPin(modelKey) { return loadPins()[modelKey] || '' }
export function setWorkerPin(modelKey, worker) {
  const pins = loadPins()
  if (worker) pins[modelKey] = worker; else delete pins[modelKey]
  try { localStorage.setItem(PIN_KEY, JSON.stringify(pins)) } catch { /* quota / private mode */ }
}

// A chat failure, kept whole (dev console — no placation): the server's
// message verbatim, its `diagnostics` object when present, request id,
// log_ref and the raw body/event for anything else.
function errorDetail({ status, message, data, event, requestId, worker }) {
  const src = event || data || {}
  const d = {
    message: String(message ?? ''),
    status: status ?? undefined,
    request_id: src.request_id || requestId || undefined,
    log_ref: src.log_ref || undefined,
    worker_pin: worker || undefined,
    diagnostics: src.diagnostics ?? undefined,
  }
  if (event) d.event = event
  else if (data !== undefined) d.body = data
  return d
}
function failTurn(modelKey, detail, { replaceEmpty = true } = {}) {
  setMessages(modelKey, prev => {
    const copy = [...prev]
    const last = copy[copy.length - 1]
    const bubble = { role: 'assistant', content: `[Error: ${detail.message}]`, error: true, errorDetail: detail }
    if (last?.role === 'assistant' && (replaceEmpty ? last.content === '' : true)) {
      copy[copy.length - 1] = { ...last, ...bubble, model: last.model, status: null }
    } else copy.push(bubble)
    return copy
  })
}
export function isStreaming(modelKey) { return !!streaming[modelKey] }
export function getAllocation(modelKey) { return allocation[modelKey] || null }

// setMessages(modelKey, next | prev => next) — mirrors the React setState API
// call sites already use, so they need no rewrite.
export function setMessages(modelKey, updater) {
  const cur = chats[modelKey] || EMPTY
  const next = typeof updater === 'function' ? updater(cur) : updater
  chats = { ...chats, [modelKey]: next }
  persist()
  commit()
}

export function clearChat(modelKey) {
  if (!(modelKey in chats)) return
  const copy = { ...chats }
  delete copy[modelKey]
  chats = copy
  persist()
  commit()
}

// The ONLY way a request should ever be aborted: an explicit user action
// (the Stop button), never a component unmount / route change.
export function stopMessage(modelKey) {
  const rid = requestIds[modelKey]
  if (rid) {
    hugpyFetch(`/api/llm/chat/cancel/${encodeURIComponent(rid)}`, { method: 'POST' }).catch(() => {})
  }
  controllers[modelKey]?.abort()
}

// Issue the chat request and stream the response straight into this module's
// store. Deliberately a plain function — NOT a hook, NOT inside a component's
// render/effect — so nothing about a component's lifecycle can touch it.
export async function sendMessage(modelKey, { model, history, system, maxTokens, attachment, worker }) {
  if (streaming[modelKey]) return
  streaming[modelKey] = true
  allocation[modelKey] = null
  commit()

  const wireMessages = history
    .filter(m => !m.error)                    // never replay failed turns as context
    .map(({ role, content, attachment: a }) => {
      const base = { role, content }
      if (a?.path) base.file = a.path
      return base
    })
  if (system) wireMessages.unshift({ role: 'system', content: system })

  const reqId = (crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`)
  requestIds[modelKey] = reqId
  const lastUser = history[history.length - 1]
  const payload = {
    model_key: modelKey,
    messages: wireMessages,
    prompt: lastUser?.content ?? '',
    request_id: reqId,
  }
  if (maxTokens) payload.max_new_tokens = maxTokens
  if (attachment?.path) payload.file = attachment.path
  if (attachment?.isImage && attachment?.dataUrl) payload.images = [attachment.dataUrl]
  if (worker) payload.alloc = { worker }

  const modelName = model?.name ?? modelKey
  setMessages(modelKey, prev => [...prev, { role: 'assistant', content: '', model: modelName }])

  const ctrl = new AbortController()
  controllers[modelKey] = ctrl
  try {
    const resp = await hugpyFetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    })
    if (!resp.ok) {
      const text = await resp.text()
      let data
      try { data = JSON.parse(text) } catch { data = text }
      const msg = (data && typeof data === 'object' && (data.error || data.detail || data.message)) || text || `HTTP ${resp.status}`
      const err = new Error(`${resp.status}: ${typeof msg === 'string' ? msg : JSON.stringify(msg)}`)
      err.detail = errorDetail({ status: resp.status, message: err.message, data, requestId: requestIds[modelKey], worker })
      throw err
    }

    const reader = resp.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const raw = line.slice(6).trim()
        if (!raw || raw === '[DONE]') continue
        try {
          const evt = JSON.parse(raw)
          if (evt.type === 'request') {
            // Server's authoritative request_id for cancellation.
            if (evt.request_id) requestIds[modelKey] = evt.request_id
          } else if (evt.type === 'status' && evt.served_by != null) {
            allocation[modelKey] = {
              servedBy: evt.served_by,
              workerId: evt.worker_id || '',
              workerName: evt.worker_name || (evt.served_by === 'local' ? 'local' : ''),
            }
            commit()
          } else if (evt.type === 'status') {
            const pct = evt.progress != null ? ` ${Math.round(evt.progress * 100)}%` : ''
            setMessages(modelKey, prev => {
              const copy = [...prev]
              const last = copy[copy.length - 1]
              if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, status: `${evt.message || evt.stage || 'working'}${pct}` }
              return copy
            })
          } else if (evt.type === 'token') {
            setMessages(modelKey, prev => {
              const copy = [...prev]
              const last = copy[copy.length - 1]
              if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, content: last.content + evt.text, status: null }
              return copy
            })
          } else if (evt.type === 'error') {
            failTurn(modelKey, errorDetail({ message: evt.message, event: evt, requestId: requestIds[modelKey], worker }), { replaceEmpty: false })
          }
        } catch { /* ignore malformed SSE line */ }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      // User stopped it — mark the partial as stopped, not an error.
      setMessages(modelKey, prev => {
        const copy = [...prev]
        const last = copy[copy.length - 1]
        if (last?.role === 'assistant') {
          copy[copy.length - 1] = { ...last, status: null, content: last.content + ' ⏹', stopped: true }
        }
        return copy
      })
    } else {
      failTurn(modelKey, e.detail || errorDetail({ message: e.message, requestId: requestIds[modelKey], worker,
        data: { name: e.name, stack: e.stack } }))
    }
  } finally {
    streaming[modelKey] = false
    delete controllers[modelKey]
    delete requestIds[modelKey]
    commit()
  }
}
