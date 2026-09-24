import { useState, useRef, useEffect, useCallback, useMemo, useSyncExternalStore } from 'react'
import ModelLiveState from '../ModelLiveState/ModelLiveState'
import { uploadFile } from '../../api'
import * as chatStore from './chatStore'
import { CopyButton, DiagnosticsView, toJsonText } from '../Diagnostics/Diagnostics'
import { holdingWorkers } from './workers'
import './ChatPanel.css'

const PLACEHOLDER_CMDS = '/system <text> · /clear · /tokens <N>'

function isVLModel(model) {
  if (!model) return false
  // Capability lives in the FULL tasks list — primary_task alone hid vision
  // on models whose primary is e.g. text-generation.
  if (Array.isArray(model.tasks) && model.tasks.includes('image-text-to-text')) return true
  const task = model.primary_task || model.task
  return task === 'image-text-to-text'
}

function ErrorDetail({ detail }) {
  if (!detail) return null
  const { diagnostics, message, ...rest } = detail
  const meta = Object.entries(rest).filter(([k, v]) => v != null && v !== '' && !['event', 'body'].includes(k) && typeof v !== 'object')
  return <div className="msg-error-detail">
    <div className="msg-error-meta">
      {meta.map(([k, v]) => <span key={k}><b>{k}</b> {String(v)}</span>)}
      <CopyButton text={() => toJsonText(detail)} label="copy diagnostics" />
    </div>
    <details className="msg-diag" open={diagnostics != null}>
      <summary>diagnostics{diagnostics == null ? ' (none from server — raw error below)' : ''}</summary>
      {diagnostics != null ? <DiagnosticsView value={diagnostics} /> : <pre className="diag-pre">{toJsonText(rest.event || rest.body || detail)}</pre>}
    </details>
  </div>
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(r.result)
    r.onerror = () => reject(r.error)
    r.readAsDataURL(file)
  })
}

function stripDataUrl(dataUrl) {
  const i = dataUrl.indexOf(',')
  return i >= 0 ? dataUrl.slice(i + 1) : dataUrl
}

export default function ChatPanel({ modelKey, model, onClose, messages = [], setMessages, chats = {}, models = [], onSwitchChat }) {
  // `messages`/`setMessages` are backed by chatStore.js (a module-level
  // singleton, not component/App state) so the conversation survives tab
  // switches, route changes, reloads — AND, critically, so it keeps being
  // written to while ChatPanel/Console is unmounted (see chatStore.js's
  // header for why that used to lose data). setMessages accepts the same
  // value-or-updater shape as a useState setter.
  //
  // `chats` is that SAME store's whole { [modelKey]: Message[] } map, handed
  // down so the top-right conversation switcher can list every populated
  // thread. It's read-only here and shares one source of truth with
  // `messages`: no separate state to drift. `onSwitchChat` is App's own
  // chat-opener (setActiveChat), so picking a model behaves exactly like
  // opening it from the model list. `models` is only for display names.
  //
  // `streaming`/`allocation`/the AbortController/request-id are NOT
  // component state — they live in chatStore too, so the in-flight request
  // and the Stop button both keep working across a remount (model switch or
  // navigating away and back), not just the message content. This component
  // subscribes to chatStore directly (useSyncExternalStore) rather than
  // holding any of that in useState/useRef.
  const [input, setInput]           = useState('')
  const [system, setSystem]         = useState('')
  const [maxTokens, setMaxTokens]   = useState(null)   // null = model max (auto-continued)
  const [attachment, setAttachment] = useState(null)   // {name, isImage, dataUrl?, path?, uploading?}
  // '' = system decides (no alloc sent); else a worker name -> alloc: {worker}
  const [workerPin, setWorkerPinState] = useState(() => chatStore.getWorkerPin(modelKey))
  const setWorkerPin = useCallback((w) => { setWorkerPinState(w); chatStore.setWorkerPin(modelKey, w) }, [modelKey])
  useEffect(() => { setWorkerPinState(chatStore.getWorkerPin(modelKey)) }, [modelKey])
  const holders = useMemo(() => holdingWorkers(model), [model])
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)
  const fileRef   = useRef(null)

  const storeSnapshot = useSyncExternalStore(chatStore.subscribe, chatStore.getSnapshot, chatStore.getSnapshot)
  const streaming  = !!storeSnapshot.streaming[modelKey]
  const allocation = storeSnapshot.allocation[modelKey] || null

  const vlCapable = isVLModel(model)

  // Options for the top-right conversation switcher, read straight off the shared
  // `chats` store (no derived state to keep in sync). Every model with a
  // non-empty saved thread is offered, PLUS the model open right now even if its
  // thread is still empty — so the switcher always shows where you are and the
  // <select> value always matches an option (no React "value not in options"
  // warning). Messages carry no timestamp, so there's nothing finer to sort by:
  // we keep the store's own (first-message insertion) order.
  const chatOptions = useMemo(() => {
    const store = chats || {}
    const keys = Object.keys(store).filter(mk => (store[mk]?.length ?? 0) > 0)
    if (modelKey && !keys.includes(modelKey)) keys.push(modelKey)
    return keys.map(mk => {
      const msgs = store[mk] || []
      const m = models.find(x => (x.model_key ?? x.key) === mk)
      return {
        modelKey: mk,
        name: m?.name ?? mk,
        count: msgs.length,
        // "active" = still generating, read straight off chatStore's live
        // per-model streaming map — true for every thread with an in-flight
        // request, not just the one currently open.
        active: !!storeSnapshot.streaming[mk],
      }
    })
  }, [chats, models, modelKey, storeSnapshot])

  useEffect(() => { inputRef.current?.focus() }, [modelKey])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const onPickFile = useCallback(async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    const isImage = file.type.startsWith('image/')
    setAttachment({ name: file.name, isImage, uploading: true })
    try {
      if (isImage) {
        // Images ride inline as base64 in the chat request (payload.images) —
        // no /uploads round-trip, so vision works without an upload or its auth.
        const dataUrl = await readFileAsDataURL(file)
        setAttachment({ name: file.name, isImage, dataUrl })
      } else {
        // Non-image files still need a server path (their text is inlined
        // server-side via the file channel).
        const res = await uploadFile(file)
        setAttachment({ name: file.name, isImage, dataUrl: null, path: res.path })
      }
    } catch (err) {
      alert(`Could not attach file: ${err?.message ?? err}`)
      setAttachment(null)
    }
  }, [])

  // Building the request and driving the fetch/SSE-stream-reading loop both
  // now live in chatStore.sendMessage — a plain module-level function, not a
  // callback closed over this component's props/state. That's the actual
  // fix for t142: this function used to run to completion (or not) tied to
  // THIS render's closure, and unmounting ChatPanel (model switch, or the
  // whole Console unmounting on a route change away from /console) left
  // `setMessages`/`setStreaming`/`abortRef` pointing at a dead fiber, so
  // in-flight tokens were silently dropped. Now `send` just hands the
  // already-known request shape off to the store and returns; the store
  // keeps running regardless of what this component does next.
  const send = useCallback(() => {
    const text = input.trim()
    const att = attachment
    if ((!text && !att) || streaming || att?.uploading) return
    setInput('')

    if (text === '/clear') { setMessages([]); return }
    if (text.startsWith('/system ')) { setSystem(text.slice(8).trim()); return }
    if (text.startsWith('/tokens ')) {
      const n = parseInt(text.slice(8).trim(), 10)
      if (!isNaN(n)) setMaxTokens(n)
      return
    }

    const userMsg = { role: 'user', content: text }
    if (att) userMsg.attachment = att
    const history = [...messages, userMsg]
    setMessages(history)
    setAttachment(null)

    chatStore.sendMessage(modelKey, { model, history, system, maxTokens, attachment: att, worker: workerPin })
  }, [input, attachment, messages, modelKey, model, system, maxTokens, streaming, setMessages, workerPin])

  // Stop the in-flight response: tell the worker to halt generation (so the
  // GPU stops doing work), then abort the fetch. Explicit user action only —
  // nothing here fires on unmount.
  const stop = useCallback(() => { chatStore.stopMessage(modelKey) }, [modelKey])

  const onKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }, [send])

  return (
    <aside className="chat-panel">
      <div className="chat-header">
        <div className="chat-title">
          <span className="chat-model">{model?.name ?? modelKey}</span>
          <ModelLiveState modelKey={modelKey} compact />
          <span className="chat-meta">
            {model?.framework} · <span title={Array.isArray(model?.tasks) && model.tasks.length > 1 ? `all tasks: ${model.tasks.join(', ')}` : undefined}>{model?.primary_task ?? model?.task}</span>
            {system && <span className="system-set" title={system}> · sys</span>}
            {' · '}{maxTokens ? `max ${maxTokens} tok` : 'unbounded (auto-continue)'}
            {vlCapable && <span className="vl-tag" title="vision-language model"> · 🖼 VL</span>}
          </span>
        </div>
        <div className="chat-header-right">
          {/* Conversation switcher: jump the panel to any other model that has a
              populated (or currently active) chat. Only shown once there's
              somewhere to switch TO — a single-entry dropdown is just noise.
              Options are recomputed live from the shared store above, and
              picking one calls App's own chat-opener, so it lands identically to
              opening that model from the model list (history loads from the
              store on open, as it already does). */}
          {chatOptions.length > 1 && (
            <select
              className="chat-switcher"
              value={modelKey}
              onChange={(e) => { const mk = e.target.value; if (mk && mk !== modelKey) onSwitchChat?.(mk) }}
              title="Switch to another model's saved conversation"
            >
              {chatOptions.map(opt => (
                <option key={opt.modelKey} value={opt.modelKey}>
                  {opt.active ? '● ' : ''}{opt.name} · {opt.count} msg{opt.count === 1 ? '' : 's'}
                </option>
              ))}
            </select>
          )}
          <span
            className={`chat-status ${streaming ? 'is-streaming' : 'is-idle'}`}
            title={streaming ? 'Generating a response…' : 'Conversation saved locally — survives tab changes and reloads'}
          >
            {streaming
              ? '● generating…'
              : `● saved · ${messages.length} msg${messages.length === 1 ? '' : 's'}`}
          </span>
          <button className="btn-clear-chat" onClick={() => setMessages([])} title="Clear conversation" disabled={streaming}>✕ clear</button>
          <button className="btn-close" onClick={onClose} title="Close chat">✕</button>
        </div>
      </div>

      <div
        className={`chat-alloc ${allocation ? (allocation.servedBy === 'local' ? 'is-local' : 'is-worker') : 'is-pending'}`}
        title="The allocation that served the most recent request"
      >
        <label className="alloc-pin" title="Where the next request runs: system decides (default), or pin one worker that holds this model (sends alloc: {worker}; the pin fails with the reason instead of rerouting)">
          <span className="alloc-label">worker</span>
          <select value={workerPin} onChange={e => setWorkerPin(e.target.value)} disabled={streaming}>
            <option value="">system decides</option>
            {holders.map(w => <option key={w.name} value={w.name} disabled={!w.online && w.name !== workerPin}>
              {w.name}{w.hot ? ' · hot' : ' · on disk'}{w.online ? '' : ' · offline'}
            </option>)}
            {workerPin && !holders.some(w => w.name === workerPin) && <option value={workerPin}>{workerPin} · not holding this model</option>}
          </select>
        </label>
        <span className="alloc-label">allocation</span>
        <span className="alloc-value">
          {!allocation
            ? '—'
            : allocation.servedBy === 'local'
              ? 'local (this node)'
              : `${allocation.workerName || allocation.workerId}${
                  allocation.workerId && allocation.workerId !== allocation.workerName
                    ? ` (${allocation.workerId})` : ''
                }`}
        </span>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Chat with <strong>{model?.name ?? modelKey}</strong></p>
            <p className="chat-hint">Commands: {PLACEHOLDER_CMDS}</p>
            <p className="chat-hint">Attach a file with 📎{vlCapable ? ' (images supported)' : ''}.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role} ${msg.error ? 'msg-error' : ''}`}>
            <span className="msg-role">
              {msg.role === 'assistant' ? (msg.model ?? 'assistant') : msg.role}
            </span>
            {msg.attachment?.isImage && msg.attachment.dataUrl && (
              <img className="msg-thumb" src={msg.attachment.dataUrl} alt={msg.attachment.name} title={msg.attachment.name} />
            )}
            {msg.attachment && !msg.attachment.isImage && (
              <span className="msg-file" title={msg.attachment.name}>📎 {msg.attachment.name}</span>
            )}
            {msg.status && !msg.content && (
              <div className="msg-status">⏳ {msg.status}</div>
            )}
            <pre className="msg-content">{msg.content}
              {msg.role === 'assistant' && streaming && i === messages.length - 1 && <span className="cursor">▌</span>}
            </pre>
            {msg.error && <ErrorDetail detail={msg.errorDetail} />}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {attachment && (
        <div className="attach-strip">
          {attachment.isImage
            ? <img src={attachment.dataUrl} alt={attachment.name} className="attach-thumb" />
            : <span className="attach-file">📎</span>}
          <span className="attach-name" title={attachment.name}>
            {attachment.name}{attachment.uploading ? ' · uploading…' : ''}
          </span>
          <button className="attach-remove" onClick={() => setAttachment(null)} disabled={streaming} title="Remove attachment">×</button>
        </div>
      )}

      <div className="chat-input-row">
        <input ref={fileRef} type="file" style={{ display: 'none' }} onChange={onPickFile} />
        <button className="btn-attach" onClick={() => fileRef.current?.click()} disabled={streaming} title="Attach file">📎</button>
        <textarea
          ref={inputRef} className="chat-input" rows={3} value={input}
          onChange={e => setInput(e.target.value)} onKeyDown={onKey}
          placeholder="Message… (Enter to send, Shift+Enter for newline)" disabled={streaming}
        />
        {streaming ? (
          <button className="btn-stop" onClick={stop} title="Stop generating">
            ⏹ Stop
          </button>
        ) : (
          <button className="btn-send" onClick={send} disabled={(!input.trim() && !attachment) || attachment?.uploading}>
            ↑ Send
          </button>
        )}
      </div>
    </aside>
  )
}
