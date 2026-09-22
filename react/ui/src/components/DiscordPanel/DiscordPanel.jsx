import { useEffect, useState, useCallback } from 'react'
import { fetchJson } from '../../api'
import FixDoc from '../FixDoc/FixDoc'
import './DiscordPanel.css'

const modelKeyOf = (m) => m.model_key ?? m.key

// A binding targets a channel and/or a user; render whichever it has.
const targetLabel = (b) => {
  const bits = []
  if (b.channel_id) bits.push(`#${b.channel_id}`)
  if (b.user_id) bits.push(`@${b.user_id}`)
  return bits.join(' + ') || '—'
}

// One binding: the model it routes to, its Discord target, and a quick compose
// box that pushes a message OUT to that target (queued for the bot to deliver).
function BindingRow({ binding, onRemove, onPing }) {
  const [msg, setMsg] = useState('')
  const [sending, setSending] = useState(false)

  const send = async () => {
    if (!msg.trim()) return
    setSending(true)
    try { await onPing(binding, msg); setMsg('') }
    finally { setSending(false) }
  }

  return (
    <div className="dc-binding">
      <span className="dc-model" title="model this Discord target talks to">🧠 {binding.model_key}</span>
      <span className="dc-arrow">→</span>
      <span className="dc-target" title="Discord channel and/or user">{targetLabel(binding)}</span>
      {binding.label && <span className="dc-label">{binding.label}</span>}
      <input className="dc-ping-input" placeholder="push a message to this target…"
             value={msg} onChange={e => setMsg(e.target.value)}
             onKeyDown={e => { if (e.key === 'Enter') send() }} />
      <button className="dc-ping-btn" onClick={send} disabled={sending || !msg.trim()}
              title="Queue an outbound message; the bot delivers it into Discord">
        {sending ? '…' : 'send'}
      </button>
      <button className="dc-remove" title="Remove binding" onClick={() => onRemove(binding)}>✕</button>
    </div>
  )
}

export default function DiscordPanel({ embedded = false, models = [] }) {
  const [bindings, setBindings] = useState([])
  const [error, setError]       = useState(null)
  const [open, setOpen]         = useState(false)
  // add-binding form
  const [modelKey, setModelKey] = useState('')
  const [channelId, setChannelId] = useState('')
  const [channels, setChannels] = useState([])   // channels the bot can see
  const [manualChan, setManualChan] = useState(false)  // typed ID vs dropdown
  const [userId, setUserId]     = useState('')
  const [users, setUsers]       = useState([])   // members the bot can see (needs members intent)
  const [manualUser, setManualUser] = useState(false)
  const [label, setLabel]       = useState('')
  const [busy, setBusy]         = useState(false)

  const load = useCallback(() => {
    fetchJson('/api/discord/bindings')
      .then(d => { setBindings(Array.isArray(d?.bindings) ? d.bindings : []); setError(null) })
      .catch(e => setError(e.message))
    fetchJson('/api/discord/channels')
      .then(d => setChannels(Array.isArray(d?.channels) ? d.channels : []))
      .catch(() => {})   // dropdown is a convenience; manual entry still works
    fetchJson('/api/discord/users')
      .then(d => setUsers(Array.isArray(d?.users) ? d.users : []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 10_000)
    return () => clearInterval(t)
  }, [load])

  const add = useCallback(async () => {
    if (!modelKey) { alert('Pick a model to bind.'); return }
    if (!channelId.trim() && !userId.trim()) { alert('Enter a channel ID and/or a user ID.'); return }
    setBusy(true)
    try {
      await fetchJson('/api/discord/bindings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_key: modelKey,
          channel_id: channelId.trim() || null,
          user_id: userId.trim() || null,
          label: label.trim() || null,
        }),
      })
      setChannelId(''); setManualChan(false); setUserId(''); setManualUser(false); setLabel('')
      load()
    } catch (e) {
      alert(`Bind failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }, [modelKey, channelId, userId, label, load])

  const remove = useCallback(async (b) => {
    if (!confirm(`Remove the binding ${b.model_key} → ${targetLabel(b)}?`)) return
    try {
      await fetchJson(`/api/discord/bindings/${encodeURIComponent(b.id)}`, { method: 'DELETE' })
      load()
    } catch (e) { alert(`Remove failed: ${e.message}`) }
  }, [load])

  const ping = useCallback(async (b, content) => {
    try {
      await fetchJson('/api/discord/outbox', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ binding_id: b.id, content }),
      })
    } catch (e) { alert(`Send failed: ${e.message}`) }
  }, [])

  const modelOptions = models.map(modelKeyOf).filter(Boolean)

  return (
    <div className="discord-panel">
      <div className={`dc-bar${embedded ? ' dc-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="dc-title">💬 Discord — model ↔ channel/user bindings</span>
        <span className="dc-count">{bindings.length} binding{bindings.length === 1 ? '' : 's'}</span>
        {error && <span className="dc-err" title={error}>registry error<FixDoc doc="registry-error" /></span>}
        {!embedded && <span className="dc-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="dc-body">
          <div className="dc-howto">
            Bind a model to a Discord <b>channel</b> and/or <b>user</b>. The hugpy bot
            (<code>hugpy bot</code>) routes <code>@mentions</code> in that channel / from that
            user to the bound model, and delivers anything you push here into Discord.
            Copy IDs from Discord with Developer Mode on (right-click → Copy ID).
          </div>

          <div className="dc-add">
            <select className="dc-sel" value={modelKey} onChange={e => setModelKey(e.target.value)}>
              <option value="">model…</option>
              {modelOptions.map(k => <option key={k} value={k}>{k}</option>)}
            </select>
            <select
              className="dc-sel"
              value={manualChan ? '__manual__' : channelId}
              title="Channels the hugpy bot can see (or enter an ID manually)"
              onChange={e => {
                const v = e.target.value
                if (v === '__manual__') { setManualChan(true); setChannelId('') }
                else { setManualChan(false); setChannelId(v) }
              }}>
              <option value="">channel…</option>
              {channels.map(c => (
                <option key={c.id} value={c.id}>
                  #{c.name}{c.guild ? ` (${c.guild})` : ''}
                </option>
              ))}
              <option value="__manual__">✏️ enter ID manually…</option>
            </select>
            {manualChan && (
              <input className="dc-in" placeholder="channel ID" value={channelId}
                     autoFocus onChange={e => setChannelId(e.target.value)} />
            )}
            <select
              className="dc-sel"
              value={manualUser ? '__manual__' : userId}
              title="Members the hugpy bot can see (or enter a user ID manually)"
              onChange={e => {
                const v = e.target.value
                if (v === '__manual__') { setManualUser(true); setUserId('') }
                else { setManualUser(false); setUserId(v) }
              }}>
              <option value="">user (optional)…</option>
              {users.map(u => (
                <option key={u.id} value={u.id}>
                  @{u.name}{u.guild ? ` (${u.guild})` : ''}
                </option>
              ))}
              <option value="__manual__">✏️ enter ID manually…</option>
            </select>
            {manualUser && (
              <input className="dc-in" placeholder="user ID" value={userId}
                     autoFocus onChange={e => setUserId(e.target.value)} />
            )}
            <input className="dc-in dc-in-label" placeholder="label (optional)" value={label}
                   onChange={e => setLabel(e.target.value)} />
            <button className="dc-add-btn" onClick={add} disabled={busy}>
              {busy ? '…' : '+ Bind'}
            </button>
          </div>

          {bindings.length === 0 && <div className="dc-empty">No models are bound to Discord yet.</div>}
          {bindings.map(b => (
            <BindingRow key={b.id} binding={b} onRemove={remove} onPing={ping} />
          ))}
        </div>
      )}
    </div>
  )
}
