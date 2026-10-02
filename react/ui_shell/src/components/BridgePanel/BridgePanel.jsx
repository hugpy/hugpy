import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchJson } from '../../api'
import './BridgePanel.css'

const modelKeyOf = (m) => m.model_key ?? m.key
const DEFER_MODES = [
  ['auto', 'auto — send replies immediately'],
  ['defer', 'defer — hold every reply for my approval'],
  ['directive', 'directive — let the model decide per message'],
]
// Same three defer modes, but labeled in keeper terms when the brain is a keeper.
const KEEPER_MODES = [
  ['defer', 'user-strict — approve every keeper reply'],
  ['directive', 'keeper-choice — keeper decides (DEFER: escalates)'],
  ['auto', 'keeper auto — send every reply'],
]

// One line of the merged transcript (Discord in, model/console out, pending).
function Msg({ m, onApprove, onReject }) {
  const inbound = m.direction === 'in'
  const pending = m.status === 'pending'
  const rejected = m.status === 'rejected'
  const who = inbound ? `← ${m.author || 'discord'}`
    : m.source === 'model' ? '→ 🤖 model'
    : m.source === 'keeper' ? '→ ⛨ keeper'
    : '→ 🧑 you'
  // Editing a candidate before send was removed 2026-07-09: it let arbitrary text
  // reach the channel. A reviewer approves the model's exact draft or rejects it.
  return (
    <div className={`br-msg ${inbound ? 'br-in' : 'br-out'}${pending ? ' br-pending' : ''}${rejected ? ' br-rejected' : ''}`}>
      <span className="br-who">{who}</span>
      <span className="br-text">{m.content}</span>
      {pending && (
        <span className="br-acts">
          <button className="br-ok" title="Approve & send the model’s draft as-is" onClick={() => onApprove(m.id)}>✓ send</button>
          <button className="br-no" title="Reject" onClick={() => onReject(m.id)}>✕</button>
        </span>
      )}
    </div>
  )
}

export default function BridgePanel({ embedded = false, models = [] }) {
  const [open, setOpen]       = useState(false)
  const [bridges, setBridges] = useState([])
  const [channels, setChannels] = useState([])
  const [error, setError]     = useState(null)
  // create form
  const [modelKey, setModelKey] = useState('')
  const [channelId, setChannelId] = useState('')
  const [manualChan, setManualChan] = useState(false)
  const [directive, setDirective] = useState('')
  const [deferMode, setDeferMode] = useState('defer')
  const [brain, setBrain] = useState('model')
  const [keeperTarget, setKeeperTarget] = useState('')
  const [logMode, setLogMode] = useState('open')
  const [busy, setBusy] = useState(false)
  // expanded transcript
  const [openId, setOpenId] = useState(null)
  const [messages, setMessages] = useState([])
  const openIdRef = useRef(null)

  const loadBridges = useCallback(() => {
    fetchJson('/api/discord/bridges')
      .then(d => { setBridges(Array.isArray(d?.bridges) ? d.bridges : []); setError(null) })
      .catch(e => setError(e.message))
    fetchJson('/api/discord/channels')
      .then(d => setChannels(Array.isArray(d?.channels) ? d.channels : []))
      .catch(() => {})
  }, [])

  const loadMessages = useCallback((id) => {
    if (!id) return
    fetchJson(`/api/discord/bridges/${encodeURIComponent(id)}/messages`)
      .then(d => { if (openIdRef.current === id) setMessages(Array.isArray(d?.messages) ? d.messages : []) })
      .catch(() => {})
  }, [])

  useEffect(() => { loadBridges(); const t = setInterval(loadBridges, 5000); return () => clearInterval(t) }, [loadBridges])
  useEffect(() => {
    openIdRef.current = openId
    if (!openId) { setMessages([]); return }
    loadMessages(openId)
    const t = setInterval(() => loadMessages(openId), 3000)
    return () => clearInterval(t)
  }, [openId, loadMessages])

  const channelLabel = (cid) => {
    const c = channels.find(x => x.id === cid)
    return c ? `#${c.name}${c.guild ? ` (${c.guild})` : ''}` : `#${cid}`
  }

  const create = useCallback(async () => {
    // A keeper-brained bridge needs no model (the attached keeper is the brain).
    if (brain === 'model' && !modelKey) { alert('Pick a model.'); return }
    if (!channelId.trim()) { alert('Pick or enter a channel.'); return }
    setBusy(true)
    try {
      // no-logs is only valid for an auto model bridge (see server guard); force
      // 'open' otherwise so the request can't be rejected on an impossible combo.
      const effLogMode = (brain === 'model' && deferMode === 'auto') ? logMode : 'open'
      await fetchJson('/api/discord/bridges', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel_id: channelId.trim(),
                               model_key: brain === 'model' ? modelKey : (modelKey || null),
                               directive: directive.trim() || null, defer_mode: deferMode,
                               brain, keeper_target: keeperTarget.trim() || null,
                               log_mode: effLogMode }),
      })
      setDirective(''); setChannelId(''); setManualChan(false); setKeeperTarget(''); setLogMode('open')
      loadBridges()
    } catch (e) { alert(`Bridge failed: ${e.message}`) }
    finally { setBusy(false) }
  }, [brain, modelKey, channelId, directive, deferMode, keeperTarget, logMode, loadBridges])

  const removeBridge = useCallback(async (b) => {
    if (!confirm(`Remove the bridge ${b.model_key} ↔ ${channelLabel(b.channel_id)}?`)) return
    try {
      await fetchJson(`/api/discord/bridges/${encodeURIComponent(b.id)}`, { method: 'DELETE' })
      if (openId === b.id) setOpenId(null)
      loadBridges()
    } catch (e) { alert(`Remove failed: ${e.message}`) }
  }, [openId, loadBridges, channels])

  const approve = useCallback(async (msgId, content) => {
    try {
      await fetchJson(`/api/discord/bridges/${encodeURIComponent(openId)}/approve`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message_id: msgId, content: content ?? null }),
      })
      loadMessages(openId)
    } catch (e) { alert(`Approve failed: ${e.message}`) }
  }, [openId, loadMessages])

  const reject = useCallback(async (msgId) => {
    try {
      await fetchJson(`/api/discord/bridges/${encodeURIComponent(openId)}/reject`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message_id: msgId }),
      })
      loadMessages(openId)
    } catch (e) { alert(`Reject failed: ${e.message}`) }
  }, [openId, loadMessages])

  // Console → channel messaging is intentionally NOT exposed in the UI: it is an
  // outbound-send vector reachable by anyone past the auth gate. The only
  // transcript action here is a destructive-to-local clear.
  const clearChat = useCallback(async () => {
    if (!openId) return
    if (!confirm('Clear this bridge’s transcript here?\n\nThis wipes the console-side history only — it does NOT delete anything from the Discord channel.')) return
    try {
      await fetchJson(`/api/discord/bridges/${encodeURIComponent(openId)}/messages`, { method: 'DELETE' })
      setMessages([])
      loadMessages(openId)
    } catch (e) { alert(`Clear failed: ${e.message}`) }
  }, [openId, loadMessages])

  const modelOptions = models.map(modelKeyOf).filter(Boolean)
  // Pending candidates awaiting approval. We only hold the transcript for the
  // open bridge, so the badge is exact there and 0 (hidden) for collapsed rows.
  const pendingCount = (b) => (b.id === openId)
    ? messages.filter(m => m.status === 'pending').length : 0
  const modeOptions = brain === 'keeper' ? KEEPER_MODES : DEFER_MODES

  return (
    <div className="bridge-panel">
      <div className={`br-bar${embedded ? ' br-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="br-title">🔗 Console ↔ Discord bridges</span>
        <span className="br-count">{bridges.length}</span>
        {error && <span className="br-err" title={error}>error</span>}
        {!embedded && <span className="br-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="br-body">
          <div className="br-howto">
            Allocate a <b>model</b> or a <b>keeper</b> to a Discord channel and supervise it from
            here: inbound messages generate a reply per the <b>directive</b>; the mode decides
            whether it sends automatically, waits for your approval, or lets the brain choose. A
            <b> keeper</b> bridge is driven by an attached keeper process
            (<code>hugpy keeper --bridge &lt;id&gt;</code>) — <b>user-strict</b> holds every keeper
            reply for your approval below; <b>keeper-choice</b> lets it decide (<code>DEFER:</code> escalates).
          </div>

          <div className="br-add">
            <select className="br-sel" value={brain}
              onChange={e => { setBrain(e.target.value); setDeferMode('defer') }} title="brain">
              <option value="model">🧠 model</option>
              <option value="keeper">⛨ keeper</option>
            </select>
            <select className="br-sel" value={modelKey} onChange={e => setModelKey(e.target.value)}>
              <option value="">{brain === 'keeper' ? 'model (optional)…' : 'model…'}</option>
              {modelOptions.map(k => <option key={k} value={k}>{k}</option>)}
            </select>
            <select className="br-sel" value={manualChan ? '__manual__' : channelId}
              onChange={e => { const v = e.target.value
                if (v === '__manual__') { setManualChan(true); setChannelId('') }
                else { setManualChan(false); setChannelId(v) } }}>
              <option value="">channel…</option>
              {channels.map(c => <option key={c.id} value={c.id}>#{c.name}{c.guild ? ` (${c.guild})` : ''}</option>)}
              <option value="__manual__">✏️ enter ID…</option>
            </select>
            {manualChan && <input className="br-in" placeholder="channel ID" value={channelId}
                                  onChange={e => setChannelId(e.target.value)} />}
            {brain === 'keeper' && (
              <input className="br-in" placeholder="keeper target (e.g. host or lxc name)"
                     value={keeperTarget} onChange={e => setKeeperTarget(e.target.value)} />
            )}
            <select className="br-sel" value={deferMode} onChange={e => setDeferMode(e.target.value)}>
              {modeOptions.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
            </select>
            <select className="br-sel"
              value={(brain === 'model' && deferMode === 'auto') ? logMode : 'open'}
              onChange={e => setLogMode(e.target.value)}
              disabled={!(brain === 'model' && deferMode === 'auto')}
              title="Transcript retention. 'no logs' keeps nothing (ephemeral) — allowed only for an auto model bridge; keeper / defer / session bridges must retain their transcript to work.">
              <option value="open">📝 open logs</option>
              <option value="none">🚫 no logs</option>
            </select>
            <button className="br-add-btn" onClick={create} disabled={busy}>{busy ? '…' : '+ Bridge'}</button>
          </div>
          <textarea className="br-directive" placeholder="directive — what to focus on, and when to defer to you…"
                    value={directive} onChange={e => setDirective(e.target.value)} rows={2} />

          {bridges.length === 0 && <div className="br-empty">No bridges yet.</div>}
          {bridges.map(b => (
            <div key={b.id} className="br-row-wrap">
              <div className="br-row">
                {b.brain === 'keeper'
                  ? <span className="br-model" title={`keeper${b.keeper_target ? `: ${b.keeper_target}` : ''}`}>⛨ {b.keeper_target || 'keeper'}</span>
                  : <span className="br-model" title="model">🧠 {b.model_key || '—'}</span>}
                <span className="br-arrow">↔</span>
                <span className="br-chan" title="Discord channel">{channelLabel(b.channel_id)}</span>
                <span className={`br-mode br-mode-${b.defer_mode}`}>
                  {b.brain === 'keeper'
                    ? (b.defer_mode === 'defer' ? 'user-strict'
                       : b.defer_mode === 'directive' ? 'keeper-choice' : 'keeper-auto')
                    : b.defer_mode}
                </span>
                {b.log_mode === 'none' && <span className="br-nolog" title="Ephemeral — no transcript retained">no-logs</span>}
                {pendingCount(b) > 0 && <span className="br-pending-badge">{pendingCount(b)} pending</span>}
                <button className="br-open" onClick={() => setOpenId(openId === b.id ? null : b.id)}>
                  {openId === b.id ? 'hide' : 'open'}
                </button>
                <button className="br-remove" title="Remove bridge" onClick={() => removeBridge(b)}>✕</button>
              </div>
              {b.directive && <div className="br-dir-show" title="directive">▸ {b.directive}</div>}
              {openId === b.id && (
                <div className="br-transcript">
                  {messages.length === 0 && <div className="br-empty">No messages yet.</div>}
                  {messages.map(m => <Msg key={m.id} m={m} onApprove={approve} onReject={reject} />)}
                  <div className="br-transcript-foot">
                    <span className="br-readonly" title="This console cannot message the channel — read-only transcript with approve/reject on model candidates.">read-only</span>
                    <button className="br-clear" title="Clear this transcript (console-side only; does not touch Discord)"
                            onClick={clearChat} disabled={messages.length === 0}>🗑 clear</button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
