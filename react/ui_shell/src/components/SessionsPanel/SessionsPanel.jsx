import { useEffect, useState, useCallback } from 'react'
import { fetchJson } from '../../api'
import FixDoc from '../FixDoc/FixDoc'
import './SessionsPanel.css'

// Comms sessions: scoped, revocable bearer tokens that let a terminal/agent
// read + post to ONE Discord channel (see abstract_hugpy_dev discord_routes).
// The console is the OPERATOR side — mint / list / revoke only. The token's own
// verbs (/api/discord/session/<token>/…) are what the agent uses, not this UI.
//
// Auth: these routes are operator-gated, satisfied by the logged-in console
// session (hugpyFetch attaches it) — no operator token handled in the browser.

// The endpoint pasted into an agent chat must be the PUBLIC hugpy URL, not a
// relative path, so it resolves from wherever the agent runs. Built from the
// origin the console is served on; off-network agents need the tailnet/public host.
const sessionEndpoint = (token) => `${window.location.origin}/api/discord/session/${token}`

function pasteBlock(channelName, endpoint, instructions) {
  const lines = [
    `You can communicate with Discord channel #${channelName} via:`,
    `  ${endpoint}`,
    `POST <endpoint>/send with JSON {"content":"..."} to post (≤1900 chars, delivered ≤8s).`,
    `GET  <endpoint>/messages?since=<ts of last seen message> to read replies (poll ~30s).`,
    `GET  <endpoint> for a usage refresher.`,
  ]
  if (instructions.trim()) lines.push('', `When/how to use it: ${instructions.trim()}`)
  return lines.join('\n')
}

function copyText(text) {
  // clipboard API needs a secure context; fall back to execCommand on plain http.
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text)
  const ta = document.createElement('textarea')
  ta.value = text; document.body.appendChild(ta); ta.select()
  document.execCommand('copy'); document.body.removeChild(ta)
  return Promise.resolve()
}

const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '—')

const sessionState = (s) =>
  s.revoked ? 'revoked'
    : (s.expires_at && s.expires_at * 1000 < Date.now()) ? 'expired' : 'live'

export default function SessionsPanel({ embedded = false }) {
  const [sessions, setSessions] = useState([])
  const [channels, setChannels] = useState([])
  const [error, setError] = useState(null)
  const [open, setOpen] = useState(false)
  // mint form
  const [channelId, setChannelId] = useState('')
  const [label, setLabel] = useState('')
  const [ttl, setTtl] = useState('')
  const [instructions, setInstructions] = useState('')
  const [busy, setBusy] = useState(false)
  const [minted, setMinted] = useState(null)   // { token, id, endpoint, channelName, block }
  const [copied, setCopied] = useState('')

  const load = useCallback(() => {
    fetchJson('/api/discord/sessions')
      .then(d => { setSessions(Array.isArray(d?.sessions) ? d.sessions : []); setError(null) })
      .catch(e => setError(e.message))
    fetchJson('/api/discord/channels')
      .then(d => setChannels(Array.isArray(d?.channels) ? d.channels : []))
      .catch(() => {})   // picker is a convenience; the list still renders
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 15_000)
    return () => clearInterval(t)
  }, [load])

  const channelName = useCallback(
    (id) => channels.find(c => String(c.id) === String(id))?.name || String(id),
    [channels])

  const mint = useCallback(async () => {
    if (!channelId) { alert('Pick a channel to mint a session for.'); return }
    setBusy(true)
    try {
      const body = { channel_id: channelId, label: label.trim() }
      if (ttl.trim()) body.ttl_hours = parseFloat(ttl)
      const d = await fetchJson('/api/discord/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const name = channelName(channelId)
      const endpoint = sessionEndpoint(d.token)
      setMinted({
        token: d.token, id: d.session.id, endpoint, channelName: name,
        block: pasteBlock(name, endpoint, instructions),
      })
      setChannelId(''); setLabel(''); setTtl(''); setInstructions('')
      load()
    } catch (e) {
      alert(`Mint failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }, [channelId, label, ttl, instructions, channelName, load])

  const revoke = useCallback(async (s) => {
    if (!confirm(`Revoke session ${s.id}? Any agent holding its token loses access immediately.`)) return
    try {
      await fetchJson(`/api/discord/sessions/${encodeURIComponent(s.id)}`, { method: 'DELETE' })
      load()
    } catch (e) { alert(`Revoke failed: ${e.message}`) }
  }, [load])

  // Remove = purge the ROW (2026-08-13 dead-weight cleanup). Dead rows just
  // leave the list; purging a LIVE session also kills its token (the server
  // treats purge of a live row as revoke-and-erase), so the confirm differs.
  const remove = useCallback(async (s, st) => {
    const warn = st === 'live'
      ? `Remove LIVE session ${s.id}? Its token stops working immediately.`
      : `Remove session ${s.id} from the list?`
    if (!confirm(warn)) return
    try {
      await fetchJson(`/api/discord/sessions/${encodeURIComponent(s.id)}?purge=1`, { method: 'DELETE' })
      load()
    } catch (e) { alert(`Remove failed: ${e.message}`) }
  }, [load])

  const pruneDead = useCallback(async () => {
    if (!confirm('Remove ALL revoked/expired sessions from the list? Live sessions are untouched.')) return
    try {
      const d = await fetchJson('/api/discord/sessions/prune', { method: 'POST' })
      load()
      if (d?.pruned != null) alert(`${d.pruned} dead session${d.pruned === 1 ? '' : 's'} removed.`)
    } catch (e) { alert(`Prune failed: ${e.message}`) }
  }, [load])

  const doCopy = useCallback(async (what, text) => {
    await copyText(text)
    setCopied(what); setTimeout(() => setCopied(''), 1500)
  }, [])

  const byGuild = {}
  for (const c of channels) (byGuild[c.guild || '—'] ||= []).push(c)

  const liveCount = sessions.filter(s => sessionState(s) === 'live').length

  return (
    <div className="sessions-panel">
      <div className={`sc-bar${embedded ? ' sc-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="sc-title">🎟️ Comms sessions — scoped channel tokens for agents</span>
        <span className="sc-count">{liveCount} live</span>
        {error && <span className="sc-err" title={error}>registry error<FixDoc doc="registry-error" /></span>}
        {!embedded && <span className="sc-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="sc-body">
          <div className="sc-howto">
            Mint a <b>scoped bearer token</b> bound to one channel and hand it to a terminal
            or agent session — it can read that channel and post to it, nothing else on the API.
            Revocable and optionally time-limited; the server stores only the token's hash.
          </div>

          <div className="sc-add">
            <select className="sc-sel" value={channelId} onChange={e => setChannelId(e.target.value)}>
              <option value="">channel…</option>
              {Object.entries(byGuild).map(([g, chs]) => (
                <optgroup key={g} label={g}>
                  {chs.map(c => <option key={c.id} value={c.id}>#{c.name}</option>)}
                </optgroup>
              ))}
            </select>
            <input className="sc-in" placeholder="label (optional)" value={label}
                   onChange={e => setLabel(e.target.value)} />
            <input className="sc-in sc-in-ttl" type="number" placeholder="ttl hours (blank = none)"
                   value={ttl} onChange={e => setTtl(e.target.value)} />
            <button className="sc-add-btn" onClick={mint} disabled={busy}>
              {busy ? '…' : '+ Mint'}
            </button>
          </div>
          <input className="sc-in sc-in-instr"
                 placeholder='when/how it should be used — folded into the paste-block (e.g. "post a summary after each step; check replies before destructive actions")'
                 value={instructions} onChange={e => setInstructions(e.target.value)} />

          {minted && (
            <div className="sc-minted">
              <div className="sc-minted-head">
                session for #{minted.channelName} — token shown <b>once</b> (id {minted.id})
              </div>
              <textarea className="sc-block" readOnly rows={minted.block.split('\n').length + 1}
                        value={minted.block} onFocus={e => e.target.select()} />
              <div className="sc-minted-actions">
                <button className="sc-add-btn" onClick={() => doCopy('block', minted.block)}>
                  {copied === 'block' ? 'copied ✓' : 'copy paste-block'}</button>
                <button className="sc-add-btn" onClick={() => doCopy('ep', minted.endpoint)}>
                  {copied === 'ep' ? 'copied ✓' : 'copy endpoint'}</button>
                <button className="sc-remove" onClick={() => setMinted(null)}>dismiss</button>
              </div>
            </div>
          )}

          {sessions.length === 0 && <div className="sc-empty">No sessions yet.</div>}
          {sessions.length > 0 && (
            <table className="sc-table">
              <thead><tr>
                <th>state</th><th>channel</th><th>label</th><th>author</th>
                <th>created</th><th>last used</th><th>expires</th><th /></tr></thead>
              <tbody>
                {sessions.map(s => {
                  const st = sessionState(s)
                  return (
                    <tr key={s.id} className={`sc-row sc-${st}`}>
                      <td><span className={`sc-pill sc-pill-${st}`}>{st}</span></td>
                      <td>#{channelName(s.channel_id)}</td>
                      <td>{s.label || '—'}</td>
                      <td>{s.author || '—'}</td>
                      <td>{when(s.created_at)}</td>
                      <td>{when(s.last_used)}</td>
                      <td>{s.expires_at ? when(s.expires_at) : 'never'}</td>
                      <td>
                        {st === 'live' &&
                          <button className="sc-remove" onClick={() => revoke(s)}>revoke</button>}
                        <button className="sc-remove" title="Remove this row from the list"
                                onClick={() => remove(s, st)}>remove</button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
          {sessions.some(s => sessionState(s) !== 'live') && (
            <div className="sc-minted-actions">
              <button className="sc-remove" onClick={pruneDead}>
                clear {sessions.filter(s => sessionState(s) !== 'live').length} dead session{sessions.filter(s => sessionState(s) !== 'live').length === 1 ? '' : 's'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
