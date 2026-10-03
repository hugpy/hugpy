// HELP TICKETS (operator 2026-10-02): findings the system filed for the
// operator — today, calibrations whose gate prediction disagreed with the
// measurement or whose unload leaked. They sit here as PRE-EXISTING APPROVALS:
// nothing happens until the operator picks an action. The navbar Help button
// shows the pending count (useHelpTicketCount).
import { useCallback, useEffect, useState } from 'react'
import { hugpyFetch } from '../../runtime/config'

const BASE = '/api/llm/help/tickets'
export const TICKETS_CHANGED = 'hugpy:help-tickets-changed'

async function getJson(url, init) {
  const r = await hugpyFetch(url, init)
  let body = null
  try { body = await r.json() } catch { body = null }
  return { ok: r.ok, status: r.status, body }
}

// The pending count for the navbar badge: on mount, once a minute, and
// whenever the panel acts on a ticket.
export function useHelpTicketCount() {
  const [n, setN] = useState(0)
  useEffect(() => {
    let off = false
    const load = () => getJson(`${BASE}?status=pending`)
      .then(r => { if (!off && r.ok) setN(Number(r.body?.pending || 0)) })
      .catch(() => {})
    load()
    const t = setInterval(load, 60000)
    window.addEventListener(TICKETS_CHANGED, load)
    return () => { off = true; clearInterval(t); window.removeEventListener(TICKETS_CHANGED, load) }
  }, [])
  return n
}

const ACTIONS = [
  { id: 'calibrate', label: '⚗ re-calibrate', title: 'run the calibration again on the same worker (same 4-bit setting)' },
  { id: 'keeper', label: '→ keeper', title: 'file it on the keeper bridge — pending your approval there' },
  { id: 'discuss', label: '💬 discuss', title: 'open a help session primed with this finding' },
  { id: 'dismiss', label: '✕ dismiss', title: 'close it without acting' },
]

export default function HelpTickets({ open, onSession }) {
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(null)
  const [note, setNote] = useState({})

  const load = useCallback(() => {
    getJson(`${BASE}?status=pending`).then(r => { if (r.ok) setData(r.body) }).catch(() => {})
  }, [])
  useEffect(() => { if (open) load() }, [open, load])

  const act = async (t, action) => {
    setBusy(`${t.id}:${action}`)
    try {
      const r = await getJson(`${BASE}/${t.id}/act`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
      })
      if (!r.ok) throw new Error(r.body?.error || `HTTP ${r.status}`)
      const msg = action === 'calibrate' ? `calibration started (${r.body?.job?.job_id || ''})`
        : action === 'keeper' ? 'filed on the keeper bridge — approve it there'
          : action === 'discuss' ? 'help session opened' : 'dismissed'
      setNote(n => ({ ...n, [t.id]: msg }))
      if (action === 'discuss' && r.body?.session?.id && onSession) onSession(r.body.session.id)
      window.dispatchEvent(new CustomEvent(TICKETS_CHANGED))
      load()
    } catch (e) {
      setNote(n => ({ ...n, [t.id]: `⚠ ${e.message}` }))
    } finally {
      setBusy(null)
    }
  }

  const tickets = data?.tickets || []
  if (!tickets.length) return null
  return (
    <div className="hp-tickets">
      <div className="hp-tickets-head">{tickets.length} pending approval{tickets.length === 1 ? '' : 's'}</div>
      {tickets.map(t => {
        const d = t.detail || {}
        return (
          <div key={t.id} className={`hp-ticket hp-ticket-${d.verdict || t.kind}`}>
            <div className="hp-ticket-title">
              {t.title}{t.occurrences > 1 ? <em> ×{t.occurrences}</em> : null}
            </div>
            <div className="hp-ticket-meta">
              {d.file ? `${d.file} · ` : ''}{d.gpu ? `${d.gpu} · ` : ''}{d.ctx ? `ctx ${Number(d.ctx).toLocaleString()} · ` : ''}
              {d.bnb ? '4-bit · ' : ''}{new Date((t.updated || t.created) * 1000).toLocaleString()}
            </div>
            <div className="hp-ticket-actions">
              {ACTIONS.map(a => (
                <button key={a.id} type="button" className="hp-btn ghost"
                        disabled={!!busy || (a.id === 'keeper' && !data?.keeper)}
                        title={a.id === 'keeper' && !data?.keeper ? 'no keeper is available on this deployment' : a.title}
                        onClick={() => act(t, a.id)}>
                  {busy === `${t.id}:${a.id}` ? '…' : a.label}
                </button>
              ))}
            </div>
            {note[t.id] && <div className="hp-ticket-note">{note[t.id]}</div>}
          </div>
        )
      })}
    </div>
  )
}
