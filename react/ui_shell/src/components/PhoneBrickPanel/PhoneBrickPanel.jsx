import { useEffect, useState, useCallback, useRef } from 'react'
import { runErrorText } from '../MetricsPanel/RunError'
import { fetchJson } from '../../api'
import { responseReason } from '../responseReason'
import { resolveApiUrl, getHugpyConfig } from '../../runtime/config'
import './PhoneBrickPanel.css'

// Consensus flag → a readable label + class. AGR=agrees with plurality,
// DIS=disagrees, NOD=no detection, null=not resolved yet (live, mid-run).
const CONSENSUS = {
  AGR: { label: 'agrees', cls: 'pb-agr' },
  DIS: { label: 'disagrees', cls: 'pb-dis' },
  NOD: { label: 'no detection', cls: 'pb-nod' },
}
const TERMINAL = ['done', 'error', 'cancelled']

// One phone row: live status, loaded model, queue depth, health ping, remove.
function PhoneRow({ phone, selected, onToggle, onRemove }) {
  const [ping, setPing] = useState(null)   // null | 'checking' | {reachable,...}
  const live = phone.live || {}

  const checkHealth = useCallback(async () => {
    setPing('checking')
    try {
      setPing(await fetchJson(`/api/phone-brick/phones/${encodeURIComponent(phone.id)}/health`))
    } catch (e) {
      setPing({ reachable: false, error: e.message })
    }
  }, [phone.id])

  return (
    <div className={`pb-phone pb-${phone.status}`}>
      <label className="pb-phone-pick" title="Include this phone in the next run">
        <input type="checkbox" checked={selected} onChange={() => onToggle(phone.id)}
               disabled={phone.status !== 'online'} />
      </label>
      <span className="pb-dot" style={{ background: phone.color }} />
      <span className="pb-name">{phone.name}</span>
      <span className="pb-status">{phone.status}</span>
      <span className="pb-url" title={phone.url}>{phone.host}:{phone.port}</span>
      {live.model_loaded != null && (
        <span className={`pb-model ${live.model_loaded ? 'pb-model-on' : 'pb-model-off'}`}
              title={live.model_path || ''}>
          {live.model_loaded ? '🧠 model loaded' : '○ no model'}
        </span>
      )}
      {live.queue_size != null && <span className="pb-queue">queue {live.queue_size}</span>}
      {ping && ping !== 'checking' && (
        <span className={`pb-ping ${ping.reachable ? 'pb-ping-ok' : 'pb-ping-bad'}`}
              title={ping.reachable ? 'central can reach this phone' : responseReason(ping)}>
          {ping.reachable ? '✓ reachable' : '✗ unreachable'}
        </span>
      )}
      <button className="pb-ping-btn" onClick={checkHealth} disabled={ping === 'checking'}
              title="Ping the phone's /status from central">
        {ping === 'checking' ? '…' : 'ping'}
      </button>
      <button className="pb-remove" title="Remove phone" onClick={() => onRemove(phone)}>✕</button>
    </div>
  )
}

// Per-phone verdict table. Rows come from live progress (consensus pending) while
// running, or the final phases (consensus resolved) once done.
function PhaseTable({ rows }) {
  if (!rows.length) return null
  return (
    <table className="pb-phases">
      <thead>
        <tr><th>phone</th><th>top class</th><th>conf</th><th>consensus</th><th>dets</th></tr>
      </thead>
      <tbody>
        {rows.map((p, i) => {
          const c = p.consensus ? (CONSENSUS[p.consensus] || { label: p.consensus, cls: '' })
                                : { label: '…', cls: 'pb-nod' }
          return (
            <tr key={i}>
              <td>{p.phone}</td>
              <td>{p.top_cls}</td>
              <td>{p.top_conf_pct}%</td>
              <td className={c.cls}>{c.label}</td>
              <td>{p.detections}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// The active/last run: live progress while it runs (with a Cancel button), then
// the annotated image + final consensus table when it finishes.
function RunView({ run, liveProgress, currentPhone, onCancel }) {
  if (!run) return null
  const running = !TERMINAL.includes(run.status)
  // Final phases once done; otherwise the live, consensus-pending rows.
  const rows = run.status === 'done' && (run.phases || []).length ? run.phases : liveProgress

  return (
    <div className="pb-run">
      <div className="pb-run-head">
        <span className={`pb-run-status pb-rs-${run.status}`}>{run.status}</span>
        <span className="pb-run-image">{run.image}</span>
        {running && (
          <button className="pb-cancel" onClick={() => onCancel(run.id)}
                  title="Cancel this run">✕ cancel</button>
        )}
      </div>

      {running && (
        <div className="pb-run-busy">
          {currentPhone ? <>Asking <b>{currentPhone}</b>… </> : 'Fanning across phones… '}
          <span className="pb-spin" />
        </div>
      )}
      {run.status === 'error' && <div className="pb-run-error">Run {run.id} failed: {runErrorText(run.error) || `run record has status=error and no error field (image ${run.image || '?'}, ${(run.phases || []).length} phases recorded)`}</div>}
      {run.status === 'cancelled' && <div className="pb-run-cancelled">Run cancelled.</div>}

      {run.status === 'done' && run.output_rel && (
        <img className="pb-run-img" alt="annotated result"
             src={resolveApiUrl(`/api/phone-brick/runs/${encodeURIComponent(run.id)}/image`)} />
      )}
      <PhaseTable rows={rows} />
    </div>
  )
}

export default function PhoneBrickPanel({ embedded = false }) {
  const [phones, setPhones]   = useState([])
  const [error, setError]     = useState(null)
  const [open, setOpen]       = useState(false)
  const [selected, setSelected] = useState({})   // phone_id -> bool
  const [image, setImage]     = useState(null)   // File
  const [run, setRun]         = useState(null)   // active/last run record
  const [liveProgress, setLiveProgress] = useState([])
  const [currentPhone, setCurrentPhone] = useState(null)
  const [busy, setBusy]       = useState(false)
  const fileRef = useRef(null)

  const load = useCallback(() => {
    fetchJson('/api/phone-brick/phones')
      .then(data => {
        if (Array.isArray(data)) { setPhones(data); setError(null) }
        else setError(`GET /api/phone-brick/phones returned a non-list: ${responseReason(data)}`)
      })
      .catch(e => setError(e.message))
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 10_000)
    return () => clearInterval(t)
  }, [load])

  // Live run progress over SSE. Keyed on the run id only, so status changes
  // pushed by the stream don't tear down and re-open the connection.
  useEffect(() => {
    if (!run || TERMINAL.includes(run.status)) return
    const es = new EventSource(
      resolveApiUrl(`/api/phone-brick/runs/${encodeURIComponent(run.id)}/stream`),
      { withCredentials: getHugpyConfig().credentials === 'include' },
    )
    es.onmessage = (e) => {
      let ev
      try { ev = JSON.parse(e.data) } catch { return }
      if (ev.type === 'progress') setLiveProgress(p => [...p, ev.phase])
      else if (ev.type === 'current') setCurrentPhone(ev.phone)
      else if (ev.type === 'status') setRun(r => r ? { ...r, status: ev.status } : r)
      else if (TERMINAL.includes(ev.type)) { setRun(ev.run); setCurrentPhone(null); es.close() }
    }
    es.onerror = () => es.close()   // hiccup; the run record still finishes server-side
    return () => es.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.id])

  const toggle = useCallback((id) => setSelected(s => ({ ...s, [id]: !s[id] })), [])

  const remove = useCallback(async (phone) => {
    if (!confirm(`Remove phone ${phone.name} from the pool?`)) return
    try {
      await fetchJson(`/api/phone-brick/phones/${encodeURIComponent(phone.id)}`, { method: 'DELETE' })
      load()
    } catch (e) { alert(`Remove failed: ${e.message}`) }
  }, [load])

  const startRun = useCallback(async () => {
    if (!image) { alert('Choose an image to analyze first.'); return }
    const ticked = phones.filter(p => selected[p.id] && p.status === 'online').map(p => p.id)
    const ids = ticked.length ? ticked : phones.filter(p => p.status === 'online').map(p => p.id)
    if (!ids.length) { alert(`No online phones to run on: ${phones.length} registered, statuses ${phones.map(p => `${p.name || p.id}=${p.status}`).join(', ') || '(none)'}`); return }

    const form = new FormData()
    form.append('image', image)
    form.append('phone_ids', ids.join(','))
    setBusy(true)
    try {
      const created = await fetchJson('/api/phone-brick/run', { method: 'POST', body: form })
      setLiveProgress([]); setCurrentPhone(null); setRun(created)
      setImage(null)
      if (fileRef.current) fileRef.current.value = ''
    } catch (e) {
      alert(`Run failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }, [image, phones, selected])

  const cancelRun = useCallback(async (runId) => {
    try {
      await fetchJson(`/api/phone-brick/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' })
    } catch (e) { alert(`Cancel failed: ${e.message}`) }
  }, [])

  const onlineCount = phones.filter(p => p.status === 'online').length
  const offlineCount = phones.length - onlineCount

  return (
    <div className="phonebrick-panel">
      <div className={`pb-bar${embedded ? ' pb-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="pb-title">📱 Phone Brick — video analytics pool</span>
        <span className="pb-count">{onlineCount} online / {phones.length} total</span>
        {error && <span className="pb-err" title={error}>registry read failed: {error}</span>}
        {!embedded && <span className="pb-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="pb-body">
          <div className="pb-howto">
            Phones join automatically when started with{' '}
            <code>PHONE_BRICK_CENTRAL</code> pointed at this console.
          </div>

          {phones.length === 0 && <div className="pb-empty">No phones have joined the pool yet.</div>}
          {phones.map(p => (
            <PhoneRow key={p.id} phone={p}
                      selected={!!selected[p.id]}
                      onToggle={toggle} onRemove={remove} />
          ))}

          <div className="pb-run-row">
            <input ref={fileRef} type="file" accept="image/*"
                   onChange={e => setImage(e.target.files?.[0] || null)} />
            <button className="pb-run-btn" onClick={startRun}
                    disabled={busy || !onlineCount}>
              {busy ? 'Starting…' : '▶ Run detection'}
            </button>
            <span className="pb-run-hint">
              {onlineCount ? 'Runs on ticked phones, or all online if none ticked.'
                           : 'No online phones.'}
            </span>
          </div>

          <RunView run={run} liveProgress={liveProgress}
                   currentPhone={currentPhone} onCancel={cancelRun} />
          {offlineCount > 0 && (
            <div className="pb-offline-note">{offlineCount} phone(s) offline (missed heartbeats).</div>
          )}
        </div>
      )}
    </div>
  )
}
