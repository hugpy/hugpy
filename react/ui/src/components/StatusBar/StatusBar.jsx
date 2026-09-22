import { useEffect, useState, useMemo } from 'react'
import { fetchJson } from '../../api'
import { useFeed } from '../../runtime/feeds'
import useSessionState from '../../hooks/useSessionState'
import PeersBar from '../PeersBar/PeersBar'
import DownloadsQueue from '../DownloadsQueue/DownloadsQueue'
import FleetResidency from './FleetResidency'
import './StatusBar.css'

function fmtBytes(n) {
  if (n == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}

// The Overview — the one consolidated fleet readout (its own tab). Pulls together
// everything that used to be scattered across the topbar and a static bar: model
// counts (installed / registered / in service), the inference queue, GPU workers,
// model slots, phone bricks, VRAM, host RAM, an alerts cluster, and the peers
// row. `models` + `workers` are lifted from the console; slots / queue / phones
// are polled here.
export default function StatusBar({ models = [], workers = [] }) {
  const [slots, setSlots]   = useState({ enabled: null, slots: [], resources: null })
  const [queue, setQueue]   = useState({ active: [], counts: {} })
  const [jobs, setJobs]     = useState([])   // CON-01: F5 job store (all transports)
  const [phones, setPhones] = useState([])
  // The VRAM / Host RAM meters double as the trigger for the fleet residency
  // breakdown below the tiles. Session-scoped like the rest of the console's
  // panel expansions: default collapsed on a fresh tab, sticky while you work.
  const [showRes, setShowRes] = useSessionState('hugpy.sess.sb.residency', false)

  // Fed by the ONE live subscription (runtime/feeds.js) instead of four timers
  // (2026-09-10). Unified job store: discord / cli / v1 GENERATION with transport
  // attribution (downloads carry kind:'download' and are filtered out below —
  // they get their own DownloadsQueue tile, not the inference Queue). Chat
  // streams still live only in the queue feed, so the queue segment merges BOTH.
  const fSlots = useFeed('slots', null)
  const fQueue = useFeed('queue', null)
  const fJobs = useFeed('jobs', null)
  const fPhones = useFeed('phones', null)
  useEffect(() => { if (fSlots) setSlots(fSlots) }, [fSlots])
  useEffect(() => { if (fQueue) setQueue(fQueue) }, [fQueue])
  useEffect(() => { if (fJobs && Array.isArray(fJobs.jobs)) setJobs(fJobs.jobs) }, [fJobs])
  useEffect(() => { if (Array.isArray(fPhones)) setPhones(fPhones) }, [fPhones])

  const m = useMemo(() => {
    const slotList = slots.slots || []

    const served = new Set()
    for (const w of workers) {
      if (w.status !== 'online') continue
      for (const k of (w.loaded_models || [])) served.add(k)
    }
    for (const s of slotList) if (s.model_key && s.healthy) served.add(s.model_key)

    // Union of the legacy live queue (in-flight chat streams) and the F5 job
    // store (discord/cli/v1 GENERATION). Model DOWNLOADS (kind:'download') are
    // EXCLUDED here — they belong to the separate DownloadsQueue tile, not the
    // inference Queue. Jobs may double-count a chat once the store learns chat —
    // dedup by request id guards that day.
    const LIVE = new Set(['pending', 'processing', 'streaming'])
    const liveJobs = jobs.filter(j => j.kind !== 'download' && LIVE.has(j.status))
    const queueIds = new Set((queue.active || []).map(r => r.request_id))
    const extraJobs = liveJobs.filter(j => !queueIds.has(j.id))
    const active  = (queue.counts?.active ?? (queue.active || []).filter(r => r.state === 'active').length)
      + extraJobs.filter(j => j.status !== 'pending').length
    const pending = (queue.counts?.waiting ?? (queue.active || []).filter(r => r.state === 'waiting').length)
      + extraJobs.filter(j => j.status === 'pending').length
    // Per-transport breakdown for the queue segment (chat rows in /llm/queue
    // carry no transport — count them as web).
    const byTransport = {}
    for (const j of extraJobs) {
      const t = j.transport || j.kind || '?'
      byTransport[t] = (byTransport[t] || 0) + 1
    }
    const chatRows = (queue.active || []).length
    if (chatRows) byTransport.web = (byTransport.web || 0) + chatRows

    const wOnline  = workers.filter(w => w.status === 'online').length
    const wPending = workers.filter(w => w.admission === 'pending').length
    const wOffline = workers.length - wOnline

    const slotsBusy = slotList.filter(s => s.model_key).length
    const slotsDown = slotList.filter(s => s.error).length

    const phOnline  = phones.filter(p => p.status === 'online').length
    const phOffline = phones.length - phOnline

    let vt = 0, vf = 0, hasV = false
    for (const w of workers) {
      if (w.status !== 'online') continue
      for (const g of (w.gpus || [])) {
        if (g.memory_total != null) { vt += g.memory_total; hasV = true }
        if (g.memory_free != null) vf += g.memory_free
      }
    }
    const res = slots.resources

    const alerts = []
    if (wPending)  alerts.push({ k: 'wpend', cls: 'warn', text: `${wPending} worker pending` })
    if (wOffline)  alerts.push({ k: 'woff',  cls: 'bad',  text: `${wOffline} worker offline` })
    if (slotsDown) alerts.push({ k: 'sdown', cls: 'bad',  text: `${slotsDown} slot down` })
    if (phOffline) alerts.push({ k: 'poff',  cls: 'bad',  text: `${phOffline} phone offline` })

    return {
      installed: models.filter(x => x.status === 'installed').length,
      registered: models.length,
      inService: served.size, active, pending, byTransport,
      wOnline, wTotal: workers.length,
      slotsEnabled: slots.enabled, slotsBusy, slotsTotal: slotList.length,
      phOnline, phTotal: phones.length,
      vramUsed: Math.max(vt - vf, 0), vramTotal: vt, hasVram: hasV,
      ramUsed: res?.used_bytes ?? null, ramTotal: res?.total_bytes ?? null,
      alerts,
    }
  }, [models, workers, slots, queue, jobs, phones])

  return (
    <div className="sb-bar">
      <div className="sb-tiles">
          <Seg k="Installed" v={m.installed} />
          <Seg k="Registered" v={m.registered} />
          <Seg k="In service" v={m.inService} />

          {/* in-flight GENERATION across every transport (CON-01) — active/pending
              with a per-transport breakdown; "idle" when nothing is in flight.
              Model downloads are NOT counted here — they live in the separate
              Downloads tile immediately after this one. */}
          <div className="sb-seg sb-queue"
               title={Object.keys(m.byTransport || {}).length
                 ? 'by transport: ' + Object.entries(m.byTransport).map(([t, n]) => `${t} ${n}`).join(' · ')
                 : 'in-flight generation across web · /v1 · discord · cli'}>
            <span className="sb-k">Queue</span>
            {m.active || m.pending ? (
              <span className="sb-v">
                <span className={m.active ? 'sb-active' : 'sb-zero'}>{m.active} active</span>
                <span className="sb-mid">·</span>
                <span className={m.pending ? 'sb-pending' : 'sb-zero'}>{m.pending} pending</span>
                {Object.entries(m.byTransport || {}).map(([t, n]) => (
                  <span key={t} className="sb-transport">{t} {n}</span>
                ))}
              </span>
            ) : <span className="sb-v sb-zero">idle</span>}
          </div>

          {/* Downloads — their OWN queue, distinct from the inference Queue tile
              above. Counts only ACTIVE (queued/running) pulls. */}
          <DownloadsQueue />

          <Seg k="Workers" v={`${m.wOnline}/${m.wTotal}`} sub="online" />
          <Seg k="Slots"
               v={m.slotsEnabled === false ? 'off' : `${m.slotsBusy}/${m.slotsTotal}`}
               sub={m.slotsEnabled === false ? undefined : 'busy'} />
          <Seg k="Phones" v={`${m.phOnline}/${m.phTotal}`} sub="online" />

          <Meter k="VRAM" used={m.vramUsed} total={m.vramTotal} show={m.hasVram}
                 expanded={showRes} onToggle={() => setShowRes(v => !v)} />
          <Meter k="Host RAM" used={m.ramUsed} total={m.ramTotal} show={!!m.ramTotal} sub="from slots"
                 expanded={showRes} onToggle={() => setShowRes(v => !v)} />

          {/* Alerts — a column, sitting inline with Host RAM and the rest of
              the tiles (not floated off on its own line). */}
          <div className={`sb-alerts${m.alerts.length ? '' : ' sb-alerts-ok'}`}>
            {m.alerts.length === 0
              ? <span className="sb-ok">✓ all healthy</span>
              : m.alerts.map(a => <span key={a.k} className={`sb-alert sb-alert-${a.cls}`}>{a.text}</span>)}
          </div>
        </div>

        {/* Fleet residency — WHICH models hold the VRAM/RAM the meters above
            summarize, and what each is doing. Toggled by either meter. */}
        {showRes && <FleetResidency workers={workers} queue={queue} />}

        {/* Peers — strictly its own row, below the metrics row. */}
        <PeersBar />
    </div>
  )
}

function Seg({ k, v, sub }) {
  return (
    <div className="sb-seg">
      <span className="sb-k">{k}</span>
      <span className="sb-v">{v}{sub && <span className="sb-sub"> {sub}</span>}</span>
    </div>
  )
}

// A meter tile. When `onToggle` is supplied the whole tile is the expander for
// the fleet-residency breakdown — the ▸/▾ caret is the discoverability cue.
function Meter({ k, used, total, show, sub, expanded, onToggle }) {
  return (
    <div className={`sb-seg sb-meter${onToggle ? ' sb-meter-x' : ''}`}
         onClick={onToggle}
         role={onToggle ? 'button' : undefined}
         tabIndex={onToggle ? 0 : undefined}
         onKeyDown={onToggle ? (e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle() } }) : undefined}
         title={onToggle ? (expanded ? 'hide fleet model residency' : 'show fleet model residency') : undefined}>
      <span className="sb-k">{k}{onToggle && <span className="sb-caret">{expanded ? '▾' : '▸'}</span>}</span>
      {show ? (
        <>
          <span className="sb-v">{fmtBytes(used)} <span className="sb-sub">/ {fmtBytes(total)}</span></span>
          <div className="sb-bar-track"><div className="sb-bar-fill" style={{ width: `${Math.min(100, Math.round((used / total) * 100))}%` }} /></div>
          {sub && <span className="sb-note">{sub}</span>}
        </>
      ) : <span className="sb-v">—</span>}
    </div>
  )
}
