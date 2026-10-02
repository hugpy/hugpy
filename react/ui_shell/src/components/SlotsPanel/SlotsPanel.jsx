import { useEffect, useState, useCallback, useMemo } from 'react'
import { fetchJson } from '../../api'
import ModelPicker from '../ModelPicker/ModelPicker'
import useSessionState from '../../hooks/useSessionState'
import './SlotsPanel.css'

function fmtBytes(n) {
  if (n == null) return '?'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// Relative age from an epoch-seconds timestamp (slot loaded_at / last_used).
function fmtAge(ts) {
  if (!ts) return null
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts)))
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`
  return `${Math.floor(secs / 86400)}d ago`
}

// Elapsed since a timestamp, for the loading timer ("45s" / "2m 5s").
function fmtDur(ts) {
  if (!ts) return ''
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts)))
  return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${secs % 60}s`
}

// A single slot supervisor: what it's serving (or loading / idle / down) + controls.
function SlotCard({ slot, nameFor, onUnload, busy, pinned, gen }) {
  const control = slot._control
  // model_key set + healthy=False + no error == still loading; +healthy == serving;
  // model_key null == idle; error == down.
  const down = !!slot.error
  const loading = !down && !!slot.model_key && !slot.healthy
  const serving = !down && !!slot.model_key && slot.healthy
  const state = down ? 'down' : serving ? 'serving' : loading ? 'loading' : 'idle'
  const stateTitle = down ? (slot.error || 'slot unreachable')
    : serving ? 'a llama-server child is loaded and serving'
    : loading ? 'model is loading into this slot — can take a while for large models'
    : 'reachable, no model loaded'
  const icon = down ? '✗' : serving ? '🔥' : loading ? '⏳' : '○'
  const rss = slot.rss_bytes || 0
  const exp = slot.expected_bytes || 0
  const pct = (exp && loading) ? Math.min(99, Math.round((rss / exp) * 100)) : null

  return (
    <div className={`sp-slot sp-st-${state}`}>
      <div className="sp-slot-head">
        <span className={`sp-state-pill sp-pill-${state}`} title={stateTitle}>
          {icon} {state}
        </span>
        <span className="sp-slot-id">slot {slot.slot_id ?? '?'}</span>
        {slot.allowed_cpus && (
          <span className="sp-tag sp-dedicated"
                title="dedicated CPU cores — kernel-enforced via cgroup AllowedCPUs (hugpy-slot-cpus)">
            🔒 cores {slot.allowed_cpus}
          </span>
        )}
        <span className="sp-slot-url" title={`control: ${control}${slot.endpoint ? `\nendpoint: ${slot.endpoint}` : ''}`}>
          {control}
        </span>
        {(serving || loading) && (
          <button className="sp-unload"
                  title={loading ? 'Cancel the load and free this slot' : 'Unload the model from this slot (frees it)'}
                  disabled={busy} onClick={() => onUnload(control)}>
            {loading ? '✕ cancel' : '⏏ unload'}
          </button>
        )}
      </div>

      {(serving || loading) && (
        <div className="sp-slot-model">
          <span className="sp-model-name">{nameFor(slot.model_key)}</span>
          {pinned && (
            <span className="sp-tag sp-tag-warn"
                  title="This model also has a systemd always-on unit — it may be resident TWICE on the GPU (this slot + the unit). Requests route to the slot, so the unit gets no traffic. Disable slots to serve via the unit, or set the model's Serving to swap/off.">
              ⚠ also systemd-pinned
            </span>
          )}
          {gen && gen.generating > 0 && (
            <span className="sp-tag sp-tag-gen" title="inference requests generating tokens on this model right now">
              ▶ generating{gen.generating > 1 ? ` ×${gen.generating}` : ''}
            </span>
          )}
          {gen && gen.waiting > 0 && (
            <span className="sp-tag sp-tag-dim" title="requests for this model waiting for the slot">
              ⏳ {gen.waiting} queued
            </span>
          )}
          {loading && (
            <span className="sp-tag sp-loading-tag" title="loading — large models can take a couple minutes">
              ⏳ loading… {fmtDur(slot.loaded_at)}
            </span>
          )}
          {slot.n_gpu_layers != null && (
            <span className="sp-tag" title="GPU layers offloaded / model total">
              ngl {String(slot.n_gpu_layers)}{slot.total_layers != null ? `/${slot.total_layers}` : ''}
            </span>
          )}
          {slot.threads != null && <span className="sp-tag" title="CPU threads">{slot.threads}t</span>}
          {rss > 0 && (
            // Serving + honest split known: lead with the truly-pinned (anon)
            // RAM; the mmap'd GGUF pages VmRSS also counts are reclaimable page
            // cache, shown as "+ N cache". While LOADING keep raw VmRSS — pages
            // faulting in (anon or file) ARE the load-progress signal.
            (!loading && slot.rss_anon_bytes != null) ? (
              <span className="sp-tag sp-tag-ram"
                    title="pinned RAM (anonymous) + mmap'd model pages (reclaimable page cache) — VmRSS counts both">
                RAM {fmtBytes(slot.rss_anon_bytes)}
                {slot.rss_file_bytes > 0 ? ` + ${fmtBytes(slot.rss_file_bytes)} cache` : ''}
              </span>
            ) : (
              // VmRSS-only fallback (pre-split worker, or mid-load). VmRSS
              // counts the mmap'd GGUF's file-backed pages — reclaimable page
              // cache, not pinned RAM (~28x overstated on ae) — so it is NEVER
              // labeled "model footprint": it is named VmRSS, ~-marked and
              // muted so it can't be read as a measured occupancy figure.
              <span className="sp-tag sp-tag-ram sp-tag-est"
                    title={loading && exp
                      ? 'VmRSS / model size on disk — rough load progress (VmRSS includes mmap\'d file pages)'
                      : 'VmRSS — includes mmap\'d file pages, overstates pinned RAM'}>
                VmRSS ~{fmtBytes(rss)}{loading && exp ? ` / ${fmtBytes(exp)} · ${pct}%` : ''}
              </span>
            )
          )}
          {slot.cpus && <span className="sp-tag" title="pinned CPU cores">cores {slot.cpus}</span>}
          {slot.gpu != null && slot.gpu !== '' && (
            <span className="sp-tag" title="pinned GPU index">gpu {String(slot.gpu)}</span>
          )}
          {slot.ctx != null && <span className="sp-tag" title="context window">ctx {slot.ctx}</span>}
          {serving && fmtAge(slot.last_used) && (
            <span className="sp-tag sp-tag-dim" title="last request served">used {fmtAge(slot.last_used)}</span>
          )}
        </div>
      )}

      {loading && (
        <div className="sp-slot-progress" title={pct != null ? `~${pct}%` : 'loading…'}>
          {pct != null
            ? <div className="sp-slot-progress-bar sp-determinate" style={{ width: `${pct}%` }} />
            : <div className="sp-slot-progress-bar" />}
        </div>
      )}

      {down && slot.error && <div className="sp-slot-err" title={slot.error}>{slot.error}</div>}

      {!down && slot.free_vram_bytes != null && (
        <div className="sp-slot-vram">VRAM free: {fmtBytes(slot.free_vram_bytes)}</div>
      )}
    </div>
  )
}

export default function SlotsPanel({ models = [], embedded = false }) {
  const [enabled, setEnabled] = useState(null)   // null = unknown, before first load
  const [slots, setSlots]     = useState([])
  const [error, setError]     = useState(null)
  const [open, setOpen]       = useSessionState('hugpy.sess.sp.open', false)  // session-sticky expansion
  const [pick, setPick]       = useState('')
  const [busy, setBusy]       = useState(false)
  const [steps, setSteps]     = useState(null)   // install dry-run steps when disabled
  const [opts, setOpts]       = useState({ threads: '', cpus: '', gpu: '', n_gpu_layers: '', ctx: '' })
  const [, setTick]           = useState(0)   // 1s re-render while loading, to tick the elapsed timer
  const [res, setRes]         = useState(null) // system RAM/cores from /api/llm/slots
  const [serving, setServing] = useState([])   // /api/llm/serving rows (serve_mode/always_on)
  const [cache, setCache]     = useState(null)  // /api/llm/cache (SSD hot-cache status)
  const [queue, setQueue]     = useState([])    // /api/llm/queue active (live inference)

  const load = useCallback(() => {
    fetchJson('/api/llm/slots')
      .then(d => {
        const data = (d && typeof d === 'object') ? d : {}
        setEnabled(data.enabled !== false)
        setSlots(Array.isArray(data.slots) ? data.slots : [])
        setRes(data.resources || null)
        setError(null)
      })
      .catch(e => setError(e.message))
    // Serving overrides — to surface systemd always-on units + flag double GPU
    // residency (a model pinned via systemd AND loaded into a slot).
    fetchJson('/api/llm/serving')
      .then(rows => setServing(Array.isArray(rows) ? rows : []))
      .catch(() => {})
    // SSD hot-cache status (used/budget + what's warmed/warming).
    fetchJson('/api/llm/cache')
      .then(c => setCache(c && c.enabled ? c : null))
      .catch(() => {})
    // Live inference queue — to denote which loaded model is generating/queued.
    fetchJson('/api/llm/queue')
      .then(r => setQueue(Array.isArray(r?.active) ? r.active : []))
      .catch(() => {})
  }, [])

  // Poll fast (2.5s) while anything is loading so the timer ticks and the card
  // flips to "serving" promptly; otherwise the usual 10s.
  const anyLoading = useMemo(
    () => slots.some(s => s.model_key && !s.healthy && !s.error),
    [slots],
  )
  const pollFast = anyLoading || queue.length > 0   // live during loads AND inference
  useEffect(() => {
    load()
    const t = setInterval(load, pollFast ? 2000 : 10_000)
    return () => clearInterval(t)
  }, [load, pollFast])

  // Smooth per-second elapsed timer while loading (local re-render only — the
  // API poll above stays at 2.5s).
  useEffect(() => {
    if (!anyLoading) return
    const t = setInterval(() => setTick(n => n + 1), 1000)
    return () => clearInterval(t)
  }, [anyLoading])

  // Show the owner when the key is collision-qualified (owner~name) so two
  // same-named variants (e.g. unsloth~/Qwen~ Coder-Next) loaded in different
  // slots are visually distinct — the bare `name` is identical for both.
  const nameFor = useCallback(
    key => {
      const m = models.find(mm => (mm.model_key ?? mm.key) === key)
      const base = m?.name || key
      return key && key.includes('~') ? `${base} (${key.split('~')[0]})` : base
    },
    [models],
  )

  const loadModel = useCallback(async () => {
    if (!pick) return
    setBusy(true)
    try {
      const body = { model_key: pick }
      for (const [k, v] of Object.entries(opts)) {
        if (v !== '' && v != null) body[k] = v   // blank = autofit/default
      }
      const r = await fetchJson('/api/llm/slots/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (r && Array.isArray(r.slots)) setSlots(r.slots)
      if (r && r.loaded === false) alert(`Could not load: ${r.reason || 'all slots busy'}`)
      setPick('')
      load()
    } catch (err) {
      alert(`Load failed: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }, [pick, opts, load])

  // Background-warm the picked model onto the SSD cache (so its next load is
  // NVMe-fast). Independent of free slots — you can warm while slots are full.
  const warmModel = useCallback(async () => {
    if (!pick) return
    setBusy(true)
    try {
      const r = await fetchJson('/api/llm/cache/warm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: pick }),
      })
      if (r && r.already_warm) alert(`${nameFor(pick)} is already on the SSD cache.`)
      else if (r && r.warming) alert(`Warming ${nameFor(pick)} to SSD in the background — it'll load fast once done.`)
      else if (r && r.error) alert(`Warm: ${r.error}`)
    } catch (err) {
      alert(`Warm failed: ${err.message}`)
    } finally {
      setBusy(false)
      load()
    }
  }, [pick, nameFor, load])

  const unload = useCallback(async (control) => {
    setBusy(true)
    try {
      await fetchJson('/api/llm/slots/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ control }),
      })
      load()
    } catch (err) {
      alert(`Unload failed: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }, [load])

  const showInstall = useCallback(async () => {
    try {
      const r = await fetchJson('/api/llm/slots/install')
      setSteps(Array.isArray(r.steps) ? r.steps : [])
    } catch (err) {
      alert(`Could not fetch install steps: ${err.message}`)
    }
  }, [])

  // Recycle the API worker to release its in-process (anon) RAM. The POST may not
  // return cleanly (the worker recycles mid-request) — that's expected; re-poll
  // after a few seconds once the fresh worker is up.
  const freeWorker = useCallback(async () => {
    if (!confirm('Recycle the API worker to free its in-process RAM?\n\n'
      + 'Frees anon memory (in-process embed/media models, caches). Slots are '
      + 'unaffected. The API blips for a few seconds.')) return
    setBusy(true)
    try { await fetchJson('/api/llm/free-worker', { method: 'POST' }) } catch (_) { /* expected */ }
    setTimeout(() => { setBusy(false); load() }, 5000)
  }, [load])

  const busyCount = useMemo(() => slots.filter(s => s.model_key).length, [slots])
  const downCount = useMemo(() => slots.filter(s => s.error).length, [slots])

  const assignable = useMemo(
    () => slots.some(s => !s.model_key && !s.error),
    [slots],
  )
  // enabled but every supervisor unreachable = pool configured, services not installed yet.
  const allDown = slots.length > 0 && downCount === slots.length
  const needsInstall = enabled === false || allDown
  // Header RAM figure: per-slot, prefer the honest pinned figure (rss_anon —
  // VmRSS also counts mmap'd GGUF pages, i.e. reclaimable page cache) when the
  // slot reports the split; raw VmRSS for older slot builds.
  const slotRam = useMemo(
    () => slots.reduce((a, s) => a + ((s.rss_anon_bytes ?? s.rss_bytes) || 0), 0),
    [slots])
  const slotThreads = useMemo(() => slots.reduce((a, s) => a + (s.threads || 0), 0), [slots])
  // A deliberate pin = the operator EXPLICITLY set serve_mode=systemd in the
  // override overlay — NOT the DEFAULT_SERVE_MODE=systemd that marks every model
  // "systemd" in the resolved row. These serve via their own always-on unit,
  // independent of slots; with slots enabled, requests still hit slots first, so
  // a pinned model can end up resident twice on the GPU.
  const pinned = useMemo(
    () => serving.filter(r => r && r.override && r.override.serve_mode === 'systemd'),
    [serving],
  )
  const pinnedKeys = useMemo(() => new Set(pinned.map(r => r.key)), [pinned])
  const doubleLoaded = useMemo(
    () => slots.filter(s => s.model_key && pinnedKeys.has(s.model_key)).map(s => s.model_key),
    [slots, pinnedKeys],
  )
  const hasConflict = enabled !== false && pinned.length > 0
  // Live inference per loaded model — to badge the slot that's generating/queued.
  const genByKey = useMemo(() => {
    const m = {}
    for (const e of queue) {
      if (!e.model_key) continue
      const g = m[e.model_key] || { generating: 0, waiting: 0 }
      if (e.state === 'active') g.generating++; else g.waiting++
      m[e.model_key] = g
    }
    return m
  }, [queue])

  return (
    <div className="slots-panel">
      <div className={`sp-bar${embedded ? ' sp-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="sp-title">🧩 Model Slots</span>
        {enabled === false
          ? <span className="sp-count">not enabled</span>
          : <span className="sp-count">{busyCount} busy / {slots.length} slots</span>}
        {downCount > 0 && (
          <span className="sp-down-chip" title="slot supervisors that are unreachable">
            ✗ {downCount} down
          </span>
        )}
        {res && res.total_bytes && (
          <span className="sp-res" title="VM RAM available (free + reclaimable cache) / total — full breakdown in the bar inside">
            RAM {fmtBytes(res.available_bytes)} free / {fmtBytes(res.total_bytes)}
            {res.cpu_count ? ` · ${res.cpu_count} cores` : ''}
          </span>
        )}
        {slotRam > 0 && (
          <span className="sp-res sp-res-pool" title="RAM + threads currently held by the slot pool">
            slots {fmtBytes(slotRam)}{slotThreads ? ` · ${slotThreads}t` : ''}
          </span>
        )}
        {error && <span className="sp-err" title={error}>slots error</span>}
        {!embedded && <span className="sp-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="sp-body">
          {res && res.total_bytes && (
            <div className="sp-mem">
              <div className="sp-mem-bar"
                   title={`VM RAM — used ${fmtBytes(res.used_bytes)} · reclaimable cache ${fmtBytes(res.cache_bytes)} · free ${fmtBytes(res.free_bytes)} → ${fmtBytes(res.available_bytes)} available of ${fmtBytes(res.total_bytes)}. Cache is auto-freed on demand, so "available" is what a load can use.`}>
                <span className="sp-mem-used"  style={{ width: `${100 * (res.used_bytes || 0) / res.total_bytes}%` }} />
                <span className="sp-mem-cache" style={{ width: `${100 * (res.cache_bytes || 0) / res.total_bytes}%` }} />
                <span className="sp-mem-free"  style={{ width: `${100 * (res.free_bytes || 0) / res.total_bytes}%` }} />
              </div>
              <div className="sp-mem-legend">
                <span><i className="sp-sw sp-sw-used" />{fmtBytes(res.used_bytes)} used</span>
                <span><i className="sp-sw sp-sw-cache" />{fmtBytes(res.cache_bytes)} cache</span>
                <span><i className="sp-sw sp-sw-free" />{fmtBytes(res.free_bytes)} free</span>
                <span className="sp-mem-avail" title="cache is reclaimable — this is what a model load can actually use">
                  {fmtBytes(res.available_bytes)} available
                </span>
                {res.cpu_count ? <span className="sp-mem-cores">· {res.cpu_count} cores</span> : null}
                <button className="sp-free-btn" onClick={freeWorker} disabled={busy}
                        title="Recycle the API worker to release its in-process (anon) RAM. Slots unaffected; brief API blip.">
                  ♻ free worker RAM
                </button>
              </div>
            </div>
          )}
          {enabled !== false && (
            <>
              {hasConflict && (
                <div className="sp-warn-banner"
                     title="Routing prefers the slot pool, so requests do NOT reach a systemd always-on unit while slots are enabled — the pinned model can sit on the GPU twice (slot + unit). Either disable slots (SLOT_COUNT=0) to serve via the unit, or set the model's Serving to swap/off.">
                  ⚠ {pinned.length} model{pinned.length > 1 ? 's' : ''} pinned via <b>systemd always-on</b> while slots are enabled.
                  Requests route to slots first, so a pinned model can be resident <b>twice</b> on the GPU
                  {doubleLoaded.length ? ` (now: ${doubleLoaded.map(nameFor).join(', ')})` : ''}.
                  Disable slots to use the unit, or set those models’ Serving to swap/off.
                </div>
              )}
              <div className="sp-load-row">
                <ModelPicker
                  models={models}
                  value={pick}
                  onPick={setPick}
                  disabled={busy}
                  placeholder={assignable ? 'Pick a model — load into a free slot, or warm to SSD…' : 'Pick a model to warm to SSD (no free slot to load)…'}
                />
                <button className="sp-load-btn" onClick={loadModel} disabled={busy || !pick || !assignable}
                        title={assignable ? 'Load into a free slot' : 'No free slot available'}>
                  {busy ? '…' : '+ Load'}
                </button>
                <button className="sp-warm-btn" onClick={warmModel} disabled={busy || !pick}
                        title="Copy this model to the SSD cache so its next load is NVMe-fast (background; works even when slots are full)">
                  ⇪ Warm
                </button>
              </div>
              {cache && (
                <div className="sp-cache-line"
                     title={`SSD hot-cache at ${cache.dir} — warmed models load NVMe-fast; LRU-evicted at the budget`}>
                  SSD cache: {fmtBytes(cache.used_bytes)} / {fmtBytes(cache.max_bytes)} · {cache.entries.length} model{cache.entries.length !== 1 ? 's' : ''}
                  {cache.warming && cache.warming.length ? ` · ⇪ warming ${cache.warming.length}` : ''}
                </div>
              )}

              <div className="sp-opts" title="Optional compute for the next load — blank = autofit/default">
                <label>threads
                  <input type="number" min="1" placeholder="def" value={opts.threads}
                         onChange={e => setOpts(o => ({ ...o, threads: e.target.value }))} />
                </label>
                <label title="dedicated CPU cores, e.g. 0-3 or 0,2,4">cores
                  <input type="text" placeholder="0-3" value={opts.cpus}
                         onChange={e => setOpts(o => ({ ...o, cpus: e.target.value }))} />
                </label>
                <label title="pin to this GPU index">gpu
                  <input type="number" min="0" placeholder="auto" value={opts.gpu}
                         onChange={e => setOpts(o => ({ ...o, gpu: e.target.value }))} />
                </label>
                <label title="GPU layers to offload (-1 = all, 0 = CPU only)">gpu-layers
                  <input type="number" placeholder="autofit" value={opts.n_gpu_layers}
                         onChange={e => setOpts(o => ({ ...o, n_gpu_layers: e.target.value }))} />
                </label>
                <label title="context window in tokens (llama-server -c); blank = model default. Independent of cores.">ctx
                  <input type="number" min="1" placeholder="def" value={opts.ctx}
                         onChange={e => setOpts(o => ({ ...o, ctx: e.target.value }))} />
                </label>
              </div>

              <div className="sp-slots">
                {slots.length === 0 && <div className="sp-none">No slots reported.</div>}
                {slots.map((s, i) => (
                  <SlotCard key={s._control || i} slot={s} nameFor={nameFor}
                            onUnload={unload} busy={busy}
                            pinned={!!s.model_key && pinnedKeys.has(s.model_key)}
                            gen={s.model_key ? genByKey[s.model_key] : null} />
                ))}
              </div>

              {pinned.length > 0 && (
                <div className="sp-pinned">
                  <div className="sp-pinned-head"
                       title="Models with a dedicated systemd always-on llama-server unit. These run 24/7 independent of the slot pool (and only receive traffic when slots are disabled).">
                    📌 explicitly pinned · systemd <span className="sp-pinned-sub">(own always-on unit, independent of slots)</span>
                  </div>
                  {pinned.map(r => {
                    const u = r.unit || {}
                    const up = u.active === true
                    const unk = u.active == null
                    const ustate = unk ? 'unit ?' : (up ? 'running' : (u.state || 'stopped'))
                    return (
                    <div key={r.key} className="sp-pinned-row">
                      <span className={`sp-unit-state sp-unit-${unk ? 'unk' : up ? 'up' : 'down'}`}
                            title={`systemd unit ${u.unit || ''}: ${ustate}${u.sub_state ? ` (${u.sub_state})` : ''}`}>
                        {unk ? '· unit ?' : up ? '● running' : `○ ${ustate}`}
                      </span>
                      <span className="sp-pinned-name">{nameFor(r.key)}</span>
                      {r.port ? <span className="sp-tag" title="dedicated unit port">:{r.port}</span> : null}
                      {r.n_gpu_layers != null && <span className="sp-tag" title="GPU layers offloaded">ngl {String(r.n_gpu_layers)}</span>}
                      {r.ctx_size != null && <span className="sp-tag" title="context window">ctx {r.ctx_size}</span>}
                      {u.rss_bytes ? <span className="sp-tag sp-tag-ram" title="unit cgroup RAM (resident)">RAM {fmtBytes(u.rss_bytes)}</span> : null}
                      {up && u.since ? <span className="sp-tag sp-tag-dim" title="unit active since">up {fmtAge(u.since)}</span> : null}
                      {doubleLoaded.includes(r.key) && (
                        <span className="sp-tag sp-tag-warn"
                              title={up
                                ? 'This pinned unit is RUNNING and the model is also in a slot — actual double GPU residency.'
                                : 'Model is in a slot; the pinned unit is configured but not running (no real double-residency).'}>
                          ⚠ also in a slot
                        </span>
                      )}
                    </div>
                  )})}
                </div>
              )}
            </>
          )}

          {needsInstall && (
            <div className="sp-install">
              <p className="sp-install-label">
                {enabled === false
                  ? <>The slot pool isn’t enabled on this install (<code>SLOT_COUNT=0</code>). Install the generic slot services once, then the app drives them over HTTP — no per-model units.</>
                  : <>The pool is enabled but no slot supervisor is reachable — the <code>abstract-hugpy-slot@</code> services aren’t installed/running yet. Install them once:</>}
              </p>
              <button className="sp-install-btn" onClick={showInstall}>Show install steps</button>
              {steps && steps.length === 0 && <div className="sp-none">No install steps reported.</div>}
              {steps && steps.length > 0 && (
                <div className="sp-steps">
                  {steps.map((st, i) => {
                    const kind = st.kind || ''
                    if (kind.startsWith('write:')) {
                      const path = kind.slice('write:'.length)
                      return (
                        <div key={i} className="sp-step">
                          <div className="sp-step-kind">write <code>{path}</code></div>
                          <pre className="sp-step-payload">{st.payload}</pre>
                        </div>
                      )
                    }
                    return (
                      <div key={i} className="sp-step">
                        <code className="sp-step-cmd" title="run with sudo">$ {st.payload}</code>
                      </div>
                    )
                  })}
                  <p className="sp-install-note">Dry run — run the writes/commands with sudo on this host.</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
