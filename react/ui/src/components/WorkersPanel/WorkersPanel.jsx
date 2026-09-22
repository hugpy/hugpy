import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchJson } from '../../api'
import { useFeed } from '../../runtime/feeds'
import { resolveApiOrigin } from '../../runtime/config'
import ModelPicker from '../ModelPicker/ModelPicker'
import FixDoc from '../FixDoc/FixDoc'
import useSessionState from '../../hooks/useSessionState'
import { allocIsGgufOnly, allocModeLabel, spillLabel } from './allocation'
import { fmtBytes } from './formatters'
import { findCatalogRow } from './catalogRow'
import { GroupAssignPanel } from './GroupAssignPanel'
import { WorkerRow } from './WorkerRow'
import './WorkersPanel.css'

export default function WorkersPanel({ models = [], embedded = false }) {
  const [workers, setWorkers] = useState([])
  const [error, setError]     = useState(null)
  const [open, setOpen]       = useSessionState('hugpy.sess.wp.open', false)  // session-sticky expansion
  const [form, setForm]       = useState({ name: '', url: '', models: '' })
  const [busy, setBusy]       = useState(false)
  const [tokens, setTokens]   = useState([])
  const [central, setCentral] = useState(null)     // /llm/slots — central's own row
  // Central-as-worker load controls (replaced the separate Model Slots panel).
  const [showCentralLoad, setShowCentralLoad] = useState(false)
  const [centralPick, setCentralPick] = useState('')
  const [centralBusy, setCentralBusy] = useState(false)
  const [newToken, setNewToken] = useState(null)   // freshly-minted plaintext, shown once
  // Transient per-worker "restarting…" flag. A restart re-execs the agent, so
  // the row goes offline→online on its own via the 10s load() poll; this just
  // reflects the click until the poll (or a fallback timeout) retires it.
  const [restarting, setRestarting] = useState({})  // worker.id -> true
  // Same transient for ⬆ Update: the worker pip-installs then restarts itself,
  // so it wears its own flag (a longer fallback — pip is slower than a re-exec).
  const [updating, setUpdating] = useState({})      // worker.id -> true

  // ── Per-worker TABS (operator, 2026-07-26) ────────────────────────────────
  // The pool used to render every worker's card stacked down the page, which
  // got long fast once the serving list stopped being a 260px box (b1b2222).
  // One card at a time instead: a tab strip picks the worker, and the selection
  // persists per browser like the other console view state.
  const [activeWorker, setActiveWorker] = useSessionState('hugpy.sess.wp.tab', '')

  // Resolve the tab selection defensively: a worker can be removed, renamed or
  // simply not be in the pool this session, and a stale id in sessionStorage
  // must never render an empty panel. Falls back to the first worker.
  const selectedWorkerId = useMemo(() => {
    if (activeWorker && workers.some(w => w.id === activeWorker)) return activeWorker
    return workers[0]?.id || ''
  }, [activeWorker, workers])
  // With a single worker there is no strip, so nothing is filtered out.
  const shownWorkers = useMemo(
    () => (workers.length > 1 ? workers.filter(w => w.id === selectedWorkerId) : workers),
    [workers, selectedWorkerId])
  // Operator model BLOCK (global serving-pool primitive): the set of model_keys
  // blocked everywhere. Seeded from the /models feed (`blocked` field on each
  // row) and kept as local state so a toggle reflects instantly without waiting
  // on the parent's models refetch (App fetches /models once). See toggleBlock.
  const propBlocked = useMemo(
    () => new Set((models || [])
      .filter(m => m && m.blocked)
      .map(m => m.model_key ?? m.key)),
    [models])
  const [blockedKeys, setBlockedKeys] = useState(propBlocked)
  useEffect(() => { setBlockedKeys(propBlocked) }, [propBlocked])
  // Setup chrome (install command, tokens, manual add) is collapsed by default so
  // the live worker list is the focus — but auto-opens when there are no workers
  // yet, since the install command is the only way to add the first one.
  const [setupOpen, setSetupOpen] = useSessionState('hugpy.sess.wp.setupOpen', false)
  // Apply-blip guard (pin / residency / slot-count): a successful /config POST
  // replies {restarting: true} and the agent re-execs ~0.5s later, so its
  // reported config stays stale for a heartbeat or two and a re-click during
  // the blip hits a dead socket (surfaced as a bare 502 — "pin is broken").
  // Track an "applying" window per worker: disable that worker's config
  // controls and soften errors until a refresh shows the change (or ~10s).
  const [applying, setApplying] = useState({})  // worker.id -> {t, kind, model?, value}
  const markApplying = useCallback((workerId, exp) => {
    setApplying(a => ({ ...a, [workerId]: { t: Date.now(), ...exp } }))
  }, [])
  useEffect(() => {
    if (Object.keys(applying).length === 0) return undefined
    const met = (w, exp) => {
      if (!w || !w.config) return false
      if (exp.kind === 'residency') {
        // Two-tier policy: static is the only stored override; everything
        // else (no entry / legacy values) reads as the on-demand default.
        const cur = w.config.residency?.[exp.model] ?? null
        return exp.value === 'static' ? cur === 'static' : cur !== 'static'
      }
      if (exp.kind === 'pinned') return !!w.config.pinned?.[exp.model] === !!exp.value
      if (exp.kind === 'pin_all') {
        // Bulk pin/unpin: cleared once every affected model matches the target.
        const p = w.config.pinned || {}
        return (exp.models || []).every(mk => !!p[mk] === !!exp.value)
      }
      if (exp.kind === 'residency_all') {
        // Bulk residency: cleared once every selected model's stored tier
        // matches the target (static stored verbatim; on-demand = no entry).
        const res = w.config.residency || {}
        return (exp.models || []).every(mk =>
          exp.value === 'static' ? res[mk] === 'static' : res[mk] !== 'static')
      }
      if (exp.kind === 'slot_count') return w.config.slot_count === exp.value
      return false
    }
    const sweep = () => setApplying(a => {
      let changed = false
      const n = { ...a }
      for (const [wid, exp] of Object.entries(a)) {
        const w = workers.find(x => x.id === wid)
        if (Date.now() - exp.t > 10_000 || met(w, exp)) { delete n[wid]; changed = true }
      }
      return changed ? n : a
    })
    sweep()                                   // a refresh may already show it
    const t = setTimeout(sweep, 10_500)       // …or age the window out
    return () => clearTimeout(t)
  }, [workers, applying])

  // Returns a promise so the caller can WAIT before scheduling the next poll
  // (see the self-scheduling effect below). Without this the backpressure loop
  // would resolve instantly and degenerate back into setInterval.
  const load = useCallback(() => {
    const pWorkers = fetchJson('/api/llm/workers')
      .then(data => {
        // A POLL MUST NEVER DESTROY GOOD DATA. This used to be
        //   setWorkers(Array.isArray(data) ? data : [])
        // so ANY non-array reply — an error object, a partial, a body returned
        // while a worker agent is mid-re-exec — blanked the whole pool list.
        // The operator saw pools "constantly disappearing simply because they
        // are being called", from a strictly sequential one-model-at-a-time
        // script: not load, just a single degraded reply landing between good
        // ones. Keep the last known roster and surface the problem instead;
        // stale-but-labelled beats empty-and-silent.
        if (Array.isArray(data)) { setWorkers(data); setError(null) }
        else setError('worker list unavailable (kept the last known roster)')
      })
      .catch(e => setError(e.message))
    // Central-as-worker: its compute (the local slot pool) renders as a
    // footer card BELOW every remote worker row, same anatomy as a remote
    // worker — it's secondary context (this console's own host), not the
    // fleet being managed.
    const pSlots = fetchJson('/api/llm/slots')
      // Same rule as the worker roster above: a falsy/degraded reply keeps the
      // last known card rather than blanking it. Only a well-formed object
      // replaces what is on screen.
      .then(d => { if (d && typeof d === 'object') setCentral(d) })
      .catch(() => {})
    // Both settle before the next poll is scheduled — neither can pile up.
    return Promise.all([pWorkers, pSlots])
  }, [])

  const loadTokens = useCallback(() => {
    fetchJson('/api/llm/enroll-tokens')
      .then(data => setTokens(Array.isArray(data) ? data : []))
      .catch(() => {})   // tokens are a secondary panel; don't surface as a registry error
  }, [])

  useEffect(() => {
    // SELF-SCHEDULING, NOT setInterval — this is backpressure, not polish.
    //
    // setInterval(load, 10_000) fired every 10s whether or not the previous
    // call had returned. With /llm/workers at 31s (measured: list_workers()
    // alone is 11.8s cold for THREE workers) that means 3-4 copies in flight
    // permanently, each holding one of central's 24 gunicorn slots. The console
    // then starves itself: the browser log showed every /api/llm/workers line
    // with NO status and NO timing while every other endpoint returned.
    //
    // That is how ONE slow backend call takes down the whole view — it
    // MULTIPLIES instead of backing off. Waiting for the response before
    // scheduling the next makes the loop self-limiting: a slow endpoint
    // lowers its own poll rate to at most one in flight, so the panel degrades
    // to "less fresh" instead of "never loads", and it can never be the reason
    // other panels starve.
    //
    // Fixing the server side (making /llm/workers fast) is necessary but not
    // sufficient — without this, any future slow endpoint reproduces it.
    let cancelled = false
    let timer = null
    const tick = () => {
      Promise.resolve(load()).finally(() => {
        if (!cancelled) timer = setTimeout(tick, 10_000)
      })
    }
    // 2026-09-10: the roster and the central slot card now arrive over the ONE
    // live subscription (runtime/feeds.js) — see the useFeed effects below —
    // so this loop is no longer started. `load()` stays for the optimistic
    // refetch right after a verb (and the backpressure note above still holds
    // for that path). `tick` is kept so re-enabling polling is one line.
    void tick
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [load])

  const fWorkers = useFeed('workers', null)
  const fSlots = useFeed('slots', null)
  useEffect(() => { if (Array.isArray(fWorkers)) { setWorkers(fWorkers); setError(null) } }, [fWorkers])
  useEffect(() => { if (fSlots && typeof fSlots === 'object') setCentral(fSlots) }, [fSlots])

  useEffect(() => { loadTokens() }, [loadTokens])

  // Load into a free central slot (409 with a clear reason when none is free)
  // — the same backend the retired Model Slots panel drove.
  const centralLoad = useCallback(async (modelKey) => {
    setCentralBusy(true)
    try {
      const r = await fetchJson('/api/llm/slots/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.loaded === false) alert(`Not loaded: ${r.reason || 'no free slot'}`)
      setShowCentralLoad(false); setCentralPick('')
      load()
    } catch (err) { alert(`Load failed: ${err.message}`) }
    finally { setCentralBusy(false) }
  }, [load])

  const centralUnload = useCallback(async (control) => {
    try {
      await fetchJson('/api/llm/slots/unload', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ control }),
      })
      load()
    } catch (err) { alert(`Unload failed: ${err.message}`) }
  }, [load])

  const issueToken = useCallback(async () => {
    const label = prompt('Label for this enrollment token (e.g. gpu-box-2):', '')
    if (label === null) return
    try {
      const r = await fetchJson('/api/llm/enroll-tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      })
      setNewToken(r)   // contains the one-time plaintext `token`
      loadTokens()
    } catch (err) { alert(`Could not issue token: ${err.message}`) }
  }, [loadTokens])

  const revokeToken = useCallback(async (tok) => {
    if (!confirm(`Revoke token "${tok.label || tok.id}"? Workers using it are refused and their agents stop.`)) return
    try {
      await fetchJson(`/api/llm/enroll-tokens/${encodeURIComponent(tok.id)}`, { method: 'DELETE' })
      loadTokens()
    } catch (err) { alert(`Revoke failed: ${err.message}`) }
  }, [loadTokens])

  const register = useCallback(async (e) => {
    e.preventDefault()
    if (!form.name.trim()) return   // url is optional; central uses source IP
    setBusy(true)
    try {
      const body = {
        name: form.name.trim(),
        models: form.models.split(',').map(s => s.trim()).filter(Boolean),
      }
      if (form.url.trim()) body.url = form.url.trim()
      await fetchJson('/api/llm/workers/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setForm({ name: '', url: '', models: '' })
      load()
    } catch (err) {
      alert(`Could not register worker: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }, [form, load])

  const assign = useCallback(async (worker, modelKey, spill) => {
    try {
      const body = { model_key: modelKey }
      if (spill) body.spill = spill
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      await load()
    } catch (err) { alert(`Assign failed: ${err.message}`) }
  }, [load])

  // Guarded place-on-worker: VRAM+RAM preflight → assign → background warm.
  // The backend refuses (409) a model that won't fit the worker's free VRAM+RAM
  // (fetchJson throws with that reason); on a pass it returns immediately and the
  // worker warms the model — residency shows up via the next heartbeat.
  const loadModel = useCallback(async (worker, modelKey, spill, force) => {
    try {
      const body = { model_key: modelKey }
      if (spill) body.spill = spill
      if (force) body.force = true
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      // Fits, but only by spilling to CPU (RAM+VRAM ok, VRAM alone not) — warn.
      if (r && r.preflight && r.preflight.gpu_resident === false && r.preflight.reason) {
        alert(`Placed — note: ${r.preflight.reason}`)
      }
      load()
    } catch (err) {
      // 409 preflight refuse arrives here carrying the clear "won't fit …" reason.
      // Offer an override for the rare case the operator knows better.
      if (confirm(`${err.message}\n\nForce-load anyway? (may OOM the worker)`)) {
        loadModel(worker, modelKey, spill, true)
      }
    }
  }, [load])

  // Group-assign: dedicate ONE model to a SET of workers in a single action.
  // Each worker goes through the same /load preflight (VRAM+RAM+disk); a worker
  // that already has the model is skipped upstream in the UI (no double-booking)
  // and defended here too. Returns per-worker outcomes for the summary.
  const groupAssign = useCallback(async (modelKey, workerList, force) => {
    const targets = workerList.filter(w => !((w.models || []).includes(modelKey)))
    const settled = await Promise.all(targets.map(async (w) => {
      try {
        const body = { model_key: modelKey }
        if (force) body.force = true
        const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(w.id)}/load`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        const spilled = r && r.preflight && r.preflight.gpu_resident === false
        return { id: w.id, name: w.name, ok: true, note: spilled ? 'CPU-spilled' : '' }
      } catch (err) {
        return { id: w.id, name: w.name, ok: false, note: err.message }
      }
    }))
    load()
    return settled
  }, [load])

  // Per-worker bulk allocate (worker <- group of models). Each model via the
  // same /load preflight; anti-duplicate is enforced in the table UI (the
  // breaker flag rides along only for the summary). Reports per-model outcome.
  const allocateMany = useCallback(async (worker, modelKeys, breaker) => {
    const settled = await Promise.all((modelKeys || []).map(async (mk) => {
      try {
        const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/load`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_key: mk }),
        })
        const spilled = r && r.preflight && r.preflight.gpu_resident === false
        return { mk, ok: true, note: spilled ? 'CPU-spilled' : '' }
      } catch (err) { return { mk, ok: false, note: err.message } }
    }))
    load()
    const okN = settled.filter(r => r.ok).length
    const failed = settled.filter(r => !r.ok)
    let msg = `Allocated ${okN}/${settled.length} to ${worker.name}${breaker ? ' (duplicate-allocation breaker was ON)' : ''}.`
    if (failed.length) msg += `\n\nRefused:\n` + failed.map(r => `  • ${r.mk} — ${r.note}`).join('\n')
    if (failed.length || okN !== settled.length) alert(msg)
  }, [load])

  const unassign = useCallback(async (worker, modelKey) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unassign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      load()
    } catch (err) { alert(`Unassign failed: ${err.message}`) }
  }, [load])

  // Operator model BLOCK / UNBLOCK (global): removes a model from the serving
  // pool everywhere (never routed / assigned / warmed / a fallback default) or
  // returns it. Optimistically updates blockedKeys so the ⛔ chip flips at once;
  // reverts on error. Block does NOT unassign — the designation row stays (inert)
  // and pin is unaffected (block outranks pin at routing time). Operator token is
  // merged by hugpyFetch, same as every mutation here.
  const toggleBlock = useCallback(async (modelKey, block) => {
    if (block && !confirm(
      `Block "${modelKey}" from the serving pool?\n\n` +
      `It will no longer be routed to, assigned, warmed, or used as a fallback ` +
      `default anywhere — files stay on disk and existing designations stay ` +
      `recorded (inert). Reversible.`)) return
    setBlockedKeys(prev => {
      const next = new Set(prev)
      if (block) next.add(modelKey); else next.delete(modelKey)
      return next
    })
    try {
      await fetchJson(
        `/api/llm/models/${encodeURIComponent(modelKey)}/${block ? 'block' : 'unblock'}`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}) })
      load()
    } catch (err) {
      // revert the optimistic flip on failure
      setBlockedKeys(prev => {
        const next = new Set(prev)
        if (block) next.delete(modelKey); else next.add(modelKey)
        return next
      })
      alert(`${block ? 'Block' : 'Unblock'} failed: ${err.message}`)
    }
  }, [load])

  // Free GPU VRAM: evict a loaded model from the worker's cache. It stays
  // assigned — only the live VRAM is reclaimed (reloads on next request).
  const freeModel = useCallback(async (worker, modelKey) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.ok === false) alert(`Free failed: ${r.error || 'unknown error'}`)
      load()
    } catch (err) { alert(`Free failed: ${err.message}`) }
  }, [load])

  // Evict an attributed model from a worker's GPU process registry: POST the
  // operator-gated evict verb (the operator token is merged by hugpyFetch, same
  // as every other mutation here). Frees the model's VRAM/RAM by killing/unloading
  // its owning pid. Idempotent — returns evicted:false + a reason when the model
  // isn't resident. Surfaces the reason, then refetches so the freed VRAM and the
  // removed registry row reflect immediately. Returns the result so the calling
  // row can clear its in-flight state.
  const evictModel = useCallback(async (worker, modelKey) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/evict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
      if (r && r.ok === false) {
        alert(`Evict failed: ${r.error || r.reason || 'unknown error'}`)
      } else if (r && r.evicted === false) {
        alert(`Not evicted: ${r.reason || 'model is not resident on this worker'}`)
      }
      load()
      return r
    } catch (err) { alert(`Evict failed: ${err.message}`) }
  }, [load])

  const freeAll = useCallback(async (worker) => {
    if (!confirm(`Unload all models from ${worker.name}'s GPU? They stay assigned and reload on demand.`)) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true }),
      })
      if (r && r.ok === false) alert(`Free failed: ${r.error || 'unknown error'}`)
      load()
    } catch (err) { alert(`Free failed: ${err.message}`) }
  }, [load])

  // Free host RAM: ask the agent to return reclaimable process/arena memory to
  // the OS. NON-destructive — loaded models stay resident (unlike freeAll), so
  // there's no scary confirm. Surfaces how much was reclaimed.
  const freeRam = useCallback(async (worker) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/free-ram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (r && r.ok === false) { alert(`Free RAM failed: ${r.error || 'unknown error'}`); return }
      const freed = r && typeof r.ram_freed === 'number' ? r.ram_freed : null
      if (freed && freed > 0) {
        alert(`Freed ${(freed / 1073741824).toFixed(1)} GiB RAM on ${worker.name}`)
      } else {
        alert(`No reclaimable RAM on ${worker.name} right now.`)
      }
      load()
    } catch (err) { alert(`Free RAM failed: ${err.message}`) }
  }, [load])

  // Restart the worker's agent: drops all loaded models and re-execs the agent
  // process. During re-exec the relay may answer 503 {error:{code:"AgentRestarting"}}
  // — that's the expected transient, not a failure. The row flips offline→online
  // on its own via the 10s load() poll; we flag "restarting…" briefly meanwhile.
  const restart = useCallback(async (worker) => {
    if (!confirm(`Restart ${worker.name}'s worker agent? It drops all loaded models and re-execs the agent.`)) return
    setRestarting(s => ({ ...s, [worker.id]: true }))
    const clear = () => setRestarting(s => { const n = { ...s }; delete n[worker.id]; return n })
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    } catch (err) {
      // 503 AgentRestarting is the agent re-execing — expected, treat as success.
      if (!/\b503\b|AgentRestarting/i.test(err.message || '')) {
        alert(`Restart failed: ${err.message}`)
        clear()
        return
      }
    }
    // Let the offline→online poll retire the indicator; clear on a fallback timeout.
    setTimeout(clear, 12_000)
    load()
  }, [load])

  // Converge a worker onto central's required version NOW (instead of waiting
  // for its next heartbeat's self-update). The worker pip-installs --no-deps and
  // re-execs, so the SAME 503 {error:{code:"AgentRestarting"}} transient applies
  // — just with a longer fallback timeout, because pip install runs first. An
  // empty body means "central's required version" server-side.
  const update = useCallback(async (worker) => {
    if (!confirm(`Update ${worker.name} to central's required version (${worker.required_pkg_version || 'unknown'})? The worker pip-installs and restarts itself.`)) return
    setUpdating(s => ({ ...s, [worker.id]: true }))
    const clear = () => setUpdating(s => { const n = { ...s }; delete n[worker.id]; return n })
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    } catch (err) {
      // 503 AgentRestarting is the agent re-execing after the install — expected.
      if (!/\b503\b|AgentRestarting/i.test(err.message || '')) {
        alert(`Update failed: ${err.message}`)
        clear()
        return
      }
    }
    setTimeout(clear, 20_000)
    load()
  }, [load])

  // Reaper (tiers-v2 slice 4): preview first, then reclaim on confirm. The
  // worker re-proves every guard at delete time; pinned/assigned/loaded/comfy
  // are never reaped.
  const reap = useCallback(async (worker) => {
    try {
      const prev = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/reap`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: true }),
      })
      const items = (prev && prev.reclaimable) || []
      if (items.length === 0) {
        alert(`Nothing to reclaim on ${worker.name}. Everything on disk is assigned, loaded, or 📌pinned.`)
        return
      }
      // Honest binary units (ruling 5): the variable is named `gib`, so divide
      // by 2^30 (GiB) — matching fmtBytes and the rest of the panel — not the
      // decimal 1e9 it used before (which computed GB but labeled GiB).
      const gib = (prev.reclaimable_bytes || 0) / 1073741824
      const list = items.slice(0, 12).map(r => `  • ${r.model_key} (${(r.bytes / 1073741824).toFixed(1)} GiB)`).join('\n')
      const more = items.length > 12 ? `\n  …and ${items.length - 12} more` : ''
      if (!confirm(`Reclaim ${gib.toFixed(1)} GiB from ${worker.name} by deleting ${items.length} unassigned model(s)?\n\n${list}${more}\n\nProtected (assigned / loaded / 🔒static) files are left untouched. 📌 Pin does not protect files — it keeps the allocation/routing only.`)) return
      const res = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/reap`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true }),
      })
      const freed = ((res && res.freed_bytes) || 0) / 1073741824
      const failed = ((res && res.results) || []).filter(r => !r.ok)
      let msg = `Reclaimed ${freed.toFixed(1)} GiB from ${worker.name}.`
      if (failed.length) msg += `\n\nSkipped ${failed.length}:\n` + failed.map(r => `  • ${r.model_key} — ${r.reason}`).join('\n')
      alert(msg)
      load()
    } catch (err) { alert(`Reclaim failed: ${err.message}`) }
  }, [load])

  // Storage over-budget: approve the LRU eviction proposal central computed.
  // Human-in-the-loop — this fires ONLY on the explicit click; nothing is
  // auto-deleted. Central re-computes + intersects the proposal at approval
  // time (drops anything since loaded/assigned/static — NOT pinned; pin no
  // longer protects files as of 2026-07-17), then the worker re-proves every
  // guard per model before deleting. Prefers the dedicated
  // /reap-approve route (adds the central intersection guard); falls back to
  // /reap — which already re-guards worker-side — if it isn't deployed yet.
  const approveEvictions = useCallback(async (worker) => {
    const proposed = (worker.storage && worker.storage.proposed_evictions) || []
    if (proposed.length === 0) {
      alert(`Nothing proposed for eviction on ${worker.name} — it isn't over budget.`)
      return
    }
    const keys = proposed.map(p => p.model_key)
    const freed = ((worker.storage && worker.storage.proposed_free_bytes) || 0) / 1e9
    const list = proposed.slice(0, 12).map(p => `  • ${p.model_key} (${(p.bytes / 1e9).toFixed(1)} GB)`).join('\n')
    const more = proposed.length > 12 ? `\n  …and ${proposed.length - 12} more` : ''
    if (!confirm(`Approve eviction on ${worker.name}?\n\nFrees ~${freed.toFixed(1)} GB by deleting ${keys.length} cold, unprotected model(s):\n\n${list}${more}\n\nLoaded / 🔒static / assigned files are never touched. 📌 Pinned files ARE eligible — pin keeps the allocation, not the bytes (they re-pull on next call). Central re-checks this list and the worker re-proves each model before deleting.`)) return
    const url = `/api/llm/workers/${encodeURIComponent(worker.id)}/reap-approve`
    const body = JSON.stringify({ model_keys: keys })
    const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }
    try {
      let res
      try {
        res = await fetchJson(url, opts)
      } catch (err) {
        // NEVER fall back to the un-intersected /reap: that path relays the
        // client's keys WITHOUT central's re-derive+intersect second guard, so a
        // stale proposal could widen the delete. /reap-approve ships in 0.1.137;
        // if it isn't live, refuse rather than weaken the guarantee.
        if (/\b404\b|not found/i.test(err.message || '')) {
          throw new Error('reap-approve is not available on this central/worker (needs 0.1.137+). Refusing to fall back to the un-guarded reaper.')
        }
        throw err
      }
      const freedGb = ((res && res.freed_bytes) || 0) / 1e9
      const failed = ((res && res.results) || []).filter(r => !r.ok)
      let msg = `Freed ${freedGb.toFixed(1)} GB from ${worker.name}.`
      if (res && res.note) msg += `\n\n${res.note}`
      if (failed.length) msg += `\n\nSkipped ${failed.length}:\n` + failed.map(r => `  • ${r.model_key} — ${r.reason}`).join('\n')
      alert(msg)
      load()
    } catch (err) { alert(`Eviction failed: ${err.message}`) }
  }, [load])

  const remove = useCallback(async (worker) => {
    if (!confirm(`Remove worker ${worker.name} from the pool?`)) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}`, { method: 'DELETE' })
      load()
    } catch (err) { alert(`Remove failed: ${err.message}`) }
  }, [load])

  // Admission gate (the persistent switch, unlike Remove which a heartbeat undoes).
  const admit = useCallback(async (worker) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/admit`, { method: 'POST' })
      load()
    } catch (err) { alert(`Admit failed: ${err.message}`) }
  }, [load])

  const block = useCallback(async (worker) => {
    if (!confirm(`Block ${worker.name}? It stops serving and its agent exits on next contact.`)) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/block`, { method: 'POST' })
      load()
    } catch (err) { alert(`Block failed: ${err.message}`) }
  }, [load])

  // Dedicate this worker to a pool (reserved for an app's traffic). A worker that
  // declares WORKER_POOL on its box re-asserts that on its next heartbeat.
  const setPool = useCallback(async (worker) => {
    const next = window.prompt(
      `Dedicated pool for "${worker.name}" (blank = general). Requests tagged for this pool route here; general traffic won't.`,
      worker.pool || '')
    if (next === null) return
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/pool`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pool: next.trim() }),
      })
      load()
    } catch (err) { alert(`Set pool failed: ${err.message}`) }
  }, [load])

  // Console-managed serving config (daylight item 3). Persists in the AGENT's
  // settings file (source of truth over env drop-ins); applies via a short
  // agent re-exec — the row blips offline→online, same worker id.
  const setConfig = useCallback(async (worker) => {
    const cur = worker.config?.slot_count
    const next = window.prompt(
      `Slot count for "${worker.name}" (0–16; currently ${cur ?? '?'}${worker.config?.slot_count_source ? ` from ${worker.config.slot_count_source}` : ''}).\n` +
      '0 = no slots (in-process only). Applies via a ~5s agent restart.',
      cur != null ? String(cur) : '')
    if (next === null || next.trim() === '') return
    const n = Number(next)
    if (!Number.isInteger(n) || n < 0 || n > 16) { alert('slot count must be an integer 0–16'); return }
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot_count: n }),
      })
      if (r && r.ok === false) alert(`Config failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'slot_count', value: n })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Config failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Per-model residency (v3 final): deep-merged into the agent's settings.
  // on-demand IS the default and is represented by NO stored override, so
  // anything except 'static' normalizes to null (clears the override).
  const setResidency = useCallback(async (worker, modelKey, mode) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ residency: { [modelKey]: mode === 'static' ? 'static' : null } }),
      })
      if (r && r.ok === false) alert(`Residency failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'residency', model: modelKey, value: mode })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Residency failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk residency (todo t12): set the RESIDENCY tier of a SELECTED set of a
  // worker's models in ONE settings-write. Central relays a single /ops/config
  // with the whole residency map (same code path + same one agent re-exec as
  // pin-all) — never one restart per model. Confirms first, surfaces per-model
  // results + counts exactly like pinAll. Residency ONLY — never touches 📌 pin.
  const setResidencyMany = useCallback(async (worker, modelKeys, mode) => {
    const keys = (modelKeys || []).filter(k => (worker.models || []).includes(k))
    if (keys.length === 0) { alert('No models selected.'); return }
    const label = mode === 'static' ? '🔒 static' : '⏲ on-demand'
    const desc = mode === 'static'
      ? 'Static is a locked seat: the model is kept on this worker and never evicted (the only tier that keeps files on disk). '
      : 'On-demand is the default: the model loads on call and yields its seat when another model needs it. '
    if (!confirm(
      `Set residency → ${label} for ${keys.length} selected model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      desc + 'The worker agent restarts once (~5s) to apply all of them together.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/residency-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_keys: keys, mode }),
      })
      if (r && r.ok === false && !r.results) {
        // A pre-relay reject (bad body/mode) — no per-model map to show.
        alert(`Residency change failed: ${r.error?.message || 'unknown'}`)
        return
      }
      if (r && r.restarting) {
        markApplying(worker.id, { kind: 'residency_all', models: keys,
                                  value: mode === 'static' ? 'static' : 'on-demand' })
      }
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Set residency → ${label}: ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Set residency failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk ALLOC (todo t15; engine gate narrowed t26; per-model % t48): set the
  // GPU allocation (spill contract) of a SELECTED set of a worker's models in
  // ONE request. Unlike residency/pin this is a central REGISTRY write
  // (assign_model, same as the single ⚙ editor) — it does NOT restart the
  // agent; the spill applies on next load. `spill` is what AllocControl
  // produces (autofit {} / max GPU {n_gpu_layers:-1} / CPU only
  // {n_gpu_layers:"off"} / explicit budgets) — the ONE contract to broadcast
  // to every selected model. `perModel` (t48, optional) is AllocControl's
  // PER-MODEL resolution of a PERCENT VRAM/RAM budget — {model_key: spill},
  // each entry already resolved against THAT model's own size — sent instead
  // of `spill` whenever it's non-empty (a percent field was actually in play);
  // otherwise every member wants the identical flat contract and `spill` alone
  // is correct, same as before t48.
  // Autofit / Max GPU / CPU only are engine-agnostic PLACEMENT INTENT and apply
  // to EVERYONE; only the explicit-budget class is GGUF-only and splits the
  // selection. Confirms first; surfaces per-model results like setResidencyMany.
  const setAllocMany = useCallback(async (worker, modelKeys, spill, perModel) => {
    const keys = (modelKeys || []).filter(k => (worker.models || []).includes(k))
    if (keys.length === 0) { alert('No models selected.'); return }
    const perModelActive = !!(perModel && Object.keys(perModel).length > 0)
    // Parity label (2026-07-24 bulk parity): BulkAllocControl posts the new
    // vocabulary — {alloc_mode:<name>} for the four zero-knob modes, {} for
    // "Default (derived)" (clear-to-derivation), and a perModel map of
    // {alloc_mode:'explicit',…} for the bulk explicit split. spillLabel() only
    // reads the legacy n_gpu_layers/budget keys, so derive the human label from
    // whichever the caller actually sent.
    const flatMode = spill && spill.alloc_mode ? String(spill.alloc_mode) : null
    const isClear = !perModelActive && (!spill || Object.keys(spill).length === 0)
    const label = perModelActive ? 'Explicit'
      : isClear ? 'Default (derived)'
      : flatMode ? allocModeLabel(flatMode)
      : spillLabel(spill)
    // Engine split (t26 + 2026-07-24): the GGUF-only class is now the
    // explicit-budget spills AND the flat modes max-ram / explicit (which ride
    // alloc_mode). When the chosen alloc is GGUF-only, pre-state the split so the
    // operator sees which members are touched before apply. (The backend
    // re-enforces this — the UI number is a courtesy, not the gate.) The KEY set
    // is identical whether broadcasting `spill` or fanning out `perModel`.
    const fwOf = (k) => String(findCatalogRow(models, k)?.framework || '').toLowerCase()
    const isGgufFw = (fw) => fw === 'gguf' || fw === 'llama_cpp'
    const ggufKeys = keys.filter(k => isGgufFw(fwOf(k)))
    const modeGgufOnly = flatMode === 'max-ram' || flatMode === 'explicit'
      || (perModelActive && Object.values(perModel).some(s => s && s.alloc_mode === 'explicit'))
    const gO = modeGgufOnly || allocIsGgufOnly(spill)
    let confirmMsg = isClear
      ? `Revert ${keys.length} selected model${keys.length === 1 ? '' : 's'} to the DERIVED default on ${worker.name}?\n\n` +
        'Clears any pinned allocation contract so each model tracks its derived default again (and improves with it as measured values land). Applies on next load — no agent restart.'
      : `Set GPU allocation → ${label} for ${keys.length} selected model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
        'The allocation is the model\'s resource contract on this worker; it applies the next time each model loads (no agent restart).'
    if (gO) {
      const m = keys.length - ggufKeys.length
      confirmMsg =
        `Set GPU allocation → ${label} (GGUF-only) on ${worker.name}?\n\n` +
        `Applies to ${ggufKeys.length} GGUF model${ggufKeys.length === 1 ? '' : 's'}; ` +
        `${m} transformers/comfy model${m === 1 ? '' : 's'} skipped with a reason (${label} is a GGUF concept; Default / Max GPU / GPU only / RAM only would apply to all).\n\n` +
        'Applies on next load — no agent restart.'
      if (ggufKeys.length === 0) {
        alert(`None of the ${keys.length} selected models are GGUF — "${label}" is GGUF-only and would touch nothing. Use Default, Max GPU, GPU only, or RAM only to affect non-GGUF models.`)
        return
      }
    }
    if (perModelActive) {
      confirmMsg += '\n\n(An explicit % split is in play — each model resolves it ' +
        'against its OWN size, so the actual GiB numbers differ per model.)'
    }
    if (!confirm(confirmMsg)) return
    try {
      // t48: send the per-model map when the percent resolution actually
      // produced one; otherwise the single shared `spill` exactly as before.
      const body = perModelActive
        ? { model_keys: keys, spills: Object.fromEntries(keys.map(k => [k, perModel[k] || {}])) }
        : { model_keys: keys, spill: spill || {} }
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/alloc-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (r && r.ok === false && !r.results) {
        alert(`Allocation change failed: ${r.error?.message || 'unknown'}`)
        return
      }
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0, skN = r?.counts?.skipped ?? 0
      // Surface skipped-with-reason (engine mismatch) + any hard errors together.
      const notable = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
      if (notable.length > 0) {
        alert(`Set alloc → ${label}: ${okN} applied` +
          (skN ? `, ${skN} skipped` : '') + (errN ? `, ${errN} failed` : '') +
          ` on ${worker.name}.\n\n` +
          notable.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) { alert(`Set alloc failed: ${err.message}`) }
  }, [load, models])

  // Tiers v2 FILES axis: pin toggles ride the same settings channel.
  const togglePin = useCallback(async (worker, modelKey, pin) => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/config`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: { [modelKey]: pin ? true : null } }),
      })
      if (r && r.ok === false) alert(`Pin failed: ${r.error?.message || 'unknown'}`)
      else if (r && r.restarting) markApplying(worker.id, { kind: 'pinned', model: modelKey, value: pin })
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Pin failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Bulk pin: 📌 pin EVERY model designated to this worker in one settings-write
  // (central relays a single /ops/config with the full pinned map — same code
  // path as the per-model pin). Sticky, so we CONFIRM first: each pinned model
  // then blocks unassign until Unpin all. Surfaces per-model results + counts.
  const pinAll = useCallback(async (worker) => {
    const keys = worker.models || []
    if (keys.length === 0) { alert(`${worker.name} has no assigned models to pin.`); return }
    const unpinned = keys.filter(k => !worker.config?.pinned?.[k])
    if (unpinned.length === 0) {
      alert(`All ${keys.length} model${keys.length === 1 ? '' : 's'} on ${worker.name} are already pinned.`)
      return
    }
    if (!confirm(
      `📌 Pin all ${keys.length} model${keys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      'Pinning is PERMANENT attribution: each pinned model then refuses unassign ' +
      '("unpin first") until you Unpin all. The worker agent restarts (~5s) to apply.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/pin-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      })
      if (r && r.restarting) markApplying(worker.id, { kind: 'pin_all', models: keys, value: true })
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Pinned ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Pin all failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  // Unpin all — the undo for Pin all. Unpins every model on the worker in one
  // /ops/config write (same relay); afterward the models can be unassigned.
  const unpinAll = useCallback(async (worker) => {
    const keys = worker.models || []
    const pinnedKeys = keys.filter(k => worker.config?.pinned?.[k])
    if (pinnedKeys.length === 0) { alert(`No pinned models on ${worker.name}.`); return }
    if (!confirm(
      `Unpin all ${pinnedKeys.length} pinned model${pinnedKeys.length === 1 ? '' : 's'} on ${worker.name}?\n\n` +
      'This is the undo for Pin all — the models can be unassigned again afterward. ' +
      'The worker agent restarts (~5s) to apply.')) return
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/unpin-all`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      })
      if (r && r.restarting) markApplying(worker.id, { kind: 'pin_all', models: keys, value: false })
      const okN = r?.counts?.ok ?? 0, errN = r?.counts?.error ?? 0
      if (errN > 0) {
        const failed = Object.entries(r.results || {}).filter(([, v]) => v !== 'ok')
        alert(`Unpinned ${okN}/${okN + errN} on ${worker.name}.\n\nFailed:\n` +
          failed.map(([mk, v]) => `  • ${mk} — ${v}`).join('\n'))
      }
      load()
    } catch (err) {
      alert(applying[worker.id]
        ? 'The agent is restarting to apply the previous change — retry in a few seconds.'
        : `Unpin all failed: ${err.message}`)
    }
  }, [load, markApplying, applying])

  const setLimits = useCallback(async (worker, limits) => {
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/limits`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limits }),
      })
      load()
    } catch (err) { alert(`Set limits failed: ${err.message}`) }
  }, [load])

  // Drop every offline worker (e.g. stale entries left by an old agent).
  const pruneOffline = useCallback(async () => {
    const stale = workers.filter(w => w.status !== 'online')
    if (!stale.length) return
    if (!confirm(`Remove ${stale.length} offline worker(s) from the pool?`)) return
    await Promise.all(stale.map(w =>
      fetchJson(`/api/llm/workers/${encodeURIComponent(w.id)}`, { method: 'DELETE' }).catch(() => {})
    ))
    load()
  }, [workers, load])

  const onlineCount = workers.filter(w => w.status === 'online').length
  const offlineCount = workers.length - onlineCount
  const pendingCount = workers.filter(w => w.admission === 'pending').length

  // Fleet rollup: VRAM used/total and models loaded across ONLINE workers.
  const fleet = useMemo(() => {
    let total = 0, free = 0, serving = 0, hasVram = false
    for (const w of workers) {
      if (w.status !== 'online') continue
      serving += (w.loaded_models || []).length
      for (const g of (w.gpus || [])) {
        if (g.memory_total != null) { total += g.memory_total; hasVram = true }
        if (g.memory_free != null) free += g.memory_free
      }
    }
    return { total, free, used: Math.max(total - free, 0), serving, hasVram }
  }, [workers])

  // The setup disclosure is forced open while the pool is empty (the install
  // command is the only first-worker path) and otherwise follows the user's
  // toggle — defaulting closed so the live worker list leads.
  const isEmpty = workers.length === 0

  // One-line installer served by THIS central node. The command must be
  // fetchable FROM THE WORKER BOX, but by default all we have is the browser's
  // origin — browsing the console on central itself (http://localhost:7002)
  // would render a command that points a remote box at its own loopback. When
  // the origin is loopback, ask central for its LAN address and render that
  // instead (the curl host is also what the script bakes as CENTRAL, so fixing
  // the display fixes the install). If central can't tell, keep the command
  // but warn.
  const browserOrigin = resolveApiOrigin()
  const originIsLoopback = useMemo(() => {
    try {
      const h = new URL(browserOrigin).hostname.replace(/^\[|\]$/g, '')
      return h === 'localhost' || h === '::1' || h === '0.0.0.0' || h.startsWith('127.')
    } catch { return false }
  }, [browserOrigin])
  const [lanOrigin, setLanOrigin] = useState(null)
  useEffect(() => {
    if (!originIsLoopback) return
    fetchJson('/api/llm/workers/central-address')
      .then(d => { if (d && d.lan_ip && d.base_url) setLanOrigin(d.base_url) })
      .catch(() => {})   // fall through to the loopback warning
  }, [originIsLoopback])
  const installCmd = `curl -fsSL ${(originIsLoopback && lanOrigin) || browserOrigin}/api/llm/workers/install.sh | bash`

  // models picked in the "Add manually" form (form.models stays a comma-joined string)
  const selectedModels = form.models.split(',').map(s => s.trim()).filter(Boolean)
  const addModel = (key) => setForm(f => {
    const cur = f.models.split(',').map(s => s.trim()).filter(Boolean)
    return cur.includes(key) ? f : { ...f, models: [...cur, key].join(',') }
  })
  const removeModel = (key) => setForm(f => ({
    ...f,
    models: f.models.split(',').map(s => s.trim()).filter(Boolean).filter(k => k !== key).join(','),
  }))

  // Cross-worker allocation map (model_key -> [{id,name}]) — powers the load
  // table's anti-duplicate lockout: a model already on another worker is shown
  // "on <worker>" and can't be re-allocated unless the breaker is flipped.
  const allocationMap = useMemo(() => {
    const map = {}
    for (const w of workers) {
      for (const mk of (w.models || [])) {
        (map[mk] ??= []).push({ id: w.id, name: w.name })
      }
    }
    return map
  }, [workers])

  return (
    <div className="workers-panel">
      <div className={`wp-bar${embedded ? ' wp-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="wp-title">🖧 GPU Workers</span>
        <span className="wp-count">{onlineCount} online / {workers.length} total</span>
        {fleet.hasVram && (
          <span className="wp-fleet" title="VRAM used / total across online workers">
            VRAM {fmtBytes(fleet.used)} / {fmtBytes(fleet.total)} · {fmtBytes(fleet.free)} free
          </span>
        )}
        {fleet.serving > 0 && (
          <span className="wp-fleet-loaded" title="models serving (resident in VRAM) across the fleet">
            🔥 {fleet.serving} serving
          </span>
        )}
        {pendingCount > 0 && (
          <span className="wp-pending-chip" title="Workers awaiting your approval — they don't serve until admitted">
            ⏳ {pendingCount} pending
          </span>
        )}
        {error && <span className="wp-err" title={error}>registry error<FixDoc doc="registry-error" /></span>}
        {offlineCount > 0 && (
          <button
            className="wp-prune"
            title="Remove all offline workers"
            onClick={e => { e.stopPropagation(); pruneOffline() }}
          >clear {offlineCount} offline</button>
        )}
        {!embedded && <span className="wp-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="wp-body">
          {/* t22 (operator): the page reads live workers first, then group
              actions, then the "Add & manage workers" tooling — so the per-worker
              cards + central footer come first, GroupAssignPanel next, and the
              wp-setup section LAST (moved below, after the central footer). */}
          {workers.length === 0 && <div className="wp-empty">No workers have joined the pool yet.</div>}
          {/* Tab strip: one worker's card at a time. `selected` falls back to the
              first worker so a stale/removed id from sessionStorage can never
              leave the panel blank. Each tab carries the worker's live status dot
              and its assignment count, so the strip stays informative without
              expanding the cards. */}
          {workers.length > 1 && (
            <div className="wp-tabs" role="tablist">
              {workers.map(w => (
                <button
                  key={w.id}
                  role="tab"
                  aria-selected={w.id === selectedWorkerId}
                  className={`wp-tab wp-tab-${w.status || 'unknown'}${w.id === selectedWorkerId ? ' wp-tab-on' : ''}`}
                  onClick={() => setActiveWorker(w.id)}
                  title={`${w.name || w.id} — ${w.status || 'unknown'}, ${(w.models || []).length} assigned`}
                >
                  <span className="wp-tab-dot" />
                  <span className="wp-tab-name">{w.name || w.id.slice(0, 8)}</span>
                  <span className="wp-tab-count">{(w.models || []).length}</span>
                </button>
              ))}
            </div>
          )}
          {shownWorkers.map(w => (
            <WorkerRow
              key={w.id}
              worker={w}
              models={models}
              allocation={allocationMap}
              onAssign={assign}
              onRefresh={load}
              onLoad={loadModel}
              onUnassign={unassign}
              onRemove={remove}
              onFree={freeModel}
              onFreeAll={freeAll}
              onFreeRam={freeRam}
              onRestart={restart}
              restarting={!!restarting[w.id]}
              onUpdate={update}
              updating={!!updating[w.id]}
              onAdmit={admit}
              onBlock={block}
              onSetPool={setPool}
              onSetLimits={setLimits}
              onSetConfig={setConfig}
              onSetResidency={setResidency}
              onSetResidencyMany={setResidencyMany}
              onSetAllocMany={setAllocMany}
              onTogglePin={togglePin}
              onPinAll={pinAll}
              onUnpinAll={unpinAll}
              onReap={reap}
              onApproveEvictions={approveEvictions}
              onEvict={evictModel}
              onAllocateMany={allocateMany}
              applying={!!applying[w.id]}
              blockedKeys={blockedKeys}
              onToggleBlock={toggleBlock}
            />
          ))}

          {/* Central itself, represented as a worker like the others — its
              compute is the local slot pool (+ in-process fallback), so it
              renders with the same row anatomy: identity, resources, and its
              "serving" list (= the slots). Rendered LAST, below every remote
              worker row: it's the console's own host, not fleet capacity —
              secondary context, not the headline. */}
          {central && (
            <div className="wp-central-footer">
              <div className="wp-worker wp-online wp-central">
                <div className="wp-worker-head">
                  <span className="wp-dot" />
                  <span className="wp-name">central</span>
                  <span className="wp-status">online</span>
                  <span className="wp-adm wp-adm-pill-approved" title="The console's own server">this host</span>
                  <span className="wp-url" title="Serves via its local slot pool; also the fallback when no worker can take a model">
                    local slot pool{central.enabled === false ? ' (disabled)' : ''}
                  </span>
                </div>
                {central.resources && (
                  <div className="wp-ram" title="Central's own RAM — what the slot preflight budgets against (minus reserves)">
                    🧠 RAM {fmtBytes(central.resources.available_bytes ?? central.resources.free_bytes)} free
                    {central.resources.total_bytes != null && <> of {fmtBytes(central.resources.total_bytes)}</>}
                    {central.resources.cpu_count != null && <> · {central.resources.cpu_count} cores</>}
                  </div>
                )}
                <div className="wp-models">
                  <span className="wp-models-label">Slots:</span>
                  {(central.slots || []).length === 0 && <span className="wp-none">— no slots —</span>}
                  {(central.slots || []).map(s => {
                    const state = s.model_key ? (s.healthy ? 'serving' : 'warming') : 'idle'
                    return (
                      <span key={s.slot_id} className={`wp-model wp-st-${state}`}>
                        <span className={`wp-state-pill wp-pill-${state}`}>
                          {s.model_key ? (s.healthy ? '🔥 serving' : '⏳ warming') : `○ slot ${s.slot_id}`}
                        </span>
                        <span className="wp-model-name">{s.model_key || '— idle —'}</span>
                        <span className="wp-model-facts">
                          {s.rss_bytes ? `${fmtBytes(s.rss_bytes)} RSS` : ''}
                          {s.n_gpu_layers != null ? ` · ngl ${s.n_gpu_layers}` : ''}
                          {s.ctx != null ? ` · ctx ${s.ctx}` : ''}
                        </span>
                        {s.model_key && (
                          <button className="wp-free" title="Unload this slot (frees its RAM/VRAM)"
                                  onClick={() => centralUnload(s._control)}>⏏</button>
                        )}
                      </span>
                    )
                  })}
                </div>
                {/* Load-a-model — same affordance as every other worker row. This
                    made the separate central-only "Model Slots" panel redundant
                    (retired); central presents as just another worker. */}
                {!showCentralLoad ? (
                  <button className="wp-load-toggle" onClick={() => setShowCentralLoad(true)}
                          title="Load a model into a free central slot">
                    ＋ load a model
                  </button>
                ) : (
                  <div className="wp-assign-row">
                    <ModelPicker
                      models={models}
                      value={centralPick}
                      onPick={setCentralPick}
                      placeholder="Load a model into a free central slot…"
                      autoFocus
                    />
                    <button className="wp-alloc-apply" disabled={!centralPick || centralBusy}
                            onClick={() => centralLoad(centralPick)}>
                      {centralBusy ? '…' : '+ Load'}
                    </button>
                    <button className="wp-load-cancel" title="Cancel"
                            onClick={() => { setShowCentralLoad(false); setCentralPick('') }}>×</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Group actions — below the live workers (t22). */}
          {workers.length > 0 && (
            <GroupAssignPanel models={models} workers={workers} onGroupAssign={groupAssign} />
          )}

          {/* Add & manage workers — install command, enrollment tokens, manual
              add. Moved BELOW the workers + group actions (t22). Stays collapsed
              by default (setupOpen defaults false); it is force-OPEN only when the
              fleet is empty (isEmpty), since then there are no workers above it and
              the install command is the only way to add the first box. */}
          <details className="wp-setup" open={setupOpen || isEmpty}
                   onToggle={e => { if (!isEmpty) setSetupOpen(e.currentTarget.open) }}>
            <summary className="wp-setup-summary">
              ＋ Add &amp; manage workers — install command, enrollment tokens, manual add
            </summary>
          <div className="wp-install">
            <span className="wp-install-label">Add a GPU box — run on the worker:</span>
            <code className="wp-install-cmd" title="Click to copy"
                  onClick={() => navigator.clipboard?.writeText(installCmd)}>
              {installCmd}
            </code>
            <span className="wp-install-note">
              Central fills in its own address and this box's reachable IP — no per-worker config.
            </span>
            {originIsLoopback && !lanOrigin && (
              <span className="wp-install-warn">
                ⚠ You're browsing central at {browserOrigin} — a worker box can't reach that
                address. Replace it with one the worker can reach (e.g. central's LAN IP).
                <FixDoc doc="worker-join" />
              </span>
            )}
          </div>

          <details className="wp-tokens">
            <summary>Enrollment tokens ({tokens.filter(t => !t.revoked).length} active) — admit machines to the fleet</summary>
            <div className="wp-tokens-body">
              <button className="wp-token-issue" onClick={issueToken}>+ Issue enrollment token</button>
              {newToken && (
                <div className="wp-token-new">
                  <strong>Copy this token now — it is shown only once:</strong>
                  <code className="wp-token-secret" title="Click to copy"
                        onClick={() => navigator.clipboard?.writeText(newToken.token)}>
                    {newToken.token}
                  </code>
                  <span className="wp-install-note">Run on the worker (bakes central + token into its unit):</span>
                  <code className="wp-install-cmd" title="Click to copy"
                        onClick={() => navigator.clipboard?.writeText(
                          `WORKER_ENROLL_TOKEN=${newToken.token} ${installCmd}`)}>
                    {`WORKER_ENROLL_TOKEN=${newToken.token} ${installCmd}`}
                  </code>
                  <button className="wp-token-dismiss" onClick={() => setNewToken(null)}>Done</button>
                </div>
              )}
              {tokens.length === 0 && <div className="wp-none">No enrollment tokens issued.</div>}
              {tokens.map(t => (
                <div key={t.id} className={`wp-token-row${t.revoked ? ' wp-token-revoked' : ''}`}>
                  <span className="wp-token-label">{t.label || '(no label)'}</span>
                  <span className="wp-token-id" title="token id">{t.id}</span>
                  {t.revoked
                    ? <span className="wp-token-state">revoked</span>
                    : <button className="wp-token-revoke" title="Revoke — its workers are refused and stop"
                              onClick={() => revokeToken(t)}>revoke</button>}
                </div>
              ))}
            </div>
          </details>

          <details className="wp-manual">
            <summary>Add manually</summary>
            <form className="wp-register" onSubmit={register}>
              <input
                placeholder="worker name (e.g. gpu-box-1)"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              />
              <input
                placeholder="worker URL (optional — central uses source IP)"
                value={form.url}
                onChange={e => setForm(f => ({ ...f, url: e.target.value }))}
              />
              <div className="wp-models-pick">
                <ModelPicker
                  models={models.filter(m => !selectedModels.includes(m.model_key ?? m.key))}
                  value=""
                  onPick={addModel}
                  placeholder={models.length ? 'models to serve (optional)…' : 'no models registered yet'}
                  disabled={!models.length}
                />
                {selectedModels.length > 0 && (
                  <div className="wp-models-chips">
                    {selectedModels.map(k => (
                      <span key={k} className="wp-models-chip">
                        {findCatalogRow(models, k)?.name || k}
                        <button type="button" className="wp-chip-x" title="Remove"
                                onClick={() => removeModel(k)}>×</button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <button type="submit" disabled={busy}>+ Add worker</button>
            </form>
          </details>
          </details>
        </div>
      )}
    </div>
  )
}
