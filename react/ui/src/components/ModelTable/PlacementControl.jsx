import { useEffect, useState, useCallback } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { getServing, invalidateServing, getShardFlag, primeShardFlag } from './servingCache'
import { LiveWorkerChip } from '../ModelLiveState/ModelLiveState'
import { archiveText } from './archiveMark'

// k56 — per-model PLACEMENT: the ordered worker preference.
//
// Designation used to be one hard worker binding; it is now an ORDERED
// candidate list. Resolution tries the workers in the stated order and takes
// the first whose admission accepts, and a model carrying a list never lands
// off it. A single entry is the degenerate case and behaves exactly as one
// designation always did — which is also why an EMPTY list is a real state
// (no preference: routing ranks by residency/capability as before), not a
// half-filled form. MODEL-scoped, so it persists through the same
// /api/llm/serving/<key> overrides call the sibling ServingControl uses —
// self-contained, no wiring through App state.
//
// d1136 (2026-09-29): the per-model / per-worker "polite load" controls
// (`no_evict`, `no_evict_by_worker`) were REMOVED from the console — no
// model-specific backend behaviour. Whether a model stays seated is the
// worker-side RESIDENCY policy (static | on-demand, ResidencyMenu), not a
// per-model eviction flag.
//
// Ranking is ↑/↓ buttons rather than drag: every other control in this detail
// panel is a plain button/select, and a drag affordance nobody else here has
// would read as a different kind of thing.

// LIVE STATE per allocated worker — the shared relay (ModelLiveState). Module
// level on purpose: declared inside PlacementControl it was a NEW component type
// on every render, so each chip unmounted + remounted every poll.
function LiveState({ modelKey, name, byName }) {
  const w = byName(name)
  return <LiveWorkerChip modelKey={modelKey} worker={w?.name || name} />
}

// `archived` — the catalog row's archive mark ({marked, at, by, reason}) or
// null. A marked model's placement is read-only here (central 409s the writes);
// every disabled control carries the recorded mark as its title.
export default function PlacementControl({ modelKey, workers = [], archived = null }) {
  const [prefs, setPrefs] = useState(null)      // null until the GET lands
  const [strict, setStrict] = useState(false)   // k-dist: hard-fence this model
  const [saved, setSaved] = useState({ prefs: [], strict: false })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [shardOn, setShardOn] = useState(null)   // multi-GPU shard-eligible (persisted setting)
  const [shardBusy, setShardBusy] = useState(false)

  // One reader for both the GET and the POST echo, so what a save leaves on
  // screen is exactly what a reload would show.
  const adopt = (ov) => {
    const list = Array.isArray(ov.worker_prefs) ? ov.worker_prefs : []
    setPrefs(list); setStrict(!!ov.strict)
    setSaved({ prefs: list, strict: !!ov.strict })
  }

  const load = useCallback(async () => {
    setMsg('')
    try {
      const d = await getServing(modelKey)   // shared, deduped, 20 s cache (see servingCache.js)
      adopt(d.override || {})
    } catch (e) {
      setPrefs([])
      setMsg(String(e.message || e))
    }
  }, [modelKey])

  useEffect(() => { load() }, [load])

  // Multi-GPU shard-eligibility is a persisted per-model flag (settings_store
  // ns 'shard_models'); toggling it opts the model in/out of sharding with NO
  // restart. A truthy flag auto-sizes from the model's own bytes on the server.
  // Fetched ONCE per model through the shared cache (servingCache.getShardFlag);
  // a toggle below primes it. Never on a render/poll.
  useEffect(() => {
    let alive = true
    getShardFlag(modelKey)
      .then(on => { if (alive) setShardOn(on) })
      .catch(e => { if (alive) { setShardOn(false); setMsg(`shard flag read failed: ${e.message || e}`) } })
    return () => { alive = false }
  }, [modelKey])

  // A listed name resolves against the live fleet by name OR id — the same
  // tolerance the backend's _pref_index applies, so what the operator sees
  // matched here is what routing matches there.
  const byName = (n) => workers.find(w => w.name === n || w.id === n)
  const archText = archived ? archiveText(archived) : ''

  const save = async () => {
    if (archived) { setMsg(`✗ ${archText}`); return }
    setBusy(true); setMsg('saving…')
    try {
      // The list is an ORDER OVER DESIGNATIONS, not a second designation store:
      // the SoT stays worker["models"] (one set per worker, which has nowhere to
      // put a cross-worker order), so listing a worker DESIGNATES the model to
      // it via the existing /assign call. Removing one does NOT unassign —
      // dropping a box from an ORDER is not a statement that the model may no
      // longer run there, and unassign has its own guards (a pinned designation
      // refuses with a 409). Use the worker row's own ✕ to undesignate.
      for (const name of prefs) {
        const w = byName(name)
        if (!w || (w.models || []).includes(modelKey)) continue
        const ra = await hugpyFetch(`/api/llm/workers/${encodeURIComponent(w.id)}/assign`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_key: modelKey }),
        })
        if (!ra.ok) {
          const da = await ra.json().catch(() => ({}))
          throw new Error(`assign to ${w.name || w.id}: ${da.error || `HTTP ${ra.status}`}`)
        }
      }
      const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Both keys go every time: an empty list and a false flag are how the
        // operator CLEARS them, and omitting a key would silently keep the old
        // value (the overrides layer merges).
        body: JSON.stringify({ worker_prefs: prefs, strict }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
      invalidateServing(modelKey)
      adopt(d.override || {})
      setMsg('✓ saved')
    } catch (e) {
      setMsg(`✗ ${e.message || e}`)
    } finally {
      setBusy(false)
    }
  }

  if (prefs === null) return <div className="mt-place">loading placement…</div>

  const move = (i, delta) => {
    const next = prefs.slice()
    const j = i + delta
    if (j < 0 || j >= next.length) return
    ;[next[i], next[j]] = [next[j], next[i]]
    setPrefs(next)
  }
  const drop = (i) => setPrefs(prefs.filter((_, k) => k !== i))
  const add = (name) => { if (name && !prefs.includes(name)) setPrefs([...prefs, name]) }

  const unlisted = workers.filter(w => !prefs.includes(w.name) && !prefs.includes(w.id))
  const dirty = (JSON.stringify(prefs) !== JSON.stringify(saved.prefs)
                 || strict !== saved.strict)

  // EFFECTIVE PLACEMENT — which candidate actually served last. Read off the
  // one ledger central already keeps (model_call_stats, stamped on the pick),
  // so this reports what happened rather than what was configured.
  let servedBy = null
  let servedAt = 0
  for (const w of workers) {
    const row = (w.model_call_stats || {})[modelKey]
    const at = row && row.last_call
    if (at && at > servedAt) { servedAt = at; servedBy = w.name || w.id }
  }
  // …and, when a load was refused, WHY — the worker's own reason string.
  const refusals = workers
    .map(w => [w, ((w.load_reports || {})[modelKey] || {})])
    .filter(([, rep]) => rep && rep.ok === false && rep.error)

  return (
    <div className="mt-place">
      <div className="mt-place-prefs">
        {prefs.length === 0 && (
          <span className="mt-place-none">
            No preference — routing ranks by designation, residency then capability.
          </span>
        )}
        {prefs.map((name, i) => {
          const w = byName(name)
          const free = w?.gpus?.[0]?.memory_free
          return (
            <span className="mt-place-chip" key={name}
                  title={w ? `${name} — ${w.status || 'unknown'}`
                           : `${name} is not a registered worker right now; it stays on the list and is skipped until it comes back`}>
              <span className="mt-place-rank">{i + 1}.</span>
              <span className={w && w.status === 'online' ? '' : 'mt-place-off'}>{name}</span>
              <LiveState modelKey={modelKey} name={name} byName={byName} />
              {free != null && <span className="mt-place-free">{(free / 2 ** 30).toFixed(1)} GiB free</span>}
              <button disabled={i === 0} title="Try this worker earlier"
                      onClick={() => move(i, -1)}>↑</button>
              <button disabled={i === prefs.length - 1} title="Try this worker later"
                      onClick={() => move(i, 1)}>↓</button>
              <button title="Remove from the preference list"
                      onClick={() => drop(i)}>✕</button>
            </span>
          )
        })}
      </div>

      <div className="mt-place-actions">
        <select value="" disabled={!!archived || !unlisted.length}
                title={archived ? archText
                  : unlisted.length ? 'Append a worker to the preference list'
                                    : 'Every known worker is already on the list'}
                onChange={e => { add(e.target.value); e.target.value = '' }}>
          <option value="">＋ add worker…</option>
          {unlisted.map(w => (
            <option key={w.id} value={w.name}>
              {w.name}{w.status === 'online' ? '' : ' (offline)'}
            </option>
          ))}
        </select>

        <label className="mt-place-toggle"
               title="Strict fences unallocated feasible workers to this preference list. Explicit worker allocations remain the routing scope; this setting cannot make an allocated model unroutable just because its preference list is stale.">
          <input type="checkbox" disabled={!!archived} checked={strict}
                 title={archived ? archText : undefined}
                 onChange={e => setStrict(e.target.checked)} />
          strict: fence unallocated workers to this preference list
        </label>

        <label className="mt-place-toggle"
               title="Shard this model across multiple workers' GPUs (llama.cpp RPC) when it does not fit a single card. Evict-to-fit-aware; opts the model in with NO restart. Dense GGUF only — MoE models CPU-offload instead of using a remote GPU.">
          <input type="checkbox" disabled={!!archived || shardOn === null || shardBusy}
                 title={archived ? archText : undefined}
                 checked={!!shardOn}
                 onChange={async (e) => {
                   const on = e.target.checked
                   setShardBusy(true); setMsg(on ? 'enabling shard…' : 'disabling shard…')
                   try {
                     const path = `/settings/shard_models/${encodeURIComponent(modelKey)}`
                     const r = on
                       ? await hugpyFetch(path, { method: 'POST',
                           headers: { 'Content-Type': 'application/json' },
                           body: JSON.stringify({ value: true }) })
                       : await hugpyFetch(path, { method: 'DELETE' })
                     if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
                     primeShardFlag(modelKey, on)
                     setShardOn(on); setMsg(on ? '✓ shard-eligible' : '✓ shard off')
                   } catch (err) { setMsg(`✗ shard: ${err.message || err}`) }
                   finally { setShardBusy(false) }
                 }} />
          shard across GPUs (multi-GPU){shardOn === null ? ' …' : ''}
        </label>

        <button disabled={!!archived || busy || !dirty} onClick={save}
                title={archived ? archText : undefined}>Save placement</button>
        {msg && <span className="mt-serve-msg">{msg}</span>}
      </div>

      {servedBy && (
        <div className="mt-place-effective">
          last served by <strong>{servedBy}</strong>
          {prefs.length > 0 && !prefs.includes(servedBy) &&
            ' — not on the list (that call predates the current preference)'}
        </div>
      )}
      {refusals.map(([w, rep]) => (
        <div className="mt-place-refusal" key={w.id} title={rep.error}>
          ⚠ {w.name}: <span className="mt-place-refusal-text" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{String(rep.error)}</span>
        </div>
      ))}
    </div>
  )
}
