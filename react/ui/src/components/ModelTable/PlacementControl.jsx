import { useEffect, useState, useCallback } from 'react'
import { hugpyFetch } from '../../runtime/config'
import { getServing, invalidateServing } from './servingCache'

// k56 — per-model PLACEMENT: the ordered worker preference + the polite load.
//
// Designation used to be one hard worker binding; it is now an ORDERED
// candidate list. Resolution tries the workers in the stated order and takes
// the first whose admission accepts, and a model carrying a list never lands
// off it. A single entry is the degenerate case and behaves exactly as one
// designation always did — which is also why an EMPTY list is a real state
// (no preference: routing ranks by residency/capability as before), not a
// half-filled form.
//
// The polite toggle (`no_evict`) is the deliberate inverse of
// declare-need-then-evict: the load may spend only genuinely free headroom and
// never displaces a resident. Both are MODEL-scoped, so they persist through
// the same /api/llm/serving/<key> overrides call the sibling ServingControl
// uses — self-contained, no wiring through App state.
//
// k62 — politeness is INDIVIDUALIZED per (model × worker), rendered as the
// operator's sketch:
//
//     worker   | polite
//     worker0  | yes
//     worker1  | no
//
// …because contention is a property of a BOX, not of a model: flux2 is polite
// on ae's contended 3090 and keeps ordinary eviction rights on computron. Each
// row's ticker cycles default → yes → no → default, writing `no_evict_by_worker`;
// the compact master toggle below stays the all-workers default (`no_evict`),
// which every row without an explicit verdict follows.
//
// Ranking is ↑/↓ buttons rather than drag: every other control in this detail
// panel is a plain button/select, and a drag affordance nobody else here has
// would read as a different kind of thing.

export default function PlacementControl({ modelKey, workers = [] }) {
  const [prefs, setPrefs] = useState(null)      // null until the GET lands
  const [polite, setPolite] = useState(false)   // the ALL-WORKERS default
  const [byWorker, setByWorker] = useState({})  // per-worker verdicts (k62)
  const [saved, setSaved] = useState({ prefs: [], polite: false, byWorker: {} })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [shardOn, setShardOn] = useState(null)   // multi-GPU shard-eligible (persisted setting)
  const [shardBusy, setShardBusy] = useState(false)

  // One reader for both the GET and the POST echo, so what a save leaves on
  // screen is exactly what a reload would show.
  const adopt = (ov) => {
    const list = Array.isArray(ov.worker_prefs) ? ov.worker_prefs : []
    const map = (ov.no_evict_by_worker && typeof ov.no_evict_by_worker === 'object')
      ? { ...ov.no_evict_by_worker } : {}
    setPrefs(list); setPolite(!!ov.no_evict); setByWorker(map)
    setSaved({ prefs: list, polite: !!ov.no_evict, byWorker: map })
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
  useEffect(() => {
    let alive = true
    hugpyFetch(`/settings/shard_models/${encodeURIComponent(modelKey)}`)
      .then(r => r.json())
      .then(d => { if (alive) setShardOn(!!(d && d.value)) })
      .catch(() => { if (alive) setShardOn(false) })
    return () => { alive = false }
  }, [modelKey])

  // A listed name resolves against the live fleet by name OR id — the same
  // tolerance the backend's _pref_index applies, so what the operator sees
  // matched here is what routing matches there.
  const byName = (n) => workers.find(w => w.name === n || w.id === n)

  const save = async () => {
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
        await hugpyFetch(`/api/llm/workers/${encodeURIComponent(w.id)}/assign`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_key: modelKey }),
        })
      }
      const r = await hugpyFetch(`/api/llm/serving/${encodeURIComponent(modelKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // All three keys go every time: an empty list, a false flag and an empty
        // map are how the operator CLEARS them, and omitting a key would
        // silently keep the old value (the overrides layer merges).
        body: JSON.stringify({ worker_prefs: prefs, no_evict: polite,
                               no_evict_by_worker: byWorker }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
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
                 || polite !== saved.polite
                 || JSON.stringify(byWorker) !== JSON.stringify(saved.byWorker))

  // ── k62: the worker × polite grid ────────────────────────────────────────
  // Rows: every worker on the preference list (in its order), then every worker
  // this model is DESIGNATED to, then any name the map still carries — an
  // orphaned verdict (the box was renamed or dropped from the list) must stay
  // visible and clearable rather than deciding routing invisibly.
  const mapKey = (name) => Object.keys(byWorker).find(
    k => k.toLowerCase() === String(name).toLowerCase())
  const explicit = (name) => {            // true | false | null (= follow default)
    const k = mapKey(name)
    return k === undefined ? null : !!byWorker[k]
  }
  const effective = (name) => {
    const e = explicit(name)
    return e === null ? polite : e
  }
  // default → yes → no → default. Three states because "follows the all-workers
  // toggle" is a real answer, distinct from an explicit no that pins this box.
  const cyclePolite = (name) => {
    const e = explicit(name)
    const k = mapKey(name) || name
    const next = { ...byWorker }
    if (e === null) next[k] = true
    else if (e === true) next[k] = false
    else delete next[k]
    setByWorker(next)
  }

  const gridRows = []
  const seen = new Set()
  const pushRow = (name) => {
    const low = String(name).toLowerCase()
    if (!name || seen.has(low)) return
    seen.add(low)
    gridRows.push(name)
  }
  prefs.forEach(pushRow)
  workers.filter(w => (w.models || []).includes(modelKey))
         // Skip a box already listed under its OTHER spelling — the backend
         // matches id and name alike, so two rows would be two tickers for one
         // worker (and the second would silently lose).
         .filter(w => !seen.has(String(w.id || '').toLowerCase()))
         .forEach(w => pushRow(w.name || w.id))
  Object.keys(byWorker).forEach(k => {
    const w = byName(k)                       // same id-or-name tolerance again
    if (w && (seen.has(String(w.id || '').toLowerCase())
              || seen.has(String(w.name || '').toLowerCase()))) return
    pushRow(k)
  })

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
  // …and, when a load was refused, WHY. A polite refusal is the expected
  // outcome of this panel's own toggle, so it belongs beside it.
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

      {gridRows.length > 0 && (
        <table className="mt-place-grid">
          <thead>
            <tr><th>worker</th><th>polite</th></tr>
          </thead>
          <tbody>
            {gridRows.map(name => {
              const w = byName(name)
              const e = explicit(name)
              const on = effective(name)
              return (
                <tr key={name}>
                  <td className={w && w.status === 'online' ? '' : 'mt-place-off'}
                      title={w ? `${name} — ${w.status || 'unknown'}`
                               : `${name} is not a registered worker right now`}>
                    {name}
                  </td>
                  <td>
                    <button className={`mt-place-tick${on ? ' on' : ''}${e === null ? ' inherited' : ''}`}
                            onClick={() => cyclePolite(name)}
                            title={e === null
                              ? `follows the all-workers default (${polite ? 'yes' : 'no'}) — click to pin this worker`
                              : `pinned ${on ? 'yes' : 'no'} for this worker — click to ${on ? 'pin no' : 'follow the all-workers default'}`}>
                      {on ? 'yes' : 'no'}{e === null ? ' *' : ''}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      {gridRows.some(n => explicit(n) === null) && (
        <div className="mt-place-note">
          * follows the all-workers default below
        </div>
      )}

      <div className="mt-place-actions">
        <select value="" disabled={!unlisted.length}
                title={unlisted.length ? 'Append a worker to the preference list'
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
               title="Polite load, ALL WORKERS: on every worker without its own verdict above, this model may take only genuinely free VRAM and will NEVER evict a resident to make space. If no candidate has room, the load is refused honestly instead of displacing someone.">
          <input type="checkbox" checked={polite}
                 onChange={e => setPolite(e.target.checked)} />
          all workers: load only into free room (never evict)
        </label>

        <label className="mt-place-toggle"
               title="Shard this model across multiple workers' GPUs (llama.cpp RPC) when it does not fit a single card. Evict-to-fit-aware; opts the model in with NO restart. Dense GGUF only — MoE models CPU-offload instead of using a remote GPU.">
          <input type="checkbox" disabled={shardOn === null || shardBusy}
                 checked={!!shardOn}
                 onChange={async (e) => {
                   const on = e.target.checked
                   setShardBusy(true); setMsg(on ? 'enabling shard…' : 'disabling shard…')
                   try {
                     const path = `/settings/shard_models/${encodeURIComponent(modelKey)}`
                     if (on) {
                       await hugpyFetch(path, { method: 'POST',
                         headers: { 'Content-Type': 'application/json' },
                         body: JSON.stringify({ value: true }) })
                     } else {
                       await hugpyFetch(path, { method: 'DELETE' })
                     }
                     setShardOn(on); setMsg(on ? '✓ shard-eligible' : '✓ shard off')
                   } catch (err) { setMsg(`✗ shard: ${err.message || err}`) }
                   finally { setShardBusy(false) }
                 }} />
          shard across GPUs (multi-GPU){shardOn === null ? ' …' : ''}
        </label>

        <button disabled={busy || !dirty} onClick={save}>Save placement</button>
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
          ⚠ {w.name}: {String(rep.error).slice(0, 220)}
        </div>
      ))}
    </div>
  )
}
