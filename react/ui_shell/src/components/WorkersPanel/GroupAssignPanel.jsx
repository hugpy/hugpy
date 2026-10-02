import { useMemo, useState } from 'react'
import ModelPicker from '../ModelPicker/ModelPicker'
import useSessionState from '../../hooks/useSessionState'
import { fmtBytes } from './formatters'

// Group-assign: pick a model, tick a set of workers (with select-all), dedicate
// it to all of them at once. Workers that already hold the model are shown
// checked+locked ("already here") and never re-fired — that IS the no-double-
// booking guard. Offline/blocked workers can't take an assignment, so they're
// locked out too. Placed at the fleet level, above the per-worker rows.
export function GroupAssignPanel({ models, workers, onGroupAssign }) {
  const [open, setOpen] = useSessionState('hugpy.sess.wp.group.open', false)
  const [pick, setPick] = useState('')
  const [sel, setSel] = useState(() => new Set())
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState(null)

  const keyOf = (m) => m.model_key ?? m.key

  // Per-worker eligibility for the picked model.
  const rows = useMemo(() => (workers || []).map(w => {
    const already = pick ? (w.models || []).includes(pick) : false
    const offline = w.status === 'offline' || w.admission === 'blocked' || w.admission === 'pending'
    return {
      w,
      already,
      offline,
      eligible: !!pick && !already && !offline,
      reason: already ? '✓ already here' : offline ? `(${w.admission === 'blocked' ? 'blocked' : w.status === 'offline' ? 'offline' : 'not admitted'})` : '',
    }
  }), [workers, pick])

  const eligibleIds = useMemo(() => rows.filter(r => r.eligible).map(r => r.w.id), [rows])
  const allSelected = eligibleIds.length > 0 && eligibleIds.every(id => sel.has(id))
  const chosen = useMemo(() => rows.filter(r => r.eligible && sel.has(r.w.id)).map(r => r.w), [rows, sel])

  const toggle = (id) => setSel(prev => {
    const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n
  })
  const toggleAll = () => setSel(allSelected ? new Set() : new Set(eligibleIds))

  const dedicate = async () => {
    if (!pick || chosen.length === 0) return
    setBusy(true); setResults(null)
    try {
      const out = await onGroupAssign(pick, chosen, false)
      setResults(out)
      // Drop the successes from the selection so a re-run only retries failures.
      const failed = new Set(out.filter(r => !r.ok).map(r => r.id))
      setSel(prev => new Set([...prev].filter(id => failed.has(id))))
    } finally { setBusy(false) }
  }

  return (
    <div className="wp-group">
      {!open ? (
        <button className="wp-group-toggle" onClick={() => setOpen(true)}
                title="Dedicate one model to several workers at once">
          ⧉ Assign a model to a group of workers…
        </button>
      ) : (
        <div className="wp-group-body">
          <div className="wp-group-head">
            <strong>Group assign</strong>
            <button className="wp-group-x" onClick={() => { setOpen(false); setPick(''); setSel(new Set()); setResults(null) }}>×</button>
          </div>
          <ModelPicker
            models={models}
            value={pick}
            onPick={(k) => { setPick(k); setSel(new Set()); setResults(null) }}
            placeholder="1. Pick a model to dedicate…"
          />
          {pick && (
            <>
              <div className="wp-group-selall">
                <label className="wp-group-check">
                  <input type="checkbox" checked={allSelected} onChange={toggleAll}
                         disabled={eligibleIds.length === 0} />
                  <span>Select all eligible ({eligibleIds.length})</span>
                </label>
              </div>
              <div className="wp-group-list">
                {rows.length === 0 && <div className="wp-none">No workers in the pool.</div>}
                {rows.map(({ w, already, eligible, reason }) => (
                  <label key={w.id}
                         className={`wp-group-row${eligible ? '' : ' wp-group-locked'}`}
                         title={eligible ? '' : reason}>
                    <input type="checkbox"
                           checked={already || sel.has(w.id)}
                           disabled={!eligible}
                           onChange={() => toggle(w.id)} />
                    <span className="wp-group-wname">{w.name}</span>
                    {reason && <span className="wp-group-note">{reason}</span>}
                    {w.disk && typeof w.disk.free_bytes === 'number' && (
                      <span className="wp-group-disk">{fmtBytes(w.disk.free_bytes)} free</span>
                    )}
                  </label>
                ))}
              </div>
              <div className="wp-group-actions">
                <button className="wp-group-go" disabled={busy || chosen.length === 0}
                        onClick={dedicate}
                        title="Preflight (VRAM+RAM+disk) then assign to each selected worker">
                  {busy ? 'Dedicating…' : `Dedicate to ${chosen.length} worker${chosen.length === 1 ? '' : 's'}`}
                </button>
              </div>
              {results && (
                <div className="wp-group-results">
                  {results.length === 0 && <span className="wp-group-ok">Nothing to do — all selected workers already had it.</span>}
                  {results.map(r => (
                    <div key={r.id} className={r.ok ? 'wp-group-ok' : 'wp-group-bad'}>
                      {r.ok ? '✓' : '✗'} {r.name}{r.note ? ` — ${r.note}` : ''}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
