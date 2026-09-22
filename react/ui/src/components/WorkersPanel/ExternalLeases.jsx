import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJson } from '../../api'

// External gpu_lease residents — wildcard processes (e.g. the bluebook OCR
// batch) riding the card as evictable pseudo-models via `hugpy-lease`.
// Read: GET /llm/workers/<id>/external (relay of the worker's /ops/residents).
// Write: POST /llm/workers/<id>/external-set — the console twin of the
// `hugpy-lease-set` CLI. The worker-side registry is authoritative and
// forwards each set to the lease supervisor's control URL, so the toggles
// apply LIVE (no job restart) and the lease's own heartbeats can't undo an
// operator's choice. A restart of the lease service reverts to its unit-file
// flags — these toggles are the "for now" lever, the unit is the "forever" one.
//
// Renders nothing when the worker has no external leases (the common case),
// so it costs no vertical space on ordinary workers.
export function ExternalLeases({ worker }) {
  const [rows, setRows] = useState(null)   // null = not loaded yet
  const [busy, setBusy] = useState(null)   // model_key of the in-flight set
  const [err, setErr]   = useState(null)
  const alive = useRef(true)

  const load = useCallback(async () => {
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/external`)
      if (!alive.current) return
      setRows(r.external || [])
    } catch {
      // Unreachable worker or older agent without /ops/residents: stay hidden
      // rather than adding a permanent warning chip to every dead row.
      if (alive.current) setRows([])
    }
  }, [worker.id])

  useEffect(() => {
    alive.current = true
    load()
    const t = setInterval(load, 30000)
    return () => { alive.current = false; clearInterval(t) }
  }, [load])

  const adjust = useCallback(async (lease, patch) => {
    setBusy(lease.model_key)
    setErr(null)
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/external-set`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: lease.model_key, ...patch }),
      })
    } catch (e) {
      setErr(`${lease.model_key}: ${e.message}`)
    } finally {
      setBusy(null)
      load()
    }
  }, [worker.id, load])

  if (!rows || rows.length === 0) return null

  return (
    <div className="wp-caps wp-leases">
      <span className="wp-cap-chip"
            title={'External gpu_lease residents: batch jobs sharing this card as pseudo-models (hugpy-lease). '
                 + 'Both toggles apply live via the worker’s /ops/external/set — no job restart. '
                 + 'They revert to the lease’s launch flags if the lease service itself restarts.'}>
        🎟 leases:
      </span>
      {rows.map(l => {
        const evictable = l.evictable !== false
        const resumes   = l.resume !== 'disabled'
        const isBusy    = busy === l.model_key
        return (
          <span key={l.model_key} className="wp-cap-chip wp-lease-chip">
            <strong title={l.note || l.model_key}>{l.model_key}</strong>
            {l.vram_gib != null && <em> {l.vram_gib}GiB</em>}
            {l.state && <em> · {l.state}</em>}
            <button className={`wp-lease-toggle${evictable ? '' : ' wp-lease-off'}`}
                    disabled={isBusy}
                    title={evictable
                      ? 'Evictable: any demand path (model load, image gen) may pause this job to reclaim VRAM. Click to protect it (external twin of 🔒 static — only force-evict touches it).'
                      : 'Protected (non-evictable): only force-evict can pause this job. Click to make it evictable again.'}
                    onClick={() => adjust(l, { evictable: !evictable })}>
              {evictable ? '⚡ evictable' : '🔒 protected'}
            </button>
            <button className={`wp-lease-toggle${resumes ? '' : ' wp-lease-off'}`}
                    disabled={isBusy}
                    title={resumes
                      ? 'Resume enabled: after an eviction pause the lease waits, re-claims VRAM and relaunches the job where it left off. Click to make a pause final.'
                      : 'Resume disabled: a pause ENDS the run (the supervisor unregisters and exits). Click to re-enable relaunch-after-pause.'}
                    onClick={() => adjust(l, { resume: resumes ? 'disabled' : 'enabled' })}>
              {resumes ? '🔁 resume' : '⏹ one-shot'}
            </button>
          </span>
        )
      })}
      {err && <span className="wp-lease-err" title={err}>⚠ {err}</span>}
    </div>
  )
}
