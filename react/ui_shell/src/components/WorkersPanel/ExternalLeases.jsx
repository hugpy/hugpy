import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJson } from '../../api'

// One host inventory: Hugpy models, registered harnesses, and raw GPU PIDs.
// Visibility comes from the GPU process scan; eviction requires a safe adapter.
export function ExternalLeases({ worker }) {
  const [rows, setRows] = useState(null)
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)
  const [chatTarget, setChatTarget] = useState(null)
  const [chatText, setChatText] = useState('')
  const [chatReply, setChatReply] = useState(null)
  const alive = useRef(true)
  const inFlight = useRef(false)

  const load = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    try {
      const result = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/activity`)
      if (!Array.isArray(result?.activity)) throw new Error('activity response has no activity list')
      if (alive.current) {
        // Central returns the presentation-ready activity inventory, already
        // excluding infrastructure and catalogued models shown in allocations.
        setRows(result.activity)
        setErr(null)
      }
    } catch (error) {
      // An unreachable worker or one delayed response is not an empty GPU.
      // Keep the last good snapshot; the next poll will refresh it.
      if (alive.current) setErr(`GPU activity refresh failed: ${error.message}`)
    } finally {
      inFlight.current = false
    }
  }, [worker.id])

  useEffect(() => {
    alive.current = true
    let timer
    let stopped = false
    const poll = async () => {
      await load()
      if (!stopped) timer = setTimeout(poll, 15000)
    }
    poll()
    return () => { stopped = true; alive.current = false; clearTimeout(timer) }
  }, [load])

  const adjust = useCallback(async (row, patch) => {
    setBusy(row.model_key)
    setErr(null)
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/external-set`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: row.model_key, ...patch }),
      })
    } catch (error) {
      setErr(`${row.model_key}: ${error.message}`)
    } finally {
      setBusy(null)
      load()
    }
  }, [worker.id, load])

  const call = useCallback(async () => {
    if (!chatTarget || !chatText.trim()) return
    setBusy(chatTarget)
    setChatReply(null)
    setErr(null)
    try {
      const result = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/external-chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: chatTarget, messages: [{ role: 'user', content: chatText.trim() }], max_tokens: 512 }),
      })
      setChatReply(result.choices?.[0]?.message?.content || JSON.stringify(result))
    } catch (error) {
      setErr(`${chatTarget}: ${error.message}`)
    } finally {
      setBusy(null)
      load()
    }
  }, [worker.id, chatTarget, chatText, load])

  if (!rows || rows.length === 0) return err ? <span className="wp-lease-err" title={err}>⚠ {err}</span> : null

  return (
    <div className="wp-caps wp-leases">
      <span className="wp-cap-chip" title="Every visible GPU compute process on this host, joined to Hugpy models and registered harnesses where known. Unregistered processes remain visible without eviction controls.">
        ◉ host model / GPU activity:
      </span>
      {rows.map(row => {
        const evictable = row.evictable !== false
        const controllable = row.kind === 'external' && !row.immutable
        const label = row.kind === 'observed'
          ? `${row.service || 'process'} (PID ${row.pids?.[0]})`
          : row.model_key || row.service || `PID ${row.pids?.[0]}`
        return (
          <span key={`${row.kind}:${label}:${row.pids?.join(',')}`} className="wp-cap-chip wp-lease-chip"
                title={`${row.note || label}\nPIDs: ${(row.pids || []).join(', ') || 'none'}\nGPU: ${(row.gpu_indices || []).join(', ') || 'none'}`}>
            <strong>{label}</strong>
            {!!row.gpu_indices?.length && <em> · GPU {row.gpu_indices.join(',')}</em>}
            {row.vram_mib > 0 && <em> · {(row.vram_mib / 1024).toFixed(1)} GiB</em>}
            {row.state && <em> · {row.state}</em>}
            {row.activity?.active_requests > 0 && <em> · {row.activity.active_requests} active</em>}
            {row.immutable && <em> · 🔒 observation only</em>}
            {row.kind === 'observed' && <em> · no adapter</em>}
            {controllable &&
              <button className={`wp-lease-toggle${evictable ? '' : ' wp-lease-off'}`}
                      disabled={busy === row.model_key}
                      title={evictable ? 'Allow Hugpy to evict this service when resources are needed' : 'Protected from demand eviction; force can still override'}
                      onClick={() => adjust(row, { evictable: !evictable })}>
                {evictable ? '⚡ evictable' : '🔒 protected'}
              </button>}
            {row.kind === 'external' && row.api_available &&
              <button className="wp-lease-toggle" onClick={() => { setChatTarget(row.model_key); setChatReply(null) }}>
                💬 Call
              </button>}
          </span>
        )
      })}
      {chatTarget && <div className="wp-lease-chat">
        <label>Call {chatTarget}
          <textarea value={chatText} onChange={event => setChatText(event.target.value)} rows={3} />
        </label>
        <button disabled={!chatText.trim() || busy === chatTarget} onClick={call}>{busy === chatTarget ? 'Calling…' : 'Send'}</button>
        <button onClick={() => setChatTarget(null)}>Close</button>
        {chatReply && <pre>{chatReply}</pre>}
      </div>}
      {err && <span className="wp-lease-err" title={err}>⚠ {err}</span>}
    </div>
  )
}
