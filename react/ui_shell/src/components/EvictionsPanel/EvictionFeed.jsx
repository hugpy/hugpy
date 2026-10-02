import { useEffect, useMemo, useState } from 'react'
import { useEvictionRuns } from './evictionStream'
import {
  fmtBytes, fmtMs, fmtClock, fmtReason,
  OUTCOME_CLASS, tierClass, buildRun, runVictims, runFreedBytes,
  failCode, provisionFailText,
} from './evictionFormat'
import './EvictionFeed.css'

// EVICTION FEED — the compact variant, embedded under the model list.
//
// The operator's use (2026-07-28): call a model from chat and watch THAT
// model's eviction behaviour on the same screen. So this is one LINE per
// headroom pass — what wanted in, what got unloaded, how it ended — and the
// candidate/skip detail (the full panel's reason for existing) folds away
// behind a click rather than being dropped.
//
// Selection HIGHLIGHTS, it does not filter. A model that never appears in an
// eviction is itself the answer to "why is this slow?", and a view that had
// silently filtered the rest away would hide the neighbours that explain it.
// The filter toggle is there when the operator explicitly wants the narrow
// view.
//
// Shares the single EventSource with the Evictions tab (see evictionStream) —
// both can be mounted at once and only one stream is open.

const MAX_ROWS = 50

function FeedRow({ run, now, selected, expanded, onToggle }) {
  const done = run.done
  const outcome = done ? (done.outcome || 'fit') : null
  const victims = runVictims(run)
  const freed = runFreedBytes(run)
  const skips = run.rows.filter(r => r.kind === 'skip')
  const elapsedMs = ((run.endTs || now / 1000) - run.startTs) * 1000

  // Highlight when the selected model was involved in ANY role — loading,
  // protected, or unloaded.
  const hit = selected && run.touched.has(selected)

  return (
    <div className={`evf-run${hit ? ' evf-hit' : ''}${done ? '' : ' evf-running'}`}>
      <button
        className="evf-line"
        onClick={onToggle}
        title={run.failureDetail || (expanded ? 'Collapse' : 'Show the candidates this pass walked')}
      >
        <span className={`evf-caret${expanded ? ' evf-caret-open' : ''}`}>›</span>

        <span className={`evf-model${run.incoming ? '' : ' evf-model-sweep'}${selected && run.incoming === selected ? ' evf-model-sel' : ''}`}>
          {run.incoming || 'headroom sweep'}
        </span>

        {/* A run that FAILED says so here and nowhere else matters — the whole
            point of the 2026-07-28 ENOSPC incident is that the operator reads
            "ENOSPC" off the collapsed list. It displaces the victim summary
            because "no eviction" is a true but useless thing to say about a
            request that never got as far as needing one. Full sentence lives in
            the row's title and in the expanded detail. */}
        {run.failure ? (
          <span className="evf-victims evf-fail" title={run.failureDetail || run.failure}>
            {run.failure}
          </span>
        ) : victims.length > 0 ? (
          <span className="evf-victims">
            <span className="evf-arrow">unloaded</span>
            {victims.map(v => (
              <span key={v.key}
                    className={`evf-victim${selected && v.model_key === selected ? ' evf-victim-sel' : ''}`}>
                {v.model_key}
              </span>
            ))}
          </span>
        ) : (
          <span className="evf-victims evf-none">
            {skips.length > 0 ? `${skips.length} protected, none unloaded` : 'no eviction'}
          </span>
        )}

        <span className="evf-spacer" />

        {freed != null && <span className="evf-freed">{fmtBytes(freed)}</span>}
        {run.worker_id && <span className="evf-chip">{run.worker_id}</span>}
        <span className="evf-time">{fmtClock(run.startTs)}</span>
        {done ? (
          <span className={`evf-outcome ${OUTCOME_CLASS[outcome] || 'ev-out-unfit'}`}>{outcome}</span>
        ) : run.failure ? (
          <span className="evf-outcome ev-out-failed">failed</span>
        ) : (
          <span className="evf-outcome ev-out-running">{fmtMs(Math.max(0, elapsedMs))}</span>
        )}
      </button>

      {expanded && (
        <div className="evf-detail">
          {run.trigger && <span className="evf-dchip">trigger {run.trigger}</span>}
          {run.needBytes != null && <span className="evf-dchip">needs {fmtBytes(run.needBytes)}</span>}

          {run.rows.length === 0 && <div className="evf-drow evf-dquiet">no candidates walked</div>}

          {run.rows.map(r => {
            // MODEL GROUPS — which iteration won, and why each other lost.
            if (r.kind === 'member') {
              return (
                <div className={`evf-drow evf-dgroup${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`}
                     key={r.key} title={r.reason || undefined}>
                  <span className="evf-dmark">member</span>
                  <span className="evf-dchip-inline">{r.group_key}</span>
                  <span className="evf-dkey">{r.model_key}</span>
                  {r.as && <span className="evf-dchip-inline">as {r.as}</span>}
                  {r.reason && <span className="evf-dwhy">{r.reason}</span>}
                </div>
              )
            }
            if (r.kind === 'memberskip') {
              return (
                <div className={`evf-drow evf-dquiet${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`}
                     key={r.key} title={r.reason || undefined}>
                  <span className="evf-dmark">skip</span>
                  <span className="evf-dkey">{r.model_key}</span>
                  <span className="evf-dwhy">{r.reason}</span>
                </div>
              )
            }
            if (r.kind === 'provision') {
              const cls = r.state === 'failed' ? 'evf-dfail'
                : r.state === 'done' ? 'evf-devict' : 'evf-dinflight'
              return (
                <div className={`evf-drow ${cls}${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`}
                     key={r.key} title={r.state === 'failed' ? provisionFailText(r) : (r.dest_path || undefined)}>
                  <span className="evf-dmark">prov</span>
                  <span className="evf-dchip-inline">{r.source || 'unknown'}</span>
                  <span className="evf-dkey">{r.model_key}</span>
                  {r.state === 'failed' && (
                    <span className="evf-dwhy">{failCode(r) ? `${failCode(r)}: ` : ''}{provisionFailText(r)}</span>
                  )}
                  {r.state === 'running' && <span className="evf-dwhy">fetching…</span>}
                  {r.state === 'done' && r.bytes != null && <span className="evf-dwhy">{fmtBytes(r.bytes)}</span>}
                  {r.state === 'done' && r.duration_ms != null && <span className="evf-dwhy">{fmtMs(r.duration_ms)}</span>}
                </div>
              )
            }
            if (r.kind === 'resolve') {
              return (
                <div className={`evf-drow evf-dfail${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`}
                     key={r.key} title={r.resolved_path || undefined}>
                  <span className="evf-dmark">resolve</span>
                  <span className="evf-dkey">{r.model_key}</span>
                  <span className="evf-dwhy">{fmtReason(r.reason) || 'could not resolve a path'}</span>
                </div>
              )
            }
            if (r.kind === 'load') {
              const cls = r.state === 'failed' ? 'evf-dfail'
                : r.state === 'done' ? 'evf-devict' : 'evf-dinflight'
              return (
                <div className={`evf-drow ${cls}${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`} key={r.key}>
                  <span className="evf-dmark">load</span>
                  {r.engine && <span className="evf-dchip-inline">{r.engine}</span>}
                  <span className="evf-dkey">{r.model_key}</span>
                  {r.state === 'failed' && <span className="evf-dwhy">{fmtReason(r.error) || 'load failed'}</span>}
                  {r.state === 'running' && <span className="evf-dwhy">loading…</span>}
                  {r.duration_ms != null && <span className="evf-dwhy">{fmtMs(r.duration_ms)}</span>}
                </div>
              )
            }
            if (r.kind === 'skip') {
              return (
                <div className={`evf-drow evf-dskip${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`} key={r.key}>
                  <span className="evf-dmark">skip</span>
                  <span className="evf-dkey">{r.model_key}</span>
                  <span className={tierClass(r.tier)}>{r.tier}</span>
                  <span className="evf-dwhy">{fmtReason(r.reason) || 'no reason recorded'}</span>
                </div>
              )
            }
            if (r.kind === 'victim') {
              const cls = r.state === 'failed' ? 'evf-dfail'
                : r.state === 'done' ? 'evf-devict' : 'evf-dinflight'
              return (
                <div className={`evf-drow ${cls}${selected && r.model_key === selected ? ' evf-drow-sel' : ''}`} key={r.key}>
                  <span className="evf-dmark">
                    {r.state === 'failed' ? 'fail' : r.state === 'done' ? 'evict' : '…'}
                  </span>
                  <span className="evf-dkey">{r.model_key}</span>
                  <span className={tierClass(r.tier)}>{r.tier}</span>
                  {r.state === 'failed' && <span className="evf-dwhy">{fmtReason(r.error)}</span>}
                  {r.freed_bytes != null && <span className="evf-dwhy">freed {fmtBytes(r.freed_bytes)}</span>}
                  {r.duration_ms != null && <span className="evf-dwhy">{fmtMs(r.duration_ms)}</span>}
                </div>
              )
            }
            if (r.kind === 'fitfail') {
              return (
                <div className="evf-drow evf-dquiet" key={r.key}>
                  <span className="evf-dmark">unfit</span>
                  <span className="evf-dwhy">needs {fmtBytes(r.need)} · {fmtBytes(r.free)} free</span>
                </div>
              )
            }
            if (r.kind === 'reclaim') return null
            return (
              <div className="evf-drow evf-dverdict" key={r.key}>
                <span className="evf-dmark">verdict</span>
                <span className="evf-dkey">{r.action}</span>
                {r.reason != null && <span className="evf-dwhy">{fmtReason(r.reason)}</span>}
              </div>
            )
          })}

          {done && fmtReason(done.reason) && (
            <div className="evf-drow evf-dquiet">
              <span className="evf-dmark">why</span>
              <span className="evf-dwhy">{fmtReason(done.reason)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * @param selectedModel model_key currently chosen in the list above; highlighted
 *                      (never filtered away) wherever it appears in a pass.
 */
export default function EvictionFeed({ selectedModel = null }) {
  const [paused, setPaused] = useState(false)
  const [onlySelected, setOnlySelected] = useState(false)
  const [expanded, setExpanded] = useState(() => new Set())
  const [now, setNow] = useState(() => Date.now())

  const { runs, conn, error, held, reconnect } = useEvictionRuns({ paused })

  // The elapsed clock only matters for a pass still in flight, so the timer
  // only runs while one is — an idle Models tab does no per-second work.
  const anyRunning = useMemo(
    () => runs.some(r => !r.events.some(e => e.stage === 'headroom.done')),
    [runs],
  )
  useEffect(() => {
    if (!anyRunning) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [anyRunning])

  const built = useMemo(() => {
    const out = []
    for (let i = runs.length - 1; i >= 0 && out.length < MAX_ROWS; i--) {
      const r = buildRun(runs[i])
      if (onlySelected && selectedModel && !r.touched.has(selectedModel)) continue
      out.push(r)
    }
    return out
  }, [runs, onlySelected, selectedModel])

  const hits = useMemo(
    () => (selectedModel ? built.filter(r => r.touched.has(selectedModel)).length : 0),
    [built, selectedModel],
  )

  const toggle = (id) => setExpanded(prev => {
    const n = new Set(prev)
    if (n.has(id)) n.delete(id); else n.add(id)
    return n
  })

  return (
    <div className="evf">
      <div className="evf-bar">
        <span className={`ev-dot ev-dot-${conn}`} title={`stream ${conn}`} />
        <span className="evf-title">Evictions</span>

        {selectedModel && (
          <span className="evf-sel" title={`Highlighting passes involving ${selectedModel}`}>
            {selectedModel}
            <span className="evf-sel-n">{hits}</span>
          </span>
        )}

        <span className="evf-spacer" />

        {selectedModel && (
          <button
            className={`evf-btn${onlySelected ? ' evf-btn-on' : ''}`}
            onClick={() => setOnlySelected(v => !v)}
            title={onlySelected
              ? 'Showing only this model — click to see its neighbours again'
              : 'Show only passes involving the selected model'}
          >
            only this
          </button>
        )}

        <button
          className={`evf-btn${paused ? ' evf-btn-on' : ''}`}
          onClick={() => setPaused(p => !p)}
          title={paused
            ? 'Resume — held events land in order; the stream never dropped'
            : 'Freeze this view. The stream stays open and keeps collecting.'}
        >
          {paused ? `paused${held ? ` (${held})` : ''}` : 'live'}
        </button>

        {conn === 'disconnected' && (
          <button className="evf-btn" onClick={reconnect}>reconnect</button>
        )}
      </div>

      {error && <div className="evf-err">{error}</div>}

      <div className="evf-list">
        {built.length === 0 ? (
          <div className="evf-empty">
            {onlySelected && selectedModel
              ? <>No eviction passes have involved <code>{selectedModel}</code> yet.</>
              : <>No eviction activity yet. Passes appear here the moment a worker
                 makes room — call a model from chat and watch it land.</>}
          </div>
        ) : built.map(r => (
          <FeedRow
            key={r.id}
            run={r}
            now={now}
            selected={selectedModel}
            expanded={expanded.has(r.id)}
            onToggle={() => toggle(r.id)}
          />
        ))}
      </div>
    </div>
  )
}
