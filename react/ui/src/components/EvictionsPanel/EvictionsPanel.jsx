import { useState, useMemo, useEffect } from 'react'
import { useEvictionRuns } from './evictionStream'
import {
  fmtBytes, fmtMs, fmtClock, fmtReason,
  OUTCOME_CLASS, tierClass, buildRun,
  failCode, provisionFailText,
} from './evictionFormat'
import './EvictionsPanel.css'

// EVICTIONS — the live trace of every make-room decision the fleet takes.
//
// The workers emit one event per STAGE of a headroom pass (start -> candidate
// walk -> evictions -> verdict -> done), tagged with a run_id. This panel groups
// those back into one card per run, newest on top, so the operator can read a
// single admission as a story: what wanted in, what was considered, what was
// PROTECTED and why, what actually got unloaded, and how it ended.
//
// The skipped-candidate rows are the point of the panel. "Why is this model
// still resident when the load refused?" is only answerable from the reason
// the planner recorded at the moment it passed the candidate over.
//
// The stream itself lives in evictionStream (shared with the compact feed in
// the Models tab) and the folding/formatting in evictionFormat, so the two
// views cannot drift and only ONE EventSource is ever open.

function TierChip({ tier }) {
  if (!tier) return null
  return <span className={tierClass(tier)}>{tier}</span>
}

// ── One run card ────────────────────────────────────────────────────────────
function RunCard({ run, now }) {
  const done = run.done
  const outcome = done ? (done.outcome || 'fit') : null
  const elapsedMs = ((run.endTs || now / 1000) - run.startTs) * 1000

  return (
    <div className={`ev-card ${done ? '' : 'ev-card-running'}`}>
      <div className="ev-head">
        <span className={`ev-model ${run.incoming ? '' : 'ev-model-sweep'}`}>
          {run.incoming || 'headroom sweep'}
        </span>
        {run.trigger && <span className={`ev-chip ev-trigger ev-trigger-${run.trigger}`}>{run.trigger}</span>}
        {run.worker_id && <span className="ev-chip ev-worker">{run.worker_id}</span>}
        {run.tier && <TierChip tier={run.tier} />}
        <span className="ev-time">{fmtClock(run.startTs)}</span>
        <span className="ev-elapsed">{fmtMs(Math.max(0, elapsedMs))}</span>
        <span className="ev-spacer" />
        {done ? (
          <span className={`ev-outcome ${OUTCOME_CLASS[outcome] || 'ev-out-unfit'}`}>{outcome}</span>
        ) : run.failure ? (
          // Died before headroom.done — provisioning, resolve or the load itself.
          // Nothing will ever close this run, so without a terminal chip it would
          // sit "running…" forever and read as slow rather than broken.
          <span className="ev-outcome ev-out-failed" title={run.failureDetail || run.failure}>failed</span>
        ) : (
          <span className="ev-outcome ev-out-running">running…</span>
        )}
      </div>

      {run.needBytes != null && (
        <div className="ev-need">needs {fmtBytes(run.needBytes)}</div>
      )}

      {run.rows.length === 0 && (
        <div className="ev-row ev-row-quiet">no candidates walked</div>
      )}

      {run.rows.map(r => {
        // ── model groups (run BEFORE provisioning: which iteration serves) ───
        if (r.kind === 'member') {
          return (
            <div className="ev-row ev-row-member" key={r.key}>
              <span className="ev-mark">member</span>
              <span className="ev-chip ev-source">{r.group_key}</span>
              <span className="ev-key">{r.model_key}</span>
              {r.as && <span className="ev-chip">as {r.as}</span>}
              <span className="ev-why">{r.reason || 'selected'}</span>
              <span className="ev-spacer" />
              {r.need_bytes != null && r.verdict === 'need' && (
                <span className="ev-num">declares {fmtBytes(r.need_bytes)}</span>
              )}
            </div>
          )
        }
        if (r.kind === 'memberskip') {
          return (
            <div className="ev-row ev-row-skip" key={r.key}>
              <span className="ev-mark">member skip</span>
              <span className="ev-key">{r.model_key}</span>
              <span className="ev-why">{r.reason || 'no reason recorded'}</span>
            </div>
          )
        }
        // ── serve pipeline (runs BEFORE any eviction; see evictionFormat) ────
        if (r.kind === 'provision') {
          const cls = r.state === 'failed' ? 'ev-row-fail'
            : r.state === 'done' ? 'ev-row-evict' : 'ev-row-inflight'
          return (
            <div className={`ev-row ${cls}`} key={r.key} title={r.dest_path || undefined}>
              <span className="ev-mark">provision</span>
              <span className="ev-chip ev-source">{r.source || 'unknown'}</span>
              <span className="ev-key">{r.model_key}</span>
              {r.state === 'failed' && (
                // The errno FIRST — it is what the operator scans for — then the
                // backend's composed sentence, which names the mount and the free
                // bytes. Anything less and this is another journalctl trip.
                <span className="ev-why">
                  {failCode(r) ? `${failCode(r)}: ` : ''}{provisionFailText(r)}
                </span>
              )}
              {r.state === 'running' && <span className="ev-why">fetching…</span>}
              <span className="ev-spacer" />
              {r.state === 'done' && r.bytes != null && (
                <span className="ev-num ev-num-strong">
                  {fmtBytes(r.bytes)}{r.duration_ms != null ? ` in ${fmtMs(r.duration_ms)}` : ''}
                </span>
              )}
              {r.state === 'done' && r.bytes == null && r.duration_ms != null && (
                <span className="ev-num">{fmtMs(r.duration_ms)}</span>
              )}
            </div>
          )
        }
        if (r.kind === 'resolve') {
          // Resolve only reports failure, so this row is unconditionally red.
          // resolved_path goes in the tooltip: it is the thing you paste into an
          // `ls`, but it is far too long to sit in the line.
          return (
            <div className="ev-row ev-row-fail" key={r.key} title={r.resolved_path || undefined}>
              <span className="ev-mark">resolve</span>
              <span className="ev-key">{r.model_key}</span>
              <span className="ev-why">{fmtReason(r.reason) || 'could not resolve a path'}</span>
            </div>
          )
        }
        if (r.kind === 'load') {
          const cls = r.state === 'failed' ? 'ev-row-fail'
            : r.state === 'done' ? 'ev-row-evict' : 'ev-row-inflight'
          return (
            <div className={`ev-row ${cls}`} key={r.key}>
              <span className="ev-mark">load</span>
              {r.engine && <span className="ev-chip ev-engine">{r.engine}</span>}
              <span className="ev-key">{r.model_key}</span>
              {r.state === 'failed' && <span className="ev-why">{fmtReason(r.error) || 'load failed'}</span>}
              {r.state === 'running' && <span className="ev-why">loading…</span>}
              <span className="ev-spacer" />
              {r.duration_ms != null && <span className="ev-num">{fmtMs(r.duration_ms)}</span>}
            </div>
          )
        }
        if (r.kind === 'skip') {
          return (
            <div className="ev-row ev-row-skip" key={r.key}>
              <span className="ev-mark">skip</span>
              <span className="ev-key">{r.model_key}</span>
              <TierChip tier={r.tier} />
              <span className="ev-why">{fmtReason(r.reason) || 'no reason recorded'}</span>
              <span className="ev-spacer" />
              {r.vram_bytes != null && <span className="ev-num">{fmtBytes(r.vram_bytes)}</span>}
              {r.idle_s != null && <span className="ev-num">idle {fmtMs(Number(r.idle_s) * 1000)}</span>}
            </div>
          )
        }
        if (r.kind === 'victim') {
          const cls = r.state === 'failed' ? 'ev-row-fail'
            : r.state === 'done' ? 'ev-row-evict' : 'ev-row-inflight'
          return (
            <div className={`ev-row ${cls}`} key={r.key}>
              <span className="ev-mark">
                {r.state === 'failed' ? 'fail' : r.state === 'done' ? 'evict' : '…'}
              </span>
              <span className="ev-key">{r.model_key}</span>
              <TierChip tier={r.tier} />
              {r.state === 'failed' && <span className="ev-why">{fmtReason(r.error) || 'eviction failed'}</span>}
              {r.state === 'running' && <span className="ev-why">unloading…</span>}
              <span className="ev-spacer" />
              {r.freed_bytes != null && <span className="ev-num ev-num-strong">freed {fmtBytes(r.freed_bytes)}</span>}
              {r.duration_ms != null && <span className="ev-num">{fmtMs(r.duration_ms)}</span>}
            </div>
          )
        }
        if (r.kind === 'fitfail') {
          return (
            <div className="ev-row ev-row-fitfail" key={r.key}>
              <span className="ev-mark">unfit</span>
              <span className="ev-why">
                needs {fmtBytes(r.need)} · {fmtBytes(r.free)} free
              </span>
            </div>
          )
        }
        if (r.kind === 'reclaim') {
          return (
            <div className="ev-row ev-row-quiet" key={r.key}>
              <span className="ev-mark">reclaim</span>
              <span className="ev-why">allocator reclaim complete</span>
            </div>
          )
        }
        // verdict
        return (
          <div className={`ev-row ev-row-verdict ev-verdict-${r.action || 'unknown'}`} key={r.key}>
            <span className="ev-mark">verdict</span>
            <span className="ev-key">{r.action}</span>
            {r.reason != null && <span className="ev-why">{fmtReason(r.reason)}</span>}
            <span className="ev-spacer" />
            {Array.isArray(r.evicted) && r.evicted.length > 0 && (
              <span className="ev-num">{r.evicted.length} evicted</span>
            )}
            {r.freed_bytes != null && <span className="ev-num">freed {fmtBytes(r.freed_bytes)}</span>}
          </div>
        )
      })}

      {done && (fmtReason(done.reason) || done.note || (Array.isArray(done.evicted) && done.evicted.length > 0)) && (
        <div className="ev-foot">
          {Array.isArray(done.evicted) && done.evicted.length > 0 && (
            <span className="ev-foot-evicted">unloaded: {done.evicted.join(', ')}</span>
          )}
          {done.reason != null && <span className="ev-foot-reason">{fmtReason(done.reason)}</span>}
          {done.note && <span className="ev-foot-note">{done.note}</span>}
        </div>
      )}
    </div>
  )
}

export default function EvictionsPanel() {
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('')
  const [now, setNow] = useState(() => Date.now())

  // One shared EventSource for the whole console — this panel and the compact
  // feed in the Models tab subscribe to the same stream (see evictionStream).
  const { runs, conn, error, held, reconnect } = useEvictionRuns({ paused })

  // Ticks the elapsed clock on runs that haven't reported headroom.done.
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const built = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const out = []
    for (let i = runs.length - 1; i >= 0; i--) {   // newest first
      const r = buildRun(runs[i])
      if (!q || r.haystack.includes(q)) out.push(r)
    }
    return out
  }, [runs, filter])

  return (
    <div className="ev-panel">
      <div className="ev-toolbar">
        <span className={`ev-dot ev-dot-${conn}`} title={`stream ${conn}`} />
        <span className="ev-conn">{conn}</span>

        <button
          className={`ev-btn ${paused ? 'ev-btn-paused' : ''}`}
          onClick={() => setPaused(p => !p)}
          title={paused
            ? 'Resume — queued events land in order; the stream never dropped'
            : 'Pause the view. The stream stays open and buffers.'}
        >
          {paused ? `Paused${held ? ` (${held})` : ''}` : 'Live'}
        </button>

        <input
          className="ev-filter"
          value={filter}
          placeholder="filter by model or worker…"
          onChange={e => setFilter(e.target.value)}
        />

        {conn === 'disconnected' && (
          <button className="ev-btn ev-btn-quiet" onClick={reconnect}>Reconnect</button>
        )}

        <span className="ev-spacer" />
        <span className="ev-count">{built.length} run{built.length === 1 ? '' : 's'}</span>
      </div>

      {error && <div className="ev-err">{error}</div>}

      {built.length === 0 && (
        <div className="ev-empty">
          {runs.length === 0 ? (
            <>
              <p className="ev-empty-line">No eviction activity recorded yet.</p>
              <p className="ev-empty-sub">
                Eviction events are emitted by the box that does the unloading, so
                they only appear from workers running a release that carries the
                worker-side emitters. Central does no local model serving, but it
                does clear the video card for a render — those runs show up here
                tagged <code>reservation</code>.
              </p>
            </>
          ) : (
            <p className="ev-empty-line">No runs match “{filter}”.</p>
          )}
        </div>
      )}

      <div className="ev-list">
        {built.map(r => <RunCard key={r.id} run={r} now={now} />)}
      </div>
    </div>
  )
}
