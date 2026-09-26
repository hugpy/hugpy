import { useEffect, useState, useCallback } from 'react'
import { fetchJson } from '../../api'
import ModelDiscovery from './ModelDiscovery'
import './SettingsPanel.css'

// SETTINGS — the console's operator-knob surface. Init slice (2026-07-25) put
// the two eviction knobs from the spec (assets/evictionflow.html) on switches;
// they had shipped env-only, reachable solely by editing a systemd drop-in.
//
// ONE of the two remains:
//
//   * LEAST REAPING is FLEET-WIDE. It gates the drop pass, which central's
//     preview runs too — a per-worker value would show the operator one victim
//     list while the fleet executed another (the Parity failure the spec exists
//     to prevent). One value, /llm/evict-policy, shipped on the heartbeat.
//   * ANTI-THRASH FLOOR (per-worker) was RETIRED 2026-07-27 by operator ruling:
//     no timeblock may block a model being evicted. Its control is gone from
//     this panel and its key is rejected by the worker's /ops/config.
//
// Every value shown is the EFFECTIVE one (read back through the same reader the
// eviction path uses) with its SOURCE, so the console can never display a typed
// value the planner disagrees with.

const SOURCE_LABEL = {
  settings: 'set here',
  fleet: 'set here',
  env: 'unit drop-in',
  default: 'default',
}

function SourceTag({ source }) {
  return (
    <span className={`sp-src sp-src-${source || 'default'}`}>
      {SOURCE_LABEL[source] || source || 'default'}
    </span>
  )
}

// Fleet DISTRIBUTION mode (operator ruling 2026-09-24) — the catch-all that
// balances hugpy's default micro-placement friction. GET /api/llm/fleet/
// distribution is open; POST is operator-gated (operator_auth._SENSITIVE), same
// as the evict-policy write below. env HUGPY_DISTRIBUTION pins the effective
// mode (source: "env"): the buttons then read-only and a notice explains why.
const DIST_HELP = {
  feasible: 'Any online worker where the model feasibly fits is a routing '
    + 'candidate; designations become an ordered preference with a feasible-set '
    + 'fallback.',
  designated: 'The legacy sealed scope: only designated / resident / granted '
    + 'homes and per-worker wildcard opt-ins serve, and an unmet preference '
    + 'refuses.',
}
const DIST_SOURCE_LABEL = { env: 'env HUGPY_DISTRIBUTION', store: 'set here', default: 'default' }

export default function SettingsPanel({ workers = [] }) {
  const [policy, setPolicy] = useState(null)      // { least_reaping, ..._source }
  const [dist, setDist] = useState(null)          // { mode, source, env_override, stored }
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState('')
  const [note, setNote] = useState('')

  const load = useCallback(() => {
    fetchJson('/api/llm/evict-policy')
      .then(d => { setPolicy(d); setError(null) })
      .catch(e => setError(e.message))
    fetchJson('/api/llm/fleet/distribution')
      .then(d => { setDist(d) })
      .catch(e => setError(e.message))
  }, [])

  useEffect(() => { load() }, [load])

  // Set the FLEET distribution mode. Operator-gated: a refusal surfaces the
  // server's message (fetchJson throws on 401/403) just like the evict-policy
  // write. When env pins the mode the reply reports env_override and the
  // unchanged effective mode — stored, but not in effect until the env clears.
  const setDistribution = (mode) => {
    setBusy('dist'); setNote('')
    fetchJson('/api/llm/fleet/distribution', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    })
      .then(d => {
        setDist(d)
        setNote(d.env_override
          ? 'Stored — but HUGPY_DISTRIBUTION pins the effective mode until the env var is cleared.'
          : `Fleet distribution set to ${d.mode}. Routing adopts it immediately.`)
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(''))
  }

  // Flip the FLEET drop-pass policy. Applies to every worker on its next beat.
  const setLeastReaping = (value) => {
    setBusy('policy'); setNote('')
    fetchJson('/api/llm/evict-policy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ least_reaping: value }),
    })
      .then(d => {
        setPolicy(d)
        setNote(value === null
          ? 'Fleet ruling cleared — workers revert to their own drop-in on the next beat.'
          : 'Fleet policy set. Every worker adopts it on its next heartbeat.')
      })
      .catch(e => setError(e.message))
      .finally(() => setBusy(''))
  }

  // The per-worker ANTI-THRASH FLOOR control lived here until 2026-07-27, when
  // the operator retired the floor itself ("is there still some timeblock on a
  // model being evicted? if so eliminate it"). The writer is gone with it: the
  // backend now REJECTS evict_min_residency_s as a retired key, so leaving the
  // control up would have shown the operator a knob that 400s on save.

  return (
    <div className="sp-panel">
      {error && <div className="sp-err">{error}</div>}
      {note && <div className="sp-note">{note}</div>}

      {/* ── FLEET: distribution mode (feasible|designated) ───────────────── */}
      <section className="sp-sec">
        <div className="sp-sec-head">
          <h3 className="sp-sec-title">Distribution</h3>
          <span className="sp-scope sp-scope-fleet">fleet-wide</span>
        </div>
        <p className="sp-desc">
          The catch-all that balances hugpy&apos;s default micro-placement.
          <strong> Feasible</strong> (the default) lets any online worker where a
          model actually fits be a routing candidate — your designations come
          first as an ordered preference, but when none can serve, routing falls
          back to any feasible worker instead of refusing.
          <strong> Designated</strong> is the legacy sealed scope: only
          designated / resident / granted homes and per-worker wildcard opt-ins
          serve, and an unmet preference refuses.
        </p>
        {dist == null ? (
          <div className="sp-loading">loading…</div>
        ) : (
          <>
            <div className="sp-row">
              <div className="sp-val">
                <span className="sp-val-main">
                  {dist.mode === 'designated' ? 'Designated' : 'Feasible'}
                </span>
                <span className={`sp-src sp-src-${dist.source || 'default'}`}>
                  {DIST_SOURCE_LABEL[dist.source] || dist.source || 'default'}
                </span>
              </div>
              <div className="sp-actions">
                <button
                  className={`sp-btn ${dist.mode === 'feasible' ? 'sp-btn-on' : ''}`}
                  disabled={busy === 'dist' || dist.env_override || dist.mode === 'feasible'}
                  title={DIST_HELP.feasible}
                  onClick={() => setDistribution('feasible')}
                >Feasible (default)</button>
                <button
                  className={`sp-btn ${dist.mode === 'designated' ? 'sp-btn-on' : ''}`}
                  disabled={busy === 'dist' || dist.env_override || dist.mode === 'designated'}
                  title={DIST_HELP.designated}
                  onClick={() => setDistribution('designated')}
                >Designated</button>
              </div>
            </div>
            {dist.env_override && (
              <p className="sp-desc sp-desc-why">
                ⚙ Pinned by the <code>HUGPY_DISTRIBUTION</code> environment
                variable — the effective mode is <strong>{dist.mode}</strong>{' '}
                until it is cleared.
                {dist.stored && dist.stored !== dist.mode &&
                  <> The stored preference is <strong>{dist.stored}</strong>.</>}
              </p>
            )}
          </>
        )}
      </section>

      {/* ── FLEET: least reaping ─────────────────────────────────────────── */}
      <section className="sp-sec">
        <div className="sp-sec-head">
          <h3 className="sp-sec-title">Eviction: least reaping</h3>
          <span className="sp-scope sp-scope-fleet">fleet-wide</span>
        </div>
        <p className="sp-desc">
          When an admission needs room, the planner walks candidates in order until
          it has freed enough, then <strong>drops any victim the rest already
          cover</strong>. That can satisfy a 15&nbsp;GiB need with one 35&nbsp;GiB
          unload instead of two smaller ones — <strong>fewer models disturbed, but
          less free headroom left behind</strong>. Turning it off restores the
          older greedy walk: more models unloaded, more headroom free. On a tight
          disk that may be what you want.
        </p>
        <p className="sp-desc sp-desc-why">
          This one is fleet-wide on purpose. Central&apos;s eviction preview runs the
          same drop pass the workers do; if they disagreed, the console would show
          you one victim list while the fleet unloaded another.
        </p>
        {policy == null ? (
          <div className="sp-loading">loading…</div>
        ) : (
          <div className="sp-row">
            <div className="sp-val">
              <span className="sp-val-main">
                {policy.least_reaping ? 'On' : 'Off (greedy walk)'}
              </span>
              <SourceTag source={policy.least_reaping_source} />
            </div>
            <div className="sp-actions">
              <button
                className={`sp-btn ${policy.least_reaping ? 'sp-btn-on' : ''}`}
                disabled={busy === 'policy' || policy.least_reaping}
                onClick={() => setLeastReaping(true)}
              >On</button>
              <button
                className={`sp-btn ${!policy.least_reaping ? 'sp-btn-on' : ''}`}
                disabled={busy === 'policy' || !policy.least_reaping}
                onClick={() => setLeastReaping(false)}
              >Off</button>
              {policy.least_reaping_source === 'fleet' && (
                <button
                  className="sp-btn sp-btn-quiet"
                  disabled={busy === 'policy'}
                  onClick={() => setLeastReaping(null)}
                >Clear ruling</button>
              )}
            </div>
          </div>
        )}
      </section>

      {/* ── PER-WORKER: anti-thrash floor — REMOVED 2026-07-27 ───────────────
          The floor it wrote no longer exists (operator ruling: no timeblock on
          a model being evicted). Documented rather than silently deleted so the
          next reader knows the control was retired, not lost. What replaced it:
          nothing — freshness is expressed as RANK in the evict order, and the
          only two things that block an eviction are 🔒static residency and a
          model that is actively answering. */}

      {/* ── FLEET: automated model discovery (the nightly review timers) ──── */}
      <ModelDiscovery />
    </div>
  )
}
