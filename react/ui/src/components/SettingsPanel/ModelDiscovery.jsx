import { Fragment, useEffect, useState, useCallback } from 'react'
import { fetchJson } from '../../api'
import ModelDossier from './ModelDossier'
import ModelRadar from './ModelRadar'
import './SettingsPanel.css'

// MODEL DISCOVERY — the automated model search's first UI (2026-08-13).
//
// The backend has run for a while with no console surface: the `review`
// package (search HF → screen → download the best → smoke-test on the GPU →
// LLM judge) fires nightly around 03:20 via systemd timers in the hugpy VM,
// one instance per saved criteria (~/.config/hugpy/review/<name>.json), and
// records everything into central's review DB. This section makes it
// visible and steerable:
//
//   * per-criteria ON/OFF — writes `enabled` into the criteria file;
//     pipeline.run() no-ops a disabled criteria, so the timer keeps firing
//     but does nothing (the root-less off switch).
//   * the search question itself (query / task / caps) — editable inline.
//   * "Run now" — POST /llm/review/run with force:true (works even while
//     the criteria is disabled: an explicit ask beats the switch).
//   * what it found — recent runs and the per-criteria leaderboard.
//
// PUT semantics matter here: /llm/review/criteria/<name> REBUILDS the file
// from the payload (absent fields revert to dataclass defaults), so every
// save round-trips the criteria's FULL dict with just the edited keys
// changed — never a patch.

const fmtTs = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '—')

// k120: the leaderboard row is now a HANDLE, not the answer. `dossier` is the
// compact summary the backend hangs on each review row (specialization, VRAM,
// verdict, margin against the incumbent); clicking a row expands the full
// dossier panel. A row with no dossier still renders exactly as it did before —
// pre-k120 reviews are in the same table and must not look broken.
function Leaderboard({ name }) {
  const [rows, setRows] = useState(null)
  const [err, setErr] = useState('')
  const [open, setOpen] = useState(null)

  useEffect(() => {
    fetchJson(`/api/llm/review/results?criteria=${encodeURIComponent(name)}&best=1&limit=8`)
      .then(d => setRows(Array.isArray(d) ? d : []))
      .catch(e => setErr(e.message))
  }, [name])

  if (err) return <div className="sp-empty">leaderboard unavailable: {err}</div>
  if (rows == null) return <div className="sp-loading">loading finds…</div>
  if (rows.length === 0) return <div className="sp-empty">Nothing recorded for this criteria yet.</div>
  return (
    <table className="sp-md-table">
      <thead>
        <tr><th>model</th><th>specialization</th><th>fit</th><th>vs incumbent</th>
            <th>verdict</th><th>reviewed</th></tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const d = (r.payload && r.payload.dossier) || null
          const isOpen = open === r.hub_id
          return (
            <Fragment key={`${r.hub_id}-${i}`}>
              <tr className="sp-md-rowlink"
                  onClick={() => setOpen(isOpen ? null : r.hub_id)}>
                <td className="sp-md-model">
                  <span className="sp-dos-caret">{isOpen ? '▾' : '▸'}</span>
                  <a href={`https://huggingface.co/${r.hub_id}`} target="_blank"
                     rel="noreferrer" onClick={e => e.stopPropagation()}>
                    {r.hub_id}
                  </a>
                </td>
                <td className="sp-md-dim">{d?.specialization || r.stage}</td>
                <td>
                  {d?.best_quant ? `${d.best_quant}` : '—'}
                  {d?.est_vram_bytes
                    ? <span className="sp-md-dim"> · {(d.est_vram_bytes / 1024 ** 3).toFixed(1)} GiB</span>
                    : null}
                </td>
                <td>
                  {d
                    ? <span className={d.beats_incumbent === 'yes' ? 'sp-dos-good-t'
                                     : d.beats_incumbent === 'no' ? 'sp-dos-bad-t' : 'sp-md-dim'}>
                        {d.beats_incumbent}
                        {d.margin != null && ` ${d.margin > 0 ? '+' : ''}${d.margin}`}
                      </span>
                    : <span className="sp-md-dim">—</span>}
                </td>
                <td>{d?.verdict || r.verdict || '—'}</td>
                <td className="sp-md-dim">{fmtTs(r.reviewed_at)}</td>
              </tr>
              {isOpen && (
                <tr>
                  <td colSpan={6}>
                    <ModelDossier criteria={name} hubId={r.hub_id} />
                  </td>
                </tr>
              )}
            </Fragment>
          )
        })}
      </tbody>
    </table>
  )
}

function CriteriaCard({ crit, onSaved, onError }) {
  const [query, setQuery] = useState(crit.query ?? '')
  const [task, setTask] = useState(crit.task ?? '')
  const [cap, setCap] = useState(crit.max_downloads_per_run ?? 2)
  const [busy, setBusy] = useState('')
  const [showFinds, setShowFinds] = useState(false)
  const [showDepth, setShowDepth] = useState(false)
  const [ranNote, setRanNote] = useState('')

  // k120 knobs. Every one defaults to the pre-k120 behaviour, so a card saved
  // before this panel existed round-trips unchanged (the PUT rebuilds the file
  // from the payload — see the header comment — so these MUST be spread from
  // `crit`, never invented here).
  const [depth, setDepth] = useState(crit.trial_depth ?? 'load-test')
  const [samples, setSamples] = useState(crit.sample_count ?? 2)
  const [compare, setCompare] = useState((crit.compare_against ?? []).join(', '))
  const [specs, setSpecs] = useState((crit.required_specializations ?? []).join(', '))
  const [licences, setLicences] = useState((crit.licenses_allowed ?? []).join(', '))
  const [research, setResearch] = useState(crit.external_research !== false)
  const [comm, setComm] = useState(crit.community !== false)
  const [radar, setRadar] = useState(!!crit.radar)
  const csv = (v) => v.split(',').map(x => x.trim()).filter(Boolean)

  const enabled = crit.enabled !== false
  const dirty = query !== (crit.query ?? '')
    || task !== (crit.task ?? '')
    || Number(cap) !== (crit.max_downloads_per_run ?? 2)
    || depth !== (crit.trial_depth ?? 'load-test')
    || Number(samples) !== (crit.sample_count ?? 2)
    || compare !== ((crit.compare_against ?? []).join(', '))
    || specs !== ((crit.required_specializations ?? []).join(', '))
    || licences !== ((crit.licenses_allowed ?? []).join(', '))
    || research !== (crit.external_research !== false)
    || comm !== (crit.community !== false)
    || radar !== !!crit.radar

  // Full-dict round-trip (see the header comment): spread the loaded crit,
  // override what changed, drop the status extras the PUT doesn't know.
  const save = useCallback((overrides) => {
    const { running, running_since, error, ...base } = crit
    const body = { ...base, query, task: task || null,
                   max_downloads_per_run: Number(cap) || 0,
                   trial_depth: depth,
                   sample_count: Number(samples) || 0,
                   compare_against: csv(compare),
                   required_specializations: csv(specs),
                   licenses_allowed: csv(licences),
                   external_research: research,
                   community: comm,
                   radar,
                   ...overrides }
    setBusy('save')
    fetchJson(`/api/llm/review/criteria/${encodeURIComponent(crit.name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(() => onSaved())
      .catch(e => onError(`Saving "${crit.name}" failed: ${e.message}`))
      .finally(() => setBusy(''))
  }, [crit, query, task, cap, depth, samples, compare, specs, licences,
      research, comm, radar, onSaved, onError])

  const runNow = useCallback(() => {
    setBusy('run'); setRanNote('')
    fetchJson('/api/llm/review/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ criteria: crit.name, force: true }),
    })
      .then(d => setRanNote(d.status === 'already_running'
        ? 'A run for this criteria is already in flight.'
        : 'Run started — results land in the tables below as stages finish.'))
      .catch(e => onError(`Run "${crit.name}" failed: ${e.message}`))
      .finally(() => { setBusy(''); onSaved() })
  }, [crit.name, onSaved, onError])

  return (
    <div className={`sp-md-card${enabled ? '' : ' sp-md-card-off'}`}>
      <div className="sp-row">
        <div className="sp-val">
          <span className="sp-val-main">{crit.name}</span>
          {crit.running && <span className="sp-scope sp-scope-fleet">running…</span>}
          {!enabled && <span className="sp-scope sp-scope-worker">off</span>}
        </div>
        <div className="sp-actions">
          <button className={`sp-btn ${enabled ? 'sp-btn-on' : ''}`}
                  disabled={busy !== '' || enabled}
                  onClick={() => save({ enabled: true })}>On</button>
          <button className={`sp-btn ${!enabled ? 'sp-btn-on' : ''}`}
                  disabled={busy !== '' || !enabled}
                  onClick={() => save({ enabled: false })}>Off</button>
          <button className="sp-btn" disabled={busy !== '' || crit.running}
                  title="Start a full run right now (works even when the nightly switch is off)"
                  onClick={runNow}>{busy === 'run' ? '…' : 'Run now'}</button>
          <button className="sp-btn sp-btn-quiet"
                  onClick={() => setShowDepth(s => !s)}>
            {showDepth ? 'hide depth' : 'how deep'}
          </button>
          <button className="sp-btn sp-btn-quiet"
                  onClick={() => setShowFinds(s => !s)}>
            {showFinds ? 'hide finds' : 'what it found'}
          </button>
        </div>
      </div>

      <div className="sp-row sp-md-edit">
        <label className="sp-md-lbl">query
          <input className="sp-in" value={query} placeholder="HF search text…"
                 onChange={e => setQuery(e.target.value)} />
        </label>
        <label className="sp-md-lbl">task
          <input className="sp-in sp-md-in-task" value={task ?? ''}
                 placeholder="e.g. text-generation"
                 onChange={e => setTask(e.target.value)} />
        </label>
        <label className="sp-md-lbl">downloads/run
          <input className="sp-in sp-md-in-cap" type="number" min="0" max="10"
                 value={cap} onChange={e => setCap(e.target.value)} />
        </label>
        {dirty && (
          <button className="sp-btn sp-btn-on" disabled={busy !== ''}
                  onClick={() => save({})}>{busy === 'save' ? '…' : 'Save'}</button>
        )}
      </div>

      {/* k120: "allow that complexity to be dictated" — the per-card depth
          knobs. Hidden behind a toggle because the common edit is still the
          query, and a settings panel that shows twelve inputs at rest is a
          panel nobody reads. */}
      {showDepth && (
        <div className="sp-row sp-md-edit sp-dos-knobs">
          <label className="sp-md-lbl">trial depth
            <select className="sp-in sp-md-in-task" value={depth}
                    onChange={e => setDepth(e.target.value)}>
              <option value="screen-only">screen-only — metadata, no download</option>
              <option value="load-test">load-test — download + load on the GPU</option>
              <option value="full-samples">full-samples — + stationary battery</option>
            </select>
          </label>
          <label className="sp-md-lbl">samples
            <input className="sp-in sp-md-in-cap" type="number" min="0" max="8"
                   value={samples} onChange={e => setSamples(e.target.value)} />
          </label>
          <label className="sp-md-lbl">compare against
            <input className="sp-in" value={compare}
                   placeholder="plot.construct, screenplay.complete"
                   title="routing-matrix operation names — the incumbent is whatever the matrix routes there"
                   onChange={e => setCompare(e.target.value)} />
          </label>
          <label className="sp-md-lbl">required specializations
            <input className="sp-in" value={specs} placeholder="code, reasoning"
                   title="screened from tags and the repo name only — no extra fetch"
                   onChange={e => setSpecs(e.target.value)} />
          </label>
          <label className="sp-md-lbl">licences allowed
            <input className="sp-in" value={licences} placeholder="apache, mit"
                   title="empty means any licence, which is the default"
                   onChange={e => setLicences(e.target.value)} />
          </label>
          <label className="sp-md-lbl sp-dos-check">
            <input type="checkbox" checked={research}
                   onChange={e => setResearch(e.target.checked)} />
            external research (card + papers)
          </label>
          <label className="sp-md-lbl sp-dos-check">
            <input type="checkbox" checked={comm}
                   onChange={e => setComm(e.target.checked)} />
            community scan (reddit / HN / discussions)
          </label>
          <label className="sp-md-lbl sp-dos-check">
            <input type="checkbox" checked={radar}
                   onChange={e => setRadar(e.target.checked)} />
            gem radar
          </label>
        </div>
      )}

      {ranNote && <div className="sp-note">{ranNote}</div>}
      {showFinds && <Leaderboard name={crit.name} />}
      {showFinds && crit.radar && <ModelRadar name={crit.name} />}
    </div>
  )
}

export default function ModelDiscovery() {
  const [crits, setCrits] = useState(null)
  const [runs, setRuns] = useState([])
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    fetchJson('/api/llm/review/status')
      .then(d => { setCrits(Array.isArray(d?.criteria) ? d.criteria : []); setError(null) })
      .catch(e => setError(e.message))
    fetchJson('/api/llm/review/runs?limit=8')
      .then(d => setRuns(Array.isArray(d) ? d : []))
      .catch(() => {})   // runs table is a convenience; cards still render
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 20_000)
    return () => clearInterval(t)
  }, [load])

  return (
    <section className="sp-sec">
      <div className="sp-sec-head">
        <h3 className="sp-sec-title">Model discovery (automated search)</h3>
        <span className="sp-scope sp-scope-fleet">nightly ~03:20</span>
      </div>
      <p className="sp-desc">
        Each card is a saved <strong>search question</strong> the fleet asks
        Hugging Face every night: candidates are screened on metadata (VRAM
        fit, quants, context, trust, downloads), the best few are downloaded
        and <strong>load-tested on the GPU</strong>, and an LLM judge files an
        adopt/trial/reject verdict. Off pauses a question without touching its
        settings; the nightly timer keeps ticking and simply skips it.
        {' '}<strong>How deep</strong> sets what each card does beyond the
        screen — the sample battery, the outside research, the community scan
        and the gem radar. Click a model in “what it found” for its full
        dossier: specialization, every quant’s VRAM, licence, what the card and
        the forums claim, the sample outputs, and the verdict with the evidence
        it cites.
      </p>
      {error && <div className="sp-err">{error}</div>}
      {crits == null && <div className="sp-loading">loading…</div>}
      {crits != null && crits.length === 0 && (
        <div className="sp-empty">
          No criteria saved on this central. (They live in
          ~/.config/hugpy/review/ next to the nightly timer.)
        </div>
      )}
      {(crits ?? []).map(c => (
        c.error
          ? <div key={c.name} className="sp-err">criteria {c.name}: {c.error}</div>
          : <CriteriaCard key={c.name} crit={c} onSaved={load} onError={setError} />
      ))}

      {runs.length > 0 && (
        <>
          <p className="sp-desc sp-desc-why">Recent runs (all criteria, newest first):</p>
          <table className="sp-md-table">
            <thead>
              <tr><th>criteria</th><th>started</th><th>screened</th><th>passed</th>
                  <th>downloaded</th><th>smoked</th><th>error</th></tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={r.run_id ?? i}>
                  <td>{r.criteria}</td>
                  <td className="sp-md-dim">{fmtTs(r.started_at)}</td>
                  <td>{r.screened ?? '—'}</td>
                  <td>{r.passed ?? '—'}</td>
                  <td>{r.downloaded ?? '—'}</td>
                  <td>{r.smoked ?? '—'}</td>
                  <td className="sp-md-err">{r.error || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  )
}
