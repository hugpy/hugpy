import { useCallback, useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import './SettingsPanel.css'

// MODEL DOSSIER — the expanded read on one discovered model (k120, 2026-08-21).
//
// The discovery leaderboard used to be model names and a score. The operator's
// note: "just model names… this needs to be comprehensive in nature… genuinely
// useful." So a leaderboard row now expands into this panel, which shows the
// eight sections the backend builds (abstract_hugpy_dev/discovery_dossier):
//
//   specialization  what it is FOR, with the per-domain weights and the
//                   author's own purpose sentences
//   weights         params / context / every quant with its OWN VRAM estimate
//   licence         + gating, as a badge, because a licence is a decision
//   research        the card digest (benchmark numbers marked CLAIMED, never
//                   measured), linked papers, and the model-written summary
//   community       reddit / HN / HF-discussion mentions, extracted claims with
//                   the quote and the link that support them, and heat
//   trial           the sample outputs and the score bar against the incumbent
//   verdict         adopt/trial/reject with the reasons AND the evidence refs
//
// TWO DISPLAY RULES, both load-bearing:
//   1. Model-generated text is always labelled. research_notes and community
//      claims are written by a local model reading sources; they render under
//      an explicit badge and never look like a measurement.
//   2. What could not be fetched is SHOWN, not hidden. `unavailable` is a list
//      the backend fills with honest causes ("arXiv timed out", "reddit 403")
//      and this panel prints it verbatim. A blank section and a failed fetch
//      must never look the same.

const GIB = 1024 ** 3
const gib = (b) => (b == null ? '—' : `${(b / GIB).toFixed(1)} GiB`)
const bn = (n) => {
  if (n == null) return '—'
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  return String(n)
}
const pct = (v) => (v == null ? '—' : `${Number(v).toFixed(1)}`)

function Badge({ tone = '', title, children }) {
  return <span className={`sp-dos-badge ${tone}`} title={title}>{children}</span>
}

function Section({ title, note, children }) {
  return (
    <div className="sp-dos-sec">
      <div className="sp-dos-sec-head">
        <span className="sp-dos-sec-title">{title}</span>
        {note && <span className="sp-md-dim">{note}</span>}
      </div>
      {children}
    </div>
  )
}

// The score-vs-incumbent bar. Two stacked tracks on ONE scale (0-100, the
// routing matrix's own quality scale) so the comparison is read, not computed.
function CompareBar({ row }) {
  const mine = row.candidate_quality
  const theirs = row.incumbent_quality
  const width = (v) => `${Math.max(0, Math.min(100, Number(v ?? 0)))}%`
  if (row.beats_incumbent === 'untested') {
    return (
      <div className="sp-dos-cmp">
        <div className="sp-dos-cmp-op">{row.operation}</div>
        <div className="sp-md-dim">untested — {row.basis}</div>
      </div>
    )
  }
  return (
    <div className="sp-dos-cmp">
      <div className="sp-dos-cmp-op">
        {row.operation}
        <Badge tone={row.beats_incumbent === 'yes' ? 'sp-dos-good' : 'sp-dos-bad'}>
          {row.beats_incumbent === 'yes' ? 'beats incumbent' : 'below incumbent'}
          {row.margin != null && ` ${row.margin > 0 ? '+' : ''}${row.margin}`}
        </Badge>
      </div>
      <div className="sp-dos-bar-row">
        <span className="sp-dos-bar-lbl">candidate</span>
        <div className="sp-dos-bar"><i style={{ width: width(mine) }} /></div>
        <span className="sp-dos-bar-num">{pct(mine)}</span>
      </div>
      <div className="sp-dos-bar-row">
        <span className="sp-dos-bar-lbl">{row.incumbent || 'incumbent'}</span>
        <div className="sp-dos-bar sp-dos-bar-inc"><i style={{ width: width(theirs) }} /></div>
        <span className="sp-dos-bar-num">{pct(theirs)}</span>
      </div>
      <div className="sp-md-dim sp-dos-basis">{row.basis}</div>
    </div>
  )
}

function Samples({ samples }) {
  if (!samples?.length) return null
  return (
    <div className="sp-dos-samples">
      {samples.map((s, i) => (
        <div key={i} className="sp-dos-sample">
          <div className="sp-dos-sample-head">
            <strong>{s.operation}</strong>
            <span className="sp-md-dim">{s.kind}</span>
            {s.seconds != null && <span className="sp-md-dim">{s.seconds}s</span>}
            {!s.ok && <Badge tone="sp-dos-bad">{s.failure || 'did not validate'}</Badge>}
          </div>
          {s.snippet && <pre className="sp-dos-snippet">{s.snippet}</pre>}
          {s.artifact_ref && (
            <div className="sp-md-dim sp-dos-artifact">artifact: <code>{s.artifact_ref}</code></div>
          )}
        </div>
      ))}
    </div>
  )
}

export default function ModelDossier({ criteria, hubId }) {
  const [doc, setDoc] = useState(null)
  const [err, setErr] = useState('')

  const load = useCallback(() => {
    setErr('')
    fetchJson(`/api/llm/review/dossier?criteria=${encodeURIComponent(criteria)}`
              + `&hub_id=${encodeURIComponent(hubId)}`)
      .then(setDoc)
      .catch(e => setErr(e.message))
  }, [criteria, hubId])

  useEffect(() => { load() }, [load])

  if (err) return <div className="sp-dos sp-empty">No dossier for this model yet ({err}).</div>
  if (!doc) return <div className="sp-dos sp-loading">loading dossier…</div>

  const { identity, specialization, weights, trust, research, community,
          trial, verdict, unavailable } = doc

  return (
    <div className="sp-dos">
      {/* ── verdict first: it is the answer, the rest is the working ── */}
      {verdict && (
        <Section title={`Verdict: ${verdict.verdict}`}
                 note={verdict.judged_by ? `judged by ${verdict.judged_by}`
                                         : 'filed from the measured numbers'}>
          <Badge tone={verdict.confidence === 'evidence-backed'
                       ? 'sp-dos-good' : 'sp-dos-warn'}>
            {verdict.confidence}
          </Badge>
          <ul className="sp-dos-reasons">
            {(verdict.reasons || []).map((r, i) => (
              <li key={i}>
                {r}
                {verdict.evidence_refs?.[i] && (
                  <code className="sp-dos-ref">{verdict.evidence_refs[i]}</code>
                )}
              </li>
            ))}
          </ul>
          {verdict.blocked && <div className="sp-note">{verdict.blocked}</div>}
        </Section>
      )}

      {/* ── specialization ── */}
      {specialization && (
        <Section title="Specialization">
          <div className="sp-dos-line">{specialization.headline || '—'}</div>
          {!!specialization.emphasis?.length && (
            <div className="sp-dos-weights">
              {specialization.emphasis.slice(0, 6).map((e, i) => (
                <span key={i} className="sp-dos-weight"
                      title={(e.evidence || []).join(' · ')}>
                  {e.domain}
                  <i style={{ width: `${Math.round(e.weight * 100)}%` }} />
                  <b>{e.weight}</b>
                </span>
              ))}
            </div>
          )}
          {!!specialization.finetune_focus?.length && (
            <ul className="sp-dos-quotes">
              {specialization.finetune_focus.map((q, i) => <li key={i}>“{q}”</li>)}
            </ul>
          )}
          {identity?.base_model && (
            <div className="sp-md-dim">
              lineage: {identity.relation || 'derived'} of{' '}
              <a href={`https://huggingface.co/${identity.base_model}`}
                 target="_blank" rel="noreferrer">{identity.base_model}</a>
            </div>
          )}
        </Section>
      )}

      {/* ── weights + licence ── */}
      {weights && (
        <Section title="Weights"
                 note={weights.params_source ? `params from ${weights.params_source}` : ''}>
          <div className="sp-dos-line">
            <Badge>{bn(weights.params)} params</Badge>
            <Badge>{weights.architecture_family || weights.architecture || 'arch ?'}</Badge>
            <Badge>ctx {weights.context_length ?? '—'}</Badge>
            {trust?.license && (
              <Badge tone="sp-dos-warn" title="licence as declared on the hub">
                {trust.license}
              </Badge>
            )}
            {trust?.gated ? <Badge tone="sp-dos-bad">gated</Badge> : null}
          </div>
          {!!weights.quants?.length && (
            <table className="sp-md-table">
              <thead><tr><th>quant</th><th>bpw</th><th>size</th><th>est VRAM</th><th>fits</th></tr></thead>
              <tbody>
                {weights.quants.map((q, i) => (
                  <tr key={i} className={q.quant === weights.best_quant ? 'sp-dos-best' : ''}>
                    <td>{q.quant}</td>
                    <td className="sp-md-dim">{q.bits_per_weight ?? '—'}</td>
                    <td>{gib(q.bytes)}</td>
                    <td>{gib(q.est_vram_bytes)}</td>
                    <td>{q.fits_vram == null ? '—' : (q.fits_vram ? 'yes' : 'no')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!!weights.notes?.length && (
            <div className="sp-md-dim">{weights.notes.join(' · ')}</div>
          )}
        </Section>
      )}

      {/* ── external research ── */}
      {research && (
        <Section title="Research (outside the download source)">
          {research.research_notes && (
            <>
              <Badge tone="sp-dos-gen">model-generated · {research.research_notes_model}</Badge>
              <p className="sp-dos-notes">{research.research_notes}</p>
            </>
          )}
          {!!research.card?.benchmark_claims?.length && (
            <div className="sp-dos-claims">
              <span className="sp-md-dim">
                claimed by the model card (not measured here):
              </span>
              {research.card.benchmark_claims.slice(0, 10).map((c, i) => (
                <Badge key={i}>{c.benchmark} {c.value}</Badge>
              ))}
            </div>
          )}
          {research.card?.limitations && (
            <details className="sp-dos-det">
              <summary>limitations (from the card)</summary>
              <pre className="sp-dos-snippet">{research.card.limitations}</pre>
            </details>
          )}
          {research.card?.training_data && (
            <details className="sp-dos-det">
              <summary>training data (from the card)</summary>
              <pre className="sp-dos-snippet">{research.card.training_data}</pre>
            </details>
          )}
          {!!research.papers?.length && (
            <ul className="sp-dos-links">
              {research.papers.map((p, i) => (
                <li key={i}>
                  <a href={p.url} target="_blank" rel="noreferrer">
                    {p.title || `arXiv:${p.arxiv_id}`}
                  </a>
                  {p.unavailable && <span className="sp-md-dim"> — {p.unavailable}</span>}
                </li>
              ))}
            </ul>
          )}
          {!!research.cited?.length && (
            <div className="sp-md-dim">sources: {research.cited.join(' · ')}</div>
          )}
        </Section>
      )}

      {/* ── community ── */}
      {community && (
        <Section title="Community" note={`heat ${community.heat}`}>
          {community.model_generated && (
            <Badge tone="sp-dos-gen">claims extracted by {community.generated_by}</Badge>
          )}
          {!!community.claims?.length && (
            <ul className="sp-dos-quotes">
              {community.claims.map((c, i) => (
                <li key={i}>
                  <b>{c.kind}</b>: {c.text}
                  {c.quote && <span className="sp-md-dim"> — “{c.quote}”</span>}
                  {c.url && <> <a href={c.url} target="_blank" rel="noreferrer">link</a></>}
                </li>
              ))}
            </ul>
          )}
          {!!community.mentions?.length && (
            <ul className="sp-dos-links">
              {community.mentions.slice(0, 8).map((m, i) => (
                <li key={i}>
                  <span className="sp-md-dim">{m.source}</span>{' '}
                  <a href={m.url} target="_blank" rel="noreferrer">{m.title || m.url}</a>
                </li>
              ))}
            </ul>
          )}
          {!community.mentions?.length && (
            <div className="sp-empty">Nobody has posted about this model on the
              sources we read.</div>
          )}
        </Section>
      )}

      {/* ── trial ── */}
      {trial && (
        <Section title="Trial" note={`${trial.depth}${trial.backend ? ` · ${trial.backend}` : ''}`}>
          {trial.blocked && <div className="sp-note">trial blocked: {trial.blocked}</div>}
          {trial.scenario_version && (
            <div className="sp-md-dim">
              stationary brief {trial.scenario_version} ({(trial.scenario_digest || '').slice(0, 12)})
              — every model is asked the same thing, which is what makes these
              numbers comparable.
            </div>
          )}
          {(trial.comparisons || []).map((c, i) => <CompareBar key={i} row={c} />)}
          <Samples samples={trial.samples} />
        </Section>
      )}

      {!!unavailable?.length && (
        <Section title="Not available">
          <ul className="sp-dos-quotes">
            {unavailable.map((u, i) => <li key={i} className="sp-md-dim">{u}</li>)}
          </ul>
        </Section>
      )}
    </div>
  )
}
