import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import './SettingsPanel.css'

// GEM RADAR (k120) — models NO card is asking about.
//
// Every saved search finds what the operator already knew to look for. The
// radar is the other half: a second pass over the SAME cached community pulls
// (reddit/HN listings the mention scan already paid for) looking for model
// names that keep coming up and that no card has ever screened.
//
// It is deliberately presented as a TIP, not a finding: nothing here has been
// VRAM-checked, licence-checked or loaded. `heat` is the recency-weighted
// mention score — the same number a real candidate's community section carries,
// so a radar row and a dossier row are on one scale.

export default function ModelRadar({ name }) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    fetchJson(`/api/llm/review/radar?criteria=${encodeURIComponent(name)}`)
      .then(setData)
      .catch(e => setErr(e.message))
  }, [name])

  if (err) return <div className="sp-empty">radar unavailable: {err}</div>
  if (!data) return <div className="sp-loading">loading radar…</div>

  const hits = data.hits || []
  return (
    <div className="sp-dos-radar">
      <div className="sp-md-dim">{data.detail || ''}</div>
      {hits.length === 0 && (
        <div className="sp-empty">
          Nothing on the radar. (It scans the cached mention pulls for models no
          card tracks; enable <code>radar</code> on this card to run it.)
        </div>
      )}
      {hits.length > 0 && (
        <table className="sp-md-table">
          <thead>
            <tr><th>model</th><th>heat</th><th>why</th></tr>
          </thead>
          <tbody>
            {hits.map((h, i) => (
              <tr key={i}>
                <td className="sp-md-model">
                  {h.hub_id
                    ? <a href={`https://huggingface.co/${h.hub_id}`}
                         target="_blank" rel="noreferrer">{h.hub_id}</a>
                    : <span title="named in posts but no hub repo matched">
                        {h.name}
                      </span>}
                </td>
                <td>{h.heat}</td>
                <td className="sp-md-dim">
                  {h.why}
                  {!!h.mentions?.length && (
                    <> ·{' '}
                      {h.mentions.slice(0, 3).map((m, j) => (
                        <a key={j} href={m.url} target="_blank" rel="noreferrer"
                           className="sp-dos-ref-link">{m.source}</a>
                      ))}
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
