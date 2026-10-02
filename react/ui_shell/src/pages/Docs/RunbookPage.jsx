// Fleet remediation runbook — the operator+agent playbook for image-gen fleet
// failures (VRAM refusals, corrupt checkpoints, missing deps, routing, etc.).
//
// SINGLE SOURCE OF TRUTH: this page renders LIVE from GET /api/fleet/runbook,
// which serves abstract_hugpy_dev/fleet_runbook.json — the SAME file the
// hugpy-agent reads to auto-remediate. Editing the runbook = edit that JSON
// only; this page and the agent both follow it. Nothing here is hand-copied.
import { useEffect, useState } from 'react'

export const RUNBOOK_TOC = [
  ['overview', 'Overview', []],
  ['ops', 'Worker /ops endpoints', []],
  ['remediations', 'Remediations', []],
]

export default function RunbookPage() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let alive = true
    fetch('/api/fleet/runbook', { headers: { Accept: 'application/json' } })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => alive && setData(j))
      .catch((e) => alive && setErr(String(e)))
    return () => {
      alive = false
    }
  }, [])

  return (
    <article className="docs-article">
      <h1>Fleet remediation runbook</h1>
      <section id="overview" className="docs-sec">
        <p>
          The single source of truth for image-generation fleet failures. This page
          renders live from <code>GET /api/fleet/runbook</code>, which serves{' '}
          <code>abstract_hugpy_dev/fleet_runbook.json</code> — the <strong>same file
          the hugpy-agent reads</strong> to auto-remediate. Edit that JSON only; this
          page and the agent both follow it.
        </p>
        {err && (
          <p className="vi-error" role="alert">
            Couldn’t load the runbook: {err}
          </p>
        )}
        {data && (
          <p>
            <strong>v{data.version}</strong> · updated {data.updated} ·{' '}
            {(data.remediations || []).length} remediations
          </p>
        )}
      </section>

      <section id="ops" className="docs-sec">
        <h2 className="docs-h2">Worker /ops endpoints</h2>
        <p>
          The agent (and an operator) executes remediations through the worker’s ops
          API — no arbitrary shell needed:
        </p>
        {data && data.ops_endpoints && (
          <ul>
            {Object.entries(data.ops_endpoints).map(([k, v]) => (
              <li key={k}>
                <code>{k}</code> — <code>{v}</code>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section id="remediations" className="docs-sec">
        <h2 className="docs-h2">Remediations</h2>
        {!data && !err && <p>Loading…</p>}
        {data &&
          (data.remediations || []).map((r) => (
            <div key={r.id} id={r.id} className="docs-h3-block">
              <h3 className="docs-h3">{r.id}</h3>
              <p>
                <strong>Signature:</strong> <code>{r.signature}</code>
              </p>
              {r.cause && (
                <p>
                  <strong>Cause:</strong> {r.cause}
                </p>
              )}
              {r.detect && (
                <p>
                  <strong>Detect:</strong> {r.detect}
                </p>
              )}
              <p>
                <strong>Remedy:</strong> {r.remedy}
              </p>
              {r.automated_by && (
                <p>
                  <strong>Automated by:</strong> {r.automated_by}
                </p>
              )}
              {r.agent_action && (
                <p>
                  <strong>Agent action:</strong>{' '}
                  <code>{JSON.stringify(r.agent_action)}</code>
                </p>
              )}
            </div>
          ))}
      </section>
    </article>
  )
}
