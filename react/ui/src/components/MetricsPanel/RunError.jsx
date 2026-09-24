// A benchmark run's error, whatever shape the server recorded: a plain string
// (older central) or the structured summary {headline, planned, runnable,
// requested_models, elapsed_s, failure_classes:{cls:{count, models, first_reason, evidence}}}.
// Every part is shown; nothing is reduced to a canned sentence.
export function runErrorText(err) {
  if (!err) return ''
  if (typeof err === 'string') return err
  return err.headline || JSON.stringify(err)
}

export default function RunError({ error, className = 'metrics-error' }) {
  if (!error) return null
  if (typeof error === 'string') return <p className={className}>{error}</p>
  const classes = Object.entries(error.failure_classes || {})
  return <div className={`${className} run-error`}>
    <p><b>{error.headline}</b></p>
    <ul className="run-error-list">
      {error.requested_models?.length > 0 && <li>requested: {error.requested_models.join(', ')}</li>}
      <li>lanes planned: {error.planned ?? 0} · runnable: {error.runnable ?? 0}{error.elapsed_s != null ? ` · ${error.elapsed_s} s spent` : ''}</li>
      {classes.map(([cls, v]) => <li key={cls}>
        <b>{cls}</b> ×{v.count}{v.models?.length ? ` (${v.models.join(', ')})` : ''}: {v.first_reason || '(no reason recorded)'}
        {v.evidence && <details className="metrics-call-detail"><summary>evidence</summary><pre>{JSON.stringify(v.evidence, null, 1)}</pre></details>}
      </li>)}
    </ul>
  </div>
}
