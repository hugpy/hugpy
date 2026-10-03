// PER-MODEL ENVIRONMENTS (operator 2026-10-02). Profiles are DB records: pinned
// packages + a base ("worker" = an overlay on the worker venv, "isolated" = a
// fresh venv). Workers build the APPROVED profiles attributed to their models
// and report state + the frozen lock; hugpy-brain may propose profiles (they
// arrive as help tickets). Attribute a profile to a model with the model's
// env knob; test one here with a pinned one-token load.
import { useCallback, useEffect, useState } from 'react'
import { fetchJson } from '../../api'

const BASE = '/api/llm/env-profiles'

function StateChip({ w }) {
  const cls = w.state === 'ready' ? 'wp-env-ok' : w.state === 'error' ? 'wp-env-bad' : 'wp-env-wait'
  const lock = (w.lock || []).join('\n')
  return (
    <span className={`wp-env-chip ${cls}`}
          title={[`${w.worker_id}: ${w.state || 'not reported'}`, w.error || '', lock ? `lock:\n${lock}` : '',
                  w.test ? `test ${w.test.ok ? 'ok' : 'failed'} with ${w.test.model}${w.test.error ? `: ${w.test.error}` : ''}` : '']
            .filter(Boolean).join('\n')}>
      {w.worker_id.slice(0, 8)} {w.state || '—'}{w.test ? (w.test.ok ? ' ✓' : ' ✗') : ''}
    </span>
  )
}

export default function EnvProfilesPanel({ workers = [] }) {
  const [profiles, setProfiles] = useState(null)
  const [error, setError] = useState(null)
  const [form, setForm] = useState({ name: '', packages: '', base: 'worker', note: '' })
  const [busy, setBusy] = useState(false)
  const [testFor, setTestFor] = useState({})

  const load = useCallback(() => {
    fetchJson(BASE).then(d => { setProfiles(d.profiles || []); setError(null) }).catch(e => setError(e.message))
  }, [])
  useEffect(() => { load() }, [load])

  const post = async (url, body) => {
    setBusy(true); setError(null)
    try {
      await fetchJson(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      load()
      return true
    } catch (e) {
      setError(e.message)
      return false
    } finally {
      setBusy(false)
    }
  }

  const create = async () => {
    const packages = form.packages.split(/[\n,]+/).map(s => s.trim()).filter(Boolean)
    if (await post(BASE, { name: form.name.trim(), packages, base: form.base, note: form.note })) {
      setForm({ name: '', packages: '', base: 'worker', note: '' })
    }
  }

  return (
    <details className="wp-envs">
      <summary className="wp-setup-summary">
        🧪 Environments{profiles ? ` (${profiles.length})` : ''} — per-model Python envs (overlay or isolated), built on the workers that serve them
      </summary>
      {error && <div className="wp-tf-err">⚠ {error}</div>}
      <table className="wp-tfh-table">
        <thead><tr><th>name</th><th>packages</th><th>base</th><th>status</th><th>workers</th><th></th></tr></thead>
        <tbody>
          {(profiles || []).map(p => (
            <tr key={p.name}>
              <td title={p.note || ''}>{p.name}</td>
              <td className="wp-tfh-result" title={(p.packages || []).join('\n')}>{(p.packages || []).join(', ') || '—'}</td>
              <td>{p.base}</td>
              <td>{p.status === 'approved' ? 'approved' : <em>proposed{p.by_kind === 'agent' ? ' by hugpy-brain' : ''}</em>}</td>
              <td>{(p.workers || []).length ? p.workers.map(w => <StateChip key={w.worker_id} w={w} />) : <em>not built yet</em>}</td>
              <td>
                {p.status !== 'approved' && (
                  <button className="wp-test-fire" disabled={busy} onClick={() => post(`${BASE}/${encodeURIComponent(p.name)}/approve`)}>✓ approve</button>
                )}
                {p.status === 'approved' && (
                  <span className="wp-env-test">
                    <select value={testFor[p.name]?.worker || ''} onChange={e => setTestFor(t => ({ ...t, [p.name]: { ...t[p.name], worker: e.target.value } }))}>
                      <option value="">worker…</option>
                      {workers.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
                    </select>
                    <input placeholder="model key" value={testFor[p.name]?.model || ''}
                           onChange={e => setTestFor(t => ({ ...t, [p.name]: { ...t[p.name], model: e.target.value } }))} />
                    <button className="wp-test-fire" disabled={busy || !testFor[p.name]?.worker || !testFor[p.name]?.model}
                            title="one pinned one-token load of the model on that worker (in its attributed environment); the result shows on the worker chip"
                            onClick={() => post(`${BASE}/${encodeURIComponent(p.name)}/test`, testFor[p.name])}>▶ test</button>
                  </span>
                )}
              </td>
            </tr>
          ))}
          {profiles && !profiles.length && <tr><td colSpan={6}><em>no environments yet</em></td></tr>}
        </tbody>
      </table>
      <div className="wp-env-form">
        <input placeholder="name (e.g. ct-015)" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
        <textarea rows={2} placeholder={'pinned packages, one per line\ncompressed-tensors>=0.15.0'} value={form.packages}
                  onChange={e => setForm(f => ({ ...f, packages: e.target.value }))} />
        <select value={form.base} onChange={e => setForm(f => ({ ...f, base: e.target.value }))}
                title="worker = an overlay on the worker's own venv (keeps its torch; small); isolated = a fresh venv">
          <option value="worker">overlay on worker venv</option>
          <option value="isolated">isolated venv</option>
        </select>
        <input placeholder="note" value={form.note} onChange={e => setForm(f => ({ ...f, note: e.target.value }))} />
        <button className="wp-test-fire" disabled={busy || !form.name.trim()} onClick={create}>＋ save environment</button>
      </div>
    </details>
  )
}
