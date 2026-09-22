import { useCallback, useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import { resolveApiOrigin } from '../../runtime/config'
import InstallLinks from './InstallLinks'
import ConsoleDownload from './ConsoleDownload'
import './ApiAccess.css'

function fmtDate(ts) {
  if (!ts) return '–'
  return new Date(ts * 1000).toLocaleString()
}

function CopyButton({ text, label = 'copy' }) {
  const [done, setDone] = useState(false)
  return (
    <button
      className="aa-copy"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true)
          setTimeout(() => setDone(false), 1500)
        })
      }}
    >
      {done ? '✓ copied' : label}
    </button>
  )
}

export default function ApiAccess({ models = [], embedded = false }) {
  const [open, setOpen] = useState(false)
  const [keys, setKeys] = useState([])
  const [requireKey, setRequireKey] = useState(false)
  const [mediaRequireKey, setMediaRequireKey] = useState(false)
  const [name, setName] = useState('')
  const [pool, setPool] = useState('')             // optional dedicated-pool binding
  const [freshKey, setFreshKey] = useState(null)   // full key, shown once
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  // Hugging Face credentials — status is read-only (token is never returned,
  // only last4); the input is write-only.
  const [hfAuth, setHfAuth] = useState(null)       // {authenticated, username, token_last4, source, note}
  const [hfToken, setHfToken] = useState('')
  const [hfBusy, setHfBusy] = useState(false)
  const [hfError, setHfError] = useState(null)

  const baseUrl = `${resolveApiOrigin()}/api/v1`
  const exampleModel =
    models.find(m => m.status === 'installed' &&
      ['text-generation', 'image-text-to-text'].includes(m.primary_task ?? m.task))?.model_key
    ?? models.find(m => m.status === 'installed')?.model_key
    ?? '<model_key>'

  const refresh = useCallback(() => {
    fetchJson('/api/keys')
      .then(d => { setKeys(d.keys ?? []); setRequireKey(!!d.require_key); setError(null) })
      .catch(e => setError(e.message))
    // The media-intelligence access gate is a separate flag (see /ml/gate).
    fetchJson('/api/ml/gate')
      .then(d => setMediaRequireKey(!!d.require_key))
      .catch(() => {})
    // Hugging Face credential status (validated against HF whoami server-side).
    fetchJson('/api/llm/hf/auth')
      .then(d => { setHfAuth(d); setHfError(null) })
      .catch(e => setHfError(e.message))
  }, [])

  useEffect(() => { if (embedded || open) refresh() }, [embedded, open, refresh])

  const createKey = useCallback(() => {
    setBusy(true)
    fetchJson('/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // `pool` binds the key to a dedicated worker pool — requests with this key
      // route to that pool's reserved workers automatically.
      body: JSON.stringify({ name, pool: pool.trim() }),
    })
      .then(k => { setFreshKey(k); setName(''); setPool(''); refresh() })
      .catch(e => alert(`Key creation failed: ${e.message}`))
      .finally(() => setBusy(false))
  }, [name, pool, refresh])

  const revokeKey = useCallback((k) => {
    if (!confirm(`Revoke key "${k.name}" (${k.prefix}…)? Calls using it will stop working.`)) return
    fetchJson(`/api/keys/${k.id}`, { method: 'DELETE' })
      .then(refresh)
      .catch(e => alert(`Revoke failed: ${e.message}`))
  }, [refresh])

  // Bulk-remove dated keys: expired always, plus (if the operator confirms the
  // age sweep) keys older than a cutoff. Never-used-only guards the age sweep so
  // a stale-but-live integration is not swept out by age alone.
  const pruneKeys = useCallback(() => {
    const expiredN = keys.filter(k => k.expired).length
    let body = { expired: true }
    let msg = expiredN
      ? `Remove ${expiredN} expired key${expiredN > 1 ? 's' : ''}?`
      : 'No expired keys.'
    const days = prompt(
      `${msg}\n\nAlso remove keys older than how many days? ` +
      `(blank = expired only; only never-used keys are swept by age)`, '')
    if (days === null) return                 // cancelled
    const n = parseFloat(days)
    if (!Number.isNaN(n) && n > 0) body = { ...body, older_than_days: n, unused_only: true }
    else if (!expiredN) return                // nothing to do
    fetchJson('/api/keys/prune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(d => { alert(`Removed ${d.count} key${d.count === 1 ? '' : 's'}.`); refresh() })
      .catch(e => alert(`Prune failed: ${e.message}`))
  }, [keys, refresh])

  const toggleRequire = useCallback(() => {
    const next = !requireKey
    fetchJson('/api/keys/require', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ require: next }),
    })
      .then(d => setRequireKey(!!d.require_key))
      .catch(e => alert(`Toggle failed: ${e.message}`))
  }, [requireKey])

  // Media-intelligence access gate — separate from the /v1 gate above and from
  // the console login. On ⇒ /ml/* inference requires a Bearer key.
  const toggleMediaGate = useCallback(() => {
    const next = !mediaRequireKey
    fetchJson('/api/ml/gate', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ require: next }),
    })
      .then(d => setMediaRequireKey(!!d.require_key))
      .catch(e => alert(`Media gate toggle failed: ${e.message}`))
  }, [mediaRequireKey])

  // Save an HF token: validated server-side against whoami BEFORE it is stored,
  // so an invalid token surfaces here and is never persisted.
  const saveHfToken = useCallback(() => {
    const tok = hfToken.trim()
    if (!tok) return
    setHfBusy(true)
    setHfError(null)
    fetchJson('/api/llm/hf/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: tok }),
    })
      .then(d => { setHfAuth(d); setHfToken('') })
      .catch(e => setHfError(e.message))
      .finally(() => setHfBusy(false))
  }, [hfToken])

  const clearHfToken = useCallback(() => {
    if (!confirm('Remove the saved Hugging Face token? HF calls go back to anonymous (rate-limited).')) return
    setHfBusy(true)
    setHfError(null)
    fetchJson('/api/llm/hf/auth', { method: 'DELETE' })
      .then(d => setHfAuth(d))
      .catch(e => setHfError(e.message))
      .finally(() => setHfBusy(false))
  }, [])

  const curl = [
    `curl ${baseUrl}/chat/completions \\`,
    `  -H "Content-Type: application/json" \\`,
    ...(requireKey ? [`  -H "Authorization: Bearer <your-key>" \\`] : []),
    `  -d '{"model": "${exampleModel}", "stream": true,`,
    `       "messages": [{"role": "user", "content": "Hello!"}]}'`,
  ].join('\n')

  return (
    <section className="api-access">
      {!embedded && (
      <div
        className="section-strip clickable"
        onClick={() => setOpen(v => !v)}
        title="Programmatic access to the local models (OpenAI-compatible)"
      >
        <span className="section-caret">{open ? '▾' : '▸'}</span>
        <span className="section-title">API access</span>
        {!open && <span className="aa-strip-hint">{baseUrl} · OpenAI-compatible</span>}
        <span className={`section-count ${requireKey ? 'aa-locked' : 'aa-open'}`}>
          {requireKey ? '🔒 key required' : 'open · no key needed'}
        </span>
      </div>
      )}

      {(embedded || open) && (
        <div className="aa-body">
          {error && <div className="aa-error">{error}</div>}

          <div className="aa-row aa-endpoint">
            <span className="aa-label">Endpoint</span>
            <code className="aa-url">{baseUrl}</code>
            <CopyButton text={baseUrl} />
            <span className="aa-note">
              OpenAI-compatible: <code>/chat/completions</code> &amp; <code>/models</code> —
              works with any OpenAI SDK via <code>base_url</code>.
            </span>
          </div>

          <div className="aa-row">
            <span className="aa-label">Auth</span>
            <label className="aa-toggle">
              <input type="checkbox" checked={requireKey} onChange={toggleRequire} />
              require an API key for /v1 calls
            </label>
            <span className="aa-note">
              {requireKey
                ? 'Calls must send Authorization: Bearer <key>.'
                : 'The API is open; keys are optional until you turn this on.'}
            </span>
          </div>

          <div className="aa-row">
            <span className="aa-label">New key</span>
            <input
              className="aa-name"
              placeholder="key name (e.g. laptop, notebook, cron)…"
              value={name}
              onChange={e => setName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') createKey() }}
            />
            <input
              className="aa-name"
              placeholder="pool (optional — dedicated worker)"
              title="Bind this key to a dedicated worker pool: its requests route to that pool's reserved workers."
              value={pool}
              onChange={e => setPool(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') createKey() }}
            />
            <button className="btn-primary" onClick={createKey} disabled={busy}>
              {busy ? '…' : '+ Create key'}
            </button>
          </div>

          <div className="aa-row aa-media-gate">
            <span className="aa-label">Media gate</span>
            <label className="aa-toggle">
              <input type="checkbox" checked={mediaRequireKey} onChange={toggleMediaGate} />
              require an API key for media-intelligence (/ml) access
            </label>
            <span className={`section-count ${mediaRequireKey ? 'aa-locked' : 'aa-open'}`}>
              {mediaRequireKey ? '🔒 key required' : 'open · no key needed'}
            </span>
            <span className="aa-note">
              {mediaRequireKey
                ? 'Media-intelligence /ml calls must send Authorization: Bearer <key>.'
                : 'Media-intelligence is open within the console; turn on to gate it.'}
              {' '}Separate from the console login and the /v1 gate above.
              {' '}Mint a key below to get a shareable <strong>live-demo link</strong> that
              opens the media UI against this backend.
            </span>
          </div>

          <div className="aa-row aa-hf-gate">
            <span className="aa-label">HF login</span>
            <div className="aa-hf-body">
              <div className="aa-hf-status">
                {hfAuth == null ? (
                  <span className="aa-note">Checking Hugging Face status…</span>
                ) : hfAuth.authenticated === true ? (
                  <span className="section-count aa-open">
                    ✓ Authenticated{hfAuth.username ? ` as ${hfAuth.username}` : ''}
                    {hfAuth.source === 'env' ? ' (env)' : ''}
                    {hfAuth.token_last4 ? ` · …${hfAuth.token_last4}` : ''}
                  </span>
                ) : hfAuth.authenticated === null ? (
                  <span className="section-count aa-locked">
                    Hugging Face unreachable — status unknown
                  </span>
                ) : hfAuth.token_last4 ? (
                  <span className="section-count aa-locked">
                    ⚠ Saved token rejected by Hugging Face{hfAuth.note ? ` (${hfAuth.note})` : ''}
                  </span>
                ) : (
                  <span className="section-count aa-locked">Anonymous — HF calls are rate-limited</span>
                )}
              </div>
              <div className="aa-hf-input">
                <input
                  className="aa-name"
                  type="password"
                  autoComplete="off"
                  placeholder="hf_… token (write-only, never shown again)"
                  value={hfToken}
                  onChange={e => setHfToken(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') saveHfToken() }}
                />
                <button className="btn-primary" onClick={saveHfToken} disabled={hfBusy || !hfToken.trim()}>
                  {hfBusy ? '…' : 'Save'}
                </button>
                {hfAuth && (hfAuth.source || hfAuth.token_last4) && (
                  <button className="aa-revoke" onClick={clearHfToken} disabled={hfBusy}>
                    clear
                  </button>
                )}
              </div>
              {hfError && <div className="aa-error">{hfError}</div>}
              <span className="aa-note">
                Save a Hugging Face token so model search, metadata, and downloads go out
                authenticated instead of anonymously rate-limited. Stored server-side (0600,
                never returned). A token is validated against HF before it is saved.
              </span>
            </div>
          </div>

          {freshKey && (
            <div className="aa-fresh">
              <div className="aa-fresh-head">
                <span>Key created — copy it now, it won't be shown again:</span>
                <button className="aa-fresh-dismiss" onClick={() => setFreshKey(null)}>✕</button>
              </div>
              <div className="aa-fresh-key">
                <code>{freshKey.key}</code>
                <CopyButton text={freshKey.key} label="copy key" />
              </div>
              {/* Shareable live-demo link: opens the media-intelligence front end
                  in live demo mode, authenticated with this key (the arm injects
                  Authorization: Bearer <key> on every call). Built here because
                  the full key only exists at mint time. */}
              <div className="aa-fresh-demo">
                <span className="aa-note">
                  Live-demo link — opens the media UI against this backend, keyed:
                </span>
                <div className="aa-fresh-key">
                  <code>{`${resolveApiOrigin()}/media/?demo=live&key=${freshKey.key}`}</code>
                  <CopyButton
                    text={`${resolveApiOrigin()}/media/?demo=live&key=${freshKey.key}`}
                    label="copy demo link"
                  />
                </div>
              </div>
              {/* Uniform station install one-liner (distribution normalization,
                  2026-08-20): the fresh key IS the install credential — the
                  installer downloads + sha256-verifies + installs the station,
                  then persists this key to ~/.config/hugpy-station/station.env
                  (0600) so the install is attributed to it. Same contract as
                  the hugpy-agent installer. Built here because the full key
                  only exists at mint time. */}
              <div className="aa-fresh-demo">
                <span className="aa-note">
                  Station install one-liner — installs hugpy Station on any Linux
                  box and provisions this key into it:
                </span>
                <div className="aa-fresh-key">
                  <code>{`curl -fsSL ${resolveApiOrigin()}/api/agent/console/install.sh | HUGPY_TOKEN=${freshKey.key} bash`}</code>
                  <CopyButton
                    text={`curl -fsSL ${resolveApiOrigin()}/api/agent/console/install.sh | HUGPY_TOKEN=${freshKey.key} bash`}
                    label="copy install cmd"
                  />
                </div>
              </div>
            </div>
          )}

          {keys.length > 0 && (
            <div className="aa-keys-toolbar">
              <button className="aa-prune" onClick={pruneKeys}
                      title="Revoke expired keys (and optionally keys older than a cutoff).">
                Remove dated keys
                {keys.some(k => k.expired) &&
                  <span className="aa-chip aa-chip-expired">{keys.filter(k => k.expired).length} expired</span>}
              </button>
            </div>
          )}

          {keys.length > 0 && (
            <table className="aa-keys">
              <thead>
                <tr><th>Name</th><th>Label</th><th>Scopes</th><th>Key</th><th>Created</th><th>Last used</th><th></th></tr>
              </thead>
              <tbody>
                {keys.map(k => (
                  <tr key={k.id}>
                    <td className="aa-key-name">
                      {k.name}
                      {k.created_by === 'install-link' && (
                        <span className="aa-chip aa-chip-install" title="Minted by a one-time install link — never a standing operator key.">install-link</span>
                      )}
                    </td>
                    <td className="aa-dim">{k.label || '–'}</td>
                    <td>{(k.scopes ?? ['full']).map(s => (
                      <span key={s} className="aa-chip">{s}</span>
                    ))}</td>
                    <td><code>{k.prefix}…</code></td>
                    <td className="aa-dim">
                      {fmtDate(k.created_at)}
                      {k.expired && <span className="aa-chip aa-chip-expired">expired</span>}
                    </td>
                    <td className="aa-dim">{fmtDate(k.last_used)}</td>
                    <td className="aa-key-actions">
                      <button className="aa-revoke" onClick={() => revokeKey(k)}>revoke</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Secure one-time install links — the hugpy-agent installer with a
              freshly minted scoped key baked into its EMBEDDED_API_KEY slot.
              Operator-gated server-side (/agent/install-links). */}
          <div className="aa-row aa-install-head">
            <span className="aa-label">Install links</span>
            <span className="aa-note">
              Secure one-time download links for the hugpy-agent installer —
              each link mints its own labeled, scoped, revocable key.
            </span>
          </div>
          <InstallLinks />
          <ConsoleDownload />

          <details className="aa-example">
            <summary>curl example</summary>
            <pre><code>{curl}</code></pre>
            <CopyButton text={curl} label="copy example" />
          </details>
        </div>
      )}
    </section>
  )
}
