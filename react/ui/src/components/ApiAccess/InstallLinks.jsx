import { useCallback, useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import './ApiAccess.css'

// Secure one-time install links (2026-07-23, operator-approved).
// Mint a labeled/scoped ONE-TIME download link for the hugpy-agent installer.
// The download bakes a freshly minted scoped key into the installer's
// EMBEDDED_API_KEY slot — the raw key is NEVER shown here (it exists only
// inside the download itself). Mint/list/revoke are operator-gated
// server-side (/agent/install-links).
//
// 2026-08-13: the surface now mints TWO kinds of link from one form —
// "hugpy-agent" (the historical installer flow, unchanged) and
// "fleet-console" (installs the console .deb AND delivers the minted key to
// ~/.fleet/console-hugpy.env — the credential the hugpy agents inside the
// installed console use). Console links are Linux-only (.deb) and their .sh
// fetch IS the consuming download. Both kinds share one ledger below; dead
// rows (used up / expired / revoked) can now be removed one-by-one
// (?purge=1) or swept in bulk (/prune) — the dead-weight cleanup.

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

const KIND_CHOICES = [
  { id: 'agent', label: 'hugpy-agent', hint: 'the hugpy-agent installer (all platforms)' },
  { id: 'console', label: 'hugpy Station', hint: 'the desktop Station .deb + its hugpy key (Linux)' },
]
const SCOPE_CHOICES = [
  { id: 'v1', hint: 'chat / models (/v1)' },
  { id: 'ml', hint: 'media intelligence (/ml)' },
  { id: 'agent-register', hint: 'fleet enroll (/agent/register)' },
  { id: 'full', hint: 'everything' },
]
const TTL_CHOICES = [
  { label: '1 hour', s: 3600 },
  { label: '24 hours', s: 86400 },
  { label: '7 days', s: 604800 },
]
const USES_CHOICES = [1, 3, 10]

export default function InstallLinks() {
  const [links, setLinks] = useState([])
  const [kind, setKind] = useState('agent')
  const [label, setLabel] = useState('')
  const [scopes, setScopes] = useState(['v1'])
  const [ttl, setTtl] = useState(86400)
  const [maxUses, setMaxUses] = useState(1)
  const [keyExpiry, setKeyExpiry] = useState('')   // optional datetime-local
  const [fresh, setFresh] = useState(null)         // the just-minted link
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(() => {
    fetchJson('/api/agent/install-links')
      .then(d => { setLinks(d.links ?? []); setError(null) })
      .catch(e => setError(e.message))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const toggleScope = useCallback((id) => {
    setScopes(prev => prev.includes(id)
      ? prev.filter(s => s !== id)
      : [...prev, id])
  }, [])

  const mint = useCallback(() => {
    if (!label.trim()) { alert('An install link needs a label.'); return }
    if (scopes.length === 0) { alert('Pick at least one scope.'); return }
    setBusy(true)
    const body = { label: label.trim(), scopes, link_ttl_s: ttl, max_uses: maxUses }
    if (keyExpiry) {
      const t = new Date(keyExpiry).getTime()
      if (Number.isFinite(t)) body.key_expires_at = t / 1000
    }
    // One form, two mints: the console kind has its own endpoint (and a
    // Linux-only `commands` map in the response — the render below follows
    // the map instead of assuming three platforms).
    const endpoint = kind === 'console'
      ? '/api/agent/console/install-links'
      : '/api/agent/install-links'
    fetchJson(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(l => { setFresh(l); setLabel(''); refresh() })
      .catch(e => alert(`Install-link mint failed: ${e.message}`))
      .finally(() => setBusy(false))
  }, [kind, label, scopes, ttl, maxUses, keyExpiry, refresh])

  const revoke = useCallback((l) => {
    if (!confirm(`Revoke install link "${l.label}"? Its key stops working too.`)) return
    fetchJson(`/api/agent/install-links/${l.link_id}`, { method: 'DELETE' })
      .then(refresh)
      .catch(e => alert(`Revoke failed: ${e.message}`))
  }, [refresh])

  // Remove = purge the ROW from the ledger (2026-08-13). For a dead link
  // that's pure cleanup (a delivered key keeps working — the server never
  // kills an exhausted link's key on purge); removing a still-ACTIVE link
  // also kills its undelivered key, so the confirm says so.
  const remove = useCallback((l) => {
    const warn = l.status === 'active'
      ? `Remove ACTIVE install link "${l.label}"? Its unused key dies with it.`
      : `Remove install link "${l.label}" from the list? (Already-installed machines keep working.)`
    if (!confirm(warn)) return
    fetchJson(`/api/agent/install-links/${l.link_id}?purge=1`, { method: 'DELETE' })
      .then(refresh)
      .catch(e => alert(`Remove failed: ${e.message}`))
  }, [refresh])

  const pruneDead = useCallback(() => {
    if (!confirm('Remove ALL used-up / expired / revoked links from the list? Active links are untouched.')) return
    fetchJson('/api/agent/install-links/prune', { method: 'POST' })
      .then(d => { refresh(); if (d?.pruned != null) alert(`${d.pruned} dead link${d.pruned === 1 ? '' : 's'} removed.`) })
      .catch(e => alert(`Prune failed: ${e.message}`))
  }, [refresh])

  // The mint response carries a ready-built `commands` map (2026-07-25) —
  // prefer it so no consumer hand-builds these strings. Fall back to
  // hand-built ones for an un-restarted central still on the old response
  // shape (the API change only takes effect after a restart), so the buttons
  // never blank out. linux/macos share the identical POSIX one-liner by
  // design — the `.sh` wrapper runs unmodified on both.
  const platformCommand = (l, plat) => {
    if (l?.commands?.[plat]) return l.commands[plat]
    if (plat === 'windows') return `irm ${l.url}.ps1 | iex`
    return `curl -fsSL ${l.url}.sh | bash`
  }

  // Same graceful-fallback shape as platformCommand, for the downloadable
  // archive (macOS double-click installer, 2026-07-25) — prefer the mint
  // response's `downloads` map, fall back to hand-building the one URL that
  // exists today for an un-restarted central still on the old response shape.
  // The .zip only: EVERY central can build one. The .pkg (tier 2) is read
  // straight from `downloads.macos_pkg` with NO fallback on purpose — it needs
  // mkbom/xar on the central host, so its absence from the map is the server
  // saying "I can't build one", and a hand-built URL would offer a button that
  // 501s.
  const downloadUrl = (l, key) => {
    if (l?.downloads?.[key]) return l.downloads[key]
    return `${l.url}.zip`
  }

  const freshIsConsole = (fresh?.kind ?? 'agent') === 'console'
  const deadCount = links.filter(l => l.status !== 'active').length

  return (
    <div className="aa-install-links">
      <div className="aa-row">
        <span className="aa-label">Install link</span>
        <select className="aa-select" value={kind} onChange={e => setKind(e.target.value)}
                title="What the link installs.">
          {KIND_CHOICES.map(k => <option key={k.id} value={k.id} title={k.hint}>{k.label}</option>)}
        </select>
        <input
          className="aa-name"
          placeholder="label (who / what box this is for)…"
          value={label}
          onChange={e => setLabel(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') mint() }}
        />
        <select className="aa-select" value={ttl} onChange={e => setTtl(Number(e.target.value))}
                title="How long the download link itself stays fetchable.">
          {TTL_CHOICES.map(t => <option key={t.s} value={t.s}>link: {t.label}</option>)}
        </select>
        <select className="aa-select" value={maxUses} onChange={e => setMaxUses(Number(e.target.value))}
                title="How many times the installer may be downloaded.">
          {USES_CHOICES.map(u => <option key={u} value={u}>{u} use{u > 1 ? 's' : ''}</option>)}
        </select>
        <button className="btn-primary" onClick={mint} disabled={busy}>
          {busy ? '…' : '+ Mint link'}
        </button>
      </div>

      <div className="aa-row">
        <span className="aa-label">Scopes</span>
        <div className="aa-scope-picks">
          {SCOPE_CHOICES.map(s => (
            <label key={s.id} className="aa-toggle" title={s.hint}>
              <input
                type="checkbox"
                checked={scopes.includes(s.id)}
                onChange={() => toggleScope(s.id)}
              />
              {s.id}
            </label>
          ))}
        </div>
        <label className="aa-toggle" title="Optional expiry for the KEY the link mints (the link's own TTL is separate).">
          key expires
          <input
            className="aa-name aa-key-expiry"
            type="datetime-local"
            value={keyExpiry}
            onChange={e => setKeyExpiry(e.target.value)}
          />
        </label>
        <span className="aa-note">
          {kind === 'console'
            ? 'Installs the hugpy Station .deb and writes a freshly minted key (with these ' +
              'scopes) to ~/.fleet/console-hugpy.env on the target box — the credential its ' +
              'hugpy agents use. The raw key is never shown here.'
            : 'The download bakes a freshly minted key (with these scopes) into the ' +
              'installer. The raw key is never shown — it exists only inside the ' +
              'one-time download.'}
        </span>
      </div>

      {error && <div className="aa-error">{error}</div>}

      {fresh && (
        <div className="aa-fresh">
          <div className="aa-fresh-head">
            <span>
              {freshIsConsole
                ? 'hugpy Station install link minted — run this on the target Linux box:'
                : 'Install link minted — hand this one-liner to the target box:'}
            </span>
            <button className="aa-fresh-dismiss" onClick={() => setFresh(null)}>✕</button>
          </div>
          <div className="aa-fresh-key">
            <span className="aa-label">Linux</span>
            <code>{platformCommand(fresh, 'linux')}</code>
            <CopyButton text={platformCommand(fresh, 'linux')} label="copy linux" />
          </div>
          {!freshIsConsole && (
            <div className="aa-fresh-key">
              <span className="aa-label">macOS</span>
              <code>{platformCommand(fresh, 'macos')}</code>
              <CopyButton text={platformCommand(fresh, 'macos')} label="copy macos" />
              <a className="aa-copy" href={downloadUrl(fresh, 'macos_zip')} download>
                download installer (.zip)
              </a>
              {fresh.downloads?.macos_pkg && (
                <a className="aa-copy" href={fresh.downloads.macos_pkg} download>
                  download installer (.pkg)
                </a>
              )}
            </div>
          )}
          {!freshIsConsole && (
            <div className="aa-fresh-key">
              <span className="aa-label">Windows</span>
              <code>{platformCommand(fresh, 'windows')}</code>
              <CopyButton text={platformCommand(fresh, 'windows')} label="copy windows" />
            </div>
          )}
          {freshIsConsole ? (
            <span className="aa-note">
              Fetching the command&apos;s URL consumes the link&apos;s use and delivers the key,
              so paste it straight into a terminal on the target box (sudo is used only for
              the apt install step). The script verifies the .deb&apos;s sha256 before
              installing.
            </span>
          ) : (
            <span className="aa-note">
              Paste the command into a terminal on the target box — don&apos;t download the script and
              double-click/run it (downloaded files aren&apos;t executable, and macOS quarantines them).
              On macOS you can instead use the zip download above: after extracting it, the first run
              needs right-click → Open (not double-click) since the installer is unsigned and Gatekeeper
              blocks a plain double-click once — after that first approval it runs normally.
              {fresh.downloads?.macos_pkg && (
                <> The .pkg gives an installer window instead of a terminal. Same unsigned first
                run (right-click → Open, or System Settings → Privacy &amp; Security → “Open Anyway”
                on macOS 15+), and because an installer shows no output it writes everything to
                ~/hugpy-agent/install.log — read that if anything goes wrong.</>
              )}
            </span>
          )}
          <span className="aa-note">
            {fresh.max_uses} download{fresh.max_uses > 1 ? 's' : ''} ·
            {' '}expires {fmtDate(fresh.expires_at)} · scopes: {fresh.scopes.join(', ')}
          </span>
        </div>
      )}

      {links.length > 0 && (
        <>
          <table className="aa-keys">
            <thead>
              <tr><th>Label</th><th>Kind</th><th>Scopes</th><th>Status</th><th>Uses</th><th>Created</th><th>Expires</th><th></th></tr>
            </thead>
            <tbody>
              {links.map(l => (
                <tr key={l.link_id}>
                  <td className="aa-key-name">{l.label}</td>
                  <td><span className="aa-chip">{(l.kind ?? 'agent') === 'console' ? 'fleet-console' : 'hugpy-agent'}</span></td>
                  <td>{(l.scopes ?? []).map(s => (
                    <span key={s} className="aa-chip">{s}</span>
                  ))}</td>
                  <td><span className={`aa-chip aa-status-${l.status}`}>{l.status}</span></td>
                  <td className="aa-dim">{(l.max_uses ?? 0) - (l.uses_left ?? 0)}/{l.max_uses}</td>
                  <td className="aa-dim">{fmtDate(l.created_at)}</td>
                  <td className="aa-dim">{fmtDate(l.expires_at)}</td>
                  <td className="aa-key-actions">
                    {l.status === 'active' && (
                      <button className="aa-revoke" onClick={() => revoke(l)}>revoke</button>
                    )}
                    <button className="aa-revoke" title="Remove this row from the ledger"
                            onClick={() => remove(l)}>remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {deadCount > 0 && (
            <div className="aa-row">
              <button className="aa-revoke" onClick={pruneDead}>
                clear {deadCount} dead link{deadCount === 1 ? '' : 's'}
              </button>
              <span className="aa-note">
                Removes every used-up / expired / revoked row above. Machines installed
                from a used-up link keep their key — this only cleans the list.
              </span>
            </div>
          )}
        </>
      )}
    </div>
  )
}
