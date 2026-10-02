// Fleet Console (desktop) — the product page for the Electron cockpit.
//
// BROCHURE-SAFE BY CONSTRUCTION: hugpy.ai serves this SPA with NO /api (the
// prod split — see prod/brochure/PROD-DEPLOY.md), and the stage.sh leak gate
// refuses any bundle carrying the keeper's identifiers. So everything below
// is static product truth, phrased against "your central" — and the live
// download section fetches the RELATIVE /api/agent/console/info, failing
// soft to the static story when there is no API (hugpy.ai) or no membership
// (anonymous console visitor). On a real console with a signed-in member it
// becomes the actual download surface.
import { useEffect, useState } from 'react'

export const FLEET_CONSOLE_TOC = [
  ['what', 'What it is'],
  ['get', 'Getting it'],
  ['inside', 'What runs inside'],
  ['workbench', 'Stations & workbenches'],
]

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return ''
  if (bytes > 1 << 20) return (bytes / (1 << 20)).toFixed(1) + ' MB'
  return Math.round(bytes / 1024) + ' KB'
}

export default function FleetConsolePage() {
  const [info, setInfo] = useState(null)   // null = no API / not entitled — stay static

  useEffect(() => {
    let alive = true
    fetch('/api/agent/console/info', { headers: { Accept: 'application/json' } })
      .then(r => (r.ok ? r.json() : null))
      .then(j => { if (alive && j?.deb) setInfo(j) })
      .catch(() => {})   // brochure / anonymous: the static page IS the product page
    return () => { alive = false }
  }, [])

  return (
    <div className="docs-page">
      <h1>hugpy Station — the desktop cockpit</h1>

      <h2 id="what">What it is</h2>
      <p>
        hugpy Station (formerly the Fleet Console) is hugpy&apos;s desktop application (Linux, Debian
        package): one window that holds your whole fleet. It talks to your
        hugpy central, lists your machines and VMs, opens terminals into any
        of them, seats coding agents next to your shells, and carries the
        full web console — chat, models, workers, media — as panes of the
        same cockpit. It is the operator&apos;s chair; the web console it
        embeds is the same one this documentation describes.
      </p>

      <h2 id="get">Getting it</h2>
      {info ? (
        <>
          <p>
            Your central is currently staging{' '}
            <strong>{info.deb.filename}</strong>
            {info.deb.size_bytes ? ` (${fmtSize(info.deb.size_bytes)})` : ''} —
            you are signed in with download access:
          </p>
          <ul>
            <li>
              <a href={info.deb.url} download>Download the .deb</a>
              {info.deb.sha256 && (
                <> — sha256 <code>{info.deb.sha256.slice(0, 16)}…</code></>
              )}, then <code>sudo apt install ./{info.deb.filename}</code>
            </li>
            {info.install?.example && (
              <li>
                Or the one-liner (downloads, verifies the sha256, installs):
                <pre><code>{info.install.example}</code></pre>
              </li>
            )}
          </ul>
          <p>
            The cleanest path for a NEW box is a <strong>one-time install
            link</strong> minted from the console&apos;s API tab (type:
            hugpy Station): its single command installs the deb <em>and</em>
            delivers a freshly minted, individually revocable API key straight
            into the app&apos;s credential file — the target box never handles
            a shared secret.
          </p>
        </>
      ) : (
        <>
          <p>
            The console is distributed through your hugpy central to signed-in
            members — it is part of the product, not a public artifact. Three
            equivalent routes, all served by your own central:
          </p>
          <ul>
            <li>
              <strong>Member download</strong> — sign in to your central&apos;s
              console and open the API tab (or the member fleet page): the
              current .deb and its sha256 are listed there.
            </li>
            <li>
              <strong>One-time install link</strong> — a console owner mints a
              single-use link whose one command installs the app <em>and</em>
              bakes in a fresh, individually revocable API key for that box.
            </li>
            <li>
              <strong>Token one-liner</strong> —
              <pre><code>{'curl -fsSL https://<your-central>/api/agent/console/install.sh | HUGPY_TOKEN=<token> bash'}</code></pre>
              downloads the current .deb from your central, verifies its
              sha256 against the staged sidecar, and installs via apt.
            </li>
          </ul>
          <p>
            Don&apos;t have a central yet? <code>pip install hugpy</code> and
            <code> hugpy serve</code> — the Installation page walks through
            it; the console then distributes itself from your own box.
          </p>
        </>
      )}

      <h2 id="inside">What runs inside</h2>
      <p>
        The app ships its own backend: the same console server the web UI
        uses, plus the hugpy agents that give every station a working seat —
        chat grounded on the machine it runs on, a keeper seat for the
        operator, and terminal surfaces for every VM. Those agents
        authenticate to your central with a console API key (the install-link
        flow above provisions it automatically; it lands in
        <code> ~/.fleet/console-hugpy.env</code> and is hot-reloaded — paste a
        new key there and the agents pick it up without a restart).
      </p>

      <h2 id="workbench">Stations &amp; workbenches</h2>
      <p>
        Each machine or VM the console manages is a <em>station</em>. A
        station can run its own embedded console (&quot;workbench mode&quot;)
        so its agents ground locally — your desktop app then acts as the
        harness that hands you from station to station, each with its own
        terminals, todo board, and agent seats. Creating a VM from the console
        provisions all of this automatically: a freshly built station boots as
        a full citizen of the fleet.
      </p>
    </div>
  )
}
