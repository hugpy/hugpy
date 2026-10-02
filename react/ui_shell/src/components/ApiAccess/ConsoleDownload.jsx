import { useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import './ApiAccess.css'

// Fleet-console distribution (console regifted to hugpy, 2026-08-04).
// Same surface as the hugpy-agent install links, but these artifacts bake no
// key, so they are plain persistent downloads: the desktop console .deb and
// the hugpy_agent wheel it calls into (the two ship as a pair — see the
// handoff's install order). /agent/console/info names whatever versions are
// currently staged server-side; a newer drop appears here with no UI change.

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return '–'
  if (bytes > 1 << 20) return (bytes / (1 << 20)).toFixed(1) + ' MB'
  return Math.round(bytes / 1024) + ' KB'
}

function ArtifactRow({ label, info, hint }) {
  if (!info) {
    return (
      <div className="aa-row">
        <span className="aa-label">{label}</span>
        <span className="aa-note">none staged on this deployment</span>
      </div>
    )
  }
  return (
    <div className="aa-row">
      <span className="aa-label">{label}</span>
      <a className="aa-copy" href={info.url} download>
        download {info.filename} ({fmtSize(info.size_bytes)})
      </a>
      {info.sha256 && (
        <span className="aa-note" title={'sha256 ' + info.sha256}>
          sha256 {info.sha256.slice(0, 12)}…
        </span>
      )}
      {hint && <span className="aa-note">{hint}</span>}
    </div>
  )
}

export default function ConsoleDownload() {
  const [info, setInfo] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetchJson('/api/agent/console/info')
      .then(setInfo)
      .catch(e => setErr(e.message || String(e)))
  }, [])

  if (err) return null // nothing staged / route absent — stay out of the way

  return (
    <div className="aa-install-links">
      <div className="aa-row">
        <span className="aa-label">hugpy Station</span>
        <span className="aa-note">
          The operator&apos;s desktop console (Linux .deb) and the hugpy-agent
          wheel it calls into — install the wheel first, then the deb, then
          launch from the desktop entry (not the bare binary).
        </span>
      </div>
      <ArtifactRow label="console (.deb)" info={info?.deb}
                   hint="sudo apt install ./<file>" />
      <ArtifactRow label="agent (.whl)" info={info?.agent_whl}
                   hint="pip install --upgrade ./<file>" />
      {info?.install?.example && (
        <div className="aa-row">
          <span className="aa-label">one-liner</span>
          <code>{info.install.example}</code>
          <button
            className="aa-copy"
            onClick={() => navigator.clipboard?.writeText(info.install.example)}
          >
            copy
          </button>
          <span className="aa-note">
            Downloads + sha256-verifies + installs the newest staged artifact
            (deb/rpm/pacman/AppImage auto-detected). HUGPY_TOKEN takes an API
            key minted above — the installer provisions it into
            ~/.config/hugpy-station/station.env on the target — or an
            operator token (authorizes the download only, never persisted).
          </span>
        </div>
      )}
    </div>
  )
}
