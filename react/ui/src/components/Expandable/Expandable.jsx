// EXPANDABLE (operator 2026-10-02, a GLOBAL rule): any text the UI truncates
// can be expanded in place — click to show it whole (wrapped), click again to
// fold. The full text is also the hover title. Use it wherever a cell ellipsises.
import { useState } from 'react'
import './Expandable.css'

export default function Expandable({ text, className = '', max = null }) {
  const [open, setOpen] = useState(false)
  const s = text == null ? '' : String(text)
  if (!s) return null
  return (
    <span className={`xp${open ? ' xp-open' : ''} ${className}`}
          style={!open && max ? { maxWidth: max } : undefined}
          title={open ? 'click to fold' : `${s}\n\n(click to expand)`}
          onClick={e => { e.stopPropagation(); setOpen(o => !o) }}>
      {s}
    </span>
  )
}

// The model's record in the DB dashboard: the styled per-model view (react/testshell)
// served under https://dev.hugpy.ai/modeldb/styled/ (LAN + WireGuard), one click from
// the raw table browser at /modeldb/. window.HUGPY_DB_UI overrides the base.
export function dbUiUrl(modelKey) {
  const base = (typeof window !== 'undefined' && window.HUGPY_DB_UI)
    || 'https://dev.hugpy.ai/modeldb/styled/'
  return `${base.replace(/\/$/, '')}/?q=${encodeURIComponent(modelKey)}`
}

export function ModelDbLink({ modelKey }) {
  if (!modelKey) return null
  return (
    <a className="xp-dblink" href={dbUiUrl(modelKey)} target="_blank" rel="noreferrer"
       title={`open ${modelKey} in the model database`} onClick={e => e.stopPropagation()}>
      {modelKey}
    </a>
  )
}
