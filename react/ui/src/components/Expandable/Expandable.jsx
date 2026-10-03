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

// The model's record in the DB mirror (react/testshell, :7014; deep link ?q=).
// window.HUGPY_DB_UI overrides the base (e.g. a proxied URL off-LAN).
export function dbUiUrl(modelKey) {
  const base = (typeof window !== 'undefined' && window.HUGPY_DB_UI)
    || `${window.location.protocol}//${window.location.hostname}:7014`
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
