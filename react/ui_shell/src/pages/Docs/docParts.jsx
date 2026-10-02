// docParts — shared building blocks for the Docs pages (split out of Docs.jsx
// when the docs went user-first, 2026-07-05). Pure presentational helpers; no
// routing knowledge lives here.
import { useState } from 'react'

/* ---- tabbed code block ---- */
export function CodeTabs({ tabs }) {
  const [i, setI] = useState(0)
  return (
    <div className="docs-codetabs">
      <div className="docs-codetabs-bar">
        {tabs.map((t, n) => (
          <button key={t.label} className={n === i ? 'on' : ''} onClick={() => setI(n)}>{t.label}</button>
        ))}
        <span className="docs-codetabs-lang">{tabs[i].lang}</span>
      </div>
      <pre><code>{tabs[i].code}</code></pre>
    </div>
  )
}

// token-colored snippet helpers
export const C = (s) => <span className="tok-c">{s}</span> // comment
export const S = (s) => <span className="tok-s">{s}</span> // string
export const K = (s) => <span className="tok-k">{s}</span> // keyword/accent

/* ---- copyable plain-text block (e.g. an example .env) ---- */
export function CopyBlock({ title, code }) {
  const [copied, setCopied] = useState(false)
  const onCopy = () => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(code).catch(() => {})
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1600)
  }
  return (
    <div className="docs-copybox">
      <div className="docs-copybox-bar">
        <span className="docs-copybox-t">{title}</span>
        <button type="button" className={`docs-copybox-btn${copied ? ' copied' : ''}`} onClick={onCopy}>
          {copied ? 'Copied ✓' : 'Copy'}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  )
}
