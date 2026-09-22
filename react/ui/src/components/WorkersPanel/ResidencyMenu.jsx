import { useEffect, useRef } from 'react'

// Residency picker (slice 8): clicking the residency tag NEVER changes
// anything — it opens this menu (same inline-expansion idiom as AllocControl).
// Only picking a DIFFERENT state fires a config change, so each ~5s agent
// restart is a deliberate choice.
//
// v3 final semantics (operator-locked): the POLICY axis has exactly two
// tiers — on-demand (the default; no stored override) and static (locked
// seat; permanent with 📌 pin). "Serving" is purely a STATE (a model in a
// slot) — the live pills (🔥 serving / ⚡ answering / ○ cold) tell that
// truth; no policy is ever called serving.
export const RESIDENCY_OPTIONS = [
  ['on-demand', '⏲ on-demand', 'loads on call; holds its slot until another model needs the seat (default)'],
  ['static', '🔒 static', 'always kept on this worker: downloaded eagerly, never evicted (the only tier that keeps files on disk)'],
]

export function ResidencyMenu({ mode, onPick, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose() }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [onClose])
  return (
    <div className="wp-res-menu" ref={ref} role="menu"
         onKeyDown={e => { if (e.key === 'Escape') onClose() }}>
      {RESIDENCY_OPTIONS.map(([value, label, desc]) => (
        <button key={value} role="menuitemradio" aria-checked={value === mode}
                className={`wp-res-opt${value === mode ? ' wp-res-opt-on' : ''}`}
                title={value === mode ? 'current state — click to close'
                  : 'applies via a ~5s agent restart'}
                onClick={() => (value === mode ? onClose() : onPick(value))}>
          <span className="wp-res-opt-label">{label}</span>
          <span className="wp-res-opt-desc">— {desc}</span>
          {value === mode && <span className="wp-res-opt-mark">✓</span>}
        </button>
      ))}
    </div>
  )
}
