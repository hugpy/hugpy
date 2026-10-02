import { fmtBytes } from './formatters'

export function SpillBadge({ spill }) {
  if (!spill || !spill.mode) return null
  const free = spill.free_vram_bytes
  const label = spill.mode === 'auto' ? 'autofit' : spill.mode
  return (
    <span className="wp-spill" title="GPU/CPU split mode reported by the worker">
      spill: {label}
      {free != null && <em> · {fmtBytes(free)} VRAM free</em>}
    </span>
  )
}
