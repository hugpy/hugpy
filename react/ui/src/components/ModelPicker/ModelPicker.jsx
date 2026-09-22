import { useEffect, useMemo, useRef, useState } from 'react'
import './ModelPicker.css'

// ModelPicker — the load-model dropdown, grown into an informative picker.
//
// A native <select> can only show a flat label, so choosing a model to load
// meant reading bare names. This renders the SAME single-pick affordance as a
// button + popover with real columns (Model | Task | Lib | Ctx | Status) and a
// filter box — every fact is already on the /api/models rows the caller holds,
// so there are no extra fetches. Click a row to pick; Escape / outside click
// closes; Enter picks the first filtered row.
//
// Callers keep ownership of the row set (e.g. WorkersPanel passes its
// `assignable` filtering) and of what "pick" means (set state / add chip).

function fmtCtx(n) {
  if (n == null) return '—'
  const v = Number(n)
  if (!Number.isFinite(v) || v <= 0) return '—'
  return v >= 1024 ? `${Math.round(v / 1024)}k` : String(v)
}

function fmtBytes(n) {
  if (n == null) return '—'
  const v = Number(n)
  if (!Number.isFinite(v) || v <= 0) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0, x = v
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++ }
  return `${x >= 100 || i === 0 ? Math.round(x) : x.toFixed(1)} ${u[i]}`
}

// The size a picked model actually costs: GGUF -> the effective quant that
// serves (never the all-quants dir sum), everything else -> on-disk footprint.
// `size_bytes` is set for every model by the /models feed; keep effective_bytes
// as a fallback for an older feed.
function sizeOf(m) {
  const s = m.size_bytes != null ? m.size_bytes : m.effective_bytes
  return s != null ? Number(s) : null
}

function keyOf(m) {
  return m.model_key ?? m.key
}

export default function ModelPicker({
  models = [],
  value = '',
  onPick,
  placeholder = 'Pick a model…',
  disabled = false,
  autoFocus = false,
}) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const rootRef = useRef(null)
  const inputRef = useRef(null)

  const current = useMemo(
    () => models.find(m => keyOf(m) === value) || null,
    [models, value],
  )

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return models
    return models.filter(m => {
      const hay = `${m.name || ''} ${keyOf(m) || ''} ${(m.tasks || []).join(' ')} ${m.framework || ''}`.toLowerCase()
      return hay.includes(needle)
    })
  }, [models, q])

  // Outside click / Escape close.
  useEffect(() => {
    if (!open) return
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false)
    }
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  useEffect(() => { if (open) inputRef.current?.focus() }, [open])
  useEffect(() => { if (autoFocus && !disabled) setOpen(true) }, [autoFocus, disabled])

  const pick = (m) => {
    onPick?.(keyOf(m), m)
    setOpen(false)
    setQ('')
  }

  return (
    <span className="mp-root" ref={rootRef}>
      <button
        type="button"
        className={`mp-trigger${current ? '' : ' mp-empty'}`}
        disabled={disabled}
        onClick={() => setOpen(o => !o)}
        title={current ? keyOf(current) : placeholder}
      >
        <span className="mp-trigger-label">{current ? (current.name || keyOf(current)) : placeholder}</span>
        <span className="mp-caret">{open ? '▴' : '▾'}</span>
      </button>

      {open && (
        <div className="mp-pop" role="listbox">
          <input
            ref={inputRef}
            className="mp-filter"
            placeholder="filter by name / task / lib…"
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && filtered.length > 0) pick(filtered[0]) }}
          />
          <div className="mp-grid mp-head">
            <span>Model</span><span>Task</span><span>Lib</span><span>Ctx</span><span>Size</span><span>Status</span>
          </div>
          <div className="mp-rows">
            {filtered.length === 0 && <div className="mp-none">no models match</div>}
            {filtered.map(m => {
              const id = keyOf(m)
              const task = m.primary_task || (m.tasks || [])[0] || '—'
              const extraTasks = (m.tasks || []).filter(t => t !== task)
              return (
                <div
                  key={id}
                  className={`mp-grid mp-row${id === value ? ' mp-current' : ''}`}
                  role="option"
                  aria-selected={id === value}
                  onClick={() => pick(m)}
                  title={id}
                >
                  <span className="mp-name">{m.name || id}</span>
                  <span className="mp-task" title={(m.tasks || []).join(', ')}>
                    {task}
                    {extraTasks.length > 0 && <em className="mp-more"> +{extraTasks.length}</em>}
                  </span>
                  <span className={`mp-fw mp-fw-${m.framework || 'unknown'}`}>{m.framework || '—'}</span>
                  <span className="mp-ctx">{fmtCtx(m.model_max_length)}</span>
                  {(() => {
                    const sz = sizeOf(m)
                    return (
                      <span className={`mp-size${sz == null ? ' mp-size-none' : ''}`}
                            title={sz == null ? 'size unknown (not on disk)'
                              : m.framework === 'gguf'
                                ? `effective quant ${m.effective_gguf || ''} — ${fmtBytes(sz)}`
                                : `${fmtBytes(sz)} on disk`}>
                        {fmtBytes(sz)}
                      </span>
                    )
                  })()}
                  <span className={`mp-status mp-status-${m.status || 'unknown'}`}>{m.status || '—'}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </span>
  )
}
