// MODEL NOTES + FLAGS (operator 2026-10-02): a note and a few flags (broken,
// trash, archive, unknown, …) per model, written by the operator here or by
// hugpy-brain through the same route (by_kind "agent"). Stored in the DB
// (model_annotations + history); every model row shows its flags.
import { useEffect, useState, useSyncExternalStore } from 'react'
import { createPortal } from 'react-dom'
import { fetchJson } from '../../api'

const store = { notes: {}, flags: [], loaded: false, loading: false }
const listeners = new Set()
const emit = () => listeners.forEach(fn => fn())

export async function refreshNotes() {
  if (store.loading) return
  store.loading = true
  try {
    const d = await fetchJson('/api/models/notes')
    store.notes = d.notes || {}
    store.flags = d.flags || []
    store.loaded = true
  } catch { /* notes are additive: no DB, no chips */ }
  store.loading = false
  emit()
}

export function useModelNotes() {
  const snap = useSyncExternalStore(fn => { listeners.add(fn); return () => listeners.delete(fn) },
    () => store, () => store)
  useEffect(() => { if (!store.loaded) refreshNotes() }, [])
  return snap
}

const FLAG_ICON = { broken: '🛠', trash: '🗑', archive: '📦', unknown: '❓', 'needs-env': '🧪', experimental: '⚗', keep: '📌' }

export function ModelFlagChips({ modelKey }) {
  const { notes } = useModelNotes()
  const n = notes[modelKey]
  if (!n || !(n.flags || []).length) return null
  return (n.flags || []).map(f => (
    <span key={f} className={`wp-note-flag wp-note-flag-${f}`}
          title={`${f}${n.note ? `\n${n.note}` : ''}\n— ${n.updated_by || '?'} (${n.by_kind || 'operator'})`}>
      {FLAG_ICON[f] || '•'} {f}
    </span>
  ))
}

function NotesPopup({ modelKey, dbRef, onClose }) {
  const { notes, flags: vocab } = useModelNotes()
  const cur = notes[modelKey] || { flags: [], note: '' }
  const [flags, setFlags] = useState(new Set(cur.flags || []))
  const [note, setNote] = useState(cur.note || '')
  const [history, setHistory] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  useEffect(() => {
    fetchJson(`/api/models/database/${encodeURIComponent(dbRef)}/notes`)
      .then(d => setHistory(d.history || [])).catch(() => {})
  }, [dbRef])
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const save = async () => {
    setBusy(true); setError(null)
    try {
      await fetchJson(`/api/models/database/${encodeURIComponent(dbRef)}/notes`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flags: [...flags], note }),
      })
      await refreshNotes()
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }
  const toggle = f => setFlags(s => { const n = new Set(s); n.has(f) ? n.delete(f) : n.add(f); return n })
  return createPortal(
    <div className="wp-tfh-overlay" onClick={onClose}>
      <div className="wp-tfh-popup wp-notes-popup" role="dialog" aria-label={`notes for ${modelKey}`} onClick={e => e.stopPropagation()}>
        <div className="wp-tfh-popup-head">
          <span className="wp-tfh-popup-title">📝 {modelKey}</span>
          <span className="wp-tf-spacer" />
          <button className="wp-tf-toggle" onClick={onClose} title="close (Esc)">✕</button>
        </div>
        <div className="wp-tfh-popup-body">
          <div className="wp-notes-flags">
            {vocab.map(f => (
              <button key={f} type="button" className={`wp-note-flag wp-note-flag-${f}${flags.has(f) ? ' wp-note-flag-on' : ''}`}
                      aria-pressed={flags.has(f)} onClick={() => toggle(f)}>
                {FLAG_ICON[f] || '•'} {f}
              </button>
            ))}
          </div>
          <textarea className="wp-notes-text" rows={5} value={note} onChange={e => setNote(e.target.value)}
                    placeholder="What do you know about this model? (hugpy-brain reads and writes these too)" />
          <div className="wp-notes-actions">
            {error && <span className="wp-tf-err">⚠ {error}</span>}
            <span className="wp-tf-spacer" />
            <button type="button" className="wp-test-fire" disabled={busy} onClick={save}>{busy ? '…' : 'Save'}</button>
          </div>
          {history.length > 0 && (
            <>
              <div className="wp-tfh-sub">history</div>
              {history.map((h, i) => (
                <div key={i} className="wp-notes-hist">
                  <em>{new Date(h.at * 1000).toLocaleString()} · {h.by || '?'} ({h.by_kind || 'operator'})</em>
                  {(h.flags || []).length > 0 && <span> [{h.flags.join(', ')}]</span>}
                  {h.note && <div>{h.note}</div>}
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body)
}

export function ModelNotesButton({ modelKey, dbRef }) {
  const [open, setOpen] = useState(false)
  const { notes } = useModelNotes()
  const has = !!(notes[modelKey] && (notes[modelKey].note || (notes[modelKey].flags || []).length))
  return (
    <>
      <button type="button" className={`wp-notes-btn${has ? ' wp-notes-btn-has' : ''}`}
              title={has ? 'notes + flags — open' : 'add a note / flag (broken, trash, archive, …)'}
              onClick={e => { e.stopPropagation(); setOpen(true) }}>📝</button>
      {open && <NotesPopup modelKey={modelKey} dbRef={dbRef || modelKey} onClose={() => setOpen(false)} />}
    </>
  )
}
