import { useCallback, useEffect, useState } from 'react'
import { fetchJson } from '../../api'
import usePriorityGroups from '../../hooks/usePriorityGroups'
import './TemplatesPanel.css'

// TASK TEMPLATES — the operator's task blueprint over groups + workers.
//
// "Templates => worker groups && => module groups": a template names the
// module groups a task needs and the workers that host them. ACTIVATE
// reserves those workers (worker.pool = template id — general traffic can no
// longer land there; traffic tagged with the pool can) and allocates every
// group's expanded membership to them. DEACTIVATE releases the pool tags;
// designations stay (cheap registry state — reactivation is instant).

function TemplateEditor({ initial, groups, workers, busy, onSave, onCancel }) {
  const [name, setName] = useState(initial?.name || '')
  const [gids, setGids] = useState(initial?.groups || [])
  const [ws, setWs] = useState(initial?.workers || [])

  const toggleGroup = useCallback(gid => {
    setGids(prev => prev.includes(gid) ? prev.filter(g => g !== gid) : [...prev, gid])
  }, [])
  const addWorker = useCallback(w => {
    if (!w) return
    setWs(prev => prev.some(p => p.toLowerCase() === w.toLowerCase()) ? prev : [...prev, w])
  }, [])

  const ok = name.trim() && gids.length > 0
  return (
    <div className="tp-editor">
      <div className="tp-row">
        <input className="tp-name" placeholder="Template name (e.g. cinema night-shoot)"
               value={name} onChange={e => setName(e.target.value)} />
      </div>
      <div className="tp-pick-head">module groups (the cast)</div>
      <div className="tp-groups">
        {groups.map(g => (
          <label key={g.id} className="tp-group-pick">
            <input type="checkbox" checked={gids.includes(g.id)}
                   onChange={() => toggleGroup(g.id)} />
            ▣ {g.name}
          </label>
        ))}
      </div>
      <div className="tp-pick-head"
           title="Optional: leave empty to derive from the groups' own worker allocations at activation time">
        workers to reserve (optional — empty derives from the groups)
      </div>
      {ws.length > 0 && (
        <div className="tp-workers">
          {ws.map(w => (
            <span key={w} className="tp-worker-chip">
              {w}
              <button type="button" title="remove"
                      onClick={() => setWs(prev => prev.filter(p => p !== w))}>✕</button>
            </span>
          ))}
        </div>
      )}
      <div className="tp-row">
        <select className="tp-worker-pick" value=""
                onChange={e => { addWorker(e.target.value); e.target.value = '' }}>
          <option value="">+ add worker…</option>
          {workers.filter(w => !ws.some(p => p.toLowerCase() === (w.name || w.id).toLowerCase()))
                  .map(w => <option key={w.id} value={w.name || w.id}>{w.name || w.id}</option>)}
        </select>
        <span className="tp-spacer" />
        <button type="button" className="tp-cancel" onClick={onCancel} disabled={busy}>cancel</button>
        <button type="button" className="tp-save" disabled={!ok || busy}
                onClick={() => onSave({ id: initial?.id, name: name.trim(), groups: gids, workers: ws })}>
          {busy ? 'saving…' : (initial ? 'save' : 'create template')}
        </button>
      </div>
    </div>
  )
}

export default function TemplatesPanel({ workers = [] }) {
  const [open, setOpen] = useState(false)
  const [templates, setTemplates] = useState([])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState(null)   // template | 'new' | null
  const [note, setNote] = useState(null)         // {id, text}
  const { groups } = usePriorityGroups()

  const load = useCallback(() => {
    fetchJson('/api/llm/templates')
      .then(d => { setTemplates(d.templates || []); setError(d.error || null) })
      .catch(e => setError(String(e.message || e)))
  }, [])
  useEffect(() => { if (open) load() }, [open, load])

  const save = useCallback(t => {
    setBusy(true); setError(null)
    const path = t.id ? `/api/llm/templates/${encodeURIComponent(t.id)}` : '/api/llm/templates'
    fetchJson(path, {
      method: t.id ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: t.name, groups: t.groups, workers: t.workers }),
    })
      .then(() => { setEditing(null); load() })
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  const act = useCallback((t, verb) => {
    const doing = verb === 'activate'
    const who = (t.workers?.length ? t.workers : t.derived_workers) || []
    if (doing && !window.confirm(
      `Activate "${t.name}": reserve ${who.join(', ') || '(derived workers)'} for this ` +
      `template's pool and allocate ${t.groups.join(', ')}? Reserved workers stop ` +
      `serving general traffic until deactivated.`)) return
    setBusy(true); setError(null)
    fetchJson(`/api/llm/templates/${encodeURIComponent(t.id)}/${verb}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    })
      .then(r => {
        setNote({
          id: t.id,
          text: doing
            ? `reserved ${(r.reserved || []).join(', ')} · ${r.designated} designated`
            : `released ${(r.released || []).join(', ') || 'nothing (no tags held)'}`,
        })
        load()
      })
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  const remove = useCallback(t => {
    if (!window.confirm(`Delete template "${t.name}"?`)) return
    setBusy(true); setError(null)
    fetchJson(`/api/llm/templates/${encodeURIComponent(t.id)}`, { method: 'DELETE' })
      .then(load)
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  return (
    <section className="templates-panel">
      <div className="tp-bar" onClick={() => setOpen(o => !o)}>
        <span className="tp-title">Task templates</span>
        <span className="tp-count">
          {templates.length
            ? `${templates.length} · ${templates.filter(t => t.active).length} active`
            : 'none'}
        </span>
        {error && <span className="tp-err">{error}</span>}
        <span className="tp-toggle">{open ? '▾' : '▸'}</span>
      </div>

      {open && (
        <div className="tp-body">
          <p className="tp-intro">
            A template maps a TASK to its <strong>module groups</strong> (what
            models it needs, in their own priority orders) and the{' '}
            <strong>workers reserved</strong> for it. Activating pools those
            workers under the template id — general traffic can no longer land
            on them; requests tagged with the pool can — and allocates every
            group to them. Deactivating releases the reservation; designations
            stay so reactivation is instant.
          </p>

          {templates.map(t => (
            <div key={t.id} className={`tp-template ${t.active ? 'tp-active' : ''}`}>
              <div className="tp-template-head">
                <span className="tp-template-name">{t.name}</span>
                {t.active && <span className="tp-badge">● reserved</span>}
                <span className="tp-spacer" />
                {note?.id === t.id && <span className="tp-note">{note.text}</span>}
                <button type="button" disabled={busy}
                        onClick={() => act(t, t.active ? 'deactivate' : 'activate')}>
                  {t.active ? 'deactivate' : 'activate'}
                </button>
                <button type="button" disabled={busy}
                        onClick={() => setEditing(editing?.id === t.id ? null : t)}>edit</button>
                <button type="button" disabled={busy || t.active} className="tp-del"
                        title={t.active ? 'deactivate first' : 'delete'}
                        onClick={() => remove(t)}>delete</button>
              </div>
              <div className="tp-template-detail">
                <span>groups: {t.groups.map(g => `▣ ${g}`).join('  ')}</span>
                <span className="tp-workers-line">
                  workers: {(t.workers?.length ? t.workers : t.derived_workers)?.join(' → ') || '—'}
                  {!t.workers?.length && t.derived_workers?.length ? ' (derived)' : ''}
                </span>
              </div>
              {editing?.id === t.id && (
                <TemplateEditor initial={t} groups={groups} workers={workers}
                                busy={busy} onSave={save} onCancel={() => setEditing(null)} />
              )}
            </div>
          ))}

          {editing === 'new'
            ? <TemplateEditor initial={null} groups={groups} workers={workers}
                              busy={busy} onSave={save} onCancel={() => setEditing(null)} />
            : <button type="button" className="tp-new" disabled={busy}
                      onClick={() => setEditing('new')}>+ new template</button>}
        </div>
      )}
    </section>
  )
}
