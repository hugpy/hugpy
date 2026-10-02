import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchJson } from '../../api'
import ModelPicker from '../ModelPicker/ModelPicker'
import useSessionState from '../../hooks/useSessionState'
import './PriorityGroupsPanel.css'

// PRIORITY GROUPS — the operator's EXPLICIT, ORDERED model fallback list.
//
// A group is a hand-written list of model keys in priority order. When a
// request names ANY member, resolution walks the list top-down and serves the
// FIRST member that is usable right now (not blocked, in the catalog, and
// servable by some worker). Nothing is inferred: this is deliberately NOT the
// derived "model groups" feature (which strips -GGUF/quant suffixes and ranks
// members by a fit pipeline, and is off by default) — an ordering the operator
// did not write is not an ordering they consented to.
//
// A BLOCKED member is skipped, never served. The group only reorders which
// candidate is TRIED; it can never make a blocked model reachable. When nothing
// in the group is usable, the request fails exactly as it does today.
//
// The Resolve preview runs the SAME walk the serve path runs, so what it shows
// is what routing does — including the reason every loser lost.

const STATUS_ICON = {
  chosen: '✓',
  blocked: '⛔',
  missing: '∅',
  'no-worker': '○',
  'lower-priority': '↓',
}

// One reorderable member row inside the editor. A "group:<id>" member is a
// nested MODULE — rendered with the group glyph so the operator sees the
// mount, not a funny-looking model key.
function MemberRow({ mk, i, n, onMove, onRemove }) {
  const isRef = String(mk).toLowerCase().startsWith('group:')
  return (
    <li className={`pg-member${isRef ? ' pg-member-ref' : ''}`}>
      <span className="pg-rank" title="priority — 1 is tried first">{i + 1}</span>
      <span className="pg-member-key" title={mk}>
        {isRef ? <>▣ {String(mk).slice(6)} <em>(group)</em></> : mk}
      </span>
      <span className="pg-member-ctl">
        <button type="button" className="pg-arrow" disabled={i === 0}
                title="Move up (tried earlier)"
                onClick={() => onMove(i, i - 1)}>▲</button>
        <button type="button" className="pg-arrow" disabled={i === n - 1}
                title="Move down (tried later)"
                onClick={() => onMove(i, i + 1)}>▼</button>
        <button type="button" className="pg-remove" title="Remove from the group"
                onClick={() => onRemove(i)}>✕</button>
      </span>
    </li>
  )
}

// The create/edit form. `initial` null == a fresh group.
function GroupEditor({ initial, models, allGroups, busy, onSave, onCancel }) {
  const [name, setName] = useState(initial?.name || '')
  const [members, setMembers] = useState(initial?.members || [])
  const [workers, setWorkers] = useState(initial?.workers || [])
  const [enabled, setEnabled] = useState(initial ? !!initial.enabled : true)
  // The fleet roster, for the add-worker picker. Fetched once per editor open:
  // the group stores NAMES/ids, so a stale roster only limits the picker, never
  // corrupts the record.
  const [roster, setRoster] = useState([])

  useEffect(() => {
    fetchJson('/api/llm/workers')
      .then(rows => setRoster((rows || []).map(w => w.name || w.id).filter(Boolean)))
      .catch(() => setRoster([]))
  }, [])

  const reorder = useCallback(setter => (from, to) => {
    setter(prev => {
      if (to < 0 || to >= prev.length) return prev
      const next = prev.slice()
      const [row] = next.splice(from, 1)
      next.splice(to, 0, row)
      return next
    })
  }, [])

  const move = useMemo(() => reorder(setMembers), [reorder])
  const moveWorker = useMemo(() => reorder(setWorkers), [reorder])

  const remove = useCallback(i => {
    setMembers(prev => prev.filter((_, j) => j !== i))
  }, [])

  const removeWorker = useCallback(i => {
    setWorkers(prev => prev.filter((_, j) => j !== i))
  }, [])

  // ModelPicker hands back (model_key, row) — the key is what a group stores.
  const add = useCallback(mk => {
    if (!mk) return
    setMembers(prev => (prev.includes(mk) ? prev : [...prev, mk]))
  }, [])

  const addWorker = useCallback(w => {
    if (!w) return
    setWorkers(prev => (
      prev.some(p => p.toLowerCase() === w.toLowerCase()) ? prev : [...prev, w]))
  }, [])

  const ok = name.trim() && members.length > 0

  return (
    <div className="pg-editor">
      <div className="pg-row">
        <input className="pg-name" placeholder="Group name (e.g. Qwen2.5-VL-7B-Instruct)"
               value={name} onChange={e => setName(e.target.value)} />
        <label className="pg-enable" title="A disabled group routes nothing. Only enabled groups claim a model key.">
          <input type="checkbox" checked={enabled}
                 onChange={e => setEnabled(e.target.checked)} />
          enabled
        </label>
      </div>

      {members.length === 0 && (
        <div className="pg-hint">
          Add models in the order they should be tried. #1 serves whenever it is
          usable; the rest are fallbacks.
        </div>
      )}

      <ol className="pg-members">
        {members.map((mk, i) => (
          <MemberRow key={mk} mk={mk} i={i} n={members.length}
                     onMove={move} onRemove={remove} />
        ))}
      </ol>

      <div className="pg-row">
        <ModelPicker models={models} onPick={add} placeholder="+ add model…" />
        <select className="pg-worker-pick" value=""
                title="Mount another group here as a module: its own ordered members expand at this position, and it inherits this group's workers unless it has its own"
                onChange={e => { if (e.target.value) add(`group:${e.target.value}`); e.target.value = '' }}>
          <option value="">+ add group…</option>
          {(allGroups || [])
            .filter(g => g.id !== initial?.id &&
                         !members.some(m => String(m).toLowerCase() === `group:${g.id}`.toLowerCase()))
            .map(g => <option key={g.id} value={g.id}>▣ {g.name}</option>)}
        </select>
      </div>

      <div className="pg-workers-head" title="Where the group's models live, in priority order. #1 is preferred; members never land on a worker off this list. Empty = no placement statement (routing unchanged).">
        workers (allocation, priority order)
      </div>
      {workers.length === 0 && (
        <div className="pg-hint">
          Optional: allocate the group to workers. #1 is preferred; models in
          this group only land on listed workers. Leave empty to route as today.
        </div>
      )}
      <ol className="pg-members">
        {workers.map((w, i) => (
          <MemberRow key={w} mk={w} i={i} n={workers.length}
                     onMove={moveWorker} onRemove={removeWorker} />
        ))}
      </ol>

      <div className="pg-row">
        <select className="pg-worker-pick" value=""
                onChange={e => { addWorker(e.target.value); e.target.value = '' }}>
          <option value="">+ add worker…</option>
          {roster.filter(w => !workers.some(p => p.toLowerCase() === w.toLowerCase()))
                 .map(w => <option key={w} value={w}>{w}</option>)}
        </select>
        <span className="pg-spacer" />
        <button type="button" className="pg-cancel" onClick={onCancel} disabled={busy}>
          cancel
        </button>
        <button type="button" className="pg-save" disabled={!ok || busy}
                onClick={() => onSave({ id: initial?.id, name: name.trim(), members, workers, enabled })}>
          {busy ? 'saving…' : (initial ? 'save' : 'create group')}
        </button>
      </div>
    </div>
  )
}

// The resolve preview: the ordered walk + the WHY for every candidate.
function ResolvePreview({ result }) {
  if (!result) return null
  const rows = result.candidates || []
  return (
    <div className="pg-preview">
      <div className="pg-preview-head">
        <span className="pg-preview-key">{result.requested}</span>
        <span className="pg-preview-arrow">→</span>
        <span className={`pg-preview-chosen ${result.chosen ? '' : 'pg-none'}`}>
          {result.chosen || 'unchanged'}
        </span>
        {result.group && (
          <span className="pg-preview-group" title="the enabled priority group that claims this key">
            group {result.group.name}
          </span>
        )}
      </div>
      <div className="pg-why">{result.why}</div>
      {rows.length > 0 && (
        <ol className="pg-cands">
          {rows.map(c => (
            <li key={`${c.position}-${c.model_key}`} className={`pg-cand pg-st-${c.status}`}>
              <span className="pg-rank">{c.position}</span>
              <span className="pg-cand-icon" title={c.status}>
                {STATUS_ICON[c.status] || '·'}
              </span>
              <span className="pg-cand-key" title={c.catalog_key || c.model_key}>
                {c.model_key}
              </span>
              {c.via && (
                <span className="pg-cand-via" title={`expanded from the nested group ${c.via}`}>
                  via ▣ {c.via}
                </span>
              )}
              <span className="pg-cand-why">{c.reason}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

export default function PriorityGroupsPanel({ models = [] }) {
  const [open, setOpen] = useSessionState('hugpy.priorityGroups.open', false)
  const [groups, setGroups] = useState([])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState(null)   // group | 'new' | null
  const [probeKey, setProbeKey] = useState('')
  const [preview, setPreview] = useState(null)
  const [allocNote, setAllocNote] = useState(null)  // {id, text} — last allocate outcome

  const load = useCallback(() => {
    fetchJson('/api/llm/model-groups')
      .then(d => { setGroups(d.groups || []); setError(d.error || null) })
      .catch(e => setError(String(e.message || e)))
  }, [])

  useEffect(() => { if (open) load() }, [open, load])

  const save = useCallback(g => {
    setBusy(true); setError(null)
    const path = g.id ? `/api/llm/model-groups/${encodeURIComponent(g.id)}`
                      : '/api/llm/model-groups'
    fetchJson(path, {
      method: g.id ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: g.name, members: g.members,
                             workers: g.workers, enabled: g.enabled }),
    })
      .then(() => { setEditing(null); load() })
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  // One-call fan-out: designate every member to every worker in the group's
  // ordered list. Designation only — loading stays the Workers panel's explicit
  // per-model gesture with its preflight.
  const allocate = useCallback(g => {
    if (!(g.workers || []).length) return
    if (!window.confirm(
      `Allocate "${g.name}": designate ${g.members.length} model(s) to ` +
      `${g.workers.join(' → ')}?`)) return
    setBusy(true); setError(null)
    fetchJson(`/api/llm/model-groups/${encodeURIComponent(g.id)}/allocate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then(r => {
        const skips = (r.outcomes || []).filter(
          o => o.status !== 'designated' && o.status !== 'already')
        setAllocNote({
          id: g.id,
          text: `${r.designated} designated` +
            (skips.length ? `, ${skips.length} skipped (${
              [...new Set(skips.map(o => o.status))].join(', ')})` : ''),
        })
        load()
      })
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  const toggle = useCallback((g, enabled) => {
    setBusy(true); setError(null)
    fetchJson(`/api/llm/model-groups/${encodeURIComponent(g.id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    })
      .then(load)
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  const remove = useCallback(g => {
    if (!window.confirm(`Delete priority group "${g.name}"?`)) return
    setBusy(true); setError(null)
    fetchJson(`/api/llm/model-groups/${encodeURIComponent(g.id)}`, { method: 'DELETE' })
      .then(load)
      .catch(e => setError(String(e.message || e)))
      .finally(() => setBusy(false))
  }, [load])

  const probe = useCallback(key => {
    const k = (key || '').trim()
    if (!k) return
    setProbeKey(k)
    fetchJson(`/api/llm/model-groups/resolve?key=${encodeURIComponent(k)}`)
      .then(setPreview)
      .catch(e => setError(String(e.message || e)))
  }, [])

  const enabledCount = useMemo(
    () => groups.filter(g => g.enabled).length, [groups])

  return (
    <section className="priority-groups-panel">
      <div className="pg-bar" onClick={() => setOpen(o => !o)}>
        <span className="pg-title">Priority groups</span>
        <span className="pg-count">
          {groups.length ? `${groups.length} · ${enabledCount} enabled` : 'none'}
        </span>
        {error && <span className="pg-err">{error}</span>}
        <span className="pg-toggle">{open ? '▾' : '▸'}</span>
      </div>

      {open && (
        <div className="pg-body">
          <p className="pg-intro">
            An <strong>explicit, ordered</strong> fallback list. Calling any member
            serves the first member that is usable right now. A <strong>blocked</strong>{' '}
            model is skipped, never served — a group cannot bypass a block. If
            nothing in the group is usable the request fails exactly as it does
            without the group.
          </p>

          {groups.map(g => (
            <div key={g.id} className={`pg-group ${g.enabled ? '' : 'pg-off'}`}>
              <div className="pg-group-head">
                <span className="pg-group-name">{g.name}</span>
                <span className="pg-group-id">{g.id}</span>
                <span className="pg-spacer" />
                <label className="pg-enable" title="Disable to stop this group routing anything">
                  <input type="checkbox" checked={!!g.enabled} disabled={busy}
                         onChange={e => toggle(g, e.target.checked)} />
                  enabled
                </label>
                <button type="button" className="pg-edit" disabled={busy}
                        onClick={() => setEditing(editing?.id === g.id ? null : g)}>
                  edit
                </button>
                <button type="button" className="pg-del" disabled={busy}
                        onClick={() => remove(g)}>delete</button>
              </div>
              <ol className="pg-members pg-members-ro">
                {(g.members || []).map((mk, i) => {
                  const isRef = String(mk).toLowerCase().startsWith('group:')
                  return (
                    <li key={mk} className={`pg-member${isRef ? ' pg-member-ref' : ''}`}>
                      <span className="pg-rank">{i + 1}</span>
                      <span className="pg-member-key">
                        {isRef ? <>▣ {String(mk).slice(6)} <em>(group)</em></> : mk}
                      </span>
                      {!isRef && (
                        <button type="button" className="pg-probe"
                                title="Preview how this key resolves right now"
                                onClick={() => probe(mk)}>resolve</button>
                      )}
                    </li>
                  )
                })}
              </ol>
              {(g.workers || []).length > 0 && (
                <div className="pg-alloc-row"
                     title="The group's worker allocation, in priority order. Members only land on these workers.">
                  <span className="pg-alloc-label">workers:</span>
                  <span className="pg-alloc-workers">{g.workers.join(' → ')}</span>
                  <span className="pg-spacer" />
                  {allocNote?.id === g.id && (
                    <span className="pg-alloc-note">{allocNote.text}</span>
                  )}
                  <button type="button" className="pg-allocate" disabled={busy}
                          title="Designate every member to every listed worker (no load/warm)"
                          onClick={() => allocate(g)}>allocate</button>
                </div>
              )}
              {editing?.id === g.id && (
                <GroupEditor initial={g} models={models} allGroups={groups} busy={busy}
                             onSave={save} onCancel={() => setEditing(null)} />
              )}
            </div>
          ))}

          {editing === 'new'
            ? <GroupEditor initial={null} models={models} allGroups={groups} busy={busy}
                           onSave={save} onCancel={() => setEditing(null)} />
            : <button type="button" className="pg-new" disabled={busy}
                      onClick={() => setEditing('new')}>+ new priority group</button>}

          <div className="pg-resolve">
            <span className="pg-title">Resolve preview</span>
            <div className="pg-row">
              <input className="pg-probe-input" placeholder="model key…"
                     value={probeKey}
                     onChange={e => setProbeKey(e.target.value)}
                     onKeyDown={e => { if (e.key === 'Enter') probe(probeKey) }} />
              <button type="button" className="pg-probe" onClick={() => probe(probeKey)}>
                resolve
              </button>
            </div>
            <ResolvePreview result={preview} />
          </div>
        </div>
      )}
    </section>
  )
}
