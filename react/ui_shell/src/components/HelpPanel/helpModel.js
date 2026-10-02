// helpModel.js — PURE functions behind the console Help panel (no React, no
// DOM, no fetch), so they run under `node --test` (helpModel.test.mjs).
//
//   buildHelpContext(where)   -> { items: [{key,label,value}], payload: {...} }
//       The "context strip": where the operator is when they press Help —
//       console route + tab, selected model / worker, the last error the console
//       showed — plus anything a caller passed explicitly (a Help click next to
//       an error hands that error in as `error`). `payload` is what the server
//       receives as `context`; `items` is what the strip renders.
//
//   recordsToView(records)    -> [{ kind: 'user'|'agent'|'activity'|'note'|'error', ... }]
//       Folds the server transcript (append-only JSONL records) into bubbles:
//       text events stream into ONE agent bubble per turn, tool/tool_result
//       become activity lines, done closes the turn.
//
//   activityLine(record)      -> short human line for a tool event.

const MAX_ERR = 600

function clip(s, n) {
  const t = String(s == null ? '' : s).trim()
  return t.length > n ? t.slice(0, n) + '…' : t
}

function searchParams(search) {
  try { return new URLSearchParams(search || '') } catch { return new URLSearchParams('') }
}

export function buildHelpContext(where = {}) {
  const q = searchParams(where.search)
  const route = where.pathname || ''
  const tab = where.tab || q.get('tab') || ''
  const model = where.model || q.get('model') || ''
  const worker = where.worker || q.get('worker') || ''
  const error = where.error || where.lastError || ''
  const items = []
  const push = (key, label, value) => {
    if (value == null || value === '') return
    items.push({ key, label, value: String(value) })
  }
  push('route', 'route', route + (tab ? ` · ${tab}` : ''))
  push('model', 'model', model)
  push('worker', 'worker', worker)
  push('error', 'last error', error ? clip(error, MAX_ERR) : '')
  if (where.extra && typeof where.extra === 'object') {
    Object.entries(where.extra).forEach(([k, v]) => push(k, k, typeof v === 'string' ? v : JSON.stringify(v)))
  }
  const payload = {}
  items.forEach(i => { payload[i.key] = i.value })
  if (where.surface) payload.surface = where.surface
  return { items, payload }
}

export function activityLine(r) {
  if (!r) return ''
  if (r.type === 'tool') {
    const name = r.name || 'tool'
    const s = clip(r.summary || '', 160)
    return s ? `${name}: ${s}` : name
  }
  if (r.type === 'tool_result') {
    const first = clip(String(r.text || '').split('\n').find(l => l.trim()) || '', 140)
    return `${r.is_error ? '✗' : '✓'} ${first || (r.is_error ? 'error' : 'ok')}`
  }
  return ''
}

export function recordsToView(records = []) {
  const out = []
  let agent = null
  const closeAgent = () => { agent = null }
  for (const r of records) {
    const k = r.kind
    if (k === 'meta' || k === 'ref') continue
    if (k === 'backend') {
      out.push({ kind: 'note', i: r.i, text: `answered by ${r.label || r.backend}` })
      continue
    }
    if (k === 'user') {
      closeAgent()
      out.push({ kind: 'user', i: r.i, text: r.text || '', context: r.context || '' })
      continue
    }
    if (k === 'error' || k === 'pull_error') {
      closeAgent()
      out.push({ kind: 'error', i: r.i, text: r.error || 'error' })
      continue
    }
    if (k === 'stopped') {
      closeAgent()
      out.push({ kind: 'note', i: r.i, text: 'stopped by operator' })
      continue
    }
    if (k !== 'event') continue
    const t = r.type
    if (t === 'text') {
      if (!agent) { agent = { kind: 'agent', i: r.i, text: '' }; out.push(agent) }
      agent.text += r.text || ''
    } else if (t === 'tool' || t === 'tool_result') {
      out.push({ kind: 'activity', i: r.i, text: activityLine(r), error: !!r.is_error,
                 detail: t === 'tool' ? r.input : r.text })
    } else if (t === 'done') {
      if (r.error || (r.rc != null && r.rc !== 0)) {
        out.push({ kind: 'error', i: r.i, text: r.error || `turn failed (rc ${r.rc})` })
      } else if (!agent && r.result) {
        out.push({ kind: 'agent', i: r.i, text: r.result })
      }
      closeAgent()
    } else if (t === 'note' || t === 'error') {
      if (r.text) out.push({ kind: t === 'error' ? 'error' : 'note', i: r.i, text: r.text })
    }
  }
  return out
}
