// Pure normalisation of per-model failure payloads (ModelLogs.jsx); shape-
// agnostic because /llm/models/<key>/failures is still being added server-side.

export const tsOf = (v) => {
  if (v == null || v === '') return null
  const n = Number(v)
  if (Number.isFinite(n)) return n < 1e12 ? n * 1000 : n
  const d = Date.parse(v)
  return Number.isFinite(d) ? d : null
}

function textSource(r, d) {
  const row = r.id != null ? `compute_actions#${r.id} ` : ''
  for (const k of ['loader_stderr', 'message']) {
    if (r[k]) return `${row}${k}`.trim()
    if (d[k]) return `${row}detail.${k}`
  }
  for (const k of ['reason', 'why', 'error']) if (r[k]) return `${row}${k}`.trim()
  return `${row}detail.loader_stderr/message and row reason/why/error`.trim()
}

function normalize(r, kindHint) {
  const d = (r && typeof r.detail === 'object' && r.detail) || {}
  const predicate = r.predicate || r.failed_predicate || d.predicate
  const kind = r.kind || r.type || kindHint || (predicate ? 'routing refusal' : 'load failure')
  return {
    kind: String(kind).replace(/_/g, ' '),
    ts: tsOf(r.ts ?? r.at ?? r.time ?? r.created_at ?? r.timestamp),
    worker: r.worker || r.worker_name || d.worker || r.worker_card || '',
    cls: r.class || d.class || '',
    predicate: predicate || '',
    request_id: r.request_id || d.request_id || '',
    text: r.loader_stderr || d.loader_stderr || r.message || d.message || r.reason || r.why || r.error || '',
    message: (r.loader_stderr || d.loader_stderr) ? (r.message || d.message || '') : '',
    // where the text was read from + the server ref that serves it whole
    source: textSource(r, d),
    id: r.id ?? null,
    logRef: r.id != null ? `compute_actions#${r.id}` : (d.log_ref || r.log_ref || ''),
    fileLogRef: d.log_ref || r.log_ref || '',
    raw: r,
  }
}

export function normalizeFailures(payload) {
  const out = []
  if (Array.isArray(payload)) payload.forEach(r => r && typeof r === 'object' && out.push(normalize(r)))
  else if (payload && typeof payload === 'object') {
    for (const [k, v] of Object.entries(payload)) {
      if (!Array.isArray(v)) continue
      const hint = /refus|routing/i.test(k) ? 'routing refusal' : /load|action/i.test(k) ? 'load failure' : undefined
      v.forEach(r => r && typeof r === 'object' && out.push(normalize(r, hint)))
    }
  }
  return out.sort((a, b) => (b.ts || 0) - (a.ts || 0))
}

