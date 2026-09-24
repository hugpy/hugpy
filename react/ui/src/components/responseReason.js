// The reason a response body records for a failure, built only from that body:
// a string `error`, a typed `{error: {code, message}}`, `reason`, `detail` /
// `message`, else the body itself (JSON). Never a canned sentence.
export function responseReason(r) {
  if (r == null) return 'empty response body'
  if (typeof r === 'string') return r || 'empty response body'
  const e = r.error
  if (typeof e === 'string' && e.trim()) return e
  if (e && typeof e === 'object') {
    const t = [e.code, e.message].filter(x => x != null && String(x).trim()).join(': ')
    return t || JSON.stringify(e)
  }
  for (const k of ['reason', 'detail', 'message']) {
    if (typeof r[k] === 'string' && r[k].trim()) return r[k]
  }
  try { return `response body: ${JSON.stringify(r)}` } catch { return String(r) }
}
