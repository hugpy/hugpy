// One graded item, explicit: what was asked, what a correct response is, what
// the model said, and the two verdicts (correct in substance / obeyed the
// format).  Never a truncated blob — the whole response is shown.
export function verdictText(v) {
  return v === true ? '✓' : v === false ? '✗' : '—'
}

// The ledger relays EVERY key of the persisted call, so a field a view expects
// to be text (expected / why / judge.reason) can arrive as a nested object.
// asText() stringifies any object whole (pretty JSON) instead of handing React
// a raw object child (React #31, which blanks the whole console).
function asText(v) {
  if (v == null) return v
  if (typeof v === 'object') { try { return JSON.stringify(v, null, 1) } catch { return String(v) } }
  return String(v)
}

export function GradedItem({ item, heading }) {
  const correct = item.pass === true ? true : item.pass === false ? false : null
  const fmt = item.format === true ? true : item.format === false ? false : null
  const judge = item.judge || null
  return <div className={`graded-item graded-item-${correct === true ? 'pass' : correct === false ? 'fail' : 'none'}`}>
    <div className="graded-item-head">
      {heading && <b>{heading}</b>}
      <span className={`graded-verdict ${correct === true ? 'metrics-pass' : correct === false ? 'metrics-fail' : ''}`} title="did the response contain the right answer (substance only)">correct {verdictText(correct)}</span>
      <span className={`graded-verdict ${fmt === true ? 'metrics-pass' : fmt === false ? 'metrics-fail' : ''}`} title="did the response obey the output instruction (only the number / nothing else / exact string)">format {verdictText(fmt)}</span>
      {item.check_pass != null && <span className="graded-sub" title="the automatic rule check before the judge">auto check {verdictText(item.check_pass)}</span>}
      {item.revised && <span className="graded-sub graded-revised" title="the judge model overturned the automatic check">revised by judge</span>}
      {judge?.model && <span className="graded-sub">judge: {judge.model}</span>}
      {judge && !judge.model && judge.status === 'unavailable' && <span className="graded-sub">no judge: {asText(judge.reason)}</span>}
    </div>
    <dl className="graded-qa">
      <dt>Prompt</dt><dd><pre>{asText(item.prompt) || '(prompt not recorded)'}</pre></dd>
      <dt>Expected response</dt><dd><pre>{asText(item.expected_answer ?? item.expected) ?? '(not recorded)'}</pre>{item.expected_answer != null && item.expected && <span className="graded-sub">rule: {asText(item.expected)}</span>}</dd>
      <dt>Model response</dt><dd><pre>{item.actual == null || item.actual === '' ? '(no answer recorded)' : asText(item.actual)}</pre></dd>
      {(item.why || judge?.reason) && <><dt>Why</dt><dd>{asText(item.why || judge.reason)}</dd></>}
      {judge?.error && <><dt>Judge error</dt><dd>{asText(judge.error)}</dd></>}
    </dl>
  </div>
}

// Flatten a grade_detail ({task: {history: [...]}}) into ordered items.
export function itemsFromDetail(detail, suite) {
  const d = typeof detail === 'string' ? (() => { try { return JSON.parse(detail) } catch { return {} } })() : (detail || {})
  const out = []
  for (const task of suite?.tasks || Object.keys(d)) {
    const v = d[task]
    const hist = v && typeof v === 'object' && Array.isArray(v.history) ? v.history : []
    hist.forEach(h => { if (h) out.push({ ...h, task }) })
  }
  return out
}
