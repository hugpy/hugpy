import { useEffect, useMemo, useState } from 'react'

const REFRESH_MS = 5000

// The raw table browser (abstract_database hugpy_model_viewer): its parent dir when
// this app is served under /modeldb/styled/, else the published site.
const RAW_DB = window.location.pathname.includes('/modeldb/') ? '../' : 'https://dev.hugpy.ai/modeldb/'
export const rawHref = (params = '') => RAW_DB + (params ? `?${params}` : '')

async function api(path, opts) {
  const r = await fetch(path, { cache: 'no-store', ...opts })
  const body = await r.text()
  let data; try { data = JSON.parse(body) } catch { throw new Error(body.slice(0, 300) || `HTTP ${r.status}`) }
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`)
  return data
}

export function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [q, setQ] = useState(() => new URLSearchParams(window.location.search).get('q') || '')  // ?q=<model> deep link (console links, 2026-10-02)
  const [allocatedOnly, setAllocatedOnly] = useState(false)
  const [tick, setTick] = useState(0)
  useEffect(() => {  // keep ?q= in the address bar so the view can be shared / reloaded
    const p = new URLSearchParams(window.location.search)
    if (q) p.set('q', q); else p.delete('q')
    const s = p.toString()
    window.history.replaceState(null, '', s ? `?${s}` : window.location.pathname)
  }, [q])

  const load = async () => {
    try { setData(await api('api/models')); setError(null); setTick(Date.now()) }
    catch (e) { setError(e.message) }
  }
  useEffect(() => { load(); const t = setInterval(load, REFRESH_MS); return () => clearInterval(t) }, [])

  // a knob write / recompute answers with the fresh row: splice it in, no full reload
  const patchModel = m => setData(d => d ? { ...d, models: d.models.map(x => x.id === m.id ? m : x) } : d)

  // TEST PAGE (operator 2026-10-02): ?ids=92,78,94,251 pins the page to the
  // feasibility test set — Anko (GGUF MoE), Qwen3.6-35B-A3B (safetensors
  // hybrid MoE, 4-bit), MN-GRAND (dense 23.5B: safetensors + GGUF).
  const onlyIds = useMemo(() => {
    const raw = new URLSearchParams(window.location.search).get('ids')
    return raw ? new Set(raw.split(',').map(x => Number(x.trim())).filter(Number.isFinite)) : null
  }, [])
  const models = useMemo(() => {
    const rows = data?.models || []
    const needle = q.trim().toLowerCase()
    return rows.filter(m => {
      if (onlyIds && !onlyIds.has(Number(m.id))) return false
      if (allocatedOnly && !(m.workers?.length)) return false
      if (!needle) return true
      const hay = [m.name, m.hub_id, m.framework, ...(m.workers || []).map(w => w.worker)].join(' ').toLowerCase()
      return hay.includes(needle)
    })
  }, [data, q, allocatedOnly, onlyIds])

  const ens = data?.ensure
  return (
    <div className="shell">
      <header>
        <h1>hugpy testshell {onlyIds ? <span className="tag ok">test page · {[...onlyIds].join(', ')}</span> : null} <span className="muted">· weights → per-worker plan → live · knobs write the DB</span>
          {' '}<a href="?ids=92,78,94,251" className="small" title="Anko (GGUF MoE) · Qwen3.6-35B-A3B (safetensors hybrid MoE, 4-bit) · MN-GRAND 23.5B dense (safetensors + GGUF)">test set</a>
          {onlyIds ? <>{' '}<a href="?" className="small">all</a></> : null}
          {' '}<a href={rawHref(q ? `table=model_full&q=${encodeURIComponent(q)}` : 'table=model_full')} className="switch" title="Raw table browser — every table, related rows, JSON">Raw DB ↗</a></h1>
        <div className="controls">
          <input placeholder="filter by model, hub id, framework, worker…" value={q} onChange={e => setQ(e.target.value)} />
          <label><input type="checkbox" checked={allocatedOnly} onChange={e => setAllocatedOnly(e.target.checked)} /> allocated only</label>
          <span className="muted">{models.length}/{data?.count ?? '…'} models · refreshed {tick ? new Date(tick).toLocaleTimeString() : '…'}</span>
          {ens && <span className="muted small" title={JSON.stringify(ens.errors)}>compute-once: {ens.running ? 'running' : 'idle'} · {ens.specs} specs, {ens.plans} plans written{Object.keys(ens.errors || {}).length ? ` · ${Object.keys(ens.errors).length} errors` : ''}</span>}
        </div>
        {error && <div className="error">API error: {error}</div>}
        <WorkerStrip workers={data?.workers || []} />
      </header>
      {models.map(m => <ModelTable key={m.id} m={m} workers={data?.workers || []} allocModes={data?.alloc_modes || []} onPatch={patchModel} />)}
      {data && !models.length && <p className="muted">no models match</p>}
    </div>
  )
}

function WorkerStrip({ workers }) {
  if (!workers.length) return null
  return (
    <div className="workers">
      {workers.map(w => (
        <span key={w.worker_id} className={`pill ${w.status || 'unknown'}`} title={w.worker_id}>
          {w.name || w.worker_id} <small>{w.status || '?'} · {fmtBytes(w.gpu_total)} GPU · {fmtBytes(w.ram_total)} RAM · {w.pkg_version || '?'}
            {w.budget && <span className="budget" title={`EARMARK rev ${w.budget.rev}: every verdict for this worker is priced against these budgets (= min(total − reserve, operator limit)). Reserves: VRAM ${fmtBytes(w.budget.vram_reserve)}, RAM ${fmtBytes(w.budget.ram_reserve)}${w.budget.gpu_limit ? `; GPU limit ${fmtBytes(w.budget.gpu_limit)}` : ''}${w.budget.ram_limit ? `; RAM limit ${fmtBytes(w.budget.ram_limit)}` : ''}.${w.budget.last_change ? ` Last change: ${w.budget.last_change.expanded ? 'EXPANDED' : ''}${w.budget.last_change.reduced ? ' REDUCED' : ''} from ${fmtBytes(w.budget.last_change.prev?.gpu_budget)} / ${fmtBytes(w.budget.last_change.prev?.ram_budget)} (rev ${w.budget.last_change.prev?.rev})` : ''}`}>
              {' · budget '}{fmtBytes(w.budget.gpu_budget)} GPU / {fmtBytes(w.budget.ram_budget)} RAM · rev {w.budget.rev}{w.budget.last_change?.reduced && !w.budget.last_change?.expanded ? ' ↓' : w.budget.last_change?.expanded ? ' ↑' : ''}
            </span>}</small>
        </span>
      ))}
    </div>
  )
}

const SPEC_HIDE = new Set(['quants'])
const WEIGHTS_HIDE = new Set(['sig', 'version', 'computed_at'])

// One table per model, straight from the model_full row.
function ModelTable({ m, workers, allocModes, onPatch }) {
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)
  const [editing, setEditing] = useState(null)   // worker_id being edited
  const [addWorker, setAddWorker] = useState('')
  const spec = Object.entries(m.spec || {}).filter(([k, v]) => !SPEC_HIDE.has(k) && v !== null && v !== undefined && v !== '' && !(Array.isArray(v) && !v.length))
  const weights = Object.entries(m.weights || {}).filter(([k, v]) => !WEIGHTS_HIDE.has(k) && v !== null && v !== undefined)
  const serving = Object.entries(m.serving || {})
  const liveBy = new Map((m.live || []).map(l => [l.worker_id, l]))
  const onModel = new Set((m.workers || []).map(w => w.worker_id))
  const spare = workers.filter(w => !onModel.has(w.worker_id) && !onModel.has(w.name))

  const run = async (label, fn) => {
    setBusy(label); setErr(null)
    try { const r = await fn(); if (r?.model) onPatch(r.model) } catch (e) { setErr(e.message) } finally { setBusy(null) }
  }
  const recompute = () => run('recompute', () => api(`api/models/${m.id}/recompute`, { method: 'POST' }))
  const saveKnobs = (wid, set, unset) => run(`knobs:${wid}`, async () => {
    const r = await api(`api/models/${m.id}/workers/${encodeURIComponent(wid)}/knobs`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ set, unset }) })
    setEditing(null); return r
  })

  return (
    <section className="model">
      <table>
        <caption>
          <strong>{m.name}</strong> <span className="muted">#{m.id}</span>{' '}<a className="small" href={rawHref(`table=model_full&f=id:${m.id}`)} title="this model's row + related tables in the raw browser">raw</a>
          {m.framework && <span className="tag">{m.framework}</span>}
          {m.hub_id && <span className="muted"> {m.hub_id}</span>}
          <span className="right">
            {err && <span className="error inline">{err}</span>}
            <button disabled={!!busy} onClick={recompute} title="drop the stored weights facts and worker plans for this model and compute them again">{busy === 'recompute' ? 'computing…' : 'recompute'}</button>
            <span className="muted"> updated {fmtTs(m.updated_at)}</span>
          </span>
        </caption>
        <tbody>
          {spec.length > 0 && <Group title="spec" rows={spec} />}
          {weights.length > 0 && <Group title={`weights · immutable, computed once ${fmtTs(m.weights?.computed_at)}`} rows={weights} />}
          {!m.weights && <tr className="group"><th colSpan={2}>weights <span className="muted">— not computed yet</span></th></tr>}
          {serving.length > 0 && <Group title="serving (model intent)" rows={serving} />}
          {m.quants?.length > 0 && (
            <>
              <tr className="group"><th colSpan={2}>quants ({m.quants.length}) <span className="muted small">— model_quant_facts, ingested from hugpy.json weights_facts</span></th></tr>
              {m.quants.map(qt => (
                <tr key={qt.file}><th>{qt.quant || qt.file}</th><td>{qt.file} <span className="muted">{fmtBytes(qt.bytes)}{qt.shards > 1 ? ` · ${qt.shards} shards` : ''}{qt.is_moe ? ` · MoE ${qt.expert_count} experts / ${qt.expert_used_count} active · experts ${fmtBytes(qt.expert_bytes)} + shared ${fmtBytes(qt.non_expert_bytes)}` : ''}{qt.kv_geo?.n_layers ? ` · ${qt.kv_geo.n_layers} layers · ${qt.kv_geo.n_kv_heads} kv heads × ${qt.kv_geo.head_dim} · trained ctx ${qt.kv_geo.ctx_train}` : ''}{qt.kv_cost ? ` · KV ${(qt.kv_cost.bytes_per_token / 1048576).toFixed(2)} MiB/token${qt.kv_cost.fixed_bytes ? ` + ${fmtBytes(qt.kv_cost.fixed_bytes)} state` : ''} → ${fmtBytes(qt.kv_cost.at_pct?.['100'])} @ trained ctx (${qt.kv_cost.basis})` : ''}{qt.error ? ` · ${qt.error}` : ''}</span></td></tr>
              ))}
            </>
          )}
          <tr className="group"><th colSpan={2}>
            workers ({m.workers?.length || 0})
            {spare.length > 0 && (
              <span className="right">
                <select value={addWorker} onChange={e => setAddWorker(e.target.value)}>
                  <option value="">add to worker…</option>
                  {spare.map(w => <option key={w.worker_id} value={w.worker_id}>{w.name || w.worker_id}</option>)}
                </select>
                <button disabled={!addWorker || !!busy} onClick={() => { saveKnobs(addWorker, {}, []); setAddWorker('') }}>add</button>
              </span>
            )}
          </th></tr>
          {m.workers?.length ? (
            <tr><td colSpan={2} className="nested">
              <table className="alloc">
                <thead>
                  <tr><th colSpan={2} className="band">intent (knobs)</th><th colSpan={4} className="band calc">computed once per worker · one verdict per quant</th><th colSpan={2} className="band live">live (heartbeat)</th></tr>
                  <tr><th>worker</th><th>knobs</th><th colSpan={4}>quants · fits · auto · 4-bit · MoE</th><th>state</th><th>updated</th></tr>
                </thead>
                <tbody>
                  {m.workers.map(w => {
                    const l = liveBy.get(w.worker_id) || {}
                    const p = w.plan
                    return [
                      <tr key={w.worker_id}>
                        <td>
                          <span className={`dot ${l.worker_status || 'unknown'}`} />{w.worker}{w.rank != null && <span className="muted small"> rank {w.rank}</span>}
                          <div className="muted small">{w.worker_id}</div>
                          <button className="small" disabled={!!busy} onClick={() => setEditing(editing === w.worker_id ? null : w.worker_id)}>{editing === w.worker_id ? 'close' : 'edit knobs'}</button>
                        </td>
                        <td><KV obj={w.knobs} />{w.pinned && Object.keys(w.pinned).length > 0 && <div className="muted small">pinned by central: <KV obj={w.pinned} /></div>}</td>
                        <td colSpan={4}>{p ? <QuantPlans plan={p} verdicts={w.verdicts} facts={m.quants} knobs={w.knobs || {}} /> : <span className="muted">not computed yet</span>}</td>
                        <td>
                          {l.loaded && <span className="tag ok">loaded</span>}
                          {l.loading && <span className="tag warn">loading</span>}
                          {!l.loaded && !l.loading && <span className="tag">idle</span>}
                          <div className="muted small">worker {l.worker_status || '?'}{l.last_picked ? ` · picked ${fmtTs(l.last_picked)}` : ''}</div>
                        </td>
                        <td className="muted small">{fmtTs(l.updated_at)}{p?.computed_at && <div>plan {fmtTs(p.computed_at)}</div>}</td>
                      </tr>,
                      editing === w.worker_id && (
                        <tr key={w.worker_id + ':edit'} className="editor"><td colSpan={8}>
                          <KnobEditor knobs={w.knobs || {}} plan={p} verdicts={w.verdicts || []} quants={m.quants || []} allocModes={allocModes} busy={busy === `knobs:${w.worker_id}`}
                                      onSave={(set, unset) => saveKnobs(w.worker_id, set, unset)} onCancel={() => setEditing(null)} />
                        </td></tr>
                      ),
                    ]
                  })}
                </tbody>
              </table>
            </td></tr>
          ) : <tr><td colSpan={2} className="muted">not allocated to any worker</td></tr>}
        </tbody>
      </table>
    </section>
  )
}

function Plan({ auto }) {
  if (!auto) return <span className="muted">—</span>
  const spill = Object.entries(auto.spill || {}).filter(([k]) => k !== 'alloc_mode')
  return (
    <div title={auto.why || ''}>
      <span className="tag ok">{auto.mode}</span>
      {spill.length > 0 && <KV obj={Object.fromEntries(spill)} />}
      {auto.why && <div className="muted small why">{auto.why}</div>}
    </div>
  )
}

// The knob gguf_file picks the quant; else the model's default variant.
function selectedFile(plan, verdicts, knobs) {
  const files = (verdicts || []).map(v => v.file)
  const want = knobs?.gguf_file ? String(knobs.gguf_file).split('/').pop() : null
  if (want && files.includes(want)) return want
  return plan?.default_variant && files.includes(plan.default_variant) ? plan.default_variant : files[0]
}

// Explicit memory plan per mode: the key-values, not a sentence.
function Memory({ mem }) {
  const modes = Object.entries(mem || {}).filter(([, v]) => v && typeof v === 'object' && 'gpu_bytes' in v)
  if (!modes.length) return <span className="muted small">{mem?.why || '—'}</span>
  return (
    <table className="mem">
      <thead><tr><th>mode</th><th>GPU</th><th>RAM</th><th>KV</th><th>ctx</th><th>ctx max f16 / q8_0 / q4_0</th><th>fits</th></tr></thead>
      <tbody>
        {modes.map(([mode, v]) => (
          <tr key={mode} className={v.fits ? '' : 'nofit'} title={v.why || ''}>
            <td>{mode}{v.n_gpu_layers != null && <span className="muted small"> {v.n_gpu_layers}/{v.n_layers}L</span>}{v.n_cpu_moe != null && <span className="muted small"> cpu_moe {v.n_cpu_moe} · len {v.leniency_pct}%</span>}</td>
            <td>{fmtBytes(v.gpu_bytes)}</td><td>{fmtBytes(v.ram_bytes)}</td><td>{v.kv_bytes != null ? fmtBytes(v.kv_bytes) : '—'}{v.kv_device === 'ram' ? <span className="muted small"> (RAM)</span> : null}</td><td>{v.ctx || '—'}</td>
            <td className="muted small">{v.ctx_max ? `${v.ctx_max.f16 ?? '—'} / ${v.ctx_max.q8_0 ?? '—'} / ${v.ctx_max.q4_0 ?? '—'}` : '—'}</td>
            <td>{v.fits ? <span className="tag ok">fits</span> : <span className="tag warn">no</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// One sub-row per QUANT under the worker (rows of model_worker_quants): its
// own fits / memory plan per mode / MoE verdict.
function QuantPlans({ plan, verdicts, facts, knobs }) {
  if (!verdicts?.length) return <span className="muted">{plan?.feasible?.why || 'no verdicts yet'}</span>
  const sel = selectedFile(plan, verdicts, knobs)
  const factBy = Object.fromEntries((facts || []).map(f => [f.file, f]))
  return (
    <table className="quants">
      <thead><tr><th>quant</th><th>fits</th><th>memory plan · per mode</th><th>MoE</th></tr></thead>
      <tbody>
        {verdicts.map(v => { const f = factBy[v.file] || {}; return (
          <tr key={v.file} className={v.file === sel ? 'selected' : ''}>
            <td title={v.file}><strong>{f.quant || v.file}</strong>{v.file === sel && <span className="tag ok small">serves</span>}
              <div className="muted small">{fmtBytes(f.bytes)}{f.is_moe ? ` · MoE ${f.expert_count}/${f.expert_used_count}` : ''}{f.kv_geo?.ctx_train ? ` · trained ctx ${f.kv_geo.ctx_train}` : ''}</div></td>
            <td>{v.fits === true ? <span className="tag ok">fits</span> : v.fits === false ? <span className="tag warn">no</span> : <span className="tag">?</span>}
              {v.kept && <div className="muted small" title={v.kept.why}>kept since budget rev {v.kept.since_rev} (↓ reduction)</div>}
              <div className="muted small">modes: {(v.modes || []).join(', ') || '—'}</div></td>
            <td><Memory mem={v.memory} />
              {v.memory?.bnb_4bit && <div className="muted small" style={{ marginTop: 4 }}>4-bit (bitsandbytes):</div>}
              {v.memory?.bnb_4bit && <Memory mem={v.memory.bnb_4bit} />}</td>
            <td><Moe moe={v.moe} /></td>
          </tr>) })}
      </tbody>
    </table>
  )
}

// MoE is a FACT per pair (offered or not), never a bare checkbox: offered means
// this exact RAM-priority split with KV at the derived ctx; otherwise the row
// says why (operator, 2026-10-01: "if it's not explicit it's not MoE").
function Moe({ moe }) {
  if (!moe) return <span className="muted">—</span>
  if (!moe.offered) return <div title={moe.why || ''}><span className="tag">{/dense/.test(moe.why || '') ? 'dense' : 'not offered'}</span><div className="muted small why">{moe.why}</div></div>
  const sp = { ...moe.spill }; delete sp.alloc_mode
  return (
    <div title={moe.why || ''}>
      <span className="tag ok">explicit · RAM priority</span>
      <KV obj={sp} />
      {moe.kv && <div className="muted small">KV {fmtBytes(moe.kv.kv_bytes)} @ ctx {moe.kv.ctx}{moe.kv.ctx_train ? ` (trained ${moe.kv.ctx_train})` : ''} · {(moe.kv.bytes_per_token / 1048576).toFixed(2)} MiB/token</div>}
      <div className="muted small why">{moe.why}</div>
    </div>
  )
}

function Feasible({ f }) {
  if (!f) return <span className="muted">—</span>
  if (!f.ok) return <span className="tag bad">no</span>
  return <>{(f.modes || []).map(x => <span key={x} className="tag">{x}</span>)}</>
}

// The knobs. Every field maps 1:1 to a key in model_workers.user_settings;
// blank = unset the key (back to auto).
const NUM = ['n_gpu_layers', 'n_cpu_moe', 'threads', 'llama_ctx', 'gpu_mem_gib', 'cpu_mem_gib', 'ctx_pct']
function KnobEditor({ knobs, plan, verdicts, quants, allocModes, busy, onSave, onCancel }) {
  const [f, setF] = useState(() => ({
    alloc_mode: knobs.alloc_mode ?? '', gguf_file: knobs.gguf_file ?? '', serve_mode: knobs.serve_mode ?? '',
    moe: knobs.moe === true ? 'on' : knobs.moe === false ? 'off' : 'auto', bnb_4bit: !!knobs.bnb_4bit,
    ...Object.fromEntries(NUM.map(k => [k, knobs[k] ?? ''])),
  }))
  const set = (k, v) => setF(s => ({ ...s, [k]: v }))
  // MoE is offered per QUANT: follow the quant the editor currently selects
  const selQ = (verdicts || []).find(v => v.file === selectedFile(plan, verdicts, { gguf_file: f.gguf_file })) || null
  const moeOffered = !!selQ?.moe?.offered
  const submit = e => {
    e.preventDefault()
    const out = {}, unset = []
    const put = (k, v) => (v === '' || v === null || v === undefined) ? unset.push(k) : (out[k] = v)
    put('alloc_mode', f.alloc_mode); put('gguf_file', f.gguf_file); put('serve_mode', f.serve_mode)
    // the MoE tick exists only where the plan offers the split; elsewhere any stored value is cleared
    put('moe', !moeOffered || f.moe === 'auto' ? '' : f.moe === 'on')
    f.bnb_4bit ? (out.bnb_4bit = true) : unset.push('bnb_4bit')
    for (const k of NUM) {
      const v = String(f[k]).trim()
      if (v === '') unset.push(k)
      else if (v === 'off') out[k] = 'off'
      else if (!isNaN(Number(v))) out[k] = Number(v)
      else return alert(`${k}: number or "off"`)
    }
    onSave(out, unset)
  }
  return (
    <form className="knobs" onSubmit={submit}>
      <label>alloc_mode <select value={f.alloc_mode} onChange={e => set('alloc_mode', e.target.value)}><option value="">auto</option>{allocModes.map(x => <option key={x}>{x}</option>)}</select></label>
      {moeOffered
        ? <label title={selQ.moe.why}>moe (= {selQ.moe.spill.n_cpu_moe} layers' experts in RAM, ctx {selQ.moe.spill.llama_ctx}) <select value={f.moe} onChange={e => set('moe', e.target.value)}><option value="auto">auto</option><option value="on">on</option><option value="off">off</option></select></label>
        : <span className="muted small" title={selQ?.moe?.why || ''}>moe: {selQ?.moe ? (/dense/.test(selQ.moe.why || '') ? 'dense' : 'not offered for this quant') : 'no plan'}</span>}
      <label>bnb_4bit <input type="checkbox" checked={f.bnb_4bit} onChange={e => set('bnb_4bit', e.target.checked)} /></label>
      <label>gguf_file <input list="q" value={f.gguf_file} onChange={e => set('gguf_file', e.target.value)} placeholder="auto" /><datalist id="q">{quants.map(q => <option key={q.file} value={q.file} />)}</datalist></label>
      <label>serve_mode <input value={f.serve_mode} onChange={e => set('serve_mode', e.target.value)} placeholder="auto" /></label>
      {NUM.map(k => <label key={k}>{k} <input value={f[k]} onChange={e => set(k, e.target.value)} placeholder="auto" /></label>)}
      <span className="actions"><button type="submit" disabled={busy}>{busy ? 'saving…' : 'save to DB'}</button> <button type="button" onClick={onCancel}>cancel</button></span>
    </form>
  )
}

function Group({ title, rows }) {
  return (
    <>
      <tr className="group"><th colSpan={2}>{title}</th></tr>
      {rows.map(([k, v]) => <tr key={k}><th>{k}</th><td><Val v={v} k={k} /></td></tr>)}
    </>
  )
}

function KV({ obj }) {
  if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) return <span className="muted">—</span>
  return (
    <dl className="kv">
      {Object.entries(obj).map(([k, v]) => <div key={k}><dt>{k}</dt><dd><Val v={v} k={k} /></dd></div>)}
    </dl>
  )
}

function Val({ v, k }) {
  if (v === null || v === undefined) return <span className="muted">null</span>
  if (typeof v === 'boolean') return <span className={v ? 'ok' : 'muted'}>{String(v)}</span>
  if (typeof v === 'number' && /bytes$/.test(k || '')) return <>{fmtBytes(v)} <span className="muted small">({v})</span></>
  if (typeof v === 'number' && /(_at|computed_at)$/.test(k || '')) return <>{fmtTs(v)}</>
  if (typeof v !== 'object') return <>{String(v)}</>
  if (Array.isArray(v)) {
    if (v.every(x => typeof x !== 'object')) return <>{v.map((x, i) => <span key={i} className="tag">{String(x)}</span>)}</>
    return <code className="json">{JSON.stringify(v)}</code>
  }
  return <KV obj={v} />
}

function fmtTs(t) {
  if (!t) return '—'
  const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t)
  return isNaN(d) ? String(t) : d.toLocaleString()
}

function fmtBytes(b) {
  if (b == null) return '—'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let n = Number(b)
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`
}
