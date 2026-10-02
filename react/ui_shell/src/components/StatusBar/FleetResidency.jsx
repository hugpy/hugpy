import { useMemo } from 'react'

// FleetResidency — the fleet-wide DISTRIBUTION view behind the Overview's VRAM
// and Host RAM meters.
//
// The meters answer "how full is the fleet"; this answers the operator's real
// question: WHICH models are sitting in each worker's VRAM and RAM right now,
// and what is each one DOING. It is the fleet-scale analog of the per-worker
// GPU/RAM bars in WorkersPanel, but keyed on models rather than totals.
//
// Everything here is derived from data the Overview already holds — the
// `workers` prop (/api/llm/workers) and the `queue` poll (/api/llm/queue). No
// new fetch, no new endpoint.
//
// Honesty rules baked in:
//   • Segment widths come from MEASURED bytes (slot vram_bytes, slot
//     rss_anon_bytes, ram_resident_bytes) — never from a declared/estimated
//     size. MEASURED-FIRST per the operator ruling of 2026-07-28: worker-side
//     measurements are the truth for residency/occupancy, so a ram row's
//     `ram_resident_bytes` (real host-RAM pages the model occupies now) wins over
//     `model_bytes`, which is the model's ON-DISK size and only ever an UPPER
//     BOUND. Older workers send no ram_resident_bytes; those rows fall back to
//     model_bytes and say so in their tooltip when a bar mixes the two.
//   • Whatever the bar total says is used but no model claims is drawn as
//     "other" (drivers, compositor, out-of-band processes), not silently
//     folded into a model.
//   • A worker that reports no `allocations` at all (older worker build) gets a
//     plain used/free bar and says so, rather than pretending to a breakdown.

const STATES = [
  ['answering', 'answering'],
  ['loading',   'loading'],
  ['serving',   'serving'],
  ['idle',      'idle (resident)'],
]

function fmtBytes(n) {
  if (n == null || !isFinite(n)) return '—'
  // Binary math (÷1024) must wear BINARY unit labels (k65 defect-1 alignment):
  // dividing by 1024 while labeling "GB" was the 1024-vs-1000 mismatch the
  // operator flagged (~2.4% off by construction). The whole stack speaks GiB.
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`
}

const num = (x) => (typeof x === 'number' && isFinite(x) && x > 0 ? x : 0)

// State precedence: answering > loading > serving > idle.
function allocState(a, key, answering, loading) {
  if (a && a.busy === true) return 'answering'
  if (answering.has(key)) return 'answering'
  if (loading.has(key)) return 'loading'
  if (a && (a.serving === true || a.healthy === true)) return 'serving'
  return 'idle'
}

function layerNote(a) {
  if (!a) return null
  const n = a.n_gpu_layers, t = a.total_layers
  if (typeof n !== 'number' || typeof t !== 'number' || t <= 0) return null
  if (n >= t) return `all ${t} layers on GPU`
  return `${n}/${t} layers on GPU`
}

// Turn a list of {key,bytes,state,...} plus the bar's used/total into drawable
// segments. `used` may disagree with the model sum in either direction (RAM
// bytes can be file-backed and not counted as used); we widen the denominator
// rather than draw a bar that overflows or lies.
function buildBar(models, used, total) {
  const sum = models.reduce((s, m) => s + m.bytes, 0)
  const other = Math.max(0, num(used) - sum)
  const denom = Math.max(num(total), sum + other, 1)
  const free = Math.max(0, denom - sum - other)
  return {
    denom, sum, other, free,
    overflow: sum > num(used) && num(used) > 0,
    segs: models.map(m => ({ ...m, pct: (m.bytes / denom) * 100 })),
    otherPct: (other / denom) * 100,
    freePct: (free / denom) * 100,
  }
}

export default function FleetResidency({ workers = [], queue = {} }) {
  const rows = useMemo(() => {
    // Only 'active' queue rows mean a model is generating right now. The queue
    // also carries 'expired' rows (stale reservations) — counting those would
    // paint half the fleet as answering forever.
    const answering = new Set()
    for (const r of (queue.active || [])) {
      if (r && r.state === 'active') {
        const k = r.model_key || r.model
        if (k) answering.add(k)
      }
    }

    const online = (workers || []).filter(w => w && w.status === 'online')

    return online.map(w => {
      const loading = new Set([
        ...(Array.isArray(w.loading) ? w.loading : []),
        ...(Array.isArray(w.provisioning) ? w.provisioning : []),
      ].filter(Boolean))
      const prog = w.provision_progress || {}

      const gpus = Array.isArray(w.gpus) ? w.gpus : []
      const gpuTotal = gpus.reduce((s, g) => s + num(g.memory_total), 0)
      const gpuFree  = gpus.reduce((s, g) => s + num(g.memory_free), 0)
      const gpuUsed  = Math.max(0, gpuTotal - gpuFree)
      const gpuName  = gpus.map(g => g.name).filter(Boolean).join(' + ')

      // RAM scale: bar_total/bar_used when the worker publishes them,
      // otherwise fall back to free_ram against whatever total we can see.
      const ramTotal = num(w.bar_total) || (num(w.free_ram) + num(w.bar_used))
      const ramUsed  = num(w.bar_used) || Math.max(0, ramTotal - num(w.free_ram))

      const allocs = Array.isArray(w.allocations) ? w.allocations : null
      const detailed = allocs != null

      const vramModels = []
      const ramModels  = []
      for (const a of (allocs || [])) {
        const key = a && a.model_key
        if (!key) continue
        const st = allocState(a, key, answering, loading)
        const layers = layerNote(a)
        const v = num(a.vram_bytes)
        if (v) vramModels.push({ key, bytes: v, state: st, layers, where: 'vram' })
        // RAM side: parked bytes (kind:'ram') and the host-side residue of a
        // partially offloaded slot both count against host RAM. Measured first:
        // ram_resident_bytes (measured page residency / torch cpu bytes) and a
        // slot's rss_anon_bytes are real occupancy; model_bytes is the on-disk
        // file size, kept only as the fallback upper bound for older workers.
        const measured = num(a.ram_resident_bytes) || num(a.rss_anon_bytes)
        const r = measured || num(a.model_bytes)
        if (r) {
          ramModels.push({ key, bytes: r, state: st, layers, where: 'ram',
                           basis: measured ? 'measured' : 'file' })
        }
      }
      vramModels.sort((x, y) => y.bytes - x.bytes)
      ramModels.sort((x, y) => y.bytes - x.bytes)
      // Only worth saying when a single bar mixes the two regimes — an all-
      // measured or all-fallback bar is unambiguous and stays uncluttered.
      const ramMixed = ramModels.some(m => m.basis === 'measured')
                    && ramModels.some(m => m.basis === 'file')
      if (ramMixed) for (const m of ramModels) m.mixed = true

      // Models that are mid-load own no bytes yet — they'd be invisible in the
      // bars, so they get their own chips with a progress fraction.
      const resident = new Set([...vramModels, ...ramModels].map(m => m.key))
      const pending = [...loading].filter(k => !resident.has(k)).map(k => ({
        key: k,
        frac: (prog[k] && typeof prog[k].frac === 'number') ? prog[k].frac : null,
      }))

      return {
        id: w.name || w.worker_id || w.id || w.host || 'worker',
        name: w.name || w.worker_id || w.host || 'worker',
        gpuName, detailed, pending,
        vram: buildBar(vramModels, gpuUsed, gpuTotal),
        vramUsed: gpuUsed, vramTotal: gpuTotal,
        ram: buildBar(ramModels, ramUsed, ramTotal),
        ramUsed, ramTotal,
      }
    })
  }, [workers, queue])

  return (
    <div className="fr-panel">
      <div className="fr-legend">
        {STATES.map(([s, label]) => (
          <span key={s} className="fr-legend-item">
            <i className={`fr-dot fr-s-${s}`} />{label}
          </span>
        ))}
        <span className="fr-legend-item"><i className="fr-dot fr-s-other" />other / unattributed</span>
        <span className="fr-legend-item"><i className="fr-dot fr-s-free" />free</span>
      </div>

      {rows.length === 0 ? (
        <div className="fr-empty">no online workers</div>
      ) : rows.map(r => (
        <div className="fr-worker" key={r.id}>
          <div className="fr-worker-head">
            <span className="fr-worker-name">{r.name}</span>
            {r.gpuName && <span className="fr-worker-gpu">{r.gpuName}</span>}
            {!r.detailed && <span className="fr-nodetail">no per-model detail</span>}
          </div>

          <Bar label="VRAM" bar={r.vram} used={r.vramUsed} total={r.vramTotal}
               detailed={r.detailed} empty="no GPU-resident models" />
          <Bar label="RAM" bar={r.ram} used={r.ramUsed} total={r.ramTotal}
               detailed={r.detailed} empty="no RAM-parked models" />

          {r.pending.length > 0 && (
            <div className="fr-list">
              {r.pending.map(p => (
                <span className="fr-chip" key={`ld-${p.key}`}
                      title={`${p.key} — loading${p.frac != null ? ` ${Math.round(p.frac * 100)}%` : ''}`}>
                  <i className="fr-dot fr-s-loading" />
                  <span className="fr-chip-key">{p.key}</span>
                  <span className="fr-chip-size">
                    loading{p.frac != null ? ` ${Math.round(p.frac * 100)}%` : ''}
                  </span>
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function segTitle(s, label) {
  const bits = [s.key, `${fmtBytes(s.bytes)} ${label}`, s.state]
  if (s.layers) bits.push(s.layers)
  // Say which regime the number came from ONLY when this bar mixes measured
  // rows with file-size fallbacks — otherwise the byte figure is unambiguous.
  if (s.mixed) bits.push(s.basis === 'measured' ? 'measured resident'
                                                : 'file size — upper bound')
  return bits.join(' — ')
}

function Bar({ label, bar, used, total, detailed, empty }) {
  if (!total) return null
  return (
    <div className="fr-row">
      <span className="fr-row-label">{label}</span>
      <div className="fr-row-body">
        <div className="fr-track"
             title={`${label}: ${fmtBytes(used)} used of ${fmtBytes(total)}`}>
          {bar.segs.map((s, i) => (
            <div key={`${s.key}-${i}`}
                 className={`fr-seg fr-s-${s.state}`}
                 style={{ width: `${s.pct}%` }}
                 title={segTitle(s, label)} />
          ))}
          {bar.otherPct > 0 && (
            <div className="fr-seg fr-s-other" style={{ width: `${bar.otherPct}%` }}
                 title={`other / unattributed — ${fmtBytes(bar.other)} ${label}`} />
          )}
          {bar.freePct > 0 && (
            <div className="fr-seg fr-s-free" style={{ width: `${bar.freePct}%` }}
                 title={`free — ${fmtBytes(bar.free)} ${label}`} />
          )}
        </div>
        <div className="fr-list">
          {bar.segs.length === 0 ? (
            <span className="fr-note">{detailed ? empty : 'no per-model detail from this worker'}</span>
          ) : bar.segs.map((s, i) => (
            <span className="fr-chip" key={`${s.key}-c-${i}`} title={segTitle(s, label)}>
              <i className={`fr-dot fr-s-${s.state}`} />
              <span className="fr-chip-key">{s.key}</span>
              <span className="fr-chip-size">{fmtBytes(s.bytes)}</span>
              {s.layers && <span className="fr-chip-layers">{s.layers}</span>}
            </span>
          ))}
          <span className="fr-note fr-note-tot">
            {fmtBytes(used)} used / {fmtBytes(total)}
          </span>
        </div>
      </div>
    </div>
  )
}
