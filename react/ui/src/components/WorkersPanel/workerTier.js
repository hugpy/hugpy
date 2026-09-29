// Per-(model, worker) STORAGE TIER for the "load a model" picker — where the
// model's bytes sit RELATIVE TO THIS WORKER (operator ask 2026-09-29: "sort by
// hot on that worker"). This is deliberately NOT central's readiness (the
// picker's "Central" column) and NOT residency (loaded/serving): it answers
// "if I allocate this here, does it load off this box's own drive?".
//
// Sources, all already on the wire (no worker-side change):
//   * worker.storage.models[]  — the worker's reaper survey materialized by
//     central's storage_proposal (GET /llm/workers). One row per model on a
//     drive this worker can read, keyed by model_key, with `store`:
//       'reapable'   the worker's OWN store root (aeb: /mnt/nvmes/.../hot990/aeb)
//       'unreapable' a local store the box hasn't declared disposable — still
//                    on this box's drive
//       'shared'     the fleet's shared/central catalog read through from here
//                    (a-brain's NFS view of llm_storage) — NOT this box's disk
//     Unlike models_local it is NOT assignment-scoped, which is what makes it
//     usable for a picker whose whole candidate set is unassigned models.
//   * worker.models_local[]    — heartbeat disk-truth (assigned ∩ on-disk +
//     system-discovered models). A superset check for rows the survey missed.
//   * model.hot_workers[]      — central's stamp on the /models row (derived
//     from models_local, by worker NAME).
//   * model.status/dir_bytes   — central's catalog: files on llm_storage.
//
// Tiers, best first:
//   hot      on THIS worker's own drive
//   shared   readable from this worker via a shared store (no local copy)
//   central  on central storage only — copies here on first call (lazy)
//   none     files nowhere this row can prove
export const TIER_RANK = { hot: 0, shared: 1, central: 2, none: 3 }

export const TIER_VIEW = {
  hot:     { glyph: '🌡 hot',     pill: 'wp-pill-hot',
             title: "on THIS worker's own drive — loads straight off local disk" },
  shared:  { glyph: '🔗 shared',  pill: 'wp-pill-shared',
             title: 'on a SHARED store this worker reads through (central catalog over the network) — no local copy on this box' },
  central: { glyph: '○ central',  pill: 'wp-pill-central',
             title: "on CENTRAL storage only — copies to this worker on the FIRST call (lazy download). Not on this box's drive yet" },
  none:    { glyph: '— none',     pill: 'wp-pill-missing',
             title: 'files not found on this worker, its shared stores, or central' },
}

const keyOf = (m) => (m && (m.model_key ?? m.key)) || ''

// The spellings a worker may report a catalog row under: model_key, name,
// hub_id, owner~name (the "~" alias central's _match_keys unifies) and the
// bare name behind an owner~name key.
export function modelSpellings(m) {
  const out = new Set()
  const add = (v) => { if (typeof v === 'string' && v) out.add(v) }
  add(keyOf(m)); add(m?.name); add(m?.hub_id)
  const hid = m?.hub_id || ''
  if (hid.includes('/')) {
    add(hid.replace('/', '~'))
    add(hid.slice(hid.lastIndexOf('/') + 1))
  }
  const k = keyOf(m)
  const at = k.indexOf('~')
  if (at > 0) add(k.slice(at + 1))
  return out
}

// Index a worker's storage survey once per render: spelling → row.
export function storageRowsByKey(worker) {
  const rows = Array.isArray(worker?.storage?.models) ? worker.storage.models : []
  const map = new Map()
  for (const r of rows) {
    if (r && typeof r.model_key === 'string' && r.model_key) map.set(r.model_key, r)
  }
  return map
}

// → { tier, rank, row, source } for one model on one worker.
//   row    the storage survey row when one matched (bytes, last_picked, store)
//   source which signal decided: 'storage' | 'models_local' | 'hot_workers' |
//          'catalog' | null
export function workerTierOf(m, worker, rowsByKey = null) {
  const spellings = modelSpellings(m)
  const rows = rowsByKey || storageRowsByKey(worker)
  let row = null
  for (const s of spellings) { if (rows.has(s)) { row = rows.get(s); break } }
  if (row) {
    const store = String(row.store || 'reapable')
    if (store === 'shared') return { tier: 'shared', rank: TIER_RANK.shared, row, source: 'storage' }
    return { tier: 'hot', rank: TIER_RANK.hot, row, source: 'storage' }
  }
  const local = Array.isArray(worker?.models_local) ? worker.models_local : []
  if (local.some(k => spellings.has(k))) return { tier: 'hot', rank: TIER_RANK.hot, row: null, source: 'models_local' }
  const wname = worker?.name || worker?.id
  const hw = Array.isArray(m?.hot_workers) ? m.hot_workers : []
  if (wname && hw.includes(wname)) return { tier: 'hot', rank: TIER_RANK.hot, row: null, source: 'hot_workers' }
  const centralHas = !!m && (m.status === 'installed' || (m.dir_bytes ?? 0) > 0)
  if (centralHas) return { tier: 'central', rank: TIER_RANK.central, row: null, source: 'catalog' }
  return { tier: 'none', rank: TIER_RANK.none, row: null, source: null }
}

// The "hot (this worker)" sort: tier ascending (hot first), then the model's
// on-worker bytes (survey row) or catalog size DESC, then name. `dir` flips
// the whole order so the header arrow behaves like every other column.
export function compareByWorkerTier(a, b, ta, tb, sizeOf, dir = 1) {
  if (ta.rank !== tb.rank) return (ta.rank - tb.rank) * dir
  const sa = ta.row?.bytes ?? sizeOf(a) ?? -1
  const sb = tb.row?.bytes ?? sizeOf(b) ?? -1
  if (sa !== sb) return (sb - sa) * dir
  return String(a?.name || keyOf(a)).localeCompare(String(b?.name || keyOf(b))) * dir
}

// ── ONE vocabulary for storage locality (operator 2026-09-29: "shouldn't be
// split nomenclature"). Two AXES, never conflated:
//   storage tier (above)   hot / shared / central / none — WHERE the bytes sit
//                          relative to this worker.
//   residency              answering / serving / loaded / loading / pulling /
//                          not loaded / failed — whether the weights are IN
//                          VRAM/RAM on this worker right now.
// GET /llm/models/status speaks an older per-worker vocabulary that mixes the
// two ('hot' = loaded-idle, 'cold' = on this drive but not loaded, 'on central'
// = central only). The mapping below is how every chip/pill renders it so the
// picker's HOT group and the worker row's allocation rows say the same word
// for the same disk fact.
export const STATUS_BASE_TIER = {
  answering: 'hot', serving: 'hot', hot: 'hot', loading: 'hot', cold: 'hot',
  'downloading from central': 'central', 'on central': 'central',
  'on another worker': 'none', missing: 'none', 'not allocated': 'none',
}

// Backend status word (`base`, else `state`) → tier, or null when the word
// carries no storage fact ('unknown', 'n/a', 'failed' without a base).
export function tierFromStatusBase(base) {
  return STATUS_BASE_TIER[String(base ?? '')] ?? null
}

// Residency words + the existing state-pill classes they render with.
export const RESIDENCY_VIEW = {
  answering:   { word: 'answering',  icon: '⚡', pill: 'wp-pill-answering', tone: 'ok' },
  serving:     { word: 'serving',    icon: '🔥', pill: 'wp-pill-serving',   tone: 'ok' },
  loaded:      { word: 'loaded',     icon: '📌', pill: 'wp-pill-loaded',    tone: 'ok' },
  loading:     { word: 'loading',    icon: '🔶', pill: 'wp-pill-heating',   tone: 'live' },
  pulling:     { word: 'pulling',    icon: '⏳', pill: 'wp-pill-pulling',   tone: 'live' },
  'not loaded': { word: 'not loaded', icon: '○', pill: 'wp-pill-cold',     tone: 'muted' },
  failed:      { word: 'failed',     icon: '✗', pill: 'wp-pill-refused',   tone: 'bad' },
}

// Backend status word → residency key. Every storage-only word ('cold', 'on
// central', 'on another worker', 'missing', 'not allocated', 'unknown') is
// simply NOT LOADED here; the storage fact lives in the tier, not the word.
export function residencyFromStatus(state) {
  switch (String(state ?? '')) {
    case 'answering': return 'answering'
    case 'serving': return 'serving'
    case 'hot': return 'loaded'
    case 'loading': return 'loading'
    case 'downloading from central': return 'pulling'
    case 'failed': return 'failed'
    default: return 'not loaded'
  }
}

// One per-(model, worker) status entry → { tier, residency, label } in the
// unified vocabulary. `tierOverride` lets a caller that holds better disk
// evidence (the worker's own storage survey) win over the backend's word.
export function unifiedStatusView(w, tierOverride = null) {
  const state = w?.state
  const base = w?.base || state
  const tier = tierOverride || tierFromStatusBase(base) || tierFromStatusBase(state)
  const residency = residencyFromStatus(state)
  const rv = RESIDENCY_VIEW[residency]
  // 'failed: <class>' keeps the class the backend put in the label.
  const resWord = residency === 'failed' && w?.label ? w.label : rv.word
  const tierWord = tier ? tier : (state === 'n/a' ? 'n/a' : 'unknown')
  return { tier, residency, icon: rv.icon, tone: rv.tone, pill: rv.pill,
           label: state === 'n/a' ? 'n/a' : `${resWord} · ${tierWord}` }
}
