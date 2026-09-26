import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchJson } from '../../api'
import { modelTask, modelTasks } from '../ModelTable/ModelTable'
import FixDoc from '../FixDoc/FixDoc'
import useSessionState from '../../hooks/useSessionState'
import useColumnLayout from '../../hooks/useColumnLayout'
import usePriorityGroups, { pgIndexOf, pgKeyForms } from '../../hooks/usePriorityGroups'
import { AllocModeMenu, BulkAllocControl } from './AllocationControls'
import { allocModeLabel, deriveAllocMode } from './allocation'
import {
  SERV_COMPACT_COLS,
  SERV_COMPACT_PX,
  SERV_DEFAULT_ORDER,
  SERV_FROZEN_COL,
  SERV_LAYOUT_KEY,
  WP_GIB,
} from './constants'
import { fmtBytes, fmtServed, midTrunc } from './formatters'
import { ExternalLeases } from './ExternalLeases'
import { ResidencyMenu } from './ResidencyMenu'
import { ResourceStrip } from './ResourceStrip'
import { SpillBadge } from './SpillBadge'
import { useNarrowContainer } from './useNarrowContainer'
import { isMeasuredResident } from './workerMetrics'
import { findCatalogRow } from './catalogRow'
import { getServing } from '../ModelTable/servingCache'
import { WorkerLoadTable } from './WorkerLoadTable'
import { useModelStatus } from '../ModelTable/useModelStatus'
import { statusFor } from '../ModelTable/modelStatus'
import { WorkerStateChips } from '../ModelTable/StatusCells'
import { responseReason } from '../responseReason'

// PIN state for one (worker, model), mirroring the backend's effective_pin
// (central/workers.py): central's recorded decision on the ``designations`` row
// wins; the legacy agent-side 📌 (worker.config.pinned) is read through ONLY
// while central holds no decision (pin_origin absent). Returns a bool.
export function effectivePin(worker, key) {
  const row = (worker?.designations || []).find(d => d && d.model_key === key)
  if (row && row.pin_origin === 'central') return !!row.pinned
  return !!(worker?.config?.pinned || {})[key]
}

// A worker row: status + GPUs (with used/free) + provisioning state, the models
// it serves with per-model load state + concise GPU allocation + free controls.
export function WorkerRow({ worker, models, allocation, onAssign, onLoad, onUnassign, onRemove, onFree, onFreeAll, onFreeRam, onRestart, onUpdate = null, onAdmit, onBlock, onSetPool, onSetLimits, onSetConfig, onSetResidency, onSetResidencyMany, onSetAllocMany, onTogglePin, onPinAll, onUnpinAll, onPruneDesignations, onReap, onApproveEvictions, onEvict, onAllocateMany, onRefresh = null, applying = false, restarting = false, updating = false, blockedKeys = null, onToggleBlock = null, distMode = 'feasible' }) {
  // The shared per-(model, worker) state vocabulary (GET /llm/models/status):
  // the same words the Models table and the Metrics picker use. Feature-
  // detected — an older central keeps the legacy pill below.
  const mstatus = useModelStatus()
  const [pick, setPick]       = useState('')
  const [newSpill, setNewSpill] = useState({})   // allocation for the next assign
  const [allocMenu, setAllocMenu] = useState(null)   // model key whose in-place alloc menu is open
  const allocAnchorRef = useRef(null)                // the open menu's trigger button (for fixed positioning)
  // Optimistic per-model alloc override: key -> spill dict, applied on top of
  // the derived mode so the cell updates the instant a mode is picked; reverted
  // (deleted) if the /assign POST rejects, and cleared once the refetch lands.
  const [allocOptimistic, setAllocOptimistic] = useState({})
  // Optimistic bitsandbytes toggles: key -> bool, applied on top of the worker
  // payload so the checkbox flips instantly while the POST + refetch land. The
  // Alloc cell beside it re-derives from the SERVER's answer, so the two settle
  // together on the next workers refresh.
  const [bnbOptimistic, setBnbOptimistic] = useState({})
  // The models on THIS worker that can actually take the specialization. Drives
  // both the bulk buttons' visibility and their count, so the label never
  // promises to change models it will skip.
  const bnbEligibleKeys = useMemo(
    () => Object.keys(worker.bnb_available || {}),
    [worker.bnb_available])

  // Bulk apply/clear. Sequential rather than Promise.all: each POST is a
  // registry write on the same worker record, and firing 44 concurrent writes
  // at one file-locked store is how you get lost updates. 44 small writes take
  // ~a second and the optimistic ticks make it feel instant anyway.
  const [moeOptimistic, setMoeOptimistic] = useState({})
  const setMoe = useCallback(async (modelKey, value) => {
    setMoeOptimistic(o => ({ ...o, [modelKey]: value === null ? undefined : value }))
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/moe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey, value }),
      })
    } catch (e) {
      setMoeOptimistic(o => { const n = { ...o }; delete n[modelKey]; return n })
      alert(`MoE split failed: ${e.message}`)
    }
    // Refetch: the Alloc column re-derives off this (forcing the split off drops
    // coder-next from explicit to max-ram), so the two must settle together.
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, onRefresh])

  const moeCapableKeys = useMemo(
    () => Object.keys(worker.moe_capable || {}), [worker.moe_capable])

  const setMoeMany = useCallback(async (value) => {
    if (!moeCapableKeys.length) return
    try {
      // One server-side sweep over the capable set — same reasoning as the 4-bit
      // bulk: the client's payload can lag a just-applied change.
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/moe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true, value }),
      })
    } catch (e) {
      alert(`MoE bulk failed: ${e.message}`)
    }
    setMoeOptimistic({})
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, moeCapableKeys, onRefresh])

  const setBnbMany = useCallback(async (enabled) => {
    const keys = Object.keys(worker.bnb_available || {})
    if (!keys.length) return
    if (enabled && !confirm(
      `Load ${keys.length} model(s) on ${worker.name || 'this worker'} at 4-bit `
      + '(bitsandbytes nf4)?\n\nEach is re-priced at ~30% of its fp16 size, so '
      + 'their allocations re-derive — several may move from RAM onto the GPU. '
      + 'Quantization costs some output quality.')) return
    setBnbOptimistic(o => {
      const n = { ...o }
      for (const k of keys) n[k] = enabled
      return n
    })
    try {
      // ONE server-side sweep, not N client POSTs. The server resolves the
      // eligible set itself, so the result cannot depend on how stale this
      // render's worker payload is — the bug that left 19 of 44 rows enabled
      // while the UI showed none ticked.
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/bnb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all: true, enabled }),
      })
    } catch (e) {
      setBnbOptimistic({})
      alert(`4-bit bulk failed: ${e.message}`)
    }
    if (typeof onRefresh === 'function') onRefresh()
  }, [worker.id, worker.name, worker.bnb_available, onRefresh])

  const setBnb = useCallback(async (modelKey, enabled) => {
    setBnbOptimistic(o => ({ ...o, [modelKey]: enabled }))
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/bnb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey, enabled }),
      })
      // Refetch so the Alloc column picks up the RE-DERIVED mode; the optimistic
      // entry is dropped once the authoritative payload carries the new value.
      if (typeof onRefresh === 'function') onRefresh()
    } catch (e) {
      setBnbOptimistic(o => { const n = { ...o }; delete n[modelKey]; return n })
      alert(`Specialization failed: ${e.message}`)
    }
  }, [worker.id, onRefresh])

  // Per-worker WILDCARD ("take all comers") routing opt-in. Optimistic-then-
  // revert like the model-groups tick: the console has no client-side operator
  // flag (the gate is server-side, operator_auth._SENSITIVE), so flip, POST, and
  // on refusal snap back with the server's message. Under Feasible distribution
  // this flag is nearly moot (any feasible worker is already a candidate); it
  // matters under Designated mode, which the label/hint says. worker.wildcard is
  // the authoritative value; the optimistic override wins until the next refetch.
  const [wildcardOpt, setWildcardOpt] = useState(undefined)  // undefined = follow payload
  const wildcardOn = wildcardOpt === undefined ? !!worker.wildcard : wildcardOpt
  const setWildcard = useCallback(async (enabled) => {
    setWildcardOpt(enabled)
    try {
      await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/wildcard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      if (typeof onRefresh === 'function') onRefresh()
    } catch (e) {
      setWildcardOpt(undefined)   // revert to the payload value
      alert(`Wildcard toggle failed: ${e.message}`)
    }
  }, [worker.id, onRefresh])
  // Drop the optimistic override once the authoritative payload agrees.
  useEffect(() => {
    if (wildcardOpt !== undefined && !!worker.wildcard === wildcardOpt) setWildcardOpt(undefined)
  }, [worker.wildcard, wildcardOpt])
  // Serving-detail cache (rulings 1+2, 2026-07-24): the per-(model,worker)
  // FEASIBLE mode set and the feasibility-DERIVED default live ONLY on
  // /llm/serving/<key> (alloc_by_worker / alloc_mode_derived), NOT on the
  // /llm/workers list this panel polls. Rather than fetch that for every serving
  // row (a per-row request storm on a worker with 100+ models), we fetch it
  // LAZILY — once, the first time a model's alloc menu is opened — and cache it
  // by model_key. The menu is the only place feasibility/derived_default are
  // needed; a closed row shows the derived-vs-pinned distinction from the local
  // override alone (see the alloc cell), no fetch required. Cache survives the
  // 10s poll (it's keyed by model_key, orthogonal to the worker refresh).
  const [servDetail, setServDetail] = useState({})   // key -> serving-GET row (or {error})
  const servFetchedRef = useRef(new Set())           // keys already fetched (dedupe)
  const fetchServDetail = useCallback((key) => {
    if (servFetchedRef.current.has(key)) return       // fetch each key at most once
    servFetchedRef.current.add(key)
    setServDetail(prev => ({ ...prev, [key]: { loading: true } }))
    getServing(key)   // shared, deduped, 20 s cache (ModelTable/servingCache.js)
      .then(row => setServDetail(prev => ({ ...prev, [key]: row || {} })))
      .catch(e => {
        // On failure, forget the key so a later open retries — feasibility is a
        // fail-OPEN read (missing data offers every mode), never a hard error.
        servFetchedRef.current.delete(key)
        setServDetail(prev => ({ ...prev, [key]: { error: e.message } }))
      })
  }, [])
  const [resMenu, setResMenu] = useState(null)   // model key whose residency picker is open
  const [limitsOpen, setLimitsOpen] = useState(false)
  const [limitsForm, setLimitsForm] = useState({ ram_max_gib: '', gpu_mem_gib: '', disk_cache_gib: '', threads: '' })
  const [ping, setPing]       = useState(null)   // null | 'checking' | {reachable, error}
  const [showLoad, setShowLoad] = useState(false) // reveal the load-a-model editor
  const [activating, setActivating] = useState(null) // model key being activated (loaded now)
  // Serving TABLE controls — a comprehensive, sortable view of the models
  // designated to THIS worker (same idiom as the "load a model" table). Search
  // and task-filter are per-mount (ephemeral, reset on unmount); the SORT is
  // session-sticky so the operator's chosen ordering survives a re-render.
  const [servQ, setServQ] = useState('')          // text search: model name / key
  const [servTask, setServTask] = useState('')    // task dropdown ('' = all tasks)
  const [servSort, setServSort] = useSessionState('hugpy.sess.wp.serving.sort', 'name')
  const [servDir, setServDir]   = useSessionState('hugpy.sess.wp.serving.dir', 'asc')
  // Multi-SELECT for bulk residency (todo t12): a Set of selected model_keys,
  // per-worker (this component instance IS one worker). Kept in local state
  // keyed by model_key, so it survives the panel's 10s load() refresh untouched
  // (the poll replaces the `worker` prop's data, never this selection); a model
  // that's since been unassigned is pruned lazily at apply time (the backend
  // also drops off-worker keys). Ephemeral: resets on unmount, like servQ.
  const [servSel, setServSel] = useState(() => new Set())
  const toggleSel = useCallback((key) => {
    setServSel(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }, [])
  const clearSel = useCallback(() => setServSel(new Set()), [])
  // Bulk-alloc editor (todo t15): the inline AllocControl expansion in the bulk
  // bar is open/closed by this flag (mirrors the per-row `editing` toggle).
  const [bulkAllocOpen, setBulkAllocOpen] = useState(false)

  // In-place alloc apply (operator ask 2026-07-24): a picked mode POSTs the
  // {alloc_mode, …} spill via the SAME /assign path the old editor used
  // (onAssign). Optimistic: stamp the spill locally so the cell flips instantly,
  // then reconcile — on success the refetch's spill_by_model becomes the truth
  // and we drop the optimistic entry; on error we drop it too (revert to the
  // unchanged server value; onAssign already surfaced the reason).
  const applyAllocMode = useCallback((key, spill) => {
    setAllocMenu(null)
    setAllocOptimistic(prev => ({ ...prev, [key]: spill }))
    Promise.resolve(onAssign(worker, key, spill))
      .finally(() => setAllocOptimistic(prev => {
        const next = { ...prev }; delete next[key]; return next
      }))
  }, [onAssign, worker])

  const checkHealth = useCallback(async () => {
    setPing('checking')
    try {
      const r = await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/health`)
      setPing(r)
    } catch (e) {
      setPing({ reachable: false, error: e.message })
    }
  }, [worker.id])

  const assignable = useMemo(() => {
    const assigned = new Set(worker.models || [])
    return models.filter(m => !assigned.has(m.model_key ?? m.key))
  }, [models, worker.models])

  const nameFor = useCallback(
    key => findCatalogRow(models, key)?.name || key,
    [models],
  )

  // A model's TRUE on-disk size from the /models feed: for GGUF the ONE effective
  // quant that serves (size_bytes/effective_bytes), never the all-quants dir sum.
  // Used to label serving rows with the real footprint instead of loaded_detail's
  // dir bytes (which double-counts sibling quants for GGUF).
  const sizeInfo = useCallback((key) => {
    const m = findCatalogRow(models, key)
    if (!m) return null
    const fw = String(m.framework || '').toLowerCase()
    const eff = m.size_bytes != null ? m.size_bytes : m.effective_bytes
    return {
      bytes: eff != null ? Number(eff) : null,
      isGguf: fw === 'gguf' || fw === 'llama_cpp',
      effGguf: m.effective_gguf,
    }
  }, [models])

  const loaded      = new Set(worker.loaded_models || [])
  const provisioning = new Set(worker.provisioning || [])
  const heating     = new Set(worker.loading || [])   // weights load in flight
  // Unified, engine-agnostic allocation view (new agents): {kind:"slot"|"ram"}
  // per resident model. When present it is the source of truth for a model's
  // residency KIND — a slot occupant (GGUF seat) and an in-RAM transformers
  // resident are reported the same way. When ABSENT (older agents) we fall
  // back to the legacy slots+loaded_models derivation below.
  const allocByKey = Array.isArray(worker.allocations)
    ? Object.fromEntries(worker.allocations.filter(a => a && a.model_key).map(a => [a.model_key, a]))
    : null
  // Residency KIND (legacy fallback): loaded_models is a merge of in-process
  // residents and slot-hosted models — the slots list tells them apart.
  // Slot-hosted = "serving" (routable supervised child); in-process =
  // "loaded" (resident in the agent itself, not in a slot).
  const slotServed  = new Set((worker.slots || [])
    .filter(s => s && s.model_key && s.healthy).map(s => s.model_key))
  // Live attribution: models whose slot is mid-request RIGHT NOW.
  const slotBusy    = new Set((worker.slots || [])
    .filter(s => s && s.model_key && s.busy).map(s => s.model_key))
  // UTIL-08 disk-truth: null until the worker reports it (older agents).
  const localSet = worker.models_local ? new Set(worker.models_local) : null

  // Per-model live-state derivation for THIS worker. Lifted verbatim out of the
  // Serving row render so the serving TABLE can BOTH sort by the derived state
  // (severity order) and render the same pill — the attribution ladder below is
  // byte-for-byte the operator-vetted logic it always was, only relocated.
  const deriveModelState = (key) => {
    const isPulling = provisioning.has(key)
    const isHeating = heating.has(key)
    const override = worker.spill_by_model?.[key]
    // The BACKEND's derived default for this (worker, model). Without it the
    // alloc cell fell back to deriveAllocMode's hardcoded 'max-gpu' for every
    // model with no persisted contract — 62 of ae's 64 — so a 67 GiB
    // transformers model that the operator's decision tree correctly resolves
    // to ram-only still displayed "⚡ Max GPU · auto". The tree was right and
    // already shipped; its answer simply never reached this cell.
    const derivedMode = worker.model_alloc_modes?.[key] || null
    // 📌 pin = PERMANENT attribution to this worker — blocks unassign. Central's
    // recorded decision (designations[].pinned) wins; the legacy agent 📌 reads
    // through only while central holds none (effectivePin).
    const isPinned = effectivePin(worker, key)
    // Full attribution ladder (each stage reported by the worker):
    //   pulling n%  — files downloading from central/HF (live progress)
    //   heating     — weights loading into VRAM/RAM right now
    //   serving     — hosted in a SLOT (routable supervised child)
    //   loaded      — resident IN-PROCESS on this machine (no slot)
    //   cold        — assigned; loads on the first request for it (or an explicit Load)
    // Residency KIND, engine-agnostic: prefer the unified allocations
    // view (a slot occupant OR an in-RAM transformers resident both count
    // as "resident"); fall back to the legacy slots+loaded_models sets.
    const alloc = allocByKey ? allocByKey[key] : undefined
    let inSlot, isAnswering, isServing
    // isIdleResident: a ram allocation EXISTS but the worker reports NO
    // measured footprint for it (no VRAM, no RSS, no cuda/cpu device, not
    // served recently). That is a hollow runner-cache entry — e.g. what an
    // evict/unload can leave behind, and the flap the operator saw:
    // it must read as a distinct, subtle idle state, NEVER as purple loaded.
    let isIdleResident = false
    if (allocByKey) {
      inSlot = !!alloc && alloc.kind === 'slot' && !!alloc.healthy
      isAnswering = inSlot && !!alloc.busy
      const ramResident = !!alloc && alloc.kind === 'ram' && isMeasuredResident(alloc)
      // "Loaded" derives from MEASURED residency, not from the mere presence
      // of a ram allocation (its runner-cache membership) — see the flap note.
      isServing = inSlot || ramResident
      isIdleResident = !!alloc && alloc.kind === 'ram' && !ramResident
    } else {
      isServing = loaded.has(key)
      inSlot = slotServed.has(key)
      isAnswering = inSlot && slotBusy.has(key)
    }
    const prog = worker.provision_progress?.[key]
    const pct = prog && prog.total_bytes > 0
      ? Math.min(99, Math.round(100 * (prog.done_bytes ?? prog.frac * prog.total_bytes) / prog.total_bytes))
      : null
    // NOT-RESIDENT tier (operator semantics ruling 2026-08-13) — three honest
    // states instead of the old binary, because "missing" used to mean merely
    // "not on this worker's drive" and read as an error for models that were
    // sitting safely in central's store all along (discovery catalogs from
    // llm_storage; the worker store is a lazy cache of it):
    //   🌡 hot     — files on THIS WORKER's drive; loads on first request.
    //   ○ cold    — files on CENTRAL storage only (catalog status:installed);
    //               they transfer to the worker on first call (lazy doctrine —
    //               assignment is attribution, not a transfer order).
    //   ○ missing — files NOWHERE (no central files either): a stale catalog
    //               row, a phantom assignment, or a download that never
    //               happened. The only one of the three that needs an operator.
    //
    // `isPulling` is live-only (central gates it on owner-alive AND
    // bytes-moving), so a dead/queued entry correctly FALLS THROUGH to these
    // states instead of masking as a phantom ⏳ pulling forever.
    const catRow = findCatalogRow(models, key)
    const centralHas = !!catRow
      && (catRow.status === 'installed' || (catRow.dir_bytes ?? 0) > 0)
    const onWorkerDisk = localSet != null && localSet.has(key)
    const notResident = !isPulling && !isHeating && !isServing && !isIdleResident
    // DISK TRUTH ONLY (operator ruling 2026-09-10: "the ui needs to simply
    // reflect whats actually going on. if the model is not on the disk, then
    // its missing, if its on the disk, then its not"). "The disk" means
    // ANYWHERE REAL: this worker's drive OR central's llm_storage. So:
    //   🌡 hot      — on THIS worker's drive
    //   ○ central  — on central storage; copies to the worker on first call
    //   ○ missing  — files NOWHERE. The only state that needs an operator.
    // 'missing' may NEVER show for files that exist somewhere — that was the
    // sam749~flux-klein-q4 complaint (on llm_storage, displayed as missing).
    const isMissing = notResident && !onWorkerDisk && !centralHas
    const state = isPulling ? 'pulling' : isHeating ? 'heating'
      : isServing ? (inSlot ? (isAnswering ? 'answering' : 'serving') : 'loaded')
      : isIdleResident ? 'idle'
      : onWorkerDisk ? 'hot'
      : centralHas ? 'central'
      : 'missing'
    const stateTitle = isPulling ? `downloading files from central/HF${pct != null ? ` — ${pct}%` : ''}`
      : isHeating ? 'weights loading into VRAM/RAM right now'
      : isServing ? (inSlot
        ? (isAnswering
          ? 'actively processing a request right now'
          : 'hosted in a slot on this worker — routable, crash-isolated server child')
        : 'resident in this worker\'s own process — dedicated to this machine, not in a slot')
      : isIdleResident ? 'a runner is cached on this worker but holds NO measured VRAM/RAM and hasn\'t served recently — not actually resident (its weights were freed, e.g. by an evict/unload). Its measured residency is the truth here, not the runner-cache membership.'
      : state === 'hot' ? 'files on THIS worker\'s drive, not loaded — weights lift into VRAM/RAM on the first request'
      : state === 'central'
        ? 'files on CENTRAL storage (llm_storage), not on this worker\'s drive yet — they copy to this worker on the FIRST CALL (lazy download). Not missing: the files exist.'
        : 'files NOWHERE — not on this worker\'s drive and not on central storage either: a stale catalog row, a phantom assignment, or a download that never happened. Serving will fail until the files are downloaded (or the key is unassigned).'
    // load_reports[key] is the recorded outcome of the LAST warm/probe attempt
    // central made on this (worker, model) — additive, central-side. We annotate
    // it ONLY where the model is NOT resident right now (cold/missing/idle/
    // heating), so a "▶ activate did nothing" always carries a visible why; a
    // live serving/answering/loaded row needs none. The pill stays DERIVED from
    // measured residency (doctrine above) — this is an annotation, not a state.
    const loadReport = worker.load_reports?.[key]
    const showLoadWhy = !!loadReport
      && (state === 'hot' || state === 'central' || state === 'missing' || state === 'idle' || state === 'heating')
    // failed = the probe reported not-ok OR reported it won't fit; ok-stale = the
    // last warm succeeded yet the model has since gone non-resident (cold again).
    // A failure older than a day is history, not a warning: a "worker
    // unreachable" probe from weeks ago was decorating healthy hot rows with ⚠.
    const loadRecent = !(loadReport?.ts > 0) || (Date.now() / 1000 - loadReport.ts) < 86_400
    const loadFailed = showLoadWhy && loadRecent && (loadReport?.ok === false || loadReport?.fit === false)
    const loadStale  = showLoadWhy && !loadFailed && loadReport?.ok === true
    return { isPulling, isHeating, override, isPinned, alloc, inSlot, derivedMode,
             isAnswering, isServing, isIdleResident, pct, isMissing, state, stateTitle,
             centralHas, loadReport, loadFailed, loadStale }
  }

  // Severity order for the State column sort: the more "live" a model is, the
  // higher it ranks (answering ▸ serving ▸ loaded ▸ heating ▸ pulling ▸ idle ▸
  // hot ▸ cold ▸ missing) — the same ladder the pills read top-to-bottom.
  // missing ranks LAST now: it is the only state that needs an operator, and
  // the "worst" sort surfaces it at the bottom edge either direction.
  const SERV_STATE_RANK = { answering: 8, serving: 7, loaded: 6, heating: 5, pulling: 4, idle: 3, hot: 2, central: 1, missing: 0 }
  // ONE unified, ORDERED column model — the single source both the <th> row and
  // every <td> render from, so a column can be dragged anywhere across the whole
  // set (the data columns and the control columns are no longer two frozen
  // groups). Each def carries: whether it's SORTABLE (the six load-table
  // identifiers stay sortable via the existing servSort/servDir; the control
  // columns stay non-sortable), `num` (right-aligned numeric), the <td> class +
  // optional dynamic title, and a `render(cx)` that returns the cell body. `cx`
  // is the per-row context { key, m, d, isSel, isBlocked, need } (d = the
  // deriveModelState bundle). The renders are the SAME JSX the fixed layout used
  // — only relocated, so pills / Alloc editor / action buttons behave identically.
  const SERV_COL_DEFS = {
    task: {
      label: 'Task', sortable: true, cls: 'wp-lt-muted',
      title: ({ m }) => (m ? modelTasks(m).join(', ') : ''),
      // first task + "+N", full list on hover; '—' when not in the catalog feed
      render: ({ m }) => (m ? `${modelTask(m)}${modelTasks(m).length > 1 ? ` +${modelTasks(m).length - 1}` : ''}` : '—'),
    },
    framework: {
      label: 'Engine', sortable: true, cls: 'wp-lt-muted',
      render: ({ m }) => m?.framework || '—',
    },
    seat: {
      // Seat — the allocation-kind badge (engine-agnostic): a slot seat or the
      // worker's own RAM. Only present when the worker reports the unified
      // allocations view; '—' otherwise.
      label: 'Seat', sortable: false, cls: 'wp-servtable-seat',
      render: ({ d }) => (d.alloc ? (
        <span className={`wp-alloc-kind wp-alloc-${d.alloc.kind}`}
              title={d.alloc.kind === 'slot'
                ? 'allocated a slot (a resource seat) on this worker'
                : 'resident in the worker’s own process (RAM allocation)'}>
          {d.alloc.kind === 'slot' ? '🎰 slot' : '🧠 RAM'}
        </span>
      ) : <span className="wp-lt-muted">—</span>),
    },
    residency: {
      // Residency POLICY tag (v3 + slice 8): on-demand (default) or static.
      // Clicking only OPENS the picker (rendered in the expansion row).
      label: 'Residency', sortable: false, cls: '',
      render: ({ key }) => (worker.config && onSetResidency ? (() => {
        const mode = worker.config.residency?.[key] === 'static' ? 'static' : 'on-demand'
        const desc = mode === 'static'
          ? ' (locked seat — never swapped out; permanent with 📌 pin)'
          : ' (default — loads on call; holds its slot until another model needs the seat)'
        return (
          <button className={`wp-residency wp-residency-${mode}`}
                  disabled={applying}
                  title={applying
                    ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                    : `Residency policy: ${mode}${desc} — click to choose (a change applies via a ~5s agent restart)`}
                  onClick={() => setResMenu(resMenu === key ? null : key)}>
            {mode === 'static' ? '🔒 static' : '⏲ on-demand'}
          </button>
        )
      })() : <span className="wp-lt-muted">—</span>),
    },
    pin: {
      // Tiers v3 — 📌 pin = PERMANENT ATTRIBUTION of the model to this worker:
      // blocks unassign, files never reaped, residency overrides survive.
      label: '📌', sortable: false, cls: '',
      render: ({ key, d }) => (worker.config && onTogglePin ? (
        <button className={`wp-pin${d.isPinned ? ' wp-pin-on' : ''}`}
                disabled={applying}
                title={applying
                  ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                  : d.isPinned
                    ? 'Pinned: this allocation survives restarts — routing to this worker is durable and unassign is refused (409). Does not download the model and does not protect its files from eviction (bytes re-pull on call). Click to unpin.'
                    : 'Pin: make this allocation survive restarts — routing to this worker becomes durable and unassign is refused (409). Does not download the model or protect its files from eviction.'}
                onClick={() => onTogglePin(worker, key, !d.isPinned)}>
          📌{d.isPinned ? '' : '?'}
        </button>
      ) : <span className="wp-lt-muted">—</span>),
    },
    name: {
      // Model — nameFor(key), MIDDLE-truncated (both ends of a model key carry
      // the distinguishing part); full key on hover, and full untruncated in the
      // compact drawer. A ⛔ blocked chip rides here when the model is blocked
      // from the serving pool. The character budget tightens in compact mode,
      // where the column is only ~28vw of a phone-width table.
      label: 'Model', sortable: true, cls: 'wp-servtable-name',
      title: ({ key }) => key,
      render: ({ key, isBlocked, compact }) => (
        <>
          {/* Compact keeps the full 14-char TAIL (it is what discriminates
              sibling repos) and gives up head characters instead — the compact
              cell wraps rather than ellipsising, so 23 chars costs two lines,
              not a cut-off tail. */}
          {compact ? midTrunc(nameFor(key), 8, 14) : midTrunc(nameFor(key))}
          {isBlocked && (
            <span className="wp-blocked-chip"
                  title="⛔ Blocked from the serving pool by the operator — not routed to, assigned, warmed, or used as a fallback anywhere. This designation stays recorded but inert (block outranks pin). Files are untouched. Use the ⛔ action to unblock.">
              ⛔ blocked
            </span>
          )}
        </>
      ),
    },
    size: {
      // Size — SAME effective-quant rule as the serving facts (sizeInfo → the
      // ONE quant that serves for GGUF), honest on-disk title; '—' when unknown.
      label: 'Size', sortable: true, num: true, cls: 'wp-lt-num wp-lt-size',
      title: ({ key }) => {
        const si = sizeInfo(key)
        const det = worker.loaded_detail?.[key]
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        return declared == null ? 'size unknown — not on disk / not reported by the feed'
          : `${fmtBytes(declared)} on disk${si && si.isGguf ? ` (effective quant${si.effGguf ? ` ${si.effGguf}` : ''}, not the all-quants dir sum)` : ''}`
      },
      render: ({ key }) => {
        const si = sizeInfo(key)
        const det = worker.loaded_detail?.[key]
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        return declared == null ? '—' : fmtBytes(declared)
      },
    },
    ctx: {
      label: 'Ctx', sortable: true, num: true, cls: 'wp-lt-num',
      render: ({ m }) => m?.model_max_length || '—',
    },
    state: {
      // State — the EXISTING pill, verbatim (FixDoc on missing, live pulling %,
      // load_reports annotation when not resident).
      label: 'State', sortable: true, cls: 'wp-servtable-state',
      render: ({ d, key }) => {
        const { isPulling, isHeating, isServing, inSlot, isAnswering, isIdleResident,
                isMissing, centralHas, pct, state, stateTitle, loadFailed, loadReport, loadStale } = d
        const srow = mstatus.index.available ? statusFor(mstatus.index, key) : null
        if (srow && (srow.workers || []).some(w => w.worker === worker.name)) {
          return <WorkerStateChips row={srow} only={worker.name} />
        }
        return (
          <>
            <span className={`wp-state-pill wp-pill-${state}`} title={stateTitle}>
              {isPulling ? `⏳ pulling${pct != null ? ` ${pct}%` : ''}`
                : isHeating ? '🔶 heating'
                : isServing ? (inSlot ? (isAnswering ? '⚡ answering' : '🔥 serving') : '📌 loaded')
                : isIdleResident ? '◍ idle'
                : state === 'hot' ? '🌡 hot'
                : state === 'central' ? '○ central'
                : '○ missing'}
              {/* FixDoc only on true missing — 'central' self-heals on first
                  call, no operator needed. */}
              {isMissing && <FixDoc doc="worker-model-missing" />}
            </span>
            {loadFailed && (
              <span className="wp-loadwhy wp-loadwhy-bad"
                    title={`${loadReport?.error || (loadReport?.fit === false ? 'probe: fit=false' : 'warm failed — the model never became resident')}${loadReport?.ts ? ` · ${fmtServed(loadReport?.ts)}` : ''}`}>
                ⚠
              </span>
            )}
            {loadStale && (
              <span className="wp-loadwhy wp-loadwhy-ok"
                    title={`last warm succeeded${loadReport?.ts ? ` ${fmtServed(loadReport?.ts)}` : ''} — not resident now`}>
                ⓘ
              </span>
            )}
          </>
        )
      },
    },
    moe: {
      // MoE — the EXPERT-SPLIT lever (operator ask 2026-07-26). Tri-state, but
      // presented as a plain checkbox on purpose: AUTO renders TICKED whenever
      // the derivation actually produced a split, so the operator sees the real
      // behaviour instead of an empty box that secretly means "on"
      // ("defaults can remain auto and should, but that also should entail a
      // checked box under the correct column, that could be switched by the
      // user"). Clicking pins the opposite state; ⟲ returns it to auto.
      label: 'MoE', sortable: false, cls: '',
      render: ({ key }) => {
        if (!worker.moe_capable?.[key]) {
          return <span className="wp-4bit-na" title={
            'No expert structure — this model is dense, so there is nothing to '
            + 'split.'}>—</span>
        }
        const ov = worker.moe_by_model?.[key]
        const pinned = ov !== undefined && ov !== null
        const on = key in moeOptimistic ? moeOptimistic[key]
                 : (pinned ? !!ov : !!worker.moe_effective?.[key])
        return (
          <span className="wp-moe">
            <label title={
              pinned
                ? `Expert split PINNED ${on ? 'on' : 'off'} by you. Click to flip; ⟲ restores auto.`
                : (on
                  ? 'Expert split ACTIVE, derived automatically — experts in RAM, everything else on the GPU. Untick to force it off.'
                  : 'Capable of an expert split, but the derivation did not apply one here (transformers MoE has no split path yet). Tick to force it on.')}>
              <input type="checkbox" className="wp-moe-box" checked={on}
                     disabled={applying}
                     onChange={e => setMoe(key, e.target.checked)} />
            </label>
            {pinned && (
              <button className="wp-moe-auto" disabled={applying}
                      title="Restore AUTO — follow the derivation for this model."
                      onClick={() => setMoe(key, null)}>⟲</button>
            )}
          </span>
        )
      },
    },
    fourbit: {
      // 4-BIT (operator ask 2026-07-26) — the bitsandbytes lever.
      // NAMED "4-bit", NOT "Quantize": the console already uses
      // "Quantization (GGUF variant)" on the Models tab (QuantControl) for a
      // DIFFERENT thing — picking WHICH pre-built GGUF quant file to download
      // (Q4_K_M vs Q8_0). This is load-time bitsandbytes on transformers
      // weights. Two distinct concepts must not share a name (operator ruling
      // 2026-07-26, keeper owns nomenclature).
      // Sits next to Alloc deliberately: it is a COMPRESSION choice that
      // re-prices the model, so the Alloc cell beside it re-derives the moment
      // this is ticked (67 GiB transformers: ram-only -> max-gpu at ~20 GiB).
      // Only rendered where it can actually work — not GGUF (llama.cpp carries
      // its own quantization), not a CPU-only worker (the 4-bit kernels are
      // CUDA-only), not an already-quantized repo. Elsewhere the cell is a
      // quiet em-dash rather than a disabled control nobody can use.
      label: '4-bit', sortable: false, cls: '',
      render: ({ key }) => {
        const avail = !!worker.bnb_available?.[key]
        if (!avail) {
          return <span className="wp-4bit-na" title={
            'No bitsandbytes specialization here: GGUF models carry their own '
            + 'quantization, the 4-bit kernels need a CUDA worker, and an '
            + 'already-quantized repo cannot be re-quantized.'}>—</span>
        }
        const on = key in bnbOptimistic ? bnbOptimistic[key] : !!worker.bnb_by_model?.[key]
        return (
          <label className="wp-4bit" title={
            on ? 'bitsandbytes 4-bit (nf4) ON — the model is priced at ~30% of '
               + 'its fp16 size, so its allocation re-derives. Untick to restore '
               + 'full precision.'
               : 'Load this model with bitsandbytes 4-bit (nf4). It is priced at '
               + '~30% of its fp16 size, so the Alloc column re-derives — often '
               + 'turning a RAM-only model into one that fits the GPU. Costs some '
               + 'quality.'}>
            <input type="checkbox" className="wp-4bit-box" checked={on}
                   disabled={applying}
                   onChange={e => setBnb(key, e.target.checked)} />
            <span className="wp-4bit-tag">{on ? '4-bit' : ''}</span>
          </label>
        )
      },
    },
    alloc: {
      // Alloc — the IN-PLACE mode picker (operator ask 2026-07-24). Clicking the
      // cell opens a compact menu anchored AT THE CLICK SITE (the five flat
      // modes); picking a zero-knob mode applies immediately, only "explicit"
      // opens extra chrome. The old full-width AllocControl expansion is retired.
      label: 'Alloc', sortable: false, cls: '',
      render: ({ key, m, d, need }) => {
        // Effective spill: an in-flight optimistic pick wins, else the persisted
        // override — so the cell flips the instant a mode is chosen.
        const effSpill = key in allocOptimistic ? allocOptimistic[key] : d.override
        // THE LABEL IS THE BACKEND's resolved mode (worker.model_alloc_modes ->
        // d.derivedMode), the ONE source of truth: central derives it from the
        // SAME spill it emits to the worker, WITH the MoE split overlaid, so a
        // stale gpu-only stamp on a MoE model reads 'explicit' here exactly as it
        // serves and as planned_split reports — never the bare-spill 'gpu-only'
        // this cell used to re-derive locally (which ignored the MoE overlay and
        // was the "gpu-only Alloc while MoE ticked" inconsistency). Only an
        // in-flight optimistic pick derives locally, so the cell still flips
        // instantly before the refetch lands; a pre-model_alloc_modes central
        // falls back to the local derivation.
        const mode = (key in allocOptimistic)
          ? deriveAllocMode(effSpill)
          : (d.derivedMode || deriveAllocMode(effSpill))
        const engineGguf = /^(gguf|llama_cpp)$/.test(String(m?.framework || '').toLowerCase())
        const isOpen = allocMenu === key
        // Ruling 2 (2026-07-24) — DERIVED vs PINNED at a glance. A model with NO
        // persisted contract TRACKS the derivation ("Auto"): the cell shows the
        // derived mode dimmed/italic with an "auto" affix. A persisted pick shows
        // solid. We decide this LOCALLY from the override (no fetch): an empty /
        // absent override == tracking-the-derivation; any non-empty override ==
        // an operator pin. (This is stricter-but-honester than the backend's
        // narrow alloc_mode_derived, which only inspects the alloc_mode key and
        // so calls a gpu-only/ram-only pin "derived" too — here an operator's
        // n_gpu_layers:-1 correctly reads as a pin. When the serving fetch has
        // landed we PREFER its alloc_mode_derived for the exact backend truth.)
        // NOTE the subtlety: explicit max-gpu ≠ blank. Blank tracks the
        // derivation as it improves (measured values land later via the
        // calibration evaluator); an explicit pick freezes the mode.
        const sd = servDetail[key]
        const isDerived = (sd && typeof sd.alloc_mode_derived === 'boolean' && !(key in allocOptimistic))
          ? sd.alloc_mode_derived
          : !(effSpill && Object.keys(effSpill).length > 0)
        // Per-(model,worker) feasibility + derived default, once the serving GET
        // for this key has landed. Fail-OPEN: undefined feasible ⇒ no disables.
        const byWorker = (sd && sd.alloc_by_worker && sd.alloc_by_worker[worker.id]) || null
        const feasibleUnion = (sd && Array.isArray(sd.alloc_modes_feasible)) ? sd.alloc_modes_feasible : null
        const feasible = byWorker && Array.isArray(byWorker.feasible) ? byWorker.feasible
          : feasibleUnion   // fall back to the model-level union when unscoped
        const derivedMode = (byWorker && byWorker.derived_default)
          || (sd && sd.alloc_mode_derived && sd.alloc_mode) || null
        return (
          <span className="wp-allocmode-anchor">
            <button className={`wp-alloc-edit${isDerived ? ' wp-alloc-derived' : ''}`} disabled={applying}
                    ref={isOpen ? allocAnchorRef : undefined}
                    title={applying
                      ? 'Agent is applying the previous change — retry in a few seconds.'
                      : isDerived
                        ? `Allocation: ${allocModeLabel(mode)} — DERIVED (no pinned contract; tracks the default as it improves). Click to change or pin.`
                        : `Allocation: ${allocModeLabel(mode)} — pinned. Click to change or revert to the derived default.`}
                    onClick={() => { setAllocMenu(isOpen ? null : key); if (!isOpen) fetchServDetail(key) }}>
              {allocModeLabel(mode)}
              {isDerived && <span className="wp-alloc-auto-affix"> · auto</span>}
            </button>
            {isOpen && (
              <AllocModeMenu
                mode={mode}
                spill={effSpill}
                worker={worker}
                need={need}
                engineGguf={engineGguf}
                feasible={feasible}
                feasibleCtx={{ modelBytes: need?.bytes ?? null,
                               vramTotal: worker.vram_total ?? null,
                               ramTotal: worker.ram_total ?? null }}
                derivedMode={derivedMode}
                anchorRef={allocAnchorRef}
                onPick={(next) => applyAllocMode(key, { alloc_mode: next })}
                onApplyExplicit={(s) => applyAllocMode(key, s)}
                onRevertDerived={() => applyAllocMode(key, {})}
                onClose={() => setAllocMenu(null)}
              />
            )}
          </span>
        )
      },
    },
    memory: {
      // Memory — the HONEST THREE-QUANTITY display (ruling 3, 2026-07-24):
      //   "<disk> disk · <resident> resident (<vram> VRAM + <anon> RAM) · <n>/<total> layers"
      // when the shipped 0.1.198+ fields are present on the row (vram_bytes,
      // rss_anon_bytes, n_gpu_layers/total_layers — all on the measured
      // allocations view d.alloc, with loaded_detail as fallback). Resident =
      // VRAM + anon RAM (the true footprint; disk is a separate universe). The
      // cell stays COMPACT — "disk · resident" — and the full breakdown rides
      // the title. Absent fields degrade to what exists (never invented): VRAM
      // 0 means resident-but-on-CPU; no measured figures fall back to the
      // declared-split line the cell always showed.
      label: 'Memory', sortable: false, cls: 'wp-servtable-mem',
      render: ({ key, d }) => {
        const det = worker.loaded_detail?.[key]
        const a = d.alloc || null
        // Prefer the measured allocations view; fall back to loaded_detail.
        const vram = a && a.vram_bytes != null ? a.vram_bytes
          : det && det.vram_bytes != null ? det.vram_bytes : null
        // Host-RAM occupancy: a slot child's anon RSS, or — for an in-process
        // (kind:'ram') model — the worker's MEASURED ram_resident_bytes (ships
        // with the next release; absent on older workers, which then degrade to
        // the disk-only line exactly as before). Either way this is a
        // measurement, never a declared/file figure.
        const anon = a && a.rss_anon_bytes != null ? a.rss_anon_bytes
          : a && a.ram_resident_bytes != null ? a.ram_resident_bytes
          : det && det.rss_anon_bytes != null ? det.rss_anon_bytes
          : det && det.ram_resident_bytes != null ? det.ram_resident_bytes : null
        const ngl = a && a.n_gpu_layers != null ? a.n_gpu_layers
          : det && det.n_gpu_layers != null ? det.n_gpu_layers : null
        const totalLayers = a && a.total_layers != null ? a.total_layers
          : det && det.total_layers != null ? det.total_layers : null
        const measuredVram = vram   // legacy name kept for the declared-split fallback below
        const si = sizeInfo(key)
        const declared = (si && si.bytes != null) ? si.bytes : det?.model_bytes
        if (declared == null && det?.gpu_pct == null && measuredVram == null) return <span className="wp-lt-muted">—</span>
        // Resident footprint = measured VRAM + anon RAM (only when at least one
        // measured figure exists; a bare declared size is NOT residency).
        const resident = (vram != null || anon != null)
          ? (vram || 0) + (anon || 0) : null
        const layersTxt = ngl == null ? ''
          : `${ngl === -1 ? 'all' : ngl}${totalLayers ? `/${totalLayers}` : ''} layers on GPU`
        const declaredTitle = declared != null
          ? `${fmtBytes(declared)} on disk${si && si.isGguf ? ` (effective quant${si.effGguf ? ` ${si.effGguf}` : ''}, not the all-quants dir sum)` : ''}`
          : ''
        // The FULL breakdown for the tooltip — the honest three quantities named.
        const fullTitle = [
          declaredTitle,
          resident != null ? `${fmtBytes(resident)} resident = ${vram != null ? fmtBytes(vram) : '0'} VRAM + ${anon != null ? fmtBytes(anon) : '0'} anon RAM` : '',
          layersTxt,
          (vram != null || anon != null)
            ? 'VRAM/RAM figures are MEASURED (nvidia-smi / rss_anon / ram_resident); VRAM 0 = running on CPU'
            : (det?.gpu_pct != null ? 'GPU split is DECLARED by the loader, not a measured VRAM read' : ''),
        ].filter(Boolean).join(' · ')
        // NOT-YET-RESIDENT: show the PLANNED division, not the on-disk size.
        // The old fallback rendered "<X> disk" — the exact number the Size
        // column already shows, which is the redundancy the operator flagged.
        // planned_split answers the question this column exists for ("where will
        // this actually go") and moves with the Alloc mode, the 4-bit lever and
        // the MoE lever, so flipping any switch visibly updates the row.
        const plan = worker.planned_split?.[key]
        if (resident == null && measuredVram == null && plan
            && (plan.gpu_bytes != null || plan.ram_bytes != null)) {
          const g = plan.gpu_bytes, r = plan.ram_bytes
          const parts = []
          if (g) parts.push(`${fmtBytes(g)} VRAM`)
          if (r) parts.push(`${fmtBytes(r)} RAM`)
          return (
            <span className="wp-model-facts wp-fact-planned" title={
              (plan.split
                ? `PLANNED expert split: ${fmtBytes(g || 0)} of non-expert tensors on the GPU, `
                  + `${fmtBytes(r || 0)} of experts in RAM. `
                : `PLANNED placement under '${plan.mode}': `)
              + (plan.split ? '' : `${fmtBytes(g || r || 0)} on ${g ? 'the GPU' : 'the CPU'}`
                 + (plan.mode === 'max-gpu' ? ' (spills whatever will not fit at load time)' : '')
                 + '. ')
              + 'Not resident yet — this is what the current Alloc mode and the '
              + '4-bit / MoE switches add up to, not a measurement.'}>
              {parts.join(' + ')}
              <span className="wp-fact-planned-tag"> planned</span>
            </span>
          )
        }
        return (
          <span className="wp-model-facts" title={fullTitle}>
            {declared != null && <span className="wp-fact-disk">{fmtBytes(declared)} disk</span>}
            {resident != null ? (
              <span className="wp-fact-resident">
                {' · '}{fmtBytes(resident)} resident
                <span className="wp-fact-split"> ({vram != null ? fmtBytes(vram) : '0'} VRAM + {anon != null ? fmtBytes(anon) : '0'} RAM)</span>
                {totalLayers != null && ngl != null && (
                  <span className="wp-fact-layers"> · {ngl === -1 ? totalLayers : ngl}/{totalLayers} layers</span>
                )}
              </span>
            ) : measuredVram != null
              ? (measuredVram > 0 ? ` · ${fmtBytes(measuredVram)} VRAM` : ' · 0 VRAM · on CPU')
              : det?.gpu_pct != null ? ` · ~${det.gpu_pct}% GPU / ${100 - det.gpu_pct}% spill` : ''}
          </span>
        )
      },
    },
    actions: {
      // Actions — ▶ activate (files exist SOMEWHERE: on this worker's drive,
      // or on central where the first call pulls them; a missing-nowhere row
      // has nothing to seat, its 📖 FixDoc is the affordance) / ⏏ free
      // (resident or hollow-idle) / × unassign (blocked while pinned) / ⛔ block.
      label: 'Actions', sortable: false, cls: 'wp-servtable-actions',
      render: ({ key, d, isBlocked }) => {
        const { state, isServing, isIdleResident, isPinned, centralHas } = d
        return (
          <>
            {onLoad && (state === 'hot' || state === 'central') && (
              <button className="wp-activate" disabled={activating === key}
                      title={activating === key
                        ? 'seating this model on the worker…'
                        : 'Activate: allocate this worker’s resources to the model now so it serves immediately. Files transfer first if not local yet.'}
                      onClick={async () => {
                        setActivating(key)
                        try { await onLoad(worker, key) }
                        finally { setActivating(null) }
                      }}>
                {activating === key ? '⏳ activating…' : '▶ activate'}
              </button>
            )}
            {(isServing || isIdleResident) && (
              <button className="wp-free"
                      title={isIdleResident
                        ? 'Clear this idle runner shell from the worker (stays assigned)'
                        : 'Unload from VRAM (stays assigned)'}
                      onClick={() => onFree(worker, key)}>⏏</button>
            )}
            <button className="wp-model-x" disabled={isPinned}
                    title={isPinned ? 'pinned — unpin first' : 'Unassign'}
                    onClick={() => onUnassign(worker, key)}>×</button>
            {onToggleBlock && (
              <button className={`wp-model-block${isBlocked ? ' wp-model-block-on' : ''}`}
                      title={isBlocked
                        ? 'Blocked from the serving pool — click to UNBLOCK (return it to routing). Block outranks pin; the designation is unchanged.'
                        : 'Block this model from the serving pool (global): never routed to / assigned / warmed / a fallback default anywhere. Files stay; designations stay (inert). Reversible.'}
                      onClick={() => onToggleBlock(key, !isBlocked)}>⛔</button>
            )}
          </>
        )
      },
    },
  }
  // Persistent, user-adjustable column order + widths (localStorage, per browser).
  const { order: servOrder, widths: servWidths, moveColumn: servMoveCol,
          setWidth: servSetWidth, reset: servResetLayout } = useColumnLayout(SERV_LAYOUT_KEY, SERV_DEFAULT_ORDER)
  const servColsAll = servOrder.map(k => ({ key: k, ...SERV_COL_DEFS[k] })).filter(c => c.label != null)
  // COMPACT MODE (operator ask 2026-07-28) — measured on the table's wrapper, so
  // a narrow panel inside a wide window compacts too. In compact the table keeps
  // only checkbox / Model / State / Memory (no horizontal scroll at all) and
  // every other column def moves into the per-row drawer, rendered from the SAME
  // def.render(cx) — the cell logic is never forked.
  const servWrapRef = useRef(null)
  const servCompact = useNarrowContainer(servWrapRef, SERV_COMPACT_PX)
  const servCols = servCompact
    // keep the operator's own left-to-right order among the surviving three
    ? servColsAll.filter(c => SERV_COMPACT_COLS.includes(c.key))
    : servColsAll
  // The columns that moved OUT of the table and into the drawer, in the same
  // order the operator arranged them (Actions is rendered last, on its own row).
  const servDrawerCols = servCompact
    ? servColsAll.filter(c => !SERV_COMPACT_COLS.includes(c.key) && c.key !== 'actions')
    : []
  const servActionsCol = SERV_COL_DEFS.actions
  // +1 for the leading select-checkbox column (todo t12 bulk residency).
  const SERV_COLSPAN = 1 + servCols.length
  // Which row's drawer is open (compact mode only) — ONE at a time: tapping the
  // same row closes it, tapping another moves it. Cleared whenever we leave
  // compact mode so a returning desktop layout never carries a stray row.
  const [servDrawer, setServDrawer] = useState(null)
  useEffect(() => { if (!servCompact) setServDrawer(null) }, [servCompact])
  // A row tap opens/closes the drawer — but ONLY when the tap landed on the row
  // BODY. Anything interactive (the select checkbox, a button, an input, a
  // select, a label, a link) keeps its own behaviour untouched.
  const servRowTap = (key) => (e) => {
    if (!servCompact) return
    const t = e.target
    if (t && typeof t.closest === 'function'
        && t.closest('button, input, select, textarea, label, a, .wp-servtable-selcol')) return
    setServDrawer(prev => (prev === key ? null : key))
  }
  // Drag-to-reorder: the key of the header currently being dragged, and the one
  // it's hovering over (for the drop-indicator). Native HTML5 drag on the <th>s;
  // the resize grip cancels its own dragstart so the two gestures never collide.
  const [servDragCol, setServDragCol] = useState(null)
  const [servDragOver, setServDragOver] = useState(null)
  // Drag-to-resize: pointer-drag a header's right edge, writing px width to the
  // hook (→ the <colgroup>). Listens on window so the drag survives leaving the
  // 6px grip.
  const startServResize = (e, key) => {
    e.preventDefault(); e.stopPropagation()
    const th = e.currentTarget.closest('th')
    const startX = e.clientX
    const startW = th ? th.getBoundingClientRect().width : (servWidths[key] || 120)
    const onMove = (ev) => servSetWidth(key, startW + (ev.clientX - startX))
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  // Task dropdown options: the union of EVERY task advertised by the models
  // ASSIGNED to this worker (a multi-task model surfaces under each of its
  // tasks — the same union rule the load table uses), narrowed to what's here.
  const servTasks = useMemo(() => {
    const assigned = new Set(worker.models || [])
    const here = models.filter(m => assigned.has(m.model_key ?? m.key))
    return [...new Set(here.flatMap(modelTasks))].sort()
  }, [models, worker.models])
  // The Size sort value MUST match the Size cell: the true effective-quant
  // footprint (sizeInfo → the ONE quant that serves for GGUF), falling back to
  // the worker's loaded_detail dir bytes; unknown (-1) sinks to the bottom asc.
  const servSizeOf = (key) => {
    const si = sizeInfo(key)
    const det = worker.loaded_detail?.[key]
    const b = (si && si.bytes != null) ? si.bytes : det?.model_bytes
    return b == null ? -1 : Number(b)
  }
  // One row per assigned model (state derived once so it can be BOTH sorted and
  // rendered), then the search + task filter, then the sort.
  const servRowsAll = (worker.models || []).map(key => ({
    key,
    m: findCatalogRow(models, key),
    d: deriveModelState(key),
  }))
  const servNeedle = servQ.trim().toLowerCase()
  let servRows = servRowsAll
  if (servNeedle) servRows = servRows.filter(({ key, m }) =>
    (m?.name || key).toLowerCase().includes(servNeedle) || key.toLowerCase().includes(servNeedle))
  // Full-list task match — a model counts under ANY task it advertises.
  if (servTask) servRows = servRows.filter(({ m }) => m && modelTasks(m).includes(servTask))
  const servGet = ({ key, m, d }) => {
    switch (servSort) {
      case 'task': return m ? (modelTask(m) || '') : ''
      case 'framework': return m?.framework || ''
      case 'size': return servSizeOf(key)
      case 'ctx': return m?.model_max_length || 0
      case 'state': return SERV_STATE_RANK[d.state] ?? -1
      case 'name':
      default: return m?.name || key
    }
  }
  const servNumeric = servSort === 'size' || servSort === 'ctx' || servSort === 'state'
  const servDirMul = servDir === 'asc' ? 1 : -1
  servRows = [...servRows].sort((a, b) => {
    const va = servGet(a), vb = servGet(b)
    if (servNumeric) return (Number(va) - Number(vb)) * servDirMul
    return String(va).localeCompare(String(vb)) * servDirMul
  })
  const toggleServSort = (k) => {
    if (servSort === k) setServDir(servDir === 'asc' ? 'desc' : 'asc')
    else { setServSort(k); setServDir('asc') }
  }
  const servArrow = (k) => servSort === k ? (servDir === 'asc' ? ' ▲' : ' ▼') : ''

  // Bulk-residency selection derivations (todo t12), computed off the CURRENT
  // filter (servRows) so "select all" and the header checkbox track exactly
  // what the operator can see. selCount counts only selected keys still visible
  // AND still designated — a stale selection (model unassigned in another tab)
  // never inflates the count or the "N selected" bar.
  // EXPLICIT priority groups: cluster this worker's serving rows under a
  // collapsible group header. Members stay together in the table's CURRENT
  // sort order (the cluster sits where its first member sorted); collapse is
  // per (worker × group) and purely visual — filters, selection and counts
  // keep operating on the flat servRows.
  const { groups: pgroupsAll } = usePriorityGroups()
  const pgIdx = useMemo(() => pgIndexOf(pgroupsAll), [pgroupsAll])
  const [pgCollapsed, setPgCollapsed] = useSessionState(
    `hugpy.sess.wp.pgcollapsed.${worker.id}`, {})
  const servDisplayRows = (() => {
    if (!pgIdx.size) return servRows
    const pgOf = (key) => {
      for (const f of pgKeyForms(key)) {
        const g = pgIdx.get(f)
        if (g) return g
      }
      return null
    }
    const byGroup = new Map()
    for (const r of servRows) {
      const g = pgOf(r.key)
      if (!g) continue
      if (!byGroup.has(g.id)) byGroup.set(g.id, { g, rows: [] })
      byGroup.get(g.id).rows.push(r)
    }
    if (!byGroup.size) return servRows
    // GROUPS FIRST (operator ask 2026-08-25): every group cluster sorts above
    // the ungrouped rows. Clusters keep the order of their first member in the
    // current sort; members keep the sort within their cluster.
    const out = []
    const emitted = new Set()
    for (const r of servRows) {
      const g = pgOf(r.key)
      if (!g || emitted.has(g.id)) continue
      emitted.add(g.id)
      const bucket = byGroup.get(g.id)
      out.push({ kind: 'pgheader', g, count: bucket.rows.length })
      if (!pgCollapsed?.[g.id]) out.push(...bucket.rows)
    }
    for (const r of servRows) {
      if (!pgOf(r.key)) out.push(r)
    }
    return out
  })()

  const servVisibleKeys = servRows.map(r => r.key)
  const selVisible = servVisibleKeys.filter(k => servSel.has(k))
  const selCount = selVisible.length
  const allVisibleSelected = servVisibleKeys.length > 0 && selCount === servVisibleKeys.length
  const someVisibleSelected = selCount > 0 && !allVisibleSelected
  const selectAllVisible = () => setServSel(prev => {
    const next = new Set(prev)
    if (allVisibleSelected) servVisibleKeys.forEach(k => next.delete(k))
    else servVisibleKeys.forEach(k => next.add(k))
    return next
  })
  const canBulkResidency = !!worker.config && !!onSetResidencyMany
  const canBulkAlloc = !!onSetAllocMany

  return (
    <div className={`wp-worker wp-${worker.status} wp-adm-${worker.admission || 'approved'}`}>
      <div className="wp-worker-head">
        <span className="wp-dot" />
        <span className="wp-name">{worker.name}</span>
        <span className="wp-status">{worker.status}</span>
        {/* Version pill — control-plane/worker skew is silent behavior drift, so
            the running package version rides in the header next to the name.
            GREEN = in sync with central's required_pkg_version, RED = skewed
            (⬆ Update converges it now), NEUTRAL = the worker didn't report a
            version, or central has no pin, or the row predates version_ok
            (older workers / showroom fixtures) — never guess red from absence. */}
        {worker.pkg_version && (
          <span className={`wp-ver ${worker.version_ok === true ? 'wp-ver-ok'
            : worker.version_ok === false ? 'wp-ver-skew' : 'wp-ver-unknown'}`}
                title={worker.version_ok === true
                  ? `abstract_hugpy_dev ${worker.pkg_version} — in sync with central`
                  : worker.version_ok === false
                    ? `worker on ${worker.pkg_version}, central requires ${worker.required_pkg_version || '?'} — use ⬆ Update to converge now (or it self-updates on its next heartbeat)`
                    : `abstract_hugpy_dev ${worker.pkg_version} — central reported no required version to compare against`}>
            v{worker.pkg_version}
          </span>
        )}
        {worker.engine_build && (
          // ENGINE build id beside the pkg version (item L, k65): the native
          // llama-server commit, so engine skew across the fleet is visible next
          // to version skew. The spawn now probes engine capability; this makes
          // the build it probes against legible.
          <span className="wp-ver wp-ver-engine"
                title={`native llama-server engine build ${worker.engine_build} — surfaced so engine skew across the fleet is visible (item L). The fat-arch rebuild is tracked separately.`}>
            ⚙ {worker.engine_build}
          </span>
        )}
        <span className={`wp-adm wp-adm-pill-${worker.admission || 'approved'}`}
              title="Operator admission gate — only approved workers serve traffic">
          {worker.admission || 'approved'}
          {/* pending/blocked are warnings — link the fix; approved needs none */}
          {worker.admission && worker.admission !== 'approved' && <FixDoc doc="worker-admission" />}
        </span>
        <span className={`wp-pool ${worker.pool ? 'wp-pool-set' : ''}`} role="button"
              title="Dedicated pool — this worker serves ONLY requests tagged for it (general traffic never lands here). Click to set/clear."
              onClick={() => onSetPool(worker)}>
          🏷 {worker.pool || 'general'}
        </span>
        {/* Per-worker WILDCARD toggle ("take all comers"). De-emphasized under
            Feasible mode, where it is nearly moot (any feasible worker is already
            a routing candidate); it matters under Designated mode. */}
        <span className={`wp-wildcard ${wildcardOn ? 'wp-wildcard-on' : ''}${distMode === 'feasible' ? ' wp-wildcard-moot' : ''}`}
              role="button" aria-pressed={wildcardOn}
              title={(wildcardOn
                ? 'Wildcard ON: this worker takes all comers — undesignated models may route here and designated models overflow here when their home workers are refused. Click to clear.'
                : 'Wildcard OFF: this worker serves only its own designated / resident / granted models. Click to opt in to "take all comers".')
                + (distMode === 'feasible'
                  ? ' — only matters in Designated mode (under Feasible, any worker where the model fits is already a candidate).'
                  : '')}
              onClick={() => setWildcard(!wildcardOn)}>
          🃏 {wildcardOn ? 'wildcard' : 'no wildcard'}
        </span>
        {worker.role === 'rpc' &&<span className="wp-role" title="shard backend (lends GPU via rpc-server)">rpc</span>}
        {worker.comfy?.available && (
          <span className="wp-role" title={`ComfyUI running on this worker${worker.comfy.version ? ` (v${worker.comfy.version})` : ''} at ${worker.comfy.url} — comfy-templated generation routes here (engine slice B)`}>
            🧩 comfy
          </span>
        )}
        <span className="wp-url" title={worker.url}>{worker.url}</span>
        <SpillBadge spill={worker.spill} />
        {(worker.gpus || []).length > 0 && worker.engine?.supports_gpu_offload === false && (
          <span className="wp-cpu-only"
                title={'This worker\'s llama-cpp-python is a CPU-only build: GGUF models run on CPU and n_gpu_layers is silently ignored, so VRAM stays idle. Rebuilding it with GPU support has real traps (missing nvcc, missing CUDA runtime libs, AVX512 SIGILL) — click 📖 for the diagnostic and the rebuild recipe that actually works.'}>
            ⚠ CPU-only engine
            <FixDoc doc="engine-cpu-only" />
          </span>
        )}
        {worker.install && worker.install.canonical === false && (
          <span className="wp-noncanon"
                title={`Non-canonical install — this worker did NOT come from the standard bootstrap/installer.\n`
                       + `unit: ${worker.install.unit || (worker.install.via_systemd ? 'systemd (name unknown)' : 'not a systemd unit')}\n`
                       + `venv: ${worker.install.venv || 'unknown'}\n`
                       + `Canonical = a hugpy-worker.service (or legacy abstract-hugpy-worker.service) user unit running from ~/hugpy-worker/venv. See WORKER-SETUP.md §1.`}>
            ⚠ non-canonical install
          </span>
        )}
        {ping && ping !== 'checking' && (
          <span className={`wp-ping ${ping.reachable ? 'wp-ping-ok' : 'wp-ping-bad'}`}
                title={ping.reachable ? 'central can reach this worker' : responseReason(ping)}>
            {ping.reachable ? '✓ reachable' : '✗ unreachable'}
            {!ping.reachable && <FixDoc doc="worker-unreachable" />}
          </span>
        )}
        <button className="wp-ping-btn" title="Ping the worker's /health from central"
                onClick={checkHealth} disabled={ping === 'checking'}>
          {ping === 'checking' ? '…' : 'ping'}
        </button>
        {loaded.size > 0 && (
          <button className="wp-free-all" title="Unload every model from this GPU (stays assigned)"
                  onClick={() => onFreeAll(worker)}>
            ⏏ free VRAM
          </button>
        )}
        {/* Maintenance controls — both shown ALWAYS (Free RAM is the fix for an
            orphaned arena that holds RAM with nothing loaded, so never gate it). */}
        <button className="wp-free-ram" title="Return reclaimable host RAM to the OS — non-destructive: loaded models stay resident"
                onClick={() => onFreeRam(worker)}>
          🧹 Free RAM
        </button>
        <button className="wp-restart" title="Restart the worker agent — drops all loaded models and re-execs the agent process"
                onClick={() => onRestart(worker)} disabled={restarting}>
          {restarting ? '↻ restarting…' : '↻ Restart'}
        </button>
        {/* ⬆ Update — converge this worker onto central's required version now
            instead of waiting for its next heartbeat. It pip-installs and
            restarts itself, so it shares the restart transient (and its flag). */}
        {onUpdate && (
          <button className={`wp-update${worker.version_ok === false ? ' wp-update-skew' : ' wp-update-noop'}`}
                  title={worker.version_ok === false
                    ? `worker on ${worker.pkg_version || '?'}, central requires ${worker.required_pkg_version || '?'} — update pip-installs the required version and restarts the agent`
                    : 'already at central’s required version — update is a no-op'}
                  onClick={() => onUpdate(worker)} disabled={updating || restarting}>
            {updating ? '⬆ updating…' : '⬆ Update'}
          </button>
        )}
        {worker.admission === 'approved'
          ? <button className="wp-block" title="Block: stop serving; the agent exits on its next contact and won't respawn"
                    onClick={() => onBlock(worker)}>⛔ block</button>
          : <button className="wp-admit" title={worker.admission === 'blocked' ? 'Unblock and allow serving' : 'Admit: allow this worker to serve'}
                    onClick={() => onAdmit(worker)}>✓ {worker.admission === 'blocked' ? 'unblock' : 'admit'}</button>}
        <button className="wp-remove" title="Remove worker (forget — a live agent re-appears as pending; use Block to evict)" onClick={() => onRemove(worker)}>✕</button>
      </div>

      {/* Resource header: VRAM · RAM · Storage as compact chips, each expanding
          its resident/detail panel below on click. Replaces the old budget bar +
          storage bar + per-GPU chips + free-RAM line. */}
      <ResourceStrip worker={worker} models={models} onApproveEvictions={onApproveEvictions} onEvict={onEvict} />

      {/* External gpu_lease residents (batch jobs as pseudo-models) with live
          evictable/resume policy toggles; renders nothing when none exist. */}
      <ExternalLeases worker={worker} />

      {/* Two-tier resource governance: the box's OWN config (caps) is the hard
          ceiling; central-set limits are clamped to it server-side. */}
      <div className="wp-caps">
        {worker.caps && Object.keys(worker.caps).length > 0 && (
          <span className="wp-cap-chip" title="Configured on the worker box itself (unit env) — central can only set limits at or below these.">
            box caps:{worker.caps.ram_max_gib != null && ` RAM ${worker.caps.ram_max_gib}GiB`}
            {worker.caps.gpu_mem_gib != null && ` · VRAM ${worker.caps.gpu_mem_gib}GiB`}
            {worker.caps.disk_cache_gib != null && ` · disk ${worker.caps.disk_cache_gib}GiB`}
            {worker.caps.threads != null && ` · ${worker.caps.threads} threads`}
          </span>
        )}
        {worker.limits && Object.keys(worker.limits).length > 0 && (
          <span className="wp-cap-chip wp-limit-chip" title="Central-set limits (≤ box caps); the worker adopts them on its next heartbeat.">
            central limits:{worker.limits.ram_max_gib != null && ` RAM ${worker.limits.ram_max_gib}GiB`}
            {worker.limits.gpu_mem_gib != null && ` · VRAM ${worker.limits.gpu_mem_gib}GiB`}
            {worker.limits.disk_cache_gib != null && ` · disk ${worker.limits.disk_cache_gib}GiB`}
            {worker.limits.threads != null && ` · ${worker.limits.threads} threads`}
          </span>
        )}
        {/* Console-managed serving config (daylight item 3): slot count lives
            in the AGENT's own settings (beats env drop-ins); the chip shows
            the EFFECTIVE value + where it came from. */}
        {worker.config?.slot_count != null && (
          <span className={`wp-cap-chip${applying ? ' wp-chip-applying' : ''}`} role="button"
                title={applying
                  ? 'The agent is restarting (~5s) to apply the previous config change — controls unlock when the new config arrives in a heartbeat.'
                  : `Worker slot pool: ${worker.config.slot_count} slot(s) — source: ${worker.config.slot_count_source || '?'}. Click to change (persists in the agent's runtime settings; applies via a ~5s agent restart).`}
                onClick={() => !applying && onSetConfig && onSetConfig(worker)}>
            🎛 slots: {worker.config.slot_count}
            {worker.config.slot_count_source && worker.config.slot_count_source !== 'settings' &&
              <em> ({worker.config.slot_count_source})</em>}
          </span>
        )}
        {applying && (
          <span className="wp-applying"
                title="The last pin/residency/slot-count change was accepted; the agent re-execs (~5s) to apply it and this worker's config controls are paused until the new config shows up in a heartbeat.">
            ⏳ applying…
          </span>
        )}
        {onSetLimits && (
          <button className="wp-limits-edit" title="Worker budget — this box's resource ceiling for ALL models combined (VRAM / RAM / disk / threads), clamped to its box caps. This is NOT a per-model allocation: per-model VRAM/RAM placement is the Alloc column on each serving row. Central-set (≤ box caps)."
                  onClick={() => {
                    setLimitsForm({
                      ram_max_gib: worker.limits?.ram_max_gib ?? '',
                      gpu_mem_gib: worker.limits?.gpu_mem_gib ?? '',
                      disk_cache_gib: worker.limits?.disk_cache_gib ?? '',
                      threads: worker.limits?.threads ?? '',
                    })
                    setLimitsOpen(o => !o)
                  }}>
            ⚙ worker budget
          </button>
        )}
        {limitsOpen && (
          <span className="wp-limits-form">
            {/* Ruling 4 (2026-07-24): an unmistakable label so this box-level
                ceiling can never again be read as per-model allocation (the
                operator conflated them). The per-model contract is the Alloc
                column; THIS is the whole-box budget. */}
            <span className="wp-limits-title" title="This box's resource ceiling across ALL models it hosts — not a per-model budget.">
              worker budget — this box's resource ceiling (all models combined):
            </span>
            <input type="number" step="1" min="0" placeholder="RAM GiB" value={limitsForm.ram_max_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, ram_max_gib: e.target.value }))} />
            <input type="number" step="1" min="0" placeholder="VRAM GiB" value={limitsForm.gpu_mem_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, gpu_mem_gib: e.target.value }))} />
            <input type="number" step="1" min="0" placeholder="disk cache GiB" title="Local model-cache ceiling for this worker. Over it, cold local models become eviction candidates in the storage proposal. Clamped to the box's own caps.disk_cache_gib — the worker's stated delegation wins."
                   value={limitsForm.disk_cache_gib}
                   onChange={e => setLimitsForm(f => ({ ...f, disk_cache_gib: e.target.value }))} />
            <input type="number" step="1" min="1" placeholder="threads" value={limitsForm.threads}
                   onChange={e => setLimitsForm(f => ({ ...f, threads: e.target.value }))} />
            <button className="wp-alloc-apply" onClick={() => {
              const limits = {}
              for (const k of ['ram_max_gib', 'gpu_mem_gib', 'disk_cache_gib', 'threads']) {
                if (limitsForm[k] !== '' && limitsForm[k] != null) limits[k] = Number(limitsForm[k])
              }
              onSetLimits(worker, limits)
              setLimitsOpen(false)
            }}>Set</button>
            <button className="wp-alloc-cancel" title="Clear all central limits"
                    onClick={() => { onSetLimits(worker, {}); setLimitsOpen(false) }}>clear</button>
          </span>
        )}
      </div>


      {/* Serving: every model DESIGNATED to this worker, as a comprehensive,
          sortable TABLE (same visual system as the "load a model" table). The
          load-table column identifiers (Model/Task/Engine/Size/Ctx) come first,
          then the serving-specific ones (State/Seat/Alloc/Residency/📌/Memory/
          Actions). "Serving" here is the whole assignment at EVERY stage of the
          attribution ladder — not only the models actively hosted right now. */}
      <div className={`wp-models wp-servtable${servCompact ? ' wp-servtable-compact' : ''}`} ref={servWrapRef}>
        <div className="wp-servtable-bar">
          <span className="wp-models-label">Serving:</span>
          {(worker.models || []).length > 0 && (
            <>
              <input className="wp-servtable-q" placeholder="filter models…" value={servQ}
                     onChange={e => setServQ(e.target.value)} />
              <select className="wp-servtable-task" value={servTask}
                      onChange={e => setServTask(e.target.value)}
                      title="Filter by task — matches ANY task a model advertises, not just its primary">
                <option value="">All tasks</option>
                {servTasks.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              {/* COMPACT-ONLY sort control. In compact mode most sortable
                  headers are not on screen to click, so the same servSort /
                  servDir state gets a select + a direction toggle here. Desktop
                  keeps header-click sorting and never renders this. */}
              {servCompact && (
                <span className="wp-servtable-sortsel">
                  <select value={servSort} onChange={e => setServSort(e.target.value)}
                          title="Sort the serving list (the sortable columns are hidden in this narrow layout)">
                    {servColsAll.filter(c => c.sortable).map(c => (
                      <option key={c.key} value={c.key}>sort: {c.label}</option>
                    ))}
                  </select>
                  <button className="wp-servtable-sortdir"
                          title={servDir === 'asc' ? 'Ascending — click for descending' : 'Descending — click for ascending'}
                          onClick={() => setServDir(servDir === 'asc' ? 'desc' : 'asc')}>
                    {servDir === 'asc' ? '▲' : '▼'}
                  </button>
                </span>
              )}
              {/* Reset the per-user column layout (order + widths) back to the
                  default. Only appears once the layout diverges from default, so
                  it's silent for operators who never touch the headers. Hidden in
                  compact mode — there is no drag/resize there to undo. */}
              {!servCompact && (servOrder.join(',') !== SERV_DEFAULT_ORDER.join(',') || Object.keys(servWidths).length > 0) && (
                <button className="wp-servtable-resetcols" onClick={servResetLayout}
                        title="Reset the serving-table columns (order + widths) to the default layout. Drag a header to reorder, or a header's right edge to resize — your layout is remembered per browser.">
                  ⟲ reset columns
                </button>
              )}
            </>
          )}
          {/* Bulk residency (todo t12): appears once one or more models in the
              current filter are selected (via the row checkboxes). Sets the
              RESIDENCY tier of the whole selection in ONE agent restart (central
              relays a single /ops/config residency map — same batching pinAll
              does). Confirms first, surfaces per-model results. Residency ONLY —
              NOT 📌 pin (that's the separate control to the right). */}
          {(canBulkResidency || canBulkAlloc) && selCount > 0 && (
            <span className="wp-bulkres wp-servtable-bulkres">
              <span className="wp-bulkres-count" title="models selected in the current filter">
                {selCount} selected
              </span>
              {/* Bulk RESIDENCY (todo t12): a ~5s agent restart. */}
              {canBulkResidency && (
                <>
                  <span className="wp-bulkres-label">set residency →</span>
                  <button className="wp-bulkres-btn wp-bulkres-ondemand" disabled={applying}
                          title={applying
                            ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                            : `Set the ${selCount} selected model${selCount === 1 ? '' : 's'} to ⏲ on-demand (the default — loads on call, yields its seat under contention). Confirms first; one ~5s agent restart.`}
                          onClick={() => onSetResidencyMany(worker, selVisible, 'on-demand')}>
                    ⏲ on-demand
                  </button>
                  <button className="wp-bulkres-btn wp-bulkres-static" disabled={applying}
                          title={applying
                            ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                            : `Set the ${selCount} selected model${selCount === 1 ? '' : 's'} to 🔒 static (locked seat — kept on this worker, never evicted). Confirms first; one ~5s agent restart.`}
                          onClick={() => onSetResidencyMany(worker, selVisible, 'static')}>
                    🔒 static
                  </button>
                </>
              )}
              {/* Bulk ALLOC (todo t15): a registry write — NO restart. The ⚙
                  button toggles the inline AllocControl below the bar (same
                  editor the per-row ⚙ uses), applied to the whole selection. */}
              {canBulkAlloc && (
                <>
                  {canBulkResidency && <span className="wp-bulkres-sep" aria-hidden="true">·</span>}
                  <span className="wp-bulkres-label">set alloc →</span>
                  <button className={`wp-bulkres-btn wp-bulkres-alloc${bulkAllocOpen ? ' wp-bulkres-alloc-open' : ''}`}
                          title={`Set the GPU allocation (Default / Max GPU / GPU only / RAM only / Max RAM / Explicit) for the ${selCount} selected model${selCount === 1 ? '' : 's'}. A registry contract applied on next load — no agent restart. Confirms first.`}
                          onClick={() => setBulkAllocOpen(o => !o)}>
                    ⚙ allocation {bulkAllocOpen ? '▾' : '▸'}
                  </button>
                </>
              )}
              <button className="wp-bulkres-clear" title="Clear the selection"
                      onClick={() => { clearSel(); setBulkAllocOpen(false) }}>clear</button>
            </span>
          )}
          {/* The bulk-alloc editor, expanded below the bar (full-width, so the
              custom-budget inputs have room) — mirrors the per-row expansion. */}
          {canBulkAlloc && selCount > 0 && bulkAllocOpen && (
            <div className="wp-bulkalloc-editor">
              <span className="wp-bulkalloc-hint">
                Apply this allocation to the {selCount} selected model{selCount === 1 ? '' : 's'}:
              </span>
              <BulkAllocControl
                count={selCount}
                bulkKeys={selVisible}
                getModelBytes={sizeInfo}
                onApply={(spill, perModel) => { setBulkAllocOpen(false); onSetAllocMany(worker, selVisible, spill, perModel) }}
                onCancel={() => setBulkAllocOpen(false)}
              />
            </div>
          )}
          {/* Bulk pin: 📌 pin (or unpin) EVERY model designated to this worker in
              one settings-write. Sticky, so pinAll confirms first. Kept on the
              RIGHT of the filter bar. */}
          {worker.config && (onPinAll || onUnpinAll) && (worker.models || []).length > 0 && (
            <span className="wp-bulkpin wp-servtable-bulk">
              {onPinAll && (
                <button className="wp-bulkpin-btn" disabled={applying}
                        title={applying
                          ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                          : 'Pin ALL of this worker’s models — permanent attribution; each then blocks unassign until unpinned. Confirms first.'}
                        onClick={() => onPinAll(worker)}>📌 pin all</button>
              )}
              {onUnpinAll && (
                <button className="wp-bulkpin-btn wp-bulkunpin-btn" disabled={applying}
                        title={applying
                          ? 'Agent is restarting to apply the previous change — retry in a few seconds.'
                          : 'Unpin ALL of this worker’s models — the undo for Pin all; lets them be unassigned again.'}
                        onClick={() => onUnpinAll(worker)}>📌✕ unpin all</button>
              )}
              {/* Prune AUTOMATED (non-pinned) designations — benchmark /
                  admission / model_group / autoplace rows whose model is not
                  loaded here and has not been called for the idle bound. Shows
                  the DRY-RUN plan first, then applies only on confirm. Never
                  touches pinned or operator designations. */}
              {onPruneDesignations && (
                <button className="wp-bulkpin-btn" disabled={applying}
                        title="Prune this worker's AUTOMATED (non-pinned) designations that are idle past the bound — shows what would be removed first, applies only on confirm. Pinned & operator designations are never touched; nothing on the worker is unloaded."
                        onClick={() => onPruneDesignations(worker)}>🧹 prune automated</button>
              )}
              {/* Bulk 4-bit (operator ask 2026-07-26), in the same row as pin
                  all / unpin all. Acts ONLY on the ELIGIBLE set — GGUF,
                  CPU-only workers and already-quantized repos are skipped
                  rather than attempted, so the count in the label is the real
                  number of models that will change. Hidden entirely when this
                  worker has nothing eligible (e.g. op, no GPU). */}
              {moeCapableKeys.length > 0 && (
                <button className="wp-bulkpin-btn wp-bulkmoe-btn" disabled={applying}
                        title={`Restore AUTO expert-split handling on all ${moeCapableKeys.length} MoE-capable model(s) — clears any forced on/off and lets the derivation decide.`}
                        onClick={() => setMoeMany(null)}>
                  ⟲ MoE auto ({moeCapableKeys.length})
                </button>
              )}
              {bnbEligibleKeys.length > 0 && (
                <>
                  <button className="wp-bulkpin-btn wp-bulk4bit-btn" disabled={applying}
                          title={`Load all ${bnbEligibleKeys.length} eligible model(s) at bitsandbytes 4-bit (nf4). Each is then priced at ~30% of its fp16 size, so their Alloc modes re-derive — models too big for the GPU can become GPU-resident. Costs some quality.`}
                          onClick={() => setBnbMany(true)}>
                    ▦ 4-bit all ({bnbEligibleKeys.length})
                  </button>
                  <button className="wp-bulkpin-btn wp-bulkunpin-btn" disabled={applying}
                          title="Restore full precision on every model that currently has the 4-bit specialization — the undo for 4-bit all."
                          onClick={() => setBnbMany(false)}>
                    ▦✕ clear 4-bit
                  </button>
                </>
              )}
            </span>
          )}
        </div>
        {(worker.models || []).length === 0 ? (
          <span className="wp-none">— nothing assigned —</span>
        ) : (
          <div className="wp-servtable-scroll">
            <table className="wp-servtable-t">
              {/* Column widths live in the <colgroup> so a resize touches ONE
                  place, not every <td>. The checkbox column is fixed; each data
                  column gets its persisted px width, or auto when never resized.
                  The FROZEN Model column is the exception: it is FLUID (a CSS
                  clamp() on .wp-servtable-namecol) and carries no manual resize,
                  so any px width a previous layout stored for it is IGNORED here
                  — the stored value is simply never read, which leaves every
                  other column's remembered width intact (no storage-key bump). */}
              <colgroup>
                <col className="wp-servtable-selcol-col" />
                {servCols.map(c => {
                  const fluid = c.key === SERV_FROZEN_COL
                  return (
                    <col key={c.key}
                         className={fluid ? 'wp-servtable-namecol' : undefined}
                         style={!fluid && servWidths[c.key] ? { width: servWidths[c.key] } : undefined} />
                  )
                })}
              </colgroup>
              <thead>
                <tr>
                  {/* Select-all-in-current-filter checkbox (todo t12). Only the
                      visible/filtered rows are (de)selected — matches what the
                      operator can see. Indeterminate when a subset is picked.
                      FIXED left — not reorderable or resizable. */}
                  <th className="wp-servtable-h wp-servtable-selcol">
                    {canBulkResidency ? (
                      <input type="checkbox" className="wp-servtable-selall"
                             checked={allVisibleSelected}
                             ref={el => { if (el) el.indeterminate = someVisibleSelected }}
                             disabled={servVisibleKeys.length === 0}
                             onChange={selectAllVisible}
                             title="Select all models in the current filter (for a bulk residency change)" />
                    ) : null}
                  </th>
                  {/* Every other column renders from the ONE ordered model, so a
                      drag reorders it anywhere. Sortable columns keep their click-
                      to-sort + arrow; all columns get a drag-reorder grip and a
                      right-edge resize handle. */}
                  {servCols.map(c => {
                    // The Model column is FROZEN left (sticky during horizontal
                    // scroll): it is neither a drag SOURCE nor a drop TARGET, so
                    // it can never leave slot 1 and nothing can land before it.
                    // Its sort click stays live; its RESIZE GRIP is gone (2026-
                    // 07-28) — the column is fluid (clamp) and mid-ellipsised, so
                    // a manual px width would fight the clamp.
                    const frozen = c.key === SERV_FROZEN_COL
                    // Compact mode has nothing to reorder into and nothing to
                    // freeze against (three columns, no horizontal scroll), so
                    // drag/resize/sticky all go inert.
                    const dragOk = !frozen && !servCompact
                    return (
                    <th key={c.key}
                        className={`wp-col-th${c.sortable ? ' wp-lt-sortable' : ' wp-servtable-h'}${c.num ? ' wp-lt-num' : ''}${frozen ? ' wp-servtable-stickycol wp-servtable-stickyname' : ''}${dragOk && servDragOver === c.key && servDragCol && servDragCol !== c.key ? ' wp-col-dragover' : ''}${servDragCol === c.key ? ' wp-col-dragging' : ''}`}
                        draggable={dragOk}
                        onDragStart={!dragOk ? undefined : (e) => { setServDragCol(c.key); e.dataTransfer.effectAllowed = 'move' }}
                        onDragOver={!dragOk ? undefined : (e) => { if (servDragCol && servDragCol !== c.key) { e.preventDefault(); setServDragOver(c.key) } }}
                        onDrop={!dragOk ? undefined : (e) => { e.preventDefault(); if (servDragCol && servDragCol !== c.key) servMoveCol(servDragCol, c.key); setServDragCol(null); setServDragOver(null) }}
                        onDragEnd={!dragOk ? undefined : () => { setServDragCol(null); setServDragOver(null) }}
                        title={frozen ? 'Click to sort · this column is pinned left (frozen while you scroll sideways) · its width is fluid — long names are shortened in the MIDDLE, hover for the full key'
                          : !dragOk ? (c.sortable ? 'Click to sort' : undefined)
                          : c.sortable ? 'Click to sort · drag to reorder · drag the right edge to resize' : 'Drag to reorder · drag the right edge to resize'}
                        onClick={c.sortable ? () => toggleServSort(c.key) : undefined}>
                      {c.label}{c.sortable ? servArrow(c.key) : ''}
                      {/* Resize grip: a pointer-drag that never triggers the sort
                          click or the column drag (both are stopped here). NOT
                          rendered for the fluid Model column, nor in compact mode
                          (where there is no spare width to hand out). */}
                      {dragOk && (
                        <span className="wp-col-resize" draggable={false}
                              onDragStart={(e) => e.preventDefault()}
                              onClick={(e) => e.stopPropagation()}
                              onPointerDown={(e) => startServResize(e, c.key)} />
                      )}
                    </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {/* Assigned models exist, but the search/task filter hid them
                    ALL — a DISTINCT empty state, never the "nothing assigned"
                    message (which would misreport the worker as empty). */}
                {servRows.length === 0 && (
                  <tr><td colSpan={SERV_COLSPAN} className="wp-servtable-nomatch">
                    no match — clear the filter to see all {(worker.models || []).length} assigned
                  </td></tr>
                )}
                {servDisplayRows.map((entry) => {
                  if (entry.kind === 'pgheader') {
                    const g = entry.g
                    const isCollapsed = !!pgCollapsed?.[g.id]
                    const total = (g.members || []).length
                    return (
                      <tr key={`pg:${g.id}`} className="wp-pgroup-row">
                        <td colSpan={SERV_COLSPAN}>
                          <button type="button" className="wp-pgroup-bar"
                                  aria-expanded={!isCollapsed}
                                  title={`priority group ${g.id} — click to ${isCollapsed ? 'expand' : 'collapse'}`}
                                  onClick={() => setPgCollapsed(c => ({ ...c, [g.id]: !c?.[g.id] }))}>
                            <span className="wp-pgroup-caret">{isCollapsed ? '▸' : '▾'}</span>
                            <span className="wp-pgroup-name">▣ {g.name}</span>
                            <span className="wp-pgroup-count">
                              {entry.count === total ? `${entry.count} models`
                                : `${entry.count} of ${total} models here`}
                            </span>
                            {(g.workers || []).length > 0 && (
                              <span className="wp-pgroup-workers">→ {g.workers.join(' → ')}</span>
                            )}
                          </button>
                        </td>
                      </tr>
                    )
                  }
                  const { key, m, d } = entry
                  const state = d.state
                  const isSel = servSel.has(key)
                  // Operator model BLOCK (global): this model is removed from the
                  // serving pool everywhere. The designation row still renders
                  // (inert + labeled) — block does not unassign and outranks pin.
                  const isBlocked = !!(blockedKeys && blockedKeys.has(key))
                  // t49: this model's own requirement (GGUF effective quant),
                  // for the AllocControl VRAM/RAM slider coupling. GGUF-only —
                  // explicit budgets (what the coupling lives inside) are
                  // already a GGUF-exclusive concept, so gate the denominator
                  // the same way rather than offering a misleading coupling on
                  // a transformers/comfy model's dir-size total.
                  const need = (() => {
                    const si = sizeInfo(key)
                    return (si && si.isGguf && si.bytes != null)
                      ? { bytes: si.bytes, gib: si.bytes / WP_GIB } : null
                  })()
                  // `compact` rides the cell context so a def can tighten its own
                  // rendering (only Model does today: a smaller midTrunc budget).
                  const cx = { key, m, d, isSel, isBlocked, need, compact: servCompact }
                  const drawerOpen = servCompact && servDrawer === key
                  return (
                    <Fragment key={key}>
                      <tr className={`wp-servtable-row wp-st-${state}${isSel ? ' wp-servtable-sel' : ''}${isBlocked ? ' wp-servtable-blocked' : ''}${drawerOpen ? ' wp-servtable-rowopen' : ''}`}
                          onClick={servRowTap(key)}>
                        {/* Row select checkbox (todo t12) — FIXED left, keyed by
                            model_key; survives the 10s refresh. Not reorderable. */}
                        <td className="wp-servtable-selcol">
                          {canBulkResidency ? (
                            <input type="checkbox" className="wp-servtable-selrow"
                                   checked={isSel}
                                   onChange={() => toggleSel(key)}
                                   title="Select this model for a bulk residency change" />
                          ) : null}
                        </td>
                        {/* Every other cell renders from the SAME ordered column
                            model as the header, so a reorder/resize moves the whole
                            column (header + body) together. */}
                        {servCols.map(c => (
                          <td key={c.key}
                              className={`${c.cls || ''}${c.key === SERV_FROZEN_COL ? ' wp-servtable-stickycol wp-servtable-stickyname' : ''}`.trim() || undefined}
                              title={c.title ? c.title(cx) : undefined}>
                            {c.render(cx)}
                          </td>
                        ))}
                      </tr>
                      {/* COMPACT DRAWER (operator ask 2026-07-28) — the SAME
                          colspanned expansion-row mechanism the residency picker
                          uses, one row at a time. It carries the FULL untruncated
                          model name, then every column that left the table as a
                          labeled chip, then the actions as a touch-target row.
                          Each chip's body is the column def's OWN render(cx): the
                          Alloc menu, residency button, 📌 pin, 4-bit and MoE
                          checkboxes are literally the same controls as desktop,
                          not a second implementation. */}
                      {drawerOpen && (
                        <tr className="wp-servtable-expand wp-servtable-drawer">
                          <td colSpan={SERV_COLSPAN}>
                            <div className="wp-servdrawer">
                              <div className="wp-servdrawer-name" title={key}>{nameFor(key)}</div>
                              {/* The model KEY when it differs from the display
                                  name — the key is what every API call and log
                                  line uses, so it must be readable somewhere. */}
                              {nameFor(key) !== key && (
                                <div className="wp-servdrawer-key">{key}</div>
                              )}
                              <div className="wp-servdrawer-chips">
                                {servDrawerCols.map(c => (
                                  <div key={c.key} className="wp-servdrawer-chip"
                                       title={c.title ? c.title(cx) : undefined}>
                                    <span className="wp-servdrawer-chip-label">{c.label}</span>
                                    <span className="wp-servdrawer-chip-body">{c.render(cx)}</span>
                                  </div>
                                ))}
                              </div>
                              {/* Actions (▶ activate / ⏏ free / × unassign / ⛔
                                  block) — the same def render, laid out as a
                                  full-width row of ≥40px touch targets. */}
                              <div className="wp-servdrawer-actions">
                                {servActionsCol.render(cx)}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {/* Full-width expansion row: the residency picker (the
                          alloc editor is now the in-place, click-site
                          AllocModeMenu — no lower-row expansion, operator ask
                          2026-07-24). Rendered as its own colspanned row so it
                          never disturbs the table's column layout. */}
                      {resMenu === key && worker.config && onSetResidency && (
                        <tr className="wp-servtable-expand">
                          <td colSpan={SERV_COLSPAN}>
                            <ResidencyMenu mode={worker.config.residency?.[key] === 'static' ? 'static' : 'on-demand'}
                                           onClose={() => setResMenu(null)}
                                           onPick={(next) => {
                                             setResMenu(null)
                                             onSetResidency(worker, key, next)
                                           }} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {!showLoad ? (
        <button className="wp-load-toggle" onClick={() => setShowLoad(true)}
                title="Load another model onto this worker">
          ＋ load a model
        </button>
      ) : (
        <WorkerLoadTable
          models={assignable}
          allocation={allocation || {}}
          workerId={worker.id}
          worker={worker}
          onAllocate={async (modelKeys, breaker) => {
            await onAllocateMany(worker, modelKeys, breaker)
            setShowLoad(false)
          }}
          onCancel={() => setShowLoad(false)}
        />
      )}
    </div>
  )
}
