// ModelLiveState — the ONE live-state relay for a model, mounted on every
// surface that executes a model call (chat, benchmark workbook, placement,
// review runs, assessment).  It reads the same shared /llm/models/status poll
// the Models table uses, so what you see beside the action is what the fleet
// is doing right now: worth label, every allocated worker's state
// (missing / cold / downloading / loading / hot / serving / answering,
// +held / failed) and the last recorded failure.  Absence is explicit.
import { useModelStatus } from '../ModelTable/useModelStatus'
import { failureView, statusFor, workerChips, worthView } from '../ModelTable/modelStatus'
import '../ModelTable/ModelStatus.css'

export function useModelLiveRow(modelKey) {
  const { index, loaded, unsupported, error } = useModelStatus()
  const row = index?.available && modelKey ? statusFor(index, modelKey) : null
  return { row, loaded, unsupported, error, available: !!index?.available }
}

// One worker's chip for a model (for rows that already name the worker).
export function LiveWorkerChip({ modelKey, worker }) {
  const { row, available } = useModelLiveRow(modelKey)
  if (!available) return <span className="ms-wchip ms-muted" title="central does not serve /llm/models/status">no status</span>
  const c = row ? workerChips(row).find(x => x.worker === worker) : null
  if (!c) return <span className="ms-wchip ms-muted" title={`${worker}: no state reported for ${modelKey}`}>no state</span>
  return <span className={`ms-wchip ms-${c.tone}${c.online ? '' : ' ms-offline'}`} title={c.title}>{c.icon} {c.label}</span>
}

export default function ModelLiveState({ modelKey, compact = false, showWorth = true, showFailure = true, only = null }) {
  const { row, loaded, unsupported, error, available } = useModelLiveRow(modelKey)
  if (!modelKey) return null
  if (unsupported) return <span className="ms-live ms-none" title="GET /llm/models/status is not served by this central">no live status (central lacks /llm/models/status)</span>
  if (!loaded) return <span className="ms-live ms-none">live status: loading…</span>
  if (error && !row) return <span className="ms-live ms-none" title={error}>live status unavailable: {error}</span>
  if (!available || !row) return <span className="ms-live ms-none">no status row for {modelKey}</span>
  const w = worthView(row)
  const f = failureView(row)
  const chips = workerChips(row).filter(c => !only || c.worker === only)
  return <span className={`ms-live${compact ? ' ms-live-compact' : ''}`}>
    {showWorth && <span className={`ms-wchip ms-${w.tone}`} title={w.title}>{w.text}</span>}
    {chips.length ? chips.map(c =>
      <span key={c.worker} className={`ms-wchip ms-${c.tone}${c.online ? '' : ' ms-offline'}`} title={c.title}><b>{c.worker}</b> {c.icon} {c.label}</span>)
      : <span className="ms-wchip ms-muted" title="no worker has this model designated, loaded, allocated or stored">no workers allocated</span>}
    {showFailure && !f.none && <span className="ms-wchip ms-bad" title={f.title}>last failure: {f.text}</span>}
  </span>
}
