// ModelTable — the installed-models grid (status, task, context, size, serving
// controls, per-model actions). Fully prop-driven: the app passes already-fetched
// models + a jobsByModel map + action callbacks. We pass the showroom fixtures and
// no-op the callbacks. Mirrors src/App/App.jsx's <ModelTable …/> composition.
import { ModelTable } from '@hugpy/ui'
import { MODELS, WORKERS, JOB } from '../../src/showroom/fixtures.js'

const noop = () => {}
const handlers = {
  onDownload: noop,
  onChat: noop,
  onDelete: noop,
  onPrune: noop,
  onSetMedia: noop,
  onCancel: noop,
  onRetry: noop,
  onAssignWorker: noop,
  onProbeWorker: noop,
}

export function Default() {
  return (
    <ModelTable
      models={MODELS}
      jobsByModel={{}}
      activeChat={null}
      workers={WORKERS}
      {...handlers}
    />
  )
}

export function Downloading() {
  // One model mid-download — surfaces the inline progress bar + cancel/retry.
  const target = MODELS[1].model_key
  const jobsByModel = { [target]: { ...JOB, model_key: target } }
  return (
    <ModelTable
      models={MODELS}
      jobsByModel={jobsByModel}
      activeChat={null}
      workers={WORKERS}
      {...handlers}
    />
  )
}
