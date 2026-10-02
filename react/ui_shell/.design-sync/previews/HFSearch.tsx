// HFSearch — the "Add models · Hugging Face Hub" search panel. Prop-driven for
// its job/queue wiring, but it self-fetches the hub search itself (GET /api/search,
// answered from the SEARCH fixture by the demo shim) and seeds the default task,
// so the results table populates with no backend. Mirrors src/App/App.jsx's
// <HFSearch embedded …/> composition.
import { HFSearch } from '@hugpy/ui'
import { JOB } from '../../src/showroom/fixtures.js'

const noop = () => {}

export function Embedded() {
  // Default embedded panel: searches the default `text-generation` task on mount
  // and renders the populated, sortable results table.
  return (
    <HFSearch
      embedded
      onJobStarted={noop}
      onCancelJob={noop}
      onRetryJob={noop}
      pendingByHub={{}}
      jobsByHub={{}}
      expanded={true}
      onToggleExpanded={noop}
    />
  )
}

export function WithDownloads() {
  // Same populated table, but two repos are mid-pipeline: one actively
  // downloading (jobsByHub) and one externally queued (pendingByHub) — varies
  // the action column (…/queued) against the idle "⬇ Options" rows.
  const jobsByHub = {
    'bartowski/Qwen2.5-7B-Instruct-GGUF': {
      ...JOB,
      model_key: 'bartowski/Qwen2.5-7B-Instruct-GGUF',
      status: 'running',
    },
  }
  const pendingByHub = { 'meta-llama/Llama-3.2-3B-Instruct': true }
  return (
    <HFSearch
      embedded
      onJobStarted={noop}
      onCancelJob={noop}
      onRetryJob={noop}
      pendingByHub={pendingByHub}
      jobsByHub={jobsByHub}
      expanded={true}
      onToggleExpanded={noop}
    />
  )
}
