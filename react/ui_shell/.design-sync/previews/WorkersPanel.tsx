// WorkersPanel — remote GPU worker pool: enrolled workers, health, enroll tokens,
// per-worker VRAM/RAM, model serving + allocation. Self-fetches
// /api/llm/workers + enroll-tokens (DemoProvider answers from fixtures);
// `models` seeds the assignment picker. The console mounts it with `embedded`
// (the full expanded panel) — see src/App/App.jsx; non-embedded collapses to a
// summary bar that only expands on click, so it can't render rich statically.
import { WorkersPanel } from '@hugpy/ui'
import { MODELS } from '../../src/showroom/fixtures.js'

export function Default() {
  return <WorkersPanel models={MODELS} embedded />
}
