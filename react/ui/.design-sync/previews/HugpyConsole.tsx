// HugpyConsole — the entire hugpy console (tabbed Models/Add/Compute/API surface)
// wrapped in its own HugpyProvider. We pass the demo-fetch shim explicitly so the
// provider's runtime config routes every panel call to canned fixtures instead of
// a real backend.
//
// The console root is `.layout { height: 100% }`, which collapses to 0 in an
// auto-height card. Give it a viewport-height ancestor (100vh) so the height
// chain resolves; paired with cfg.overrides.HugpyConsole single-mode + a fixed
// capture viewport, the full Models view renders.
import { HugpyConsole } from '@hugpy/ui'
import { demoFetch } from '../../src/showroom/demoFetch.js'

export function Console() {
  return (
    <div style={{ height: '100vh', width: '100%' }}>
      <HugpyConsole fetch={demoFetch} />
    </div>
  )
}
