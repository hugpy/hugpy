// Early, boot-time install of the showroom demo shim.
//
// On the dedicated demo host (demo.hugpy.ai) the backend is a hard 404 wall: the
// vhost proxies / to the SPA but 404s every /api and /v1 path. The showroom
// wraps the REAL <Console/> in a canned fetch shim (see ./demoFetch), but that
// shim was historically installed inside <ShowroomConsole>'s first render —
// which is far too late. The very first things the app does on ANY route are:
//   • <AuthProvider> resolves GET /api/auth/config (top of the tree, before the
//     console gate is even reached),
//   • <Landing> (the "/" index) fires GET /api/readiness via <ReadinessPanel>.
// Both go out as real network calls and 404 on the demo host, spraying console
// errors, before <ShowroomConsole> ever mounts.
//
// The fix: install the shim at module-load, before React renders, whenever the
// showroom is the guaranteed destination — i.e. on the exact demo host. This is
// idempotent with <ShowroomConsole>'s own installDemoMode() (guarded by the
// `installed` flag in ./demoFetch), so the brochure flow on hugpy.ai
// (?demo=1/probe → ShowroomConsole installs on mount) is untouched: isDemoHost()
// is false there, so this early hook does nothing and the existing
// showroom-active conditions still drive install exactly as before.

import { isDemoHost } from '../runtime/localCentral.js'
import { installDemoMode } from './demoFetch.js'

/** Install the demo shim now iff this is the dedicated demo host. Idempotent. */
export function installDemoModeIfDemoHost() {
  if (isDemoHost()) installDemoMode()
}
