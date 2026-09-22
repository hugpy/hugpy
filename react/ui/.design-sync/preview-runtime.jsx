// Preview harness for /design-sync — NOT shipped in the @hugpy/ui npm package.
//
// Bundled into window.HugpyUI as an extraEntry so it shares the SAME runtime
// config singleton and react-router instance as the panels. Exposes one export,
// DemoProvider, used as cfg.provider so every preview card renders the real
// data-driven panels fully populated from the repo's own showroom fixtures —
// no backend, no network. This mirrors how src/showroom/ShowroomConsole drives
// the public, backend-less /console demo.

import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { HugpyProvider, AuthProvider } from '../dist-lib/hugpy-ui.mjs'
import { demoFetch, installDemoMode } from '../src/showroom/demoFetch.js'

export function DemoProvider({ children }) {
  // Global, config-independent patches (Phone-Brick EventSource consensus replay,
  // alert/confirm -> toast). Installed in a lazy initializer so it runs during
  // this provider's render, before any child panel mounts and fires effects —
  // the same ordering ShowroomConsole relies on. Idempotent across cards.
  React.useState(() => {
    installDemoMode()
    return null
  })
  // Panels read their fetch from the bundle's process-wide config; HugpyProvider
  // (the bundle's own export) writes `demoFetch` into that singleton during
  // render, so every hugpyFetch/fetchJson call is answered from fixtures.
  // MemoryRouter satisfies the components that use react-router (Landing, Navbar,
  // BrandMark, the Auth forms) without a host app router. AuthProvider (mode
  // "open", so no /api/auth/config round-trip) satisfies everything that reads
  // useAuth() from context — HugpyConsole, PrivateRoute, and the auth forms.
  return React.createElement(
    MemoryRouter,
    null,
    React.createElement(
      HugpyProvider,
      { fetch: demoFetch, baseUrl: '' },
      React.createElement(AuthProvider, { mode: 'open' }, children),
    ),
  )
}
