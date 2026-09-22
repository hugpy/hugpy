// Public entry point for the @hugpy/ui package.
//
// Layered surface:
//   • runtime config — point the UI at any hugpy backend
//   • individual components — compose your own console
//   • <HugpyConsole/> — the whole console as one drop-in
//
// Consumers import the stylesheet once: `import '@hugpy/ui/style.css'`.
// That bundle carries the design tokens (CSS custom properties on :root) plus
// every component's styles.

import './index.css'

// ── runtime config / backend wiring ────────────────────────────────────────
export {
  configureHugpy,
  getHugpyConfig,
  resetHugpyConfig,
  resolveApiUrl,
  resolveApiOrigin,
  hugpyFetch,
} from './runtime/config'
export type { HugpyRuntimeConfig } from './runtime/config'

export { HugpyProvider, useHugpyConfig } from './runtime/HugpyProvider'
export type { HugpyProviderProps } from './runtime/HugpyProvider'

// ── low-level API helpers ───────────────────────────────────────────────────
export { fetchJson, uploadFile } from './api'
export type { ApiErrorPayload, UploadFileResponse } from './api'

// ── individual panels ───────────────────────────────────────────────────────
export {
  ApiAccess,
  Landing,
  ChatPanel,
  HFSearch,
  ModelTable,
  PeersBar,
  WorkersPanel,
  PhoneBrickPanel,
  AgentNodesPanel,
  DiscordPanel,
  BridgePanel,
} from './components'

// ── auth (opt-in; requires react-router-dom; forms also require @mui/material) ─
// AuthProvider is the single source of truth for auth state + networking; the
// forms are pure UI that consume it. Configure via props (base/mode/endpoints/
// fetch/credentials) or let it auto-discover from GET /api/auth/config.
//   • AuthProvider/useAuth, LoginForm, PrivateRoute — dependency-light (no MUI)
//   • Login/Register/ChangePassword/Logout — MUI-styled form components
export {
  AuthProvider,
  useAuth,
  LoginForm,
  PrivateRoute,
  Login,
  Logout,
  Register,
  ChangePassword,
  getAuthConfig,
  getAuthBase,
} from './Auth'
export type {
  AuthState,
  AuthUser,
  AuthResult,
  AuthEndpoints,
  AuthContextValue,
  AuthProviderProps,
  AuthConfig,
} from './Auth'

// ── layered: the whole console ──────────────────────────────────────────────
export { HugpyConsole } from './HugpyConsole'
export type { HugpyConsoleProps } from './HugpyConsole'
