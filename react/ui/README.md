# @hugpy/ui

Embeddable React UI for [hugpy](https://hugpy.ai) — the panels that make up the
hugpy console (chat, model table, GPU workers, HF search, API keys, Discord
bindings), plus a single drop-in `<HugpyConsole/>`.

This package is **layered**:

- **runtime config** — point the UI at any hugpy backend
- **individual panels** — compose your own console
- **`<HugpyConsole/>`** — the whole console in one component

## Install

```bash
npm install @hugpy/ui
# peers (most React apps already have these):
npm install react react-dom react-router-dom
# only if you use the MUI-styled auth forms (Login/Register/ChangePassword/Logout):
npm install @mui/material @mui/styled-engine
```

Import the stylesheet once, anywhere in your app:

```js
import '@hugpy/ui/style.css'
```

The stylesheet carries the design tokens (CSS custom properties on `:root`) and
every panel's styles.

## Quick start — the whole console

```jsx
import { HugpyConsole } from '@hugpy/ui'
import '@hugpy/ui/style.css'

export default function App() {
  return <HugpyConsole baseUrl="https://api.hugpy.ai" />
}
```

`<HugpyConsole/>` is router-free — no `<BrowserRouter>` required.

## Quick start — individual panels

Wrap the part of your tree that uses hugpy panels in a `<HugpyProvider>` (it sets
the backend address the panels talk to), then drop panels in:

```jsx
import { HugpyProvider, ChatPanel, ModelTable } from '@hugpy/ui'
import '@hugpy/ui/style.css'

<HugpyProvider baseUrl="https://api.hugpy.ai">
  <ModelTable models={models} /* … */ />
  <ChatPanel /* … */ />
</HugpyProvider>
```

## Backend wiring

Every API call resolves through the runtime config. The default `baseUrl` is
empty, meaning **same-origin / relative paths** — that's how the bundled console
works behind its proxy, unchanged.

```ts
configureHugpy({
  baseUrl: 'https://api.hugpy.ai', // '' = same origin (default)
  headers: () => ({ Authorization: `Bearer ${getToken()}` }), // optional, per-request
  credentials: 'include',          // optional, cookie auth across origins
  fetch: myFetch,                  // optional, custom fetch
})
```

`<HugpyProvider>` accepts the same options as props and applies them for its
subtree. You can also call `configureHugpy(...)` once at startup instead.

### Resolvers (exported for advanced use)

- `resolveApiUrl(path)` — relative `/api/...` → absolute against `baseUrl`
- `resolveApiOrigin()` — the configured origin (falls back to `window.location.origin`)
- `hugpyFetch(path, init)` — `fetch` with URL resolution + configured headers
- `fetchJson(path, init)` — `hugpyFetch` + JSON parsing with real error messages

## Auth (optional)

Auth is opt-in and follows one convention: **`AuthProvider` is the single source
of truth for both auth state and auth networking.** The forms are pure UI that
call the context — you never wire up fetch calls yourself.

```jsx
import { AuthProvider, useAuth, LoginForm } from '@hugpy/ui'

<AuthProvider>
  <LoginForm />
</AuthProvider>

// anywhere inside:
const { state, signIn, signOut, signUp, changePassword, refresh } = useAuth()
// state: { status: 'checking' | 'guest' | 'authed', user? }
```

By default the provider asks the API instance how to authenticate via
`GET /api/auth/config` → `{ mode: 'open' }` (no login wall) or
`{ mode: 'external', base }` (login against a separate auth service). Override
any of it with props:

```jsx
<AuthProvider
  base="https://auth.example.com"          // auth service origin
  mode="external"                            // or "open"
  endpoints={{ login: '/signin', me: '/session' }}  // per-path overrides
  credentials="include"                      // cookie auth (default)
  fetch={myFetch}                            // inject headers, etc.
/>
```

Two tiers of UI: `LoginForm` and `PrivateRoute` are dependency-light (no MUI);
the richer `Login` / `Register` / `ChangePassword` / `Logout` forms are
MUI-styled (need `@mui/material`, an **optional** peer). All require
`react-router-dom`.

## Exports

| Export | What |
|---|---|
| `HugpyConsole` | full console, one component |
| `HugpyProvider`, `useHugpyConfig` | backend config context |
| `configureHugpy`, `getHugpyConfig`, `resetHugpyConfig` | imperative config |
| `resolveApiUrl`, `resolveApiOrigin`, `hugpyFetch`, `fetchJson`, `uploadFile` | request helpers |
| `ChatPanel`, `ModelTable`, `WorkersPanel`, `HFSearch`, `PeersBar`, `ApiAccess`, `DiscordPanel`, `PhoneBrickPanel`, `Landing` | panels |
| `AuthProvider`, `useAuth` | auth context (state + signIn/signOut/signUp/changePassword/refresh) |
| `LoginForm`, `PrivateRoute` | lightweight auth UI (no MUI) |
| `Login`, `Register`, `ChangePassword`, `Logout` | MUI-styled auth forms |
| `getAuthConfig`, `getAuthBase` | auth-config discovery helpers |

## Building the package (maintainers)

```bash
npm run build:lib    # → dist-lib/ (ESM + CJS + .d.ts + style.css)
```

`react`, `react-dom`, `react-router-dom`, and `@mui/*` are externalized (peer
dependencies; MUI is optional). The console app itself still builds with
`npm run build` (webpack → `dist/`); the library build is additive and
independent.
