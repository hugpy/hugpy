# Building with @hugpy/ui

`@hugpy/ui` is a **dark, monospace, terminal-style console** for self-hosted LLM
serving (models, GPU workers, chat, API keys, Discord/phone bridges). Components are
pre-styled React parts imported from `window.HugpyUI.*`. Compose them; style your own
layout glue with the design tokens below.

## Setup & wrapping

- **Stylesheet:** the design tokens (`:root` custom properties) and every component's
  CSS ship in the bound `styles.css` closure — they're already loaded. Don't restyle
  the components; they carry their own classes.
- **`HugpyProvider`** — wrap the tree in it ONCE near the root. Every data-driven panel
  (ModelTable, WorkersPanel, HFSearch, ApiAccess, Discord/Bridge/PhoneBrick panels,
  Landing) reads its backend wiring from this context and from the process-wide config
  it sets (`baseUrl`, `fetch`, `headers`, `credentials`). Without it they call same-origin
  `/api/*` and render their empty/loading state.
- **`AuthProvider`** — required by `Login`/`Register`/`ChangePassword`/`LoginForm`/
  `Logout`/`PrivateRoute` and `HugpyConsole`; anything calling `useAuth()` throws
  `"useAuth must be used inside <AuthProvider>"` without it. Props: `mode` (`"open"` |
  `"external"`), `base`, `endpoints`, `fetch`.
- **`HugpyConsole`** is the whole console in one component (tabs: Models / Add / Compute /
  API). It expects a height-bearing parent (its root is `height:100%`) — give it a sized
  container (e.g. `100vh`). It still needs `AuthProvider` above it.
- Routing components (Navbar, Landing, the auth forms) use `react-router-dom` — they need
  a Router ancestor in a real app.

## Styling idiom — CSS custom-property tokens (NOT utility classes, NOT style props)

The whole UI is **monospace** (`font-family: var(--font-mono)`) on **dark layered
surfaces**. For your own layout/containers, use these real tokens — never invent hexes:

| Group | Tokens |
|---|---|
| Surfaces (dark→raised) | `--bg` (page) · `--surface` · `--surface-1` (panels) · `--surface-2` (controls) · `--surface-3` (menus/popovers) |
| Text | `--text` (body) · `--bright` (emphasis) · `--muted` (secondary) |
| Borders | `--border` · `--border-strong` |
| Accent | `--accent` / `--cyan` (signature cyan) · `--accent-dim` |
| Status | `--green`/`--success` · `--yellow`/`--warning` · `--red`/`--danger` · `--blue` · `--purple` |
| Shape / type | `--radius` (6px) · `--radius-lg` (10px) · `--shadow-menu` · `--font-mono` |

Components style themselves through per-part class families (e.g. `.chat-*`, `.model-table`,
`.landing-*`, `.navbar-*`, `.wp-*`, and buttons like `.btn-primary`/`.btn-send`). You don't
write those — they're shipped. Read the bound `styles.css` (and a component's `.prompt.md`
/ `.d.ts`) before styling anything adjacent.

## Idiomatic example

```tsx
import { HugpyProvider, ModelTable } from '@hugpy/ui'

export function ModelsView({ models, jobsByModel }) {
  const noop = () => {}
  return (
    <HugpyProvider baseUrl="https://api.hugpy.ai">
      <div style={{ padding: 16, background: 'var(--surface-1)', color: 'var(--text)',
                    fontFamily: 'var(--font-mono)', borderRadius: 'var(--radius-lg)' }}>
        <h2 style={{ color: 'var(--bright)' }}>Models</h2>
        <ModelTable
          models={models} jobsByModel={jobsByModel} activeChat={null}
          onDownload={noop} onChat={noop} onDelete={noop} onPrune={noop}
          onSetMedia={noop} onCancel={noop} onRetry={noop}
          onAssignWorker={noop} onProbeWorker={noop}
        />
      </div>
    </HugpyProvider>
  )
}
```

# HugpyUI (@hugpy/ui@0.2.0)

This design system is the published @hugpy/ui React library, bundled as a single
browser global. All 19 components are the real upstream code.

## Where things are

- `_ds_bundle.js` — the whole-DS bundle at the project root; loads every component to `window.HugpyUI`. First line is a `/* @ds-bundle: … */` metadata header.
- `styles.css` — the single stylesheet entry: it `@import`s the tokens, fonts, and component styles (`_ds_bundle.css`). Link this one file.
- `components/<group>/<Name>/<Name>.prompt.md` (example JSX + variants), `<Name>.d.ts` (types), `<Name>.html` (variant grid).
- `tokens/*.css` — CSS custom properties, names verbatim from upstream.
- `fonts/` — `@font-face` files + `fonts.css` (when the package ships fonts).

For a specific component, `read_file("components/<group>/<Name>/<Name>.prompt.md")`.

## Loading

Add these two lines to your page once (React must be on the page first):

```html
<link rel="stylesheet" href="styles.css">
<script src="_ds_bundle.js"></script>
```

Components are then available at `window.HugpyUI.*`. Mount into a dedicated child node (e.g. `<div id="ds-root">`), not the host page's own React root, so the two trees don't collide:

```jsx
const { ApiAccess } = window.HugpyUI;
ReactDOM.createRoot(document.getElementById('ds-root')).render(<ApiAccess />);
```

Wrap the tree in the provider — most components read theme/i18n from context:

```jsx
<DemoProvider>{children}</DemoProvider>
```

## Tokens

25 CSS custom properties from @hugpy/ui. Names are
preserved verbatim from upstream. They are declared inside `_ds_bundle.css` (this DS ships one compiled stylesheet rather than separate token files).

- **color** (5): `--surface`, `--surface-1`, `--surface-2`, …
- **typography** (1): `--font-mono`
- **radius** (2): `--radius`, `--radius-lg`
- **shadow** (1): `--shadow-menu`
- **other** (16): `--bg`, `--border`, `--muted`, …

## Components

### general
- `ApiAccess`
- `BridgePanel`
- `ChatPanel`
- `DiscordPanel`
- `HFSearch`
- `HugpyConsole`
- `Landing`
- `ModelTable`
- `PeersBar`
- `PhoneBrickPanel`
- `WorkersPanel`

### auth
- `AuthProvider`
- `ChangePassword`
- `Login`
- `LoginForm`
- `Logout`
- `PrivateRoute`
- `Register`

### runtime
- `HugpyProvider`
