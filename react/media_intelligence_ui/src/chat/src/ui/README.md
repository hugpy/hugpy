# Hugpy chat — convo-conformed shell

Full Tailwind v4 + design-token rewrite of the Hugpy chat page so the
markup, class vocabulary, and structural hooks line up with the convo
HTML you pasted in. Composer + sidebar + thread header + disclaimer all
present. Your sampling controls (model / max tokens / temp / top-p /
stream) stay visible inline, restyled — no hidden popovers.

## Drop-in file map

These files replace what's under
`app/src/Components/tools/hugpy/src/chat/src/` in your tree. Path on the
left, what it does on the right.

```
src/main.tsx                  page shell — sidebar + header + main + thread + composer
src/imports/index.ts          re-exports (CSS module no longer surfaced)

src/styles/chat.css           SINGLE css entry — import this once
src/styles/tokens.css         design tokens (light + dark) bound to Tailwind @theme
src/styles/components.css     composer-btn, btn, __menu-item, status pills, etc.

src/ui/Sidebar.tsx            left rail (chat history slot is a placeholder)
src/ui/ThreadHeader.tsx       top bar
src/ui/Thread.tsx             scrollable transcript (replaces ChatDisplay)
src/ui/MessageTurn.tsx        one user msg + one assistant msg pair
src/ui/ChatResponse.tsx       restyled
src/ui/ChatTurnActions.tsx    restyled
src/ui/Composer.tsx           unified-composer form (replaces ChatInputPanel)
src/ui/ComposerControls.tsx   restyled sampling-knob row (replaces ChatControls)
src/ui/Disclaimer.tsx         footer strip
src/ui/Icons.tsx              icon registry (inline SVGs — no sprite dep)
src/ui/index.ts               barrel
```

Files **not** touched (still good as-is):

- `src/utilities/**`  — streamChat / generateChat / cancelChat / history / functions
- `src/imports/types.ts`  — ChatEntry, ChatRequest, StreamEvent
- `src/imports/constants.ts`  — CHAT_MODEL_OPTIONS registry

The old `src/imports/chat.module.css` and the old
`src/ui/{ChatDisplay,ChatInputPanel,ChatControls}.tsx` can be deleted —
nothing references them anymore.

## Tailwind v4 setup

This component pulls Tailwind v4 with CSS-first config (no
`tailwind.config.js` required). If your host app already has Tailwind v4
running, skip to step 3 and merge the `@theme` block from
`src/styles/tokens.css` into your existing entry.

```bash
# 1. install
npm i -D tailwindcss @tailwindcss/vite
```

```ts
// 2. vite.config.ts — register the plugin
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    // ...existing
    tailwindcss(),
  ],
});
```

```ts
// 3. host app entry (e.g. App.tsx or main.tsx)
import "path/to/hugpy/src/chat/src/styles/chat.css";
```

That's the whole setup. No PostCSS config, no JS Tailwind config, no
`@tailwind base/components/utilities` lines — Tailwind v4 does it all
from the CSS entry.

## Theme switching

Explicit wiring, no smart defaults. There is **no**
`prefers-color-scheme` auto-switch. To switch themes, toggle one
attribute on the root:

```ts
document.documentElement.dataset.theme = "dark";   // or remove for light
```

All token values live in `tokens.css` keyed off `:root` (light) and
`[data-theme="dark"]`. To add a third theme variant, add a third
selector block — don't fork the `@theme`.

## Behavior changes worth knowing

- **Textarea name changed**: `prompt` → `prompt-textarea` (matches
  convo). `main.tsx` reads `formData.get("prompt-textarea")` to match.
  If anything else in your tree submitted this form, update the name.
- **No more CSS-module `styles` import**: components reference Tailwind
  utilities + the named classes in `components.css` directly. The
  `styles` re-export is gone from `imports/index.ts`. If host code
  imported `styles` from this module, update or remove.
- **`ChatDisplay` / `ChatInputPanel` / `ChatControls` are gone**: they
  were folded into `Thread` / `Composer` / `ComposerControls`.
- **Icons are inline SVG, not sprite-referenced**: no external
  `sprites-core-*.svg` dependency. Adding icons = adding entries to
  `ICONS` in `Icons.tsx`.

## Convo hooks preserved (1:1 with the source HTML)

Greppable structural hooks kept intact so future-you can map between
this tree and convo's DOM without rediscovering them:

```
<main id="main" tabIndex={-1}>
<div id="thread" class="group/thread flex flex-col min-h-full">
<div id="thread-bottom-container" class="sticky bottom-0 ...">
<aside id="stage-slideover-sidebar" aria-label="Sidebar">
<div id="sidebar-header">
<form data-type="unified-composer">
<div data-composer-surface="true" class="corner-superellipse/1.1 ...">
<textarea name="prompt-textarea" aria-label="Chat with {modelLabel}">
<button data-testid="composer-plus-btn" id="composer-plus-btn">
<button data-testid="composer-send-button">
<button data-testid="composer-stop-button">
<button data-testid="create-new-chat-button">
<button data-testid="close-sidebar-button">
<article data-conversation-turn={chat.id}>
<div data-message-author-role="user" data-message-id="...">
<div data-message-author-role="assistant" data-message-id="...">
.composer-btn / .composer-send-btn / .__menu-item / .icon / .icon-lg
bg-token-* / text-token-* / border-token-*
--sidebar-width / --header-height / --thread-content-margin-* / --composer-radius
```

## Stubs you'll want to wire next

These are visible in the UI but presently just `console.info`:

- composer plus-btn (add files / attachments)
- composer mic / dictation button
- Sidebar "Search chats", "Images" menu rows
- Sidebar "Recents" history list (rendered "No saved chats yet")
- `ChatTurnActions` switch-model / sources / more menus

All of them are real `<button>` elements with the correct ARIA + test
IDs already in place, so wiring them to real handlers later doesn't
disturb the markup.
