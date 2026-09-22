# Cleanup — files to delete before installing this version

The earlier rebuild left dead files in your tree that:

- reference a `styles` CSS-module that no longer exists, or
- live in the wrong folder (e.g. `imports/Icons.tsx`, `imports/sidebar.tsx`), or
- duplicate something that already lives elsewhere.

If you leave them in place, `tsc` and Vite will both complain on first
build. Delete the following files outright — nothing in this version
imports any of them:

```
src/ui/ChatDisplay.tsx          # folded into Thread.tsx + MessageTurn.tsx
src/ui/ChatInputPanel.tsx       # folded into Composer.tsx
src/ui/ChatControls.tsx         # folded into ComposerControls.tsx
src/ui/RenderChat.tsx           # replaced by ChatResponse.tsx
src/ui/Icons2.tsx               # duplicate of Icons.tsx
src/imports/Icons.tsx           # duplicate, wrong folder
src/imports/sidebar.tsx         # duplicate, wrong folder, wrong case
src/imports/chat.module.css     # CSS module — no longer used
```

Verify nothing references the dead names afterwards:

```sh
grep -rn "styles\.\|ChatDisplay\|ChatInputPanel\|ChatControls\|RenderChat\|chat\.module\.css" src/chat/src
```

Should come back empty. If it doesn't, that's your remaining work.

## Files in this version that REPLACE the originals

Drop these on top of `src/chat/src/`:

```
src/main.tsx                    # new shell — empty-state + docked-state branches
src/styles/chat.css             # CSS entry (unchanged from last round)
src/styles/tokens.css           # convo's actual token values (unchanged)
src/styles/components.css       # composer-btn, __menu-item, control-chip-* (refined)
src/ui/Sidebar.tsx              # refined to match convo's sidebar structure
src/ui/ThreadHeader.tsx         # model name + chevron-down trigger
src/ui/Thread.tsx               # scrollable transcript only; empty-state lives in main.tsx
src/ui/MessageTurn.tsx          # convo's actual look: no bubble for assistant, soft pill for user
src/ui/ChatResponse.tsx         # inline prose, code detection drops to <pre>
src/ui/ChatTurnActions.tsx      # icon-first ghost buttons (hover-revealed by parent)
src/ui/Composer.tsx             # pill — controls REMOVED from inside the surface
src/ui/ComposerControls.tsx     # thin chip strip living BELOW the composer
src/ui/Disclaimer.tsx           # small muted footer text
src/ui/Icons.tsx                # adds chevron-down, logo-dot
src/ui/index.ts                 # barrel without the dead exports
src/imports/index.ts            # unchanged
```

`src/utilities/**`, `src/imports/types.ts`, `src/imports/constants.ts`,
and your top-level `src/index.ts` barrel are untouched.
