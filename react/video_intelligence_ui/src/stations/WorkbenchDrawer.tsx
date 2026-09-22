// WorkbenchDrawer — the ONE overlay window behind the sidebar's
// `.vi-side-toolbar` deep-links, with Console | ComfyUI | Active as tabs inside it.
//
// ── WHY ONE WINDOW (operator ask, 2026-08-06) ──────────────────────────────
// "Having the 3 options as tab selectable within the overlay window would be a
// huge convenience." Before this, each panel owned its own drawer instance, its
// own open/size state and its own storage key. That made switching between them
// a close-then-reopen — and worse, the scrim sits at z-index 90 OVER the sidebar,
// so while any drawer was open the other two toggles were not even clickable.
// Now there is a single drawer: one open flag, one half/full size, one persisted
// record, and a tab strip in its header.
//
// ── WHAT EACH FILE STILL OWNS ──────────────────────────────────────────────
// The shell (drawerShell.tsx) owns the window: portal-to-<body>, sizing,
// Esc-to-close, persistence, the tab strip, and the mount policy. Each panel file
// owns its own tab descriptor and nothing else — crucially, its own security
// posture, which this host must never normalise:
//   * ConsolePanel  — first-party, same-origin iframe, deliberately UNSANDBOXED
//                     (a sandbox without allow-modals breaks its confirm() flows).
//   * ComfyPanel    — user-supplied FOREIGN endpoint: sanitize-then-sandbox gate
//                     (comfyEmbed.ts) plus its endpoint form in the header.
//   * ActivePanel   — a real component whose MOUNT starts a 2s poll.
// They render as three separate elements side by side, so "one window" never
// means "one frame whose src is swapped": the console frame and the comfy frame
// are distinct <iframe>s with independent sandbox attributes, exactly as before.
//
// ── STATE LIVES HERE, NOT IN THE PORTAL ────────────────────────────────────
// This component is mounted by WorkbenchSidebar and stays mounted whether the
// drawer is open or shut, so tab-owned state hoisted into it (ComfyUI's endpoint
// field) survives both tab switches and a close/reopen.
import { useComfyDrawerTab } from "./ComfyPanel";
import { consoleDrawerTab } from "./ConsolePanel";
import { activeDrawerTab } from "./ActivePanel";
import { toksDrawerTab } from "./ToksPanel";
import { DrawerShell, useDrawerState, type LegacyDrawerKey } from "./drawerShell";

type DrawerTabId = "console" | "comfy" | "active" | "toks";

// Strip order = the order the panels arrived in (Console t52, ComfyUI 2026-08-04,
// Active 2026-08-06), which is also the toolbar's existing left-to-right order —
// so the tab strip and the deep-link row read the same way round.
const TAB_IDS = ["console", "comfy", "active", "toks"] as const;

// ONE key now holds {open, size, tab}, replacing the three per-panel records.
const STORAGE_KEY = "vi.workbench.drawer.v1";

// The pre-tabs keys, in strip order. useDrawerState reads them once (first one
// found OPEN wins, so an operator who left ComfyUI open full-screen reopens on
// the ComfyUI tab, full-screen) and then clears them. See migrateLegacy.
const LEGACY_KEYS: readonly LegacyDrawerKey<DrawerTabId>[] = [
  { key: "vi.console.panel.v1", tab: "console" },
  { key: "vi.comfy.panel.v1", tab: "comfy" },
  { key: "vi.active.panel.v1", tab: "active" },
];

export function WorkbenchDrawer() {
  const drawer = useDrawerState<DrawerTabId>(STORAGE_KEY, TAB_IDS, LEGACY_KEYS);
  // ComfyUI's tab is a hook (it owns the endpoint state); the other two are pure
  // descriptors. All three are rebuilt every render — a descriptor is just JSX,
  // and an unselected tab's body element is only MOUNTED if the shell renders it.
  const comfy = useComfyDrawerTab();
  const tabs = [consoleDrawerTab(), comfy, activeDrawerTab(), toksDrawerTab()];

  return (
    <DrawerShell state={drawer} tabs={tabs} tablistLabel="Overlay panels" />
  );
}
