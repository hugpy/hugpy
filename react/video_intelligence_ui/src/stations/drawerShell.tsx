// The SHARED expand-into-the-screen drawer — the shell behind the sidebar's
// `.vi-side-toolbar` toggles (Console, ComfyUI, and — 2026-08-06 — Active).
//
// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
// t52 built the shell inside ConsolePanel. ComfyPanel (2026-08-04) reproduced it
// verbatim so that "the same as console" would be literally true rather than
// approximately true — the right instinct, but by the third overlay (the operator
// ask 2026-08-06: "Active as an overlay too") a copied shell is three chances to
// drift. So the shell is factored out HERE, exactly as it already existed:
// same portal-to-<body>, same `.vi-console-panel*` classes, same localStorage
// open/size persistence, same Esc-to-close, same half/full toggle. Nothing about
// the rendered DOM changed in the move — the class names deliberately keep the
// `vi-console-panel` prefix (they are the drawer's classes, not the console's).
//
// ── ONE WINDOW, THREE TABS (operator ask 2026-08-06, second round) ──────────
// "Having the 3 options as tab selectable within the overlay window would be a
// huge convenience." Three independent drawers meant three open/size states and
// no way to get from one to the next without closing and reopening — and, since
// the scrim covers the sidebar at z-index 90, the toolbar toggles are not even
// clickable while a drawer is open. So there is now exactly ONE drawer instance
// (one open flag, one size, one persisted record) whose header carries a
// `role="tablist"` strip; each DrawerTab contributes its own body, header extras
// and header actions. Selecting a tab swaps the body in place — no close/reopen.
//
// ── MOUNTING POLICY: PER TAB, VIA `keepMounted` (deliberate, read this) ─────
// An unselected tab is one of two things, and the difference is load-bearing:
//   * keepMounted: true  → rendered but `hidden`. `display:none` does NOT tear
//     down or reload an <iframe>, so Console and ComfyUI keep their live frame
//     (and its scroll position, its login session, its half-finished ComfyUI
//     graph) across tab switches. Reloading a ComfyUI workspace every time the
//     operator glanced at Active would make the tab strip worse than the three
//     separate drawers it replaces.
//   * keepMounted: false → not rendered at all. This is for a body whose MOUNT
//     is what starts work: ActivePanel's ActiveProcessesStation polls /media/jobs
//     every 2s for as long as it is mounted, so "hidden but mounted" would keep
//     that poll running behind the Console frame — exactly the always-on poll the
//     panel's own header forbids. Unmounting is how the poll stops; there is no
//     separate pause seam to reach for.
// The whole portal only exists while `open`, so a CLOSED drawer still mounts
// nothing at all — closing remains the way to drop every frame and every poll.
//
// ── PORTALED TO <body> (do not "simplify" this) ─────────────────────────────
// WorkbenchSidebar lives inside `.vi-workbench-sidebar`, which grows a `transform`
// at the <62rem breakpoint (the off-canvas drawer). A transformed ancestor becomes
// the containing block for `position: fixed` descendants, which would trap an
// "expand into the screen" panel inside the sidebar's own narrow box. Portaling to
// <body> sidesteps that regardless of where the toggle button lives in the tree.
import {
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

export type DrawerSize = "half" | "full";

interface PersistedState<T extends string> {
  open: boolean;
  size: DrawerSize;
  /** Which tab was showing — restored so reopening lands where you left off. */
  tab: T;
}

/** A pre-tabs per-panel storage key and the tab it became. See migrateLegacy. */
export interface LegacyDrawerKey<T extends string> {
  key: string;
  tab: T;
}

function parse<T extends string>(
  raw: string | null,
  tabs: readonly T[],
  fallback: PersistedState<T>,
): PersistedState<T> | null {
  if (!raw) return null;
  try {
    const p = JSON.parse(raw);
    return {
      open: typeof p.open === "boolean" ? p.open : fallback.open,
      size: p.size === "full" ? "full" : "half",
      // An unknown tab id (a renamed/removed panel in an older record) falls
      // back to the first tab rather than leaving the strip with nothing selected.
      tab: tabs.includes(p.tab) ? (p.tab as T) : fallback.tab,
    };
  } catch {
    return null; // corrupt record — quiet fallback
  }
}

/**
 * One-time upgrade from the three pre-tabs keys (`vi.console.panel.v1`,
 * `vi.comfy.panel.v1`, `vi.active.panel.v1`) to the single tabbed record.
 *
 * An operator who left, say, the ComfyUI drawer open and full-screen should find
 * the unified drawer open, full-screen, on the ComfyUI tab — not collapsed. So:
 * the first legacy key found OPEN wins outright (open + size + its tab); if none
 * was open, the first parseable record still donates its `size` so the half/full
 * preference survives. The stale keys are cleared by useDrawerState's mount
 * effect (a side effect belongs there, not in a render-phase initializer).
 */
function migrateLegacy<T extends string>(
  tabs: readonly T[],
  legacy: readonly LegacyDrawerKey<T>[],
  fallback: PersistedState<T>,
): PersistedState<T> {
  let size: DrawerSize | null = null;
  for (const entry of legacy) {
    const prev = parse(localStorage.getItem(entry.key), tabs, fallback);
    if (!prev) continue;
    if (prev.open) return { open: true, size: prev.size, tab: entry.tab };
    if (size == null) size = prev.size;
  }
  return size == null ? fallback : { ...fallback, size };
}

function readPersisted<T extends string>(
  storageKey: string,
  tabs: readonly T[],
  legacy: readonly LegacyDrawerKey<T>[],
): PersistedState<T> {
  const fallback: PersistedState<T> = {
    open: false,
    size: "half",
    tab: tabs[0] as T,
  };
  try {
    // The unified record wins whenever it exists — migration is strictly the
    // "never seen the tabbed drawer before" path, so it can only ever run once.
    return (
      parse(localStorage.getItem(storageKey), tabs, fallback) ??
      migrateLegacy(tabs, legacy, fallback)
    );
  } catch {
    return fallback; // storage unavailable (private mode) — quiet fallback
  }
}

function persist<T extends string>(
  storageKey: string,
  state: PersistedState<T>,
): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(state));
  } catch {
    /* best-effort — the in-memory state still drives this session */
  }
}

export interface DrawerState<T extends string = string> {
  open: boolean;
  size: DrawerSize;
  tab: T;
  /** Open the drawer ON a tab — what a toolbar deep-link does. */
  openTab: (tab: T) => void;
  /** Switch tabs without touching open/size — what the in-window strip does. */
  selectTab: (tab: T) => void;
  close: () => void;
  toggleSize: () => void;
}

/**
 * The drawer's open/size/tab state, persisted under ONE key.
 *
 * Persisting every change is the t52 spec ("feels stable across visits"):
 * reopening /video restores exactly the open/closed + half/full state left
 * behind, instead of always starting collapsed. The selected tab rides along in
 * the same record for the same reason — the operator who lives in ComfyUI should
 * not have to re-pick it every visit.
 *
 * Note the SINGLE state object: the pre-tabs version kept two useState calls,
 * each re-reading storage in its initializer. That was fine while the read was
 * pure, but migration must observe the legacy keys exactly once, so one read /
 * one initializer is now load-bearing.
 */
export function useDrawerState<T extends string>(
  storageKey: string,
  tabs: readonly T[],
  legacy: readonly LegacyDrawerKey<T>[] = [],
): DrawerState<T> {
  const [state, setState] = useState<PersistedState<T>>(() =>
    readPersisted(storageKey, tabs, legacy),
  );

  useEffect(() => {
    persist(storageKey, state);
  }, [storageKey, state]);

  // Drop the pre-tabs keys once, after the first commit has already read them.
  // They are pure UI chrome (open/half) and nothing reads them any more, so
  // leaving them would only litter storage with records that can never win again.
  useEffect(() => {
    try {
      for (const entry of legacy) localStorage.removeItem(entry.key);
    } catch {
      /* storage unavailable — nothing to clean up */
    }
    // Intentionally mount-only: `legacy` is a module-level constant at every
    // call site, and re-running this would be a no-op anyway.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openTab = useCallback(
    (tab: T) => setState((s) => ({ ...s, open: true, tab })),
    [],
  );
  const selectTab = useCallback(
    (tab: T) => setState((s) => ({ ...s, tab })),
    [],
  );
  const close = useCallback(() => setState((s) => ({ ...s, open: false })), []);
  const toggleSize = useCallback(
    () =>
      setState((s) => ({ ...s, size: s.size === "full" ? "half" : "full" })),
    [],
  );

  // ESC closes — only listens while the panel is actually open, mirroring
  // StationShell's own Escape-closes-the-mobile-drawer effect.
  useEffect(() => {
    if (!state.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.open, close]);

  return {
    open: state.open,
    size: state.size,
    tab: state.tab,
    openTab,
    selectTab,
    close,
    toggleSize,
  };
}

/** One tab of the unified drawer, contributed by its own panel file. */
export interface DrawerTab<T extends string = string> {
  id: T;
  /** Tab-strip label, and the noun in "Close X" — a plain string for a11y. */
  label: string;
  /** Tooltip, shared by the toolbar deep-link and the in-window tab. */
  hint: string;
  /** Extra class on the toolbar deep-link (e.g. "vi-console-toggle") — sized by app.css. */
  toggleClassName: string;
  /** Toolbar deep-link content (e.g. "▤ Console") — may carry a glyph. */
  toggleLabel: ReactNode;
  /** The dialog's accessible name while this tab is selected (e.g. "hugpy console"). */
  ariaLabel: string;
  /** Extra class on the panel element while selected (e.g. "vi-comfy-panel"). */
  panelClassName?: string;
  /** Optional header content between the tab strip and the actions (e.g. Comfy's endpoint form). */
  headerExtra?: ReactNode;
  /** Optional actions between Half/Full and the ✕ (e.g. "Open in new tab"). */
  headerActions?: ReactNode;
  /** The tab body. */
  body: ReactNode;
  /**
   * Stay in the DOM (just `hidden`) while another tab is selected — TRUE for
   * iframe bodies so they don't reload, FALSE for a body whose mount starts
   * work (a poll). See this file's mounting-policy note; every tab must make
   * this choice deliberately, so it is not optional.
   */
  keepMounted: boolean;
}

export interface DrawerShellProps<T extends string> {
  /** Open/size/tab state — from useDrawerState, owned by the caller. */
  state: DrawerState<T>;
  /** The tabs, in strip order. The first is the fallback for an unknown id. */
  tabs: readonly DrawerTab<T>[];
  /** aria-label for the tab strip itself. */
  tablistLabel: string;
}

/**
 * The toolbar deep-links + the (portaled) drawer window with its tab strip.
 *
 * WHY THREE TOOLBAR BUTTONS SURVIVED the collapse to one window: the row's idiom
 * is one button per named destination (Console, ComfyUI, Active, Share — Share is
 * a separate modal and keeps its own). Collapsing to a single "▤ Panels" button
 * would trade a named one-click destination for an unnamed click-then-pick, and
 * would leave Share as the odd one out in a row of one generic button. So each
 * button stays a DEEP LINK: it opens the one window already showing that tab.
 * They cost nothing extra now — they share one open/size state.
 */
export function DrawerShell<T extends string>({
  state,
  tabs,
  tablistLabel,
}: DrawerShellProps<T>) {
  const { open, size, tab, openTab, selectTab, close, toggleSize } = state;
  const active = tabs.find((t) => t.id === tab) ?? tabs[0];
  const closeText = `Close ${active.label}`;

  return (
    <>
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`${t.toggleClassName} vi-btn vi-btn-sm`}
          // Accurate disclosure state: this tab's panel is only showing when the
          // drawer is open AND it is the selected one.
          aria-expanded={open && tab === t.id}
          onClick={() => openTab(t.id)}
          title={t.hint}
        >
          {t.toggleLabel}
        </button>
      ))}

      {/* Portaled to document.body — see the header note. */}
      {open &&
        createPortal(
          <>
            <button
              type="button"
              className="vi-console-scrim"
              aria-label={closeText}
              onClick={close}
            />
            <div
              className={`vi-console-panel vi-console-panel-${size}${
                active.panelClassName ? ` ${active.panelClassName}` : ""
              }`}
              role="dialog"
              aria-modal="true"
              aria-label={active.ariaLabel}
            >
              <div className="vi-console-panel-bar">
                {/* The strip replaces the old single title — the selected tab IS
                    the title, and it reuses the segmented `.vi-gen-mode` control
                    the sidebar's own tab strip uses, so the two read alike. */}
                <div
                  className="vi-gen-mode vi-drawer-tabs"
                  role="tablist"
                  aria-label={tablistLabel}
                >
                  {tabs.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      role="tab"
                      id={`vi-drawer-tab-${t.id}`}
                      aria-selected={t.id === tab}
                      // Only claim to control a panel that is actually in the
                      // DOM — a non-keepMounted tab has no element to point at
                      // while it is unselected, and a dangling aria-controls is
                      // worse than none.
                      {...(t.keepMounted || t.id === tab
                        ? { "aria-controls": `vi-drawer-panel-${t.id}` }
                        : {})}
                      className={`vi-gen-mode-btn${
                        t.id === tab ? " vi-gen-mode-btn-active" : ""
                      }`}
                      onClick={() => selectTab(t.id)}
                      title={t.hint}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                {active.headerExtra}
                <div className="vi-console-panel-actions">
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    onClick={toggleSize}
                    title={
                      size === "full"
                        ? "Shrink to half the screen"
                        : "Expand to fill the screen"
                    }
                  >
                    {size === "full" ? "Half" : "Full"}
                  </button>
                  {active.headerActions}
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm"
                    onClick={close}
                    aria-label={closeText}
                    title="Close (Esc)"
                  >
                    ✕
                  </button>
                </div>
              </div>

              {/* Bodies. An unselected keepMounted tab stays in the DOM behind
                  `hidden` (iframes survive); an unselected non-keepMounted tab is
                  removed outright (its poll stops). See the mounting-policy note. */}
              {tabs.map((t) => {
                const selected = t.id === tab;
                if (!selected && !t.keepMounted) return null;
                return (
                  <div
                    key={t.id}
                    id={`vi-drawer-panel-${t.id}`}
                    role="tabpanel"
                    aria-labelledby={`vi-drawer-tab-${t.id}`}
                    className="vi-drawer-tabpanel"
                    hidden={!selected}
                  >
                    {t.body}
                  </div>
                );
              })}
            </div>
          </>,
          document.body,
        )}
    </>
  );
}
