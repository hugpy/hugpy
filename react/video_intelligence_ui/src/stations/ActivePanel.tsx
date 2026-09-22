// ActivePanel — the console-wide Active Processes view in the shared
// expand-into-the-screen drawer (operator ask 2026-08-06: "'Active' as an
// overlay", the same treatment ComfyUI got on 2026-08-04). Watching what the box
// is doing is something you want to do FROM the station you are working in, not
// by navigating away from it.
//
// Since the tab round (2026-08-06) this file contributes ONE TAB to the single
// unified drawer (drawerShell.tsx → WorkbenchDrawer.tsx) — same window as Console
// and ComfyUI, same `.vi-console-panel*` chrome. Its body is the EXISTING
// ActiveProcessesStation, mounted verbatim. Not a second, divergent renderer of
// the same feed: the `/active` tab and this overlay are literally the same
// component, so the placement line, the awaiting_capacity hold, the progress
// bars, the timeline expander and Cancel all behave identically in both places.
//
// ── POLLING: `keepMounted: false` IS THE PAUSE (read before changing it) ────
// The cadence lives in useMediaJobs, which polls ONLY while a consumer is
// mounted (mount → immediate fetch + 2s interval; unmount → interval cleared,
// mounted-ref flipped so a late response never setStates). There is no separate
// pause seam — mount IS the switch. So this tab declares `keepMounted: false`,
// and the poll runs exactly when the operator is watching it:
//   * drawer closed          → the shell renders no portal at all → no poll;
//   * drawer open, other tab → this body is REMOVED (not merely hidden) → no poll;
//   * drawer open, this tab  → mounted → 2s poll.
// Console and ComfyUI take the opposite choice (`keepMounted: true`) because
// their bodies are iframes, where staying mounted costs nothing and unmounting
// would throw away a live session. Do NOT flip this one to true to "keep the
// list warm": a hidden-but-mounted body would poll every 2s behind the ComfyUI
// frame on every station, which is precisely what the drawer was built to avoid.
import { ActiveProcessesStation } from "./ActiveProcessesStation";
import type { DrawerTab } from "./drawerShell";

/** Active Processes' tab in the unified drawer. */
export function activeDrawerTab(): DrawerTab<"active"> {
  return {
    id: "active",
    // The short chrome word, the way the Console tab says "Console" — the
    // station card below carries its own "Active Processes" heading, so a longer
    // label here would only stack the same words twice.
    label: "Active",
    hint: "Open Active Processes alongside this station — no tab switch",
    toggleClassName: "vi-active-toggle",
    toggleLabel: "▤ Active",
    ariaLabel: "Active processes",
    panelClassName: "vi-active-panel",
    keepMounted: false, // see the polling note above — this is the pause
    body: (
      /* The station body is a `.station-card` built for a page column, so it
         gets its own scroll region here (the other tabs' bodies are iframes,
         which scroll themselves). */
      <div className="vi-active-panel-body">
        <ActiveProcessesStation />
      </div>
    ),
  };
}
