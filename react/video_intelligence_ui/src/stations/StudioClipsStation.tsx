// The STUDIO station — a SINGLE-PLANE workspace over the cinema-studio spine (Movie-tab
// style): one scroll with the generate controls on top and the clip viewer + list
// inline below. It renders the shared StudioPlane (the same plane the Generate station's
// "studio" mode mounts), passing registerBridge so this standalone station is the
// "Send to Studio" target.
//
// History (house rule: never delete, archive):
//   • the original single-page test station → _archive/StudioClipsStation.2026-07-07.pre-workspace.tsx.bak
//   • the two-tab (Generate | Library) workspace → _archive/StudioClipsStation.2026-07-08.tabs-workspace.tsx.bak
// The two tabs are retired: the single plane shows generation AND the viewer AND the
// list together, so there is nothing a tab covered that the plane does not. All the
// behaviors are preserved inside the plane's pieces — preset shapes + hero gating,
// prompt_note badge, the durable clip list + 6s poll + Session-Library push, the
// per-row Details expander, filters (now a compact collapsible strip), per-clip actions
// (play / copy path / use-as-source / cancel), failed-row inline errors, and the
// studioBridge "Send to Studio" staging (owned here via registerBridge).
import type { StationSpec } from "./types";
import { StudioPlane } from "./studio/StudioPlane";

export function StudioClipsStation({ spec }: { spec: StationSpec }) {
  return (
    <section className="station-wide vi-studio-clips" aria-label="Studio workspace">
      {/* The single plane. registerBridge: this standalone station is the target of the
          shell sidebar's "Send to Studio" (the Generate-mode plane does not register). */}
      <StudioPlane registerBridge />
    </section>
  );
}
