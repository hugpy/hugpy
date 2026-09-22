// The Studio workbench — the studio sub-tabs nav + the active station body. The
// session-library sidebar and the surrounding `.vi-workbench` grid now live at the
// shell level (StationShell owns them once, for EVERY route), so this component
// renders ONLY into the shell's `.vi-workbench-main` column. StationShell mounts
// ONE WorkbenchStation for the whole studio group (routed on the dynamic
// `/:workbenchTab` segment), so switching sub-tabs never remounts it. The wrapped
// station bodies (each a `.station-card`, some using the `.vi-studio` grid) render
// untouched inside the main column.
import { NavLink, Navigate, useParams } from "react-router-dom";
import { STATIONS, DEFAULT_STATION_ID } from "./registry";
import { PlaceholderStation } from "./PlaceholderStation";
import type { StationSpec } from "./types";

// The workbench sub-tabs, straight off the registry — every built station now
// shares the "studio" group (image-crop, audio-crop, frames, generate), so this
// one strip IS the whole tab bar.
const STUDIO_STATIONS = STATIONS.filter((s) => s.group === "studio");
// The fallback must be a station that actually HAS a tab. This used to be
// STUDIO_STATIONS[0], taken before the `!navHidden` filter applied to the strip
// below — so if the first registered station were ever navHidden (studio-clips
// and, since 2026-08-04, comfy both are), the redirect target would be a page
// with no tab in the strip and nothing rendering as active.
const DEFAULT_STUDIO_ID =
  STUDIO_STATIONS.find((s) => !s.navHidden)?.id ?? DEFAULT_STATION_ID;

// Same active-guard the shell uses: the real component when active, else the
// honest placeholder. Kept local so the workbench mirrors StationBody exactly.
function StudioBody({ spec }: { spec: StationSpec }) {
  const Body = spec.status === "active" && spec.component ? spec.component : null;
  return Body ? <Body spec={spec} /> : <PlaceholderStation spec={spec} />;
}

interface WorkbenchStationProps {
  /** Mobile drawer open-state (owned by StationShell). */
  sidebarOpen: boolean;
  /** Toggle the mobile session-library drawer. */
  onToggleSidebar: () => void;
}

export function WorkbenchStation({
  sidebarOpen,
  onToggleSidebar,
}: WorkbenchStationProps) {
  const { workbenchTab } = useParams();

  const active = STUDIO_STATIONS.find((s) => s.id === workbenchTab);
  // A non-studio param (or bare group route) lands on the default studio sub-tab.
  if (!active) return <Navigate to={`/${DEFAULT_STUDIO_ID}`} replace />;

  // Renders directly into the shell's `.vi-workbench-main` column: the sub-tabs
  // nav then the active body. No sidebar/grid here — the shell owns those. The
  // leading control in the strip is the mobile-only "Session library" hamburger
  // (hidden at >=62rem via CSS); it toggles the shell's off-canvas drawer.
  return (
    <>
      <nav className="vi-subtabs" aria-label="Studio">
        <button
          type="button"
          className="vi-hamburger"
          aria-label="Session library"
          title="Session library"
          aria-expanded={sidebarOpen}
          onClick={onToggleSidebar}
        >
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
            <path
              d="M2 4.5h14M2 9h14M2 13.5h14"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </button>
        {STUDIO_STATIONS.filter((s) => !s.navHidden).map((s) => (
          <NavLink
            key={s.id}
            to={`/${s.id}`}
            className={({ isActive }) =>
              isActive ? "vi-subtab vi-subtab-active" : "vi-subtab"
            }
          >
            {s.title}
          </NavLink>
        ))}
      </nav>
      <div className="vi-workbench-body">
        <StudioBody spec={active} />
      </div>
    </>
  );
}
