// The arm shell: the sitewide hugpy navbar + the persistent session-library
// sidebar + the routed workbench. Every built station now shares ONE workbench
// (the `group:"studio"` group): its `.vi-subtabs` strip — rendered by
// WorkbenchStation on the dynamic `/:workbenchTab` route — is the only tab bar,
// and the old Studio/Generate top-level split is retired. The sidebar mounts
// ONCE here so it stays static across every sub-tab (including Generate). Deep
// links (/video/frames, /video/generate …) resolve under the router basename.
import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { DEFAULT_STATION_ID } from "./registry";
import { WorkbenchStation } from "./WorkbenchStation";
import { WorkbenchSidebar } from "./WorkbenchSidebar";
import { SidebarRegistryProvider } from "./SidebarRegistry";
import Navbar from "../nav/Navbar";
import DemoBanner from "../demo/DemoBanner";

export function StationShell() {
  // Narrow-width (<62rem) only: the sidebar becomes an off-canvas drawer toggled
  // by the tab-strip hamburger (WorkbenchStation). At >=62rem CSS keeps the
  // sidebar in its grid column regardless of this flag, so it stays inert there.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  // Close the drawer whenever the route/sub-tab changes. Desktop is unaffected —
  // the sidebar shows via the grid no matter what this flag says.
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Close the drawer on Escape while it is open.
  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sidebarOpen]);

  return (
    <div className="vi-shell">
      {/* Nav + demo banner = ONE header stack pinned at the top (operator
          directive 2026-07-21). In .vi-shell (a 100dvh flex column that never
          scrolls — content scrolls deep inside the workbench body) the stack is
          simply the non-growing first flex item, so the banner sits directly
          under the nav and pushes the content region down with no offset math. */}
      <header className="vi-header-stack">
        <Navbar />
        <DemoBanner />
      </header>
      <main className="vi-main">
        {/* Round 9: the whole workbench (the shared sidebar AND the routed stations)
            shares the sidebar registry, so a mounted station can register knobs into
            the sidebar's Settings tab / in-flight rows into Active Processes. */}
        <SidebarRegistryProvider>
        <div className={sidebarOpen ? "vi-workbench is-open" : "vi-workbench"}>
          <aside className="vi-workbench-sidebar">
            <WorkbenchSidebar />
          </aside>
          <div className="vi-workbench-main">
            <Routes>
              {/* The single studio workbench: ONE WorkbenchStation on the dynamic
                  sub-tab param. It stays mounted across sub-tab switches (incl.
                  Generate → GenerateStation as the 4th sub-tab) so the sidebar is
                  static. React-Router ranks the `*` redirect below it, so `/`
                  (index) and any unknown path land on the default station. The
                  sidebar toggle is drilled in so the mobile hamburger can live as
                  the leading element of the sub-tab strip. */}
              <Route
                path="/:workbenchTab"
                element={
                  <WorkbenchStation
                    sidebarOpen={sidebarOpen}
                    onToggleSidebar={() => setSidebarOpen((v) => !v)}
                  />
                }
              />
              <Route path="*" element={<Navigate to={`/${DEFAULT_STATION_ID}`} replace />} />
            </Routes>
          </div>
          {/* Mobile-only scrim behind the drawer — click to close. Fixed-position,
              so its DOM slot here is cosmetic; CSS hides it at >=62rem. */}
          {sidebarOpen && (
            <button
              type="button"
              className="vi-scrim"
              aria-label="Close session library"
              onClick={() => setSidebarOpen(false)}
            />
          )}
        </div>
        </SidebarRegistryProvider>
      </main>
    </div>
  );
}
