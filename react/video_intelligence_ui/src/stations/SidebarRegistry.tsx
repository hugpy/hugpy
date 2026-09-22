// Round 9 — the ONE left sidebar of the arm shell grows a THIRD, GUARDED tab:
// "Settings". The shell's WorkbenchSidebar owns the [ Active Processes | Session
// Library | Settings ] tab strip, but the Settings tab's CONTENTS (and whether it
// appears at all) belong to whatever station is currently mounted. This tiny context
// is the seam:
//
//   • A station that has config knobs REGISTERS on mount (registerSettings) — that
//     flips `settingsActive` true so the sidebar reveals the Settings tab. Stations
//     that do NOT register (crop stations, and — for now — the Generate image/scene/
//     movie modes) are completely unaffected: no third tab, zero behaviour change.
//   • The sidebar publishes the Settings tabpanel's DOM node via `setSettingsHost`
//     (a callback ref). The mounted station renders its knobs INTO that node with a
//     React portal, so the knob STATE lives in the station (reactive) while the DOM
//     lands in the shared sidebar.
//   • The same pattern carries STUDIO's in-flight renders into the shared Active
//     Processes tab (studio jobs are not in the jobTracker — StationId has no
//     "studio" — so they cannot join it the normal way). `activeExtraHost` is a
//     portal target rendered at the foot of the Active Processes panel;
//     `activeExtraActive` lets the panel suppress its "no active processes" empty
//     copy when a station has portaled rows in.
//
// Additive by construction: a station outside the provider gets an inert fallback
// (no-op setters, everything false/null) so nothing throws.
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export interface SidebarRegistryValue {
  /** The Settings tabpanel DOM node (portal target) — null until that tab renders. */
  settingsHost: HTMLElement | null;
  /** Callback ref the sidebar attaches to its Settings tabpanel. */
  setSettingsHost: (el: HTMLElement | null) => void;
  /** True when ≥1 station has registered Settings content — gates the Settings tab. */
  settingsActive: boolean;
  /** A station calls this on mount; the returned fn unregisters on unmount. */
  registerSettings: () => () => void;

  /** The Active-Processes "extra rows" portal target (studio in-flight renders). */
  activeExtraHost: HTMLElement | null;
  setActiveExtraHost: (el: HTMLElement | null) => void;
  /** True when a station has registered extra Active rows — suppresses empty copy. */
  activeExtraActive: boolean;
  registerActiveExtra: () => () => void;
}

const FALLBACK: SidebarRegistryValue = {
  settingsHost: null,
  setSettingsHost: () => {},
  settingsActive: false,
  registerSettings: () => () => {},
  activeExtraHost: null,
  setActiveExtraHost: () => {},
  activeExtraActive: false,
  registerActiveExtra: () => () => {},
};

const SidebarRegistryContext = createContext<SidebarRegistryValue | null>(null);

export function SidebarRegistryProvider({ children }: { children: ReactNode }) {
  const [settingsHost, setSettingsHost] = useState<HTMLElement | null>(null);
  const [settingsCount, setSettingsCount] = useState(0);
  const [activeExtraHost, setActiveExtraHost] = useState<HTMLElement | null>(null);
  const [activeExtraCount, setActiveExtraCount] = useState(0);

  // Reference-counted so overlapping mounts (or React 18 StrictMode's double-invoke)
  // never leave the tab stuck visible or hidden.
  const registerSettings = useCallback(() => {
    setSettingsCount((n) => n + 1);
    return () => setSettingsCount((n) => Math.max(0, n - 1));
  }, []);
  const registerActiveExtra = useCallback(() => {
    setActiveExtraCount((n) => n + 1);
    return () => setActiveExtraCount((n) => Math.max(0, n - 1));
  }, []);

  const value = useMemo<SidebarRegistryValue>(
    () => ({
      settingsHost,
      setSettingsHost,
      settingsActive: settingsCount > 0,
      registerSettings,
      activeExtraHost,
      setActiveExtraHost,
      activeExtraActive: activeExtraCount > 0,
      registerActiveExtra,
    }),
    [settingsHost, settingsCount, activeExtraHost, activeExtraCount, registerSettings, registerActiveExtra],
  );

  return (
    <SidebarRegistryContext.Provider value={value}>
      {children}
    </SidebarRegistryContext.Provider>
  );
}

/** Read the sidebar registry. Returns an inert fallback outside a provider (never throws). */
export function useSidebarRegistry(): SidebarRegistryValue {
  return useContext(SidebarRegistryContext) ?? FALLBACK;
}
