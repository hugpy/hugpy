// Station contract for the video-intelligence arm — the PageSpec analog at this
// layer. A station is a capability page: it consumes MediaRefs, builds ONE frozen
// spec type, enqueues it on the job bus, and subscribes to results by job_id.
// The registry (registry.ts) is the single place stations are declared; the
// shell renders whatever is registered. Registries over globals, top to bottom.
import type { ComponentType } from "react";

export type StationStatus = "planned" | "active";

export interface StationSpec {
  /** Stable id — also the route segment (/#/<id> equivalent under the router). */
  id: string;
  /** Tab label, matching the sample image's top-tab affordance. */
  title: string;
  /** One-line purpose shown on the placeholder card. */
  blurb: string;
  /** ROADMAP.md phase that turns this station real. */
  phase: string;
  /** Backend spec type this station will build (from the architecture map). */
  buildsSpec: string;
  /** Planned affordances, rendered as the placeholder checklist. */
  planned: string[];
  /**
   * Workbench group id. Every built station now shares the single "studio"
   * group and appears as a sub-tab of the one workbench (the old Studio/Generate
   * top-level split is retired). Kept optional so a future station can opt out.
   */
  group?: string;
  /**
   * Hide this station from the sub-tab NAV strip while keeping its ROUTE live —
   * WorkbenchStation still resolves /<id>, so deep-links (e.g. the sidebar's
   * "Send to Studio") keep working. Retires a redundant tab from the nav without
   * deleting the page.
   */
  navHidden?: boolean;
  status: StationStatus;
  /**
   * The real body, once the station is built. When present AND status==="active"
   * the shell renders this instead of the PlaceholderStation. Receives its own
   * StationSpec so a component can read its id/title/phase.
   */
  component?: ComponentType<{ spec: StationSpec }>;
}
