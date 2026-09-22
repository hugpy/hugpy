// The station registry — every capability page of the video-intelligence arm,
// declared in one place. Order = tab order. All four start as "planned"
// placeholders (Phase 0); each flips to "active" when its roadmap phase lands.
// See ROADMAP.md for the sequence and hugpy_video_intelligence_map.md for the
// schemas/jobs each station drives.
import type { StationSpec } from "./types";
import { ImageCropStation } from "./ImageCropStation";
import { AudioCropStation } from "./AudioCropStation";
import { FrameExtractStation } from "./FrameExtractStation";
import { GenerateStation } from "./GenerateStation";
import { ComfyStation } from "./ComfyStation";
import { StudioClipsStation } from "./StudioClipsStation";
import { IdentitiesStation } from "./studio/IdentitiesStation";
import { ActiveProcessesStation } from "./ActiveProcessesStation";
import { ScriptFirstStation } from "./ScriptFirstStation";
import { OracleStation } from "./oracle/OracleStation";

export const STATIONS: readonly StationSpec[] = [
  {
    id: "generate",
    title: "Studio",
    blurb:
      "Ordered multimodal prompt (text + images + video) → image, with video parts resolved by the chain registry.",
    phase: "Phases 6–7",
    buildsSpec: "GenerateImageSpec",
    planned: [
      "MediaInputReceptacle collecting an ordered text/image/video prompt",
      "Cropped frames from other stations selectable via FieldSpec propagation",
      "Model dropdown filtered to text2image / image2image tasks",
      "video → frame_extract → frame-pick → conditioning chain (never runner special-casing)",
    ],
    group: "studio",
    status: "active",
    component: GenerateStation,
  },
  {
    id: "studio-clips",
    title: "Studio",
    blurb:
      "Cinema-studio workspace on a single plane: the clip viewer on top with tier-first generate controls (t2v/i2v/v2v, morphing param panel, start image + source clip) below. Produced clips land in the shared left-sidebar Session Library (extend / restyle from there).",
    phase: "Studio",
    buildsSpec: "StudioI2VSpec (enqueues + plays cinema-studio clips)",
    planned: [
      "Single plane: preset tiers (real vs synthetic) + morphing param panel + start-image/source-clip inputs → Generate (POST /video/studio/i2v)",
      "Inline viewer + durable clip list (GET /video/studio/clips): list-left / player-right, autoplay follows the newest clip",
      "Range-seekable playback via GET /video/studio/clip/<job_id> + per-row creation-params Details expander",
      "Compact collapsible filters (status/capability/model) + per-clip actions: play, copy path, use-as-source, cancel",
    ],
    group: "studio",
    // Retired from the nav strip: this standalone "Studio" page mirrors the studio
    // MODE of the renamed Studio (ex-Generate) tab, so it's redundant in the nav.
    // Route kept live (WorkbenchStation still resolves /studio-clips) so the
    // sidebar's "Send to Studio" / "Restyle" deep-links keep working.
    navHidden: true,
    status: "active",
    component: StudioClipsStation,
  },
  {
    id: "identities",
    title: "Identities",
    blurb:
      "The identity-profile library: procure a character's curated reference set once — create, rename, edit its notes/references, and archive — for reuse anywhere via identity_profile:<slug>. The Studio Generate/Movie tabs' \"👤 From profile\" picker is where a saved one gets attached to a render; this station is where it's made and maintained.",
    phase: "Studio",
    buildsSpec: "IdentityProfile (create / PATCH update / archive)",
    planned: [
      "Create panel: name + notes + reference images, via upload or the session-library picker",
      "Grid of saved profiles: thumbnail strip, name, created date, notes preview, reference count",
      "Inline per-card edit: rename (display-only — the slug stays stable) + notes + add/remove references",
      "Archive (never-delete doctrine) per profile",
      "Stage (b), next: '✦ Generate character sheet' — turnaround canon generation from a profile",
    ],
    group: "studio",
    status: "active",
    component: IdentitiesStation,
  },
  {
    id: "image-crop",
    title: "Image Crop",
    blurb:
      "Spatial crop station: draw one or many bboxes over an image, enqueue crop jobs through the bus.",
    phase: "Phase 3",
    buildsSpec: "CropSpec (spatial axis)",
    planned: [
      "SpatialRegionEditor — draggable/resizable bbox, aspect lock, zoom/pan",
      "Numeric x/y/w/h inputs + model-native presets (1:1, 16:9, 1024²)",
      "Multi-crop queue off one source",
      "Enqueue crop → subscribe by job_id → result in the store",
    ],
    group: "studio",
    status: "active",
    component: ImageCropStation,
  },
  {
    id: "audio-crop",
    title: "Audio Crop",
    blurb:
      "Temporal crop station: waveform region editing over uploaded audio or a video's extracted track.",
    phase: "Phase 5",
    buildsSpec: "CropSpec (temporal axis)",
    planned: [
      "TemporalRegionEditor — waveform, draggable region handles, timeline zoom",
      "Scrub + region-loop playback, numeric start/end, spectrogram toggle",
      "Video sources routed through the audio_extract chain first",
      "Multi-region clip queue",
    ],
    group: "studio",
    status: "active",
    component: AudioCropStation,
  },
  {
    id: "frames",
    title: "Frames & Models",
    blurb:
      "The Options panel: fps / quality / format / window / max-frames knobs plus the model dropdown, producing the frame grid.",
    phase: "Phase 4",
    buildsSpec: "FrameExtractSpec",
    planned: [
      "Explicit knobs — fps, quality (range per fmt), fmt, window, max_frames (loud cap)",
      "Model dropdown sourced from the ModelConfig registry, filtered by task",
      "Virtualized/paginated frame grid; frames land as MediaRefs, never inlined",
      "Parallel ffmpeg fan-out on the worker (wmtrim pattern)",
    ],
    group: "studio",
    status: "active",
    component: FrameExtractStation,
  },
  {
    id: "comfy",
    title: "ComfyUI",
    blurb:
      "Viewer station: embed a running ComfyUI (an HTTPS endpoint) in an iframe, with a user-editable, session-persisted URL.",
    phase: "Viewer",
    buildsSpec: "— (embeds an external endpoint; enqueues no job)",
    planned: [
      "Editable endpoint URL, persisted per-tab in sessionStorage (default from VITE_HUGPY_COMFY_URL)",
      "Scheme-allowlisted iframe src (http/https only) — rejects javascript:/data:/blob:/file:",
      "Mixed-content guidance: an HTTPS page can't frame http:// or localhost endpoints",
      "Always-available Open-in-new-tab fallback for endpoints that refuse embedding (X-Frame-Options)",
    ],
    group: "studio",
    status: "active",
    // 2026-08-04 (operator): ComfyUI is reached from the SIDEBAR OVERLAY
    // (ComfyPanel, beside Console) rather than a studio sub-tab, so it leaves the
    // strip. The station stays registered and navHidden so an existing /comfy
    // bookmark still resolves here instead of bouncing to the default tab —
    // WorkbenchStation resolves `/:workbenchTab` against the UNFILTERED list and
    // only the strip filters on !navHidden.
    navHidden: true,
    component: ComfyStation,
  },
  {
    id: "script",
    title: "Script",
    blurb:
      "Script-first generation: an immutable input snapshot, one authored plot + screenplay, a locked continuity bible and shot plan, and every segment prompt compiled from that lock as a SIBLING — never from another generated prompt.",
    phase: "k104/k110/k114 (oracle video)",
    buildsSpec: "GenerationSnapshot → ProductionLock → SegmentSpec[] (POST /video/script/runs)",
    planned: [
      "Input snapshot: which prompts were captured, with hashes; prompts persisted after the run began are shown EXCLUDED with the reason",
      "Plot + screenplay authored on the live text route or hand-edited, validated through the same constructor — validator errors verbatim, never a coerced artifact",
      "Lock the screenplay, continuity bible, audio master and shot plan together; after the lock, edits are revisions with a mandatory reason",
      "Segment cards fanning out from one 'Locked artifacts' parent: prompt, provenance digests, per-attempt model/seed/params, and a Regenerate that busies only its own card",
      "Promote an accepted output as a source for a NEW run — refused re-entry into the run that made it, by digest",
    ],
    group: "studio",
    status: "active",
    component: ScriptFirstStation,
  },
  {
    id: "oracle",
    title: "Oracle",
    blurb:
      "The in-between of generation, made visible: the steward's self-audit of the selector, a performance run's DAG node states with per-node selection decisions, scorecards and repair plans, and an honest spatial-overlays placeholder.",
    phase: "or-k18 / k114 (oracle video)",
    buildsSpec: "— (reads GET/POST /api/oracle/steward, POST /api/oracle/selection, GET /api/oracle/producers, GET /api/oracle/ledger/summary, GET /video/jobs/<id> manifests; enqueues no job)",
    planned: [
      "Steward report: findings by severity, matrix_stale banner, last run time, re-run (POST applies bounded rebalancing)",
      "DAG state list from the run manifest's dag block: node id, state, producer model (receipt → ledger), lease (journal-only, said so)",
      "Per-node drawer: selection decision + candidates by model, scorecard (confidence, disagreements), repair plan",
      "Spatial overlays tab: spatial manifest JSON + evaluator metrics when present, else 'no spatial conditioning'",
    ],
    group: "studio",
    status: "active",
    component: OracleStation,
  },
  {
    id: "active",
    title: "Active",
    blurb:
      "Console-wide Active Processes: every in-flight media build, its phase (incl. the awaiting_capacity GPU hold), and WHERE it physically executes (host · gpu · process) with reserved VRAM and Cancel.",
    phase: "Observability",
    buildsSpec: "— (reads GET /video/jobs; enqueues no job)",
    planned: [
      "Live in-flight jobs across every station/transport (GET /video/jobs, ~2s)",
      "Placement line per job — ae · cuda:0 · P-studio — with an 'external' cue",
      "awaiting_capacity hold phase with the shortfall reason + overtaken count",
      "Reserved VRAM when a GPU reservation is held; honest progress (no fake bars); Cancel",
    ],
    group: "studio",
    status: "active",
    component: ActiveProcessesStation,
  },
] as const;

export function stationById(id: string): StationSpec | undefined {
  return STATIONS.find((s) => s.id === id);
}

/**
 * Default landing station.
 *
 * PINNED, not derived from STATIONS[0]. It used to be `STATIONS[0].id`, which
 * quietly coupled "which page you land on" to "which tab is leftmost" — so the
 * 2026-08-04 tab reorder (crops/frames moved next to Active) would have silently
 * moved the landing page too. Naming it makes the landing a decision rather than
 * a side effect of array order; reorder the strip freely without touching where
 * a bare "/" or an unknown tab resolves.
 *
 * Studio is the landing because it is the workbench people actually start in;
 * the crop/frames utilities are secondary, which is exactly why they moved right.
 */
export const DEFAULT_STATION_ID = "generate";

// The two-tab top nav (Studio / Generate) is retired: every built station now
// shares the single `group:"studio"` workbench, whose ONE `.vi-subtabs` strip
// (rendered by WorkbenchStation) IS the whole tab bar. No SECTIONS model is
// derived here anymore — the shell renders the persistent sidebar + the dynamic
// `/:workbenchTab` route and nothing hardcodes a station list.
