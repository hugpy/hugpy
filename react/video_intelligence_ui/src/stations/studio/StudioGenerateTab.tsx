// The Studio GENERATE SURFACE — ONE source of truth for the studio generate CONTROLS
// (preset tiers + morphing param panel + hero inputs + Generate). It is the TOP of
// the single Studio plane (StudioPlane), which renders it above the inline clip
// viewer + list; that one plane is mounted in both places the studio appears — the
// standalone Studio station (StudioClipsStation) and the Generate station's "studio"
// mode (StudioGenerateMode). It is a pure CONTROLLED component — the source clip + its
// staged mode + the clip list arrive as props, so each mount owns that wiring (and the
// studioBridge registration) while this form stays identical in both. (The file keeps
// its historical name; the exported component is `StudioGenerateSurface`.)
//
// Tier-first: pick a curated preset (grouped REAL tiers vs SYNTHETIC previews), and
// the form MORPHS into that generator's PREFORMED SHAPE instead of showing one
// uniform panel:
//
//   • t2v shape       — image inputs are hidden entirely (text-only note);
//   • i2v "needs a start image" shape — the START IMAGE input becomes the dominant
//     element near the top, with the source-clip picker offered as the "or extend a
//     clip" alternative;
//   • v2v (restyle) shape — the SOURCE CLIP picker gets that hero prominence;
//   • id_lock (identity lock) shape — the REFERENCE-IMAGE slot list (1–4, ordered) is
//     the dominant accent-bordered element, with an OPTIONAL composition-control input
//     (a static pose/depth/sketch anchor) subdued below;
//   • fully custom (no preset) — the current full panel, inputs inline + optional.
//
// Required-vs-optional visual language: a required-and-empty input gets the accent
// attention treatment and the Generate gate names it; optional inputs stay subdued.
//
// Every affordance reuses an EXISTING pattern: presets from useStudioPresets; the
// image UPLOAD path is the Generate station's upload→ingest→guard-kind flow
// (POST /uploads → POST /video/ingest → MediaRef) and the hero drop surface is the
// shared DropReceptacle; the LIBRARY pick reads useMediaLibrary; the source-clip pick
// reads the plane's clip list; "Send to Studio" / "use as source" staging arrives via
// studioBridge as the `source` prop; the enqueue body uses the exact route field names,
// dropped by JSON.stringify when unset.
//
// FIELDS ARE DISTINCT: `start_image` (i2v, exactly 1 — the first frame) and
// `reference_images` (id_lock, 1–4 ORDERED — identity to lock; the hash keys on order)
// are separate route fields with separate slot lists; `control_image`+`control_kind`
// are id_lock-only and both-or-neither.
import { useCallback, useEffect, useRef, useState } from "react";
// Sub-tab input memory — see the block of useSessionState calls in
// StudioGenerateSurface and the rationale in video/sessionForm.ts.
import {
  nextSessionSeq,
  useSessionRef,
  useSessionState,
} from "../../video/sessionForm";
import { createPortal } from "react-dom";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { hugpyConfig, mediaBytesUrl } from "../../config";
import { addToLibrary, useMediaLibrary } from "../../video/mediaLibrary";
import { UnnamedWarnPanel, WarnUnnamedToggle, useWarnUnnamedPref, deriveNameFrom } from "../../video/UnnamedWarn";
import { LengthRow } from "../../video/LengthRow";
import { enqueueResultSchema, mediaRefSchema, uploadResultSchema } from "../../video/contract";
import type { MediaRef } from "../../video/contract";
import { DropReceptacle } from "../../video/DropReceptacle";
import type { GoalComposerAttach, GoalComposerAttached } from "../GoalComposer";
import { usePromptAssist } from "../../video/usePromptAssist";
import { PromptAssistButtons } from "../../video/PromptAssistButtons";
import { getSessionId } from "../../session";
import { isCanned } from "../../demo/mode";
import { DEMO_STUDIO_PROMPT } from "../../demo/seed";
import type { StudioPreset } from "../../video/useStudioPresets";
import { useProjects } from "../../video/useProjects";
import type { StudioMode } from "../../video/studioBridge";
import {
  VACE_W,
  VACE_H,
  VACE_FPS,
  VACE_BUDGET_GB,
  STUDIO_MODEL_IDS,
  STUDIO_REAL_FLOOR_GB,
  STUDIO_SAFE_FILL_GB,
  bindsSyntheticTier,
  isSyntheticPreset,
  shortId,
  clipToMediaRef,
  LibraryImageGrid,
  type Clip,
} from "./studioShared";
// B2: the Movie authoring surface the mode switch renders. Self-contained (owns its
// own goal state, enqueue + poll, and playback) — see StudioMovieComposer.tsx.
import { StudioMovieComposer } from "./StudioMovieComposer";
// IDENTITY PROFILES (stage a): the shared "Identity" control block (From profile /
// Save as identity profile) — one additive row in the reference area below.
import { IdentityProfileControls, type SelectedProfile } from "./IdentityProfileControls";
// The SHARED prompt-card system (k88/k89): the Clip surface's single prompt card
// rides the same shell + [Prompt][Negative] tabs the Scene/Movie/Cinema lists use,
// plus the standard-negative composition helper and the sidebar About expander.
import { AboutExpander, PromptCard, PromptCardSettings, PromptFieldTabs, composeNegative } from "./promptCard";
// ── COMPATIBILITY AWARENESS (operator ruling 2026-07-29) ───────────────────────────
// The fleet's MEASURED render table (GET /video/render/presets) + the pure derivations
// over it. Everything this surface offers — capabilities, model pins, geometry, the
// budget floor, the proven badges — is now filtered by what the fleet can actually
// render and by what the user has already attached, instead of by hand-maintained
// mirrors of backend facts. Courtesy layer only: the route's capability gate stays
// authoritative, and every consumer below falls back to its prior list while
// `renderLoaded` is false, so a slow discovery GET can never empty a picker.
import { useRenderPresets } from "../../video/useRenderPresets";
import {
  CLIP_CAPABILITIES,
  MOVIE_CAPABILITY,
  capabilityOffers,
  capabilityForAttachments,
  capabilityLabel,
  acceptedSlots,
  geometryForCapability,
  withinGeometry,
  modelsForCapability,
  suggestedBudgetGb,
  minBudgetGbForCapability,
  provenForCapability,
  fillIfEmpty,
  type Attachments,
} from "../../video/renderCompat";

// `motion` joined the union on 2026-07-29: it is a SERVABLE capability (video_routes
// widened the control_image gate from id_lock-only on 2026-07-27, making the VACE
// structural-control branch reachable) and this surface already owns the only input it
// needs — the pose/depth/sketch control still. It was simply never offered.
type Capability = "t2v" | "i2v" | "v2v" | "id_lock" | "motion";
type ControlKind = "" | "pose" | "depth" | "sketch";
type UploadPhase = "idle" | "uploading" | "ingesting";
/** Where an ingested image slots (a single upload/library flow feeds all three). */
type ImageTarget = "start" | "reference" | "control";

// i2v `start_image` is a SINGLE first frame (spine takes one). id_lock `reference_images`
// is a DIFFERENT field: 1–4 ordered stills the render locks identity to (order is
// semantic — the content hash keys on it), max 4 enforced by the route.
const MAX_START_IMAGES = 1;
const MAX_REFERENCE_IMAGES = 4;

// Stable per-slot key for the reorderable reference list (order matters, and the same
// still may legitimately appear twice — so key by slot identity, not by uri).
// The counter is SESSION-persisted now that the reference list itself survives a
// reload: a module `let` would restart at 0 and mint `ref_1` a second time,
// colliding with a restored slot.
function refKey(): string {
  return `ref_${nextSessionSeq("studio.clip.refKeySeq")}`;
}

export interface StudioGenerateSurfaceProps {
  presets: StudioPreset[];
  presetsLoading: boolean;
  /** The host's durable clip list — the source-clip picker chooses from it. */
  clips: Clip[];
  /** Host-owned staged/picked source clip (Send to Studio / use-as-source). */
  source: MediaRef | null;
  setSource: (m: MediaRef | null) => void;
  /** Consume mode from the last external stage (drives capability on handoff). */
  sourceMode: StudioMode;
  /** Bumps on each external stage so the capability handoff runs exactly once. */
  stageSeq: number;
  /** Quiet-refresh the clip list after an enqueue — the new job then appears in the
   *  inline clip list below (same plane), no navigation. */
  onEnqueued: () => void;
  /** Round 8: the DOM host for the left-sidebar "Settings" tab. When present, the knob
   *  grid renders THERE (via a studio-local portal) instead of the center — folding the
   *  former inline knob column into the sidebar Settings tab (movie-shape canonical). Null
   *  until the sidebar mounts its Settings panel; the knob state lives here regardless. */
  settingsHost: HTMLElement | null;
  /** When PROVIDED, the top-level Generate tab (Clip / Cinema) owns the clip-vs-movie
   *  choice, so this surface is LOCKED to that value and the internal Clip|Movie switcher
   *  is hidden. When ABSENT (the standalone /studio-clips station mount), the surface keeps
   *  its own internal switcher + useState. "movie" is the Cinema surface (label only — the
   *  wire/tester value stays "movie"). */
  lockedSurfaceMode?: "clip" | "movie";
}

// Labels now come from renderCompat (ONE table both studio surfaces read) so a
// capability can never be named two things in two pickers. Kept as a local alias so the
// existing call sites read unchanged.
const CAP_LABELS: Record<Capability, string> = {
  t2v: capabilityLabel("t2v"),
  i2v: capabilityLabel("i2v"),
  v2v: capabilityLabel("v2v"),
  id_lock: capabilityLabel("id_lock"),
  motion: capabilityLabel("motion"),
};

// A compact source-clip poster that plays inline ON DEMAND — the studio-local twin of
// the Session Library's <RefThumb>, factored so the SELECTED-source summary and every
// picker row share ONE poster/preview renderer (no duplicated <video> markup). The
// still is a muted, metadata-only <video> seeked to an early frame; /video/media is
// Range-aware so only the seeked byte range loads until the viewer presses play.
// Clicking the poster (or a row's Preview button) flips `playing`, swapping in a
// <video controls autoPlay> so the clip is previewable IN PLACE. The poster lives in a
// <button> (it is a non-interactive muted still, so nesting it is valid); the playing
// state is a bare <div> wrapper (a <video controls> must NEVER sit inside a <button>).
// EXPORTED (operator ask 2026-07-12): StudioViewer's "▤ Play from library" panel reuses
// this verbatim for its own poster/id rows, so the viewer's picker and this surface's
// source-clip picker share one poster implementation instead of a fourth copy.
export function SourceThumb({
  uri,
  name,
  playing,
  onToggle,
}: {
  uri: string;
  name: string;
  playing: boolean;
  onToggle: () => void;
}) {
  const poster = `${mediaBytesUrl(uri)}#t=0.1`;
  if (playing) {
    return (
      <div className="vi-studio-src-thumb vi-studio-src-thumb--playing">
        <video
          src={poster}
          controls
          autoPlay
          playsInline
          preload="metadata"
          aria-label={`${name} preview`}
        />
      </div>
    );
  }
  return (
    <button
      type="button"
      className="vi-studio-src-thumb vi-studio-src-thumb-btn"
      onClick={onToggle}
      title={`Preview ${name}`}
      aria-label={`Preview ${name}`}
    >
      <video src={poster} muted playsInline preload="metadata" aria-hidden="true" />
      <span className="vi-studio-src-play" aria-hidden="true">
        ▶
      </span>
    </button>
  );
}

export function StudioGenerateSurface({
  presets,
  presetsLoading,
  clips,
  source,
  setSource,
  sourceMode,
  stageSeq,
  onEnqueued,
  settingsHost,
  lockedSurfaceMode,
}: StudioGenerateSurfaceProps) {
  // ── SUB-TAB INPUT MEMORY (operator ask 2026-08-06) ──────────────────────────
  // Every OPERATOR-AUTHORED field below is `useSessionState` rather than
  // `useState`, so it survives this surface being unmounted — which happens on
  // every Clip↔Cinema switch (the two surfaces swap), every Scene/Movie↔Clip
  // switch (StudioGenerateMode unmounts), and every workbench tab switch
  // (WorkbenchStation renders one station body). Same tuple, same updaters; the
  // store is session-scoped and per-tab. See video/sessionForm.ts for why a store
  // beats keep-mounted-but-hidden here. Transient UI below (pickers, previews,
  // upload phases, busy/error flags, in-flight job ids) deliberately stays on
  // plain useState — restoring a stale "uploading…" would be its own bug.
  //
  // B2 mode switch — Clip (this existing surface, default) | Movie/Cinema (StudioMovieComposer).
  // When the top-level Generate tab locks the choice (Clip / Cinema tabs), `lockedSurfaceMode`
  // is the effective value and the internal switcher below is hidden; the standalone
  // /studio-clips station passes nothing, so its own internal state + switcher drive it.
  const [internalSurfaceMode, setInternalSurfaceMode] = useSessionState<"clip" | "movie">(
    "studio.clip.surfaceMode",
    "clip",
  );
  const surfaceMode = lockedSurfaceMode ?? internalSurfaceMode;

  const [capability, setCapability] = useSessionState<Capability>(
    "studio.clip.capability",
    "i2v",
  );
  const [presetId, setPresetId] = useSessionState("studio.clip.presetId", "");

  const [width, setWidth] = useSessionState("studio.clip.width", 512);
  const [height, setHeight] = useSessionState("studio.clip.height", 512);
  const [fps, setFps] = useSessionState("studio.clip.fps", 24);
  const [seed, setSeed] = useSessionState("studio.clip.seed", 0);
  // Text fields (blank = "unset"): the route/model fills the default. Kept as
  // strings so "empty" is distinct from 0.
  const [vramBudget, setVramBudget] = useSessionState("studio.clip.vramBudget", "");
  // MERGE-NOT-OVERWRITE bookkeeping: true once the OPERATOR has typed in the budget
  // field. A preset/capability change fills a budget the user never touched (it is a
  // workflow slot) but never overwrites a number they chose. One ref, not state — it
  // only ever gates a write, so it must not cause a render of its own. SESSION-scoped
  // for the same reason the values are: the autofit effect below runs on every mount,
  // so a mount that restored a hand-typed budget but a fresh `false` here would
  // immediately overwrite it.
  const budgetTouched = useSessionRef("studio.clip.budgetTouched", false);
  // Same discipline for GEOMETRY. The snap below is a workflow slot and fires on every
  // capability change and once when the measured table lands — which would otherwise
  // overwrite a width the operator typed in the first moments of the page. v2v is
  // deliberately exempt: its inputs are disabled (a hard router-backed lock), so there is
  // nothing there for a user to have touched.
  const geomTouched = useSessionRef("studio.clip.geomTouched", false);
  const [steps, setSteps] = useSessionState("studio.clip.steps", "");
  const [cfg, setCfg] = useSessionState("studio.clip.cfg", "");
  const [modelId, setModelId] = useSessionState("studio.clip.modelId", "");
  // Showroom flavor (?demo=1 only): open with a cinematic example prompt (matches
  // the sunset sample gens) so the brochure's Studio surface tells a coherent story
  // out of the box. Seed lives in src/demo/ and is injected ONLY behind isCanned() —
  // the live path stays an empty prompt.
  const [prompt, setPrompt] = useSessionState("studio.clip.prompt", () =>
    isCanned() ? DEMO_STUDIO_PROMPT : "",
  );
  const [negative, setNegative] = useSessionState("studio.clip.negative", "");
  // k89: the card's [standard negative] tick — default OFF to preserve the Clip
  // surface's blank-negative default; when on, the standard artifact/junk set is
  // COMPOSED into the payload at submit (composeNegative — the wire is unchanged).
  const [stdNegative, setStdNegative] = useSessionState("studio.clip.stdNegative", true);
  // k89: the bottom "[▸ Tester]" collapsible that now hosts the cross-model sweep
  // (the run row's ghost 🧪 button is retired).
  const [clipTesterOpen, setClipTesterOpen] = useState(false);
  // LLM PROMPT-ASSIST (Enhance / Generate) — the same shared hook + button cluster the
  // Generate station uses (POSTs promptAssistUrl). Here the draft is the single clip
  // prompt and apply replaces it wholesale. context.kind = "scene" (a clip is a short
  // motion sequence, so the assistant leans into camera/motion phrasing).
  const assist = usePromptAssist({ kind: "scene", specKey: "studio" });
  // Optional human PROJECT name (choose-or-type). Threaded into the enqueue as
  // `project` (auto-archive NAME); blank = auto-named (the default). The list of
  // existing names backs the Settings datalist; free-typing a new one creates it.
  const [project, setProject] = useSessionState("studio.clip.project", "");
  // EXPLICIT CLIP LENGTH (operator 2026-08-13: "the length of the video
  // generations has been ambiguous") — requested_frames, blank = the bound
  // model's default (81 for real models). The spine clamps to the model
  // ceiling and snaps to the 4k+1 temporal cadence.
  const [requestedFrames, setRequestedFrames] = useSessionState("studio.clip.frames", "");
  // UNNAMED WARNING (operator 2026-08-13): explicit per-section option,
  // default OFF, localStorage-remembered. The clip's NAME is its `project`
  // (the auto-archive name every generate route accepts).
  const [warnUnnamed, setWarnUnnamed] = useWarnUnnamedPref("clip");
  const [nameWarn, setNameWarn] = useState(false);
  const [pendingRun, setPendingRun] = useState(false);
  // Auto-derived project names may be replaced by the next ✨ Generate; a
  // human-typed project is never overwritten.
  const autoProjectRef = useRef("");

  // i2v start image (cap 1) — the `start_image` field.
  const [startImages, setStartImages] = useSessionState<MediaRef[]>(
    "studio.clip.startImages",
    [],
  );
  // id_lock reference images (1–4, ordered) — the `reference_images` field.
  // Only the server MediaRef travels (never pixels), so this is as cheap to keep
  // in session storage as the media library's own entries.
  const [referenceImages, setReferenceImages] = useSessionState<
    Array<{ key: string; ref: MediaRef }>
  >("studio.clip.referenceImages", []);
  // IDENTITY PROFILES (stage a): when the reference set came from a SAVED profile,
  // this holds it — the enqueue then sends `identity_profile:<slug>` (canonical) and
  // omits raw reference_images. ANY ad-hoc edit (upload/library-pick/remove/reorder)
  // clears it back to null: the set is then an "unsaved identity" sent as raw paths.
  const [selectedProfile, setSelectedProfile] = useSessionState<SelectedProfile | null>(
    "studio.clip.selectedProfile",
    null,
  );
  // id_lock optional composition control — `control_image` + `control_kind` (both/neither).
  const [controlImage, setControlImage] = useSessionState<MediaRef | null>(
    "studio.clip.controlImage",
    null,
  );
  const [controlKind, setControlKind] = useSessionState<ControlKind>(
    "studio.clip.controlKind",
    "",
  );

  const [imagePhase, setImagePhase] = useState<UploadPhase>("idle");
  const [imageError, setImageError] = useState<string | null>(null);
  const imageBusy = imagePhase === "uploading" || imagePhase === "ingesting";

  // ── Subject-clip switcher (operator ask 2026-07-12) ──
  // "Generate new clip" is a two-step flow: pendingSubjectStage marks the INTENT
  // (source cleared, capability forced to t2v, note shown) between the click and the
  // next successful enqueue; awaitingSubjectJobId then names the in-flight job whose
  // completion (read off the `clips` poll the viewer/library already share) auto-stages
  // its clip as the new subject. Only one is ever truthy at a time.
  const [pendingSubjectStage, setPendingSubjectStage] = useState(false);
  const [awaitingSubjectJobId, setAwaitingSubjectJobId] = useState<string | null>(null);

  // Advanced escape hatch for the model pin: off by default so the menu is the ratified
  // set, on when an operator deliberately wants every studio model id (labelled
  // unverified for the active capability, never presented as equivalent).
  const [showAllModels, setShowAllModels] = useState(false);

  const [showLibPicker, setShowLibPicker] = useState(false); // start/reference (one active per shape)
  const [showControlLib, setShowControlLib] = useState(false);
  const [showSourcePicker, setShowSourcePicker] = useState(false);
  // Which picker row is previewing inline (job_id), and whether the SELECTED-source
  // summary poster is previewing. Only one renderSource variant mounts at a time (hero
  // is a single value), so these component-level flags are never shared across two
  // visible pickers. Cleared on select/clear so a stale row never keeps auto-playing.
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [sourcePreview, setSourcePreview] = useState(false);

  const [genBusy, setGenBusy] = useState(false);
  const [genMsg, setGenMsg] = useState<string | null>(null);

  // STUDIO TESTER (operator 2026-08-05) — fan the CURRENT prompt across every
  // servable model of this surface's type (clip/movie) via POST studioTesterUrl.
  // Operator-gated + background job; per-model results stream to the generation
  // log directly below this surface (StudioAssistLogPanel). One prompt in, a
  // battery out — no per-model clicking.
  const [testerBusy, setTesterBusy] = useState(false);
  // The FILTERED sweep (operator 2026-08-05): each surface computes the model
  // subset that fits its current capability + geometry (see clipTesterModels /
  // movieTesterModels below) and passes it as `models`. An EMPTY subset means the
  // measured render table has not landed yet (or the current geometry sits outside
  // every ratified envelope) — we omit `models` so the backend sweeps the whole
  // type roster, the same courtesy-layer fallback every picker here uses when
  // `renderLoaded` is false. A dead button would be a worse failure than a wider
  // sweep.
  const onTestAllModels = useCallback(
    async (opts: {
      category: string;
      models: string[];
      width?: number;
      height?: number;
      fps?: number;
      seed?: number;
      startImage?: string;
      steps?: number;
      cfg?: number;
      requestedFrames?: number;
      negative?: string;
    }) => {
      const p = prompt.trim();
      if (!p) {
        setGenMsg("Enter a prompt to test across every model of this type.");
        return;
      }
      setTesterBusy(true);
      setGenMsg(null);
      try {
        const body: Record<string, unknown> = { category: opts.category, prompt: p };
        if (opts.width != null) body.width = opts.width;
        if (opts.height != null) body.height = opts.height;
        if (opts.fps != null) body.fps = opts.fps;
        if (opts.seed != null) body.seed = opts.seed;
        if (opts.startImage) body.start_image = opts.startImage;
        if (opts.steps != null) body.steps = opts.steps;
        if (opts.cfg != null) body.cfg = opts.cfg;
        if (opts.requestedFrames != null) body.requested_frames = opts.requestedFrames;
        if (opts.negative) body.negative = opts.negative;
        if (opts.models.length > 0) body.models = opts.models;
        const res = await request<unknown>(hugpyConfig.studioTesterUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          meta: { specKey: "studio", operation: "studio.tester.enqueue" },
        });
        if (!mounted.current) return;
        if (!res.ok) {
          setGenMsg(`Tester failed: ${describeAppError(errorOf(res))}`);
          return;
        }
        setGenMsg(
          opts.models.length > 0
            ? `Testing this prompt across ${opts.models.length} ${opts.category} ` +
                `model${opts.models.length === 1 ? "" : "s"} that fit the current filter — ` +
                `each result appears in the log below as it finishes.`
            : `Testing this prompt across every ${opts.category} model — each model's ` +
                `result appears in the log below as it finishes.`,
        );
      } finally {
        if (mounted.current) setTesterBusy(false);
      }
    },
    [prompt],
  );

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const library = useMediaLibrary();
  const libraryImages = library.filter((it) => it.ref.kind === "image");
  // Existing project names for the Settings "Project" combobox (choose-or-type).
  const { projects: knownProjects, refresh: refreshProjects } = useProjects(true);

  const preset = presets.find((p) => p.id === presetId) ?? null;
  const isT2V = capability === "t2v";
  const isI2V = capability === "i2v";
  const isV2V = capability === "v2v";
  const isIdLock = capability === "id_lock";
  const isMotion = capability === "motion";

  // ── the fleet's measured render table + everything derived from it ───────────────
  // One session-cached GET shared with the Movie composer (module singleton in the
  // hook). `renderLoaded` gates EVERY filter below: while it is false each picker shows
  // its full prior list, so a slow or failed discovery call degrades to exactly the
  // behavior this surface had before, never to an empty menu.
  const {
    presets: renderPresets,
    unavailable: renderUnavailable,
    loaded: renderLoaded,
    renderBox,
  } = useRenderPresets(true);

  // What the user has attached RIGHT NOW — the input side of the compatibility matrix.
  // Every picker below narrows against this, which is the operator's "every input the
  // user makes filters the options subsequently offered".
  const attached: Attachments = {
    startImage: startImages.length > 0,
    sourceVideo: !!source,
    referenceImages: referenceImages.length > 0,
    controlImage: !!controlImage,
  };

  // The capability menu, as data: servable-on-this-fleet AND compatible-with-what-is-
  // attached, each unavailable row carrying the backend's OWN refusal wording for its
  // tooltip. Not a gate — an incompatible capability renders DISABLED with the reason,
  // because a user who attached a start image and cannot find restyle is worse off than
  // one who can see why it is greyed out.
  const capOffers = capabilityOffers(
    renderPresets,
    renderUnavailable,
    attached,
    CLIP_CAPABILITIES,
  );
  const offerFor = (c: string) => capOffers.find((o) => o.capability === c);

  // The ratified geometry envelope for the active capability (832x480@16 on every video
  // row today). Drives the snap-on-pick below and the out-of-envelope flag — the same
  // footgun guard v2v always had, now derived for every capability instead of hardcoded
  // for one.
  const capGeometry = renderLoaded ? geometryForCapability(renderPresets, capability) : null;
  const geomOutOfEnvelope =
    renderLoaded && !withinGeometry(renderPresets, capability, width, height);

  // The MINIMUM budget that can bind a real model for this capability, read off the
  // preset table's `vram_envelope_gb` (the exact number the router compares a budget
  // against). Replaces the hand-mirrored STUDIO_REAL_FLOOR_GB, which had drifted to 6 —
  // below every real row in the table (cheapest is 8.2).
  const capFloorGb = renderLoaded ? minBudgetGbForCapability(renderPresets, capability) : null;
  const capSuggestGb = renderLoaded ? suggestedBudgetGb(renderPresets, capability) : null;

  // Model pins that stay inside the ratified set for this capability. The escape hatch
  // below keeps every studio model id reachable — an operator may still pin off-menu,
  // and the option says so rather than pretending the pin is verified.
  const capModels = renderLoaded ? modelsForCapability(renderPresets, capability) : [];
  const modelOffMenu =
    renderLoaded && modelId.trim() !== "" && !capModels.includes(modelId.trim());

  // STUDIO TESTER rosters (operator 2026-08-05) — the model subset each surface's
  // "Test all models" sweep is scoped to, so a battery only ever runs the models
  // that fit the CURRENT FILTER, not every servable model blindly.
  //   CLIP: the ratified models for the ACTIVE capability, kept only while the
  //   current width×height is inside that capability's envelope. Geometry is a
  //   PER-CAPABILITY envelope in the render table (not per-model), so this is
  //   correctly all-or-nothing — out of envelope ⇒ [] ⇒ the button falls back to
  //   the whole roster (see onTestAllModels).
  //   CINEMA: the movie/assemble capability's ratified models, which via the
  //   preset's `composes` already fold in the per-segment t2v/i2v bindings the
  //   movie actually renders.
  const clipTesterModels =
    renderLoaded && withinGeometry(renderPresets, capability, width, height)
      ? capModels
      : [];

  // Has this capability ever produced pixels here? Badged, never hidden — `false` is the
  // table telling the truth about itself.
  const capProven = renderLoaded ? provenForCapability(renderPresets, capability) : null;

  // Which input slots the ACTIVE capability accepts, from the route's own gates.
  const slots = acceptedSlots(capability);

  // Geometry tooltips, named rather than anonymous: a lock the user cannot explain is a
  // lock they will fight. Both quote the RATIFIED preset the envelope came from.
  const geomLockTitle = capGeometry
    ? `Restyle is locked to ${capGeometry.width}×${capGeometry.height} @ ${capGeometry.fps}fps — the envelope “${capGeometry.presetTitle}” is ratified at. The router refuses anything outside it.`
    : `Restyle is locked to ${VACE_W}×${VACE_H} @ ${VACE_FPS}fps (the VACE envelope).`;
  const geomEnvelopeTitle = capGeometry
    ? `Ratified envelope for this capability: ${capGeometry.width}×${capGeometry.height} @ ${capGeometry.fps}fps (“${capGeometry.presetTitle}”). Outside it there is no proven binding.`
    : undefined;

  const addStartImage = useCallback((ref: MediaRef) => {
    setStartImages((prev) => (prev.length >= MAX_START_IMAGES ? prev : [...prev, ref]));
    setShowLibPicker(false);
  }, []);
  const removeStartImage = useCallback((idx: number) => {
    setStartImages((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const addReferenceImage = useCallback((ref: MediaRef) => {
    setReferenceImages((prev) =>
      prev.length >= MAX_REFERENCE_IMAGES ? prev : [...prev, { key: refKey(), ref }],
    );
    setShowLibPicker(false);
    setSelectedProfile(null); // an ad-hoc add makes this an UNSAVED identity
  }, []);
  const removeReferenceImage = useCallback((idx: number) => {
    setReferenceImages((prev) => prev.filter((_, i) => i !== idx));
    setSelectedProfile(null);
  }, []);
  const moveReferenceImage = useCallback((idx: number, dir: -1 | 1) => {
    setReferenceImages((prev) => {
      const j = idx + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = prev.slice();
      const [it] = next.splice(idx, 1);
      next.splice(j, 0, it);
      return next;
    });
    setSelectedProfile(null); // reorder diverges from the profile's canonical order
  }, []);

  // IDENTITY PROFILES: attach a profile's resolved reference set. A profile pick
  // REPLACES the current set (a profile IS the whole identity); ad-hoc callers append.
  const attachIdentityRefs = useCallback((refs: MediaRef[], replace: boolean) => {
    if (replace) {
      setReferenceImages(refs.slice(0, MAX_REFERENCE_IMAGES).map((ref) => ({ key: refKey(), ref })));
    } else {
      setReferenceImages((prev) => {
        const room = MAX_REFERENCE_IMAGES - prev.length;
        return room <= 0 ? prev : [...prev, ...refs.slice(0, room).map((ref) => ({ key: refKey(), ref }))];
      });
    }
    setShowLibPicker(false);
  }, []);

  // Capability handoff: when the shell stages a source (Send to Studio / use as
  // source), adopt its mode — restyle ⇒ v2v, extend ⇒ i2v. Keyed on stageSeq so it
  // runs once per stage, never clobbering a manual capability choice.
  //
  // CROSS-TIER CARRY: an "extend" stage normally lands in i2v, but if the surface is
  // ALREADY holding reference images, land in v2v scene-continuity instead — v2v shows
  // AND sends the carried refs, so chaining scene→scene keeps the identity lock without
  // re-picking (i2v would silently drop the refs). The reference slots themselves are
  // never cleared by a stage, so the lock rides along.
  //
  // 2026-07-29: the landing capability is now VALIDATED against the compatibility
  // matrix instead of assumed. The intent ordering is unchanged (an explicit "restyle"
  // stage means v2v; carried references keep the identity lock alive rather than letting
  // i2v silently drop them), but each candidate is checked with `capabilityForAttachments`
  // so a stage can never land the form on a capability that refuses what is attached.
  useEffect(() => {
    if (stageSeq === 0) return;
    const next: Attachments = {
      startImage: startImages.length > 0,
      sourceVideo: true, // a stage IS a staged source clip
      referenceImages: referenceImages.length > 0,
      controlImage: !!controlImage,
    };
    const intent: Capability =
      sourceMode === "restyle" || referenceImages.length > 0 ? "v2v" : "i2v";
    const resolved = capabilityForAttachments(renderPresets, next, intent);
    setCapability((resolved as Capability) ?? intent);
    setPresetId("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stageSeq]);

  // FOOTGUN GUARD, GENERALIZED (2026-07-29). This used to be a v2v-only hardcode of the
  // 832x480@16 VACE envelope. It is now derived: whenever the capability changes, the
  // geometry snaps to the envelope the RATIFIED preset for that capability declares, and
  // the budget floor comes from that preset's `vram_envelope_gb`. Every video row on the
  // fleet is 832x480@16 today, so in practice this makes t2v/i2v/id_lock/motion land on
  // the same guaranteed-to-render envelope v2v was already pinned to — which is also the
  // route's own default (video_routes: a 512x512 default "was a GUARANTEED FAIL for two
  // whole capabilities").
  //
  // The v2v HARD LOCK below (disabled inputs + the submit block) is unchanged: the router
  // refuses an out-of-envelope restyle outright, so that one stays a lock rather than a
  // snap. Other capabilities snap once and remain editable, with an honest flag if the
  // operator moves outside the envelope.
  //
  // MERGE DISCIPLINE: the budget is filled only when the operator has not typed one.
  useEffect(() => {
    if (isV2V) {
      setWidth(capGeometry?.width ?? VACE_W);
      setHeight(capGeometry?.height ?? VACE_H);
      setFps(capGeometry?.fps ?? VACE_FPS);
      if (!budgetTouched.current && vramBudget.trim() === "") {
        setVramBudget(String(capSuggestGb ?? VACE_BUDGET_GB));
      }
      return;
    }
    if (capGeometry && !geomTouched.current) {
      setWidth(capGeometry.width);
      setHeight(capGeometry.height);
      setFps(capGeometry.fps);
    }
    if (!budgetTouched.current && vramBudget.trim() === "" && capSuggestGb != null) {
      setVramBudget(String(capSuggestGb));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capability, capGeometry?.presetId, capSuggestGb]);

  // ── APPLYING A PRESET IS A MERGE, NEVER AN OVERWRITE (operator ruling 2026-07-29) ──
  //
  // A preset supplies the WORKFLOW — capability, geometry, cadence, budget. The user's
  // SESSION INPUTS are theirs: prompt, negative, identity, every attachment, and a seed
  // they chose all survive the apply and are slotted into the workflow around them.
  //
  // WHAT THIS REPLACES, precisely, because it was a data-loss bug and not a preference:
  // the previous body ended `setPrompt(p.prompt ?? ""); setNegative(p.negative ?? "")`.
  // Most studio presets carry no prompt of their own, so picking one BLANKED whatever
  // the operator had just written (and any prompt-assist enhancement they had run on
  // it). It also stamped the preset's seed and budget over hand-typed values. The
  // Generate station's `applyPresetLoad` had the right instinct all along — fill only
  // what the preset actually names — and this is that rule, tightened for text to
  // fill-only-when-empty (`fillIfEmpty`).
  const onPickPreset = useCallback(
    (id: string) => {
      setPresetId(id);
      const p = presets.find((x) => x.id === id) ?? null;
      if (!p) return;
      const cap = (p.capability as Capability) ?? "i2v";
      setCapability(cap);
      // A manual preset pick abandons any armed/in-flight "generate new subject"
      // wait (see the subject-clip switcher below) — the operator has taken over.
      setPendingSubjectStage(false);
      setAwaitingSubjectJobId(null);
      // WORKFLOW SLOTS — the preset owns these. Geometry prefers the RATIFIED envelope
      // for the capability over the curated preset's own numbers when the two disagree,
      // because the ratified table is the one measured against the render box; the
      // curated preset's geometry is intent.
      const geom = renderLoaded ? geometryForCapability(renderPresets, cap) : null;
      // An explicit preset pick IS a request for that workflow's envelope, so it hands
      // geometry back to the presets (a later hand-edit takes it again).
      geomTouched.current = false;
      setWidth(geom?.width ?? p.width);
      setHeight(geom?.height ?? p.height);
      setFps(geom?.fps ?? p.fps);
      // SESSION INPUTS — preserved. A seed the operator moved off 0 is a choice; a seed
      // still at 0 is untouched scaffolding a preset may fill.
      setSeed((cur) => (cur !== 0 ? cur : typeof p.seed === "number" ? p.seed : cur));
      if (!budgetTouched.current) {
        const floor = renderLoaded ? suggestedBudgetGb(renderPresets, cap) : null;
        // Never offer a budget under the capability's real floor — a preset curated
        // before the floor was measured would otherwise silently bind the synthetic tier.
        const wanted =
          typeof p.vram_budget_gb === "number"
            ? floor != null
              ? Math.max(p.vram_budget_gb, floor)
              : p.vram_budget_gb
            : floor;
        if (wanted != null) setVramBudget(String(wanted));
      }
      // The preset's example prompt is a scaffold for an EMPTY box, never a replacement.
      setPrompt((cur) => fillIfEmpty(cur, p.prompt));
      setNegative((cur) => fillIfEmpty(cur, p.negative));
    },
    [presets, renderPresets, renderLoaded],
  );

  // Manual capability change drops any picked preset (so it can't fight the choice); the
  // geometry/budget snap for the new capability runs in the derived effect above.
  //
  // 2026-07-29: it now also RE-VALIDATES the user's other inputs against the new
  // capability instead of leaving them to fail at render time. Nothing is deleted — the
  // attachments are the user's — but a model pin that cannot serve the new capability is
  // cleared back to Auto (the router then picks a model that CAN), and an incompatible
  // attachment is named in the callout so the operator can act on it.
  const onPickCapability = useCallback(
    (c: Capability) => {
      setCapability(c);
      setPresetId("");
      // Same abandon-on-manual-override as onPickPreset above.
      setPendingSubjectStage(false);
      setAwaitingSubjectJobId(null);
      if (renderLoaded && modelId.trim() !== "") {
        const ok = modelsForCapability(renderPresets, c);
        if (ok.length > 0 && !ok.includes(modelId.trim())) setModelId("");
      }
    },
    [renderPresets, renderLoaded, modelId],
  );

  // ── image upload → ingest → guard kind=image, then route the MediaRef to its slot
  //    (start | reference | control). One flow for all three; joins the Session Library. ──
  const onPickImageFile = useCallback(
    async (file: File, target: ImageTarget) => {
      setImageError(null);
      setImagePhase("uploading");
      const fd = new FormData();
      fd.append("file", file);
      fd.append("sid", getSessionId());
      const up = await request<unknown>(hugpyConfig.uploadUrl, {
        method: "POST",
        body: fd,
        meta: { specKey: "studio", operation: "upload" },
      });
      if (!mounted.current) return;
      if (!up.ok) {
        setImageError(describeAppError(errorOf(up)));
        setImagePhase("idle");
        return;
      }
      const upParsed = uploadResultSchema.safeParse(okValue(up));
      if (!upParsed.success) {
        setImageError("Malformed upload response.");
        setImagePhase("idle");
        return;
      }
      setImagePhase("ingesting");
      const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
        method: "POST",
        body: JSON.stringify({ path: upParsed.data.path }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "ingest" },
      });
      if (!mounted.current) return;
      if (!ing.ok) {
        setImageError(describeAppError(errorOf(ing)));
        setImagePhase("idle");
        return;
      }
      const media = mediaRefSchema.safeParse(okValue(ing));
      if (!media.success) {
        setImageError("Malformed ingest response.");
        setImagePhase("idle");
        return;
      }
      if (media.data.kind !== "image") {
        setImageError(`Ingested asset is "${media.data.kind}", not an image — pick a still.`);
        setImagePhase("idle");
        return;
      }
      addToLibrary(media.data, "upload", media.data.mime);
      if (target === "start") addStartImage(media.data);
      else if (target === "reference") addReferenceImage(media.data);
      else setControlImage(media.data);
      setImagePhase("idle");
    },
    [addStartImage, addReferenceImage],
  );

  // ── video upload → ingest → guard kind=video → straight to the staged subject. ──
  // The subject-clip switcher's "Upload video" path (operator ask 2026-07-12): the
  // EXACT mirror of onPickImageFile above (same upload/ingest endpoints, same MediaRef
  // shape) but guarding the OTHER kind this surface consumes, landing straight in
  // `source` instead of a slot list. Shares imagePhase/imageError with the image
  // uploads — one in-flight-upload state machine for every upload this surface offers,
  // so an image upload and a video upload never race each other.
  const onPickVideoFile = useCallback(
    async (file: File) => {
      setImageError(null);
      setImagePhase("uploading");
      const fd = new FormData();
      fd.append("file", file);
      fd.append("sid", getSessionId());
      const up = await request<unknown>(hugpyConfig.uploadUrl, {
        method: "POST",
        body: fd,
        meta: { specKey: "studio", operation: "studio.subject.upload" },
      });
      if (!mounted.current) return;
      if (!up.ok) {
        setImageError(describeAppError(errorOf(up)));
        setImagePhase("idle");
        return;
      }
      const upParsed = uploadResultSchema.safeParse(okValue(up));
      if (!upParsed.success) {
        setImageError("Malformed upload response.");
        setImagePhase("idle");
        return;
      }
      setImagePhase("ingesting");
      const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
        method: "POST",
        body: JSON.stringify({ path: upParsed.data.path }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "studio.subject.ingest" },
      });
      if (!mounted.current) return;
      if (!ing.ok) {
        setImageError(describeAppError(errorOf(ing)));
        setImagePhase("idle");
        return;
      }
      const media = mediaRefSchema.safeParse(okValue(ing));
      if (!media.success) {
        setImageError("Malformed ingest response.");
        setImagePhase("idle");
        return;
      }
      if (media.data.kind !== "video") {
        setImageError(`Ingested asset is "${media.data.kind}", not a video — pick a clip.`);
        setImagePhase("idle");
        return;
      }
      addToLibrary(media.data, "upload", media.data.mime);
      setSource(media.data);
      setSourcePreview(false);
      setShowSourcePicker(false);
      setPendingSubjectStage(false);
      setAwaitingSubjectJobId(null);
      setImagePhase("idle");
    },
    [setSource],
  );

  // ── "generate new clip" — the switcher's third swap path. Clears the staged
  // subject, forces the no-source t2v shape (renderSource itself stops rendering in
  // t2v, so the always-visible note below carries the "what's happening" signal in
  // its place), and arms pendingSubjectStage so the NEXT successful enqueue starts the
  // auto-stage wait below. ──
  const onGenerateNewSubject = useCallback(() => {
    setSource(null);
    setSourcePreview(false);
    setShowSourcePicker(false);
    setAwaitingSubjectJobId(null);
    setCapability("t2v");
    setPresetId("");
    setPendingSubjectStage(true);
    setGenMsg(null);
  }, [setSource]);

  // Watch the shared clip list for the job armed by onGenerateNewSubject (via
  // onGenerate below). `clips` is the SAME 6s-polled list the viewer/library read, so
  // this rides that poll instead of opening a second one. Once the awaited job shows
  // up playable, its clip becomes the staged subject automatically; a failed/cancelled
  // job stops the wait with an honest message instead of hanging forever.
  useEffect(() => {
    if (!awaitingSubjectJobId) return;
    const match = clips.find((c) => c.job_id === awaitingSubjectJobId);
    if (!match) return; // not in the list yet — keep waiting for the next poll tick
    if (match.playable) {
      const ref = clipToMediaRef(match);
      if (ref) {
        setSource(ref);
        setSourcePreview(false);
        setGenMsg("The new clip is done — staged as the subject.");
      }
      setAwaitingSubjectJobId(null);
      return;
    }
    const st = match.status ?? "";
    if (st === "failed" || st === "cancelled") {
      setGenMsg(`The subject render ${st} — add a subject another way, or try Generate again.`);
      setAwaitingSubjectJobId(null);
    }
  }, [clips, awaitingSubjectJobId, setSource]);

  // ── required-input gate + preformed SHAPE ──
  // v2v: a source clip is REQUIRED (nothing to repaint otherwise) — always the hero.
  // i2v with a requires_source preset: a START IMAGE or a SOURCE CLIP (extend) is
  //   required — the start image is the hero, the source clip the alternative.
  // id_lock: >=1 REFERENCE image is required (the route 400s otherwise) — reference
  //   slots are the hero; the composition control is optional + both-or-neither.
  // t2v / synthetic i2v / custom: no required input; inputs stay inline + optional.
  const presetNeedsInput = preset?.requires_source === true;
  const hasImage = startImages.length > 0;
  const hasSource = !!source;
  const hasReference = referenceImages.length > 0;
  const canAddImage = startImages.length < MAX_START_IMAGES;
  const canAddReference = referenceImages.length < MAX_REFERENCE_IMAGES;
  const needSource = isV2V;
  const needImageOrSource = isI2V && presetNeedsInput;
  const needReference = isIdLock;
  // MOTION requires a control still and it IS the capability — the exact mirror of
  // id_lock's reference rule (video_routes: "a motion request with no control carries NO
  // conditioning this runner can tell apart from a plain t2v/v2v render").
  const needControl = isMotion;
  const hasControl = !!controlImage;
  // both-or-neither: image XOR kind set is invalid. Applies to BOTH capabilities that
  // accept a control (id_lock optional, motion required).
  const controlHalfSet = (isIdLock || isMotion) && !!controlImage !== (controlKind !== "");
  // The route refuses this pair by name: the VACE control channel takes ONE input and
  // source_video wins the elif, so the control still would be silently dropped and the
  // render would come back a plain restyle. Block it here rather than let the 400 land.
  const controlAndSource = isMotion && hasControl && hasSource;
  const missingRequired =
    (needSource && !hasSource) ||
    (needImageOrSource && !hasImage && !hasSource) ||
    (needReference && !hasReference) ||
    (needControl && !hasControl) ||
    controlAndSource ||
    controlHalfSet;

  const hero: "start-image" | "source" | "reference" | "control" | null = isMotion
    ? "control"
    : isIdLock
      ? "reference"
      : isV2V
        ? "source"
        : needImageOrSource
          ? "start-image"
          : null;
  const startImageAttention = needImageOrSource && !hasImage && !hasSource;
  const sourceAttention =
    (needSource && !hasSource) || (needImageOrSource && !hasImage && !hasSource);
  const referenceAttention = needReference && !hasReference;

  // ── Composer attach idiom (operator ask 2026-07-11, option A) ──
  // The prompt composer's inline attach toolbar + chips and the hero shape editor below
  // are TWO ACCESS PATHS TO THE SAME STATE — one source of truth, two views. We adopt
  // GoalComposer's documented attach idiom (attachments = toolbar buttons, attached =
  // removable chips) and wire its callbacks to THIS file's OWN pickers / MediaRef flows,
  // offering only the inputs the ACTIVE shape accepts (mirroring the render* predicates
  // above). The buttons open the SAME `showLibPicker` / `showSourcePicker` the hero uses,
  // so the library pick routes to start (i2v) or reference (v2v / id_lock) exactly as the
  // hero's own lib-picker routing does; the chips reflect the SAME startImages / source /
  // referenceImages / controlImage state the hero renders, and reuse the hero's existing
  // removes. Applicability decides whether a button EXISTS (t2v gets no toolbar — omitted
  // cleanly by GoalComposer when the array is empty); capacity decides whether it is
  // DISABLED (a filled slot mirrors the hero's "you're at the cap" note, not a vanished
  // affordance). The negative composer stays plain.
  //
  // 2026-07-29: these three are no longer hand-written capability unions — they are the
  // route's own input gates, transcribed once in renderCompat's ACCEPTS matrix and read
  // here via `acceptedSlots`. Same values for the four original capabilities; `motion`
  // gets its control slot for free, and a future gate change is a one-line edit in the
  // module that mirrors the route rather than a hunt through two components.
  const acceptsStartImage = slots.startImage; // i2v carries the subject from a start still
  const acceptsSourceClip = slots.sourceVideo; // i2v extend / v2v restyle take a source clip
  const acceptsReferenceImage = slots.referenceImages; // v2v continuity / id_lock lock

  const promptAttachments: GoalComposerAttach[] = [];
  // i2v ⇒ start image; v2v / id_lock ⇒ reference image (mutually exclusive — never both).
  if (acceptsStartImage) {
    promptAttachments.push({
      label: "＋ image from library",
      onClick: () => setShowLibPicker(true),
      disabled: !canAddImage,
      title: canAddImage
        ? "Pick a start image from the session library"
        : "A start image is already added — remove it to pick another.",
    });
  } else if (acceptsReferenceImage) {
    promptAttachments.push({
      label: "＋ reference image",
      onClick: () => setShowLibPicker(true),
      disabled: !canAddReference,
      title: canAddReference
        ? "Pick a reference image from the session library"
        : `Maximum ${MAX_REFERENCE_IMAGES} reference images.`,
    });
  }
  if (acceptsSourceClip) {
    promptAttachments.push({
      label: "＋ video",
      onClick: () => setShowSourcePicker(true),
      title: isV2V ? "Pick the source clip to restyle" : "Pick a source clip to extend",
    });
  }

  // Chips mirror the picked state the ACTIVE shape shows, with the hero's own removes.
  const promptAttached: GoalComposerAttached[] = [];
  if (acceptsStartImage) {
    startImages.forEach((img, idx) =>
      promptAttached.push({
        ref: img,
        label: "start image",
        onRemove: () => removeStartImage(idx),
      }),
    );
  }
  if (acceptsSourceClip && source) {
    promptAttached.push({
      ref: source,
      label: source.asset_id.slice(0, 8),
      // Same clear the hero's Clear button performs (drop the ref + stop any preview).
      onRemove: () => {
        setSource(null);
        setSourcePreview(false);
      },
    });
  }
  if (acceptsReferenceImage) {
    referenceImages.forEach((slot, idx) =>
      promptAttached.push({
        ref: slot.ref,
        label: `ref ${idx + 1}`,
        onRemove: () => removeReferenceImage(idx),
      }),
    );
  }
  if (slots.controlImage && controlImage) {
    promptAttached.push({
      ref: controlImage,
      // Mirrors the hero's control Remove (setControlImage(null)); kind is left as-is,
      // exactly like that button — the both-or-neither gate then flags the half-set state.
      label: controlKind ? `control · ${controlKind}` : "control",
      onRemove: () => setControlImage(null),
    });
  }

  // Preset-aware honesty copy for the id_lock reference hero (lock-person / lock-thing /
  // scene-bias templates each want different guidance; the generic id_lock keeps a
  // neutral line). Shown as the accent callout while the required refs are empty.
  const referenceCallout =
    presetId === "lock-person"
      ? "Add 1–4 photos of the SAME person; multiple angles (front, 3/4, profile) help hold the face and build across motion. Order matters."
      : presetId === "lock-thing"
        ? "Add up to 4 clean shots of the object (a plain background and a couple of angles help). Its shape, markings and colour are preserved. Order matters."
        : presetId === "scene-bias"
          ? "Add a photo of the place. Honest: a place BIASES the setting, lighting and palette — it does not hard-lock like a person or object does."
          : `This generator locks identity to these reference images — add 1–${MAX_REFERENCE_IMAGES} (a person, place, or thing). Order matters.`;

  const missingReason =
    needReference && !hasReference
      ? "Identity lock needs at least one reference image (1–4) — add one above."
      : needControl && !hasControl
        ? "Pose / structure control needs a control still (pose, depth or sketch) — it IS the control that makes this a motion render rather than a plain restyle."
        : controlAndSource
          ? "A control still and a source clip cannot be combined — the control channel takes one input and the source clip wins, so the still would be silently ignored. Clear one (control still ⇒ motion, source clip ⇒ restyle)."
          : controlHalfSet
            ? "Composition control is both-or-neither — add a control image AND a kind, or clear both."
            : needSource && !hasSource
              ? "Restyle (v2v) needs a source clip — add one above."
              : "This generator needs a start image (or a source clip to extend) — add one above.";

  // v2v stays a HARD block (the router refuses an out-of-envelope restyle outright). The
  // envelope itself is now the ratified preset's rather than a mirrored constant.
  const v2vW = capGeometry?.width ?? VACE_W;
  const v2vH = capGeometry?.height ?? VACE_H;
  const v2vBadGeom =
    isV2V && (width > v2vW || height > v2vH || width <= 0 || height <= 0);
  const blockSubmit = missingRequired || v2vBadGeom;

  // ── SYNTHETIC-TIER HONESTY BANNER (operator ask 2026-07-12) ──
  // "Leaving VRAM budget blank silently binds the synthetic (colored-noise) tier"
  // bit the operator repeatedly — the footnote on the knob below was never enough.
  // Only t2v and i2v have a synthetic last-resort model registered at all
  // (studio/models_seed.py: synthetic-t2v / synthetic-i2v are the only synthetic
  // ModelConfigs) — v2v and id_lock have NO synthetic candidate, so a too-low
  // budget there fails the router honestly instead of rendering fake frames. That
  // structurally excludes both: isV2V is caught by `!(isT2V || isI2V)` directly,
  // and separately v2v's own footgun-guard effect above already floors a blank
  // budget to VACE_BUDGET_GB the moment restyle is picked — so this predicate
  // mirrors the surface's OWN "would this bind sub-real" logic rather than
  // re-deriving a parallel one.
  //
  // ⚠ THE FLOOR IS NOW MEASURED, NOT MIRRORED (2026-07-29). `STUDIO_REAL_FLOOR_GB` was a
  // hand-kept copy of a backend constant and it had drifted to 6 — UNDER the cheapest
  // real row in the ratified table (wan2.1-t2v-1.3b @fp16 is 8.2). A budget of 7 would
  // therefore bind the synthetic prover while this banner stayed silent. The floor now
  // comes from the capability's own `vram_envelope_gb`, with the old constant kept only
  // as the fallback for a page whose discovery GET has not landed.
  const realFloorGb = capFloorGb ?? STUDIO_REAL_FLOOR_GB;
  const safeFillGb = capSuggestGb ?? STUDIO_SAFE_FILL_GB;
  const wouldBindSynthetic =
    (isT2V || isI2V) && bindsSyntheticTier(vramBudget, realFloorGb);

  // Post-commit re-run after the warn panel names the clip: state has settled,
  // this effect's closure is fresh, so the submit sees the new project name.
  useEffect(() => {
    if (!pendingRun) return;
    setPendingRun(false);
    void onGenerate(true);
  });

  const onGenerate = useCallback(async (skipNameCheck = false) => {
    if (!skipNameCheck && warnUnnamed && project.trim() === "") {
      setNameWarn(true);
      return;
    }
    setNameWarn(false);
    if (blockSubmit) {
      setGenMsg(
        v2vBadGeom
          ? "Restyle geometry must be 832×480 landscape (the VACE envelope)."
          : missingReason,
      );
      return;
    }
    const budgetNum = vramBudget.trim() === "" ? undefined : Number(vramBudget);
    if (budgetNum !== undefined && (Number.isNaN(budgetNum) || budgetNum <= 0)) {
      setGenMsg("VRAM budget must be a positive number (or blank for the route default).");
      return;
    }
    const framesNum = requestedFrames.trim() === "" ? undefined : Number(requestedFrames);
    if (framesNum !== undefined && (!Number.isInteger(framesNum) || framesNum < 1)) {
      setGenMsg("Length must be a whole number of frames \u2265 1 (or blank for the model default).");
      return;
    }
    const stepsNum = steps.trim() === "" ? undefined : Number(steps);
    if (
      stepsNum !== undefined &&
      (!Number.isInteger(stepsNum) || stepsNum < 1 || stepsNum > 100)
    ) {
      setGenMsg("Steps must be a whole number in [1, 100] (or blank for the model default).");
      return;
    }
    const cfgNum = cfg.trim() === "" ? undefined : Number(cfg);
    if (cfgNum !== undefined && (Number.isNaN(cfgNum) || cfgNum < 0 || cfgNum > 20)) {
      setGenMsg("CFG must be a number in [0, 20] (or blank for the model default).");
      return;
    }

    setGenBusy(true);
    setGenMsg(null);
    try {
      // The route accepts control_image on id_lock (optional composition blocking) AND
      // on motion (required, definitional) — `slots.controlImage` is that gate.
      const controlReady = (isIdLock || isMotion) && !!controlImage && controlKind !== "";
      const body: Record<string, unknown> = {
        capability,
        resolution: { width, height, fps },
        seed,
        vram_budget_gb: budgetNum,
        steps: stepsNum,
        cfg: cfgNum,
        requested_frames: requestedFrames.trim() !== "" ? Number(requestedFrames) : undefined,
        model_id: modelId.trim() ? modelId.trim() : undefined,
        // Optional auto-archive NAME (choose-or-type). Blank = auto-named (default).
        project: project.trim() ? project.trim() : undefined,
        prompt: prompt.trim() ? prompt : undefined,
        // k89: the standard set is composed in when the card's tick is on — an
        // empty user negative + the tick still sends the standard exclusions.
        negative_prompt: (() => {
          const eff = composeNegative(negative, stdNegative);
          return eff !== "" ? eff : undefined;
        })(),
        // i2v start image (single first frame). t2v/v2v/id_lock drop it.
        start_image: isI2V && startImages[0] ? startImages[0].uri : undefined,
        // Source clip: i2v (extend) / v2v (restyle) only. Prefer the jail-resolvable
        // abs uri; fall back to the catalog asset id.
        source_video: (isI2V || isV2V) && source?.uri ? source.uri : undefined,
        source_asset_id:
          (isI2V || isV2V) && source && !source.uri ? source.asset_id : undefined,
        // IDENTITY (unified): a SAVED profile is the canonical identity — send its slug
        // (the route resolves it server-side; a later profile edit re-resolves). Raw
        // reference_images ride only for an UNSAVED identity (no profile bound).
        identity_profile:
          (isIdLock || isV2V) && selectedProfile ? selectedProfile.slug : undefined,
        // ORDERED reference images (the hash keys on order — preserve it). id_lock: the
        // identity to lock (required). v2v: an OPTIONAL identity held across the restyle
        // (scene-continuity). The route accepts reference_images on BOTH capabilities.
        reference_images:
          (isIdLock || isV2V) && !selectedProfile && referenceImages.length
            ? referenceImages.map((s) => s.ref.uri)
            : undefined,
        // id_lock optional composition control (both-or-neither; sent only when both set).
        control_image: controlReady ? (controlImage as MediaRef).uri : undefined,
        control_kind: controlReady ? controlKind : undefined,
      };
      const res = await request<unknown>(hugpyConfig.studioI2VEnqueueUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        meta: { specKey: "studio", operation: "studio.i2v.enqueue" },
      });
      if (!mounted.current) return;
      if (!res.ok) {
        setGenMsg(`Enqueue failed: ${describeAppError(errorOf(res))}`);
        return;
      }
      // Subject-clip auto-stage (armed by onGenerateNewSubject): the FIRST successful
      // enqueue after "Generate new clip" arms the wait on ITS job id — later manual
      // generates (pendingSubjectStage already false by then) are plain renders.
      if (pendingSubjectStage) {
        setPendingSubjectStage(false);
        const enq = enqueueResultSchema.safeParse(okValue(res));
        if (enq.success) {
          setAwaitingSubjectJobId(enq.data.job_id);
          setGenMsg(
            "Enqueued — once this clip is produced it becomes the staged subject automatically.",
          );
        } else {
          setGenMsg(
            "Enqueued, but no job id came back to track — stage the produced clip as the subject manually (From library) once it's done.",
          );
        }
      } else {
        setGenMsg("Enqueued — once the clip is produced it plays in the viewer above and lands in the Session Library.");
      }
      onEnqueued();
      // A project name coined here should appear in the combobox next time.
      if (project.trim()) refreshProjects();
    } finally {
      if (mounted.current) setGenBusy(false);
    }
  }, [
    blockSubmit,
    v2vBadGeom,
    missingReason,
    capability,
    width,
    height,
    fps,
    seed,
    vramBudget,
    steps,
    cfg,
    modelId,
    prompt,
    negative,
    stdNegative,
    isI2V,
    isV2V,
    isIdLock,
    isMotion,
    startImages,
    source,
    referenceImages,
    selectedProfile,
    controlImage,
    controlKind,
    pendingSubjectStage,
    onEnqueued,
    project,
    refreshProjects,, warnUnnamed, project]);

  const synthPresets = presets.filter((p) => isSyntheticPreset(p));
  const realPresets = presets.filter((p) => !isSyntheticPreset(p));
  // The identity-lock family (id_lock TEMPLATES + the v2v scene-continuity template) gets
  // its OWN first-class group so the lock/continuity cards are not buried among the plain
  // t2v/i2v/v2v tiers.
  const lockPresets = realPresets.filter(
    (p) => p.capability === "id_lock" || p.id === "scene-continuity",
  );
  const generalPresets = realPresets.filter((p) => !lockPresets.includes(p));
  const sourceClips = clips.filter((c) => c.playable && c.output?.uri);

  // The image-library thumbnail grid used below is now the SHARED
  // `LibraryImageGrid` (studioShared.tsx) — extracted out of this file's original
  // inline closure so the Movie composer's start-image switcher reuses the exact
  // same picker instead of a parallel implementation (operator ask 2026-07-12).

  // A preset card (rendered as a <button> so `key` sits on an intrinsic element).
  function presetCard(p: StudioPreset) {
    const selected = p.id === presetId;
    return (
      <button
        key={p.id}
        type="button"
        className={selected ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"}
        onClick={() => onPickPreset(p.id)}
        aria-pressed={selected}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: "0.2rem",
          textAlign: "left",
          padding: "0.55rem 0.7rem",
          flex: "1 1 15rem",
          minWidth: "13rem",
          maxWidth: "20rem",
          whiteSpace: "normal",
          height: "auto",
        }}
      >
        <span style={{ fontWeight: 600 }}>{p.name}</span>
        <span style={{ fontSize: "0.78rem", opacity: 0.85 }}>
          {p.capability}
          {typeof p.vram_budget_gb === "number" ? ` · ${p.vram_budget_gb} GB` : ""}
          {p.requires_source || p.requires_reference ? " · needs input" : ""}
          {p.recommended ? ` · ${p.recommended}` : ""}
        </span>
        {p.description ? (
          <span style={{ fontSize: "0.75rem", opacity: 0.7, lineHeight: 1.35 }}>
            {p.description}
          </span>
        ) : null}
        {p.prompt_note ? (
          <span style={{ fontSize: "0.72rem", fontStyle: "italic", opacity: 0.7 }}>
            ⓘ {p.prompt_note}
          </span>
        ) : null}
      </button>
    );
  }

  // ── start-image input (variant-driven: hero = dominant/accented; inline = subdued) ──
  function renderStartImage(variant: "hero" | "inline") {
    const isHero = variant === "hero";
    const drawEye = isHero && startImageAttention;
    const thumb = isHero ? "9rem" : "5rem";
    return (
      <section aria-label="Start image" style={{ marginTop: isHero ? 0 : "0.9rem" }}>
        <p
          className="vi-comfy-label"
          style={{ marginBottom: "0.35rem", opacity: isHero ? 1 : 0.75 }}
        >
          Start image{isHero ? " — required for this generator" : " (optional)"}
        </p>
        {drawEye && (
          <p className="vi-input-callout" role="note">
            This generator needs a start image — the subject is carried from it. Drop one below
            (or extend a source clip instead).
          </p>
        )}
        {hasImage && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.5rem" }}>
            {startImages.map((img, idx) => (
              <div key={img.uri} style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                <img
                  src={mediaBytesUrl(img.uri)}
                  alt="start"
                  style={{
                    width: thumb,
                    height: thumb,
                    objectFit: "cover",
                    borderRadius: "0.4rem",
                    background: "#000",
                    border: "1px solid var(--vi-border)",
                  }}
                />
                <button
                  type="button"
                  className="vi-btn vi-btn-sm vi-btn-ghost"
                  onClick={() => removeStartImage(idx)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
        {canAddImage ? (
          isHero && !hasImage ? (
            <DropReceptacle
              onFile={(f) => void onPickImageFile(f, "start")}
              accept="image/*"
              busy={imageBusy}
              title="Drop a start image or click to browse"
              hint="This generator carries the subject from this still."
              className={drawEye ? "vi-dropzone--required" : undefined}
            />
          ) : (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
              <label className="vi-btn vi-btn-accent vi-file-label">
                {imageBusy ? "Working…" : "Upload a still"}
                <input
                  type="file"
                  accept="image/*"
                  className="vi-file-input"
                  disabled={imageBusy}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void onPickImageFile(f, "start");
                    e.target.value = "";
                  }}
                />
              </label>
              <button
                type="button"
                className="vi-btn vi-btn-ghost"
                onClick={() => setShowLibPicker((v) => !v)}
                aria-expanded={showLibPicker}
              >
                {showLibPicker ? "Hide library" : `Pick from library (${libraryImages.length})`}
              </button>
            </div>
          )
        ) : (
          <p className="vi-comfy-hint" role="note">
            Multi-image conditioning here is identity lock — switch capability to Identity lock to
            add 1–4 reference images.
          </p>
        )}
        {imageError && (
          <p className="vi-error" role="alert" style={{ marginTop: "0.4rem" }}>
            {imageError}
          </p>
        )}
        {showLibPicker && canAddImage && (
          <LibraryImageGrid images={libraryImages} onPick={(ref) => addStartImage(ref)} />
        )}
      </section>
    );
  }

  // ── reference-image SLOT LIST: 1–4 ordered stills, reorderable. The SAME list feeds
  //    two shapes — id_lock (variant "required": the identity to lock, accent-gated) and
  //    v2v scene-continuity (variant "optional": an identity held across the restyle). ──
  function renderReferenceSlots(variant: "required" | "optional") {
    const required = variant === "required";
    const drawEye = required && referenceAttention;
    return (
      <section aria-label="Reference images" style={{ marginTop: 0 }}>
        <p
          className="vi-comfy-label"
          style={{ marginBottom: "0.35rem", opacity: required ? 1 : 0.75 }}
        >
          {required
            ? `Reference images — required (1–${MAX_REFERENCE_IMAGES}, order matters)`
            : `Hold an identity (optional) — 1–${MAX_REFERENCE_IMAGES} reference images, order matters`}
        </p>
        {required ? (
          drawEye && (
            <p className="vi-input-callout" role="note">
              {referenceCallout}
            </p>
          )
        ) : (
          <p className="vi-comfy-hint" role="note" style={{ marginTop: 0 }}>
            Optionally carry a subject&apos;s identity across the restyle — add 1–
            {MAX_REFERENCE_IMAGES} reference images (order matters). Leave empty for a plain restyle.
          </p>
        )}
        <div className={drawEye ? "vi-input-required" : undefined}>
          {hasReference && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", marginBottom: "0.5rem" }}>
              {referenceImages.map((slot, idx) => (
                <div
                  key={slot.key}
                  style={{ display: "flex", flexDirection: "column", gap: "0.2rem", alignItems: "center" }}
                >
                  <div style={{ position: "relative" }}>
                    <img
                      src={mediaBytesUrl(slot.ref.uri)}
                      alt={`reference ${idx + 1}`}
                      style={{
                        width: "7rem",
                        height: "7rem",
                        objectFit: "cover",
                        borderRadius: "0.4rem",
                        background: "#000",
                        border: "1px solid var(--vi-border)",
                      }}
                    />
                    <span
                      aria-hidden="true"
                      style={{
                        position: "absolute",
                        top: 3,
                        left: 3,
                        background: "var(--vi-accent)",
                        color: "#000",
                        borderRadius: "0.3rem",
                        padding: "0 0.4rem",
                        fontSize: "0.72rem",
                        fontWeight: 700,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {idx + 1}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: "0.2rem" }}>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      disabled={idx === 0}
                      onClick={() => moveReferenceImage(idx, -1)}
                      title="Move earlier"
                    >
                      ◀
                    </button>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      disabled={idx === referenceImages.length - 1}
                      onClick={() => moveReferenceImage(idx, 1)}
                      title="Move later"
                    >
                      ▶
                    </button>
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => removeReferenceImage(idx)}
                      title="Remove"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
          {canAddReference ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
              <label className="vi-btn vi-btn-accent vi-file-label">
                {imageBusy ? "Working…" : hasReference ? "Add another reference" : "Upload a reference"}
                <input
                  type="file"
                  accept="image/*"
                  className="vi-file-input"
                  disabled={imageBusy}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void onPickImageFile(f, "reference");
                    e.target.value = "";
                  }}
                />
              </label>
              <button
                type="button"
                className="vi-btn vi-btn-ghost"
                onClick={() => setShowLibPicker((v) => !v)}
                aria-expanded={showLibPicker}
              >
                {showLibPicker ? "Hide library" : `Pick from library (${libraryImages.length})`}
              </button>
              <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
                {referenceImages.length}/{MAX_REFERENCE_IMAGES}
              </span>
            </div>
          ) : (
            <p className="vi-comfy-hint" role="note">
              Maximum {MAX_REFERENCE_IMAGES} reference images.
            </p>
          )}
          {imageError && (
            <p className="vi-error" role="alert" style={{ marginTop: "0.4rem" }}>
              {imageError}
            </p>
          )}
          {showLibPicker && canAddReference && (
            <LibraryImageGrid images={libraryImages} onPick={(ref) => addReferenceImage(ref)} />
          )}
          {/* IDENTITY PROFILES (stage a): attach a saved identity profile's reference
              set, or save the current one — the durable form of this identity. */}
          <IdentityProfileControls
            refCount={referenceImages.length}
            currentRefUris={referenceImages.map((s) => s.ref.uri)}
            selectedProfile={selectedProfile}
            onAttach={attachIdentityRefs}
            onSelectProfile={setSelectedProfile}
            maxRefs={MAX_REFERENCE_IMAGES}
            disabled={imageBusy}
          />
        </div>
      </section>
    );
  }

  // ── id_lock OPTIONAL composition control (subdued): a single pose/depth/sketch still. ──
  //    2026-07-29: the SAME block now serves the `motion` capability, where the control
  //    is not optional decoration but the whole render — so it takes the hero treatment
  //    (accent gating + the required copy) exactly like the reference slots do on id_lock.
  function renderControl(variant: "optional" | "required" = "optional") {
    const required = variant === "required";
    const drawEye = required && !hasControl;
    return (
      <section aria-label="Composition control" style={{ marginTop: required ? 0 : "0.9rem" }}>
        <p className="vi-comfy-label" style={{ opacity: required ? 1 : 0.75, marginBottom: "0.2rem" }}>
          {required
            ? "Control still — required (pose, depth or sketch)"
            : "Composition control (optional)"}
        </p>
        {/* Conditional required-input warning — KEPT (functional, k89), wording
            condensed; the general what-a-control-is prose lives in "About Clip". */}
        {drawEye && (
          <p className="vi-input-callout" role="note">
            This generator is driven BY the control still — a STATIC pose/depth/sketch
            anchor repeated across every frame. A source clip cannot ride along (the
            control channel takes one input and the clip would win).
          </p>
        )}
        <p className="vi-comfy-hint" role="note" style={{ marginTop: 0 }}>
          Both-or-neither: set the image and its kind together.
        </p>
        <div
          style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "flex-end", marginTop: "0.4rem" }}
        >
          {controlImage ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
              <img
                src={mediaBytesUrl(controlImage.uri)}
                alt="composition control"
                style={{
                  width: "5.5rem",
                  height: "5.5rem",
                  objectFit: "cover",
                  borderRadius: "0.4rem",
                  background: "#000",
                  border: "1px solid var(--vi-border)",
                }}
              />
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={() => setControlImage(null)}
              >
                Remove
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
              <label className="vi-btn vi-btn-ghost vi-file-label">
                {imageBusy ? "Working…" : "Upload control image"}
                <input
                  type="file"
                  accept="image/*"
                  className="vi-file-input"
                  disabled={imageBusy}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void onPickImageFile(f, "control");
                    e.target.value = "";
                  }}
                />
              </label>
              <button
                type="button"
                className="vi-btn vi-btn-ghost"
                onClick={() => setShowControlLib((v) => !v)}
                aria-expanded={showControlLib}
              >
                {showControlLib ? "Hide library" : "Pick from library"}
              </button>
            </div>
          )}
          <label className="vi-comfy-field" style={{ maxWidth: "10rem" }}>
            <span className="vi-comfy-label">Control kind</span>
            <select
              className="vi-knob-input"
              value={controlKind}
              onChange={(e) => setControlKind(e.target.value as ControlKind)}
            >
              <option value="">— none —</option>
              <option value="pose">pose</option>
              <option value="depth">depth</option>
              <option value="sketch">sketch</option>
            </select>
          </label>
        </div>
        {controlHalfSet && (
          <p className="vi-comfy-hint" role="alert" style={{ marginTop: "0.3rem" }}>
            Composition control is both-or-neither — add a control image AND a kind, or clear both.
          </p>
        )}
        {showControlLib && !controlImage && (
          <LibraryImageGrid
            images={libraryImages}
            onPick={(ref) => {
              setControlImage(ref);
              setShowControlLib(false);
            }}
          />
        )}
      </section>
    );
  }

  // ── source-clip input (hero = dominant/accented; inline = subdued; inline-alt =
  //    the "or extend a clip" alternative under a hero start image) ──
  function renderSource(variant: "hero" | "inline" | "inline-alt") {
    const isHero = variant === "hero";
    const isAlt = variant === "inline-alt";
    const drawEye = isHero && sourceAttention;
    const label = isAlt
      ? "…or use a source clip (extend from its last frame)"
      : isHero
        ? "Source clip — required (the clip to repaint)"
        : isV2V
          ? "Source clip — required"
          : "Source clip (optional — extend from its last frame)";
    return (
      <section aria-label="Source clip" style={{ marginTop: isHero ? 0 : "0.9rem" }}>
        <p
          className="vi-comfy-label"
          style={{ marginBottom: "0.35rem", opacity: isHero ? 1 : 0.75 }}
        >
          {label}
        </p>
        {drawEye && (
          <p className="vi-input-callout" role="note">
            This generator needs a source clip to repaint — pick one below (or send a clip to the
            studio from the Library / Movie Maker).
          </p>
        )}
        <div className={drawEye ? "vi-input-required" : undefined}>
          {source ? (
            <>
              <div className="vi-studio-src-selected">
                <SourceThumb
                  uri={source.uri}
                  name={source.asset_id.slice(0, 8)}
                  playing={sourcePreview}
                  onToggle={() => setSourcePreview((v) => !v)}
                />
                <p className="vi-comfy-hint" role="note" style={{ margin: 0 }}>
                  <strong>{isV2V ? "Restyling" : "Extending"}</strong>{" "}
                  <code>{source.asset_id.slice(0, 8)}</code>
                  {source.width != null && source.height != null
                    ? ` · ${source.width}×${source.height}`
                    : ""}
                  {source.duration_s != null ? ` · ${Math.round(source.duration_s)}s` : ""}
                </p>
              </div>
              {/* Subject-clip switcher (operator ask 2026-07-12): "there isn't a way to
                  actually switch it out" — this row is the ONE concrete front door to
                  all three swap paths the operator named. It CONSOLIDATES the 2026-07-10
                  "Change" button (clear + reopen the library picker, kept as "From
                  library" below — removed from here, not duplicated) plus the two new
                  paths, alongside the plain unstage. */}
              <div className="vi-studio-src-actions" role="group" aria-label="Change subject clip">
                <button
                  type="button"
                  className="vi-btn vi-btn-sm"
                  onClick={() => {
                    setSource(null);
                    setSourcePreview(false);
                    setShowSourcePicker(true);
                  }}
                >
                  ⇄ From library
                </button>
                <label className="vi-btn vi-btn-sm vi-file-label">
                  {imageBusy ? "Working…" : "⇪ Upload video"}
                  <input
                    type="file"
                    accept="video/*"
                    className="vi-file-input"
                    disabled={imageBusy}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void onPickVideoFile(f);
                      e.target.value = "";
                    }}
                  />
                </label>
                <button type="button" className="vi-btn vi-btn-sm" onClick={onGenerateNewSubject}>
                  ✦ Generate new
                </button>
                <button
                  type="button"
                  className="vi-btn vi-btn-sm vi-btn-ghost"
                  onClick={() => {
                    setSource(null);
                    setSourcePreview(false);
                  }}
                >
                  ✕ Clear
                </button>
              </div>
            </>
          ) : (
            <div className="vi-studio-src-actions" role="group" aria-label="Add subject clip">
              <button
                type="button"
                className={isHero ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"}
                onClick={() => setShowSourcePicker((v) => !v)}
                aria-expanded={showSourcePicker}
              >
                {showSourcePicker ? "Hide clips" : `⇄ From library (${sourceClips.length})`}
              </button>
              <label
                className={`${isHero ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"} vi-file-label`}
              >
                {imageBusy ? "Working…" : "⇪ Upload video"}
                <input
                  type="file"
                  accept="video/*"
                  className="vi-file-input"
                  disabled={imageBusy}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void onPickVideoFile(f);
                    e.target.value = "";
                  }}
                />
              </label>
              <button type="button" className="vi-btn vi-btn-ghost" onClick={onGenerateNewSubject}>
                ✦ Generate new clip
              </button>
            </div>
          )}
          {imageError && (
            <p className="vi-error" role="alert" style={{ marginTop: "0.4rem" }}>
              {imageError}
            </p>
          )}
          {showSourcePicker && !source && (
            <ul
              style={{
                listStyle: "none",
                margin: "0.5rem 0 0",
                padding: 0,
                maxHeight: "14rem",
                overflowY: "auto",
              }}
            >
              {sourceClips.length === 0 ? (
                <li className="vi-comfy-hint">No playable studio clips yet.</li>
              ) : (
                sourceClips.map((c) => {
                  // Restructured row (mirrors the Library's LibraryRow): a poster/preview
                  // thumbnail + metadata + a DISTINCT select affordance. The row is no
                  // longer one big <button> — a playable <video> can't nest in a button.
                  const ref = clipToMediaRef(c);
                  const w = c.output?.width;
                  const h = c.output?.height;
                  const dur = c.output?.duration_s;
                  const isPreviewing = previewId === c.job_id;
                  const togglePreview = () =>
                    setPreviewId((cur) => (cur === c.job_id ? null : c.job_id));
                  return (
                    <li key={c.job_id} className="vi-studio-src-row">
                      {ref ? (
                        <SourceThumb
                          uri={ref.uri}
                          name={shortId(c.job_id)}
                          playing={isPreviewing}
                          onToggle={togglePreview}
                        />
                      ) : (
                        <div className="vi-studio-src-thumb vi-lib-placeholder" aria-hidden>
                          <span>clip</span>
                        </div>
                      )}
                      <div className="vi-studio-src-body">
                        <div className="vi-studio-src-meta" title={c.job_id}>
                          <code>{shortId(c.job_id)}</code>
                          {w != null && h != null ? (
                            <span>
                              {w}×{h}
                            </span>
                          ) : null}
                          {dur != null ? <span>{Math.round(dur)}s</span> : null}
                        </div>
                        <div className="vi-studio-src-actions">
                          <button
                            type="button"
                            className="vi-btn vi-btn-sm vi-btn-ghost"
                            onClick={togglePreview}
                            disabled={!ref}
                            aria-pressed={isPreviewing}
                          >
                            {isPreviewing ? "◾ Stop" : "▶ Preview"}
                          </button>
                          <button
                            type="button"
                            className="vi-btn vi-btn-sm vi-btn-accent"
                            onClick={() => {
                              if (ref) setSource(ref);
                              setShowSourcePicker(false);
                              setPreviewId(null);
                              // A manual library pick takes back control from any
                              // armed/in-flight "generate new subject" wait.
                              setPendingSubjectStage(false);
                              setAwaitingSubjectJobId(null);
                            }}
                            disabled={!ref}
                            title={
                              ref
                                ? isV2V
                                  ? "Use this clip as the restyle source"
                                  : "Extend from this clip's last frame"
                                : "This clip has no streamable output yet"
                            }
                          >
                            {isV2V ? "Use · restyle" : "Use · extend"}
                          </button>
                        </div>
                      </div>
                    </li>
                  );
                })
              )}
            </ul>
          )}
        </div>
      </section>
    );
  }

  // Round 8: the Movie .vi-knob grid renders into the LEFT-SIDEBAR "Settings" tab (via a
  // studio-local portal into `settingsHost`), NOT the center — folding the former inline
  // knob column into the sidebar (movie-shape canonical). Knob STATE stays in this
  // component; the portal only relocates where the inputs render.
  // ── PRESET HONESTY (2026-07-29): a preset offered = a preset whose capability is
  //    servable on this fleet, and an UNPROVEN capability says so on the row.
  //
  // The curated studio presets (GET /video/studio/presets) are INTENT; the ratified
  // render table (GET /video/render/presets) is MEASUREMENT. Joining them per capability
  // is what turns the dropdown from "eight things we wrote down" into "the things that
  // will come back as pixels, and which of them have". Nothing is hidden: an option that
  // cannot be picked right now renders DISABLED carrying the backend's own reason, so
  // the fleet's shape stays visible instead of silently shrinking the menu.
  function presetOption(p: StudioPreset) {
    const o = renderLoaded ? offerFor(p.capability) : undefined;
    const unproven = o?.servable === true && o.proven === false;
    const label = `${p.name}${unproven ? " · unproven" : ""}`;
    const disabled = !!o && !o.offered;
    return (
      <option key={p.id} value={p.id} disabled={disabled} title={o?.reason || p.description || ""}>
        {label}
        {disabled ? " — unavailable" : ""}
      </option>
    );
  }

  const knobGrid = (
    <div className="vi-studio-knobs">
      {/* ABOUT — the condensed what-is-this / how-to prose (k89), collapsed at the
          top of the Settings-tab content. Absorbs the descriptive halves of the old
          inline callouts; the conditional required-input warnings stay on the form. */}
      <div className="vi-studio-knob-wide" style={{ flex: "1 1 100%" }}>
        <AboutExpander title="About Clip">
          <p>
            Generates ONE studio clip from a prompt plus whatever inputs the chosen
            generator needs.
          </p>
          <p>
            Pick a template or capability here and the form morphs to that shape:
            text→video needs nothing, image→video a start image (or a source clip to
            extend), restyle a source clip, identity lock 1–4 reference images, and
            pose/structure a control still.
          </p>
          <p>
            A missing required input is flagged on the card and gates Generate until
            you add it.
          </p>
          <p>
            Finished clips play in the viewer above and land in the Session Library;
            the Tester at the bottom sweeps one prompt across every model.
          </p>
        </AboutExpander>
      </div>
      {/* TEMPLATE dropdown — the tier chooser, mirroring Movie's right-rail PRESET
          dropdown. Picking one = clicking a tier card: sets capability/shape + prefills
          the knobs below. (The center cards are retired — this is the one chooser.) */}
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-studio-template">Template</label>
        <select
          id="vi-studio-template"
          className="vi-knob-select"
          value={presetId}
          onChange={(e) => (e.target.value ? onPickPreset(e.target.value) : setPresetId(""))}
        >
          <option value="">Custom — no template</option>
          {generalPresets.length > 0 && (
            <optgroup label="Tiers">{generalPresets.map((p) => presetOption(p))}</optgroup>
          )}
          {lockPresets.length > 0 && (
            <optgroup label="Identity lock & continuity">
              {lockPresets.map((p) => presetOption(p))}
            </optgroup>
          )}
          {synthPresets.length > 0 && (
            <optgroup label="Synthetic previews">{synthPresets.map((p) => presetOption(p))}</optgroup>
          )}
        </select>
        <span className="vi-knob-hint">
          A template sets the WORKFLOW — capability, geometry, cadence, budget — and slots
          it around what you have already written and attached. Your prompt, negative,
          identity, images and any seed you chose are never overwritten; an empty prompt
          box is the only thing a template&apos;s example text fills.
          {renderLoaded ? " Rows marked “unproven” have a ratified path that has not yet produced a clip on this fleet." : ""}
        </span>
      </div>
      {/* PROJECT — choose an existing name or free-type a new one (Movie's "PROJECT NAME
          (OPTIONAL)" idiom). Threaded into the enqueue as `project`; blank = auto-named. */}
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-studio-project">Project name (optional)</label>
        <input
          id="vi-studio-project"
          className="vi-knob-input"
          type="text"
          list="vi-studio-project-list"
          value={project}
          placeholder="auto-named — or pick / type a project"
          onChange={(e) => setProject(e.target.value)}
        />
        <datalist id="vi-studio-project-list">
          {knownProjects.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
        <span className="vi-knob-hint">
          Choose an existing project or type a new name to create it. Blank keeps the
          auto-named default.
        </span>
      </div>
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-studio-cap">Capability</label>
        <select
          id="vi-studio-cap"
          className="vi-knob-select"
          value={capability}
          onChange={(e) => onPickCapability(e.target.value as Capability)}
        >
          {/* SOURCED FROM THE SERVABLE SET (2026-07-29), not from a hand-curated union.
              Every row here is a capability at least one RATIFIED preset covers; a row
              that the fleet cannot serve, or that the inputs you have already attached
              rule out, renders DISABLED with the backend's own reason as its tooltip
              rather than vanishing — the same disabled-with-honest-title idiom the
              movie start switcher uses for "start from a clip".

              id_lock stays non-selectable-by-hand: it is set by an identity-lock
              TEMPLATE or by attaching a profile/references. It must still RENDER while
              active — this is a controlled <select>, and a value with no matching
              <option> shows blank. */}
          {capOffers.map((o) => {
            const viaTemplate = o.capability === "id_lock";
            const disabled = !o.offered || (viaTemplate && !isIdLock);
            if (viaTemplate && !isIdLock && !o.servable) return null;
            const suffix = !o.servable
              ? " — not on this fleet"
              : viaTemplate
                ? " — via template"
                : !o.offered
                  ? " — incompatible with your inputs"
                  : o.proven === false
                    ? " · unproven"
                    : "";
            return (
              <option
                key={o.capability}
                value={o.capability}
                disabled={disabled}
                title={o.reason || undefined}
              >
                {o.label}
                {suffix}
              </option>
            );
          })}
          {/* THE OTHER HALF OF THE ANSWER, and the reason /video/render/presets publishes
              it: every capability the enum declares that NO ratified preset covers, with
              the MEASURED blocker as its tooltip. Shown disabled rather than omitted so a
              user learns "studio cannot render 'keyframe' on this fleet: … no end-frame
              input exists in the spine" at DISCOVERY time instead of after a render. The
              wording is the registry's, verbatim. */}
          {renderUnavailable.length > 0 && (
            <optgroup label="Not renderable on this fleet">
              {renderUnavailable.map((u) => (
                <option
                  key={u.capability}
                  value={u.capability}
                  disabled
                  title={u.refusal || u.reason || undefined}
                >
                  {capabilityLabel(u.capability)} — {u.reason ? u.reason.slice(0, 90) : "no preset covers it"}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <span className="vi-knob-hint">
          {renderLoaded
            ? `What this fleet can render today${renderBox ? ` on ${renderBox}` : ""}. Greyed-out rows say why — hover for the measured reason.`
            : "Loading what this fleet can render…"}
        </span>
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-w">width (px)</label>
        <input
          id="vi-studio-w"
          className="vi-knob-input"
          type="number"
          min={64}
          step={8}
          value={width}
          disabled={isV2V}
          title={isV2V ? geomLockTitle : geomEnvelopeTitle}
          onChange={(e) => {
            geomTouched.current = true;
            setWidth(Number(e.target.value) || 0);
          }}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-h">height (px)</label>
        <input
          id="vi-studio-h"
          className="vi-knob-input"
          type="number"
          min={64}
          step={8}
          value={height}
          disabled={isV2V}
          title={isV2V ? geomLockTitle : geomEnvelopeTitle}
          onChange={(e) => {
            geomTouched.current = true;
            setHeight(Number(e.target.value) || 0);
          }}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-fps">fps</label>
        <input
          id="vi-studio-fps"
          className="vi-knob-input"
          type="number"
          min={1}
          max={60}
          value={fps}
          disabled={isV2V}
          title={isV2V ? geomLockTitle : geomEnvelopeTitle}
          onChange={(e) => {
            geomTouched.current = true;
            setFps(Number(e.target.value) || 0);
          }}
        />
        {geomOutOfEnvelope && !isV2V && (
          <span className="vi-knob-flag">
            Outside the ratified envelope{capGeometry ? ` (${capGeometry.width}×${capGeometry.height})` : ""} —
            this geometry has no proven binding for {CAP_LABELS[capability]} and the router may refuse it.
          </span>
        )}
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-seed">seed</label>
        <input
          id="vi-studio-seed"
          className="vi-knob-input"
          type="number"
          min={0}
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value) || 0)}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-vram">vram budget (GB)</label>
        <input
          id="vi-studio-vram"
          className="vi-knob-input"
          type="number"
          min={0}
          step={0.5}
          value={vramBudget}
          placeholder="route default"
          onChange={(e) => {
            // Marks the field OPERATOR-OWNED: from here on a preset or capability change
            // fills nothing over it. Clearing the field hands it back to the presets.
            budgetTouched.current = e.target.value.trim() !== "";
            setVramBudget(e.target.value);
          }}
        />
        <span className="vi-knob-hint">
          {capFloorGb != null
            ? `Blank = autofit to the card. ${CAP_LABELS[capability]} needs at least ${capFloorGb} GB to bind a real model — measured, not guessed.`
            : "Blank = route default (sub-real binds synthetic)."}
        </span>
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-steps">steps</label>
        <input
          id="vi-studio-steps"
          className="vi-knob-input"
          type="number"
          min={1}
          max={100}
          value={steps}
          placeholder="model default"
          onChange={(e) => setSteps(e.target.value)}
        />
      </div>
      <div className="vi-knob">
        <label htmlFor="vi-studio-cfg">cfg</label>
        <input
          id="vi-studio-cfg"
          className="vi-knob-input"
          type="number"
          min={0}
          max={20}
          step={0.5}
          value={cfg}
          placeholder="model default"
          onChange={(e) => setCfg(e.target.value)}
        />
      </div>
      {/* MODEL PIN — FILTERED BY CAPABILITY (2026-07-29).
          It used to offer STUDIO_MODEL_IDS: a static, unfiltered mirror of a backend seed
          file. Nothing in it knew which models can serve the capability you had chosen,
          so pinning wan2.1-t2v-1.3b under capability=v2v was a first-class menu choice
          that could only fail at render time. The menu is now the models that appear in a
          RATIFIED preset for the active capability.

          THE ESCAPE HATCH IS DELIBERATE. "Show every studio model" keeps the whole list
          reachable — an operator may legitimately want to pin off-menu (to prove a new
          binding, or to reproduce an old job) — and every off-menu row is LABELLED
          unverified rather than quietly presented as equivalent. An unroutable pin still
          comes back as errors-as-data on the job; this only changes what we recommend. */}
      <div className="vi-knob vi-studio-knob-wide">
        <label htmlFor="vi-studio-model">model pin</label>
        <select
          id="vi-studio-model"
          className="vi-knob-select"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
        >
          <option value="">Auto — router picks by budget + resolution</option>
          {capModels.length > 0 && (
            <optgroup label={`Ratified for ${CAP_LABELS[capability]}`}>
              {capModels.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </optgroup>
          )}
          {(showAllModels || !renderLoaded || modelOffMenu) && (
            <optgroup label={renderLoaded ? "Other studio models — unverified for this capability" : "Studio models"}>
              {STUDIO_MODEL_IDS.filter((id) => !capModels.includes(id)).map((id) => (
                <option key={id} value={id}>
                  {id}
                  {renderLoaded ? " · unverified" : ""}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        {renderLoaded && (
          <label className="vi-knob-hint" style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showAllModels}
              onChange={(e) => setShowAllModels(e.target.checked)}
            />
            Advanced — show every studio model (off-menu pins are unverified for this capability)
          </label>
        )}
        {modelOffMenu && (
          <span className="vi-knob-flag">
            {modelId} has no ratified preset for {CAP_LABELS[capability]} — the router may
            refuse the pin, and it will say so on the job rather than fall back silently.
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="vi-studio-generate">
      {/* B2 mode switch: Clip (existing surface, default, unchanged) | Cinema (movie).
          Rendered ONLY on the standalone /studio-clips mount (no lockedSurfaceMode). When
          the top-level Generate tab locks the choice (Clip / Cinema tabs), this switcher is
          hidden — the tab IS the choice. The "movie" surfaceMode value is unchanged (label
          reads "Cinema"; the wire/tester value stays "movie"). */}
      {!lockedSurfaceMode && (
      <div
        role="tablist"
        aria-label="Studio generate mode"
        style={{ display: "flex", gap: "0.4rem", marginBottom: "0.7rem" }}
      >
        <button
          type="button"
          role="tab"
          aria-selected={surfaceMode === "clip"}
          className={surfaceMode === "clip" ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"}
          onClick={() => setInternalSurfaceMode("clip")}
        >
          Clip
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={surfaceMode === "movie"}
          className={surfaceMode === "movie" ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"}
          onClick={() => setInternalSurfaceMode("movie")}
        >
          Cinema
        </button>
      </div>
      )}
      {surfaceMode === "movie" ? (
        /* Movie mode portals its own "Movie templates" dropdown into the SAME Settings-tab
           host the Clip knobGrid uses (the composer owns that portal).
           NB no "Test all models" here: Cinema is the multi-GOAL composer (its prompt
           lives per-goal inside StudioMovieComposer, not in this surface's single
           `prompt`), so a single-prompt sweep doesn't map. The top-level Movie tab
           already offers a movie-model sweep from one prompt. */
        <StudioMovieComposer settingsHost={settingsHost} />
      ) : (
        <>
      {/* Round 8: the knob grid renders in the left-sidebar Settings tab (portaled into
          settingsHost); null until the sidebar mounts its Settings panel. */}
      {settingsHost ? createPortal(knobGrid, settingsHost) : null}
      {/* Subject-clip "generate new" note — ALWAYS visible (unlike renderSource, which
          stops rendering entirely once onGenerateNewSubject forces the t2v shape), so
          the "what's happening" signal survives that shape change. Clears itself the
          moment the wait resolves (success, failure, or a manual override elsewhere). */}
      {(pendingSubjectStage || awaitingSubjectJobId) && (
        <p className="vi-input-callout" role="note">
          {awaitingSubjectJobId
            ? "Waiting for the new clip to finish rendering — it becomes the staged subject automatically."
            : "No subject staged. Write a prompt and press Generate below — the clip it produces becomes the staged subject."}
        </p>
      )}
      {/* ── 1) TEMPLATE/TIER CARDS — RETIRED from the centre (Round 9): template
             selection now lives in the Settings-tab dropdown (mirrors Movie's PRESET
             dropdown). The card grid is kept below, ARCHIVED + never rendered (`false &&`),
             per the no-delete house rule. ── */}
      {false && (
        <section aria-label="Studio templates" className="vi-studio-templates">
        <p className="vi-knob-headline">
          Choose a template{" "}
          <span className="vi-knob-headline-sub">
            the tier you pick is the mode — it sets the shape and prefills the knobs below
          </span>
        </p>
        {presetsLoading ? (
          <p className="vi-knob-hint">Loading templates…</p>
        ) : generalPresets.length === 0 ? (
          <p className="vi-knob-hint">No real tiers registered.</p>
        ) : (
          <div className="vi-studio-card-grid">{generalPresets.map((p) => presetCard(p))}</div>
        )}

        {lockPresets.length > 0 && (
          <>
            <p className="vi-knob-subhead">
              Identity lock &amp; scene continuity{" "}
              <span className="vi-knob-headline-sub">
                — lock a person / thing / place to reference images, or continue a prior scene
              </span>
            </p>
            <div className="vi-studio-card-grid">{lockPresets.map((p) => presetCard(p))}</div>
          </>
        )}

        {synthPresets.length > 0 && (
          <>
            <p className="vi-knob-subhead">
              Synthetic previews{" "}
              <span className="vi-knob-headline-sub">
                — no GPU; the prompt is recorded, not rendered
              </span>
            </p>
            <div className="vi-studio-card-grid">{synthPresets.map((p) => presetCard(p))}</div>
          </>
        )}
        {presetId && (
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            onClick={() => setPresetId("")}
            style={{ marginTop: "0.5rem" }}
          >
            Clear template (custom)
          </button>
        )}
        </section>
      )}

      {/* ── 2) COMPOSER — prompt + negative, in the CENTER. Round 8: the Movie knob
             grid moved to the left-sidebar Settings tab (portaled above), so the
             center is cards → composer → shape heroes → Run (movie-shape canonical). ── */}
      <section aria-label="Prompt" className="vi-studio-composer">
        {/* k89: the clip's prompt + negative rebuilt on the SHARED PromptCard +
             PromptFieldTabs (the same shell every Studio prompt list renders). A
             single-prompt surface: no up/down/split/✕ callbacks (the action cluster
             hides), and deliberately NO PromptListToolbar — a count/select toolbar
             over one card would be noise. Same state; the attach toolbar + chips
             move into the uniform attach-row position below the assist cluster. */}
        <ol className="vi-gen-parts">
          <PromptCard
            badge="clip"
            index={0}
            noun="clip"
            headerHint={
              <span className="vi-comfy-hint" style={{ opacity: 0.75 }}>
                {isV2V
                  ? "video → video (restyle)"
                  : isMotion
                    ? "pose / structure → video"
                    : hasImage
                      ? "image → video"
                      : "text → video"}
                {capProven === false ? " · unproven on this fleet" : ""}
              </span>
            }
          >
            <PromptFieldTabs
              idPrefix="vi-studio-clip"
              promptLabel="prompt"
              prompt={prompt}
              onPromptChange={setPrompt}
              promptPlaceholder={
                isV2V ? "repaint this scene as … (describe the new look)" : "prompt"
              }
              promptExtras={
                preset?.prompt_note ? (
                  <span className="vi-knob-hint">
                    <em>ⓘ {preset.prompt_note}</em>
                  </span>
                ) : undefined
              }
              negativeLabel="negative prompt"
              negative={negative}
              onNegativeChange={setNegative}
              negativePlaceholder="what to avoid"
              stdNegative={stdNegative}
              onStdNegativeChange={setStdNegative}
            />
            {/* PROMPT ASSIST — Enhance enriches the clip prompt, Generate writes a fresh
                one (seeded by any current text). Same shared cluster as the Generate
                station; errors surface inline and never clobber the existing prompt. */}
            <PromptAssistButtons
              busy={assist.assistBusy}
              error={assist.assistError}
              onDismissError={assist.dismissError}
              onEnhance={() => void assist.runAssist("detail", prompt, setPrompt)}
              onGenerate={() =>
                void (async () => {
                  let fresh = "";
                  await assist.runAssist("generate", prompt, (p) => {
                    fresh = p;
                    setPrompt(p);
                  });
                  if (!fresh.trim()) return;
                  // NEGATIVE RIDES ALONG (operator 2026-08-13): generated
                  // exclusions APPEND to any typed negative; std tick untouched.
                  await assist.runNegative({ subject: fresh }, (neg) => {
                    const cur = negative.trim().replace(/,\s*$/, "");
                    setNegative(cur ? `${cur}, ${neg}` : neg);
                  });
                  // NAME RIDES ALONG: derive the project name from the fresh
                  // prompt when blank or itself auto-derived.
                  const t = project.trim();
                  if (t === "" || t === autoProjectRef.current) {
                    const derived = deriveNameFrom(fresh);
                    if (derived) {
                      autoProjectRef.current = derived;
                      setProject(derived);
                    }
                  }
                })()
              }
              canEnhance={prompt.trim() !== ""}
              models={assist.assistModels}
              model={assist.assistModel}
              onModelChange={assist.setAssistModel}
            />
            {/* ATTACH ROW — the same chips + buttons GoalComposer used to render
                inline, now in the uniform per-card attach position. Same state and
                callbacks (promptAttachments / promptAttached above); the chip markup
                mirrors GoalComposer's .vi-goal-composer-chip idiom verbatim. */}
            {promptAttached.length > 0 && (
              <div className="vi-goal-composer-chips">
                {promptAttached.map((a, i) => (
                  <span key={`${a.ref.uri}-${i}`} className="vi-goal-composer-chip">
                    {a.ref.kind === "image" ? (
                      <img
                        src={mediaBytesUrl(a.ref.uri)}
                        alt={a.label ?? "attachment"}
                        className="vi-goal-composer-chip-thumb"
                      />
                    ) : (
                      <span className="vi-goal-composer-chip-kind">{a.ref.kind}</span>
                    )}
                    <span className="vi-goal-composer-chip-label">
                      {a.label ?? a.ref.asset_id.slice(0, 8)}
                    </span>
                    {a.onRemove && (
                      <button
                        type="button"
                        className="vi-goal-composer-chip-x"
                        onClick={a.onRemove}
                        aria-label="Remove attachment"
                        title="Remove"
                      >
                        ✕
                      </button>
                    )}
                  </span>
                ))}
              </div>
            )}
            {promptAttachments.length > 0 && (
              <div className="vi-gen-part-attach">
                {promptAttachments.map((a, i) => (
                  <button
                    key={`${a.label}-${i}`}
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={a.onClick}
                    disabled={a.disabled}
                    title={a.title}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
            )}

            {/* SETTINGS EXPANDER (operator ask 2026-08-13): the clip prompt
                component hosts the FULL knob set, same as Scene/Movie/Cinema
                prompt components. Clip is a SINGLE-prompt surface, so these
                bind the SAME state as the left-column Settings tab — change a
                knob in either place and both update; comprehensive by
                construction (model pin included). The TEMPLATE chooser stays
                on the Settings tab only: it morphs the form shape (required
                inputs), it is not a render knob. */}
            <PromptCardSettings label="settings — model · size · steps · cfg · seed · budget">
              <div className="vi-knob vi-studio-knob-wide" style={{ minWidth: "16rem" }}>
                <label htmlFor="vi-clipcard-model">model pin</label>
                <select
                  id="vi-clipcard-model"
                  className="vi-knob-select"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                >
                  <option value="">Auto — router picks by budget + resolution</option>
                  {capModels.length > 0 && (
                    <optgroup label={`Ratified for ${CAP_LABELS[capability]}`}>
                      {capModels.map((id) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {(showAllModels || !renderLoaded || modelOffMenu) && (
                    <optgroup label={renderLoaded ? "Other studio models — unverified for this capability" : "Studio models"}>
                      {STUDIO_MODEL_IDS.filter((id) => !capModels.includes(id)).map((id) => (
                        <option key={id} value={id}>
                          {id}
                          {renderLoaded ? " · unverified" : ""}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor="vi-clipcard-width">width</label>
                <input id="vi-clipcard-width" className="vi-knob-input" type="number" min={64} step={8}
                       value={width} onChange={(e) => setWidth(Number(e.target.value) || 0)} />
              </div>
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor="vi-clipcard-height">height</label>
                <input id="vi-clipcard-height" className="vi-knob-input" type="number" min={64} step={8}
                       value={height} onChange={(e) => setHeight(Number(e.target.value) || 0)} />
              </div>
              {/* LENGTH ROW (operator 2026-08-13): fps | frames | seconds — any two
                  set the third; the dimmed one is derived, last-touched wins. */}
              <LengthRow
                idPrefix="vi-clipcard-len"
                fps={fps}
                onFps={(n) => setFps(n)}
                frames={requestedFrames}
                onFrames={setRequestedFrames}
              />
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor="vi-clipcard-steps">steps</label>
                <input id="vi-clipcard-steps" className="vi-knob-input" type="number" min={1} max={100}
                       value={steps} placeholder="model default" onChange={(e) => setSteps(e.target.value)} />
              </div>
              <div className="vi-knob" style={{ maxWidth: "7rem" }}>
                <label htmlFor="vi-clipcard-cfg">cfg</label>
                <input id="vi-clipcard-cfg" className="vi-knob-input" type="number" min={0} max={20} step={0.5}
                       value={cfg} placeholder="model default" onChange={(e) => setCfg(e.target.value)} />
              </div>
              <div className="vi-knob" style={{ maxWidth: "8rem" }}>
                <label htmlFor="vi-clipcard-seed">seed</label>
                <input id="vi-clipcard-seed" className="vi-knob-input" type="number" min={0}
                       value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              </div>
              <div className="vi-knob" style={{ maxWidth: "9rem" }}>
                <label htmlFor="vi-clipcard-vram">VRAM budget GB</label>
                <input id="vi-clipcard-vram" className="vi-knob-input" type="number" min={1}
                       value={vramBudget} placeholder="autofit" onChange={(e) => setVramBudget(e.target.value)} />
              </div>
              <div className="vi-knob" style={{ maxWidth: "12rem" }}>
                <label htmlFor="vi-clipcard-project">project (archive name)</label>
                <input id="vi-clipcard-project" className="vi-knob-input" type="text"
                       value={project} placeholder="optional" onChange={(e) => setProject(e.target.value)} />
                <WarnUnnamedToggle checked={warnUnnamed} onChange={setWarnUnnamed} />
              </div>
            </PromptCardSettings>
          </PromptCard>
        </ol>
      </section>

      {/* Required-input callout — surfaces the missing prerequisite PROMINENTLY next to
          the composer now that the tier cards (which used to carry the "needs input" cue)
          are retired. The required input itself is the hero shape directly below. */}
      {missingRequired && (
        <p className="vi-studio-required-callout" role="note">
          ⚠ {missingReason}
        </p>
      )}

      {/* ── 3) SHAPE EDITOR (heroes) — morphs per preformed shape; stays dominant when
             required (accent gating) even though it now sits below the knob grid. ── */}
      {hero === "start-image" && (
        <section
          aria-label="Required input"
          className="vi-studio-shape"
          style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
        >
          {renderStartImage("hero")}
          {renderSource("inline-alt")}
        </section>
      )}
      {hero === "source" && (
        <section
          aria-label="Required input"
          className="vi-studio-shape"
          style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}
        >
          {renderSource("hero")}
          {/* scene-continuity: OPTIONALLY hold a subject's identity across the restyle. */}
          {renderReferenceSlots("optional")}
        </section>
      )}
      {hero === "reference" && (
        <section
          aria-label="Required input"
          className="vi-studio-shape"
          style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
        >
          {renderReferenceSlots("required")}
          {renderControl()}
        </section>
      )}
      {/* motion (2026-07-29): the control still IS the render, so it is the hero and the
          only input this shape offers — no source clip (the route refuses the pair), no
          references (a non-VACE-reference branch would silently ignore them). */}
      {hero === "control" && (
        <section
          aria-label="Required input"
          className="vi-studio-shape"
          style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
        >
          {renderControl("required")}
        </section>
      )}

      {/* LOWER inputs — only when NOT a hero shape (custom / synthetic / t2v). */}
      {!hero && isI2V && <div className="vi-studio-shape">{renderStartImage("inline")}</div>}
      {!hero && !isT2V && !isIdLock && <div className="vi-studio-shape">{renderSource("inline")}</div>}
      {!hero && isT2V && (
        <p className="vi-knob-hint" role="note" style={{ marginTop: "0.9rem" }}>
          Text-to-video: no image inputs needed — the clip is a function of prompt + seed +
          geometry.
        </p>
      )}

      {/* ── SYNTHETIC-TIER HONESTY BANNER — loud, right above Generate, so the
             operator sees it at the moment it matters instead of a footnote three
             fields up. Only fires when the CURRENT form state would actually bind
             synthetic (see wouldBindSynthetic above); the one-click fill jumps
             straight past STUDIO_REAL_FLOOR_GB with headroom (STUDIO_SAFE_FILL_GB). ── */}
      {wouldBindSynthetic && (
        <div
          className="vi-studio-required-callout"
          role="alert"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.6rem",
            flexWrap: "wrap",
            marginBottom: "0.6rem",
          }}
        >
          <span>
            ⚠ This will render the SYNTHETIC TEST tier (colored noise, no real model). Set VRAM
            budget ≥ {realFloorGb} for a real render on the GPU worker
            {capFloorGb != null ? " — the measured envelope of the cheapest real model for this capability" : ""}.
          </span>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-accent"
            disabled={genBusy}
            onClick={() => {
              budgetTouched.current = true;
              setVramBudget(String(safeFillGb));
            }}
          >
            Set budget to {safeFillGb}
          </button>
        </div>
      )}

      {/* ── 4) RUN ROW — the Movie-mode .vi-station-footer idiom (flags/errors on the
             left, the primary Run affordance + status on the right). ── */}
      <div className="vi-station-footer vi-studio-run">
        <div className="vi-station-footer-main">
          {missingRequired && <span className="vi-knob-flag">{missingReason}</span>}
          {v2vBadGeom && (
            <span className="vi-knob-flag">
              Restyle geometry must be 832×480 landscape (the VACE envelope).
            </span>
          )}
          {genMsg && (
            <p className="vi-knob-hint" role="status" style={{ margin: 0 }}>
              {genMsg}
            </p>
          )}
        </div>
        {nameWarn && (
          <UnnamedWarnPanel
            noun="clip"
            onProceed={(name) => {
              setNameWarn(false);
              if (name) setProject(name);
              setPendingRun(true);
            }}
            onDismiss={() => setNameWarn(false)}
          />
        )}
        <div className="vi-station-footer-run">
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={genBusy || blockSubmit}
            onClick={() => void onGenerate()}
            title={blockSubmit ? (v2vBadGeom ? "Fix the geometry" : missingReason) : undefined}
          >
            {genBusy
              ? "Enqueuing…"
              : isV2V
                ? "Restyle clip"
                : isIdLock
                  ? "Generate (identity lock)"
                  : isMotion
                    ? "Generate (pose / structure)"
                    : isI2V
                      ? "Generate (i2v)"
                      : "Generate (t2v)"}
          </button>
          {/* (The cross-model "🧪 Test all models" ghost that sat here moved into
              the bottom Tester collapsible below — k89.) */}
        </div>
      </div>

      {/* CLIP TESTER (k89) — no row-based tester on this surface, so the bottom
          "[▸ Tester]" collapsible holds just the cross-model sweep folded in from
          the run row (same handler, untouched). Its enqueue/result feedback rides
          the shared genMsg line in the run row above; per-model results stream to
          the generation log below this surface. */}
      <div className="vi-gen-tester">
        <div className="vi-gen-tester-head">
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            aria-expanded={clipTesterOpen}
            onClick={() => setClipTesterOpen((v) => !v)}
            title="Sweep the prompt across every clip model"
          >
            {clipTesterOpen ? "▾" : "▸"} Tester
          </button>
          <span className="vi-comfy-hint">
            Sweep one prompt across every clip model — results stream to the
            generation log below.
          </span>
        </div>
        {clipTesterOpen && (
          <div className="vi-gen-tester-actions">
            {!isT2V && !(isI2V && hasImage) && (
              <span className="vi-knob-hint" role="note">The current sweep supports text-to-video and image-to-video with a start image. Choose one of those shapes to run it.</span>
            )}
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-tester"
              disabled={testerBusy || !prompt.trim() || (!isT2V && !(isI2V && hasImage))}
              onClick={() =>
                void onTestAllModels({
                  category: surfaceMode,
                  models: clipTesterModels,
                  width,
                  height,
                  fps,
                  seed,
                  startImage: isI2V ? startImages[0]?.uri : undefined,
                  steps: steps.trim() ? Number(steps) : undefined,
                  cfg: cfg.trim() ? Number(cfg) : undefined,
                  requestedFrames: requestedFrames.trim() ? Number(requestedFrames) : undefined,
                  negative: negative.trim() || undefined,
                })
              }
              title={!isT2V && !(isI2V && hasImage) ? "This sweep currently supports text-to-video or image-to-video with a start image." : "Run this prompt across each model that fits the current filter; one battery row per model."}
            >
              {testerBusy ? "Starting…" : "🧪 Test all models"}
            </button>
          </div>
        )}
      </div>
        </>
      )}
      {/* STUDIO-ASSIST LIVE LOG: renders in the sidebar's ACTIVE tab now (operator
          correction 2026-08-05 — "active" meant Active Processes, not the active
          generate surface). StudioPlane portals it via reg.activeExtraHost. */}
    </div>
  );
}
